from __future__ import annotations
import importlib.util
import sys

# Define your required packages
required = {'pymatgen', 'monty', 'tabulate','numpy', 'matplotlib', 'pandas','param','xarray', 'seaborn'}

missing = []

for pkg in required:
    # This checks if the package exists without importing it
    package_spec = importlib.util.find_spec(pkg)
    if package_spec is None:
        missing.append(pkg)

if missing:
    print(f"Error: Missing required libraries: {', '.join(missing)}")
    print("Please install them using: pip install " + " ".join(missing))
    sys.exit(1) # Stop the script safely
else:
    print("All dependencies found.")


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import numpy as np
from collections import OrderedDict, defaultdict
#from typing import List
from tabulate import tabulate
from monty.termcolor import cprint
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.periodic_table import Element
import param
import seaborn as sns





class FatbandsPlotter:
    """
    This class implements tool to plot extract data from the FATBANDS.nc file from
    ABINIT, and generate fatband plots.
    """

    
    def __init__(self, nc_file, **params):
        ######################## constants ######################
        self.Ha_to_eV=27.2114 ###  1 hartree to eV = 27.2114 eV
        ########################
        self.ncfile = xr.open_dataset(nc_file)

        self.prtdos=self.ncfile['prtdos']
        self.atom_spc = self.ncfile['atom_species']
        self.atom_number = self.ncfile['atomic_numbers']                
        self.lmax = self.ncfile['lmax_type']
        self.iatsph = self.ncfile["iatsph"] - 1 ### first index = 0
        self.prtdos = self.ncfile["prtdos"]
        self.prtdosm = self.ncfile["prtdosm"]
        self.dos_fractions = None

        self.no_bands = self.ncfile.dims['max_number_of_states']
        self.nkpoints = self.ncfile.dims['number_of_kpoints']
        self.eigenv = self.ncfile['eigenvalues']
        self.natsph = self.ncfile.dims['natsph']
        self.natom = self.ncfile.dims['number_of_atoms']
        # If usepaw == 0, lmax_type gives the max l included in the non-local part of Vnl
        #   The wavefunction can have l-components > lmax_type, especially if vloc = vlmax.
        self.mbesslang = self.ncfile.dims["ndosfraction"]//self.natsph
        self.nsppol = self.ncfile.dims['number_of_spins']
        self.l_to_symbol = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4:'g'}

        ## atom species and their index
        self.species_map = {i+1: Element.from_Z(n).symbol for i, n in enumerate(self.atom_number)} 
        ## atom species and their max angular momentum
        self.lmax_map = {self.species_map[i+1]: l for i, l in enumerate(self.lmax.values)}
        ## lmax of the full system
        self.lsize = max(self.lmax_map.values()) + 1 #### lmax of the system + 1
        ## atom type for each atom in the system
        self.elements_system = {idx: self.species_map[val] for idx, val in enumerate(self.atom_spc.values)}
        ## lmax for each atom in the system
        self.lmax_atoms = {idx: self.lmax_map[val] for idx, val in enumerate(self.elements_system.values())}

        if not (self.prtdos == 3):
            raise ValueError(
                "Fatbands plots with LM-character require prtdos = 3 "                
            )

        
        
        ## separate atom index per chemical symbol, .e.g, ['Pb': [1,2,3], 'C':[2,3,...], ...]
        self.symbol2indices={}
        for index, symbol  in self.elements_system.items():
            if symbol not in self.symbol2indices:
                self.symbol2indices[symbol]=[]
            self.symbol2indices[symbol].append(index)
        for symbol in self.symbol2indices:
            self.symbol2indices[symbol]=np.array(self.symbol2indices[symbol])


    def export_variables(self):
            """
            Creates a pandas DataFrame containing variable names, 
            their dimensions, and their shapes.
            """
            data = []
            filename='./variables.csv'
            for var_name in self.ncfile.data_vars:
                var = self.ncfile[var_name]
                data.append({
                    "Variable Name": var_name,
                    "Dimensions": ", ".join(var.dims),
                    "Shape": var.shape,
                })
            df_vars = pd.DataFrame(data)
            df_vars.to_csv(filename, index=False)
            print(f"Variable metadata exported to: {filename}")
            return pd.DataFrame(data)

    def export_dimensions(self):
            """
            Creates a CSV summarizing global dimensions and their sizes.
            """
            filename_dimensions = './dimensions.csv'
            
            # Accessing dimensions from the dataset object
            dim_list = [
                {"Dimension Name": name, "Size": size} 
                for name, size in self.ncfile.dims.items()
            ]
            
            #print(self.ncfile.dims.items())
            #print(self.ncfile.dims)


            df_dims = pd.DataFrame(dim_list)
            df_dims.to_csv(filename_dimensions, index=False)
            print(f"Dimensions summary exported to: {filename_dimensions}")
            return df_dims

    @lazy_property
    def wal_sbk(self):
        # Read dos_fraction_m from file and build wal_sbk array of shape
        # [natom, lmax, nsppol, mband, nkpt].
        #
        # In abinit the **Fortran** array has shape 
        #--> self.ncfile['dos_fractions'](ndosfraction, number_of_spins, max_number_of_states, number_of_kpoint)
        # ndosfraction=natoms*5
        #   dos_fractions(nkpt,mband,nsppol,ndosfraction)
        #
        # Note that Abinit allows the users to select a subset of atoms with iatsph. Moreover the order
        # of the atoms could differ from the one in the structure even when natom == natsph (unlikely but possible).
        # To keep it simple, the code always operate on an array dimensioned with the total number of atoms
        # Entries that are not computed are set to zero and a warning is issued.
    
        if self.prtdos != 3:
            raise RuntimeError(f"The file does not contain L-DOS since {self.prtdos=}")

        wshape = (self.natom, self.mbesslang, self.nsppol, self.no_bands, self.nkpoints)

        if self.natsph == self.natom and np.all(self.iatsph == np.arange(self.natom)):
            # All atoms have been calculated and the order if ok.
            wal_sbk = np.reshape(self.ncfile['dos_fractions'].values, wshape)
            #print(wal_sbk.shape)

        else:
            wal_sbk = np.zeros(wshape)
            if self.natsph == self.natom and np.any(self.iatsph != np.arange(self.natom)):
                print("Will rearrange filedata since iatsp != [1, 2, ...])")
                filedata = np.reshape(self.ncfile['dos_fractions'].values, wshape)
                for i, iatom in enumerate(self.iatsph):                    
                    wal_sbk[iatom] = filedata[i]
            else:
                print("natsph < natom. Will set to zero the PJDOS contributions for the atoms that are not included.")
                assert self.natsph < self.natom
                filedata = np.reshape(self.ncfile['dos_fractions'].values,
                                        (self.natsph, self.mbesslang, self.nsppol, self.no_bands, self.nkpoints))
                for i, iatom in enumerate(self.iatsph):
                    wal_sbk[iatom] = filedata[i]
        
        return wal_sbk    # Return it

    @lazy_property
    def walm_sbk(self):
        # Read dos_fraction_m from file and build walm_sbk array of shape
        # [natom, lmax**2, nsppol, mband, nkpt].
        #
        # In abinit the **Fortran** array has shape
        # dos_fractions_m(dos_fractions_m_lastsize, number_of_spins, max_number_of_states, number_of_kpoints)
        # dos_fractions_m_lastsize=mbesslang**2
        # Note that Abinit allows the users to select a subset of atoms with iatsph. Moreover the order
        # of the atoms could differ from the one in the structure even when natom == natsph (unlikely but possible).
        # To keep it simple, the code always operate on an array dimensioned with the total number of atoms
        # Entries that are not computed are set to zero and a warning is issued.

        if not (self.prtdos == 3 and self.prtdosm != 0):
            cprint("Fatbands plots with LM-character require prtdosm != 0`", "red")
            return None
        
        wshape = (self.natom, self.mbesslang**2, self.nsppol, self.no_bands, self.nkpoints)

        if self.natsph == self.natom and np.all(self.iatsph == np.arange(self.natom)):
            # All atoms have been calculated and the order if ok.
            walm_sbk = np.reshape(self.ncfile['dos_fractions_m'].values, wshape)
            #print(walm_sbk.shape)

        else:
            # Need to transfer data. Note np.zeros.
            walm_sbk = np.zeros(wshape)
            if self.natsph == self.natom and np.any(self.iatsph != np.arange(self.natom)):
                print("Will rearrange filedata since iatsp != [1, 2, ...])")
                filedata = np.reshape(self.ncfile['dos_fractions_m'].values, wshape)
                for i, iatom in enumerate(self.iatsph):
                    walm_sbk[iatom] = filedata[i]
            else:
                print("natsph < natom. Will set to zero the PJDOS contributions for the atoms that are not included.")
                assert self.natsph < self.natom
                filedata = np.reshape(self.ncfile['dos_fractions_m'].values,
                                     (self.natsph, self.mbesslang**2, self.nsppol,  self.no_bands, self.nkpoints))
                for i, iatom in enumerate(self.iatsph):
                    walm_sbk[iatom] = filedata[i]

        # In principle, this should never happen (unless there's a bug in Abinit or a
        # very bad cancellation between the FFT and the PS-PAW term (pawprtden=0).
        num_neg = np.sum(walm_sbk < 0)
        if num_neg:
            print("WARNING: There are %d (%.1f%%) negative entries in LDOS weights" % (
                  num_neg, 100 * num_neg / walm_sbk.size))

        return walm_sbk



    def get_wl_symbol(self, symbol, spin=None, band=None) -> np.ndarray:
        """
        Return the l-dependent DOS weights for a given atomic type specified in terms of the
        chemical symbol ``symbol``. The weights are summed over m and over all atoms of the same type.
        If ``spin`` and ``band`` are not specified, the weights
        for all spins and bands else the contribution for (spin, band) are returned.
        """
        if spin is None and band is None:
            wl = np.zeros((self.lsize, self.nsppol, self.no_bands, self.nkpoints))
            for iat in self.symbol2indices[symbol]:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l] += self.wal_sbk[iat, l]
        else:
            assert spin is not None and band is not None
            wl = np.zeros((self.lsize, self.nkpoints))
            for iat in self.symbol2indices[symbol]:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l, :] += self.wal_sbk[iat, l, spin, band, :]

        return wl


    def get_wl_sets(self, atom_subset, spin=None, band=None) -> np.ndarray:
        """
        Return l-dependent DOS weights for a given atomic subset.
        The weights are summed over m and over all atoms of the same type.
        If ``spin`` and ``band`` are not specified, the weights
        for all spins and bands else the contribution for (spin, band) are returned.
        """
        if spin is None and band is None:
            wl = np.zeros((self.lsize, self.nsppol, self.no_bands, self.nkpoints))
            for iat in atom_subset:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l] += self.wal_sbk[iat, l]
        else:
            assert spin is not None and band is not None
            wl = np.zeros((self.lsize, self.nkpoints))
            for iat in atom_subset:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l, :] += self.wal_sbk[iat, l, spin, band, :]

        return wl


    def get_wlm_sets(self, atom_subset, l_val=None, spin=None, band=None) -> np.ndarray:
        """
        Return m-resolved DOS weights for a given atomic subset and angular momentum l.
        The weights are summed the atoms in the substed for each m value
        If ``spin`` and ``band`` are not specified, the weights
        for all spins and bands else the contribution for (spin, band) are returned.
        """
        m_vals=2 * l_val + 1
        if spin is None and band is None:
            wl = np.zeros((m_vals, self.nsppol, self.no_bands, self.nkpoints))
            for iat in atom_subset:
                for m in range(m_vals):
                    wl[m] += self.walm_sbk[iat, l_val**2+m]
        else:
            assert spin is not None and band is not None
            wl = np.zeros((m_vals, self.nkpoints))
            for iat in atom_subset:
                for m in range(m_vals):
                    wl[m, :] += self.walm_sbk[iat, l_val**2+m, spin, band, :]

        return wl


    def get_wlm_symbol(self, symbol, l_val=None, spin=None, band=None) -> np.ndarray:
        """
        Return m-resolved DOS weights for a given atomic subset and angular momentum l.
        The weights are summed the atoms in the substed for each m value
        If ``spin`` and ``band`` are not specified, the weights
        for all spins and bands else the contribution for (spin, band) are returned.
        """
        m_vals=2 * l_val + 1
        if spin is None and band is None:
            wl = np.zeros((m_vals, self.nsppol, self.no_bands, self.nkpoints))
            for iat in self.symbol2indices[symbol]:
                for m in range(m_vals):
                    wl[m] += self.walm_sbk[iat, l_val**2+m]
        else:
            assert spin is not None and band is not None
            wl = np.zeros((m_vals, self.nkpoints))
            for iat in self.symbol2indices[symbol]:
                for m in range(m_vals):
                    wl[m, :] += self.walm_sbk[iat, l_val**2+m, spin, band, :]

        return wl






    def get_spilling(self, spin=None, band=None):
        """
        Return the spilling parameter --> electronic part that is not captured by local basis set
        If ``spin`` and ``band`` are not specified, the method returns the spilling for all states
        as a [nsppol, mband, nkpt] numpy array else the spilling for (spin, band) with shape [nkpt].
        """
        if spin is None and band is None:
            sp = np.zeros((self.nsppol, self.no_bands, self.nkpoints))
            for iatom in range(self.natom):
                #print(iatom)
                for l in range(self.lmax_atoms[iatom]+1):
                    sp += self.wal_sbk[iatom, l]
        else:
            assert spin is not None and band is not None
            sp = np.zeros((self.nkpoints))
            for iatom in range(self.natom):
                for l in range(self.lmax_atoms[iatom]+1):
                    sp += self.wal_sbk[iatom, l, spin, band, :]

        return 1.0 - sp


    @lazy_property
    def bands_eV(self):               
        bands = self.ncfile['eigenvalues'].transpose("number_of_spins", "max_number_of_states", "number_of_kpoints")
        ##################### convert to eV and shift to E0
        bands_in_eV = bands * self.Ha_to_eV
        return  bands_in_eV

    @staticmethod
    def get_high_contrast_colors(n):
        # 'hls' or 'husl' spreads the colors evenly across the human visual spectrum
        return sns.color_palette("husl", n)

    @staticmethod
    def get_colors(n):
        #"""Returns a list of RGBA color values for n_atoms."""
        # 'tab10' has 10 colors. 'tab20' has 20.
        if n <= 10:
            cmap = plt.get_cmap('tab10')
        else:
            cmap = plt.get_cmap('tab20')
        # cmap(i) returns a tuple like (0.12, 0.46, 0.70, 1.0)
        # This list comprehension returns exactly what you asked for.
        return [cmap(i) for i in range(n)]


    def _save_and_handle_kwargs(self, fig, **kwargs):
            """
            Internal helper to handle figure saving and common Matplotlib tweaks.
            Removes 'save_path' and 'dpi' from kwargs to prevent conflicts.
            """
            # 1. Extract and remove our custom 'save_path'
            save_path = kwargs.pop('save_path', None)
            
            if save_path is not None:
                if not save_path:
                    raise ValueError("save_path cannot be an empty string.")

                # 2. Extract and remove 'dpi' with a default
                dpi = kwargs.pop('dpi', 300)
                
                # 3. Extract 'bbox_inches' with a default
                bbox = kwargs.pop('bbox_inches', 'tight')

                # 4. Save using the remaining kwargs (e.g., facecolor, transparent)
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox, **kwargs)
                print(f"Figure saved to: {save_path}")
            
            # Optional: return the remaining kwargs if you need them for show()
            return kwargs

    def _apply_standard_formatting(self, ax, l, xticks=None, xval_ticks=None, ylims=None, xlims=None):
        """
        Helper to apply consistent styling, labels, and axis limits to band plots.
        """
        # Set the title based on the angular momentum symbol
        l_name = self.l_to_symbol.get(l, str(l))
        ax.set_title(f'l = {l_name}', pad=35, fontsize=14, fontweight='bold')
        
        # Axis labels
        ax.set_xlabel('K-point', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)

        # Handle Y-axis limits (Energy)
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])

        # Handle X-axis limits (K-points)
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])

        # Handle Custom K-point Ticks (High Symmetry Points)
        if xval_ticks is not None:
            # If positions are provided, we can set them. 
            # Labels (xticks) are optional but recommended if xval_ticks is used.
            ax.set_xticks(xval_ticks)
            if xticks is not None:
                ax.set_xticklabels(xticks)
        elif xticks is not None:
            # If the user gave labels but forgot positions, it's an error
            raise ValueError("xticks provided without xval_ticks. Cannot place labels.")

        # Standard Legend Formatting
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1), # Places legend outside the plot area
            borderaxespad=0.,
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=False
        )
        


###########################################################################################
###########################################################################################
###########################################################################################
    def plot_fatbands_symbol(self, e0=0, band_list=None, l=None, symbol=None, symbol_m=None, 
                            colors=None, colors_m=None, fact=1.0, transparency=0.5,
                            xticks=None, xval_ticks=None, ylims=None, xlims=None, **kwargs):
        """
        Plot the L and LM-decomposed electronic fatbands for the atom type given by symbol and symbol_m respectively.
        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            l: Angular momentum used to calculate the orbital projection.
            colors and colors_m: list containing the colores used for plotting the contributions 
                                 of the L and LM resolve plots respectively.
            symbol: Define the atomic species for the the L decomposition.
            symbol_m: Define the atomic species for the the LM decomposition.
            fact: float used to scale the stripe size.
            transparency: controls the transparency of the stripes
            xticks: list with labels of the ticks.
            xval_ticks: list containing the values where the ticks will be located.
            ylims: list used to define limits for the y-axis.
            xlims: list used to define limits for the x-axis.
            save_path: for saving the figure in the specified path './path/name.png'.
            dpi: resolution of the saved fig.
            format: format of the fig, e.g, pdf, png, etc.
        Returns: |matplotlib-Figure|
        """
        if l is None:            
            raise ValueError("Angular momentum not specified.")
        if symbol is None and symbol_m is None:
            raise ValueError("No atom types were specified.")

        ####################PREPARE AND VALIDATE SYMBOL LISTS ####################################
        valid_species = set(self.species_map.values())
        
        atm_symbols = ([symbol] if isinstance(symbol, str) else list(symbol)) if symbol is not None else []
        atm_symbols_m = ([symbol_m] if isinstance(symbol_m, str) else list(symbol_m)) if symbol_m is not None else []

        ################### CHECK FOR OVERLAP ###################
        # Plot the L and LM contribution of the same atoms is redundant.
        overlap = set(atm_symbols).intersection(set(atm_symbols_m))
        if overlap:
            raise ValueError(f"Elements cannot be repeated in both 'symbol' and 'symbol_m'. "
                             f"Repeated: {list(overlap)}")

        # Validate existence in file
        for s in atm_symbols + atm_symbols_m:
            if s not in valid_species:
                raise ValueError(f"Incorrect element: '{s}'. Valid species: {valid_species}")

        ################################ COLOR MANAGEMENT ########################
        m_vals = 2 * l + 1
        n_main = len(atm_symbols)
        n_m = len(atm_symbols_m) * m_vals
        
        full_palette = self.get_colors(n_main + n_m)

        if atm_symbols and colors is None:
            colors = full_palette[:n_main]
        elif atm_symbols and len(colors) != n_main:
            raise ValueError(f"Colors must contain {n_main} elements.")

        if atm_symbols_m and colors_m is None:
            colors_m = full_palette[n_main:]
        elif atm_symbols_m and len(colors_m) != n_m:
            raise ValueError(f"Colors_m must contain {n_m} elements.")

        ################################################# INITIALIZE PLOT ###################################
        fig, ax = plt.subplots(figsize=(8, 6))
        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list


        idx_to_name = {
            0: 's',
            1: 'py',
            2: 'pz',
            3: 'px',
            4: 'dxy',
            5: 'dyz',
            6: 'dz2',
            7: 'dxz',
            8: 'dx2-y2',
            9: 'f_y3x2',
            10: 'f_xyz',
            11: 'f_yz2',
            12: 'f_z3',
            13: 'f_xz2',
            14: 'f_zx2y2',
            15: 'f_x3y2',
            16: 'g0',
            17: 'g1',
            18: 'g2',
            19: 'g3',
            20: 'g4',
            21: 'g5',
            22: 'g6',
            23: 'g7',
            24: 'g8'
        }

        ############################ PLOTTING LOOPS ##################################
        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                # Background band structure
                ax.plot(x, ebands[spin, band, :], color='black', zorder=0)
                
                y_base = ebands[spin, band, :]

                ############## DRAW TOTAL L #########################
                for idx, s in enumerate(atm_symbols):
                    w = self.get_wl_symbol(s, spin=spin, band=band)[l] * (fact / 2)
                    ax.fill_between(x, y_base - w, y_base + w, alpha=transparency, 
                                    facecolor=colors[idx], label=s if ib == 0 else None)

                ############## DRAW TOTAL M #########################
                for idx, s in enumerate(atm_symbols_m):
                    wlk_m = self.get_wlm_symbol(symbol=s, l_val=l, spin=spin, band=band) * (fact / 2)
                    for mval in range(m_vals):
                        w = wlk_m[mval]
                        c = colors_m[mval + m_vals*idx]
                        lbl = f"{s}-{idx_to_name[l**2 + mval]}" if ib == 0 else None
                        ax.fill_between(x, y_base - w, y_base + w, alpha=transparency, 
                                        facecolor=c, label=lbl)

        #####################5. FINAL FORMATTING ###########################
        self._apply_standard_formatting(ax, l, xticks, xval_ticks, ylims, xlims)
        self._save_and_handle_kwargs(fig, **kwargs)

        return fig, ax
        
###########################################################################################
###########################################################################################
###########################################################################################

    def plot_fatbands_atomsets(self, e0=0, band_list=None, l=None, 
                              atom_set=None, atom_set_m=None, 
                              colors=None, colors_m=None,
                              fact=1.0, transparency=0.5, 
                              xticks=None, xval_ticks=None, 
                              ylims=None, xlims=None, **kwargs):
        """
        Plot the m-decomposed electronic fatbands for a specific L for each atomic subset.
        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            l: Angular momentum used to calculate the orbital projection.
            colors and colors_m: list containing the colores used for plotting the contributions 
                                 of the L and LM resolve plots respectively.
            atom_set: Define the subset for the the L decomposition.
            atom_set_m: Define the subset for the the LM decomposition.           
            fact: float used to scale the stripe size.
            transparency: controls the transparency of the stripes
            xticks: list with labels of the ticks.
            xval_ticks: list containing the values where the ticks will be located.
            ylims: list used to define limits for the y-axis.
            xlims: list used to define limits for the x-axis.
            save_path: for saving the figure in the specified path './path/name.png'.
            dpi: resolution of the saved fig.
            format: format of the fig, e.g, pdf, png, etc.
        Returns: |matplotlib-Figure|
        """
        if l is None:            
            raise ValueError("Angular momentum not specified.")
        if atom_set is None and atom_set_m is None:
            raise ValueError("At least one atom subset (atom_set or atom_set_m) must be specified")

        ######NORMALIZE AND VALIDATE ATOM SETS###############
        def normalize_sets(aset):
            if aset is None: return []
            # Handle [1, 2] -> [[1, 2]]
            if isinstance(aset[0], (int, np.integer)):
                aset = [aset]
            # Validate indices
            flat = [idx for subset in aset for idx in subset]
            if any(i >= self.natom or i < 0 for i in flat):
                raise ValueError(f"Invalid atom indices in {aset}")
            return aset

        sets = normalize_sets(atom_set)
        sets_m = normalize_sets(atom_set_m)

        ################### CHECK FOR OVERLAP ###################
        # Plot the L and LM contribution of the same atoms is redundant.
        # Compare them as tuples
        set_tuples = [tuple(sorted(s)) for s in sets]
        set_m_tuples = [tuple(sorted(s)) for s in sets_m]
        overlap = set(set_tuples).intersection(set(set_m_tuples))
        if overlap:
            raise ValueError(f"The following atom subsets are repeated: {overlap}")

        ################################ COLOR MANAGEMENT ########################
        m_vals = 2 * l + 1
        n_main = len(sets)
        n_m = len(sets_m) * m_vals
        
        all_colors = self.get_colors(n_main + n_m)

        if sets and colors is None:
            colors = all_colors[:n_main]
        if sets_m and colors_m is None:
            colors_m = all_colors[n_main:]

        ################################################# INITIALIZE PLOT ###################################
        fig, ax = plt.subplots(figsize=(8, 6))
        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list
        idx_to_name = {
            0: 's',
            1: 'py',
            2: 'pz',
            3: 'px',
            4: 'dxy',
            5: 'dyz',
            6: 'dz2',
            7: 'dxz',
            8: 'dx2-y2',
            9: 'f_y3x2',
            10: 'f_xyz',
            11: 'f_yz2',
            12: 'f_z3',
            13: 'f_xz2',
            14: 'f_zx2y2',
            15: 'f_x3y2',
            16: 'g0',
            17: 'g1',
            18: 'g2',
            19: 'g3',
            20: 'g4',
            21: 'g5',
            22: 'g6',
            23: 'g7',
            24: 'g8'
        }

        # Plot underlying band structure
        for spin in range(self.nsppol):               
            for i in mybands:
                ax.plot(x, ebands[spin,i,:], color='black', zorder=0)

        ########################### PLOTTING LOOPS #############################################
        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                y_base = ebands[spin, band, :]
                
                ############## DRAW TOTAL L SUBSETS #########################
                for idx, a_set in enumerate(sets):
                    w = self.get_wl_sets(atom_subset=a_set, spin=spin, band=band)[l] * (fact/2)
                    ax.fill_between(x, y_base-w, y_base+w, facecolor=colors[idx], 
                                    alpha=transparency, label=f"Set {idx+1}" if ib==0 else None)

                ############## DRAW M-DECOMPOSED SUBSETS ########################
                for idx, a_set in enumerate(sets_m):
                    w_m = self.get_wlm_sets(atom_subset=a_set, spin=spin, l_val=l, band=band) * (fact/2)
                    for mval in range(m_vals):
                        w = w_m[mval]
                        c = colors_m[mval + m_vals*idx]
                        lbl = f"Set {idx+1}-{idx_to_name[l**2+mval]}" if ib==0 else None
                        ax.fill_between(x, y_base-w, y_base+w, facecolor=c, alpha=transparency, label=lbl)

        ####################### FORMATTING (Standard for both) #################################
        self._apply_standard_formatting(ax, l, xticks, xval_ticks, ylims, xlims)
        self._save_and_handle_kwargs(fig, **kwargs)
        return fig, ax








 


# --- Execution ---
viewer = FatbandsPlotter("../sample_data/Pb_SiCo_FATBANDS.nc")
#viewer.walm_sbk

#self.iatsph = self.ncfile["iatsph"]

# Export both files
#print(viewer.species_map)
#print(viewer.lmax_map)
#print(viewer.elements_system)
#print(viewer.lmax_atoms)
#print(viewer.iatsph.values)
#print(viewer.wal_sbk)
Pb=atom1=[0,1,2]
#Gr=atom1=[3,4,5,6,7,8,9,10]
#SiC=list(range(11,50))
at_sets=[Pb]#,Gr,SiC]


viewer.plot_fatbands_symbol(band_list=list(range(150,250)), e0=2.77561,
                                l=1,ylims=[-2,2], symbol='Pb',
                                xval_ticks=[0,30,60,90])
#viewer.plot_fatbands_l(band_list=list(range(150,250)), l=0)
plt.show()

#viewer.plot_fatbands_l(band_list=list(range(150,250)), e0=2.77561,
#                                l=1,ylims=[-2,2],
#                                xval_ticks=[0,30,60,90],
#                                save_path='./test_2.png')
#plt.show()

#for i in range(444):
#    plt.plot(viewer.get_bands[0,i,:])
#plt.show()
#print(viewer.symbol2indices)
#wl=viewer.get_wl_sets(atom_set=[0,1,2])
#print(wl.shape)
#sp=viewer.get_spilling()
#print(sp.shape)
#print(sp[0,222,:])
#print(viewer.lsize)
#test=viewer.wal_sbk()
#dimensions=viewer.export_dimensions()

#print(viewer.ncfile['eigenvalues'])
#print(viewer.ncfile.dims['max_number_of_states'])

#if viewer.prtdos == 3:
# print('it is taking the value', viewer.prtdos.values)
#print(viewer.prtdos.values)

#print(viewer.mbesslang)
#eigenv=viewer.ncfile['eigenvalues']
#elements, lmax= viewer.atomic_species()
#print(elements)
#print(lmax)
#print(viewer.atomic_species())
#print(viewer.ncfile['lmax_type'].values)
#print(elements_system)



#plt.plot(kpoints,eigenv[0,:,:])
#plt.show()







