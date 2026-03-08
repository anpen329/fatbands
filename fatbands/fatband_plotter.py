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
            print(walm_sbk.shape)

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


    def get_wl_symbol_sets(self, atom_subset, spin=None, band=None) -> np.ndarray:
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


    def get_wlm_symbol_sets(self, atom_subset, l_val, spin=None, band=None) -> np.ndarray:
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

    def plot_fatbands_l(self, e0=0, band_list=None, spin=None, l=None, colors=None, symbol=None,
                         fact=1.0, transparency=0.5,xticks=None,xval_ticks=None, ylims=None, xlims=None, **kwargs):

        """
        Plot the electronic fatbands for a specific L. Atoms are grouped by type.

        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            l: Angular momentum used to calculate the orbital projection.
            colors: list containing the colores used for each stripe.
            symbols: Atom type(s) included in the fatbands. Can be a string e.g 'Pb'. A list or an array of strings. 
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
        # string elements in self.species_map.values()
        valid_species = set(self.species_map.values())
  
        #  If nothing is specified, use all species in the system
        if symbol is None:            
            atm_symbols = valid_species 

        elif isinstance(symbol, str):
            # If it's a single string "Si", wrap it in a list ["Si"]
            atm_symbols = [symbol]
        elif isinstance(symbol, (list, np.ndarray)):
            # If it's already a list or numpy array, use it directly
            atm_symbols = list(symbol)
        else:
            # Handle Xarray or other iterables safely
            atm_symbols = list(symbol)
        #Check that every requested symbol actually exists in the file
        
        for s in atm_symbols:
            if s not in valid_species:
                raise ValueError(f"Incorrect element specified: '{s}'. "
                                 f"Valid species in this file are: {valid_species}")


        fig, ax= plt.subplots(figsize=(8, 6))

        if colors is None:
            colors = self.get_colors(len(self.species_map))
        elif len(colors) != len(self.species_map):
            raise ValueError("Colors must contain the same elements as atom types.")

        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list
        
        for i in mybands:
            ax.plot(x, ebands[0,i,:], color='black')

        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                yup = ebands[spin, band,:]
                ydown = yup
                for idx, symbol in enumerate(atm_symbols):
                    wlk = self.get_wl_symbol(symbol, spin=spin, band=band) * (fact / 2)
                    w = wlk[l]
                    #print(w.shape)
                    y1, y2 = yup + w, ydown - w
                    # Add width around each band. Only the [0,0] plot has the legend.
                    ax.fill_between(x, yup, y1, alpha=transparency, facecolor=colors[idx])
                    ax.fill_between(x, ydown, y2, alpha=transparency, facecolor=colors[idx],
                                    label=symbol if ib == 0 else None)
        ax.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.015), 
            ncol=len(self.species_map), 
            shadow=False, 
            frameon=True
        )
        ax.set_title('l=' + self.l_to_symbol[l], pad=35)
        ax.set_xlabel('K-point')
        ax.set_ylabel('Energy (eV)')

        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])

        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])

        if xval_ticks is not None:
            # Si hay posiciones, las ponemos. Si además hay etiquetas, se pasan como 'labels'
            ax.set_xticks(xval_ticks, labels=xticks)
        elif xticks is not None:
            # Si hay etiquetas pero no posiciones (xval_ticks es None)
            raise ValueError("Values for the ticks not defined")

        self._save_and_handle_kwargs(fig, **kwargs)


        return fig, ax

    def plot_fatbands_l_atomsets(self, e0=0, band_list=None, spin=None, l=None, atom_set=None, colors=None,
                         fact=1.0, transparency=0.5,xticks=None,xval_ticks=None, ylims=None, xlims=None, **kwargs):

        """
        Plot the electronic fatbands for a specific L for each subset of atoms defined

        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            l: Angular momentum used to calculate the orbital projection.
            atom_set: subsets of atoms for which the orbital projected will be calculated.
            colors: list containing the colores used for each stripe.
            fact: float used to scale the stripe size.
            transparency: controls the transparency of the stripes
            xticks: list with labels of the ticks.
            xval_ticks: list containing the values where the ticks will be located.
            ylims: List used to define limits for the y-axis.
            xlims: List used to define limits for the x-axis.
            save_path: for saving the figure in the specified path './path/name.png'.
            dpi: resolution of the saved fig.
            format: format of the fig, e.g, pdf, png, etc.
        Returns: |matplotlib-Figure|
        """
        #### Checking if the atom indices are correct ####

        if atom_set is None:
            raise ValueError("At least one atom subset must be specified")
    
        # If it's a list of ints, make it a list of one list
        if atom_set is not None:
            # Check if the first element is an integer. 
            # If so, the user passed [1, 2, 3] instead of [[1, 2, 3]]
            if isinstance(atom_set[0], (int, np.integer)):
                atom_set = [atom_set]

        flat_at_list = [atom_idx for subset in atom_set for atom_idx in subset]
        missing_elements = set(flat_at_list) - set(range(self.natom))
        if missing_elements:
            raise ValueError(f"The following atom indices are not valid: {missing_elements}")

        if colors is None:
            colors = self.get_colors(len(atom_set))
        elif len(colors) != len(atom_set):
            raise ValueError("Colors must contain the same elements as atomic subsets.")

        

        fig, ax= plt.subplots(figsize=(8, 6))


        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list

        for i in mybands:
            ax.plot(x, ebands[0,i,:], color='black')

        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                yup = ebands[spin, band,:]
                ydown = yup
                for set_idx, at_set in enumerate(atom_set):
                    wlk = self.get_wl_symbol_sets(atom_subset=at_set, spin=spin, band=band) * (fact / 2)
                    w = wlk[l]
                    #print(w.shape)
                    y1, y2 = yup + w, ydown - w
                    # Add width around each band. Only the [0,0] plot has the legend.
                    ax.fill_between(x, yup, y1, alpha=transparency, facecolor=colors[set_idx])
                    ax.fill_between(x, ydown, y2, alpha=transparency, facecolor=colors[set_idx],
                                    label=f"set {set_idx+1}"  if ib == 0 else None)
        ax.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.015), 
            ncol=len(atom_set), 
            shadow=False, 
            frameon=True
        )
        ax.set_title('l=' + self.l_to_symbol[l], pad=35)
        ax.set_xlabel('K-point')
        ax.set_ylabel('Energy (eV)')

        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])

        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])

        if xval_ticks is not None:
            # Si hay posiciones, las ponemos. Si además hay etiquetas, se pasan como 'labels'
            ax.set_xticks(xval_ticks, labels=xticks)
        elif xticks is not None:
            # Si hay etiquetas pero no posiciones (xval_ticks es None)
            raise ValueError("Values for the ticks not defined")


        self._save_and_handle_kwargs(fig, **kwargs)

        return fig, ax

    def plot_fatbands_mview(self, e0=0, band_list=None, spin=None, l=None, atom_set=None, colors=None,
                         fact=1.0, transparency=0.5,xticks=None,xval_ticks=None, ylims=None, xlims=None, **kwargs):
        """
        Plot the electronic fatbands grouped by LM.

        Args:
            iatom: Index of the atom in the structure.
            e0: Option used to define the zero of energy in the band structure plot. Possible values:
                - ``fermie``: shift all eigenvalues to have zero energy at the Fermi energy.
                -  Number e.g ``e0 = 0.5``: shift all eigenvalues to have zero energy at 0.5 eV
                -  None: Don't shift energies, equivalent to ``e0 = 0``
            fact:  float used to scale the stripe size.
            ylims: Set the data limits for the y-axis. Accept tuple e.g. ``(left, right)``
                   or scalar e.g. ``left``. If left (right) is None, default values are used
            blist: List of band indices for the fatband plot. If None, all bands are included

        Returns: |matplotlib-Figure|
        """
        fig, ax= plt.subplots(figsize=(8, 6))

        m_vals = 2 * l + 1


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
        
        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list

        
#        for spin in range(self.nsppol):
#            ebands.plot_ax(ax, e0, spin=spin, **self.eb_plotax_kwargs(spin))


        if atom_set is None:
            raise ValueError("At least one atom subset must be specified")
    
        # If it's a list of ints, make it a list of one list
        if atom_set is not None:
            # Check if the first element is an integer. 
            # If so, the user passed [1, 2, 3] instead of [[1, 2, 3]]
            if isinstance(atom_set[0], (int, np.integer)):
                atom_set = [atom_set]

        flat_at_list = [atom_idx for subset in atom_set for atom_idx in subset]
        missing_elements = set(flat_at_list) - set(range(self.natom))
        if missing_elements:
            raise ValueError(f"The following atom indices are not valid: {missing_elements}")

        if colors is None:
            colors = self.get_colors(len(atom_set) * (2*l+1))
#        elif len(colors) != len(atom_set):
#            raise ValueError("Colors must contain the same elements as atomic subsets.")

        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                ax.plot(x, ebands[0,band,:], color='black', zorder=0) # band structure
                yup = ebands[spin, band,:]
                ydown = yup
                for set_idx, at_set in enumerate(atom_set): 
                    for mval in range(m_vals):                         
                        wlk_m = self.get_wlm_symbol_sets(atom_subset=at_set, spin=spin, l_val=l, band=band) * (fact / 2)
                        w = wlk_m[mval]
                        y1, y2 = yup + w, ydown - w
                        # Add width around each band.
                        ax.fill_between(x, yup, y1, alpha=transparency, facecolor=colors[mval + m_vals*set_idx])
                        ax.fill_between(x, ydown, y2, alpha=transparency, facecolor=colors[mval + m_vals*set_idx],
                        label=f"set {set_idx+1} - {idx_to_name[l**2 + mval]}"  if ib == 0 else None)                        

        ax.legend(
            loc='upper left',           # The point on the legend we are 'holding'
            bbox_to_anchor=(1.02, 1),   # Coordinates: slightly right of the plot (x=1.02), at the top (y=1)
            ncol=1,                     # Standard single column for a vertical list
            shadow=False, 
            frameon=True
        )
        ax.set_title('l=' + self.l_to_symbol[l], pad=35)
        ax.set_xlabel('K-point')
        ax.set_ylabel('Energy (eV)')

        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])

        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])

        if xval_ticks is not None:
            # Si hay posiciones, las ponemos. Si además hay etiquetas, se pasan como 'labels'
            ax.set_xticks(xval_ticks, labels=xticks)
        elif xticks is not None:
            # Si hay etiquetas pero no posiciones (xval_ticks es None)
            raise ValueError("Values for the ticks not defined")


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
viewer.plot_fatbands_mview(band_list=list(range(150,250)), e0=2.77561,
                                l=1,ylims=[-2,2], atom_set=at_sets,
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
#wl=viewer.get_wl_symbol_sets(atom_set=[0,1,2])
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







