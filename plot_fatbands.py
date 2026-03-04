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





class NcFileViewer:
    """
    This class implements toool to inspect dimensions and variables stored in a netcdf file.

    Relyes on the API provided by `AbinitNcFile` defined in core.mixins.py
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
        self.dos_fractions = None

        self.no_bands = self.ncfile.dims['max_number_of_states']
        self.nkpoints = self.ncfile.dims['number_of_kpoints']
        self.eigenv = self.ncfile['eigenvalues']
        self.natsph = self.ncfile.dims['natsph']
        self.natom = self.ncfile.dims['number_of_atoms']
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
                "Fatbands plots with LM-character require `prtdos = 3 "                
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
        # In abinit the **Fortran** array has shape --> self.ncfile['dos_fractions']
        #   dos_fractions(nkpt,mband,nsppol,ndosfraction)
        #
        # Note that Abinit allows the users to select a subset of atoms with iatsph. Moreover the order
        # of the atoms could differ from the one in the structure even when natom == natsph (unlikely but possible).
        # To keep it simple, the code always operate on an array dimensioned with the total number of atoms
        # Entries that are not computed are set to zero and a warning is issued.
        #for i, iatom in enumerate(self.iatsph.values):
            #print(i,'      ',iatom)       
        if self.prtdos != 3:
            raise RuntimeError(f"The file does not contain L-DOS since {self.prtdos=}")

        wshape = (self.natom, self.mbesslang, self.nsppol, self.no_bands, self.nkpoints)

        if self.natsph == self.natom and np.all(self.iatsph == np.arange(self.natom)):
            # All atoms have been calculated and the order if ok.
            wal_sbk = np.reshape(self.ncfile['dos_fractions'].values, wshape)
            #print(wal_sbk.shape)

        else:
            # Need to transfer data. Note np.zeros.
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


    def get_wl_symbol_sets(self, atom_set, spin=None, band=None) -> np.ndarray:
        """
        Return l-dependent DOS weights for a given atomic subset.
        The weights are summed over m and over all atoms of the same type.
        If ``spin`` and ``band`` are not specified, the weights
        for all spins and bands else the contribution for (spin, band) are returned.
        """
        if spin is None and band is None:
            wl = np.zeros((self.lsize, self.nsppol, self.no_bands, self.nkpoints))
            for iat in atom_set:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l] += self.wal_sbk[iat, l]
        else:
            assert spin is not None and band is not None
            wl = np.zeros((self.lsize, self.nkpoints))
            for iat in atom_set:
                for l in range(self.lmax_atoms[iat]+1):
                    wl[l, :] += self.wal_sbk[iat, l, spin, band, :]

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
    def get_atom_colors(n):
        #"""Returns a list of RGBA color values for n_atoms."""
        # 'tab10' has 10 colors. 'tab20' has 20.
        if n <= 10:
            cmap = plt.get_cmap('tab10')
        else:
            cmap = plt.get_cmap('tab20')
        # cmap(i) returns a tuple like (0.12, 0.46, 0.70, 1.0)
        # This list comprehension returns exactly what you asked for.
        return [cmap(i) for i in range(n)]

   
    def plot_fatbands_l(self, e0=0, band_list=None, spin=None, l=None,
                         fact=1.0, alpha=0.5,xticks=None,xval_ticks=None, ylims=None, xlims=None, **kwargs):

        """
        Plot the electronic fatbands for a specific L grouped by atom type

        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            fact: float used to scale the stripe size.
            l: Angular momentum used to calculate the orbital projection
            ylims: List used to define limits for the y-axis
            xlims: List used to define limits for the x-axis 
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            save_path: for saving the figure in the specified path './path/name.png'.
            dpi: resolution of the saved fig.
            format: format of the fig, e.g, pdf, png, etc.
        Returns: |matplotlib-Figure|
        """


        fig, ax= plt.subplots(figsize=(8, 6))


        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list
        colors = self.get_atom_colors(len(self.species_map))
        for i in mybands:
            ax.plot(x, ebands[0,i,:], color='black')

        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                yup = ebands[spin, band,:]
                ydown = yup
                for symbol in self.species_map:
                    wlk = self.get_wl_symbol(self.species_map[symbol], spin=spin, band=band) * (fact / 2)
                    w = wlk[l]
                    #print(w.shape)
                    y1, y2 = yup + w, ydown - w
                    # Add width around each band. Only the [0,0] plot has the legend.
                    ax.fill_between(x, yup, y1, alpha=0.5, facecolor=colors[symbol-1])
                    ax.fill_between(x, ydown, y2, alpha=0.5, facecolor=colors[symbol-1],
                                    label=self.species_map[symbol] if ib == 0 else None)
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

        # 1. Handle Saving Logic
        save_path = kwargs.pop('save_path')
        if save_path:
            # Set defaults for savefig, but allow kwargs to override them
            dpi = kwargs.pop('dpi', 300)
            # bbox_inches='tight' to keep the legend in the frame
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
            print(f"Figure saved to: {save_path}")


        return fig, ax

    def plot_fatbands_l_atomsets(self, e0=0, band_list=None, spin=None, l=None, atom_set=None,
                         fact=1.0, alpha=0.5,xticks=None,xval_ticks=None, ylims=None, xlims=None, **kwargs):

        """
        Plot the electronic fatbands for a specific L for each subset of atoms defined

        Args:
            e0: Option used to define the zero of energy in the band structure plot.
            fact: float used to scale the stripe size.
            l: Angular momentum used to calculate the orbital projection.
            atom_set: subsets of atoms for which the orbital projected will be calculated.
            ylims: List used to define limits for the y-axis.
            xlims: List used to define limits for the x-axis.
            band_list: List of band indices for the fatband plot. If None, all bands are included.
            save_path: for saving the figure in the specified path './path/name.png'.
            dpi: resolution of the saved fig.
            format: format of the fig, e.g, pdf, png, etc.
        Returns: |matplotlib-Figure|
        """
        #### Checking if the atom indices are correct ####
        flat_at_list = [atom_idx for subset in atom_set for atom_idx in subset]
        missing_elements = set(flat_at_list) - set(range(self.natom))
        if missing_elements:
            raise ValueError(f"The following atom indices are not valid: {missing_elements}")


        fig, ax= plt.subplots(figsize=(8, 6))


        ebands = self.bands_eV - e0        
        x = np.arange(self.nkpoints)
        mybands = list(range(self.no_bands)) if band_list is None else band_list
        colors = self.get_atom_colors(len(self.species_map))
        for i in mybands:
            ax.plot(x, ebands[0,i,:], color='black')

        for spin in range(self.nsppol):            
            for ib, band in enumerate(mybands):
                yup = ebands[spin, band,:]
                ydown = yup
                for set_idx, at_set in enumerate(atom_set):
                    wlk = self.get_wl_symbol_sets(atom_set=at_set, spin=spin, band=band) * (fact / 2)
                    w = wlk[l]
                    #print(w.shape)
                    y1, y2 = yup + w, ydown - w
                    # Add width around each band. Only the [0,0] plot has the legend.
                    ax.fill_between(x, yup, y1, alpha=0.5, facecolor=colors[set_idx])
                    ax.fill_between(x, ydown, y2, alpha=0.5, facecolor=colors[set_idx],
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

        # 1. Handle Saving Logic
        save_path = kwargs.pop('save_path')
        if save_path:
            # Set defaults for savefig, but allow kwargs to override them
            dpi = kwargs.pop('dpi', 300)
            # bbox_inches='tight' to keep the legend in the frame
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
            print(f"Figure saved to: {save_path}")

        return fig, ax











# --- Execution ---
viewer = NcFileViewer("./Pb_SiCo_FATBANDS.nc")


#self.iatsph = self.ncfile["iatsph"]

# Export both files
#print(viewer.species_map.values())
#print(viewer.lmax_map)
#print(viewer.elements_system)
#print(viewer.lmax_atoms)
#print(viewer.iatsph.values)
#print(viewer.wal_sbk)
#viewer.plot_fatbands_l(band_list=list(range(150,250)), l=1, xticks=['G','K','M','K'],xval_ticks=[0,30,60,90])
#viewer.plot_fatbands_l(band_list=list(range(150,250)), l=0)
#plt.show()
Pb=atom1=[0,1,2]
Gr=atom1=[3,4,5,6,7,8,9,10]
SiC=list(range(11,56))
at_sets=[Pb,Gr,SiC]
viewer.plot_fatbands_l_atomsets(band_list=list(range(150,250)), e0=2.77561,
                                l=1, atom_set=at_sets, xticks=['G','K','M','K'], 
                                xval_ticks=[0,30,60,90],
                                save_path='./test_1.png')
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
