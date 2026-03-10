from fatbands import FatbandsPlotter



############# example of usage
# Initialize with the example file
plotter = FatbandsPlotter("path_to_file_FATBANDS.nc")

# Call plot function for L (symbol) and LM (symbol_m) resolved fatbands for te atom type specified.
viewer.plot_fatbands_symbol(band_list=list(range(150,250)), e0=2.77561,
                                l=1,ylims=[-2,2], symbol=['Si','C'], symbol_m='Pb'
                                xticks=['G','K','M','K'], 
                                xval_ticks=[0,30,60,90],
                                save_path='./plot.png')


# Call plot function for L (atom_set) and LM (atom_set_m) resolved fatbands for the subsets specified.
Pb=atom1=[0,1,2]
Gr=atom1=[3,4,5,6,7,8,9,10]
SiC=list(range(11,50))

viewer.plot_fatbands_atomsets(band_list=list(range(150,250)), e0=2.77561,
                                l=1,ylims=[-2,2], atom_set=[0,1,2], atom_set_m=[[3,4,5,6,7,8,9,10],list(range(11,50))]
                                xticks=['G','K','M','K'], 
                                xval_ticks=[0,30,60,90],
                                save_path='./plot.png')