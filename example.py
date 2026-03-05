from fatbands import FatbandsPlotter



############# example of usage
# Initialize with the example file
plotter = FatbandsPlotter("sample_data/Pb_SiCo_FATBANDS.nc")

# Call plot function for atomic subsets l=p
Pb=atom1=[0,1,2]
Gr=atom1=[3,4,5,6,7,8,9,10]
SiC=list(range(11,50))
at_sets=[Pb,Gr,SiC]
plotter.plot_fatbands_l_atomsets(band_list=list(range(150,250)), 
                                e0=2.77561,
                                l=1, 
                                atom_set=at_sets, 
                                xticks=['G','K','M','K'], 
                                xval_ticks=[0,30,60,90],
                                save_path='./test_1.png')

# Call plot function for l=p. Data grouped by atom type

plotter.plot_fatbands_l(band_list=list(range(150,250)), 
                                e0=2.77561,
                                l=1, 
                                xticks=['G','K','M','K'], 
                                xval_ticks=[0,30,60,90],
                                save_path='./test_2.png')