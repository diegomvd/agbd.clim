from pathlib import Path
from natsort import natsorted
import rioxarray as riox
from rioxarray.merge import merge_arrays


raster_paths = [str(file) for file in Path("./data/prediction/potential_AGBD_maps/").glob('*')]
raster_paths = [str(file) for file in Path("./data/training/processing/0agbd_instances/all_predictors/").glob('*')]


merged_path = './data/prediction/potential_AGBD_Mgha_1km2_contemporary_climate.tif' 
merged_path = './data/training/processing/0agbd_instances/number_predictors_out_of_range.tif'

if not Path(merged_path).parents[0].exists():
    Path(merged_path).parents[0].mkdir()

# Sort paths to ensure memory efficient merge by merging nearby windows first
sorted_paths = natsorted(raster_paths, key=str)

raster_list = [ riox.open_rasterio(file) for file in sorted_paths ] 

aux_list=[]

# Merge rasters by pairs until there's only one.
while len(raster_list)>1:
    # Discard left-out raster to make sure there is an even number to merge. Normally, this should be entered in the first iteration if ever.
    if not (len(raster_list)%2 == 0):
        aux_list.append(raster_list[-1])
        raster_list = raster_list[0:-1]

    raster_list = [ merge_arrays([tup[0],tup[1]]) for tup in zip(raster_list[0::2], raster_list[1::2]) ]
    print("Remaining rasters {}".format(len(raster_list)))

# Merge left-out raster with the result of the progressive merges.
if len(aux_list)>0:
    print("merging aux")
    raster_aux = merge_arrays(aux_list)   

    print("merging final")
    raster_final = merge_arrays([raster_aux,raster_list[0]])

    raster_final.rio.to_raster(merged_path)
else :
    if len(raster_list)>1:
        print("Error")
    else:    
        raster_list[0].rio.to_raster(merged_path)  


