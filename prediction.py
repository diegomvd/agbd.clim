"""
Module including functions to create windowed CSV prediction data, to elaborate predictions given a model and a prediction dataset and to merge windowed predictions together in a single raster. Also includes functions related to the creation of 0 aboveground biomass density instances.
"""

import pandas as pd
from pathlib import Path
import re
import numpy as np
import geopandas as gpd
import rasterio
from geoutils import raster_from_array, add_features, create_windows
from units import rescale_bioclimatic_predictors
from shapely.geometry import box
from fiona.crs import from_epsg
from rasterio import mask
import json


# Create the paths within the package rather than letting the user define them, this requires creating directories

"""
Functions related to predicting using a trained model.
"""

def create_prediction_dataset(predictor_dict: dict, land_polygons: list, windows_per_side: int, savedir):
    """
    Creates the prediction dataset by sampling values of each predictor in land territories. Each prediction dataset stores also the row and column of each point and the window parameters in the filename.
    :param predictor_dict: Dictionary with predictor names as keys and filenames and datatypes as values.
    :param land_polygons: A list of polygons delineating all land reginos in the world.
    :param windows_per_side: Number of processing windows per dimension. The total number of windos is windows_per_side^2.
    :return: A list storing all paths to the prediction datasets.
    """

    prediction_datasets = []

    def get_predictor_value(arr):
        return  data_predictor[arr[0],arr[1]]

    window_list = create_windows(list(predictor_dict.values())[0][0], windows_per_side)

    # Open first bioclimatic raster as a reference.
    with rasterio.open(list(predictor_dict.values())[0][0]) as src: 
        
        # Read the raster masking as no data all the pixels in waterbodies.    
        data, _ = rasterio.mask.mask(src, land_polygons, crop = False, indexes = 1)

        i = 0
        for window, c, r in window_list:
            i=i+1
            print("window {}".format(i))
            col_off, row_off, w, h = (int(x) for x in window.flatten())

            savepath = savedir + "/predictors_data_col_{}_row_{}_width_{}_height_{}.csv".format(col_off,row_off,w,h)

            if not Path(savepath).exists():

                data_window = data[row_off:row_off+h,col_off:col_off+w]
                # Get the indices in the data tile that fall in land.
                indices_rowcol = np.transpose(np.nonzero(data_window!=src.nodata))

                # Stop the calculation if all the tile falls in a waterbody.
                if len(indices_rowcol)>0:       

                    # Prepare the NumPy array were results will be saved.
                    predictors_array = indices_rowcol.copy()

                    # Iterate over predictors.
                    for predictor in predictor_dict:
                        fname = predictor_dict[predictor][0] 

                        with rasterio.open(fname) as src_predictor:
                            
                            # Read the window and extract the predictor value in each inland cell.
                            data_predictor = src_predictor.read(1,window=window)    
                            data_predictor = np.apply_along_axis(get_predictor_value,1,indices_rowcol)

                            # Progressively build the array adding new columns for each sampled predictor.
                            predictors_array = np.concatenate((predictors_array,np.transpose([data_predictor])), axis=1)

                    columns = ['row','col']+list(predictor_dict.keys())
                    predictors_df = pd.DataFrame({ name : predictors_array[:,i] for i,name in enumerate(columns)})

                    predictors_df = rescale_bioclimatic_predictors(data=predictors_df, predictor_names = list(predictor_dict.keys()))

                    predictors_df.to_csv(savepath,index=False)

                    prediction_datasets.append(savepath)
            else:
                prediction_datasets.append(savepath)        
        
    return prediction_datasets

def predict(model, datasets:list, savedir):
    """
    Predicts potential AGBD for every prediction dataset. Results are saved in CSV files that store thepredicted AGBD value, and the row and column for the corresponding window. Window parameters are saved in the filenames. 
    :param model: The trained prediction model.
    :param datasets: A list storing the paths to the prediction dataset for every window.
    :param savedir: The directory where predictions should be saved.
    :return: A list of paths to the prediction files.  
    """
 
    prediction_results = []

    for file in [Path(dataset) for dataset in datasets]:

        window_params = re.findall("predictors_data(.*)",file.name)[0]

        savepath = savedir+"AGBD_potential{}".format(window_params)

        if not Path(savepath).is_file(): 

            try:
                predictors_data = pd.read_csv(file)
            except:
                continue 

            # TODO: This is being extra safe as prediction datasets shuold not be empty.    
            if len(predictors_data.index)>0:
        
                indices = predictors_data[["row","col"]]

                X = predictors_data.drop(["row","col"],axis="columns")

                X.replace([np.inf, -np.inf], np.nan, inplace=True)

                nan_ids = X[X.isnull().any(axis=1)].index
                
                X = X.dropna()
                indices = indices.drop(nan_ids)

                X = X.reset_index(drop=True)
                indices = indices.reset_index(drop=True)
                
                try:
                    agbd = model.predict(X)
                    agbd_df = pd.DataFrame({"agbd":agbd})
                    predicted = pd.concat([agbd_df,indices], axis = "columns")
                except:
                    print("Could not make prediction.")
                    continue

            try:            
                predicted.to_csv(savepath,index=False)
                prediction_results.append(savepath)
            except: 
                # TODO: this should not happen anymore, could be removed.
                print("Windows without data are not saved.")

        else:
            prediction_results.append(savepath)
    
    return prediction_results

def create_potential_agbd_maps(prediction_results: list, reference_rasterfile: str, savedir: str):
    """
    Creates georeferenced potential AGBD maps from the prediciton CSV files and saves them as rasters in TIFF format. Window information sotred in the filenames is used to map row and column indices to their corresponding coordinates. 
    :param prediction_results: A list storing the paths to the prediction files.
    :param reference_rasterfile: The path to a raster file used as a reference to calculate coordinates of predicted values. It should point to a predictor raster.
    :param savedir: Directory where resulting maps should be saved. 
    :return:  A liststoring the paths to the potential AGBD raster files.
    """

    map_paths = []

    with rasterio.open(reference_rasterfile) as src:

        src_transform = src.transform

        for path in prediction_results:

            window_str = re.findall("AGBD_potential_(.*).csv",Path(path).name)[0].split('_')
            window_params = dict( zip( window_str[::2], window_str[1::2]) )

            width = int(window_params["width"])
            height = int(window_params["height"])
            row_off = int(window_params["row"])
            col_off = int(window_params["col"])
            
            agbd_array = np.full((height,width), -9999, dtype=np.int16)

            data = pd.read_csv(path)

            for _, row_df in data.iterrows():

                row = int(row_df["row"])
                col = int(row_df["col"])
                val = row_df["agbd"]

                agbd_array[row,col] = int(val)

            window_transform = rasterio.windows.transform(rasterio.windows.Window(col_off, row_off, width, height), src_transform) 

            lon, lat = window_transform * (0,0)

            savepath = savedir + "potential_AGBD_lat_{}_lon_{}.tif".format(int(lat),int(lon))

            path = raster_from_array(agbd_array, window_transform, dtype=np.int16, nodata = -9999, filename = savepath) 
            
            map_paths.append(path)
  
    return map_paths        
    

"""
Functions to include 0AGBD points in the training dataset.
"""    

def map_0_agbd(tree_absence_file, predictors_range_file, dtype, nodata, savedir):
    """
    Creates a binary map where values of 1 indicate that a a location is likely to have 0 aboveground biomass density (AGBD) due harsh to climatic conditions. The resulting map requires to process (1) a map depicting tree absence, (2) a map depicting the number of predictors that fall outside of the training range value in each location. Potential 0 AGBD locations must satisfy tree absence and a positive number of predictors with values outside of the training range. 
    :param tree_absence_file: Path to the map depicting tree absence.
    :param predictors_range_file: Path to the map depicting the unmber of predictors with values outside of the training range.
    :param dtype: Data type of the resulting raster.
    :param nodata: No-data value of the resulting raster.
    :param savedir: Directory with maps depicting potential 0 AGBD locations are saved.
    :return: The path to the resulting raster. 
    """

    with rasterio.open(tree_absence_file) as density:
        bounds = rasterio.transform.array_bounds(density.height, density.width, density.transform)
        bbox = box(*bounds)

        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
        geo = geo.to_crs(crs=density.crs.data)

        # Parse features from GeoDataFrame such that rasterio can handle them
        coordinates = [json.loads(geo.to_json())['features'][0]['geometry']]

        with rasterio.open(predictors_range_file) as range:
            out_image, out_transform = mask.mask(range, coordinates, crop=True)
            
            out_image = np.where(out_image>=1, 1, 0)
            out_image = out_image[0,:,:] 
            out_image = out_image + density.read(1)
            out_image = np.where(out_image==2,1,0)

        out_transform = density.transform

        lon = re.findall("lon_(.*)_lat", Path(tree_absence_file).stem)[0]
        lat = re.findall("lat_(.*)", Path(tree_absence_file).stem)[0]

        savepath = savedir + "0agbd_lon_{}_lat_{}.tif".format(lon,lat)
        fname = raster_from_array(out_image,out_transform,dtype,nodata,savepath)
    
    return fname 

def get_0_agbd_indices(id, raster_file):
    """
    Given a binary map get all the indices (rows and columns) where the value equals 1. 
    :param id: Window label corresponding to the raster files. Windows describe different geographical locations but store data with identical indexing. Knowledge of the specific window is required to apply the correct transform to the index data.
    :param raster_file: Path to the raster file sotring the map to process.
    :return: A NumPy array storing rows, columns and window label for each positive location. Window label is identical through the whole array.
    """

    with rasterio.open(raster_file, "r") as src:
        
        data = src.read(1)
        rows, cols = np.where(data==1)
        idarr = np.ones(shape=rows.shape,dtype=int)*int(id)

        # TODO: window label (id) could be returned alone. Return would then be a tuple (array, id).
        rowcol_array = np.array( list( zip(rows, cols, idarr) ), dtype="f,f,i")

    return rowcol_array  

def get_0_agbd_coordinates(agbd_0_indices, agbd_0_files):
    """
    Get coordinates of 0 aboveground biomass density (AGBD) points, given their indices and the window they belong to. 
    :param agbd_0_indices: A NumPy array storing indices of 0 AGBD points and the correspoding window label.
    :agbd_0_files: A dictionary storing as keys the window label and as values the paths to the rasters storing data for their corresponding window.
    :return: A pandas DataFrame storing the coordinates in degrees and EPSG:4326 coordinate reference system (latitude and longitude) of every selected 0 AGBD point.  
    """
   
    agbd_0_coordinates = pd.DataFrame()

    for instance in agbd_0_indices:
        
        row = instance[0]
        col = instance[1]
        id = instance[2]

        with rasterio.open(agbd_0_files[id],'r') as src:

            x, y = src.xy(row=row,col=col) 
            coord = pd.DataFrame.from_dict({"lon":[x],"lat":[y]})
            agbd_0_coordinates = pd.concat([agbd_0_coordinates,coord],axis="rows")

    return agbd_0_coordinates

def add_0_agbd_instances(agbd_df: pd.DataFrame, agbd_0_files, n_instances, predictor_dict):
    """
    Adds 0 aboveground biomass density (AGBD) instances to the training dataset given a series of maps depicting potential locations and a number of desired instances. The method also adds te bioclimatic conditions at each instance and its coordinates. 
    :param agbd_df: A pandas DataFrame storing the training dataset.
    :param agbd_0_files: A list of paths to the rasters storing the potential 0 AGBD locations in each window.
    :param n_instances: Number of 0 AGBD instances to add to the training dataset.
    :param predictor_dict: A dictionary sotring as keys predictor names and as values their corresponding raster files.
    :return: A pandas DataFrame including the initial training dataset plus the 0 AGBD instances.
    """

    agbd_0_indices_all = np.array([],dtype="f,f,i") 
    for i, file in enumerate(list(agbd_0_files)):
        agbd_0_indices_window = get_0_agbd_indices(i, file)
        agbd_0_indices_all = np.append(agbd_0_indices_all, agbd_0_indices_window)
    
    # Among all the possible 0 AGBD select the specified number of instances. 
    rng = np.random.default_rng()
    agbd_0_indices_selected = rng.choice(agbd_0_indices_all, size=n_instances, replace=False)

    agbd_0_coordinates = get_0_agbd_coordinates(agbd_0_indices_selected,agbd_0_files)

    agbd_0_gdf = gpd.GeoDataFrame(
        agbd_0_coordinates,
        geometry=gpd.points_from_xy(agbd_0_coordinates.lon, agbd_0_coordinates.lat)
    )
    agbd_0_gdf.crs = "EPSG:4326"

    agbd_0_gdf = add_features(agbd_0_gdf,predictor_dict)

    agbd_0_gdf = rescale_bioclimatic_predictors(agbd_0_gdf,predictor_dict.keys())

    agbd_0_gdf["agbd"] = 0.0

    agbd_0_df = pd.DataFrame(agbd_0_gdf.drop(columns=['geometry']))    

    agbd_0_df["cluster"] = agbd_df["cluster"].min()-1

    agbd_data_new = pd.concat([agbd_df,agbd_0_df],axis="rows").reset_index(drop=True)  

    return agbd_data_new



