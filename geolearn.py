"""
Module featuring a series of functions to facilitate machine learning with geospatial data
"""

from sklearn import cluster
import pandas as pd
from geoutils import get_radian_coordinates
from sklearn.metrics.pairwise import haversine_distances
import rasterio
import numpy as np
from geoutils import raster_from_array, create_windows, sum_rasters
from units import inverse_rescale_bioclimatic_predictors
from pathlib import Path
import os
import re

"""
Functions related to spatial clustering
"""

def geospatial_clustering(coordinates, min_cluster_size: int, distance_threshold: float):
    """
    Cluster geolocated instances based on proximity in a spherical surface using haversine metric and a hierarchical density-based clustering algorithm (Scikit Learn HDBSCAN implementation).
    :param coordinates: An array of coordinates in degrees and EPSG:4326 coordinate system (latitude and longitude) where each row corresponds to an instance.
    :param min_cluster_size: Clusters smaller than this size wil be discarded as outliers.
    :param distance_threshold: Clusters at a distance closer than this threshold will be merged together. 
    :return: The cluster labels for each geolocated instance.  
    """
    
    kms_per_radian = 6371.0088
    cs_eps = distance_threshold / kms_per_radian
    
    # Define the clustering algorithm
    hdb = cluster.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon = cs_eps, metric='haversine')

    # Perform clustering
    labels = hdb.fit_predict(coordinates)

    return labels

def get_clusters_centroids(clusters, id_col: str):
    """
    Calculate the centroid of all points belonging to a cluster. Uses EPSG:4326 coordinate reference system.
    :param clusters: A GeoDataFrame describing point geometries where instances are associated to a cluster label. 
    :param id_col: Name of the column storing cluster labels.
    :return: A GeoDataFrame storing cluster labels together with their centroid. 
    """

    clusters.crs = "EPSG:4326"
    clusters_centroids = clusters.dissolve(by=id_col,aggfunc="first").to_crs('+proj=cea')
    clusters_centroids['geometry'] = clusters_centroids.geometry.centroid
    
    # This removes latitude and longitude records for the first instance of every cluster.
    clusters_centroids = clusters_centroids.reset_index()[[id_col,'geometry']]
    clusters_centroids = clusters_centroids.to_crs(clusters.crs)

    return clusters_centroids

"""
Functions related to spatial cross-validation
"""

def add_spatial_cv_folds(data: pd.DataFrame, autocorrelation_distance: float, lat_col: str, lon_col: str):
    """
    Creates spatial cross-validation folds by clustering instances in a dataset based on geographical distance. The method uses a hierarchical clustering algorithm with a distance threshold corresponding to the autocorrelation distance characteristic of the dataset. 
    :param data: A Pandas DataFrame storing the dataset to split in spatial folds.
    :param autocorrelation_distance: The autocorrelation distance of the dataset instances in kilometers.
    :param lat_col: Name of the column storing latitude data.
    :param lon_col: Name of the column storing longitude data. 
    :return: A copy of the dataset including a column with spatial cross-validation labels named: "fold_{autocorrelation_distance}km".  
    """
    new_data = data.copy()

    coordinates = pd.DataFrame(data[[lat_col,lon_col]])
    radian_coordinates = get_radian_coordinates(coordinates,lat_col,lon_col)

    kms_per_radian = 6371.0088

    distance_matrix = haversine_distances(radian_coordinates)*kms_per_radian

    clustering = cluster.AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=autocorrelation_distance,
        metric="precomputed",
        linkage="average"
    )

    labels = clustering.fit_predict(distance_matrix)

    label_df = pd.DataFrame(columns=["fold_{}km".format(autocorrelation_distance)],index=new_data.index)
    label_df["fold_{}km".format(autocorrelation_distance)] = labels

    new_data = new_data.join(label_df)

    return new_data


""" 
Functions related to assessing predictors' coverage in training vs. prediction datasets.
"""

def get_predictors_range(predictor_data: pd.DataFrame) :
    """
    Calculate the range (min and max values) of predictor variables.
    :param predictor_data: A pandas DataFrame storing the predictor values for every instance. 
    :return: A pandas DataFrame storing minimum and maximum values for each predictor.
    """
    range = predictor_data.describe().loc[['min','max']]  
    range = inverse_rescale_bioclimatic_predictors(range,range.columns)

    return range

def get_out_of_range_geodata(predictor: str, window, col_off, row_off, datafile, range):
    """
    From a window of a global predictor map create a new binary map where 1 indicates the predictor is out of the training range and a value of 0 indicates it is within.
    :param predictor: Predictor name.
    :param window: Raster window to be processed.
    :param col_off: Window's column offset.
    :param row_off: Window's row offset. 
    :param datafile: The raster file for the predictor's global map.
    :param range: A predictor range pandas DataFrame storing minimum and maximum values of each predictor in the training dataset. 
    :return: (1) a NumPy array storing the resulting binary map, (2) the processing window's transform object, (3) longitude of the upper-left window corner, (4) latitude of the upper-left window corner.  
    """

    min = range[predictor].loc["min"]
    max = range[predictor].loc["max"]
    
    with rasterio.open(datafile) as src:

        raster = src.read(1, window = window)
        transform_src = src.transform
        (lon, lat) = transform_src * (col_off,row_off)
        transform_win = rasterio.windows.transform(window, transform_src)
        
        raster = np.where(raster!=src.nodata ,raster,np.nan)
        out_of_range = ((raster > max) | ( raster < min ))
        raster = np.where(out_of_range,1,0)

    return raster, transform_win, lon, lat

def map_predictors_out_of_range(training_data: pd.DataFrame, window_side, predictor_dict : dict, savedir, dtype, nodata):
    """
    Process global predictor maps to create a series of binary maps for each predictor where 1 indicates the the predictor is out of the training dataset's range, and a value of 0 that it is within the training range. Multiple maps are created for each predictor according to the number of windows used for processing. Processing by windows prevents memory overflows. 
    :param training_data: A pandas DataFrame sotring the training dataset.
    :param window_side: Number of windows in each dimension. The total number of windows in window_side^2.
    :param predictor_dict: A dictionary sotring as keys predictor names and as values their corresponding raster files.
    :param savedir: Directory where resulting maps should be saved.
    :param dtype: Data type of resulting rasters.
    :para nodata: No-data value in resulting rasters.
    :return: A dictionary storing as keys the predictor names and as values the list of paths to the resulting raster files.
    """

    predictors = predictor_dict.keys()

    range = get_predictors_range( training_data[list(predictors)] )

    predictor_paths = {}

    for predictor in predictors:

        predictor_fname = predictor_dict[predictor][0] 

        window_list = create_windows(predictor_fname, n_divisions = window_side)

        savedir_windows = savedir + "{}/".format(predictor)
        
        if not os.path.exists(savedir_windows):
            os.mkdir(savedir_windows)

        fnames = []
        for window, col_off, row_off in window_list:

            raster, transform, lon, lat = get_out_of_range_geodata(predictor, window, col_off, row_off, predictor_fname,range)

            out_path = savedir_windows + "predictor_{}_out_of_range_lon_{}_lat_{}.tif".format(predictor,int(lon),int(lat))
        
            fname_temp = raster_from_array(raster,transform,dtype,nodata,out_path)
            fnames.append(fname_temp)

        predictor_paths[predictor] = fnames
           
    return predictor_paths

def map_number_of_predictors_out_of_range(training_data, window_side, predictor_dict : dict, dtype, nodata, savedir):
    """
    Process global predictor maps to create a series of maps storing the number of predictors with values outside of the training range in each pixel. Multiplemaps are created according to the number of windows used for data processing. Processing by windows prevents memory overflows. 
    :param training_data: A pandas DataFrame sotring the training dataset.
    :param window_side: Number of windows in each dimension. The total number of windows in window_side^2.
    :param predictor_dict: A dictionary sotring as keys predictor names and as values their corresponding raster files.
    :param dtype: Data type of resulting rasters.
    :para nodata: No-data value in resulting rasters.
    :param savedir: Directory where resulting maps should be saved.
    :return: A list of paths to the resulting raster files (1 per processing window).
    """

    # First create the out-of-range maps fro every predictor individually.
    out_of_range_maps = map_predictors_out_of_range(training_data, window_side, predictor_dict, savedir, dtype, nodata)

    window_list = create_windows(list(out_of_range_maps.values())[0][0], n_divisions = 1)

    savedir_windows = savedir + "all_predictors/"
    if not os.path.exists(savedir_windows):
        os.mkdir(savedir_windows)
    
    ref_map_list = [Path(mapfile).stem for mapfile in list(out_of_range_maps.values())[0]]

    window_coordinates_dict = {  re.findall("range_(.*)" ,fname)[0]: [] for fname in ref_map_list }

    for predictor in out_of_range_maps: 
        for fname in out_of_range_maps[predictor]:
            coords = re.findall("range_(.*)" ,Path(fname).stem)[0]
            flist = window_coordinates_dict[coords]
            flist.append(fname)
            window_coordinates_dict[coords] = flist

    for c in window_coordinates_dict:
        fmaps = window_coordinates_dict[c]

        prefix = "number_predictors_out_of_range"
        result_files = sum_rasters(fmaps, window_list, savedir_windows, prefix, dtype, nodata)

    # TODO: check if this really builds a list.
    raster_paths = Path(savedir_windows).glob('*.tif')

    return raster_paths