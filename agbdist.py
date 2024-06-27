"""
This module contains functions to estimate Aboveground Biomass Density in clusters of trees given the trees Aboveground Biomass and the tree density at their location. s
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
import scipy.stats as stats
from geoutils import get_radian_coordinates, raster_from_array, create_windows
from geolearn import geospatial_clustering, get_clusters_centroids
import geopandas as gpd
import rasterio
from pathlib import Path
import os


def cluster_trees(tree_data: pd.DataFrame, min_cluster_size:int, distance_threshold: float, lat_col: str, lon_col: str):
    """
    This function clusters tree observations together based on geolocation using a hierarchical density-based clustering algorithm (HDBSCAN). 
    :param tree_data: pandas DataFrame containing tree data, must include their coordinates in EPSG:4326 WGS84. 
    :param min_cluster_size: minimum number of tree instances to form a cluster. Clusters with less than this number of instances are discarded as outliers.
    :param distance_threshold: tree clusters at a distance smaller than this threshold are merged in the same cluster. Setting a distance threshold avoids the obtention of many small clusters in regions with high density of tree instnaces, in favor of fewer, larger clusters.
    :param lat_col: the column name storing trees' latitude.
    :param lon_col: the column name storing trees' longitude.
    :return: a pandas DataFrame storing the initial tree data plus the identifier of their cluster and their cluster location based on the centroid of every tree within that cluster. 
    """
    
    coordinates = pd.DataFrame(tree_data[[lat_col,lon_col]])
    
    radian_coordinates = get_radian_coordinates(coordinates,lat_col,lon_col)

    clusters = geospatial_clustering(radian_coordinates,min_cluster_size,distance_threshold)

    # Create label DataFrame and join with tree dataset. 
    cluster_df = pd.DataFrame(columns=["cluster"],index=tree_data.index)
    cluster_df["cluster"] = clusters
    tree_data_upd = tree_data.copy()
    tree_data_upd = tree_data_upd.join(cluster_df)

    cluster_gdf = gpd.GeoDataFrame(
        tree_data_upd[["cluster",lat_col,lon_col]], 
        geometry = gpd.points_from_xy(tree_data_upd[lon_col], tree_data_upd[lat_col])
    )

    clusters_centroids = get_clusters_centroids(cluster_gdf,"cluster")
    clusters_centroids[lon_col+"_cluster"] = clusters_centroids.geometry.x
    clusters_centroids[lat_col+"_cluster"] = clusters_centroids.geometry.y
    tree_data_upd = pd.merge(tree_data_upd,clusters_centroids,on='cluster')

    return tree_data_upd

def calibrate_cluster_agb_distribution(agb_samples, niter: int, bwmin: float, bwmax: float, kernel, in_splits: int, out_splits: int):
    """
    Calibrates the bandwidth of the kernel density estimation (KDE) of trees' aboveground biomass (AGB) distribution in a cluster. Calibration is carried over wit a nested cross-validation scheme using a randomized search to explore the bandwidth space. AGB distributions are heavy-tailed, thus the AGB values are log-transformed to ensure that the KDE algorithm estimates close-to-normal distributions.

    :param agb_samples: Array storing the aboveground biomass of each tree in the cluster.
    :param int niter: Number of repetitions in the inner cross-validation loop. 
    :param float bwimn: Lower boundary for the bandwidth parameter.
    :param float bwmax: Higher boundary for the bandwidth parameter.
    :param kernel: Kernel to be used in the (KDE) algorithm.
    :param int in_splits: Number of inner folds in the nested cross-validation scheme.
    :param int out_splits: Number of outer folds in the nested cross-validation scheme. 
    :return: A triple storing (1) the tuned calibrated KDE estimator, (2) the cross-validation score, (3) the optimal bandwidth.  
    """

    bw_dist = {"bandwidth":  stats.uniform(bwmin,bwmax) }
    kde = KernelDensity(kernel=kernel)

    inner_cv = KFold(n_splits=in_splits, shuffle=True)
    outer_cv = KFold(n_splits=out_splits, shuffle=True)

    random_search = RandomizedSearchCV(
        estimator=kde, param_distributions=bw_dist, n_iter=niter, cv=inner_cv
    )

    agb_samples_log = np.log(agb_samples)

    best_params = random_search.fit(agb_samples_log).best_params_

    best_bw = best_params["bandwidth"]

    score = cross_val_score(
        random_search, X=agb_samples_log, cv = outer_cv
    ).mean()

    kde_tuned = KernelDensity(kernel=kernel, bandwidth = best_bw).fit(agb_samples_log)

    return (kde_tuned, score, best_bw)

def sample_cluster_agb_distribution(agb_distribution, tree_density):
    """
    Estimate total aboveground biomass (AGB) in a cluster with given AGB distribution (in logarithmic space) and tree density. The distribution is sampled a number of times equal to the tree density value and each sample is summed. The sum corresponds to the total AGB in one unit area (depends on tree density units). 
    :param agb_distribution: An empirically estimated distribution of log-transformed AGB values.
    :param tree_density: Average number of trees per unit area in a cluster of trees.
    :return: Estimated total AGB in one unit area (actual unit depends on tree density data) of a forest represented by a cluster of trees. 
    """

    agb = np.nan
    if np.isfinite(tree_density):

        int_tree_density = int(tree_density)

        if int_tree_density<1:
            agb = 0
        else: 
            agb = np.sum(np.exp(agb_distribution.sample(int_tree_density)))    

    return agb 

def estimate_cluster_biomass_density(agb_samples, tree_density, kdeparams: dict, n_replicas: int):       
    """
    Estimates aboveground biomass density (AGBD) in a cluster given instances of trees' aboveground biomass (AGB) and average tree density. This involves the empirical estimation of the AGB distribution in the cluster using a kernel density estimation (KDE) algorithm. The AGB distribution is sampled according to the average tree density in the cluster and sampled values are summed to obtain an estimation of AGBD at the cluster's location. The sampling process is replicated a number of times to ensure the AGBD estimation is robust.
    :param agb_samples: Array storing the aboveground biomass of each tree in the cluster.
    :param tree_density: Average number of trees per unit area in a cluster of trees.
    :param kdeparams: A dictionary storing the parameters to be used for calibration of the KDE algorithm.
    :param n_replicas: Number of replications of the AGB distribution sampling and summing process.
    :return: A tuple storing (1) the median AGBD across replicas, (2) the mean AGBD across replicas, (3) the standard deviation of accross replicas AGBD, (4) the cross-validation score of the calibrated KDE, (5) the calibrated bandwidth of the KDE algorithm.  
    """
    agb_distribution, score, bw = calibrate_cluster_agb_distribution(agb_samples,kdeparams["niter"],kdeparams["bwmin"],kdeparams["bwmax"],kdeparams["kernel"],kdeparams["in_splits"],kdeparams["out_splits"])
    
    agbd_replicas = []
    for i in range(n_replicas):
        agbd_replicas.append(sample_cluster_agb_distribution(agb_distribution,tree_density))

    median_agbd = np.median(np.array(agbd_replicas)) 
    mean_agbd = np.mean(np.array(agbd_replicas)) 
    std_agbd = np.std(np.array(agbd_replicas))   

    return (median_agbd, mean_agbd, std_agbd, score, bw)  

def estimate_agbd(data: pd.DataFrame, agb_col: str, dens_col: str, cluster_col: str, kdeparams: dict, n_replicas: int, save_tmp: bool, save_path: str):
    """
    Estimate aboveground biomass density (AGBD) in every cluster given a dataset of trees' aboveground biomass (AGB) and tree densities. AGBD calculation in each cluster is based on the empirical estimation of the cluster's AGB distribution using a kernel density estimation algorithm. THe AGB distribution is sampled a number of times according to tree density to estimate AGBD in each cluster.
    :param data: Pandas DataFrame storing tree AGB instances associated with a cluster Id and tree density at the location of the sampled tree. 
    :param agb_col: Name of the column storing the AGB data.
    :param dens_col: Name of the column storing the tree density data.
    :param cluster_col: Name of the column storing the cluster Id.
    :param kdeparams: A dictionary storing the parameters to be used for calibration of the KDE algorithm.
    :param n_replicas: Number of replications of the AGB distribution sampling and summing process.
    :param save_tmp: If False AGBD estimations are saved after every cluster is calculated. If True results are progressively saved. 
    :param save_path: Path to save the results of the AGBD estimation procedure for every cluster fo trees.
    :return: Pandas DataFrame storing AGBD estimation data (median, mean, standard deviation) and tree density data for every cluster. The cross-validation score and optimal bandwidth selected during the calibration of the KDE algorithm are also stored in the DataFrame.   
    """

    nsplits = np.max(np.array([kdeparams["out_splits"],kdeparams["in_splits"]]))

    data_filt = data[[cluster_col,agb_col,dens_col]]

    groups = data_filt[cluster_col].unique()

    abd = pd.DataFrame()

    for g in groups:
        
        datag = data_filt[data_filt[cluster_col] == g]

        td_mean = np.mean(datag[dens_col])
        td_std = np.std(datag[dens_col])

        agb_array = np.array(datag[agb_col]).reshape(-1,1)
        
        if (not np.isnan(td_mean)):

            if agb_array.shape[0] >= nsplits:

                med_agbd, mean_agbd, std_agbd, score, bw = estimate_cluster_biomass_density(agb_array,td_mean,kdeparams,n_replicas)

                row = {
                    "cluster":g,
                    "mean_td": td_mean,
                    "std_td": td_std,
                    "med_agbd": med_agbd,
                    "mean_agbd": mean_agbd,
                    "std_agbd": std_agbd,
                    "score": score,
                    "bw" : bw
                    }
            else:
                print("Strange behavior:")
                row = {
                    "cluster":g,
                    "mean_td": td_mean,
                    "std_td": td_std,
                    "med_agbd": np.nan,
                    "mean_agbd": np.nan,
                    "std_agbd": np.nan,
                    "score": np.nan,
                    "bw" : np.nan
                    }

            
            abdrow = pd.DataFrame([row])
            abd = pd.concat([abd,abdrow], axis=0, ignore_index=True)

            if save_tmp:
                abd.to_csv(save_path,index = False)

        else: 
            print("Cluster {} has undefined average tree density (nan)".format(g))        

    return abd

def map_tree_absence(datafile : str, window_side, nodata, savedir : str):
    """
    Creates a tree absence map, where 1 means absence and 0 means presence, based on a map of tree density. 
    :param datafile: The path to the base tree density map.
    :param window_side: Lenght of the window side used for processing the tree density map by chunks.
    :param nodata: nodata value for the resulting tree absence map. 
    :param savedir: Directory where tree absence maps should be saved. 
    :return: A list of paths to the tree absence maps for every processing window.
    """

    window_list = create_windows(datafile, window_side)

    with rasterio.open(datafile) as src:

        savedir_windows = savedir + "tree_absence/"
        if not os.path.exists(savedir_windows):
            os.mkdir(savedir_windows)
        
        for window, col_off, row_off in window_list:

            raster = src.read(1, window = window)
            raster = np.where(raster==0,1,0)
            transform_src = src.transform
            (lon, lat) = transform_src * (col_off,row_off)
            transform_win = rasterio.windows.transform(window,transform_src)

            savepath = savedir_windows + "tree_absence_lon_{}_lat_{}.tif".format(int(lon),int(lat))

            fname_tmp = raster_from_array(raster,transform_win,raster.dtype,nodata,savepath)

        raster_paths = Path(savedir_windows).glob('*.tif') 
    
    return raster_paths
