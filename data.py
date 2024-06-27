"""
Module to download the data required to reproduce the analysis.
"""

from urllib.request import urlretrieve
import re
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

def fetch_tree_size_data():
    """
    Downloads tree data from the Tallo database to a predefined location within the project. 
    :return: The path to the database.
    """

    filename = "./data/tree_size_data/tallo_test.csv"

    if not Path(filename).parents[0].exists():
        Path(filename).parents[0].mkdir()

    path = filename

    if not Path(filename).exists():    
        tallo_url = "https://zenodo.org/records/6637599/files/Tallo.csv?download=1"
        path, message = urlretrieve(tallo_url,filename)

    return path

def fetch_bioclimatic_data():
    """
    Downloads bioclimatic data from CHELSA Bioclim to a predefined location within the project. 
    :return: The directory where bioclimatic data is stored, relative to the source of the project.
    """

    urls_file = "./data/bioclimatic_data/envidatS3paths.txt"
    
    with open(urls_file) as urls:
        for url in urls:
            name = re.findall('bio/(.*)',url[:-2])[0]
            path = "./data/bioclimatic_data/{}".format(name)
            if not Path(path).exists():
                response = urlretrieve(url,path)

    return Path(urls_file).parents[0]

def fetch_tree_density_data():
    """
    Downloads the tree density raster file from Crowther et al., 2015 to a predefined location within the project. 
    :return: The path to the raster file.
    """

    tree_density_url = "https://elischolar.library.yale.edu/cgi/viewcontent.cgi?filename=1&article=1000&context=yale_fes_data&type=additional"

    filename = "./data/tree_density/crowther_tree_density_biome_revised_model.zip"
    if not Path(filename).parents[0].exists():
        Path(filename).parents[0].mkdir()
 
    if not Path(filename).exists():
        path, message = urlretrieve(tree_density_url,filename)
    
    raster_path = "./data/tree_density/Crowther_Nature_Files_Revision_01_WGS84_GeoTiff/Crowther_Nature_Biome_Revision_01_WGS84_GeoTiff.tif"

    if not Path(raster_path).exists():
        with ZipFile(filename) as myzip:
                raster_path = myzip.extract(myzip.namelist()[1], path=Path(filename).parents[0])
                raster_path = './' + raster_path
    
    return raster_path


def fetch_geo_trees_data():
    """
    Downloads data from GEO-TREES experimental plots with in-situ forest biomass measurements.
    :return: The path to the excel file.
    """

    filename = "./data/validation/geo_trees_data/FOS_Plots_v2019.04.10.xlsx"

    if not Path(filename).parents[0].exists():
        Path(filename).parents[0].mkdir(parents=True)

    path = filename

    if not Path(filename).exists():    
        geotrees_url = "https://data.geo-trees.org/Data/FOS_Plots_v2019.04.10.xlsx"        
        path, message = urlretrieve(geotrees_url,filename)

    pd.DataFrame(pd.read_excel(path)).to_csv("./data/validation/geo_trees_data/FOS_Plots_v2019.04.10.csv", index= False)    

    return path

def fetch_walker_agcd_map():
    """
    Downloads the potential AGBD map elaborated in Walker et al. (2022) using remote-sensing data and machine learning methods.
    :return: The path to the raster file. 
    """

    filename = "./data/validation/walker_2022_data/Base_Pot_AGB_MgCha_500m.tif"

    if not Path(filename).parents[0].exists():
        Path(filename).parents[0].mkdir(parents=True)

    path = filename

    try:
        if not Path(filename).exists():    
            walker_agbd_map_url = "https://dataverse.harvard.edu/api/access/datafile/6298964"   
            path, message = urlretrieve(walker_agbd_map_url,filename)
    except: 
        print("Function fetch_walker_agcd_map is not working as expected. Please click on the following link: \n https://dataverse.harvard.edu/api/access/datafile/6298964 to download the data \n and save the raster file within ./data/walker_2022_data/ \n Sorry for the inconvenience.")        


    return path

def fetch_ecoregion_data():
    """
    Downloads Ecoregion and Biome data with Olson et al. (2017) classification from WWF.(??)
    :return: The path to the shapefile.
    """

    filename = "./data/ecoregions/Ecoregions2017.zip"

    if not Path(filename).parents[0].exists():
        Path(filename).parents[0].mkdir(parents=True)

    path = filename

    if not Path(filename).exists():    
        ecoregions_url = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"        
        path, message = urlretrieve(ecoregions_url,filename)

    path = "./data/ecoregions/Ecoregions2017.shp"

    if not Path(path).exists():
        with ZipFile(filename) as myzip:
                myzip.extractall(Path(filename).parents[0])    

    return path