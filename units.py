"""
Module featuring methods for unit conversion.
"""

import pandas as pd

def kg_per_km2_to_mg_per_ha(x):
    """
    Transform kg/km^2 to Mg/ha.
    :param x: Mass density in kg/km^2.
    :return: The corresponding mass density in Mg/ha.
    """
    return x * 0.00001 

def mg_per_ha_to_kg_per_km2(x):
    """
    Transform Mg/ha to kg/km^2. 
    :param x: Mass density in Mg/ha.
    :return: The corresponding mass density in kg/km^2.
    """
    return x * 10**5

def chelsa_bioclim_transformation(name, x):
    """
    Transforms a bioclimatic variable from CHELSA scaling to its real value.
    :param name: Name of the CHELSA bioclimatic variable. 
    :param x: Value of the bioclimatic variable scaled and offseted.
    :return: Real value of the bioclimatic variable in the reported units. 
    """

    transformation_dict = {
        'bio1' : {"scale":0.1, "offset":-273.15},
        'bio2' : {"scale":0.1, "offset":0},
        'bio3' : {"scale":0.1, "offset":0},
        'bio4' : {"scale":0.1, "offset":0},
        'bio5' : {"scale":0.1, "offset":-273.15},
        'bio6' : {"scale":0.1, "offset":-273.15},
        'bio7' : {"scale":0.1, "offset":0},
        'bio8' : {"scale":0.1, "offset":-273.15},
        'bio9' : {"scale":0.1, "offset":-273.15},
        'bio10' : {"scale":0.1, "offset":-273.15},
        'bio11' : {"scale":0.1, "offset":-273.15},
        'bio12' : {"scale":0.1, "offset":0},
        'bio13' : {"scale":0.1, "offset":0},
        'bio14' : {"scale":0.1, "offset":0},
        'bio15' : {"scale":0.1, "offset":0},
        'bio16' : {"scale":0.1, "offset":0},
        'bio17' : {"scale":0.1, "offset":0},
        'bio18' : {"scale":0.1, "offset":0},
        'bio19' : {"scale":0.1, "offset":0},
        'gsl' : {"scale":1.0, "offset":0},
    }

    try:
        params = transformation_dict[name]
        scale = params["scale"]
        offset = params["offset"]
        ret = x*scale + offset
    except:
        ret = x 

    return ret  

def inverse_chelsa_bioclim_transformation(name, x):
    """
    Transforms a bioclimatic variable in its real value to the CHELSA scaling.
    :param name: Name of the CHELSA bioclimatic variable. 
    :param x: Value of the bioclimatic variable using CHELSA scaling.
    :return: Real value of the bioclimatic variable in the reported units. 
    """

    transformation_dict = {
        'bio1' : {"scale":0.1, "offset":-273.15},
        'bio2' : {"scale":0.1, "offset":0},
        'bio3' : {"scale":0.1, "offset":0},
        'bio4' : {"scale":0.1, "offset":0},
        'bio5' : {"scale":0.1, "offset":-273.15},
        'bio6' : {"scale":0.1, "offset":-273.15},
        'bio7' : {"scale":0.1, "offset":0},
        'bio8' : {"scale":0.1, "offset":-273.15},
        'bio9' : {"scale":0.1, "offset":-273.15},
        'bio10' : {"scale":0.1, "offset":-273.15},
        'bio11' : {"scale":0.1, "offset":-273.15},
        'bio12' : {"scale":0.1, "offset":0},
        'bio13' : {"scale":0.1, "offset":0},
        'bio14' : {"scale":0.1, "offset":0},
        'bio15' : {"scale":0.1, "offset":0},
        'bio16' : {"scale":0.1, "offset":0},
        'bio17' : {"scale":0.1, "offset":0},
        'bio18' : {"scale":0.1, "offset":0},
        'bio19' : {"scale":0.1, "offset":0},
        'gsl' : {"scale":1.0, "offset":0},
    }

    try:
        params = transformation_dict[name]
        scale = params["scale"]
        offset = params["offset"]
        ret = (x - offset)/scale
    except:
        ret = x 

    return ret 

def rescale_bioclimatic_predictors(data: pd.DataFrame, predictor_names):
    """
    Rescale the values of CHELSA bioclimatic predictors in a dataset to their real value in the reported units. 
    :param data: A pandas DataFrame storing the dataset. 
    :param predictor_names: A list storing the names of the bioclimatic predictors to re-scale. names must match columns in the dataset.
    :return: A copy of the initial DataFrame with the real values for the bioclimatic predictors.  
    """

    data_new = data.copy()
    for predictor in predictor_names:
        data_new[predictor] = data_new[predictor].apply(lambda val: chelsa_bioclim_transformation(predictor,val))

    return data_new    

def inverse_rescale_bioclimatic_predictors(data: pd.DataFrame, predictor_names):
    """
    Rescale true values of bioclimatic predictors in a dataset to their CHELSA-scaling value. 
    :param data: A pandas DataFrame storing the dataset. 
    :param predictor_names: A list storing the names of the bioclimatic predictors to re-scale. names must match columns in the dataset.
    :return: A copy of the initial DataFrame with the real values for the bioclimatic predictors.  
    """

    data_new = data.copy()
    for predictor in predictor_names:
        data_new[predictor] = data_new[predictor].apply(lambda val: inverse_chelsa_bioclim_transformation(predictor,val))

    return data_new  
