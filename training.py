"""
Module with functions needed to build an instance of an extreme gradient boosting regressor for predicting AGBD.
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

# Custom transformer for sklearn pipeline to extract specific columns representing predictors
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self    

# Custom transformer for sklearn pipeline to logarithmmize precipitation related predictors (heavy-tailed distributions in the natural space).
class LogarithmizeWaterObservables(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        log_preds = selected_water_predictors(self.cols)
        if len(log_preds)>0:
            df[log_preds] = np.log(df[log_preds]+1)       
        return df    

def keep_predictor(plist,pname,pval: float):
    """
    Returns a list containing the predictors to be used for training.
    :param plist: A list of predictor names.
    :param pname: The name of a potential predictor to include in the list.
    :param pval: If larger than 0.5 the predictor is included.
    :return: The updated list.
    """
    if pval>0.5:
        plist.append(pname)
    return plist

def select_predictors(predictor_dict):
    """
    Creates a list storing the name of the predictors to be used for training.
    :param predictor_dict: A dictionary with poredictor names as keys and their selection value as values. Predictors are kept if their selectino value is larger than 0.5. Selection values are 1 or 0.
    :return: The list storing the names of the predictors to be used for training.
    """

    plist = []
    for predictor in predictor_dict:
        plist = keep_predictor(plist,predictor,predictor_dict[predictor])
    return plist    

def log_function(x):
    """
    Natural logarithm function adapted to 0 values. The logarithm is applied to x+1 instead of x.
    :param x: The value to logarithmize.
    :return: The natural logarithm of x+1. 
    """
    if x.any()<=0:
        print("Cannot logarithmize negative values.")
    return np.log(x+1)

def inverse_log_function(x):
    """
    The inverse of the adapted log function f(x)=log(x+1).
    :param x: The log value.
    :return: The value in the natural space.
    """
    return np.exp(x)-1

def selected_water_predictors(predictor_set):
    """
    Creates a list storing the name of the precipitation-related predictors selected for model training. 
    :param predictor_set: List of selected predictors.
    :return: A list storing the names of the precipitation-related predictors selected for training and tha must be logarithmized.
    """
    water_related = ["bio12","bio13","bio14","bio15","bio16","bio17","bio18","bio19"]
    return [f for f in water_related if f in predictor_set]

def instantiate_xgbr(parameters, hp_names: dict):
    """
    Creates an Extreme Gradient Boosting Regressor (XGBR) instance from the optimal parameters found during model calibration. By default the objective function is the squared error to penalize large prediction errors more than smaller ones.
    :param parameters: Optimal parameters found during model calibration and predictor selection procedure.
    :param hp_names: A dictionary storing as keys the XGBR hyper-parameter names and as values the user-defined names used during calibration to refer to those parameters. For example: the learning rate is usually called e. 
    :return: The parametrized XGBR instance.
    """

    regressor = XGBRegressor(
        n_estimators=int(parameters[hp_names["n_estimators"]]),
        learning_rate=float(parameters[hp_names["learning_rate"]]),
        max_depth = int(parameters[hp_names["max_depth"]]),
        min_child_weight=float(parameters[hp_names["min_child_weight"]]),
        subsample=float(parameters[hp_names["subsample"]]),
        min_split_loss = float(parameters[hp_names["min_split_loss"]]),
        max_delta_step = float(parameters[hp_names["max_delta_step"]]),
        eval_metric = "rmse",
        objective='reg:squarederror',
    )

    return regressor

def get_predictor_set(parameters, predictor_names: list):
    """
    Given the set of optimal parameters and predictors selected during the calibration procedure this method creates a list storing the final set of predictors to be used for training.
    :param parameters: Optimal parameters found during model calibration and predictor selection procedure.
    :param predictor_names: A list storing the names of every potential predictor.
    :return: A list storing the final set of predictors to be used for training. 
    """

    parameters = parameters.reset_index()
    predictor_selection_dict = { name : parameters[name][0] for name in predictor_names }
    predictor_set = select_predictors(predictor_selection_dict)

    return predictor_set

def training_pipeline(parameters, hp_names: dict, predictor_names: list, logarithmize_agbd: bool = True):
    """
    Returns the full training pipeline given the optimal hyperparameters and predictors found during the calibration procedure. 
    :param parameters: Optimal parameters found during model calibration and predictor selection procedure.
    :param hp_names: A dictionary storing as keys the XGBR hyper-parameter names and as values the user-defined names used during calibration to refer to those parameters. For example: the learning rate is usually called e.
    :param predictor_names: A list storing the names of every potential predictor.
    :param logarithmize_agbd: True if target aboveground biomass density values are to be logarithmized for training. Defaults to True.
    :return: A pipeline ready for training.
    """

    xgbr = instantiate_xgbr(parameters,hp_names)

    if logarithmize_agbd:
        xgbr =  TransformedTargetRegressor(regressor=xgbr,func=log_function,inverse_func=inverse_log_function)

    predictor_set = get_predictor_set(parameters, predictor_names)

    estimator = Pipeline([
                ("col_extract", ColumnExtractor(predictor_set)),
                ("log_water", LogarithmizeWaterObservables(predictor_set)),
                ("regressor",xgbr)
            ])    
    
    return estimator


