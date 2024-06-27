import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self    
    
class LogarithmizeWaterObservables(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X , y=None):
        return self
    def transform(self, X):
        df = X.copy()
        log_preds = log_predictors(self.cols)
        df[log_preds] = np.log(df[log_preds]+1)     
        return df    

def keep_predictor(plist,pname,pval):
    if pval>0.5:
        plist.append(pname)
    return plist

def build_predictor_list(predictor_dict):
    plist = []
    for predictor in predictor_dict:
        plist = keep_predictor(plist,predictor,predictor_dict[predictor])
    return plist    

def func(x):
    return np.log(x+1)

def inverse_func(x):
    return np.exp(x)-1

def log_predictors(predictor_list):
    all = ["bio12","bio13","bio14","bio15","bio16","bio17","bio18","bio19"]
    return [f for f in all if f in predictor_list]
   
# Read data from the corresponding outer fold.
data_all_folds = pd.read_csv("agbd_climate_training_dataset_spatialkfolds_1000km.csv")

# This line sets aside the outer spatial CV fold from the inner CV loop used for calibration. The best model for each spatial fold is then evaluated in an outer loop.
data = data_all_folds[data_all_folds["fold_1000km"] != int(id)].reset_index(drop=True)

# Dictionary of predictors.
predictor_dict = {
    "bio1" : bio1,
    "bio2" : bio2,
    "bio3" : bio3,
    "bio4" : bio4,
    "bio5" : bio5,
    "bio6" : bio6,
    "bio7" : bio7,
    "bio8" : bio8,
    "bio9" : bio9,
    "bio10" : bio10,
    "bio11" : bio11,
    "bio12" : bio12,
    "bio13" : bio13,
    "bio14" : bio14,
    "bio15" : bio15,
    "bio16" : bio16,
    "bio17" : bio17,
    "bio18" : bio18,
    "bio19" : bio19,
    "gsl": gsl
}


if seed<0:
    seed = -1*seed

# Extract the target and predictors.
predictor_list = build_predictor_list(predictor_dict)
npredictors = len(predictor_list)

if npredictors > 0:

    n_splits = len(np.unique(data["fold_1000km"]))
    cvs = np.zeros(n_splits)
    
    mdint = int(md)
    
    bst = XGBRegressor(
            n_estimators=3000,
            learning_rate=e,
            max_depth = mdint,
            min_child_weight=mcw,
            subsample=subsample,
            min_split_loss = g,
            max_delta_step = mds,
            random_state = seed,
            eval_metric = "rmse",
            objective='reg:squarederror',
            )
    
    # Log-transform target observable.
    regr = TransformedTargetRegressor(regressor=bst,func=func,inverse_func=inverse_func)

    # Log-transform precipitation related variables and binarize biogeographic realm.
    log_preds = log_predictors(predictor_list)
    if len(log_preds)>0:
        estimator = Pipeline([
            ("col_extract", ColumnExtractor(predictor_list)),
            ("log_water", LogarithmizeWaterObservables(predictor_list)),
            ("regressor",regr)
        ])
    else:
        estimator = Pipeline([
            ("col_extract", ColumnExtractor(predictor_list)),
            ("regressor",regr)
        ])
             
    # Set up inner CV with LeaveOneGroupOut
    logo = LeaveOneGroupOut()

    X=data
    y=data["agbd"]
    g=data["fold_1000km"]

    cvs = cross_val_score(estimator,X,y,groups=g,cv=logo,scoring="neg_mean_absolute_error")    

    # Outputs for multi-objective optimization with NSGA-II with OpenMole.
    error = -1.0*np.mean(cvs)

else:

    # Predicted value is the average biomass density.
    n_folds = len(np.unique(data["fold_1000km"]))
    errors = np.zeros(n_folds)
    
    for k,fold in enumerate(np.unique(data["fold_1000km"])):
        data_fold = data[data["fold_1000km"] == fold ]
        data_reduced = data[data["fold_1000km"] != fold ]
    
        y = np.array(data_fold["agbd"])
        avg = np.mean(data_reduced["agbd"])
        err = np.sqrt(np.mean( np.power((y - avg),2) ))
        
        errors[k] = err
    
    error = np.mean(errors)    
