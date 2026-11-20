import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from .schema import CAT,NUM,EXTRA_NUM

def pipeline(kind,seed=42):
    numerics=NUM+EXTRA_NUM
    cat=Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('ohe',OneHotEncoder(handle_unknown='ignore',min_frequency=30,max_categories=80,sparse_output=False))])
    # Dense one hot can be memory intensive for >100k observations; categorical cap is deliberate.
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numerics),('cat',cat,CAT)],sparse_threshold=0)
    choices={
     'Ridge':Ridge(),
     'ElasticNet':ElasticNet(max_iter=3000,random_state=seed),
     'RandomForest':RandomForestRegressor(n_estimators=100,max_depth=18,min_samples_leaf=3,n_jobs=-1,random_state=seed),
     'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=31,random_state=seed)
    }
    return Pipeline([('pre',pre),('model',choices[kind])])

PARAMS={
 'Ridge':{'model__alpha':[1.0,10.0,100.0]},
 'ElasticNet':{'model__alpha':[0.001,0.01,0.1],'model__l1_ratio':[0.2,0.7]},
 'RandomForest':{'model__max_depth':[12,20],'model__min_samples_leaf':[2,5]},
 'HistGradientBoosting':{'model__learning_rate':[0.05,0.1],'model__max_leaf_nodes':[15,31]},
}
