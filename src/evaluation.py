import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def scores(actual,pred):
    actual=np.asarray(actual,dtype=float);pred=np.asarray(pred,dtype=float)
    return {'MAE_VND':float(mean_absolute_error(actual,pred)),
            'RMSE_VND':float(np.sqrt(mean_squared_error(actual,pred))),
            'R2':float(r2_score(actual,pred)),
            'MAPE_percent':float(100*np.mean(np.abs((actual-pred)/actual)))}
