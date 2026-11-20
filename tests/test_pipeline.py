import numpy as np,pandas as pd
from src.schema import prepare,FEATURES
from src.model import pipeline
from src.evaluation import scores

def test_pipeline_smoke():
 n=80
 df=pd.DataFrame({'province_name':['Hồ Chí Minh']*n,'price':np.linspace(1e9,5e9,n),'area':np.linspace(40,140,n),'property_type_name':['Nhà']*n,'district_name':['Tân Bình']*n})
 p=prepare(df);assert len(p)==n
 assert np.isfinite(p['area']).all()
 m=pipeline('Ridge');m.fit(p[FEATURES],np.log(p['price']))
 pred=np.exp(m.predict(p[FEATURES]));assert np.all(pred>0)
 assert scores(p['price'],pred)['MAE_VND']>=0
