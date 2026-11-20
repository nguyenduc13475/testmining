import numpy as np
import pandas as pd

TARGET='price'
NUM=['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count']
CAT=['property_type_name','district_name','ward_name','street_name','project_name','house_direction','balcony_direction']
FIELDS=CAT+NUM

def prepare(df: pd.DataFrame, training: bool=True) -> pd.DataFrame:
    df=df.copy()
    for c in FIELDS+[TARGET,'province_name','published_at']:
        if c not in df: df[c]=np.nan
    if training:
        df=df[df['province_name'].astype('string').str.contains('Hồ Chí Minh|Ho Chi Minh',case=False,na=False,regex=True)].copy()
    for c in NUM+[TARGET]: df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in CAT:
        df[c]=df[c].astype('string').fillna('Không rõ').str.strip().replace('', 'Không rõ').astype(str)
    for c in NUM:
        df.loc[(df[c]<0)|(df[c]>1e7),c]=np.nan
    df.loc[~df['area'].between(5,5000),'area']=np.nan
    for c in ['floor_count','bedroom_count','bathroom_count']:
        df.loc[~df[c].between(0,100),c]=np.nan
    df.loc[~df['frontage_width'].between(0.1,100),'frontage_width']=np.nan
    df.loc[~df['house_depth'].between(0.1,1000),'house_depth']=np.nan
    df.loc[~df['road_width'].between(0.1,100),'road_width']=np.nan
    df['published_at']=pd.to_datetime(df['published_at'],errors='coerce',utc=True)
    df['published_month']=df['published_at'].dt.month.astype(float)
    df['published_year']=df['published_at'].dt.year.astype(float)
    df['aspect_ratio']=df['house_depth']/df['frontage_width'].replace(0,np.nan)
    df['aspect_ratio']=df['aspect_ratio'].clip(0,100)
    return df

EXTRA_NUM=['published_month','published_year','aspect_ratio']
FEATURES=NUM+EXTRA_NUM+CAT
