"""Train only on listing features that are not the label or label-derived.
NOTE: Output artifacts are created only after the dataset has actually been processed.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold
from sklearn.inspection import permutation_importance
from src.schema import prepare,FEATURES
from src.model import pipeline,PARAMS
from src.evaluation import scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/raw/hcm_listings.parquet')
    p.add_argument('--out',default='outputs')
    p.add_argument('--max-rows',type=int,default=80000,help='0=all eligible observations; default limits RAM and runtime')
    p.add_argument('--cv',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not Path(a.data).exists():raise SystemExit('Missing dataset. Run python scripts/download_data.py')
    raw=pd.read_csv(a.data) if str(a.data).lower().endswith(".csv") else pd.read_parquet(a.data)
    summary={'raw_hcm_rows':int(len(raw)),'raw_duplicate_rows':int(raw.duplicated().sum()),'raw_missing':{k:int(v) for k,v in raw.isna().sum().items()}}
    df=prepare(raw,training=True)
    df=df.drop_duplicates().copy()
    # Remove implausible advertisements; report counts, never make up observations.
    df=df[df['price'].between(100_000_000,500_000_000_000)&df['area'].between(5,5000)].copy()
    df['price_per_m2']=df['price']/df['area']
    df=df[df['price_per_m2'].between(1_000_000,3_000_000_000)].copy()
    summary['eligible_before_sampling']=int(len(df))
    # Deterministic sampling; test and calibration disjoint after sampling.
    if a.max_rows and len(df)>a.max_rows:df=df.sample(n=a.max_rows,random_state=a.seed)
    summary['sample_size']=len(df)
    if len(df)<100:raise ValueError('Insufficient valid records')
    # HOLD OUT test (20%). Never use test for tuning or interval calibration.
    train_dev,test=train_test_split(df,test_size=.20,random_state=a.seed)
    train,calib=train_test_split(train_dev,test_size=.125,random_state=a.seed)
    xtrain=train[FEATURES];ytrain=np.log(train['price'].to_numpy())
    xcal=calib[FEATURES];ycal=calib['price'].to_numpy()
    xtest=test[FEATURES];ytest=test['price'].to_numpy()
    cv=KFold(n_splits=a.cv,shuffle=True,random_state=a.seed)
    trials=[];best_models={}
    for name in PARAMS:
        print('Training',name,flush=True)
        search=RandomizedSearchCV(pipeline(name,a.seed),PARAMS[name],n_iter=min(4,int(np.prod([len(x) for x in PARAMS[name].values()]))),scoring='neg_mean_absolute_error',cv=cv,n_jobs=1,random_state=a.seed,refit=True,error_score='raise')
        search.fit(xtrain,ytrain)
        pred=np.exp(np.clip(search.predict(xtest),0,35))
        metric=scores(ytest,pred)
        trials.append({'model':name,'cv_log_MAE':-search.best_score_,**metric,'params':json.dumps(search.best_params_,ensure_ascii=False)})
        best_models[name]=search.best_estimator_
    leaderboard=pd.DataFrame(trials)
    # Select by CV only, NEVER by test scores.
    winner=min(trials,key=lambda x:x['cv_log_MAE'])['model']
    model=best_models[winner]
    # Refit on development training only; calibration stays untouched for interval.
    residual=np.abs(np.log(ycal)-model.predict(xcal))
    # Split conformal two-sided multiplicative prediction interval (approx 90% marginal coverage)
    q=float(np.quantile(residual,min(1,np.ceil((len(residual)+1)*.90)/len(residual)),method='higher'))
    test_predictions=np.exp(np.clip(model.predict(xtest),0,35))
    coverage=float(np.mean((ytest>=test_predictions*np.exp(-q))&(ytest<=test_predictions*np.exp(q))))
    leaderboard.to_csv(out/'metrics.csv',index=False)
    summary.update({'winner_cv':winner,'calibration_n':len(calib),'test_n':len(test),'conformal_nominal_coverage':.90,'conformal_empirical_test_coverage':coverage,'conformal_log_radius':q,'params_seed':a.seed})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump({'model':model,'features':FEATURES,'log_radius':q,'winner':winner,'metrics':next(x for x in trials if x['model']==winner)},out/'model.joblib')
    pd.DataFrame({'actual_VND':ytest,'predicted_VND':test_predictions}).to_csv(out/'test_predictions.csv',index=False)
    plt.figure(figsize=(7,5));plt.scatter(np.log10(ytest),np.log10(test_predictions),s=3,alpha=.15);lims=[8,12];plt.plot(lims,lims,'k--');plt.xlim(lims);plt.ylim(lims);plt.xlabel('log10 Actual VND');plt.ylabel('log10 Predicted VND');plt.tight_layout();plt.savefig(out/'actual_vs_pred.png',dpi=170);plt.close()
    plt.figure(figsize=(8,4));df['price'].div(1e9).clip(upper=100).hist(bins=60);plt.xlabel('Listed price, billion VND (clipped at 100 for visualization)');plt.ylabel('Listings');plt.tight_layout();plt.savefig(out/'price_distribution.png',dpi=170);plt.close()
    imp=permutation_importance(model,xtest.sample(min(300,len(xtest)),random_state=a.seed),np.log(test.loc[xtest.sample(min(300,len(xtest)),random_state=a.seed).index,'price']),n_repeats=3,random_state=a.seed,scoring='neg_mean_absolute_error')
    pd.DataFrame({'feature':FEATURES,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False).to_csv(out/'permutation_importance.csv',index=False)
    print('Done:',winner,'results:',out)
if __name__=='__main__':main()
