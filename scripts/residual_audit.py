"""CO3029 residual diagnostics. Uses existing local data; does not modify the app/model.

Reproduces the generalization_audit splits/seeds for paired comparisons.
Price bands based on true prices are used for OFFLINE evaluation only.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.schema import FEATURES
from src.model import pipeline
from scripts.learning_curve import clean_listings
from scripts.generalization_audit import make_proxy

BANDS=[0,3,10,30,float('inf')]
LABELS=['<3 tỷ','3–10 tỷ','10–30 tỷ','≥30 tỷ']
COLS=['split','repeat','segment','group','n','MAE_billion_VND','RMSE_billion_VND','MAPE_percent','signed_bias_billion_VND','median_signed_billion_VND','actual_mean_billion_VND','predicted_mean_billion_VND','underpredict_pct','overpredict_pct','sum_absolute_error_billion_VND']


def metrics(y,pred,split,repeat,kind,label,mask):
    actual=np.asarray(y[mask],dtype=float)
    predicted=np.asarray(pred[mask],dtype=float)
    if len(actual)==0:
        return None
    err=predicted-actual
    return dict(zip(COLS,[split,repeat,kind,str(label),len(actual),np.abs(err).mean()/1e9,np.sqrt(np.square(err).mean())/1e9,100*np.mean(np.abs(err)/actual),err.mean()/1e9,np.median(err)/1e9,actual.mean()/1e9,predicted.mean()/1e9,100*np.mean(err<0),100*np.mean(err>0),np.abs(err).sum()/1e9]))


def diagnostics(actual,pred,split,repeat,types,min_group_n):
    actual=np.asarray(actual,dtype=float)
    pred=np.asarray(pred,dtype=float)
    n=len(actual)
    rows=[metrics(actual,pred,split,repeat,'overall','ALL',np.ones(n,dtype=bool))]
    pricegroup=pd.cut(pd.Series(actual/1e9),BANDS,labels=LABELS,right=False)
    for band in LABELS:
        mask=(pricegroup==band).to_numpy()
        if mask.sum()>=min_group_n:
            rows.append(metrics(actual,pred,split,repeat,'actual_price_band',band,mask))
    v=types.fillna('Không rõ').astype(str).to_numpy()
    for typ in pd.Series(v).value_counts().index:
        mask=v==typ
        if mask.sum()>=min_group_n:
            rows.append(metrics(actual,pred,split,repeat,'property_type',typ,mask))
    return rows


def tail_metrics(actual,pred,split,repeat):
    residual=np.asarray(pred,dtype=float)-np.asarray(actual,dtype=float)
    abs_err=np.abs(residual)
    total=abs_err.sum()
    n=len(abs_err)
    rows=[]
    for pct in (1,5,10):
        count=max(1,int(np.ceil(n*pct/100)))
        selection=np.argpartition(abs_err,-count)[-count:]
        share=100*abs_err[selection].sum()/total if total else 0
        rows.append({'split':split,'repeat':repeat,'top_percent':pct,'top_n':count,'absolute_error_share_percent':share,
                     'threshold_billion_VND':float(np.min(abs_err[selection])/1e9)})
    for q in (50,75,90,95,99):
        rows.append({'split':split,'repeat':repeat,'percentile':q,'absolute_error_percentile_billion_VND':float(np.percentile(abs_err,q)/1e9),
                     'absolute_percent_error_percentile':float(np.percentile(100*abs_err/np.asarray(actual),q))})
    return rows


def plot_groups(summary,out):
    order=['random','group_purged','temporal']
    groups=summary[summary.segment=='actual_price_band']
    for value,ylab,filename in [('signed_bias_billion_VND','Mean signed error (billion VND)','signed_bias_by_band.png'),
                                ('MAE_billion_VND','MAE (billion VND)','mae_by_band.png'),
                                ('MAPE_percent','MAPE (%)','mape_by_band.png')]:
        fig,ax=plt.subplots(figsize=(9,4.8))
        x=np.arange(len(LABELS))
        for i,split in enumerate(order):
            part=groups.loc[groups.split==split].set_index('group').reindex(LABELS)
            ax.bar(x+(i-1)*.24,part[value].to_numpy(dtype=float),width=.23,label=split)
        ax.set_xticks(x,LABELS);ax.set_ylabel(ylab);ax.axhline(0,color='black',linewidth=.8)
        ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
    # Bar plot of calibration gap for the same price buckets.
    fig,ax=plt.subplots(figsize=(9,4.8))
    group=groups.loc[groups.split=='group_purged'].set_index('group').reindex(LABELS)
    x=np.arange(len(LABELS))
    ax.bar(x-.18,group.actual_mean_billion_VND,width=.35,label='Actual mean')
    ax.bar(x+.18,group.predicted_mean_billion_VND,width=.35,label='Predicted mean')
    ax.set_xticks(x,LABELS);ax.set_ylabel('Price (billion VND)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'actual_vs_predicted_by_band.png',dpi=160);plt.close(fig)


def run(df,out,train_n,test_n,repeats,seed,min_group_n,protocols):
    if train_n<100 or test_n<100 or repeats<1: raise ValueError('train/test>=100 and repeats>=1')
    df=df.reset_index(drop=True)
    if len(df)<=train_n+test_n:raise ValueError('Insufficient eligible rows')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected=set(protocols)
    random_dev,random_test=train_test_split(df,test_size=test_n,random_state=seed,shuffle=True)
    plans={}
    proxy_note='Proxy matches are NOT verified same properties.'
    purged_count=None
    if 'random' in selected: plans['random']=(random_dev,random_test)
    if 'group_purged' in selected:
        keys=make_proxy(df)
        test_keys=set(keys.iloc[random_test.index]);dev_keys=keys.iloc[random_dev.index]
        keep=(~dev_keys.isin(test_keys)).to_numpy()
        purged_count=int((~keep).sum())
        plans['group_purged']=(random_dev.iloc[keep],random_test)
    cutoff=None
    if 'temporal' in selected:
        dated=df[df.published_at.notna()].sort_values('published_at')
        if len(dated)<=test_n+train_n:raise ValueError('Insufficient dated listings')
        temporal_test=dated.iloc[-test_n:]
        cutoff=temporal_test.published_at.iloc[0]
        temporal_dev=dated.loc[dated.published_at<cutoff]
        plans['temporal']=(temporal_dev,temporal_test)
    for split,(dev,test) in plans.items():
        if len(dev)<train_n:raise ValueError(f'{split}: insufficient train pool')
    info={'eligible_rows':int(len(df)),'train_n':train_n,'test_n':test_n,'repeats':repeats,'seed':seed,
          'protocols':list(plans),'purged_candidate_rows':purged_count,'temporal_cutoff_utc':str(cutoff) if cutoff is not None else None,
          'holdout':'Reuses exact generalization_audit holdout and sampling seeds; random and group_purged share test.',
          'grouping':'Actual price bands used ONLY for offline evaluation, not to select models at inference.',
          'proxy_note':proxy_note,'temporal_note':'Temporal has a different test distribution; cross-protocol metrics are not causal comparisons.',
          'repetitions':'Shared test and correlated training subsets; standard deviations are descriptive, not independent confidence intervals.',
          'no_overwrites':'New experiment folder; existing data/model/old outputs remain untouched.'}
    (out/'residual_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    rows=[];tails=[]
    for split,(dev,test) in plans.items():
        y=test.price.to_numpy(dtype=float)
        print(f'Protocol={split} / train pool={len(dev):,} / test={len(test):,}',flush=True)
        for repeat in range(1,repeats+1):
            train=dev.sample(n=train_n,random_state=seed+1000+repeat-1)
            estimator=pipeline('HistGradientBoosting',seed+repeat-1)
            estimator.set_params(model__learning_rate=.1,model__max_leaf_nodes=31)
            t0=time.perf_counter()
            estimator.fit(train[FEATURES],np.log(train.price.to_numpy(dtype=float)))
            pred=np.exp(np.clip(estimator.predict(test[FEATURES]),0,35))
            this=diagnostics(y,pred,split,repeat,test.property_type_name,min_group_n)
            rows.extend(this);tails.extend(tail_metrics(y,pred,split,repeat))
            pd.DataFrame(rows,columns=COLS).to_csv(out/'residual_runs.csv',index=False)
            pd.DataFrame(tails).to_csv(out/'residual_tail_runs.csv',index=False)
            overall=this[0]
            print(f'  repeat {repeat}/{repeats}, MAE={overall["MAE_billion_VND"]:.3f} bn, signed bias={overall["signed_bias_billion_VND"]:+.3f} bn, elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    allruns=pd.DataFrame(rows)
    means=allruns.groupby(['split','segment','group'],sort=False,as_index=False).agg({
        **{c:'mean' for c in COLS if c not in ('split','repeat','segment','group')},'repeat':'count'})
    means=means.rename(columns={'repeat':'repeats'})
    means.to_csv(out/'residual_summary.csv',index=False)
    tail_df=pd.DataFrame(tails)
    tail_means=tail_df.groupby(['split','top_percent'],dropna=True,as_index=False)[['absolute_error_share_percent','threshold_billion_VND']].mean()
    tail_means.to_csv(out/'residual_tail_summary.csv',index=False)
    quant_df=tail_df.dropna(subset=['percentile']).groupby(['split','percentile'],as_index=False)[['absolute_error_percentile_billion_VND','absolute_percent_error_percentile']].mean()
    quant_df.to_csv(out/'residual_quantiles.csv',index=False)
    plot_groups(means,out)
    print('Saved residual audit to',out,flush=True)
    return means


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/raw/hcm_listings.parquet')
    p.add_argument('--out',default='outputs/experiments/residual_final')
    p.add_argument('--train-n',type=int,default=80000)
    p.add_argument('--test-size',type=int,default=20000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-group-n',type=int,default=100)
    p.add_argument('--protocols',nargs='+',choices=['random','group_purged','temporal'],default=['random','group_purged','temporal'])
    a=p.parse_args()
    file=Path(a.data)
    if not file.is_file():p.error(f'Existing data not found at {file}; use --data path/to/your/local/file')
    raw=pd.read_csv(file) if file.suffix.lower()=='.csv' else pd.read_parquet(file)
    df=clean_listings(raw)
    del raw
    run(df,a.out,a.train_n,a.test_size,a.repeats,a.seed,a.min_group_n,a.protocols)

if __name__=='__main__':main()
