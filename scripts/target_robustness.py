"""CO3029 experiment 5: price audit and target/loss robustness.

Uses local data only. Uses a fixed validation holdout for model selection and
fixed test holdout for reporting. Baseline and alternatives have identical
training subsets per repetition and identical validation/test rows.
No trained models or prior outputs are modified.
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
from src.evaluation import scores
from scripts.learning_curve import clean_listings
from scripts.generalization_audit import make_proxy

CANDIDATES = {
    'log_squared': ('squared_error', 'log'),
    'log_absolute': ('absolute_error', 'log'),
    'raw_squared': ('squared_error', 'billion'),
    'raw_absolute': ('absolute_error', 'billion'),
    'raw_gamma': ('gamma', 'billion'),
}
BANDS = [(0, 3, '<3b'), (3, 10, '3-10b'), (10, 30, '10-30b'), (30, float('inf'), '>=30b')]


def price_audit(train, out):
    """Audit price/m2 distribution on training sample only, no row deletion."""
    work = train[['property_type_name', 'price', 'area']].copy()
    work['billion_vnd'] = work.price / 1e9
    work['million_vnd_per_m2'] = work.price / work.area / 1e6
    records = []
    for name, grp in [('ALL', work)] + list(work.groupby('property_type_name', sort=True)):
        ratio = grp['million_vnd_per_m2'].to_numpy(dtype=float)
        logs = np.log(ratio)
        q1, q3 = np.quantile(logs, [0.25, 0.75])
        low, high = np.exp([q1-3*(q3-q1), q3+3*(q3-q1)])
        row = {'property_type':name, 'n':len(grp),
               'price_p50_billion':float(np.quantile(grp.billion_vnd,.5)),
               'price_p95_billion':float(np.quantile(grp.billion_vnd,.95)),
               'price_p99_billion':float(np.quantile(grp.billion_vnd,.99)),
               'ppm2_p01_million':float(np.quantile(ratio,.01)),
               'ppm2_p05_million':float(np.quantile(ratio,.05)),
               'ppm2_p50_million':float(np.quantile(ratio,.50)),
               'ppm2_p95_million':float(np.quantile(ratio,.95)),
               'ppm2_p99_million':float(np.quantile(ratio,.99)),
               'ppm2_p999_million':float(np.quantile(ratio,.999)),
               'log_iqr_lower_million':float(low),
               'log_iqr_upper_million':float(high),
               'flagged_low_n':int((ratio<low).sum()),
               'flagged_high_n':int((ratio>high).sum())}
        records.append(row)
    pd.DataFrame(records).to_csv(out/'training_price_audit.csv', index=False)
    return records


def metrics(y, pred):
    result = scores(y, pred)
    err = pred - y
    abserr = np.abs(err)
    n = len(y)
    top = max(1, int(np.ceil(n*.05)))
    top_ae = np.partition(abserr, n-top)[n-top:].sum()
    result.update({
        'bias_billion_VND':float(err.mean()/1e9),
        'p95_absolute_error_billion_VND':float(np.quantile(abserr,.95)/1e9),
        'top5_absolute_error_share_percent':float(100*top_ae/max(abserr.sum(),1e-10)),
    })
    return result


def segments(y, pred, source, protocol, repeat, candidate, split):
    rows = []
    kinds = [('ALL',np.ones(len(y),dtype=bool))]
    for lo,hi,label in BANDS:
        kinds.append((label,(y/1e9>=lo)&(y/1e9<hi)))
    property_types = source['property_type_name'].astype('string').fillna('Không rõ').to_numpy()
    for name, count in pd.Series(property_types).value_counts().items():
        if count >= 100:
            kinds.append(('type:'+str(name),property_types==name))
    for label, mask in kinds:
        if mask.sum()<2:
            continue
        row = {'protocol':protocol,'repeat':repeat,'candidate':candidate,
               'split':split,'group':label,'n':int(mask.sum())}
        row.update(metrics(y[mask], pred[mask]))
        rows.append(row)
    return rows


def build_splits(df, protocol, seed, test_n, val_n):
    # Match the exact fixed random test set in generalization_audit.py.
    random_dev, random_test = train_test_split(df,test_size=test_n,random_state=seed,shuffle=True)
    if protocol == 'group_purged':
        proxy=make_proxy(df)
        test_keys=set(proxy.loc[random_test.index])
        dev=random_dev.loc[~proxy.loc[random_dev.index].isin(test_keys).to_numpy()]
        test=random_test
    elif protocol == 'random':
        dev,test=random_dev,random_test
        proxy=None
    elif protocol == 'temporal':
        dated=df.loc[df.published_at.notna()].sort_values('published_at')
        test=dated.iloc[-test_n:]
        cutoff=test.published_at.iloc[0]
        dev=dated.loc[dated.published_at<cutoff]
        proxy=None
    else:
        raise ValueError('Unknown protocol')
    if len(dev)<=val_n:
        raise ValueError('Not enough development samples for validation')
    if protocol == 'temporal':
        # Latest development rows are the validation set; the earlier rows train.
        val=dev.iloc[-val_n:]
        trainpool=dev.iloc[:-val_n]
    else:
        trainpool, val=train_test_split(dev,test_size=val_n,random_state=seed+77,shuffle=True)
        if protocol == 'group_purged':
            val_keys=set(proxy.loc[val.index])
            trainpool=trainpool.loc[~proxy.loc[trainpool.index].isin(val_keys).to_numpy()]
    return trainpool,val,test


def run(df,out,train_n,test_n,val_n,repeats,seed,protocol,candidates):
    if repeats<1 or train_n<100 or val_n<100 or test_n<100:
        raise ValueError('repeats >=1 and train/validation/test sizes >=100 required')
    df=df.reset_index(drop=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    trainpool,val,test=build_splits(df,protocol,seed,test_n,val_n)
    if len(trainpool)<train_n:
        raise ValueError(f'Train pool too small: {len(trainpool)} < {train_n}')
    info={'eligible_rows':len(df),'protocol':protocol,'seed':seed,'train_n':train_n,
          'validation_n':len(val),'test_n':len(test),'train_pool_n':len(trainpool),
          'repeats':repeats,'candidates':candidates,
          'heldout':'Same fixed validation and test sets across all candidates and repetitions.',
          'selection':'Select only by validation MAE; test metrics are reported separately and must not be used for tuning.',
          'group_note':'Price band from actual price is used ONLY for offline diagnostic groups.',
          'audit_note':'Training price-per-m2 thresholds are diagnostic only; no test-based cleaning or automatic row deletion.',
          'prior_test_note':'This dataset/test has already been explored in earlier experiments, so final test metrics are exploratory, not a pristine blind estimate.',
          'no_overwrite':'Outputs are separate; no app/model changes.'}
    (out/'experiment_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    records=[]
    for rep in range(1,repeats+1):
        train=trainpool.sample(n=train_n,random_state=seed+1000+rep-1)
        if rep==1:
            price_audit(train,out)
        print(f'Repetition {rep}/{repeats} (train={len(train):,}, validation={len(val):,}, test={len(test):,})',flush=True)
        for candidate in candidates:
            loss, scale=CANDIDATES[candidate]
            model=pipeline('HistGradientBoosting',seed+rep-1)
            model.set_params(model__learning_rate=0.1,model__max_leaf_nodes=31,model__loss=loss)
            y_train=(np.log(train.price.to_numpy(dtype=float)) if scale=='log' else train.price.to_numpy(dtype=float)/1e9)
            t0=time.perf_counter()
            model.fit(train[FEATURES],y_train)
            elapsed=time.perf_counter()-t0
            for split,subset in [('validation',val),('test',test)]:
                rawpred=model.predict(subset[FEATURES])
                pred=(np.exp(np.clip(rawpred,0,35)) if scale=='log' else np.clip(rawpred,0.1,500)*1e9)
                y=subset.price.to_numpy(dtype=float)
                chunk=segments(y,pred,subset,protocol,rep,candidate,split)
                for row in chunk:
                    row['fit_seconds']=round(elapsed,3)
                records.extend(chunk)
            main=[r for r in records if r['candidate']==candidate and r['repeat']==rep and r['group']=='ALL']
            valrow=next(r for r in main if r['split']=='validation')
            tstrow=next(r for r in main if r['split']=='test')
            print(f'  {candidate:14s} val MAE={valrow["MAE_VND"]/1e9:.3f}b, test MAE={tstrow["MAE_VND"]/1e9:.3f}b, test R2={tstrow["R2"]:.4f}',flush=True)
            pd.DataFrame(records).to_csv(out/'robustness_runs.csv',index=False)
    frame=pd.DataFrame(records)
    aggregate=(frame.groupby(['candidate','split','group'],as_index=False,sort=False)
               .agg(repeats=('repeat','nunique'),n=('n','mean'),
                    MAE_billion_VND=('MAE_VND',lambda x:x.mean()/1e9),
                    MAE_std_billion_VND=('MAE_VND',lambda x:x.std(ddof=1)/1e9 if len(x)>1 else np.nan),
                    RMSE_billion_VND=('RMSE_VND',lambda x:x.mean()/1e9),
                    R2=('R2','mean'),MAPE_percent=('MAPE_percent','mean'),
                    bias_billion_VND=('bias_billion_VND','mean'),
                    p95_abs_error_billion_VND=('p95_absolute_error_billion_VND','mean'),
                    top5_error_share_percent=('top5_absolute_error_share_percent','mean'),
                    mean_fit_seconds=('fit_seconds','mean')))
    aggregate.to_csv(out/'robustness_summary.csv',index=False)
    val_summary=aggregate[(aggregate.split=='validation')&(aggregate.group=='ALL')].sort_values('MAE_billion_VND')
    winner=val_summary.iloc[0]['candidate']
    selection={'criterion':'mean_validation_MAE_billion_VND','selected_candidate':winner,
               'validation_ranking':val_summary[['candidate','MAE_billion_VND','R2']].to_dict('records'),
               'test_note':'Tests are exploratory because previous analysis has repeatedly inspected related holdouts.'}
    (out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf8')
    overall=aggregate[aggregate.group=='ALL']
    for metric,ylabel,filename in [
        ('MAE_billion_VND','MAE (billion VND)','robustness_mae.png'),
        ('R2','R-squared','robustness_r2.png'),
        ('bias_billion_VND','Signed bias (billion VND)','robustness_bias.png')]:
        piv=overall.pivot(index='candidate',columns='split',values=metric).reindex(candidates)
        fig,ax=plt.subplots(figsize=(9,4.7))
        xx=np.arange(len(piv)); width=.35
        ax.bar(xx-width/2,piv.validation,width,label='Validation')
        ax.bar(xx+width/2,piv.test,width,label='Test')
        ax.set_xticks(xx,piv.index,rotation=17)
        ax.set_ylabel(ylabel)
        ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
    print(f'Validation-selected candidate: {winner}. Results: {out}',flush=True)
    return aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/raw/hcm_listings.parquet')
    p.add_argument('--out',default='outputs/experiments/target_robustness_final')
    p.add_argument('--train-n',type=int,default=80000)
    p.add_argument('--test-size',type=int,default=20000)
    p.add_argument('--validation-size',type=int,default=10000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--protocol',choices=['random','group_purged','temporal'],default='group_purged')
    p.add_argument('--candidates',nargs='+',choices=list(CANDIDATES),default=list(CANDIDATES))
    args=p.parse_args()
    path=Path(args.data)
    if not path.is_file():
        raise SystemExit(f'Missing local data: {path}. Pass --data pointing to your existing Parquet/CSV; no downloading.')
    raw=pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_parquet(path)
    df=clean_listings(raw)
    del raw
    print(f'Eligible rows={len(df):,}; protocol={args.protocol}',flush=True)
    run(df,args.out,args.train_n,args.test_size,args.validation_size,args.repeats,
        args.seed,args.protocol,args.candidates)


if __name__=='__main__':
    main()
