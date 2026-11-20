"""Generalization audit for CO3029. Uses local parquet/CSV; never downloads.

Three evaluations:
  random: random listing holdout, usual training pool;
  group_purged: IDENTICAL holdout, but remove from training pool any records
                with the same conservative near-duplicate listing proxy;
  temporal: newest 20k dated listings, strictly older training pool.
All fits use the baseline HistGradientBoosting pipeline and fixed parameters.
These are observational diagnostics, not evidence of causality.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
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


def normalize_text(s):
    if pd.isna(s):
        return ''
    s = unicodedata.normalize('NFKC', str(s)).casefold().strip()
    return re.sub(r'\s+', ' ', s)


def make_proxy(df):
    """Conservative proxy: normalized non-generic title + district + street + rounded area.
    Missing/short titles or missing location are set to per-row unique keys.
    This is *not* an actual property ID; both false matches/misses are possible.
    """
    names = df['name'].map(normalize_text) if 'name' in df else pd.Series('', index=df.index)
    district = df['district_name'].map(normalize_text)
    street = df['street_name'].map(normalize_text)
    valid = (names.str.len() >= 22) & (district.str.len() >= 3) & ~street.isin(['', 'không rõ', 'nan'])
    areas = df['area'].round(0).astype('Int64').astype('string')
    raw_key = names + '|' + district + '|' + street + '|' + areas.fillna('?')
    # The dataset was deduplicated and has a stable RangeIndex at this point.
    unique_key = pd.Series('unique:' + pd.RangeIndex(len(df)).astype(str), index=df.index)
    result = pd.Series(np.where(valid, 'proxy:' + raw_key, unique_key), index=df.index)
    return result


def metric_rows(y, pred, segment, repeat, split, min_group_n):
    records = []
    def emit(kind, label, mask):
        n = int(mask.sum())
        if n < (2 if kind == 'overall' else min_group_n):
            return
        metrics = scores(y[mask], pred[mask])
        records.append({'split':split,'repeat':repeat,'segment':kind,'group':str(label),'n':n,**metrics})

    emit('overall', 'ALL', np.ones(len(y), dtype=bool))
    price_b = pd.cut(y/1e9, bins=[0,3,10,30,np.inf], right=False,
                     labels=['<3 tỷ','3–10 tỷ','10–30 tỷ','≥30 tỷ'])
    for cat in price_b.categories:
        emit('actual_price_band', cat, np.asarray(price_b == cat))
    types = segment['property_type_name'].astype('string').fillna('Không rõ')
    for name, n in types.value_counts().items():
        if n >= min_group_n:
            emit('property_type', name, (types.to_numpy() == name))
    return records


def run(df, out, train_n, test_n, repeats, seed, min_group_n):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if min(train_n,test_n) < 100 or repeats < 1:
        raise ValueError('train/test >=100, repeats>=1 required')
    if len(df) <= train_n+test_n:
        raise ValueError('Not enough eligible rows for train + test')
    # Unique positional index to avoid accidental duplicate alignment.
    df = df.reset_index(drop=True)
    print('Creating near-duplicate proxy keys...',flush=True)
    proxy = make_proxy(df)
    random_dev, random_test = train_test_split(df, test_size=test_n, random_state=seed, shuffle=True)
    keys_test = set(proxy.iloc[random_test.index].tolist())
    is_purged = ~proxy.iloc[random_dev.index].isin(keys_test)
    purged_dev = random_dev.loc[is_purged.to_numpy()]
    shared = int((~is_purged).sum())
    risky_test = int(proxy.iloc[random_test.index].isin(set(proxy.iloc[random_dev.index])).sum())
    if len(purged_dev) < train_n:
        raise ValueError(f'Group-purged train pool too small ({len(purged_dev):,} < {train_n:,})')

    # Strict chronological separation, excluding any timestamp tie at the boundary.
    dated = df.loc[df['published_at'].notna()].sort_values('published_at')
    if len(dated) <= train_n + test_n:
        raise ValueError(f'Only {len(dated):,} valid timestamps; insufficient for strict temporal holdout')
    temporal_test = dated.iloc[-test_n:]
    cutoff = temporal_test['published_at'].iloc[0]
    temporal_dev = dated.loc[dated['published_at'] < cutoff]
    if len(temporal_dev) < train_n:
        raise ValueError('Insufficient strictly older listings for temporal training; lower --train-n or --test-size')
    # Report gaps; different holdouts should NOT be directly interpreted as solely split effects.
    plans = [('random', random_dev, random_test),
             ('group_purged', purged_dev, random_test),
             ('temporal', temporal_dev, temporal_test)]
    print(f'Eligible {len(df):,}; random test {len(random_test):,}; '
          f'purged training candidates {len(purged_dev):,} (removed {shared:,}); '
          f'proxy-flagged random test listings {risky_test:,}; '
          f'time cutoff {cutoff}',flush=True)
    metadata = {
        'eligible_rows':len(df),'train_n':train_n,'test_n':test_n,'repeats':repeats,
        'seed':seed,'min_group_n':min_group_n,
        'random_vs_group_same_test':True,'purged_training_candidates':len(purged_dev),
        'purged_candidate_rows':shared, 'random_test_proxy_overlap_count':risky_test,
        'random_test_proxy_overlap_pct':100*risky_test/len(random_test),
        'temporal_cutoff_utc':str(cutoff),
        'temporal_training_end_utc':str(temporal_dev['published_at'].max()),
        'temporal_test_end_utc':str(temporal_test['published_at'].max()),
        'temporal_note':'Temporal test differs from random test; raw metric differences mix period/data shift with split effects.',
        'proxy_note':'Only exact match of normalized long title, district, street and rounded area; not an actual property identity. False negatives likely.',
        'metric_note':'Price bands defined from true test price only, for offline error analysis; never used for prediction.',
        'model':'HistGradientBoosting, max_leaf_nodes=31, learning_rate=0.1, as learning_curve baseline',
        'repeat_note':'same holdout and separate seeded subsets; repeated fits are correlated, sample std is descriptive only.',
        'project_outputs':'No trained model, existing app, or previous experiment outputs are modified.',
    }
    (out/'generalization_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    records=[]
    for split, dev, test in plans:
        print(f'\nSPLIT {split}: pool={len(dev):,}, holdout={len(test):,}',flush=True)
        x_test = test[FEATURES]
        y_test = test['price'].to_numpy(dtype=float)
        for repeat in range(1,repeats+1):
            train = dev.sample(n=train_n, random_state=seed + 1000 + repeat - 1)
            estimator = pipeline('HistGradientBoosting', seed + repeat - 1)
            estimator.set_params(model__learning_rate=0.1,model__max_leaf_nodes=31)
            t0 = time.perf_counter()
            estimator.fit(train[FEATURES], np.log(train['price'].to_numpy(dtype=float)))
            prediction = np.exp(np.clip(estimator.predict(x_test),0,35))
            elapsed = round(time.perf_counter()-t0,2)
            rows = metric_rows(y_test,prediction,test,repeat,split,min_group_n)
            for row in rows:
                row['fit_seconds'] = elapsed
            records += rows
            pd.DataFrame(records).to_csv(out/'generalization_runs.csv',index=False)
            allrow=next(r for r in rows if r['segment']=='overall')
            print(f' {repeat}/{repeats}: MAE={allrow["MAE_VND"]/1e9:.3f} tỷ VND, R2={allrow["R2"]:.4f}, elapsed={elapsed:.1f}s',flush=True)
    results = pd.DataFrame(records)
    grouped = (results.groupby(['split','segment','group'],sort=False,as_index=False)
               .agg(repeats=('repeat','count'),test_n=('n','mean'),
                    MAE_billion_VND=('MAE_VND',lambda x: x.mean()/1e9),
                    MAE_std_billion_VND=('MAE_VND',lambda x: x.std(ddof=1)/1e9 if len(x)>1 else float('nan')),
                    RMSE_billion_VND=('RMSE_VND',lambda x: x.mean()/1e9),
                    R2=('R2','mean'),
                    R2_std=('R2',lambda x: x.std(ddof=1) if len(x)>1 else float('nan')),
                    MAPE_percent=('MAPE_percent','mean'),
                    mean_fit_seconds=('fit_seconds','mean')))
    grouped.to_csv(out/'generalization_summary.csv',index=False)
    overall = grouped.loc[grouped.segment=='overall'].set_index('split').reindex([x[0] for x in plans])
    for m,ylabel,filename in [('MAE_billion_VND','MAE (tỷ VND)','generalization_mae.png'),
                              ('R2','R²','generalization_r2.png')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        v=overall[m]
        ax.bar(v.index.tolist(),v.to_numpy())
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Evaluation protocol')
        ax.grid(axis='y',alpha=.25)
        fig.tight_layout()
        fig.savefig(out/filename,dpi=150)
        plt.close(fig)
    for split in ['random','group_purged','temporal']:
        price=grouped.loc[(grouped.split==split)&(grouped.segment=='actual_price_band')]
        if len(price):
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.bar(price['group'],price['MAE_billion_VND'])
            ax.set_ylabel('MAE (tỷ VND)')
            ax.set_title(f'Error by actual price band — {split}')
            ax.grid(axis='y',alpha=.2)
            fig.tight_layout()
            fig.savefig(out/f'price_band_{split}.png',dpi=150)
            plt.close(fig)
    print('\nOverall results:\n', overall[['MAE_billion_VND','R2','MAPE_percent']].to_string(),flush=True)
    print('Saved files to',out,flush=True)
    return grouped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/raw/hcm_listings.parquet')
    p.add_argument('--out',default='outputs/experiments/generalization_final')
    p.add_argument('--train-n',type=int,default=80000)
    p.add_argument('--test-size',type=int,default=20000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-group-n',type=int,default=100)
    a=p.parse_args()
    path=Path(a.data)
    if not path.is_file():
        raise SystemExit(f'File not found: {path}. Point --data to existing local CSV/Parquet (no download).')
    print(f'Loading local dataset {path}',flush=True)
    raw=pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_parquet(path)
    df=clean_listings(raw)
    del raw
    run(df,a.out,a.train_n,a.test_size,a.repeats,a.seed,a.min_group_n)

if __name__=='__main__':
    main()
