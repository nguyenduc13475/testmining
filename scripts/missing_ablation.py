"""Controlled missing-feature ablation on previously downloaded HCMC listings.

Three strategies, same holdout and nested training samples across repeat/strategy:
  baseline   : existing median/most-frequent imputation, category 'Không rõ'
  drop_high  : remove high-missing features (rate measured on training subset only)
  indicators : baseline + explicit training-derived missingness indicators

No writes to existing models/metrics. This is exploratory, not a causal study.
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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.schema import FEATURES, NUM, EXTRA_NUM, CAT
from src.model import pipeline
from src.evaluation import scores
from scripts.learning_curve import clean_listings

STRATEGIES = ('baseline', 'drop_high', 'indicators')
NUMERIC = list(NUM) + list(EXTRA_NUM)


def missing_mask(frame, feature):
    if feature in CAT:
        col = frame[feature].astype('string')
        return col.isna() | col.str.strip().isin(['', 'Không rõ'])
    return frame[feature].isna()


def augmented(frame, indicator_for):
    data = frame[FEATURES].copy()
    for feature in indicator_for:
        data[f'missing__{feature}'] = missing_mask(data, feature).astype('int8')
    return data


def estimator_for(strategy, drop, indicator_for, seed):
    estimator = pipeline('HistGradientBoosting', seed)
    estimator.set_params(model__learning_rate=0.1, model__max_leaf_nodes=31)
    if strategy == 'baseline':
        return estimator
    orig = estimator.named_steps['pre']
    # Reuse exactly the original preprocessing for retained numeric/categorical fields.
    num_transformer = orig.named_transformers_['num'] if hasattr(orig, 'transformers_') else orig.transformers[0][1]
    cat_transformer = orig.named_transformers_['cat'] if hasattr(orig, 'transformers_') else orig.transformers[1][1]
    kept_num = [c for c in NUMERIC if c not in drop]
    kept_cat = [c for c in CAT if c not in drop]
    transformers = []
    if kept_num:
        transformers.append(('num', num_transformer, kept_num))
    if kept_cat:
        transformers.append(('cat', cat_transformer, kept_cat))
    if strategy == 'indicators':
        transformers.append(('missing', 'passthrough', [f'missing__{c}' for c in indicator_for]))
    estimator.set_params(pre=ColumnTransformer(transformers, sparse_threshold=0))
    return estimator


def execute(df, out, train_n=80000, test_n=20000, repeats=3, threshold=0.7, seed=2026,
            strategies=STRATEGIES):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (0 < threshold <= 1):
        raise ValueError('--threshold must be in (0,1]')
    if train_n < 100 or test_n < 10 or repeats < 1:
        raise ValueError('train-n>=100, test-size>=10, repeats>=1 required')
    if test_n + train_n > len(df):
        raise ValueError(f'Not enough valid rows: {len(df):,}; require at least {train_n+test_n:,}')
    development, test = train_test_split(df, test_size=test_n, random_state=seed, shuffle=True)
    test_y = test['price'].to_numpy()
    detail = []
    audit = []
    # Missingness policy is determined only from training data, not the holdout.
    for rep in range(repeats):
        train = development.sample(n=train_n, random_state=seed+1000+rep)
        rates = {feat: float(missing_mask(train, feat).mean()) for feat in FEATURES}
        dropped = sorted([c for c, v in rates.items() if v >= threshold])
        # Only include indicators of features that exhibit missingness in train.
        indicator_for = sorted([c for c, v in rates.items() if 0.0 < v < 1.0])
        for feat in FEATURES:
            audit.append({'repeat': rep+1, 'feature': feat,
                          'train_missing_pct': round(rates[feat]*100, 5),
                          'drop_high': feat in dropped,
                          'indicator_added': feat in indicator_for})
        print(f'Repeat {rep+1}/{repeats}: drop_high={dropped}', flush=True)
        for strategy in strategies:
            model = estimator_for(strategy, dropped if strategy == 'drop_high' else [],
                                  indicator_for if strategy == 'indicators' else [], seed+rep)
            train_x = augmented(train, indicator_for) if strategy == 'indicators' else train[FEATURES]
            test_x = augmented(test, indicator_for) if strategy == 'indicators' else test[FEATURES]
            t0 = time.perf_counter()
            print(f'  Fit {strategy}: train={train_n:,} / test={test_n:,}', flush=True)
            model.fit(train_x, np.log(train['price'].to_numpy()))
            pred = np.exp(np.clip(model.predict(test_x), 0, 35))
            metrics = scores(test_y, pred)
            row = {'strategy': strategy, 'repeat': rep+1, 'train_n': train_n,
                   'test_n': test_n, 'dropped_n': len(dropped) if strategy == 'drop_high' else 0,
                   'indicator_n': len(indicator_for) if strategy == 'indicators' else 0,
                   'elapsed_seconds': round(time.perf_counter()-t0, 2), **metrics}
            detail.append(row)
            pd.DataFrame(detail).to_csv(out/'missing_ablation_runs.csv', index=False)
            print(f"    MAE={metrics['MAE_VND']/1e9:.4f} billion; R2={metrics['R2']:.4f}", flush=True)
    pd.DataFrame(audit).to_csv(out/'feature_missing_audit.csv', index=False)
    runs = pd.DataFrame(detail)
    summary = runs.groupby('strategy', as_index=False).agg(
        repeats=('repeat', 'count'),
        MAE_billion_VND=('MAE_VND', lambda x: x.mean()/1e9),
        MAE_std_billion_VND=('MAE_VND', lambda x: x.std(ddof=1)/1e9 if len(x)>1 else float('nan')),
        RMSE_billion_VND=('RMSE_VND', lambda x: x.mean()/1e9),
        R2=('R2', 'mean'), R2_std=('R2', lambda x: x.std(ddof=1) if len(x)>1 else float('nan')),
        MAPE_percent=('MAPE_percent', 'mean'),
        mean_fit_seconds=('elapsed_seconds', 'mean'))
    summary.to_csv(out/'missing_ablation_summary.csv', index=False)
    for metric, err, label, filename in [
        ('MAE_billion_VND', 'MAE_std_billion_VND', 'MAE (billion VND)', 'missing_ablation_mae.png'),
        ('R2', 'R2_std', 'R-squared', 'missing_ablation_r2.png')]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(summary['strategy'], summary[metric], yerr=summary[err].fillna(0), capsize=5)
        ax.set_ylabel(label)
        ax.set_xlabel('Missing-value strategy')
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(out/filename, dpi=160)
        plt.close(fig)
    meta = {'dataset_rows_eligible': len(df), 'train_n': train_n, 'test_n': test_n,
            'repeats': repeats, 'seed': seed, 'threshold': threshold,
            'strategies': list(strategies),
            'split': 'fixed train_test_split(df, test_size=test_n, seed=seed) identical to learning_curve.py when seed/test_n match',
            'selection': 'repeat train subset seeded with seed+1000+repeat_index, same as learning_curve.py',
            'missingness': 'computed on training data only; unknown categorical strings count as missing',
            'std_note': 'sample standard deviation across overlapping training subsets; descriptive, not an uncertainty interval',
            'caveat': 'Random listing split may include near duplicate advertisements; scores need spatial/time holdout before deployment.'}
    (out/'missing_ablation_info.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSUMMARY\n', summary.to_string(index=False), '\nSaved to:', out, flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/raw/hcm_listings.parquet')
    p.add_argument('--out', default='outputs/experiments/missing_ablation')
    p.add_argument('--train-n', type=int, default=80000)
    p.add_argument('--test-size', type=int, default=20000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threshold', type=float, default=0.70)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
    args = p.parse_args()
    path = Path(args.data)
    if not path.is_file():
        raise SystemExit(f'No local dataset at {path}. Use --data to point to existing CSV/Parquet; no download required.')
    print(f'Reading existing dataset: {path}', flush=True)
    raw = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_parquet(path)
    df = clean_listings(raw)
    print(f'Raw={len(raw):,}; eligible={len(df):,}', flush=True)
    execute(df, args.out, args.train_n, args.test_size, args.repeats, args.threshold,
            args.seed, args.strategies)

if __name__ == '__main__':
    main()
