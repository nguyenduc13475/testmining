"""Controlled sample-size study, reusing the already-downloaded HCMC parquet.

A fixed holdout is sampled before creating nested training subsets. All
preprocessing parameters are fitted on each training subset inside Pipeline.
Outputs are isolated from the application model and existing metrics.
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
from src.schema import prepare, FEATURES
from src.model import pipeline, PARAMS
from src.evaluation import scores


def clean_listings(raw):
    """Exactly mirror the eligibility rules in scripts/train.py."""
    df = prepare(raw, training=True).drop_duplicates().copy()
    df = df[df['price'].between(100_000_000, 500_000_000_000) & df['area'].between(5, 5000)].copy()
    df['price_per_m2'] = df['price'] / df['area']
    return df[df['price_per_m2'].between(1_000_000, 3_000_000_000)].copy()


def run_experiment(df, sizes, model_names, test_n, seed, repeats, output, max_seconds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if test_n < 10 or test_n >= len(df):
        raise ValueError('Invalid --test-size (must be >=10 and smaller than eligible rows)')
    # Always select the exact same test indices, independent of --sizes.
    development, test = train_test_split(df, test_size=test_n, random_state=seed, shuffle=True)
    sizes = sorted(set(sizes))
    if sizes[0] < 100 or sizes[-1] > len(development):
        raise ValueError(f'Training sizes must fall between 100 and {len(development):,}')

    # Deterministic nested training samples: training set of size N is a prefix
    # of training set of size M, whenever N<M for the same repetition.
    test_x = test[FEATURES]
    test_y = test['price'].to_numpy()
    rows = []
    started = time.perf_counter()
    # Subsamples can vary across repetitions; holdout NEVER changes.
    for rep in range(repeats):
        ordered = development.sample(frac=1, random_state=seed + 1000 + rep)
        for size in sizes:
            train = ordered.iloc[:size]
            x_train = train[FEATURES]
            y_train = np.log(train['price'].to_numpy())
            for model_name in model_names:
                estimator = pipeline(model_name, seed + rep)
                if model_name == 'HistGradientBoosting':
                    estimator.set_params(model__learning_rate=0.1, model__max_leaf_nodes=31)
                if model_name == 'RandomForest':
                    estimator.set_params(model__max_depth=20, model__min_samples_leaf=2)
                t0 = time.perf_counter()
                print(f'Fit model={model_name}, samples={size:,}, repeat={rep+1}/{repeats}', flush=True)
                estimator.fit(x_train, y_train)
                predict = np.exp(np.clip(estimator.predict(test_x), 0, 35))
                metrics = scores(test_y, predict)
                row = {
                    'model': model_name, 'repeat': rep + 1,
                    'train_n': size, 'test_n': len(test),
                    'fit_and_predict_seconds': round(time.perf_counter() - t0, 2),
                    **metrics,
                }
                rows.append(row)
                # Save after each fit, so interrupted experiments retain evidence.
                pd.DataFrame(rows).to_csv(output / 'learning_curve.csv', index=False)
                print(f"   MAE={metrics['MAE_VND']/1e9:.3f} billion VND, "
                      f"R2={metrics['R2']:.4f}, time={row['fit_and_predict_seconds']:.1f}s", flush=True)
                if max_seconds is not None and time.perf_counter()-started >= max_seconds:
                    print('Time budget reached. Partial results saved.', flush=True)
                    break
            if max_seconds is not None and time.perf_counter()-started >= max_seconds:
                break
        if max_seconds is not None and time.perf_counter()-started >= max_seconds:
            break

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError('No models were fitted')
    agg = (results.groupby(['model', 'train_n'], as_index=False)
           .agg(MAE_billion_VND=('MAE_VND', lambda s: s.mean() / 1e9),
                RMSE_billion_VND=('RMSE_VND', lambda s: s.mean() / 1e9),
                R2=('R2', 'mean'),
                MAPE_percent=('MAPE_percent', 'mean'),
                repetitions=('repeat', 'count'),
                elapsed_seconds=('fit_and_predict_seconds', 'mean')))
    agg.to_csv(output / 'learning_curve_summary.csv', index=False)
    for metric, ylabel, filename in [
        ('MAE_billion_VND', 'MAE (billion VND)', 'learning_curve_mae.png'),
        ('R2', 'R-squared', 'learning_curve_r2.png'),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for model, part in agg.groupby('model'):
            part = part.sort_values('train_n')
            ax.plot(part['train_n'], part[metric], marker='o', label=model)
        ax.set_xlabel('Number of training examples')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    info = {
        'eligible_before_sampling': int(len(df)),
        'test_n': int(len(test)),
        'development_pool_n': int(len(development)),
        'sizes_requested': sizes,
        'models': model_names,
        'repeats_requested': repeats,
        'seed': seed,
        'holdout_note': 'One fixed randomly selected holdout across ALL sample sizes; separate from the existing train.py split.',
        'subset_note': 'Training subsets are nested within a repetition; each Pipeline fits imputation, one-hot encoder, and scaler on train only.',
        'comparison_caveat': 'Exploratory learning-curve analysis; do not compare this holdout to old outputs/metrics.csv as if it were the same test set.',
        'rows_completed': len(rows),
    }
    (output / 'learning_curve_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Experiment files saved to {output}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw/hcm_listings.parquet')
    parser.add_argument('--out', default='outputs/experiments')
    parser.add_argument('--sizes', nargs='+', type=int, default=[3000, 10000, 30000, 80000])
    parser.add_argument('--models', nargs='+', choices=list(PARAMS), default=['HistGradientBoosting'])
    parser.add_argument('--test-size', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f'Dataset not found: {path}. Set --data to existing local Parquet/CSV. Do not re-download.')
    print('Reading existing local dataset:', path, flush=True)
    raw = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_parquet(path)
    raw_count = len(raw)
    df = clean_listings(raw)
    print(f'Raw rows={raw_count:,}; eligible rows={len(df):,}', flush=True)
    del raw
    run_experiment(df, args.sizes, args.models, args.test_size, args.seed,
                   args.repeats, args.out, args.max_seconds)


if __name__ == '__main__':
    main()
