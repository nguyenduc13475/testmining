"""Export actual missing-value ablation results as a LaTeX table + images."""
from pathlib import Path
import argparse
import shutil
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/experiments/missing_ablation')
    parser.add_argument('--dest', default='report/generated')
    args = parser.parse_args()
    inp, dest = Path(args.input), Path(args.dest)
    src = inp / 'missing_ablation_summary.csv'
    if not src.exists():
        raise SystemExit(f'Missing {src}; run missing_ablation.py first.')
    data = pd.read_csv(src)
    dest.mkdir(parents=True, exist_ok=True)
    labels = {'baseline': 'Giữ tất cả', 'drop_high': 'Bỏ cột thiếu nhiều',
              'indicators': 'Thêm chỉ báo thiếu'}
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Ablation dữ liệu khuyết thiếu (số liệu trung bình trên các lần lặp).}',
             r'\begin{tabular}{lrrrr}',r'\toprule',
             r'Chiến lược & MAE (tỷ VND) & Độ lệch chuẩn & $R^2$ & MAPE (\%) \\',
             r'\midrule']
    for _, row in data.iterrows():
        label = labels.get(row['strategy'], row['strategy']).replace('_', r'\_')
        lines.append(f"{label} & {row['MAE_billion_VND']:.3f} & {row['MAE_std_billion_VND']:.3f} & {row['R2']:.4f} & {row['MAPE_percent']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\label{tab:missing-ablation}', r'\end{table}']
    (dest / 'missing_ablation_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for f in ['missing_ablation_mae.png', 'missing_ablation_r2.png']:
        if (inp/f).exists():
            shutil.copy2(inp/f, dest/f)
    print('Exported', dest/'missing_ablation_table.tex')

if __name__ == '__main__':
    main()
