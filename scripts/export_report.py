"""Generate LaTeX table and insert real experiment figures, never fabricated metrics."""
from pathlib import Path
import pandas as pd
p=Path('outputs/metrics.csv');o=Path('report/generated');o.mkdir(exist_ok=True,parents=True)
if not p.exists():raise SystemExit('Run training before exporting actual metrics.')
df=pd.read_csv(p)
def fmt(v):return f'{v:,.3f}'
lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Kết quả thực nghiệm trên tập kiểm thử độc lập.}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Mô hình & MAE (tỷ) & RMSE (tỷ) & $R^2$ & MAPE (\%) \\',r'\midrule']
for _,r in df.iterrows():lines.append(f"{r['model']} & {fmt(r['MAE_VND']/1e9)} & {fmt(r['RMSE_VND']/1e9)} & {fmt(r['R2'])} & {fmt(r['MAPE_percent'])} \\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\label{tab:results}',r'\end{table}']
(o/'metrics_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
for source in ['actual_vs_pred.png','price_distribution.png']:
 f=Path('outputs')/source
 if f.exists():
  import shutil
  shutil.copy2(f,o/source)
print('Wrote',o/'metrics_table.tex')

# Optional learning-curve study; do not fabricate numbers when missing.
exp = Path('outputs/experiments')
curve = exp / 'learning_curve_summary.csv'
if curve.exists():
    import shutil
    study = pd.read_csv(curve)
    for img in ('learning_curve_mae.png', 'learning_curve_r2.png'):
        if (exp / img).exists():
            shutil.copy2(exp / img, o / img)
    safe = study.sort_values(['model','train_n'])
    result = [r'\begin{table}[ht]',r'\centering\small',
              r'\caption{Ảnh hưởng của kích thước tập huấn luyện trên cùng một tập kiểm thử cố định.}',
              r'\begin{tabular}{lrrrr}',r'\toprule',
              r'Mô hình & Số mẫu train & MAE (tỷ VND) & $R^2$ & Số lần chạy \\',r'\midrule']
    for _, row in safe.iterrows():
        result.append(f"{row['model']} & {int(row['train_n']):,} & {row['MAE_billion_VND']:.3f} & {row['R2']:.4f} & {int(row['repetitions'])} " + r"\\")
    result += [r'\bottomrule',r'\end{tabular}',r'\label{tab:learning-curve}',r'\end{table}']
    (o / 'learning_curve_table.tex').write_text('\n'.join(result)+'\n', encoding='utf-8')
    print('Wrote', o / 'learning_curve_table.tex')
