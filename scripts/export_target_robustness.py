"""Export experiment 5 aggregate results to LaTeX, without modifying main.tex."""
import argparse
import shutil
from pathlib import Path
import pandas as pd


def tex_escape(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='outputs/experiments/target_robustness_final')
    p.add_argument('--report-dir',default='report/generated')
    args=p.parse_args()
    src=Path(args.input); dest=Path(args.report_dir)
    dest.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(src/'robustness_summary.csv')
    overall=data[data['group']=='ALL'].copy()
    rows=[r'\begin{tabular}{llrrrr}',r'\hline',
          r'Method & Split & MAE (billion VND) & $R^2$ & MAPE (\%) & Bias (billion VND) \\',r'\hline']
    for _,r in overall.iterrows():
        rows.append(f"{tex_escape(r['candidate'])} & {tex_escape(r['split'])} & {r['MAE_billion_VND']:.3f} & {r['R2']:.3f} & {r['MAPE_percent']:.2f} & {r['bias_billion_VND']:+.3f} "+r'\\')
    rows += [r'\hline',r'\end{tabular}']
    (dest/'target_robustness_table.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    for name in ('robustness_mae.png','robustness_r2.png','robustness_bias.png'):
        if (src/name).exists(): shutil.copyfile(src/name,dest/name)
    print('Created',dest/'target_robustness_table.tex')
    print('Add to report/main.tex manually when reviewing your own results.')

if __name__=='__main__':main()
