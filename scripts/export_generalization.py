"""Export tested generalization audit table and plots into report/generated."""
import argparse
import shutil
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='outputs/experiments/generalization_final')
    p.add_argument('--dest',default='report/generated')
    a=p.parse_args()
    root,dest=Path(a.input),Path(a.dest)
    f=root/'generalization_summary.csv'
    if not f.is_file():
        raise SystemExit('Missing '+str(f))
    data=pd.read_csv(f)
    data=data.loc[data['segment']=='overall'].set_index('split')
    dest.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{Đánh giá khả năng khái quát hóa (trung bình các lần lặp; hai protocol random và group-purged dùng chung tập kiểm thử).}',
           r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Protocol & MAE (tỷ VND) & $R^2$ & MAPE (\%) & Số mẫu test \\',r'\midrule']
    labels={'random':'Ngẫu nhiên','group_purged':'Loại gần trùng theo proxy','temporal':'Theo thời gian'}
    for k in ['random','group_purged','temporal']:
        if k not in data.index: continue
        row=data.loc[k]
        lines.append(f'{labels[k]} & {row.MAE_billion_VND:.3f} & {row.R2:.4f} & {row.MAPE_percent:.2f} & {int(row.test_n):,} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\label{tab:generalization}',r'\end{table}']
    (dest/'generalization_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for pth in root.glob('*.png'):
        shutil.copy2(pth,dest/pth.name)
    print('Exported',dest/'generalization_table.tex')

if __name__=='__main__': main()
