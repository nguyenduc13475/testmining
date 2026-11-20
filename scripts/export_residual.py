"""Export results of residual audit to LaTeX tables and report/generated/ images."""
import argparse
import shutil
from pathlib import Path
import pandas as pd

def escape(s):
    return str(s).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')

def table(df, cols, labels, caption):
    out=['\\begin{table}[htbp]','\\centering','\\small','\\begin{tabular}{'+'l'+'r'*(len(cols)-1)+'}', '\\hline', ' & '.join(labels)+r' \\', '\\hline']
    for _,r in df.iterrows():
        vals=[]
        for c in cols:
            val=r[c]
            if isinstance(val,(int,float)) or hasattr(val,'item') and isinstance(val.item(),(int,float)):
                vals.append(f'{val:.3f}')
            else: vals.append(escape(val))
        out.append(' & '.join(vals)+r' \\')
    out.extend(['\\hline','\\end{tabular}','\\caption{'+caption+'}','\\end{table}'])
    return '\n'.join(out)+'\n'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='outputs/experiments/residual_final')
    p.add_argument('--output',default='report/generated')
    a=p.parse_args();src=Path(a.input);dst=Path(a.output);dst.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(src/'residual_summary.csv')
    part=s[(s.split=='group_purged')&(s.segment=='actual_price_band')].copy()
    if part.empty:part=s[(s.split=='random')&(s.segment=='actual_price_band')].copy()
    text=table(part,['group','n','MAE_billion_VND','MAPE_percent','signed_bias_billion_VND','underpredict_pct'],
               ['Giá thực','Số mẫu','MAE (tỷ)','MAPE (\\%)','Bias (tỷ)','Dự báo thấp (\\%)'],
               'Sai số theo nhóm giá thực tế; giá thực chỉ dùng để phân tích ngoại tuyến.')
    (dst/'residual_price_table.tex').write_text(text,encoding='utf-8')
    t=pd.read_csv(src/'residual_tail_summary.csv')
    t=t[t.split=='group_purged'] if (t.split=='group_purged').any() else t[t.split=='random']
    (dst/'residual_tail_table.tex').write_text(table(t,['top_percent','absolute_error_share_percent'],['Top mẫu sai (\\%)','Tỷ trọng tổng lỗi (\\%)'],
                          'Tỷ trọng tổng sai số tuyệt đối do các trường hợp sai lớn nhất.'),encoding='utf-8')
    for file in ['signed_bias_by_band.png','mae_by_band.png','mape_by_band.png','actual_vs_predicted_by_band.png']:
        if (src/file).exists():shutil.copy2(src/file,dst/file)
    print('Exported LaTeX tables and 4 figures to',dst)
    print('In report/main.tex, include tables using \\input{generated/residual_price_table.tex} etc.')
if __name__=='__main__':main()
