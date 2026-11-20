"""Download original 10 Parquet shards and extract HCMC records only.
Does not redistribute raw dataset. Requires internet and ~2GB free cache space.
"""
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO='vduydong/vietnam-real-estates-2'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=REPO)
    p.add_argument('--output',default='data/raw/hcm_listings.parquet')
    p.add_argument('--cache-dir',default='data/cache')
    a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        print(f'Exists: {out} ({len(pd.read_parquet(out,columns=["price"])):,} rows)');return
    files=[x for x in HfApi().list_repo_files(a.repo,repo_type='dataset') if x.endswith('.parquet')]
    if not files: raise RuntimeError('No original Parquet files found. Inspect repo and use --repo.')
    print('Dataset:',a.repo,'Parquet files:',len(files))
    parts=[]
    for i,name in enumerate(files,1):
        path=hf_hub_download(repo_id=a.repo,filename=name,repo_type='dataset',cache_dir=a.cache_dir)
        table=pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=50000):
            df=batch.to_pandas()
            if 'province_name' not in df: raise ValueError('Missing province_name; wrong source schema')
            mask=df['province_name'].astype('string').str.contains('Hồ Chí Minh|Ho Chi Minh',case=False,na=False,regex=True)
            if mask.any():parts.append(df.loc[mask].copy())
        print(f'[{i}/{len(files)}] {name}: accumulated={sum(len(x) for x in parts):,}')
    if not parts: raise RuntimeError('No HCMC rows found')
    final=pd.concat(parts,ignore_index=True)
    final.to_parquet(out,index=False,compression='zstd')
    print('Saved:',out,'rows:',len(final))
    print('Expected published HCMC count: 1,357,526; investigate if mismatch, never silently override.')

if __name__=='__main__':main()
