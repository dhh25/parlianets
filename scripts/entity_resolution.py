import argparse
import os

from tqdm import tqdm
import re
from io import StringIO
import polars as pl
from glob import glob
import pandas as pd

from scripts.preprocessing_utils import get_entities_of_interest, get_country_name

def get_prefix(x, prefixes, path_to_meta):
    print(x, prefixes[:5])
    for y in prefixes:
        if x.startswith(y):
            return f"{path_to_meta}/{y}-meta.tsv"
    return ""


def fix_tsv_formatting(meta_filename):
    fixed_tsv = str()
    with open(meta_filename, "r", encoding="utf-8") as rf_meta:
        for line in rf_meta:
            fixed_tsv += re.sub(r"\s*\n", "\n", line)

    return StringIO(fixed_tsv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    country_codes = sorted(filter(lambda x:not x.startswith("."), os.listdir(args.in_dir)))
    os.makedirs(args.out_dir, exist_ok=True)
    entities = get_entities_of_interest()
    for iso2_cc in tqdm(country_codes):
        country = get_country_name(iso2_cc)
        path_to_meta = f"../ParlaMint_data/ParlaMint-{iso2_cc}-en/ParlaMint-{iso2_cc}-en.txt"
        meta_files = sorted(glob(path_to_meta + "/**/**-meta.tsv"))
        df = pl.scan_parquet(f"{args.in_dir}/{iso2_cc}/**/*_entities.parquet").filter(
            (pl.col("name_type") == "LOC") & (pl.col("text") != country)
        ).with_columns(pl.col("text").str.to_lowercase()
                       ).filter(pl.col("text").is_in(entities)
        ).collect().to_pandas()

        speaker_meta = pd.concat([pd.read_csv(fix_tsv_formatting(f), sep="\t") for f in meta_files], ignore_index=True)
        speaker_meta["Speaker_birth"] = [x if isinstance(x,int) else None for x in speaker_meta.Speaker_birth]
        df = df.merge(speaker_meta, how="left").rename(columns={"text":"entity", "ID":"text_id"})
        print(iso2_cc, len(df))
        df = pl.from_pandas(df, schema_overrides=dict(Speaker_birth=int))
        df.write_parquet(f"{args.out_dir}/{iso2_cc}_filtered.parquet", compression="zstd")
