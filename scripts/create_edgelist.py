import argparse
import os

from tqdm import tqdm
import re
from io import StringIO
import polars as pl
from glob import glob
import pandas as pd
import country_converter as coco




if __name__ == '__main__':
    out_dir = '../ParlaMint_edgelists'
    os.makedirs(out_dir, exist_ok=True)
    df = pl.read_csv(coco.COUNTRY_DATA_FILE, separator="\t").select([pl.col("name_short"), pl.col("ISO2")])
    mapping = {k.lower():v for k,v in zip(df.get_column("name_short"),df.get_column("ISO2"))} | {"us": "US", "usa": "US", "holland": "NL"}
    mapping["greece"] = "GR"
    for f in sorted(glob("../ParlaMint_entities_filtered/**.parquet")):
        iso2_prelim = f.split("/")[-1].split("_")[0]
        iso2_cc = iso2_prelim[:2]
        pl.scan_parquet(f).with_columns(
            pl.col("entity").map_elements(lambda x: mapping.get(x,x), return_dtype=pl.String).alias("target_country"),
            pl.lit(iso2_cc, dtype=pl.String).alias("source_country"),
        ).sink_parquet(f"{out_dir}/{iso2_prelim}_edges.parquet", compression="zstd")
