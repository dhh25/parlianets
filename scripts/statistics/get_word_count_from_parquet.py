from lxml import etree
import os
import sys
import pandas as pd
import polars as pl
import argparse
from glob import glob
from tqdm import tqdm

def process_file(input_file, output_file):
    pl.scan_parquet(input_file
                                 ).select(
        pl.col("id").alias("text_id"),
        pl.col("xml").cast(pl.String).str.count_matches("</w>", literal=True).alias("word_count")
    ).sink_parquet(output_file, compression='zstd')

    # parquet_df["word_count"] = parquet_df["xml"].map(lambda x: str(x).count("</w>"))

    # return parquet_df[["id", "word_count"]]


# the script should result in an identical parquet file with an additional column containing the number of words (<w> elements)
# def main(args):
#     if len(args) != 2:
#         print('Usage\npython get_word_count_from_parquet.py input_file output_file')
#         return
#     [input_file, output_file] = args
#     process_file(input_file, output_file)
#
#     # result_df.to_parquet(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()
    countries = sorted([f for f in os.listdir(args.input_folder) if os.path.isdir(f"{args.input_folder}/{f}")])
    for country in tqdm(countries):
        input_files = sorted(glob(f"{args.input_folder}/{country}/*_texts.parquet"))
        out_folder = f"{args.output_folder}/{country}"
        os.makedirs(out_folder, exist_ok=True)
        for input_file in tqdm(input_files):
            output_file = f"{out_folder}/{os.path.basename(input_file).replace('_texts','_wc')}"
            process_file(input_file, output_file)
    # main(sys.argv[1:])
