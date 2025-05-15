import os
import sys
from tqdm import tqdm
import argparse
from glob import glob
import polars as pl
import shutil

from lookup_table import find_session_files, create_temp_directory, extract_data
from extract import extract_entities

def generate_entity_data(root, iso_cc):
    extract_dir = create_temp_directory(iso_cc)
    files = find_session_files(root)
    for file in tqdm(files):
        df = extract_entities(file)
        out_filename = f'{extract_dir}/{os.path.basename(file)}.parquet'
        df.to_parquet(out_filename, compression='zstd')
    return extract_dir

def preprocess(root, iso_cc):
    lookup_dir = extract_data(root, iso_cc)
    extract_dir = generate_entity_data(root, iso_cc)

    return (lookup_dir, extract_dir)

# def main(args):
#     if len(args) != 2:
#         print('Usage\npython preprocess.py ParlaMint-CC.TEI.ana/ ISO_CC')
#         return
#     [root, iso_cc] = args
#     (lookup_dir, extract_dir) = preprocess(root, iso_cc)
#     print(f'Output\nLookup\t{lookup_dir}\nNamEnt\t{extract_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tei_ana_folder', type=str)
    parser.add_argument('out_folder', type=str)
    args = parser.parse_args()
    root = args.tei_ana_folder
    iso_cc = root.split("ParlaMint-")[-1].split("-en")[0]
    years = sorted([x for x in os.listdir(root) if len(x) == 4])
    lookup_dir, extract_dir = preprocess(root, iso_cc)
    out_folder = f"{args.out_folder}/{iso_cc}"
    os.makedirs(out_folder, exist_ok=True)
    for year in years:
        lookup_files = glob(lookup_dir + "/*" + year + "*.parquet")
        df = pl.read_parquet(lookup_files)
        df.write_parquet(f"{out_folder}/{year}_texts.parquet", compression='zstd')
        extract_files = glob(extract_dir + "/*" + year + "*.parquet")
        df = pl.read_parquet(extract_files)
        df.write_parquet(f"{out_folder}/{year}_entities.parquet", compression='zstd')
    shutil.rmtree(lookup_dir)
    shutil.rmtree(extract_dir)

    # main(sys.argv[1:])
