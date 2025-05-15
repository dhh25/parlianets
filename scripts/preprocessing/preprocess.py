import os
import sys
from tqdm import tqdm

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

def main(args):
    if len(args) != 2:
        print('Usage\npython preprocess.py ParlaMint-CC.TEI.ana/ ISO_CC')
        return
    [root, iso_cc] = args
    (lookup_dir, extract_dir) = preprocess(root, iso_cc)
    print(f'Output\nLookup\t{lookup_dir}\nNamEnt\t{extract_dir}')

if __name__ == '__main__':
    main(sys.argv[1:])
