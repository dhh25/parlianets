import re
import pandas as pd
from lxml import etree
import sys
from extract.py import extract_entities, merge_files 

# Pass meta_filename and tei_filename from pair files to produce data for single session
def process_files(tei_filename, out_filename, meta_filename=None):
    if meta_filename:
        df = merge_files(meta_filename, tei_filename)
    else:
        df = extract_entities(tei_filename)
        df['Text_ID'] = df['Text_ID'].astype(str)

    if out_filename.endswith('.parquet'):
        df.to_parquet(out_filename)
    else:
        df.to_csv(out_filename, index=False)

def main(args):
    if len(args) not in [3, 2]:
        print('Usage\npython extract.py tei_filename meta_filename output_filename\nor\npython extract.py tei_filename output_filename')
        return
    if len(args) == 3:
        [tei_filename, meta_filename, out_filename] = args
        process_files(tei_filename, out_filename, meta_filename=meta_filename)
    elif len(args) == 2:
        [tei_filename, out_filename] = args
        process_files(tei_filename, out_filename)


if __name__ == '__main__':
    main(sys.argv[1:])
