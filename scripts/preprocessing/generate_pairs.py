# Generate csv with TEI meta file pairs to iterate over by running extract.py
# for each output row

import sys
import pandas as pd
from pathlib import Path

def extract_sessions_endswith(root, endswith):
    ret = list()
    for year in root.iterdir():
        if not year.is_dir(): continue
        for file in year.iterdir():
            if str(file).endswith(endswith):
                ret.append(file)

    return ret

def pair_files(path, out):
    ret = pd.DataFrame(columns=['tsv_filename', 'tei_filename'])
    meta_files = list()
    tei_files = list()
    for child in path.iterdir():
        name = str(child)
        if name.endswith(r'.txt'):
            meta_files = meta_files + extract_sessions_endswith(child, 'meta.tsv')
        if name.endswith(r'TEI.ana'):
            tei_files = tei_files + extract_sessions_endswith(child, 'ana.xml')

    meta_files = sorted(meta_files)
    tei_files = sorted(tei_files)

    ret['tsv_filename'] = meta_files
    ret['tei_filename'] = tei_files

    ret.to_csv(out, index=False)

def main(args):
    if len(args) == 0:
        print('Usage\npython generate_pairs.py corpus-CC.ana/ output.csv')
        return
    [path, out] = args
    pair_files(Path(path), out)

if __name__ == '__main__':
    main(sys.argv[1:])