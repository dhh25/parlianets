import glob
import re
import sys
from lxml import etree
import pandas as pd
import random
from tqdm import tqdm
import os
import string
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='lookup_table_exceptions.log', level=logging.INFO)

def process_file(filename):
    xml = r'{http://www.w3.org/XML/1998/namespace}'
    tree = etree.parse(filename)
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = tree.getroot()
    ids = list()
    strings = list()
    for tag_u in root.findall('.//tei:u', namespaces=namespaces):
        names = list(tag_u.findall('.//tei:name', namespaces=namespaces))
        if len(names) == 0:
            continue
        id = tag_u.attrib[f'{xml}id']
        text = etree.tostring(tag_u, encoding='utf-8')
        ids.append(id)
        strings.append(text)

    return pd.DataFrame({'id':ids, 'xml':strings})

def extract_data(root, cc):
    while True:
        try:
            dir = f'temp/lookup/{''.join(random.choices(string.ascii_lowercase, k=16))}'
            os.makedirs(dir, exist_ok=False)
            break
        except:
            pass

    for file in tqdm(glob.glob(f'{root}/**/*.ana.xml', recursive=True)):
        match = re.search(r'\d{4}-\d{2}-\d{2}', file)

        if not match:
            continue

        out_filename = f'{dir}/{os.path.basename(file)}.parquet'

        date = match.group()
        try:
            data = process_file(file)
        except exception as e:
            logger.warning(f"the following exception occurred during processing of file {file}: {e}")
            
        data['date'] = date
        data['iso_cc'] = cc
        data['date'] = pd.to_datetime(data['date']).dt.date
        data.to_parquet(out_filename, compression='zstd')
    return dir

def main(args):
    if len(args) == 0:
        print('Usage\npython lookup_table.py corpus_directory country_code')
    dir = args[0]
    cc = args[1]
    out = extract_data(dir, cc)
    print(f'Wrote data to {out}')

if __name__ == '__main__':
    main(sys.argv[1:])
