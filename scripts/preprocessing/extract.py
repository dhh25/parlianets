import re
import pandas as pd
from lxml import etree
import sys
from pathlib import Path

def extract_entities(filename):
    xml = r'{http://www.w3.org/XML/1998/namespace}'
    tree = etree.parse(filename)
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = tree.getroot()
    cols=['Text_ID', 'name_type', 'position', 'text']
    df = pd.DataFrame(columns=cols)

    ids = list()
    name_types = list()
    positions = list()
    texts = list()

    for speech in root.findall('.//tei:u', namespaces=namespaces):
        id = speech.attrib[xml+'id']
        for sententece in speech.findall('.//tei:s', namespaces=namespaces):
            names = list(sententece.findall('.//tei:name', namespaces=namespaces))
            # Todo extract lemmatized version
            for name in names:
                fragment_id = name.find('.//tei:w', namespaces=namespaces).attrib[xml+'id']
                text = name.xpath('string()')
                text = re.sub(r'\n', ' ', text)
                ids.append(id)
                name_types.append(name.attrib['type'])
                positions.append(fragment_id)
                texts.append(text)
    df['Text_ID'] = ids
    df['name_type'] = name_types
    df['position'] = positions
    df['text'] = texts
    
    return df

def merge_files(meta_filename, tei_filename):
    out = extract_entities(tei_filename)
    out['Text_ID'] = out['Text_ID'].astype(str)

    table = pd.read_csv(meta_filename, sep='\t')
    table['Text_ID'] = table['Text_ID'].astype(str)

    merged = table.merge(out, left_on='Text_ID', right_on='Text_ID', how='inner')

    return merged

# Pass meta_filename and tei_filename from pair files to produce data for single session
def process_files(meta_filename, tei_filename, out_filename):
    df = merge_files(meta_filename, tei_filename)

    if out_filename.endswith('.parquet'):
        df.to_parquet(out_filename)
    else:
        df.to_csv(out_filename, index=False)

def main(args):
    if len(args) != 3:
        print('Usage\npython extract.py tei_filename meta_filename output_filename')
        return
    [tei_filename, meta_filename, out_filename] = args
    process_files(meta_filename, tei_filename, out_filename)

if __name__ == '__main__':
    main(sys.argv[1:])
