import re
import pandas as pd
from lxml import etree
import sys
from io import StringIO
from pathlib import Path


def fix_tsv_formatting(meta_filename):
    fixed_tsv = str()
    with open(meta_filename, "r", encoding="utf-8") as rf_meta:
        for line in rf_meta:
            fixed_tsv += re.sub(r"\s*\n", "\n", line)
    
    return StringIO(fixed_tsv)


def extract_entities(filename):
    xml = r'{http://www.w3.org/XML/1998/namespace}'
    tree = etree.parse(filename)
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = tree.getroot()
    cols=["ID", 'name_type', 'position', 'text']
    df = pd.DataFrame(columns=cols)

    ids = list()
    #session_ids = list()
    name_types = list()
    positions = list()
    texts = list()

    for speech in root.findall('.//tei:u', namespaces=namespaces):
        id = speech.attrib[xml+'id']
        #session_id = filename.split(".ana.xml")[0]
        for sententece in speech.findall('.//tei:s', namespaces=namespaces):
            names = list(sententece.findall('.//tei:name', namespaces=namespaces))
            # Todo extract lemmatized version
            for name in names:
                fragment_id = name.find('.//tei:w', namespaces=namespaces).attrib[xml+'id']
                text = name.xpath('string()')
                text = re.sub(r'\n', ' ', text)
                ids.append(id)
                #session_ids.append(session_id)
                name_types.append(name.attrib['type'])
                positions.append(fragment_id)
                texts.append(text)
    df['ID'] = ids
    #df["Text_ID"] = session_ids
    df['name_type'] = name_types
    df['position'] = positions
    df['text'] = texts
    
    return df

def merge_files(meta_filename, tei_filename):
    pd.set_option('display.max_column', None)

    out = extract_entities(tei_filename)
    out['ID'] = out['ID'].astype(str)
    print(out.head())

    table = pd.read_csv(fix_tsv_formatting(meta_filename), sep='\t')
    table['ID'] = table['ID'].astype(str)
    print(table.head())

    merged = table.merge(out, left_on='ID', right_on='ID', how='inner')

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
