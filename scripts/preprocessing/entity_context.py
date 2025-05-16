import polars as pl
from lxml import etree
import re

def find_name(position, root):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    xmlspec = r'{http://www.w3.org/XML/1998/namespace}'
    idstring = f'{xmlspec}id'

    return root.find(f'.//*[@{idstring}=\'{position}\']', namespaces=namespaces).getparent()

def to_text_content(node):
    rawtxt = ''.join(node.itertext())
    return re.sub(r'\s+', ' ', rawtxt).strip()

def xml_from_text_id(text_id, texts_df):
    xmlstr = texts_df.filter(pl.col('id') == text_id)['xml'][0].decode()
    return etree.fromstring(xmlstr)

def process_name_node(position, unode, levels=0, transform_fn=to_text_content):
    node = find_name(position, unode)

    while(levels > 0):
        if node.getparent() is None:
            break
        node = node.getparent()
        levels -= 1
    
    return transform_fn(node)

def extract_hierarchical(position, text_id, texts_df, levels=0, transform_fn=to_text_content):
    unode = xml_from_text_id(text_id, texts_df)
    return process_name_node(position, unode, levels, transform_fn)