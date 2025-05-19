import re
from itertools import product

import country_converter as coco
import pandas as pd
import polars as pl
from lxml import etree
import numpy as np


def get_years_from_filenames(filenames):
    return sorted({int(re.search(r"[1,2]\d{3}", x).group(0)) for x in filenames if x.endswith("parquet")})


# TODO this is still (very) heuristic -> e.g., OECD, G7, G20 etc. missing -> you need to decide what should go in here
def get_entities_of_interest():
    countries = set(pd.read_csv(coco.COUNTRY_DATA_FILE, sep="\t").name_short.str.lower().values) | {"usa", "united states", "united states of america", "us", "holland"}
    regions = {"europe", "eu", "european union", "nato", "un", "united nations"}
    continents = {"europe", "africa", "america", "asia", "australia", "oceania"}
    directions1 = {"south", "north"}
    directions2 = {"west", "east"}
    directions1alt = {"southern", "northern"}
    directions2alt = {"western", "eastern"}
    directions3 = {" ".join(x) for x in product(directions1, directions2, continents)}
    directions4 = {"-".join(x) for x in product(directions1, directions2, continents)}
    directions5 = {" ".join(x) for x in product(directions1, continents)}
    directions6 = {" ".join(x) for x in product(directions2, continents)}
    directions7 = {" ".join(x) for x in product(directions1alt, continents)}
    directions8 = {"-".join(x) for x in product(directions2alt, continents)}
    return countries | regions | continents | directions3 | directions4 | directions5 | directions6 | directions7 | directions8


def is_of_interest(text, entities_of_interest):
    return text.lower() in entities_of_interest


def to_text_content(node):
    rawtxt = ''.join(node.itertext(r'{http://www.tei-c.org/ns/1.0}w', r'{http://www.tei-c.org/ns/1.0}pc'))
    return re.sub(r'\s+', ' ', rawtxt).strip()


def find_name(position, root):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    xmlspec = r'{http://www.w3.org/XML/1998/namespace}'
    idstring = f'{xmlspec}id'

    return root.find(f'.//*[@{idstring}=\'{position}\']', namespaces=namespaces).getparent()


def xml_from_text_id(text_id, texts_df, texts_index_df):
    idx = texts_index_df.filter(pl.col('id') == text_id).select('idx').collect().item()
    xmlstr = texts_df.slice(idx, 1).select('xml').collect().item().decode()
    return etree.fromstring(xmlstr)


def process_name_node(position, unode, levels=0, transform_fn=to_text_content):
    node = find_name(position, unode)

    while(levels > 0):
        if node.getparent() is None:
            break
        node = node.getparent()
        levels -= 1

    return transform_fn(node)


def get_country_name(iso2_cc):
    return coco.convert(iso2_cc, to="short")


def isword(n):
    return n.tag == r'{http://www.tei-c.org/ns/1.0}w'

def ispunct(n):
    return n.tag == r'{http://www.tei-c.org/ns/1.0}pc' 

# Returns index of first element after width words
def limitwindow(nodel, width):
    if width == 0:
        return 0

    if width > len(nodel):
        return len(nodel)

    cs = np.cumsum(np.array([isword(n) for n in nodel]))
    index = np.argwhere(cs == width)

    if index.shape[0] == 0:
        return len(nodel)

    return int(index[0].item()) + 1


def extract_word_window(position, text_id, texts_df, texts_index_df, width=0, transform_fn=to_text_content):
    unode = xml_from_text_id(text_id, texts_df, texts_index_df)
    namenode = find_name(position, unode)

    # TODO improve
    pre = list() # list of preceding nodes
    post = list() # list of subsequent nodes
    li = pre

    for node in unode.iter(r'{http://www.tei-c.org/ns/1.0}w', r'{http://www.tei-c.org/ns/1.0}pc'):
        if node == namenode[0]:
            li = post

        if node.getparent() == namenode:
            continue

        li.append(node)

    pre = pre[::-1]
    pre_exl = pre[limitwindow(pre, width):]
    post_exl = post[limitwindow(post, width):]

    # Remove unwanted nodes from tree to enable arbitrary node-based transform
    for node in pre_exl:
        node.getparent().remove(node)
    for node in post_exl:
        node.getparent().remove(node)

    return transform_fn(unode)
