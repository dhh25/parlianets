import polars as pl
from lxml import etree
import re
import numpy as np

def find_name(position, root):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    xmlspec = r'{http://www.w3.org/XML/1998/namespace}'
    idstring = f'{xmlspec}id'

    return root.find(f'.//*[@{idstring}=\'{position}\']', namespaces=namespaces).getparent()

def to_text_content(node):
    rawtxt = ''.join(node.itertext(r'{http://www.tei-c.org/ns/1.0}w', r'{http://www.tei-c.org/ns/1.0}pc'))
    return re.sub(r'\s+', ' ', rawtxt).strip()

def to_lemmas(node):
    content = []
    for n in node.iter(r'{http://www.tei-c.org/ns/1.0}w', r'{http://www.tei-c.org/ns/1.0}pc'):
        if isword(n):
            content.append(n.attrib['lemma'].strip())
        if ispunct(n):
            content.append(n.text.strip())

    return ' '.join(content)

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

def extract_word_window(position, text_id, texts_df, width=0, transform_fn=to_text_content):
    unode = xml_from_text_id(text_id, texts_df)
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

def extract_paragraph_window(position, text_id, texts_df, pgraph=0, transform_fn=to_text_content):
    unode = xml_from_text_id(text_id, texts_df)
    namenode = find_name(position, unode)
    segment = namenode.getparent().getparent()

    children = list(unode.getchildren())
    segment_index = -1

    for i, child in enumerate(children):
        if child == segment:
            segment_index = i
            break

    min_id = max(0, segment_index - pgraph)
    max_id = min(len(children), segment_index + pgraph)

    for i, child in enumerate(children):
        if i < min_id or i > max_id:
            unode.remove(child)

    return transform_fn(unode)
