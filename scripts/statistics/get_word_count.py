from lxml import etree
import os
import sys
import pandas as pd


def get_indices_n_word_count(filepath):
    text_id = str(os.path.split(filepath)[1].split(".ana.xml")[0])
    xml = r'{http://www.w3.org/XML/1998/namespace}'
    tree = etree.parse(filepath)
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = tree.getroot()

    for usage_tag in root.findall(".//tei:tagUsage", namespaces=namespaces):
        if "gi" in usage_tag.attrib.keys() and usage_tag.attrib["gi"] == "w":
            return text_id, str(usage_tag.attrib["occurs"])
    
    return text_id, None



def count_words(input_dir):
    wc_dict = dict()
    
    for format_folder in os.listdir(input_dir): 
        if format_folder.endswith(".TEI.ana"):
            for year_folder in os.listdir(os.path.join(input_dir, format_folder)):
                if not os.path.isdir(os.path.join(input_dir, format_folder, year_folder)) or not year_folder.isnumeric():
                    continue
                
                for xml_file in os.listdir(os.path.join(input_dir, format_folder, year_folder)):
                    if xml_file.endswith(".ana.xml"):
                        index, word_count = get_indices_n_word_count(os.path.join(input_dir, format_folder, year_folder, xml_file))
                        wc_dict[index] = word_count
        
    return pd.Series(data=wc_dict)



# the script should result in one series covering the number of words (<w> elements) in a parliamentary session for one parliament
def main(args):
    if len(args) != 2 or not os.path.isdir(args[0]):
        print('Usage\npython get_word_count.py input_directory output_file')
        return
    [input_dir, output_file] = args
    result_series = count_words(input_dir)

    result_series.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    main(sys.argv[1:])
