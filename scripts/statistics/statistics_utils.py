import json
import os
from io import StringIO
import re

# build country code dictionaries. Returns two dictionaries, one indexed by common name, one by country capital
def build_country_code_dicts():
    cn_dict = dict()
    cap_dict = dict()
    curr_dir = os.getcwd()
    with open(os.path.join(curr_dir, "restcountries_all.json"), "r", encoding="utf-8") as json_file:
        json_all = json.load(json_file)
    
    for country in json_all:
        cn_dict[country["name"]["common"].lower()] = country["cca2"]
        if "capital" in country.keys():
            for capital in country["capital"]:
                cap_dict[capital.lower()] = country["cca2"]
    
    return cn_dict, cap_dict


# converts a given named entity to an ISO 3166-1 alpha-2 country code. 
# expects a named entity and two country codes dictionary (one indexed by common name and one by country capital)
# will return a country code only if given a country name or name of country capital. 
# returns None otherwise
def ne2cc(ne, cc_dict, cap_dict):
    ne_lower = ne.lower()
    if ne_lower in cc_dict.keys():
        return cc_dict[ne_lower]
    elif ne_lower in cap_dict.keys():
        return cap_dict[ne_lower]
    else:
        return None


def fix_tsv_formatting(meta_filepath):
    fixed_tsv = str()
    with open(meta_filepath, "r", encoding="utf-8") as rf_meta:
        for line in rf_meta:
            fixed_tsv += re.sub(r"\s*\n", "\n", line)
    
    return StringIO(fixed_tsv)
