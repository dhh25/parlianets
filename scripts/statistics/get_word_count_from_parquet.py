from lxml import etree
import os
import sys
import pandas as pd


def process_file(input_file):
    parquet_df = pd.read_parquet(input_file)

    parquet_df["word_count"] = parquet_df["xml"].map(lambda x: str(x).count("</w>"))

    return parquet_df[["id", "word_count"]]


# the script should result in an identical parquet file with an additional column containing the number of words (<w> elements)
def main(args):
    if len(args) != 2:
        print('Usage\npython get_word_count_from_parquet.py input_file output_file')
        return
    [input_file, output_file] = args
    result_df = process_file(input_file)

    result_df.to_parquet(output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
