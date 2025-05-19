import shutil

import polars as pl
import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import islice
import logging
from pathlib import Path
import os
from multiprocessing import Pool, cpu_count
import argparse
import sys

from topic_extraction import get_model, get_tokenizer, extract_topics
from preprocessing.entity_context import extract_hierarchical_nodf, extract_word_window_nodf
from preprocessing_utils import get_years_from_filenames

def xml_from_text_id_lazy(text_id, texts_df, texts_index_df):
    idx = texts_index_df.filter(pl.col('id') == text_id).select('idx').collect().item()
    return texts_df.slice(idx, 1).select('xml').collect().item().decode()

def main(target_dir=None, scratch=None, filtered_dir=None, texts_file=None, batch_size=100, context=None, context_length=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    texts_file_filename = os.path.basename(texts_file)
    iso2_cc=os.path.basename(os.path.dirname(texts_file))
    year=texts_file_filename[:4]

    # scan parquet files as lazyframes
    logging.info("Loading parquet datasets...")
    year_str = str(year)
    lf_entities = pl.scan_parquet(f'{filtered_dir}/{iso2_cc.upper()}_filtered.parquet').select([pl.col("entity"), pl.col("position"), pl.col("text_id"), pl.col("name_type"), pl.col("Date")]).filter(pl.col("Date").str.starts_with(year_str))
    lf_texts = pl.scan_parquet(texts_file)

    # create an index to retrieve the xml text from the id (full texts make lf_texts too large to load into memory)
    texts_index = lf_texts.select('id').with_row_index('idx')

    logging.info(f"{iso2_cc}-{year}: Loading model and tokenizer...")
    model = get_model()
    tokenizer = get_tokenizer()
    logging.info(f"{iso2_cc}-{year}: Model and tokenizer loaded.")

    num_entities = lf_entities.select(pl.len()).collect().item()
    logging.info(f"{iso2_cc}-{year}: Processing {num_entities} entities...")

    results = []
    batch_count = 0
    save_every = batch_size
    output_dir = f'{scratch}/{Path(target_dir).name}'
    os.makedirs(output_dir, exist_ok=True)

    for i in trange(num_entities):
        # get one row from the lazyframe
        entity = lf_entities.slice(i, 1).collect().to_dicts()[0]
        position = entity['position']
        text_id = entity['text_id']

        xmlstr = xml_from_text_id_lazy(text_id, lf_texts, texts_index)

        # extract the sentence from the xml text
        sentence = extract_hierarchical_nodf(position, xmlstr, levels=1)

        logging.info(f"{iso2_cc}-{year}: Entity: {entity['entity']}, {entity['name_type']}")

        if not sentence:
            logging.warning(
                    f"{iso2_cc}-{year}: Skipping entity {i}: could not extract sentence.")
            continue

        # extract the context from the xml text
        if context == 'segment':
            context = extract_hierarchical_nodf(position, xmlstr, levels=2)
        elif context == 'words':
            context = extract_word_window_nodf(position, xmlstr, width=context_length)
        topics = extract_topics(sentence, context, model, tokenizer)

        results.append({
                "text_id": text_id,
                "position": position,
                "entity": entity['entity'],
                "sentence": sentence,
                "context": context,
                "topics": str(topics)
            })

        if len(results) >= save_every:
                batch_path = f"{output_dir}/{Path(target_dir).name}_{year}_{batch_count}.parquet"
                pl.DataFrame(results).write_parquet(batch_path)
                logging.info(f"{iso2_cc}-{year}: Saved batch {batch_count} to {batch_path}")
                results.clear()
                batch_count += 1

    if results:
        batch_path = f"{output_dir}/{Path(target_dir).name}_{year}_{batch_count}.parquet"
        pl.DataFrame(results).write_parquet(batch_path)
        logging.info(f"{iso2_cc}-{year}: Saved final batch {batch_count} to {batch_path}")

    logging.info(f"{iso2_cc}-{year}: Processing complete.")

#target_dir=None, scratch=None, entities_file=None, texts_file=None,
#batch_size=100, iso2_cc=None, year="", context=None, context_length=None):

# if __name__ == "__main__":
#     main(
#         target_dir='./test/target',
#         scratch='./test/scratch',
#         filtered_dir='./data', # directory containing CC_filtered.parquet files
#         texts_file='./data/FI/2021_texts.parquet', # text file, iso2cc and year inferred from this
#         batch_size=100,
#         context='segment'
#     )