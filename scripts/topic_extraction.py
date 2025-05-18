import shutil

import polars as pl
import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import islice
import logging
from pathlib import Path
import os
import country_converter as coco
from multiprocessing import Pool, cpu_count
import argparse

from scripts.preprocessing_utils import get_years_from_filenames, get_entities_of_interest, is_of_interest, \
    to_text_content, xml_from_text_id, process_name_node, get_country_name

os.makedirs("topic_modeling_results/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.StreamHandler(sys.stdout),
        logging.FileHandler("topic_modeling_results/logs/topic_modeling.log", mode='a', encoding='utf-8')
    ]
)


def extract_hierarchical(text_id, position, texts_df, texts_index_df, levels=0, transform_fn=to_text_content):
    unode = xml_from_text_id(text_id, texts_df, texts_index_df)
    return process_name_node(position, unode, levels, transform_fn)


def get_tokenizer(model="xlm-roberta-large"):
    tokenizer = AutoTokenizer.from_pretrained(model)

    return tokenizer


def get_model(model="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model, trust_remote_code=True)

    return model


# run the model on the sentence and context to extract the top 5 topics
def extract_topics(sentence, context, model, tokenizer):

    # Tokenize the input sentence and context
    inputs = tokenizer(sentence,
                       context,
                       return_tensors="pt",
                       max_length=300,  # we limited the input to 300 tokens during finetuning
                       padding="max_length",
                       truncation=True
                       )

    # Get the model's predictions
    logits = model(**inputs).logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    probabilities = {model.config.id2label[index]: round(
        probability * 100, 2) for index, probability in enumerate(probabilities)}
    probabilities = dict(sorted(probabilities.items(),
                         key=lambda item: item[1], reverse=True))

    return dict(islice(probabilities.items(), 5))


def main(target_dir=None, batch_size=100, country=None, year=None, filter_=None):
    # scan parquet files as lazyframes
    logging.info("Loading parquet datasets...")
    if year is None:
        year = ""
    if filter_ is None:
        filter_ = True
    entities_of_interest = get_entities_of_interest()
    lf_entities = pl.scan_parquet(f'{target_dir}/**/{year}*_entities.parquet'
                ).filter(filter_ & (pl.col('text') != country)
                ).with_columns(
                    pl.col("text").str.to_lowercase(),
                    pl.col("ID").str.extract("[1-2][0-9]{3}-[0-9]{2}-[0-9]{2}", group_index=0).alias("date")
                ).filter(pl.col("text").is_in(entities_of_interest))
    lf_texts = pl.scan_parquet(f'{target_dir}/**/{year}*_texts.parquet')

    # create an index to retrieve the xml text from the id (full texts make lf_texts too large to load into memory)
    texts_index = lf_texts.select('id').with_row_index('idx')

    logging.info(f"{country}-{year}: Loading model and tokenizer...")
    model = get_model()
    tokenizer = get_tokenizer()
    logging.info(f"{country}-{year}: Model and tokenizer loaded.")

    num_entities = lf_entities.select(pl.len()).collect().item()
    logging.info(f"{country}-{year}: Processing {num_entities} entities...")

    results = []
    batch_count = 0
    save_every = batch_size
    output_dir = f'./topic_modeling_results/{Path(target_dir).name}'
    os.makedirs(output_dir, exist_ok=True)

    for i in trange(num_entities):
        # get one row from the lazyframe
        entity = lf_entities.slice(i, 1).collect().to_dicts()[0]
        position = entity['position']
        text_id = entity['ID']
        date = entity['date']

        # if not is_of_interest(entity['text'], entities_of_interest):
        #     continue


        # logging.info(f"{country}-{year}: Processing sentence {i}")

        # extract the sentence from the xml text
        sentence = extract_hierarchical(
                text_id, position, lf_texts, texts_index, levels=1)

        logging.info(f"{country}-{year}: Entity: {entity['text']}, {entity['name_type']}")
        # logging.info(f"{country}-{year}: Sentence: {sentence}")
        if not sentence:
            logging.warning(
                    f"{country}-{year}: Skipping entity {i}: could not extract sentence.")
            continue

        # extract the context from the xml text
        context = extract_hierarchical(
                text_id, position, lf_texts, texts_index, levels=2)
        # remove the sentence from the context
        context = context.replace(
                sentence, '') if context != sentence else context

        # logging.info(f"{country}-{year}: Context: {context}")
        topics = extract_topics(sentence, context, model, tokenizer)

        # logging.info(f"{country}-{year}: Topics: {topics}")
        # logging.info(f'{country}-{year}: ----')

        results.append({
                "text_id": text_id,
                "position": position,
                "sentence": sentence,
                "context": context,
                "topics": str(topics),
                "date": date,
                "entity": entity['text']
            })

        if len(results) >= save_every:
                batch_path = f"{output_dir}/{Path(target_dir).name}_{year}_{batch_count}.parquet"
                pl.DataFrame(results).write_parquet(batch_path)
                logging.info(f"{country}-{year}: Saved batch {batch_count} to {batch_path}")
                results.clear()
                batch_count += 1

    if results:
        batch_path = f"{output_dir}/{Path(target_dir).name}_{year}_{batch_count}.parquet"
        pl.DataFrame(results).write_parquet(batch_path)
        logging.info(f"{country}-{year}: Saved final batch {batch_count} to {batch_path}")

    logging.info(f"{country}-{year}: Processing complete.")


def merge_parquets(iso2_cc, out_folder, remove=True):
    in_folder = f"./topic_modeling_results/{iso2_cc}"
    df = pl.read_parquet(in_folder)
    os.makedirs(out_folder, exist_ok=True)
    df.write_parquet(f"{out_folder}/{iso2_cc}_topics.parquet", compression='zstd')
    if remove:
        shutil.rmtree(in_folder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iso2_cc", type=str, help="ISO2 country code for country of interest, e.g., FI")
    parser.add_argument("--preprocessed_dir", type=str, help="Path to preprocessed files (entities + texts), e.g., ../ParlaMint_preprocessed")
    parser.add_argument("--out_dir", type=str, help="Directory to write merged parquet file, e.g., ../ParlaMint_topics")
    parser.add_argument("--batch_size", "-b", type=int, default=100, help="Batch size")
    parser.add_argument("--parallelize", "-p", type=int, default=0,
                        help="Number of cores to parallelize over. 0 means no parallelization; passing a negative number p computes cpu_count() - p.")
    args = parser.parse_args()

    iso2_cc = args.iso2_cc
    preprocessed_dir = f"{args.preprocessed_dir}/{iso2_cc}"
    batch_size = args.batch_size
    country = get_country_name(iso2_cc)


    years = get_years_from_filenames(os.listdir(preprocessed_dir))
    parallelize = cpu_count() - args.parallelize if args.parallelize < 0 else args.parallelize
    parallelize = min(parallelize, len(years)) # restrict to number of years

    # TODO filtering api currently awkward -> refactor
    filter_ = pl.col('name_type') == 'LOC'
    if parallelize != 0:
        with Pool(parallelize) as pool:
            pool.starmap(main, [(preprocessed_dir, batch_size, country, year, filter_) for year in years])
    else:
        for year in years:
            main(preprocessed_dir, batch_size, country, year)

    merge_parquets(iso2_cc, args.out_dir, remove=True)