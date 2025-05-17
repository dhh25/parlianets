import polars as pl
from lxml import etree
import re
import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from itertools import islice
import logging
from pathlib import Path
import os
import country_converter as coco
import pandas as pd
from multiprocessing import Pool, cpu_count

os.makedirs("topic_modeling_results/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.StreamHandler(sys.stdout),
        logging.FileHandler("topic_modeling_results/logs/topic_modeling.log", mode='a', encoding='utf-8')
    ]
)


def find_name(position, root):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    xmlspec = r'{http://www.w3.org/XML/1998/namespace}'
    idstring = f'{xmlspec}id'

    return root.find(f'.//*[@{idstring}=\'{position}\']', namespaces=namespaces).getparent()

def to_text_content(node):
    rawtxt = ''.join(node.itertext())
    return re.sub(r'\s+', ' ', rawtxt).strip()

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
    lf_entities = pl.scan_parquet(f'{target_dir}/**/{year}*_entities.parquet').filter(filter_ & (pl.col('text') != country))
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

        if not is_country(entity['text']):
            continue


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
                "topics": str(topics)

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

# TODO proper file for complete list of non-country entities of interest
COUNTRIES = set(pd.read_csv(coco.COUNTRY_DATA_FILE, sep="\t").name_short.str.lower().values) | {"europe", "eu", "european union", "nato", "un", "United Nations"}

def is_country(text):
    return text.lower() in COUNTRIES


if __name__ == "__main__":
    country = "Finland"
    target_dir = f'../ParlaMint_preprocessed/{coco.convert(country, to="iso2")}'
    years = sorted({int(x[:4]) for x in os.listdir(target_dir)})
    batch_size = 100
    filter_ = pl.col('name_type') == 'LOC'
    with Pool(cpu_count() - 4) as pool:
        pool.starmap(main, [(target_dir, batch_size, country, year, filter_) for year in years])