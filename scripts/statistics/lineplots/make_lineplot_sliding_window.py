import os

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import date


def get_full_country_name(country_code):
    return {"AT": "Austria", "BA": "Bosnia and Herzegovina", "BE": "Belgium", "BG": "Bulgaria", 
            "CZ": "Czechia", 
            "EE": "Estonia", 
            "ES": "Spain", 
            "FI": "Finland", 
            "FR": "France", 
            "GR": "Greece", 
            "HR": "Croatia", 
            "HU": "Hungary", 
            "IS": "Iceland", 
            "IT": "Italy", 
            "LV": "Latvia", 
            "NL": "Netherlands", 
            "NO": "Norway", 
            "PL": "Poland", 
            "PT": "Portugal", 
            "RS": "Serbia", 
            "SI": "Slovenia", 
            "TR": "Turkey"}[country_code]


def get_default_countries():
    # returns all the country codes for countries in the current version of ParlaMint. 
    # I am skipping DK, and SE at the moment, since the dates in the IDs are malformed. Also GB and UA, 
    # as they seem to never mention Ukraine
    return ",".join(["AT", "BA", "BE", "BG", "CZ", "EE", "ES", "FI", "FR", 
                     "GR", "HR", "HU", "IS", "IT", "LV", "NL", "NO", "PL", 
                     "PT", "RS", "SI", "TR"])


def merge_dates(a, b):
    return f"{a}-{b}"


def calculate_rel_freq(abs_freq, corp_size, norm_level):
    return (abs_freq / corp_size) * norm_level


def build_wordcount_df(wc_folder, source_countries, start_date, end_date):
    result_df = pl.DataFrame()
    for country in os.listdir(wc_folder):
        if country in source_countries:
            for file in os.listdir(os.path.join(wc_folder, country)):
                curr_df = pl.read_parquet(os.path.join(wc_folder, country, file)).with_columns(pl.col("text_id").str.extract(r".*(\d{4}-\d{2}-\d{2}).*", 1).str.strptime(pl.Date, format="%Y-%m-%d"
                                                       ).alias("Date_formatted"), pl.lit(country).alias("country"))
                result_df = pl.concat([result_df, curr_df])
    
    result_df = result_df.filter(pl.col("Date_formatted").is_between(start_date, end_date)).sort("Date_formatted")
    #print(result_df.filter(pl.col("country") == "DK"))
    
    return result_df


def get_window_corpus_size(lower_boundary, upper_boundary, wordcount_df, cc):
    return wordcount_df.filter(pl.col("country") == cc, pl.col("Date_formatted"
                                      ).is_between(lower_boundary, upper_boundary, closed="left")).select(pl.sum("word_count"
                                      )).item()


def create_df_n_visualize(wc_folder, el_folder, target_country, source_countries, start_date, end_date, window_size, dataframe_filename, plot_filename):
    final_window_df = pl.DataFrame()

    start_date = date(int(start_date.split("-")[0]), int(start_date.split("-")[1]), int(start_date.split("-")[2]))
    end_date = date(int(end_date.split("-")[0]), int(end_date.split("-")[1]), int(end_date.split("-")[2]))
    wordcount_df = build_wordcount_df(wc_folder, source_countries, start_date, end_date)

    for country in os.listdir(el_folder):
        # for now, I am not considering smaller political entities (like Catalonia, Gallicia, etc.)
        if "not found" in country or "-" in country:
            continue
        
        # build the dataframe with mention counts for each window
        country_code = country.split("_")[0]
        if country_code not in source_countries:
            continue

        edges_df = pl.read_parquet(os.path.join(el_folder, country))
        if country_code == "UA":
            print(edges_df.filter(pl.col("target_country") == target_country))
        edges_df = edges_df.with_columns(pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date_formatted")
                                         ).filter(pl.col("target_country") == target_country, pl.col("Date_formatted").is_between(start_date, end_date)).sort("Date_formatted"
                                         ).group_by_dynamic("Date_formatted", include_boundaries=True, every=f"{window_size}d", 
                                         closed="left").agg(pl.len().alias("mentions_in_window"))
        
        # add window timespan names
        try:
            edges_df = edges_df.with_columns(pl.concat_str([pl.col("_lower_boundary").cast(pl.String).str.extract(r".*(\d{4}-\d{2}-\d{2}).*", 1), 
                                            pl.col("_upper_boundary").cast(pl.String).str.extract(r".*(\d{4}-\d{2}-\d{2}).*", 1)], separator=" - "
                                            ).alias("window_timespan"))
        except:
            print(f"error at country {country_code}")
            raise Exception
        # get word counts per window
        corpus_sizes = (
        wordcount_df.filter(pl.col("country") == country_code)
        .group_by_dynamic("Date_formatted", every=f"{window_size}d", closed="left", include_boundaries=True)
        .agg(pl.sum("word_count").alias("window_word_count")))

        # add the normalized mention count
        try:
            edges_df = edges_df.join(corpus_sizes, on=["_lower_boundary", "_upper_boundary"], how="left")
        except Exception as e:
            raise Exception(f"ERROR at {country}, edges_df:\n{edges_df}\n\n{corpus_sizes}\n\n{wordcount_df.filter(pl.col('country') == country_code)}")
        edges_df = edges_df.with_columns(((pl.col("mentions_in_window") / pl.col("window_word_count")) * 100000).alias("mentions_normalized"))
        
        # add source country code
        edges_df = edges_df.with_columns(pl.lit(country_code).alias("source_country"))

        # merge with final dataframe
        final_window_df = pl.concat([final_window_df, edges_df])
    
    final_window_df.write_parquet(dataframe_filename)
    final_window_df = final_window_df.with_columns(pl.col("_lower_boundary").alias("window_start_date"))

    pdf = final_window_df.to_pandas()
    pdf["source_country_full"] = pdf["source_country"].map(lambda x: get_full_country_name(x))
    
    # draw lineplot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=pdf, x='window_start_date', y='mentions_normalized', hue='source_country_full')
    plt.ylabel(f"mentions per 100 000 words")

    # Format x-axis to show only the year
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.xlabel("")

    plt.legend(ncol=2)
    plt.title(f"Mentions of Ukraine for various countries from {start_date} to {end_date}")
    plt.savefig(plot_filename)
    plt.show()


if __name__ == "__main__":
    all_countries = get_default_countries()
    #temp_relevant_countries = ",".join(["PL", "CZ", "ES", "IT", "EE", "SK", "SI", "RS", "IS", "BA"])

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dataframe_filename", type=str)
    parser.add_argument("output_plot_filename", type=str)
    parser.add_argument("target_country", type=str)
    parser.add_argument("--source_countries", type=str, default=all_countries, required=False)
    parser.add_argument("--wordcounts_folder", type=str, default="ParlaMint_wordcounts", required=False)
    parser.add_argument("--edgelist_folder", type=str, default="ParlaMint_edgelists", required=False)
    parser.add_argument("--start_date", type=str, default="2015-01-01", required=False)
    parser.add_argument("--end_date", type=str, default="2022-12-31", required=False)
    parser.add_argument("--window_size", type=str, default="90", required=False)
    args = parser.parse_args()

    create_df_n_visualize(args.wordcounts_folder, args.edgelist_folder, args.target_country, 
                          args.source_countries.split(","), args.start_date, args.end_date, 
                          args.window_size, args.output_dataframe_filename, args.output_plot_filename)
