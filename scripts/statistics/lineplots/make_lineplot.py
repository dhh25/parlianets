import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

start_year = "2015"
end_year = "2022"
source_countries_list = ["PL", "CZ", "ES", "IT", "SK", "SI", "RS", "IS", "BA"]
target_country = "UA"


def calculate_rel_freq(abs_freq, corp_size):
    return (abs_freq / corp_size) * 100000


def get_year_corpus_size(country_code, year, wc_folder):
    for folder in os.listdir(wc_folder):
        if folder == country_code:
            if not os.path.isfile(os.path.join(wc_folder, folder, f"{year}_wc.parquet")):
                continue
            wc_df = pd.read_parquet(os.path.join(wc_folder, folder, f"{year}_wc.parquet"))
            return sum(wc_df["word_count"])
    
    return None


# create a final dataframe that has years as rows and the columns are source country, relative frequency
# create also temporary lists for each source country which can be visualised on a plot
def create_df_n_visualize(wc_folder, el_folder, plot_filename):
    year_range = range(int(start_year), int(end_year) + 1)
    result_df = pd.DataFrame(columns=["year"] + source_countries_list)
    result_df["year"] = list(year_range)

    for country in os.listdir(el_folder):
        # for now, I am not considering smaller political entities (like Catalonia, Gallicia, etc.)
        if "not found" in country or "-" in country:
            continue
        country_code = country.split("_")[0]
        edges_df = pd.read_parquet(os.path.join(el_folder, country))
        years_list = list()
        rel_freqs_list = list()

        for year in year_range:
            relevant_year_df = edges_df[(edges_df["Date"].str.contains(f"{year}-")) & (edges_df["target_country"] == target_country)]
            #print(relevant_year_df.head())
            absolute_freq = relevant_year_df.shape[0]
            year_corpus_size = get_year_corpus_size(country_code, year, wc_folder)

            years_list.append(year)
            if year_corpus_size:
                rel_freqs_list.append(calculate_rel_freq(absolute_freq, year_corpus_size))
            else:
                rel_freqs_list.append(np.nan)
        
        result_df[country_code] = rel_freqs_list
    
    result_df_long = pd.melt(result_df, id_vars=['year'], value_vars=source_countries_list, var_name='country', value_name='mentions')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=result_df_long, x='year', y='mentions', hue='country')
    plt.ylabel("mentions per 100,000 words")
    plt.xticks(result_df_long['year'])
    plt.title(f"Mentions of Ukraine for various countries from {start_year} to {end_year}")
    plt.savefig(plot_filename)
    plt.show()

    return result_df_long


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wordcounts_folder", type=str)
    parser.add_argument("edgelist_folder", type=str)
    parser.add_argument("output_dataframe_filename", type=str)
    parser.add_argument("output_plot_filename", type=str)
    args = parser.parse_args()

    relative_freq_df = create_df_n_visualize(args.wordcounts_folder, args.edgelist_folder, args.output_plot_filename)
    relative_freq_df.to_parquet(args.output_dataframe_filename)
