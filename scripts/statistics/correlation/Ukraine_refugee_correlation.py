import polars as pl
import os
from datetime import date
from scipy.stats import pearsonr

refugee_data = pl.read_csv("UA_refugees_by_country.tsv", separator="\t")

print(refugee_data)

mentions_data = pl.read_parquet("../lineplots/RU_sliding_years.parquet").filter(pl.col("source_country") != "FI").rename({"source_country": "country_code"})

print(mentions_data)

before_date = pl.lit("2021-03-01").str.strptime(pl.Datetime)
after_date = pl.lit("2022-03-01").str.strptime(pl.Datetime)
before_df = mentions_data.filter((pl.col("_lower_boundary") <= before_date), (pl.col("_upper_boundary") >= before_date))
after_df = mentions_data.filter((pl.col("_lower_boundary") <= after_date), (pl.col("_upper_boundary") >= after_date))

#print(set(before_df["source_country"].to_list()))
#print(set(before_df["source_country"].to_list()) - set(after_df["source_country"].to_list()))
refugee_data = refugee_data.join(before_df.select(pl.col("country_code"), pl.col("mentions_normalized").alias("mentions_before")), how="left", on="country_code")
refugee_data = refugee_data.join(after_df.select(pl.col("country_code"), pl.col("mentions_normalized").alias("mentions_after")), how="left", on="country_code")
refugee_data = refugee_data.filter(pl.col("mentions_before").is_not_null()).with_columns((pl.col("mentions_after") - pl.col("mentions_before")).alias("delta_mentions"))

with pl.Config(tbl_rows=-1):
    print(refugee_data)

print(pearsonr(refugee_data["distance_to_RU"], refugee_data["delta_mentions"]))
