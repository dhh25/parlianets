import argparse
from topic_extract_fgp import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=str, required=False)
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--tfile', type=str, required=False)
    parser.add_argument('--efile', type=str, required=False)
    parser.add_argument('--filtered_dir', type=str, required=False)
    parser.add_argument('--preproc_dir', type=str, required=False)
    parser.add_argument('--cache_dir', type=str, required=False)
    args = parser.parse_args()

    main(
        target_dir='topic_extract',
        scratch=args.scratch,
        filtered_dir=args.filtered_dir, # directory containing CC_filtered.parquet files
        texts_file=args.tfile, # text file, iso2cc and year inferred from this
        batch_size=100,
        context_type='segment',
        cache_dir=args.cache_dir
    )
