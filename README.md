# parlianets
#parlianets @ DHH25

## Current Data (see Drive)
```
.
|-- ParlaMint_data (size: XXL, not in Drive), downloaded and extracted from source
|-- ParlaMint_entities_filtered (size: S, complete in Drive), created from ParlaMint_preprocessed
|-- ParlaMint_preprocessed (size: L, complete in Drive), created from ParlaMint_data
|-- ParlaMint_topics (size: M, currently computing, will go into Drive), created from ParlaMint_preprocessed and ParlaMint_entities_filtered
```

## Usage Hints
Sample invocation of new `topic_extraction.py` for the `AT` corpus, assuming you are in the `scripts` folder, 
assuming you have `ParlaMint_entities_filtered` and `ParlaMint_preprocessed/AT` in `root`:
```bash
python topic_extraction.py --preprocessed_dir ../ParlaMint_preprocessed --filtered_dir ../ParlaMint_entities_filtered --out_dir ../ParlaMint_topics --batch_size 100 --iso2_cc AT -p 8
```
This will save a new batch every 100 entities and parallelize available years over 8 cores.