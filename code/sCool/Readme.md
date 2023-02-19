# Read Me 
This is the implementation of SCOOL for Affiliation-Normalization.

The configuration is in **config.py**, including:
- Parameters for boosting retrieval;
- Index name & hosts for database;
- Number of processes for multiprocessing;

The main function for normalization is **scool_normalize** in **scool.py**, which returns a list of normalized names for the input list of original names.

The function **scool_normalize_with_score** in **scool.py** returns a list of tuples (NormalizedName, Confidence) for the input list of orginal names. The confidence values can be used to discriminate the samples as open set or not.

This baseline is not suitable for OSV, and thus no corresponding functions.

Suggested environment to run the code:
- Ubuntu 18.04.02
- Python 3.7.3
- fuzzywuzzy
- elasticsearch
- multiprocessing
- tqdm
