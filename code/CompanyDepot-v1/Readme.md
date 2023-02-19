# Read Me 
This is the implementation of **CompanyDepot** for Affiliation-Normalization.

The configuration is in **config.py**, including:

- Index name & hosts for database;

- Paths to load norm2popular, norm2id, id2norm and norm2len;
- Paths to load train and test files;
- Paths where indri_file is saved after running ranklib;
- Paths to save results;

To reproduce CompanyDepot，you need to first install **ranklib** which is in  [The Lemur Project / Wiki / RankLib (sourceforge.net)](https://sourceforge.net/p/lemur/wiki/RankLib/) and run the following steps:

1. run CompanyDepot.py with `python CompanyDepot.py --mode "preprocess"`
2. run ranklib with features saved in step 1 and modify the saved path in the config.py
3. run CompanyDepot.py with `python CompanyDepot.py --mode "after_ranklib"`
4. run CompanyDepot.py with `python CompanyDepot.py --mode "get_result"`

The main function for normalization is **get_result** in **CompanyDepot.py**, which returns the features for ranklib to process.

The function for CSC is **get_performance** in **CompanyDepot.py**, which returns the accuracy, precision, recall and F1.

Note that this baseline is not suitable for OSV, and thus no corresponding functions.

Suggested environment to run the code:
- Ubuntu 18.04.02
- Python 3.7.3
- heapq
- elasticsearch
- sklearn
- argparse
- pickle
- multiprocessing
- re

