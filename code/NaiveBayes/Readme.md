# Read Me 
This is the implementation of NaiveBayes based Affiliation-Normalization.

The configuration is in **config.py**, including:
- File path of train data in .txt format;
- File path of pickle cache for NaiveBayes;
- Number of processes for multiprocessing;

Before the main functions, the **preprocess.py** script should be run once, to generate the cache file to the path assigned in **config.py**. The train data, in .txt format, contains records with **SurfaceName \t\t NormalizedName**.

The main function for normalization is **bayes_normalize** in **bayes.py**, which returns a list of normalized names for the input list of original names.

The function for OSC is **bayes_normalize_with_score** in **bayes.py**, which returns a list of tuples (NormalizedName, Confidence) for the input list of orginal names. The confidence values can be used to discriminate the samples as open set or not.

The function for OSV is **bayes_discriminate** in **bayes.py**, which returns a list of JS distance for the two input lists of original names. The JSD values can be used to verify whether the pair of samples belong to same entity.

Suggested environment to run the code:
- Ubuntu 18.04.02
- Python 3.7.3
- re
- nltk
