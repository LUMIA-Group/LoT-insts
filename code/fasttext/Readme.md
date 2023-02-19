# read me
This is the implementation of fasttext for Affiliation-Normalization.

The configuration is in config.py, including:
- input, output path for processes
- arguments of model in training

In DataProcess.py, labels are transformed to num and their mapping relation is stored in label2num.pkl and num2label.pkl.

In trainAndscore.py, fasttext model is trained and score is calculate by function **getscore**

Suggested environment to run the code:

- Ubuntu 18.04.02
- Python 3.7.2
- tqdm
- pickle
- sklearn
- fasttext
