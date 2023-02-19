# Dataset description

## Download link

+ Google drive
https://drive.google.com/drive/folders/1qbUxrgO4j8LvV5pXxe8tvmcB2inqIJog?usp=sharing


## Dataset Partition

Our Dataset include train/valid/test/open partition. Different methods may use differnt partition. For example,  there is no validition file used in search-based method.


For CSC task:

| Method | Train File Path | Validation File Path | Test File Path | ES_index File Path | LTR File Path
| :----| ----: | :----: | :----: | :----: |  :----: |
| Naive Bayes| train/train.txt | - | test/csc_test.txt | - | -
| sCool | - | - | test/csc_test.txt | train/train.txt | -
| CDv1 | - | - | test/csc_test.txt | train/es_train.txt | ltr_train.txt
| BERT | train/train_part.txt | dev/dev.txt | test/csc_test.txt | - | -
| Our Method | train/train_part.txt | dev/dev.txt | test/csc_test.txt | - | -

For OSC task

| Method |  Test File Path 
| :----| ----: 
| all| test/test_osc.txt | 

For OSV task
| Method |  Validation File Path |Test File Path 
| :----| :----: | :----:
| all| dev/dev_osv.txt | test/test_osv.txt


## Data File Format
## Train
There are four files in train folder:

 - train.txt
- train_part.txt
- es_train.txt
- ltr_train.txt

every line in these files is a mapping, it is organized by 
```
[original_name]\t\t[normalized_name]
```
+ sample
```
university of nijmegen , holland		radboud university nijmegen
pennsylvania state university's center for cell research, university park 16802.		pennsylvania state university
```
you can use python code to read it easily.

## Validation

Validation set also called development set(dev set). It is created for some machine learning method. There are two tasks have dev set.

### Dev set for CSC
As same as train file, every line in these files is a mapping, it is organized by 
```
[Original_name]\t\t[Normalized_name]
```
### Dev set for OSV
 every line in this file is a triplet, it is organized by
```
[first_institution_name]\t\t[second_institution_name]\t\t[label]
```
label=0 means first_institution_name and second_institution_name is not the same institution, 

label=1 means first_institution_name and second_institution_name is the same institution. 
+ sample 
```
pulmonary engineering group, clinic of anaesthesiology and intensive care therapy, university clinic dresden, germany		diies, "mediterranea" university of reggio calabria, 89060 reggio calabria, italy		0
suffolk county community college, selden ny united states of america		department of english, suffolk county community college, long island, new york, usa		1
```

## Test
There are three tasks.
### Test set for CSC
every line in these files is a mapping, it is organized by 
```
[original_name]\t\t[normalized_name]
```
+ sample
```
engineering administration, chesapeake midstream partners, oklahoma city, ok, united states		chesapeake energy
abteilung fur allgemeinchirurgie und abteilung fur mikroskopische anatomie, universitat hamburg, hamburg 20, germany		university of hamburg
```
### Test set for OSC
every line in this file is a triplet, it is organized by
```
[original_name]\t\t[normalized_name]\t\t[label]
```
label=0 means it is an institution in train set, 

label=1 means it is an institution in train set(in open set).
+ sample
```
huematology department, st bartholomew's hospital, london ecla 7be		st bartholomew s hospital		1
department of surgery, university of nevada las vegas medicine, las vegas, nv.		university of nevada las vegas		0
```
### Test set for OSV
 every line in this file is a triplet, it is organized by
```
[first_institution_name]\t\t[second_institution_name]\t\t[label]
```
label=0 means first_institution_name and second_institution_name is not the same institution, 

label=1 means first_institution_name and second_institution_name is the same institution. 
+ sample 
```
pulmonary engineering group, clinic of anaesthesiology and intensive care therapy, university clinic dresden, germany		diies, "mediterranea" university of reggio calabria, 89060 reggio calabria, italy		0
suffolk county community college, selden ny united states of america		department of english, suffolk county community college, long island, new york, usa		1
```
