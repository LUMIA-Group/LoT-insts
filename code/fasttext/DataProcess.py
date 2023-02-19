from tqdm import tqdm_notebook as tqdm
import pickle
from config import *

labels = set()
label2num = {}
num2label = {}

def writeNewTXT(path,newpath): #transfrom labels into num
    f = open(newpath,'w')
    with open(path,encoding = 'utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines):
            words = line.split('\t\t',1)
            words[1] = words[1].replace('\n','')
            if(label2num.__contains__(words[1])):
                f.write(words[0])
                f.write('\t')
                f.write("__label__")
                f.write(str(label2num[words[1]]))
                f.write('\n')
    f.close()

with open(trainpath,encoding = 'utf-8') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        words = line.split('\t\t',1)
        words[1] = words[1].replace('\n','')
        labels.add(words[1])
labels = list(labels)

for i in range(len(labels)): 
    label2num[labels[i]] = i
    num2label[i] = labels[i]
    
pickle.dump(label2num,open(label2numpath,'wb'))
pickle.dump(num2label,open(num2labelpath,'wb'))  #write mapping file

writeNewTXT(trainpath,newtrainpath)
writeNewTXT(devpath,newdevpath)
writeNewTXT(testpath,newtestpath)