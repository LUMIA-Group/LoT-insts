# BERT-based Method
BERT-based method including three methods shown in our paper. They are:
+ BERT:Fine-tune original BERT directly
+ BERT(RS):Fine-tune original BERT with resampling strategy
+ BERT(FP+RS):Further pretrain in institution corpus firstly, then fine-tune it on train dataset.

The implementation for this are briefly introduced as follows:

## Fine-tune with resampling strategy
**Original/fine_tune.py** is the main file for fine-tune. It can be used by setting some parameters easily. It is listed as follows:
```
# Resampling rate
# RS=1 means no Resampling
RS_EXP = 0.3
# Model_path is the checkpoint path of pretrained model,If it is empty, the bert-uncased-base(From Huggingface Transformer) model will be used.
Model_path = '../checkpoint/epoch_2'
# EPOCH_NUM is number of epochs
EPOCH_NUM = 100
# Train_batch_size
Train_batch_size = 512
# Validation_batch_size
Validation_batch_size = 512
# SAVE_CHECKPOINT_PATH to set where to save the fine-tuned model
SAVE_CHECKPOINT_PATH = './checkpoint/'
# Learning_rate
Learning_rate = 5e-5
# DEVICE_NAME means which device will be used
DEVICE_NAME = "cuda:0"
```

When you want to fine-tune original BERT directly, you can just set 
```
RS_EXP = 1
RW_EXP = 0
```
When you want to fine-tune original BERT with resampling strategy, you can set the ```RS_EXP```.

The same file content for **RS/fine_tune.py** and **FP_and_RS/fine_tune.py**.



## Further-Pretrain
**FP_and_RS/further_pretrain.py** is adapted from this file [run_mlm_no_trainer.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py).

the document for this script in [Language model training](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)



## Evaluate For CSC task
**Evaluate/Evaluate_CSC_task.ipynb** is the implementation for evaluating a model's perfomance for CSC task.

You can set the model which you want to evaluate.


## Evaluate For OSC task

**Evaluate/Evaluate_OSC_task.ipynb** is the implementation for evaluating a model's perfomance for OSC task.

You can set the model which you want to evaluate.

## Evaluate For OSV task
**Evaluate/Evaluate_OSV_task.ipynb** is the implementation for evaluating a model's perfomance for OSV task.

You can set the model which you want to evaluate.


# Requirements
Suggested environment to run the code:

+ Ubuntu 18.04.02
+ pytorch 1.7.1
