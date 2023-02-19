# The Pretrained Character-level BERT Model

## Overview

There are four directorys here in this project. They are:

+ **dataset**: Store pre - training corpus and vocabulary files.
+ **logs**: Store log information during model pre-training/training, such as model configuration, changes in loss values, performance on validation sets.
+ **saved_model**: Store checkpoint files saved during model pre-training/training.
+ **src**: The code directory




## Quick Start
To run our code. The overall process is as follows:
1. Proper configuration of project information. That is, configure which mode(cpu/single gpu/multiple gpus) you will you use in src/cs/\__init\__.py file(cs means common setting). 
```
# Setting of distributed training

# There are three choices, you can use any mode depend on your environment
# 'cpu': use CPU to train
# 'single': use single GPU
# 'distributed': use distributed GPUs
device_mode = 'single'
rank = 0

if device_mode == 'cpu':
    use_cuda = False
    use_dist = False
    num_devices = 1
    device = torch.device('cpu')
elif device_mode == 'single':
    use_cuda = True
    use_dist = False
    num_devices = 1
    device_no = 2  
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_no}'
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(device)
elif device_mode == 'distributed':
    use_cuda = True
    use_dist = True

    tcp_url = 'tcp://10.10.10.11:23457'  # It must be the first GPU's IP address, arbitrary port value
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'  

    workers = [
        {'hostname': 'main-server-11', 'device_id': 0},
        {'hostname': 'main-server-11', 'device_id': 1},
        {'hostname': 'main-server-11', 'device_id': 2},
        {'hostname': 'main-server-11', 'device_id': 3},
    ]
```
When you use 'single' mode, you should set *device_no* to indicate which GPU to use.
When you use 'distributed' mode, you should set *tcp_url* to indicate which GPUs to use. The IP address must be the first GPU's IP address(the gpu in which sever/computer) and the port value is arbitrary. The GPU information used is then listed in workers.

2. Start a dataset manger for multiprocessing. For different tasks, edit src/start_manger.py file to fit corresponing tasks. And then run src/start_manger.py file. 
For example, for pretrain tasks, you should import start_mager function from tasks.masked_lm.manager.
```
from tasks.masked_lm.manager import start_manager
# from tasks.bert_classifier.manager import start_manager
# from tasks.bert_metric.manager import start_manager

if __name__ == '__main__':
    start_manager()
```

3. Configure *config.py* file for the corresponding task.

For all tasks, if you pretrain/train from scratch, you should set
```
last_training_time = 0
last_step = -1
```
if you train from a checkpoint, you should set these two variables to the appropriate values.


+ For Pretrain
The file you should configure is src/tasks/masked_lm/config.py. There are many parameters such as AnnBertConfig(Ann means affiliation name normalization, i.e. institution name normalization), and learning rate, batch size, training steps etc. And most importantly, you should set *manager_host* value to the host's ip address which used to start manger in step 2. And *manager_port* is arbitrary. 

```
manager_host = '10.10.10.8'
manager_port = 50009
```

+ For Classifier
The file you should configure is src/tasks/bert_classifier/config.py. There are many parameters such as AnnBertConfig(Ann means affiliation name normalization, i.e. institution name normalization), and learning rate, batch size, training steps etc. And most importantly, you should set *manager_host* value to the host's ip address which used to start manger in step 2. And *manager_port* is arbitrary. 

```
manager_host = '10.10.10.2'
manager_port = 50002
```
Then, you could config the *resampling_exp* to use different resampling stragety. 
```
# Resample_exp = 1 means no resampling
resample_exp = 0.3
```

+ For Model Use Contrastive Loss
The file you should configure is src/tasks/bert_metric/config.py. There are many parameters such as AnnBertConfig(Ann means affiliation name normalization, i.e. institution name normalization), and learning rate, batch size, training steps etc. And most importantly, you should set *manager_host* value to the host's ip address which used to start manger in step 2. And *manager_port* is arbitrary. 

```
manager_host = '10.10.10.3'
manager_port = 50003
```
Then, you could config the *resampling_exp* to use different resampling stragety. 
```
# Resample_exp = 1 means no resampling
resample_exp = 0.3
```

4. Start pretrain/train process.
Just run src/main.py to start.
```
python main.py
```
Note that you should the code corresponding to the task, and comment out the other code. For example, if you pretrain character-level model, you will use
```
from tasks.masked_lm.train import train
# from tasks.bert_classifier.train import train
# from tasks.bert_metric.train import train
```

**Note**: all pkl files used in code can be avaliable in https://drive.google.com/drive/folders/1J3oL3EZm9Idzy4Ij611fbk_T6qoc8Cn-?usp=sharing.

# Requirements
Suggested environment to run the code:
+ Ubuntu 18.04.02
+ pytorch 1.7.1
