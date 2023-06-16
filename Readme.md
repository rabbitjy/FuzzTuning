# FuzzTuning

code for our acl 2023 paper [Understanding Programs by Exploiting (Fuzzing) Test Cases](https://arxiv.org/pdf/2305.13592.pdf)

## Requirements

torch 1.7.0

transformers 4.18.0

## Pretrained Model

Download [codebert-base](https://huggingface.co/microsoft/codebert-base) and [unixcoder-base](https://huggingface.co/microsoft/unixcoder-base), and move the files to ```./microsoft/codebert-base``` and ```./microsoft/unixcoder-base```

## Clone Detection (POJ-104)

### Task Definition

Given a code and a collection of candidates as the input, the task is to return Top K codes with the same semantic. Models are evaluated by MAP@R score. MAP@R is defined as the mean of average precision scores, each of which is evaluated for retrieving R most similar samples given a query. For a code (query), R is the number of other codes in the same class, i.e. R=499 in this dataset.


### Dataset

We take [POJ-104](https://arxiv.org/pdf/1409.5718.pdf) dataset on this task as an exmaple.

#### Download and Preprocess

Download POJ dataset from [programs.tar.gz](https://drive.google.com/file/d/1x0nucnROMhDDxyJmoUnYWfggRPk9pQUu/view?usp=drive_link) and download POJ fuzzing data from [POJ104.io.tar.gz](https://drive.google.com/file/d/1uLj_d1bKl4HbIos_4p1Fg9Z3jKodAHn-/view?usp=drive_link). 
```shell
mv programs.tar.gz POJ104.io.tar.gz ./dataset
cd dataset
tar zxvf programs.tar.gz
tar zxvf POJ104.io.tar.gz
mv fuzz output
python preprocess_clone.py
cd ..
```

#### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train_fuzz_clone.jsonl, valid_fuzz_clone.jsonl, test_fuzz_clone.jsonl.

The processed .jsonl files are also at ```./dataset```.

For each file, each line in the uncompressed file represents one function. One row is illustrated below.

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of example
   - **fuzz:** the fuzzing test cases including input and output pair

#### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Problems | #Examples |
| ----- | --------- | :-------: |
| Train | 64        |  32,000   |
| Dev   | 16        |   8,000   |
| Test  | 24        |  12,000   |

###  Evaluator

We use the script provided from [website](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104) to evaluate predictions for this task, and report MAP@R score.


### Pipeline-FuzzTuning-Clone

We also provide a pipeline of FuzzTuning on this task in ```./clone/run.sh```. 


## Code Classification (POJ-104)

### Task Definition
The  code classification task requires that we assign the same label to programs that were implemented to solve the same problem and achieve the same goal. Models are evaluated by accuracy. 


### Dataset

We take [POJ-104](https://arxiv.org/pdf/1409.5718.pdf) dataset on this task as an example.


#### Download and Preprocess

The process for dataset is similar with clone detection unless
```shell
python preprocess_cls.py
```

#### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train_fuzz_cls.jsonl, valid_fuzz_cls.jsonl, test_fuzz_cls.jsonl.

The processed .jsonl files are also at ```./dataset```.

For each file, each line in the uncompressed file represents one function. One row is illustrated below.

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of example
   - **fuzz:** the fuzzing test cases including input and output pair

#### Data Statistics

Data statistics of the dataset are shown in the below table:

|       |  #Examples |
| ----- |  -------   |
| Train |   31,338   |
| Dev   |   10,294   |
| Test  |   10,368   |


### Pipeline-FuzzTuning-Cls

We also provide a pipeline of FuzzTuning on this task in ```./cls/run.sh```. 


## Reference
Please cite our work in your publications if it helps your research:

<pre><code>@article{zhao2023understanding,
  title={Understanding Programs by Exploiting (Fuzzing) Test Cases},
  author={Zhao, Jianyu and Rong, Yuyang and Guo, Yiwen and He, Yifeng and Chen, Hao},
  journal={arXiv preprint arXiv:2305.13592},
  year={2023}
}</code></pre>
