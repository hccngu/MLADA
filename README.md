# Meta-Learning Adversarial Domain Adaptation Network for Few-Shot Text Classification

This repository contains the code and data for our ACL 2021 paper:

[*Meta-Learning Adversarial Domain Adaptation Network for Few-Shot Text Classification*](https://aclanthology.org/2021.findings-acl.145.pdf)

If you find this work useful and use it on your own research, please cite our paper.

`````
@inproceedings{MLADA:conf/acl/HanFZQGZ21,
  author    = {Chengcheng Han and
               Zeqiu Fan and
               Dongxiang Zhang and
               Minghui Qiu and
               Ming Gao and
               Aoying Zhou},
  title     = {Meta-Learning Adversarial Domain Adaptation Network for Few-Shot Text
               Classification},
  booktitle = {Findings of the Association for Computational Linguistics: {ACL/IJCNLP}
               2021, Online Event, August 1-6, 2021},
  series    = {Findings of {ACL}},
  volume    = {{ACL/IJCNLP} 2021},
  pages     = {1664--1673},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
}
`````

### Overview

We propose an adversarial domain adaptation network to enhance meta-learning framework, with the objective of improving the model’s adaptive ability for new tasks in new domains. 

We first utilize two neural networks competing against each other, separately playing the roles of a domain discriminator and a meta-knowledge generator. The adversarial network is able to strengthen the adaptability of the meta-learning architecture. 

Moreover, we aggregate transferable features generated by the meta-knowledge generator with sentence-specific features to produce high-quality sentence embeddings. 

Finally, we utilize a ridge regression classifier to obtain final classification results.  

Figure 1 gives an overview of our model.

![image-20210908204754657](C:\Users\30485\AppData\Roaming\Typora\typora-user-images\image-20210908204754657.png)

### Data

We ran experiments on a total of 4 datasets. You may unzip our processed data file `data.zip` and put the data files under `data/` folder.

| Dataset | Notes |
|---|---|
| 20 Newsgroups ([link](http://qwone.com/~jason/20Newsgroups/ "20 Newsgroups")) | Processed data available. We used the `20news-18828` version, available at the link provided.
| Reuters-21578 ([link](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html "Reuters")) | Processed data available. |
| Amazon reviews ([link](http://jmcauley.ucsd.edu/data/amazon/ "Amazon")) | We used a subset of the product review data. Processed data available. |
| HuffPost&nbsp;headlines&nbsp;([link](https://www.kaggle.com/rmisra/news-category-dataset "HuffPost")) | Processed data available. |

Please download pretrained word embedding file `wiki.en.vec` from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec) and put it under `pretrain_wordvec/` folder.

### Quickstart
After you have finished configuring the `data/` folder and the `pretrain_wordvec/` folder, you can run our model with the following commands. 
```
cd bin
sh mlada.sh
```
You can also adjust the model by modifying the parameters in the `malada.sh` file.

### Dependencies
- Python 3.7
- PyTorch 1.6.0
- numpy 1.18.5
- torchtext 0.7.0
- termcolor 1.1.0
- tqdm 4.46.0
- CUDA 10.2
