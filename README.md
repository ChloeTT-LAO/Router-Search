<div align="center">

<h1> R1-Router: Learning to Route Queries Across Knowledge Bases for Step-wise Retrieval-Augmented Reasoning </h1>


<h5 align="center"> 

<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/hmhm1229/R1-Router'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>

[Chunyi Peng]()<sup>1</sup>,
[Zhipeng Xu]()<sup>1</sup>,
[Zhenghao Liu](https://edwardzh.github.io/)<sup>1</sup>,
[Yishan Li]()<sup>3</sup>,
[Yukun Yan]()<sup>2</sup>,
[Zhiyuan Liu]()<sup>2</sup>,
[Yu Gu]()<sup>1</sup>
[Minghe Yu]()<sup>1</sup>
[Ge Yu]()<sup>1</sup>
[Maosong Sun]()<sup>2</sup>

<sup>1</sup>Northeastern University, <sup>2</sup>Tsinghua University, <sup>3</sup>ModleBest.Inc

<h5 align="center"> If you find this project useful, please give us a star🌟.
</h5>
</div>

## Environment
For training, answer generation, and evaluation processes:
```bash
conda create -n router python=3.11
conda activate router
pip install requirements_ag.txt
```
For retriever and corpus construction processes:
```bash
conda create -n retriever python=3.11
conda activate retiever
pip install requirements_r.txt
```

## Corpora Construction
For the text corpus, you can download `enwiki-20241020` from [Google Drive](https://). Then preprocess, and index it with the following commands:
```bash
conda activate retriever
wikiextractor enwiki-20241020-pages-articles-multistream.xml.bz2 -o wiki_extracted
python wiki_preprocess.py
```
For the image corpus, you can directly download [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR). To embed and index it you can follow the [repository](https://github.com/TIGER-AI-Lab/UniIR)

For the table corpus, you can download, embed and index Open-WikiTable following the [repository](https://github.com/sean0042/Open_WikiTable), or you can download directly the one we have already preprocessed from [here](https://huggingface.co/hmhm1229/table-retriever). 

## Retrievers
For the Text-Image Retriever, you can directly download [UniIR](https://huggingface.co/TIGER-Lab/UniIR)

For the Table Retriever, you can train it with the help of the [repository](https://github.com/sean0042/Open_WikiTable), or you can download it directly from [here](https://huggingface.co/hmhm1229/table-retriever). 

## Training
If you do not want to train the model, you can download [R1-Router](https://huggingface.co/hmhm1229/R1-Router) and skip this section to [Evaluation](#evaluation)
### Data Synthesis
If you want to use the ready-to-use synthetic data directly, you can skip this section to [Step-GRPO Training](#step-grpo-training)

First, we need to synthesis the data step by step:
```bash
bash src/data_synthesis/data_synthesis.sh
```
### Step-GRPO Training
Our training framework is based on [EasyR1](https://github.com/hiyouga/EasyR1), only you need to do is to download it and replace some files with the files in `./Easy-R1`.
Then start training with the command:
```bash
conda activate ag
bash examples/run_qwen2_5_vl_7b_stepgrpo.sh
```
## Evaluation
We provide the evaluation pipeline for the R1-Router:
```bash
bash evaluation.sh
```

## Acknowledgement 
Our work is built on the following codebases, and we are deeply grateful for their contributions.
- [EasyR1](https://github.com/hiyouga/EasyR1)
- [UniIR](https://huggingface.co/TIGER-Lab/UniIR)
- [Open-WikiTable](https://github.com/sean0042/Open_WikiTable)
- [OmniSearch](https://github.com/Alibaba-NLP/OmniSearch)

## Citation
```
@artile{
  
}
```