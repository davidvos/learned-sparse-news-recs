# LSNR: Learned Sparse News Recommendations

This repository contains the supporting material for the paper 'Learned Sparse News Recommendations for Short-History Users and Out-of-Distribution Items'.

News recommendation methods typically rely on long sequences of user interactions to learn dense representations that model the general interests of users. In this work, we identify two settings where such methods underperform: (i) users with short interaction histories, and (ii) candidate news articles whose topics were not seen during training. We propose learned sparse news recommendations (LSNR), the first news recommendation approach that learns lexical representations for users and news articles. Unlike dense models that compress signals into low-dimensional embeddings, LSNR preserves interpretable lexical features, enabling it to capture stronger signals from limited data and generalize more effectively across domains. Experiments on two public datasets show that LSNR outperforms dense baselines in settings with short-history users and out-of-distribution items. In long history and in-domain settings, LSNR matches dense methods in performance.
## Get Started

This project uses Weights and Biases logging. To use this feature, you need to create a Weights and Biases account and login. Alternatively, you can remove the corresponding code in the ```src/train.py``` file.

1. Clone the repository by running the following command:
```bash
git clone https://github.com/davidvos/learned-sparse-news-recs.git
```

2. Navigate to the repository by running the following command:
```bash
cd learned-sparse-news-recs
```

3. Download the MIND (small) and EBNeRD (small) datasets from the following links:
   * MIND
     * [Train](https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip)
     * [Validation](https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip)
   * [EBNeRD](https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip)

```bash
wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
```
  
4. Unzip the downloaded files and put the unzipped files in the folders of the dataset that you want to pre-process and use. 

First create a data folder:
```bash
mkdir data
mkdir data/mind data/mind/train data/mind/val data/mind/test data/ebnerd data/ebnerd/test
```

For MIND: 
```bash
unzip MINDsmall_train.zip -d data/mind/train
unzip MINDsmall_dev.zip -d data/mind/val
unzip MINDsmall_dev.zip -d data/mind/test
```
For EBNeRD:
```bash
unzip ebnerd_small.zip -d data/ebnerd
mv data/ebnerd/validation data/ebnerd/val
cp -r data/ebnerd/val/* data/ebnerd/test
```

5. To create folder for the out-of-distribution setting we need to copy the MIND and EBNeRD datasets to the new folder. You can do this by running the following commands:
```bash
mkdir data/mind_news data/mind_news/train data/mind_news/val data/mind_news/test
mkdir data/mind_sports data/mind_sports/train data/mind_sports/val data/mind_sports/test
mkdir data/ebnerd_nyheder data/ebnerd_nyheder/train data/ebnerd_nyheder/val data/ebnerd_nyheder/test
mkdir data/ebnerd_sport data/ebnerd_sport/train data/ebnerd_sport/val data/ebnerd_sport/test

cp -r data/mind/* data/mind_news/
cp -r data/mind/* data/mind_sports/
cp -r data/ebnerd/* data/ebnerd_nyheder/
cp -r data/ebnerd/* data/ebnerd_sport/
```

6. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

7. Pre-process the MIND and EBNeRD datasets by running the following commands:
```bash
python src/data_preprocess_mind.py
```
```bash
python src/data_preprocess_ebnerd.py
```
These commands will pre-process the data for the regular and user cold-start settings. To pre-process the data for the out-of-distrubution setting, run the following commands:
```bash
python src/data_preprocess_mind_ood.py
```
```bash
python src/data_preprocess_ebnerd_ood.py
```

1. For the ```--dataset``` argument you can choose between 'mind', 'ebnerd', 'mind_news', 'mind_sports', 'ebnerd_nyheder' and 'ebnerd_sport'. For the ```--model``` argument you can choose between 'lsnr', 'nrms', 'naml' and 'lstur'. Train a model by running the following command:
```bash
python src/train.py --dataset mind --model lsnr --batch_size 32 --eval_every 1000 --num_workers 4 --num_epochs 10 --learning_rate 0.00001
```

1. Evaluate the model by running the following command:
```bash
python src/evaluate.py --dataset mind --model lsnr --batch_size 32 --num_workers 4 --n_epochs 10 --learning_rate 0.00001
```
