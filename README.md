# LSNR: Learned Sparse News Recommendations

This repository contains the supporting material for the paper 'Reliable and Controllable Learned Sparse News Recommendations for Short-History Users'.

Modern news recommenders based on dense representations face a \emph{reliability} problem: they systematically provide a lower quality of service to the large segment of users with short interaction histories. They also face a \emph{controllability} problem: dense models sacrifice the controllable balance between performance and efficiency inherent to earlier sparse methods. To address these challenges, we propose learned sparse news recommendations (LSNR). LSNR features a novel user encoder that learns to generate sparse, bag-of-words-style representations of user interests, allowing explicit term matching as a robust signal for recommendation. LSNR's sparsity regularization provides a direct mechanism to control the performance-sparsity trade-off. Experiments on the MIND and Adressa datasets show that LSNR significantly outperforms state-of-the-art dense models for short-history users while remaining competitive for users with long histories. We show that learned sparse models can promote both reliability and practical control over news recommendation systems.

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
These commands will pre-process the data for the regular and user cold-start settings.

1. For the ```--dataset``` argument you can choose between 'mind' and 'ebnerd'. For the ```--model``` argument you can choose between 'lsnr', 'nrms', 'naml' and 'lstur'. Train a model by running the following command:
```bash
python src/train.py --dataset mind --model lsnr --batch_size 32 --eval_every 1000 --num_workers 4 --num_epochs 10 --learning_rate 0.00001
```

1. Evaluate the model by running the following command:
```bash
# Evaluate on validation set (default)
python src/evaluate.py --dataset mind --model lsnr --batch_size 32 --num_workers 4 --learning_rate 0.00001

# Evaluate on test set
python src/evaluate.py --dataset mind --model lsnr --batch_size 32 --num_workers 4 --learning_rate 0.00001 --eval_split test
```

The evaluation script now performs retrieval on all candidate news from the specified eval/test set, rather than just using the candidates for each specific user behavior. This provides a more comprehensive retrieval evaluation where each user's preferences are matched against the entire candidate pool from all users in the dataset.
