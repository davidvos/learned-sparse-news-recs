# LSNR: Learned Sparse News Recommendations

This repository contains the supporting material for the paper 'Learned Sparse News Recommendations for Topical Distribution Shifts in Zero-Shot and User Cold-Start Settings'.

News recommendation approaches typically rely on long sequences of user interactions to learn dense representations that model the dynamics between users and news articles. This reliance makes it challenging for current approaches to recommend relevant news under topical distribution shifts, such as in out-of-domain settings and for cold-start users with limited previous interactions. We hypothesize that learned sparse embeddings can better tackle the problem of these distribution shifts by using lexical matching signals than dense representation methods. We propose learned sparse news recommendations (LSNR), a news recommendation approach that uses lexical embeddings for users and news articles. Experiments on two benchmark datasets demonstrate that LSNR is more robust under topical distribution shifts than dense representation methods. LSNR outperforms existing news recommendation methods based on dense representations in zero-shot out-of-domain and user cold-start scenarios.

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
4. Unzip the downloaded files and put the unzipped files in the folders of the dataset that you want to pre-process and use. For example: 
```bash
# unzip folder into existing folder
unzip MINDsmall_train.zip -d data/mind/train
unzip MINDsmall_dev.zip -d data/mind/val
```
This will allow pre-processing and training on the MIND dataset for regaular and cold-start users.
5. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
6. Pre-process the MIND and EBNeRD datasets by running the following commands:
```bash
python src/data_preprocess_mind.py
```
```bash
python src/data_preprocess_ebnerd.py
```
These commands will pre-process the data for the regular and user cold-start settings. To pre-process the data for the zero-shot setting, modify and run the following commands:
```bash
python src/data_preprocess_mind_zero_shot.py
```
```bash
python src/data_preprocess_ebnerd_zero_shot.py
```
7. For the ```--dataset``` argument you can choose between 'mind', 'ebnerd', 'mind_zero_shot_news_sports', 'mind_zero_shot_sports_news', 'ebnerd_zero_shot_news_sports' and 'ebnerd_zero_shot_news_sports'. For the ```--model``` argument you can choose between 'lsnr' and 'lsnr_cat', 'nrms', 'naml' and 'lstur'. Train a model by running the following command:
```bash
python src/train.py --dataset mind --model lsnr
```
8. Evaluate the model by running the following command:
```bash
python src/evaluate.py --dataset mind --model lsnr
```
