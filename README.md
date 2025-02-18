# LSNR: Learned Sparse News Recommendations

This repository contains the supporting material for the paper 'LSNR: Learned Sparse News Recommendations for Cold-Start Users'.

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
3. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
4. Pre-process the MIND and EBNeRD datasets by running the following commands:
```bash
python src/data_preprocess_mind.py
```
```bash
python src/data_preprocess_ebnerd.py
```
5. For the ```--dataset``` argument you can choose between 'mind' and 'ebnerd'. For the ```--model``` argument you can choose between 'lsnr' and 'lsnr_cat', 'nrms', 'naml' and 'lstur'. Train a model by running the following command:
```bash
python src/train.py ---dataset mind --model lsnr
```
6. Evaluate the model by running the following command:
```bash
python src/evaluate.py --dataset mind --model lsnr
```
