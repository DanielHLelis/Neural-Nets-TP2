# TP2 - Neural Nets

## Authors

- D. H. Lelis - 12543822
- Samuel Veronez - 12542626
- Fernando Campos - 12542352

## Dataset

The main target here is developing for classifying portuguese text. We have
found some open datasets appropriate for usage:

- OLID-BR
  - This is the main dataset we are using.
  - License: CC BY 4.0
  - URL: <https://www.kaggle.com/datasets/dougtrajano/olidbr>
  - Author: Douglas Trajano
  - Description: The Offensive Language Identification Dataset for Brazilian
    Portuguese (OLID-BR). It is composed of 7943 (extendable to 13,538)
    comments from different sources, such as YouTube and Twitter.

- ToLD-Br
  - License: CC BY-SA 4.0
  - URL: <https://github.com/JAugusto97/ToLD-Br>
  - Paper: <https://aclanthology.org/2020.aacl-main.91>
  - Authors: João A. Leite, Diego F. Silva, Kalina Bontcheva and Carolina Scarton
  - Description: The Toxic Language Detection Dataset for Brazilian Portuguese
    (ToLD-Br) is a dataset for the task of toxic language detection in Brazilian
    Portuguese composed of tweets.

## Dependencies

### Conda or Mamba

When using `conda`, `micromamba` and its derivatives, you can create a new
environment, activate it, and add the dependencies with the following commands:

```bash
# Create the environment
conda create -n toxicity-classification python=3.11 -c conda-forge

# Activate the environment
conda activate toxicity-classification

# Install dependencies for visualization
conda install jupyter polars numpy pandas pyarrow plotly matplotlib seaborn scipy -c conda-forge

# Install dependencies for text processing (stanza, spacy, unidecode and nltk)
conda install stanza nltk spacy unidecode -c conda-forge

# Install dependencies for machine learning (scikit-learn and pytorch)
conda install scikit-learn -c conda-forge

# For PyTorch, recommend checking for your specific setup at: https://pytorch.org
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia

# For transformers, we use some pre-trained models from Hugging Face, so we need to install it
conda install transformers -c conda-forge

# Install optuna for hyperparameter optimization
conda install optuna ray-tune -c conda-forge
```

For installing everything at once, you can use the `environment.yml` file:

```bash
# For Conda
conda env create -n toxicity-classification -f environment.yml

# For Micromamba
micromamba create -n toxicity-classification -f environment.yml
```
