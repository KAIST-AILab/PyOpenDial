#!/usr/bin/env bash

conda env create -f environment_chitchat.yml
conda activate py-opendial-chitchat

PYTHON_BIN=~/anaconda3/envs/py-opendial-chitchat/bin/python

git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI; $PYTHON_BIN setup.py develop
cd ..

$PYTHON_BIN -m spacy download en


cd example_domains/chitchat
mkdir checkpoints
wget -O checkpoints/last_checkpoint https://www.dropbox.com/s/cs6zd9yntn6ixea/last_checkpoint?dl=1

wget -O parameters.zip https://www.dropbox.com/s/n2jbjyq32x6jgr6/parameters.zip?dl=1
unzip parameters.zip
