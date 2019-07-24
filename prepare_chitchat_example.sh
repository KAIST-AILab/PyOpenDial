#!/usr/bin/env bash

conda env create -f environment_chitchat.yml
conda activate py-opendial-chitchat

git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI; python setup.py develop
cd ..

cd example_domains/chitchat
mkdir checkpoints
wget -O checkpoints/last_checkpoint https://www.dropbox.com/s/cs6zd9yntn6ixea/last_checkpoint?dl=1

wget -O parameters.zip https://www.dropbox.com/s/n2jbjyq32x6jgr6/parameters.zip?dl=1
unzip parameters.zip
