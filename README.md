# PyOpenDial

This is a Python implementation of OpenDial (http://www.opendial-toolkit.net/, https://github.com/plison/opendial) 

by KAIST AIPR Lab (http://ailab.kaist.ac.kr)

## Installation
### 1. Download this repository:
```
git clone https://github.com/KAIST-AILab/PyOpenDial.git
```

### 2. Install conda environment
```
cd PyOpenDial
conda env create -f environment.yml
conda activate py-opendial
```

## How to run

### (Optional) To use Google Speech API

#### 1. Please create service account key:
https://cloud.google.com/text-to-speech/docs/quickstart-protocol

#### 2. Please write down your json file path on the 'GOOGLE_APPLICATION_CREDENTIALS' in settings.yml

```
# Execution
python main.py
```
You can try various examples in example_domains, including 'example_domains/negotiation/negotiation.xml' and 'example_domains/example-flightbooking.xml'.

The detailed information of the negotiation dialogue can be found in the followings:
- https://www.aclweb.org/anthology/D17-1259
- https://github.com/facebookresearch/end-to-end-negotiator)
