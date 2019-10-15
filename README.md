# ai-recepit-art
## Run
You need to create 2 virtual environment to run this project, one for Bert Server, one for Generator.
### Bert
* change to directory of Bert.
```shell script
virtualenv venv
source venv/bin/activate
pip install tensorflow==1.14.0
pip install bert-serving-server
```
* Download checkpoint from https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip and unzip it. Remember to change `model_dir` according to your directory
```shell script
bert-serving-start -model_dir ./tmp/chinese_L-12_H-768_A-12 -num_worker=1
```
### Generator
* Change to directory of ai-recepit-art
* Download checkpoint from https://drive.google.com/drive/folders/1W42ZRVCr3o2I_xwUNZFY_AQp4juUSahR?usp=sharing, and move files to ai-recepit-art/ckpt
```shell script
virtualenv cgan
source cgan/bin/activate
pip install -r requirement
mkdir result
mkdir ckpt
python api.py
```
