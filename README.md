# Art Generation based on Receipt Data

## Setup
You need to create 2 virtual environment to run this project, one for Bert Server, one for Generator.
### Bert
* change to directory of Bert.
```shell script
python3 -m venv venv
source venv/bin/activate
pip install tensorflow==1.14.0
pip install bert-serving-server
```
* Download checkpoint from https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip and unzip it. Remember to change `model_dir` according to your directory
```shell script
bert-serving-start -model_dir ./tmp/chinese_L-12_H-768_A-12 -num_worker=1
```
you should see the following message:

```
I:WORKER-0:[__i:gen:559]:ready and listening!
I:VENTILATOR:[__i:_ru:164]:all set, ready to serve request!
```

### Generator - API
* Change to directory of ai-recepit-art
* Download checkpoint from https://drive.google.com/drive/folders/1W42ZRVCr3o2I_xwUNZFY_AQp4juUSahR?usp=sharing, and move files to ai-recepit-art/ckpt

```shell script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir result
mkdir ckpt
python api.py
```

#### Test on API
If anything goes right, now this API is served at localhost:5000, you can use Postman to test the API as follows:  
Add `form-data`(in `Body`)as showed in picture, and post.  
![image](https://i.ibb.co/b7rTtcp/postman.png)
* Finally you will get response with 2 fake urls, one for original image, one for stylized image.
* The result is stored in `ai-recepit-art/result`.  
* Here are some result:  
<div align=center><img width="200" height="200" src="https://i.ibb.co/DD3cjtY/result1.png" alt="original"/></div>
<div align=center><img width="200" height="200" src="https://i.ibb.co/pyr7NYB/result1-stylized.png" alt="stylized"/></div>

### Generator - Web Demo
Besides API, you can also run generator as web demo.
* Make sure Bert is running, you are in directory of ai-recepit-art, you have installed all the requirements and you have downloaded checkpoints.  
```shell script
mkdir static
python gui_demo.py
```
Now you can visit localhost:5000 to enjoy the magic of generator.  
<div align=center><img width="500" src="https://i.ibb.co/jM2zw62/2019-10-21-230813.png" /></div>