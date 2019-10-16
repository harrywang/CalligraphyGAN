# Art Generation based on Receipt Data

## Setup
You need to create 2 virtual environment to run this project, one for Bert Server, one for Generator.
### Setup Bert Server
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

### Setup Generation
* Change to directory of ai-recepit-art
* Download checkpoint from https://drive.google.com/drive/folders/1W42ZRVCr3o2I_xwUNZFY_AQp4juUSahR?usp=sharing, and move files to ai-recepit-art/ckpt

Setup the virtual environment and folders:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir result
mkdir ckpt
```

### Run the Demo
Make sure you are in the ai-recepit-art folder and virtual environment is activated and run `python demo.py`.

```
(venv) dami:ai-recepit-art harrywang$ python demo.py
2019-10-16 14:19:37.282856: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-16 14:19:37.314537: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f852fdcbd50 executing computations on platform Host. Devices:
2019-10-16 14:19:37.314559: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Enter the dish name: 西红柿鸡蛋
generating the character ......
resizing and denoising the character ......
applying style transfer ......
Done! Please check the files in the result folder
```

## Run and test the API
Run `python api.py` to start a Flask dev server to serve the API at localhost:5000, you can use Postman to test the API as follows:  
Add `form-data`(in `Body`)as showed in picture, and post.  
![image](https://i.ibb.co/b7rTtcp/postman.png)
* Finally you will get response with 2 fake urls, one for original image, one for stylized image.
* The result is stored in `ai-recepit-art/result`.  
* Here are some results:  
<div align=center><img width="200" height="200" src="https://i.ibb.co/DD3cjtY/result1.png" alt="original"/></div>
<div align=center><img width="200" height="200" src="https://i.ibb.co/pyr7NYB/result1-stylized.png" alt="stylized"/></div>
