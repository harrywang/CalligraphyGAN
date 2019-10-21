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
initializing ......
2019-10-16 16:06:54.619176: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-16 16:06:54.645629: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7ff14eaea0a0 executing computations on platform Host. Devices:
2019-10-16 16:06:54.645649: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Enter dish name - [default:红烧肉卤鸡蛋]:
generating the character...
resizing and denoising the character...
[?] Choose a style:: Rousseau
   Picasso
   Pollock
  Rousseau
   Rothko
   deKooning

applying style transfer...
Finishing up...
```
Sample outputs:
![1571256428_final](https://user-images.githubusercontent.com/595772/66955036-970f4400-f02f-11e9-85b2-16e2d018aeab.jpg)
![1571256477_final](https://user-images.githubusercontent.com/595772/66955039-98407100-f02f-11e9-8469-83b93f6c9164.jpg)

## Run and test the API
Run `python api.py` to start a Flask dev server to serve the API at localhost:5000, you can use Postman to test the API as follows:  
Add `form-data`(in `Body`)as showed in picture, and post.  
![image](https://i.ibb.co/b7rTtcp/postman.png)
* Finally you will get response with 2 fake urls, one for original image, one for stylized image.
* The result is stored in `ai-recepit-art/result`.  
* Here are some results:  
<div align=center><img width="200" height="200" src="https://i.ibb.co/DD3cjtY/result1.png" alt="original"/></div>
<div align=center><img width="200" height="200" src="https://i.ibb.co/pyr7NYB/result1-stylized.png" alt="stylized"/></div>

## Web Demo
Besides API, you can also run generator as web demo.
* Make sure Bert is running, you are in directory of ai-recepit-art, you have installed all the requirements and you have downloaded checkpoints.  
```shell script
mkdir static
python gui_demo.py
```
Now you can visit localhost:5000 to enjoy the magic of generator.  
<div align=center><img width="500" src="https://i.ibb.co/jM2zw62/2019-10-21-230813.png" /></div>
