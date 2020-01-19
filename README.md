# Art Generation based on Receipt Data

## Introduction
This is the implementation of *Taming Generative Modeling and Natural Language Processing for Creative Customer Engagement*.  
  
This is a creative framework based on Conditional Generative Adversarial Networks and Contextual Neural Language Model to generate artworks that have intrinsic meaning and aesthetic value.  
  
Input a dish name or description in Chinese, and we can get a image representing these Chinese characters.  
  
The whole framework is composed of 3 parts -- **Bert**, **CGAN** and **Oil Painting**.

### Dataset
We use 100 Chinese characters to test our framework.  
  
Examples from the dataset (Chinese character 好):  
![dataset_sample](https://i.ibb.co/HBNy5T7/dataset-sample.png)
### Bert
>Inspired by https://github.com/hanxiao/bert-as-service  

In this part, we developed a simple algorithm based on BERT to map the input text with arbitrary number of characters into five characters from the 100 characters.
### CGAN
In this part, we use 100 Chinese characters as training data to train a generator.  
This generator take a 100-dimensional vector as input, 
and each dimension in this vector represents the weight of each Chinese character in the data set.
### Oil Painting
>Inspired by https://github.com/ctmakro/opencv_playground  

In this part, we convert generated image into oil painting.


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
Sample outputs. They are GAN result, result after noise reduction, result after oil painting,
result after style transfer and style image respectively.
![1579404347_final](https://i.ibb.co/ZLr8xyt/1579404347-final.jpg)

## Run and test the API
Run `python api.py` to start a Flask dev server to serve the API at localhost:5000, you can use Postman to test the API as follows:  
Add `form-data`(in `Body`)as showed in picture, and post.  
![api_test](https://i.ibb.co/qsxrrsZ/api-test.png)
* Finally you will get response with 2 fake urls, one for original image, one for stylized image.
* The result is stored in `ai-recepit-art/result`.  
* Here are some results:  
<div align=center><img width="200" height="200" src="https://i.ibb.co/dmRVP9m/1579412691-convert.png" alt="original"/></div>
<div align=center><img width="200" height="200" src="https://i.ibb.co/KX0HbW5/1579412691-stylized.png" alt="stylized"/></div>

## Web Demo
Besides API, you can also run generator as web demo.
* Make sure Bert is running, you are in directory of ai-recepit-art, you have installed all the requirements and you have downloaded checkpoints.  
```shell script
mkdir static
python gui_demo.py
```
Now you can visit localhost:5000 to enjoy the magic of generator.  
<div align=center><img width="500" src="https://i.ibb.co/p4MYWKZ/web-demo.png" /></div>
