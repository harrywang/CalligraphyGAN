# Abstract Art via CalligraphyGAN  

[demo](http://54.223.165.220:8501)  

## Introduction
This is the implementation of *A Framework and Dataset for Abstract Art Generation via CalligraphyGAN*, 
for [NeurIPS Workshop, Machine Learning for Creativity and Design](https://neurips2020creativity.github.io/).  

This is a creative framework based on Conditional Generative Adversarial Networks and Contextual Neural Language Model to generate artworks that have intrinsic meaning and aesthetic value.

Input a dish name or description in Chinese, and we can get a image representing these Chinese characters.  

The whole framework is composed of 3 parts -- **Bert**, **CalligraphyGAN** and **Oil Painting**.

### Dataset

More information for dataset on [Chinese Calligraphy Dataset](https://github.com/zhuojg/chinese-calligraphy-dataset).  
We use 1000 characters in this dataset to test our framework.

### Bert
>adapted based on https://github.com/huggingface/transformers  

In this part, we developed a simple algorithm based on BERT to map the input text with arbitrary number of characters into five characters from the 100 characters.

### CalligraphyGAN
In this part, we use 1000 Chinese characters as training data to train a generator.  
This generator take a 1000-dimensional vector as input,
and each dimension in this vector represents the weight of each Chinese character in the data set.

### Oil Painting
>adapted based on by https://github.com/ctmakro/opencv_playground  

In this part, we convert generated image into oil painting.

## Web Demo  

We use [Streamlit](https://www.streamlit.io/) to build a demo to show our model, and we have deployed it [here](http://54.223.165.220:8501).

<div align=center><img width="500" src="images/web_demo.png" /></div>

You can also run demo by yourself following the instruction below:

### Using Docker  

We have packaged our web demo using Docker, just make sure you have Docker installed correctly.  
```shell script
docker run -d -p 8501:8501 zhuojg1519/calligraphy.ai
```

Then visit `http://localhost:8501` to enjoy the magic.  

### Setup on Local  

- Change to directory of `calligraphy.ai`  
- Download checkpoint from https://drive.google.com/drive/folders/1W42ZRVCr3o2I_xwUNZFY_AQp4juUSahR?usp=sharing, 
and move files to `calligraphy.ai/ckpt`  

- Setup the virtual environment and folders  
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir result
mkdir ckpt
```

- Then run
```shell script
streamlit run st_demo.py
```
- Now you can visit `localhost:8501` to enjoy it.
