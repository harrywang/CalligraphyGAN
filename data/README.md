# Dataset

## 100 Chinese Characters

We use 100 Chinese characters to test our framework.  

Examples from the dataset (Chinese character 好):  
![dataset_sample](https://i.ibb.co/HBNy5T7/dataset-sample.png)

## Chenzhongjian Calligraphy Dataset

From http://163.20.160.14/, we crawled hundreds of thousands of Chinese characters pictures written by different calligraphers.  
They are all packaged as hdf5 files.  
>
>link for data: https://mega.nz/file/oG4ikCoR  
>
### Brief Information 
Examples from the dataset (Chinese character 阿):  
![dataset_sample](https://i.ibb.co/ZW3SKsc/data-sample.png)  
Total number of images: 350656  
Total number of writers: 48  
Total number of characters: 7721  
Image size: 140×140  
Characters with more than 100 images: 173  
Characters with more than 50 images: 3917  
Characters with more than 30 images: 6054  

The EDA of dataset is in `eda.ipynb`.  

We encode the characters and writers with int, as shown in the csv files `label_character_all.csv` and `label_writer_all.csv`.  
In the hdf5 file, each line represents a image(a numpy array with size of `(140, 140)`) and its labels(encoded results for characters and writers).
