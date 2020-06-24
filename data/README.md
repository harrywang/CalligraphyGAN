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

### Chenzhongjian Dataset Preprocess

This directory contains the code for preprocessing Chenzhongjian Dataset.

#### Package Data

1. Download original Chenzhongjian Dataset from `https://mega.nz/file/NGgV2QYQ`.
2. Unzip data into `./data_process/chenzhongjian`.
3. Run `embedding.py` and it will generate csv files of labels.
>```shell script
>python embedding.py --data_dir ./chenzhongjian --output_dir ./
>```
4. Run `write_data.py` and it will generate a new hdf5 file named `chenzhongjian_all.hdf5` in output directory.
>```shell script
>python write_data.py --data_dir ./chenzhongjian --label_dir ./ --output_dir ./
>```

#### How It Works

1. Encode characters and writers using integer. We convert traditional characters used for labels into simplified characters.
2. Convert images into 140×140.
3. Write images, corresponding Chinese characters and corresponding writers to the hdf5 file as rows.  


#### Load Data

Read data from the hdf5 file using function `data_loader` defined in `write_data.py`.
```python
from write_data import get_embedding, data_loader

data_path = './chenzhongjian_all.hdf5'
writer_csv = './label_writer_all.csv'
character_csv = './label_character_all.csv'
data_images, data_character, data_writer = data_loader(data_path)
characters, writers = get_embedding(character_csv, writer_csv)
```
Check `./eda.ipynb` for detailed usage example.
