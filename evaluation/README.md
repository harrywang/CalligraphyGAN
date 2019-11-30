# Evaluation for CGAN-Calligraphy
## Evaluate with FID
* Generate images for evluation using `generate_target_word` or `generate_target_vector` in evalution/util
```python
generate_target_word('且', checkpoint_dir='../', result_dir='./且', size=5)
```
* Calculate FID Matrix for word_list
---
There is an example - `evaluation/fid_calculation.py`.  
Run this file by, notice that you should in the folder `evaluation`
Run this file by, notice that you should in the folder `evaluation`
```shell script
python fid_calculation.py
```
This will generate results for 5 words and calculate the matrix.

## Evaluation with white space
Calculate the proportion of white parts in the image.  
**How it works:**   
If the grayscale value of a pixel is equal or more than threshold, 
it will be considered as white.  
**Test:**  
Run function `white_space_ratio` to get the result of a image.
```python
white_space_ratio(image_path='./example_image_path', threshold=240)
```
