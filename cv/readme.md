# Oil painting
## Setup
* Create virtual environment
```shell script
python3 -m venv venv
source venv/bin/activate
```
* Install OpenCV and threadpool
```shell script
pip install opencv-python, threadpool
```
## Test
Run
```shell script
python painterfun.py
```
to see the oil painting of `flower.jpg` in folder `flower`. And program will generate results for every epoch.
## Parameters
You can change the parameters of function `load` and `r` to change the original image or the other affect.
* Parameter for `load` function is the path of the original image, change it according to your need.
* `r` function is the main function for generator, and it has 3 paramters.
    1. **epoch**  
    How many times will the program run.
    2. **target_color**  
    Set the target color of the generated image. It should be a BGR color list and value for each channel should be between 0 and 1.  
    For example, it can be `[[0, 1, 1], [0.6, 1, 0.5]]`. And the program will randomly pick colors to generate strokes.  
    By default it it `None`.
    3. **random_color**  
    When `target_color` is `None`, program will use original color of image or pick colors randomly if this parameter is set to `True`.
