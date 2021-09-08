# ImageSegmentation
Performing image segmentation using a gaussian mixture model. 

**University:** Athens University of Economics and Business  
**Department:** Informatics  
**Subject:** Machine Learning

**Writer:**  Andreas Gouletas (@BrainBroader)

## Table of Contents
* [Description](#description)
* [Datasets](#datasets)
* [Technologies](#technologies)
* [Prerequisites](#prerequisities)
* [Execution Instructions](#execution-instructions)

### Description 
It is an implementation of a gaussian mixture model that performs image segmentation in 3 dimensional input data. The model uses the Expectation Maximization algorithm 
for training.

### Datasets
 Any image(RGB). 

### Technologies

The technologies used that are worth mentioning, are:

   * Python
   * Numpy
   * Matplotlib


### Prerequisities

Before you execute the given program, you need to:

    1.get an image of your choice.
    2.check if you have installed the libraries mention in Section "Technologies".

If you haven't previously installed the libraries mentioned above, you can use the provided requirements.txt file, by running the following command:

cd path-to-project
pip install -r requirements.txt
 
### Execution Instructions
To execute the program the following command is used:
```
python main.py arg1 arg2
```
where 

* arg1 is the path to the image.
* arg2 is the number of image segments you want (It must be an integer).


Running example, 
```
python main.py img.jpg 2
```

