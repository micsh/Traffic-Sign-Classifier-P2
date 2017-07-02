# Traffic-Sign-Classifier-P2

### Overview

This is the project submission for Udacity's Self Driving Car Nano Degree: **'*Project: Build a Traffic Sign Recognition Program*'**.

In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train a model so it can decode traffic signs from natural images by using the German Traffic Sign Dataset. After the model is trained, we will then test our model program on new images of traffic signs from the web.

The implementation uses **TensorFlow**.

The full solution is provided in the [Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb) notebook.

An HTML version of the notebook: [Traffic_Sign_Classifier.html](Traffic_Sign_Classifier.html).

## Project: *Build a Traffic Sign Recognition Program*

---

The goals / steps of this project are the following:
* Load the data-set
* Explore, summarize and visualize the data-set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[rsample]: ./images/rsample.jpg "random sample"
[distribution]: ./images/distribution.jpg "distribution"
[wresults]: ./images/web_results.jpg "web results"
[wresultsp]: ./images/web_results_pred.jpg "web results pred"
[visualization]: ./images/visualization.jpg "visualization"


[rn]: ./images/100_1607_small.jpg "right of way"
[rw1]: ./images/6432962a1f.jpg "road work"
[rw2]: ./images/images.jpg "road work"
[pr]: ./images/Arterial_small.jpg "priority road"
[ne]: ./images/Do-Not-Enter_small.jpg "no entry"
[gc]: ./images/Radfahrer-Absteigen_small.jpg "general caution"
[st1]: ./images/Stop-sign_small.jpg "stop"
[60]: ./images/ce4d143c7d.jpg "60"
[y]: ./images/cfe15da5b3.jpg "yield"
[st2]: ./images/dc3f68f38a.jpg "stop"


#### Loading the data

Unzip the data and load it with pickle, there are 3 files to load:

train.p
valid.p
test.p


```python
import pickle

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
```

#### Data Set Summary & Exploration

We have a total of 51,839 sample, devided into 34,799 for training, 4,410 for validation and 12,630 for a final test.

```python
n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
```

The images in this dataset have been resized to 32x32, giving us the shape of (32, 32, 3) with the color channels.

```python
image_shape = X_train[0].shape
```

There are a total of 43 categories for the traffic signs.

```python
import numpy as np

n_classes = np.sum(np.bincount(y_train) > 0)
```

#### Visualization of the dataset.

I have drawn a random sample from each category. First thing that is obvious is that many images are very low quality and would be hard to categorize even for a human.

![random sample][rsample]

Looking at the distribution of the various categories, it's clear that not all categories have an equal representation.

![distribution][distribution]

#### Design and Test a Model Architecture

I have started with a simple approach, using the LeNet model and only normalizing the input from [0..255] to [0..1].

##### I have decided to see how far I can go with just the model, and no data augmentation or pre-processing.
The thinking here is to be able to better understand how well does the model learn. The quality of the data-set is obviously paramount for allowing the learning to happen, but I wanted to start with a benchmark, and therefore no helping the model! 
From the point of view of a data scientist, having a good toolset for data augmentation is invaluable to solve a given task. We would also very much like to be able to learn from a few examples as possible (they are expensive), and data augmentation is one (mechanical) way to achieve that.
But I wanted to start with the simplest implementation and work from there; turns out we can get a pretty good score, even without data augmentation.

##### My final model has the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x36	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36    				|
| Fully Connected	    | outputs 800   								|
| Dropout	            | keep probabilty 0.5  							|
| Fully Connected	    | outputs 400   								|
| Dropout	            | keep probabilty 0.5							|
| Fully connected		| outputs 43    								|
|						|												|
 


#### To train the model, I used the Adam optimizer with a learning rate of 0.0009, batch size of 256 for 55 epochs.
* I have removed the bias terms from the model.
* I have increased the batch size, thinking it might help the model get a big enough sample to get good gradients, after all, there are 43 different categories to classify and they aren't distributed uniformly.
* I have not used early stopping (the validation accuracy peaked 98% on epoch 39).
* I am using dropout, after both fully-connected layers, to help the model generalize and not overfit.

#### My final model results are:
* validation set accuracy of 97.1%
* test set accuracy of 95.4%

I have decided to start from the LeNet model, partially because it was readily available, and partially because the data-set input fits the model. I didn't want to go to the newest models (such as ResNet) as I thought that was an overkill. 

I have played my own gradient descent, and tweaked the learning rate, batch size, layers, activations, and more, to see what works, and what doesn't.

### Test a Model on New Images

#### I have searched the web for traffic sign images, and found the following 10 images:

![road work][rw1] ![road work][rw2] ![60][60] ![general caution][gc] ![priority road][pr]
![no entry][ne] ![stop][st1] ![stop][st2] ![aright of way][rn] ![yield][y]

#### Here are the results of the prediction:

![web results][wresults]

The model only gets wrong the 'general caution' sign and mistakes it to 'no passing'.

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. 

#### Looking at the softmax values to asses how certain the model is

![web results pred][wresultsp]

The model is pretty confident about all of them (lowest proababily goes to the stop sign), but gets one of them wrong. This a very small sample, so we should take the accuracy on them with a grain of salt.

### Visualizing the Neural Network 

One of the visualizations that helped me (with tweaking the model too) was the activation of the first layer. Here's an example:

input image:

![60][60]

first layer activation:

![visualization][visualization]
