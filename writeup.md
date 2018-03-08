# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1a]: ./images/train_distribution.png "Training Set Class Distribution"
[image1b]: ./images/validation_distribution.png "Validation Set Class Distribution"
[image1c]: ./images/test_distribution.png "Testing Set Class Distribution"
[image2a]: ./images/with_color.png "With Color"
[image2b]: ./images/gray_scale.png "Grayscaling"
[image2c]: ./images/equalize_histogram.png "Histogram Equalization"
[image3]: ./images/augmentation.png "Random Noise"
[image4]: ./images/1.png "Traffic Sign 1"
[image5]: ./images/2.png "Traffic Sign 2"
[image6]: ./images/3.png "Traffic Sign 3"
[image7]: ./images/4.png "Traffic Sign 4"
[image8]: ./images/5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/freddycct/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the labels are distributed across the train, validation and test data sets. It is important to make sure that the distribution of labels in validation and test sets are almost similar.

![alt text][image1a]
![alt text][image1b]
![alt text][image1c]

I also randomly chose some images and visualize how it would look like after converting the images to grayscale. Some images are darker, so I apply histogram equalization in order to increase the contrast of the patterns in the images.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because most traffic signs can be identified without using the color information. It is also faster to train because reducing the images to grayscale reduces the amount of numerical values by 3 times.

Here is an example of a traffic sign image before and after grayscaling, follow by histogram equalization

![alt text][image2a]
![alt text][image2b]
![alt text][image2b]

As a last step, I normalized the image data because the deep neural network converge faster with more numerical stability if the range of values are consistent across the training examples.

I decided to generate additional data because I am not getting a higher test accuracy.

To add more data to the the data set, I rotate each image slightly, the random rotation is generated from a normal distribution of mean 0 and standard deviation of 15

Here is an example of an original image and an augmented image:

![alt text][image3]

However, the augmented images did not show any significant improvements for my results.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried Inception Model and LeNet. Inception Model did not give much improvement, so I stick to LeNet. The code for inception is also included in my notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32 x 32 x 1 Grayscale image   							| 
| Convolution 5 x 5 | 1 x 1 stride, valid padding, outputs 28 x 28 x 6  |
| Dropout |
| RELU					       |	
| Max pooling	2 x 2 | 1 x 1 stride, valid padding, outputs 27 x 27 x 6  |
| Convolution 5 x 5	| 1 x 1 stride, valid padding, outputs 23 x 23 x 16 |
| Dropout |
| RELU |
| Max pooling 2 x 2 | 1 x 1 stride, valid padding, outputs 22 x 22 x 16 |
| Flatten          | outputs 7744 |
| Fully connected		| outputs 120 |
| Dropout |
| RELU |
| Fully connected		| outputs 84 |
| Dropout |
| RELU |
| Fully connected		| outputs 43 |
| Softmax	| outputs 43 |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer, with learning rate of 0.001 for 200 epochs, with 1000 as batch size. All parameters have L2 regularization of 1e-5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.979 
* test set accuracy of 0.953

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? **I started with LeNet and I chose it because it was simple to train.**
* What were some problems with the initial architecture? **Probably overfitted to the training set.**
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. **Added dropout layers**
* Which parameters were tuned? How were they adjusted and why? **I tried various kernel sizes and changing number of filters, but did not improve my results significantly.**
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
**Convolution layer helps to detect the edges within the traffic signs. Dropout helps to generalize the model by making the network smaller because when outputs are randomly chosen to be 0, the architecture becomes simpler.**

If a well known architecture was chosen:
* What architecture was chosen? **I tried Inception Model**
* Why did you believe it would be relevant to the traffic sign application? **Well known model that is able to go deep while using less parameters than other well-known architecture.**
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? **Training error < Validation error < testing error, with training and validation error decreasing after every epoch.**
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because relatively similar images are not in the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| Children crossing     			| Children crossing								|
| Bicycles crossing					| Bicycles crossing											|
| Road work	      		| Road work	 					 				|
| Speed limit (80km/h)		| Stop     							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 49-50th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a general caution (probability of 0.998), and the image does contain a general caution. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.998 | General caution |
| 0.00143 | Traffic signals |
| 0.000414 | Speed limit (70km/h) |
| 0.000247 | Road narrows on the right |
| 0.00024 | Pedestrians |

For the second image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.991 | Children crossing |
| 0.00489 | Beware of ice/snow |
| 0.00236 | Bicycles crossing |
| 0.00151 | Slippery road |
| 0.000135 | Road narrows on the right |

For the third image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.996 | Bicycles crossing |
| 0.0015 | Children crossing |
| 0.000876 | Road narrows on the right |
| 0.000608 | Speed limit (60km/h) |
| 0.000312 | Road work |

For the fourth image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 | Road work |
| 7.04e-05 | Speed limit (60km/h) |
| 1.88e-05 | Road narrows on the right |
| 1.07e-05 | Ahead only |
| 8.38e-06 | Dangerous curve to the right |

For the fifth image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93 | Stop |
| 0.0318 | Keep right |
| 0.026 | Speed limit (60km/h) |
| 0.0111 | Speed limit (120km/h) |
| 0.000381 | No vehicles |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

