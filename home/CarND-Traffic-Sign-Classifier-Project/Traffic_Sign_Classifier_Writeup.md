# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Traing_dataset.png "Training Visualization"
[image2]: ./images/Validation_dataset.png "Validation Visualization"
[image3]: ./images/Test_dataset.png "Test Visualization"
[image4]: ./images/color_sign.png "Original Dataset Image"
[image5]: ./images/gray_sign.png "Grayscale Dataset Image"
[image6]: ./images/gray3_sign.png "Grayscale 3 Channel Dataset Image"
[image7]: ./images/new_images/Sign1.jpg "Traffic Sign 1"
[image8]: ./images/new_images/Sign2.jpg "Traffic Sign 2"
[image9]: ./images/new_images/Sign3.jpg "Traffic Sign 3"
[image10]: ./images/new_images/Sign4.jpg "Traffic Sign 4"
[image11]: ./images/new_images/Sign5.jpg "Traffic Sign 5"
[image12]: ./images/new_images/new_image_snapshot.png "New Image Snapshot"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! [project code]./Traffic_Sign_Classifier.html

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the original dataset and combined it with a grayscale copy of the same image. This was used to inflate the dataset for less represented labels and improve training. I used the length of the arrays `len()` to get most of the details about the quantity of the dataset and `.shape` to get the image shape. 

The dataset details are below:

* The size of training set: 69598 (80.3%)
* The size of the validation set: 4410 (5.1%)
* The size of test set: 12630 (14.6%)
* The shape of a traffic sign image: (32, 32)
* The number of unique classes/labels in the data set: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the datasets are separated by labels and how many of each are present. By doubling the dataset with a color and grayscale image I was able to get more quanitity on a few labels that had small data samples.

![alt text][image1]
Training Dataset

![alt text][image2]
Validation Dataset

![alt text][image3]
Test Dataset

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I took the original training dataset and made a grayscale copy of them and combined the two sets to form larger samples of color and grayscale images but I had to make the grayscale image 3 channel to combine the sets. After having a split of color and grayscale I normalized the dataset.

I chose to create an additional grayscale dataset to suppliment the original without making a direct copy of it, in hopes that the Neural Net would pick up on different information present in both versions of the same image. Image normalization is to assist in the processing of the images for the Neural Net and ensure that the pixels have a similar data distribution.

Below I have examples of the original image, grayscale, and 3 channel grayscale:

![alt text][image4]
Original Image

![alt text][image5]
Grayscale Image

![alt text][image6]
3 Channel Grayscale Image

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

_________________________________________________________________________
| Layer           | Input    | Output   | Description (stride / padding) |
|=================|==========|==========|================================|
| Input           | --       | 32x32x3  | Dataset images                 |
| Convolution     | 32x32x3  | 30x30x12 | 1x1 / Valid                    |
| ReLU            | --       | --       | --                             |
| Dropout         | --       | --       | Training 80% / Test 100%       |
| Max Pooling     | 30x30x12 | 15x15x12 | 2x2                            |
| Convolution     | 15x15x12 | 14x14x24 | 1x1 / Valid                    |
| ReLU            | --       | --       | --                             |
| Dropout         | --       | --       | Training 80% / Test 100%       |
| Max Pooling     | 14x14x24 | 7x7x24   | 2x2                            |
| Convolution     | 7x7x24   | 6x6x36   | 1x1 / Valid                    |
| ReLU            | --       | --       | --                             |
| Dropout         | --       | --       | Training 80% / Test 100%       |
| Max Pooling     | 6x6x36   | 3x3x36   | 2x2                            |
| Flatten         | --       | --       | --                             |
| Fully Connected | 324      | 172      | --                             |
| ReLU            | --       | --       | --                             |
| Fully Connected | 172      | 86       | --                             |
| ReLU            | --       | --       | --                             |
| Fully Connected | 86       | 43       | 43 for each of the labels      |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a configuration of:

Optimizer: Adam Optimizer
Batch Size: 50
Number of Empochs: 25
Learning Rate: 0.00099

A lot of it was trial and error, a few things I noticed were the increase in Epochs generally led to greater accuracies, as did smaller batches. The learning rate seemed to be best leaving it at 0.00099 to make more steps to gradient descent.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of: 99.7%
* validation set accuracy of: 94.1% 
* test set accuracy of: 91.5%

I started my model based on the LeNet structure but added an extra Convolution Layer and added Dropouts to improve training. This was done with the intention of getting more defined details and reduced noise for the Neural Net. Performance also increased with the extra convolution layer. The LeNet architecture is a well known working model that is suitable for a project like this and with the information gained from the lessons this was the strongest candidate for basing it off of. Initially I was stuck with accuracies in the 80s and was not able to break the 90% mark and that prompted those adjustments. The accuracy fluctuates a fair amount each run but proves to be more consistent than the prior model.

I used the combination of an additional convolution layer and dropout to have greater chances of higher accuracies. By implementing dropout I was able to lower the chances of overfitting and the convolution layer was able to get more details from the images.

In the Jupyter Notebook the code cells can be located under Step 2: Train, Validate, and Test the Model. I have commented Train, Validate, and Test for the respective cells so that each step can be identified easily.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]
The first image is a speed limit sign and is very clear with high contrast with the colors. I would expect this to be one of the better predictions. The number could be mixed with several other traffic signs that are in a circular shape.

![alt text][image8]
Sign two is a speed bump sign again with high contrast but the speed bump could come off as a horizontal line due to the low resolution and definition in the image.

![alt text][image9]
Sign three is another speed limit sign in a circular shape which is similar to many other signs. This could be misclassified due to shape alone.

![alt text][image10]
Sign four is a circular shape but with a blue background which is not as common. The roundabout sign has three arrows that form another circle. Identifying the multiple circles and details within the sign might prove difficult.

![alt text][image11]
Sign five is a triangle shape with a snowflake in the middle. This is a common sign shape but the snowflake would be the unique element. Depending on the details picked up from the Neural Net this may be difficult to differentiate from other images found in the same shape.

![alt text][image12]
New Image Snapshot of images, shape, count, and label.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
________________________________________________________
| Image                     | Prediction               |
|===========================|==========================|
| (4)  Speed limit (70km/h) | (4)  Speed limit (70km/h)|
| (22) Bumpy Road           | (23) Slippery Road       |
| (1)  Speed limit (30km/h) | (2)  Speed limit (50km/h)|
| (40) Roundabout mandatory | (40) Roundabout mandatory|
| (30) Beware of ice/snow   | (2)  Speed limit (50km/h)|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The images were randomly sourced from the internet and very likely that they were very different from the training, validation, and test datasets making it harder for the Neural Network to work well in classifying the signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

___________________________________
| Sign 1 Probability | Prediction |
|====================|============|
| 14.6%              | 4          |
| 11.4%              | 1          |
|  9.2%              | 0          |
|  6.1%              | 2          |
|  1.0%              | 40         |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
___________________________________
| Sign 2 Probability | Prediction |
|====================|============|
|  5.7%              | 40         |
|  4.3%              | 16         |
|  3.6%              | 2          |
|  3.1%              | 5          |
|  1.8%              | 38         |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
___________________________________
| Sign 3 Probability | Prediction |
|====================|============|
|  4.6%              | 2          |
|  1.5%              | 38         |
| -0.1%              | 1          |
| -0.4%              | 40         |
| -0.7%              | 10         |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
___________________________________
| Sign 4 Probability | Prediction |
|====================|============|
| 12.4%              | 2          |
|  8.7%              | 1          |
|  6.8%              | 40         |
|  4.3%              | 5          |
|  3.2%              | 37         |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
___________________________________
| Sign 5 Probability | Prediction |
|====================|============|
|  8.3%              | 23         |
|  7.5%              | 22         |
|  3.7%              | 30         |
|  2.5%              | 31         |
|  2.1%              | 25         |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


