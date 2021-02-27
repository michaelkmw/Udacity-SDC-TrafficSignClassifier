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

[image1]: ./output_images/visualization.jpg "Visualization"
[image2]: ./output_images/preprocess.jpg "Pre-processing"
[image3]: ./output_images/test.jpg "Test Image"
[image4]: ./output_images/softmax.jpg "Top 5 Softmax Prob"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/michaelkmw/Udacity-SDC-TrafficSignClassifier/blob/master/P3.ipynb)

A HTML version is also available [here](https://github.com/michaelkmw/Udacity-SDC-TrafficSignClassifier/blob/master/P3.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To visualize the training dataset, 5 random images are selected from the dataset for plotting. The title of each image displays the corresponding labels. In addition, histograms of the training, validation, and testing dataset are plotted to visualize the distribution of classes within the dataset. It is discovered that all 3 datasets contain higher distribution of classes 0-20. This could potentially prevent the neural network from training to recognize classes 20 or above. Potential future improvement would be to augment the dataset to include more representation of classes 20 or above.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It is rationalized that traffic signs can generally be identified by their shapes and graphic features. Therefore, one of the pre-processing steps is to convert the images from color to grayscale. By removing the color information, the size requirement of the neurons and the training dataset to effectively train the network is reduced.

In addition, as recommended by the lectures on gradient convergence, the datasets are normalized to have zero mean and equal variance. Example of images that have been pre-processed are displayed below:

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x16   |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 24x24x64   |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 24x24x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 12x12x64                  |
| Fully connected       | outputs 1024                                  |
| RELU                  |                                               |
| Dropout               | 0.5 dropout rate                              |
| Fully connected       | outputs 120                                   |
| RELU                  |                                               |
| Dropout               | 0.5 dropout rate                              |
| Fully connected       | outputs 43                                    |
| Softmax               |                                               |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, `tf.train.AdamOptimizer` is used to minimize the loss function, which is the `tf.reduce_mean` of the softmax cross entropy `tf.nn.softmax_cross_entropy_with_logits` of the model output.

The following are the hyperparameters of the model:

| Hyperparameter  | Value  |
|:---------------:|:----- :|
| epochs          | 100    |
| batch size      | 128    |
| learning rate   | 0.0001 |
| dropout rate    | 0.5    |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.965
* test set accuracy of 0.951

The first architecture chosen was the LeNet from lectures. With the LeNet, the validation set accuracy remain below 0.93 even after reducing the learning rate, increasing the epochs, and increasing the depths of the convolution layer.

Using the LeNet architecture as the baseline, experiements were conducted to modify the layers of the architecture. The first modification made was to add dropout of 50% to both fully connected layers to reduce overfitting, which resulted in a small improvement in the validation set accuracy.

Afterwards, a major focus was put on the effects of max pooling in the first 2 layers. Since the images of traffic signs contain more features than simple alphabets that the LeNet was designed for, and since the image resolution was poor (32x32px) to begin with, it is hypothesized that max pooling in the first 2 layers would remove too much resolution for training. There were two models tested:

| Model 1 Layer         |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x48   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 14x14x48                  |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x192  |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 10x10x192   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x192                   |

| Model 2 Layer         |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x16   |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 24x24x64   |
| RELU                  |                                               |
| Convolution 5x5       | 1x1 stride, same padding, outputs 24x24x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 12x12x64                  |

Both models provide roughly the same amount of parameters for training. Model 1 uses max pooling in the first layer, whereas Model 2 is devoid of pooling before the fully connected layers. With the same set of Hyper-parameters, it was noticed that Model 2 can converge to validation accuracy of 0.96 or above, where as Model 1 fails to achieve accuracy higher than 0.94.

Regarding the model's hyper-parameters, epochs was fixed to 100 for all baselines and experiments. Batch size was unchanged from the LeNet model from lecture, at 128. Learning rate was tuned to 0.0001 to help improve the validation set accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that was found on Google image search:

![alt text][image3]

All of these images are taken from different perspectives such that they don't contain direct front view of the traffic signs. They are also taken in different lighting conditions.

As will be discussed below, the 3rd image is actually not a German-style speed limit sign. The model actually failed to classify this sign. 

The 4th image belong to class 25, which is under-represented in the training, validation, and test dataset as discussed in the above.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| Priority road         | Priority road                                 |
| No entry              | No entry                                      |
| Speed limit (100km/h) | Speed limit (30km/h)                          |
| Road work             | Road work                                     |
| Stop                  | Stop                                          |
| Speed limit (100km/h) | Speed limit (30km/h)                          |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This is noticeably lower than the accuracy of the test set. A closer look reveals that the one image that fails is the first 100km/h speed limit sign. It was later found out that the speed limit sign does not correspond to the German-style 100km/h speed limt sign design. A correct image was added as the 6th test image, and the model was able to correctly classify the image. This implies that the trained model is unable to generalize the features of a speed limit sign to correctly classify differently-styled designs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 9th to 11th cells of the Ipython notebook.

For all 6 images, the model is absolutely certain about the classification of each iamge. This is evident in the  1st softmax probabilities being nearly 1.0. Since the trained model is absolutely certain about mis-classification of the 3rd image, which shows a non-German speed limit sign, it is suspected that the model might have overfitted the training data.

![alt text][image4]

A future improvement to be made to the model would be to experiment with a different architecture (such as the AlexNet) to see how the performance changes. Other regularization techniques such as 1x1 convolution and inception could be considered.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


