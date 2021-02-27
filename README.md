# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This project builds deep convolutional neural network with Tensorflow to classify German traffic signs from the [German Traffic Sign Benchmark Website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

The pipeline consists of the following:
- Image preprocessing through grayscale and normalization
- Deep convolutional neural network with layers of convolution and ReLU activation and application of max pooling and dropout regularization
- Optimization through Adam Optimizer on the softmax cross entropy of the neural network output
- Training, validation, and testing pipeline utilizing mini-batching
- Testing of the pipeline on new images of German traffic signs found on Google image search
 
The code, test images, and test videos originated from [CarND-Traffic-Sign-Classifier-Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)

Installation
---

To run this project, follow the setup guideline in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)

**NOTE:** If Miniconda is used, install Jupyter Notebook before creating the `carnd-term1` environment

```
conda install jupyter notebook
```

Usage
---

For a quickview of the results, open `P3.html` in the repository

**NOTE:** Activate the `carnd-term1` environment before opening the notebook

```
activate carnd-term1
```

Alternatively, change the environment in the notebook through `Kernel -> Change kernel`

This project is coded using Jupyter Notebook. There are two ways to run the notebook:

1. Open Jupyter Notebook, navigate to the repository directory, and open `P3.ipynb`

2. Open Terminal/Anaconda prompt, navigate to the repository directory, enter the following command, and open `P3.ipynb`

```
jupyter notebook
```

Once the notebook is opened, the cells in the notebook can be run to detect lane lines on the roads in test images and videos.

Images `test_images` folder and videos from main folder will be processed, and the results are saved to `output_images` and main folder, respectively.