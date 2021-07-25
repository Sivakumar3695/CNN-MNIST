# CNN-MNIST
A vectorized implementation of CNN for MNIST digit recognition. This CNN project is written from the scratch and is not based on any libraries. The vectorization of CNN convolution operation is based on this paper: http://lxu.me/mypapers/vcnn_aaai15.pdf

For those unfamiliar with the MNIST dataset, *Training dataset* refers to 60000 handwritten data in MNIST and *Test dataset* refers to 10000 handwritten data in MNIST

Major operations in this projects are as follows:
1. MNIST digit recognition training
2. Recognizing custom handwritten digit images
3. Feature Visualization 
    - layer and channel wise
    - for the given input image
4. Evaluating test data with a pre-trained model. 

*NOTE:* When a new model is trained, its weights will be stored in a new .npz file. The file name will be in the following format: weight_details{{model_str}}.npz
    for example: weight_detailsC(16,3)-P2-C(32,3)-P2-C(64,3)-P2-FC512-FC10.npz
    
Once the training for a new model is done, the following details will be provided:
1. Accuracy
2. Prediction
3. Loss at the end of training
4. Time taken for the process to complete.

Following packages are prerequisites to run this project:
1. matplotlib - to generate Means Loss Vs Training Epoch graph, Mean activation value of every channel in a given layer for feature visualization in bar graph formay
2. keras - to load MNIST training and test dataset
3. numpy - to carry out vectorised implementation.
4. opencv - to process custom images provided for digit recognition, to generate feature visualization images based on the Numpy vector data
5. scipy - scipy.ndimage is needed to calculate the centre of mass for a custom handwritten digit to align the digit in the same way MNIST dataset handles. 

   Important details from MNIST dataset: "
        1. All images are size normalized to fit in a 20x20 pixel box
        2. Digits are centered in a 28x28 image using the center of mass
        3. Digits are written in white color in black background" 
        
Based on this information, the user provided input images are altered to make them predictable by the pre-trained model.

# Training a model
To keep things simple, 3 pre-trained models are already available in the repo. 

The model training details of these models are given below:
----------------------------------------------------
Model : C(8,5)-P2-C(16,3)-P2-C(32,3)-P2-FC256-FC10
Final Loss: 0.033796041456661571984
Accuracy: 99.11%
Precision: 95.56%
Total time taken: ~ 28 minutes
----------------------------------------------------
Model : C(16,3)-P2-C(32,3)-P2-C(64,3)-P2-FC512-FC10
Final Loss: 0.029360860885127495471
Accuracy: 99.20%
Precision: 96.01%
Total time taken: ~ 86 minutes
----------------------------------------------------
Model : C(8,5)-P2-C(16,3)-P2-FC128-FC10
Final Loss: 0.029360860885127495471
Accuracy: 98.13%
Precision: 94.37%
Total time taken: ~ 25 minutes
----------------------------------------------------

All these model training are done in CPU and no GPU is involved in these trainings.
For a new model training, the following details are obtained from the user:
1. Model string 
   e.g: C(8,5)-P2-C(16,3)-P2-FC128-FC10
     C(8,5) = Convolution Layer with 8 Channels of 5x5 filter
     P2 = Max Pooling Layer with 2x2 max pooling filter
     FC128 = Fully connected layer with 128 units
2. Learning rate (e.g: 0.0001)
3. Training epoch (e.g: 5) => Number of times 60000 training dataset has to be iterated forward and backward for model training.
![Screenshot from 2021-07-24 21-15-41](https://user-images.githubusercontent.com/29046579/126873841-fea67074-90c5-42ca-9e59-d439dfb39e3f.png)

Any new model can be trained and the weights of such model will be store in the .npz file inside the folder ***./weight_output/*** with the file name ***weight_details{{model_str}}.npz***

# Predict custom handwritten digits
As mentioned above, the custom digit images are transformed based in the MNIST information.

NOTE: User image shown here is reduced to 28x28 px to properly fit it in this page. To download original image, please click [here](https://user-images.githubusercontent.com/29046579/126874071-2d64a689-12b6-4423-b499-c6454b2152a0.jpg)

User Image (original resolution of the image:720 x 885 px): <img src="https://user-images.githubusercontent.com/29046579/126874071-2d64a689-12b6-4423-b499-c6454b2152a0.jpg" width="28" height="28">

Transformed Image: ![1](https://user-images.githubusercontent.com/29046579/126874100-76d2fad0-6764-4a63-9a3c-68fec8011fe5.jpg)

Prediction with C(16,3)-P2-C(32,3)-P2-C(64,3)-P2-FC512-FC10: Actual Number:1.jpg, Predicted:1 with 99.72% confidence

All the custom images should be put inside the folder ***custom_test_images***. All the images in this folder will be preprocessed and provided to the pre-trained CNN model for prediction.

![Screenshot from 2021-07-24 21-52-49](https://user-images.githubusercontent.com/29046579/126874816-482c2296-89d3-4972-b3f0-6ba577c1c78d.png)

*NOTE:* The dark handwritten digits on pure white background will result in better prediction.

# Feature Visualization
Feature Visualization by optimization is a technique to make a neural network models interpretable by humans. It is a growing area in deep learning as many fields like medical imaging require model decision interpretability to understand why and how the model came to a conclusion and makes predictions.
The implementation of feature visualization is based on the knowledge gained from: https://distill.pub/2017/feature-visualization/

## Feature visualization - Layer and channel wise:
To generate an image (from scratch) that excites a given layer and channel to maximum level. The layer and channel will be obtained from the user.
The generated output will be avialable in the folder ***./feature_visualize/output/*** with the file name: ***feature_visualize_1.jpg***

![Screenshot from 2021-07-25 08-07-05](https://user-images.githubusercontent.com/29046579/126885826-c6cd82bd-4518-4510-a42c-82edce4023e0.png)


## Feature visualization - for a given image:
To generate a collaged image of the given image (in MNIST image format - black background + white digit) with 3x3 grid and feature visualization images of top 5 channel/units in a given layer that has maximum mean activation value for the given input image.
Before proceeding with the feature visualization for a given image, please place the input image in the folder ***./feature_visualize/*** with the file name ***input.jpg***. 

The following outputs will be generated in the folder ***./feature_visualize/output/***
1. Top 5 feature visualize image. The file names will be in the format ***feature_visualize_{{NUMBER}}.jpg***
2. ***mean_activation_bar.jpg*** => a bar graph showing the mean activation data of every channel in a given layer
3. ***collaged_feature_visualize.jpg*** =>  collage of the input images with 3x3 grid (color of the digit will be grayed to make comparison) and the top 5 feature visualizing image.

![Screenshot from 2021-07-25 08-23-55](https://user-images.githubusercontent.com/29046579/126886128-e032a031-b5c1-470b-9a2f-9b6eb724792d.png)

To know more about feature visualization in this project, please click [here](https://github.com/Sivakumar3695/CNN-MNIST/wiki/Feature-Visualization)

# Evaluation of test data with the pre-trained model
Any pre-trained mode can be used to evaluate the test data at any time. The precision and accuracy details will be computed for the pre-trained model. This feature can be used to know the ability of any pre-trained model in the digit recognition process.

![Screenshot from 2021-07-25 08-25-19](https://user-images.githubusercontent.com/29046579/126886156-5ff2f3ce-6b43-46e4-a8d0-9ee8904f37ed.png)

Based on the inputs provided in the above image, test data will be evaluated for the model C(16,3)-P2-C(32,3)-P2-C(64,3)-P2-FC512-FC10
