# CarND-Behavioral-Cloning-P3

The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


My project includes the following files:
- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
``python drive.py model.h5``  
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works

# Model Architecture and Training Strategy
I started with the basic NN similar to the lessons and got the basic setup working. Then moved to LENET before moving to NVIDIA model shown below.  

![model.png](https://github.com/abhisheksreesaila/CarND-Behavioral-Cloning-P3/raw/master/model.png)

The first layer of the network performs image normalization followed by cropping layer which takes out the sky and car section in every image. The convolutional layers are designed to perform feature extraction. I use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a (1,1) strided convolution with a 3×3 kernel size in the final two convolutional layers followed by three fully connected layers, leading to a final output control value. 

### Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting which is added after the first 5 convolutions layers. Data Augmentation was also performed by flipping the images.
### Model parameter tuning
The model used an Adam optimizer, so the learning rate was not tuned manually. 3 EPOCHS was sufficient to delivers good results (2% loss). 
### Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used all the 3 cameras images to ensure car learns from all 3 angles and hence drives safely. Also, I used the sample data provided in the GitHub repo for this project. I checked the images in a slide show to ensure we have a good set and was not guessing.  

# Evaluate the model
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set and on the validation set. I was convinced it was good enough to be taken on-road.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
