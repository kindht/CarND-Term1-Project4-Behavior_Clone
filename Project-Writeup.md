# **Behavioral Cloning** 

## Writeup

---

**Use Deep Learning to Clone Driving Behavior**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0.0]: ./Records/history_run/0.0-Run-Basic.png "Basic"

[image0.1]: ./Records/history_run/0.1-Run-Good.png "Run Good"
[image0.2]: ./Records/history_run/0.2-Run-Good.png "Run Good"
[image0.3]: ./Records/history_run/0.3-Run-Good.png "Run Good"
[image0.4]: ./Records/history_run/0.4-Run-Good.png "Run Good"
[image0.5]: ./Records/history_run/0.5-Run-Good.png "Run Good"

[image0.00]: ./Records/history_run/1.Run-Bad.png "Bad LeNet"

[image1.0]: ./Records/history_loss/0.history_pilot_overfit.png "Over"
[image1.1]: ./Records/history_loss/1.history_loss_Final.png "Final loss"
[image1.2]: ./Records/history_loss/1.history_plot_Final.png "Final plot"


[image2.1]: ./Records/images/0.data-land01.png "Curve 1"
[image2.2]: ./Records/images/0.data-land02.png "Curve 1"
[image2.3]: ./Records/images/3.data-water.png  "Curve water"
[image2.4]: ./Records/images/4.data-hard.png  "Hard"

[image3.center]: ./Records/images/img_center.png "image"
[image3.left]: ./Records/images/img_left.png "image"
[image3.right]: ./Records/images/img_right.png "image"
[image3.flip]: ./Records/images/img_flipped.png "image"
[image3.crop]: ./Records/images/img_crop.png "image"

[image4.data]: ./Records/images/driving_log-my-data.png "data"

[image5.history-loss]: ./Records/history_loss/2.history_plot_water.png 
[image5.water-bad]: ./Records/history_run/2.Run-Water-Bad.jpg 

[image6.Network]: ./Records/images/Network.png 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Project4-Writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy in Summary

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network, which includes **5 convolutional layers** , where first 3 convolutional layers have **5x5** filter sizes with strides of **2** and depths of 24, 36 and 48 respectively, followed by 2 convolutional layers with **3x3** filter sizes and strides of **1**, depths of 64 , followed by **5 fully connected layers**,the final output layer has **1 node for predition**(model.py lines 109-125) 

 The first input layer of the model is a **Cropping layer** (keras Cropping2D layer) to crop images to desired size, then the data is normalized and mean centered around zero using a Keras **Lambda layer** (code line 90). The model also includes **ReLu** layers to introduce nonlinearity

#### 2. Attempts to reduce overfitting in the model

The model contains **Dropout layers** in order to reduce overfitting (model.py lines 117,119,121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-81, 137-141). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 136).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a **combination of images** captured by simulator from 3 cameras mounted on the car: the **center, left and right** camera. In this way, I not only have center lane driving images, but also can simulate the vehicle being further off the center line. 

I also collected more driving data along **a couple of curves** where the vehicle is prone to drift off to the side.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy in Details

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to **start from a simple model**, then **build a more advanced model** and finally **finetune the model untill to obtain ideal results**.

My **first step** was to build a very basic model (```build_Basic_Model()``` function in model.py line 92-95).

With this basic model, I used a small data set to **go through the whole training and testing process**, including how to load data, how to create training samples, how to compile, train and save the model, finally how to run the simulator to see how well the car was driving around track one. 

![alt text][image0.0]

Then I built a model with the classic **LeNet** architecture. (```build_LeNet_Model``` function in model.py line 98-107). I used this model to train the **sample data** provided by the project, and the vehicle could move forward well for some while before it ran out of the track.


**Finally** I implemented a more powerful **convolution neural network model** similar to [PilotNet](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture published by Nvidia. (```buid_PilotNet_Model()``` model.py line 109-125).  I thought this model might be appropriate because it was specifically designed for the purpose of automatic driving. The model creates **a deep neural network** that can be trained to **translate camera images into steering commands** by studing human drivers.

I trained my **PilotNet model** with the **sample dataset** providec by project , it was surprising that the vehicle already could run a full lap without going out of track , only for a couple of times, it would run close to the sides of the track. It convinced me that the model is very powerful.

In order to gauge how well the model was working, I split my image and steering angle data into **a training and validation set**, and plot the **history loss**, I found that although the vehicle seemed run well in the simulator,  the model had a low mean squared error on the training set but it had a high mean squared error on the validation set. This implied that the model was **overfitting**. 

![alt text][image1.0]

To combat the overfitting, I first implemented a **generator** to generate more data. The generator is also a great way work with large amounts of data by pulling pieces of the data and processing them on the fly only when data is needed, which is much more memory-efficient.

Then I started to **modify the model** by adding **one more fully connected layer** to do better prediction , and adding **3 dropout layers** to reduce overfitting,  I later **finetuned** the model by **trying difffent numbers for nodes** in the fully connected layers


At the end, the vehicle is able to **drive autonomously around the track without leaving the road!**

![alt text][image0.1]
![alt text][image0.3]

The **history loss** of final model, both training set and validation set have low losses finally.

![alt text][image1.2]
![alt text][image1.1]



#### 2. Final Model Architecture

The final model architecture (model.py lines 88-90 and 109-125) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						|
| Cropping         		| 90 x320x3 RGB image   						|
| Lambda             	| 90 x320x3 RGB image   						|
| Convolution1 5x5      | 2x2 stride, VALID padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution2 5x5	    | 2x2 stride, VALID padding, outputs 20x77x36   |
| RELU					|												|
| Convolution3 5x5      | 2x2 stride, VALID padding, outputs 8x37x48 	|
| RELU					|												|
| Convolution4 3x3	    | 1x1 stride, VALID padding, outputs 6x35x64    |
| RELU					|												|
| Convolution5 3x3	    | 1x1 stride, VALID padding, outputs 4x33x64    |
| RELU					|												|
| Fully connected		| Input units 8448(4x33x64), Output units 100   |
| Fully connected		| Input units 100, Output units 100             |
| Fully connected		| Input units 100, Output units 50              |
| Fully connected		| Input units 50, Output units 15               |
| Fully connected		| Input units 15, Output units 1                |

The output of the model is logits coming out of last fully connected layer, which predicts the steering angle for the car

Here is a visualization of the architecture 

![alt text][image6.Network]

#### 3. Creation of the Training Set & Training Process

#### 1) Collect more data
**At the begining**,  I used the **sample dataset** provided by the project to do basic training and testing.  Later **to capture more driving behaviors**, I recorded **my own data on track one** using center lane driving.

I specially recorded more data long the **sharp curves** or where land is confusing to the vehicle because the vehicle is prone to drive out of track there, here are examples:

![alt text][image2.1]
![alt text][image2.3]

I also took a drive on track two

![alt text][image2.4]

I used the images captured from both side(left and right) cameras to **simulate the vehicle recovering from the left side and right side of the road back to center**, I set a 'correction' parameter as 0.2 to calculate the corresponding steering angles for the left and right camera images. (model.py line 58-61) . (I once tried to set it as 0.15 but results were not good)
```
correction = 0.2  # parameter to tune
angle_center = float(cur_sample[3])
angle_left   = angle_center + correction  
angle_right  = angle_center - correction
```


Here is an example image captured by center , left, and right cameras:

![alt text][image3.center]
![alt text][image3.left]
![alt text][image3.right]


During training, I feed images from side perspectives to the model as if they were coming from the center camera, so that the vehicle would **learn to steer if the car drifts off to the left or the right**.

#### 2) Augmentation of the data sat
 I **flipped center images** taking the opposite sign of the steering angles (model.py line 63-65),  thinking that this would help the model generize better. In this way, I obtain data samples just like I record data by runing in an opposite direction on the track. For example, here is an image that has then been flipped (to the above center image):

![alt text][image3.flip]

After the collection process, I had **56529 images** (there are 18843 rows/points of samples in records where each sample has 3 camera images, center, left and right). I then preprocessed this data in the generator as mentioned above. (i.e. fliping and left/right corrections)

![alt text][image4.data]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by ploting the history loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Other Ideas
I also tried another architecture by adding **one more convolutional layer** instead of one more fully connected layer:

`model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))`

The training and validation losses were quickly going down, it seems this model should be better, but the model couldn't work well when using simulator to run the test, the vehicle was able to successfully pass a long way but finally fell into the water.

By testing this model, I was impressed by the power of convolution layers because it could help the model converge much faster.

One interesting thing is that I removed  activations of 'relu' in the part where the fully connected layers are,  the results were even better than those with 'relu' in terms of losses.  


![alt text][image5.history-loss]

```
def build_PilotNet_Model(model):
    model.add(Conv2D(24, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(36, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(48, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
    model.add(Flatten())

    model.add(Dropout(drop_rate))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

```



