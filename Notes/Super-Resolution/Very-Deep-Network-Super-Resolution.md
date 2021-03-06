## Introduction
- The main objective is to generate a very high resolution Image (like DSLR cameras) from Low Resolution Input (VGA, QVGA)

- Main application areas like - Security, Smartphones, Criminal Investigations and surveillance
  
## Methods
### Deep Learning methods
#### 1.1 SRCNN
- SRCNN is a prominent method.
- It has three layers  
    - Patch Extraction layer and representation (9x9)
    - Non linear mapping (1x1)
    - Reconstruction (5x5)

![](resources/vdsr1.png)

#### 1.2 VDSR

- In this approach a pair of layers (CNN and nonlinear) is cascaded repeatedly
- An interpolated low resoultion (ILR) goes through layers and transforms into a high resolution image.
- A residual image is predicted by network.
- This residual is added to the ILR to give the required output
- ReLU is the most common non-linear unit used.

##### 1.2.1 Features
- Very deep network (20) and large receptive fields (41 x 41)
- Loss functions
- SRCNN has a very small learning rate. Setting high learning rates lead to vanishing gradients.Adjustable gradient clipping is solution (i.e. clip gradients to [-theta/gamma, theta/gamma] where gamma is the learning rate)
  
- Use of scale augmentation to boost performance
###### 1.2.2 Hyperparameters

- Learing Rate = 0.1 decreased by a factor 10 every 20 epochs
- Epochs = 80
- Depth = 20
- Batch Size = 64
- Weight Initialization = Group normalization
- Activation = ReLU / Parametric ReLU
- Training Dataset = Berkely segmentation dataset
- Momentum = 0.9
- Weight Decay = 0.0001

###### 1.2.3 Comparision

![](resources/vdsr1.JPG)

### Classical methods

#### 1.1 Super Resolution Optical Flow
- Introduced by Simon Baker and Takeo Kanade
- Best suited for facial features
- Steps 
   -  Takes input a conventional video stream
   -  Simultaneously compute optical flow and super-resolution of entire video.
  
- Flow of method
  
  Registration -> Warping -> Fusion  -> Deblurring
- Any Optical Flow algorithm can be used.

  ![](./resources/vdsr2.JPG)

