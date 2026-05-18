![image 1](Week7-Lecture-2025_images/imageFile1.png)

42028: Deep Learning and Convolutional Neural Network

# Week-7 Lecture

Convolutional Neural Network (CNN) - 3

Outline

- • Transfer Learning strategies

- • Classic CNN architectures case studies

- • AlexNet

- • Inception/GoogleNet

- • Understanding Inception and ResNet

- • 1X1 Convolution and it’s use

- • Motivation behind Inception

- • Inception V1, V2 and V3 Modules

- • Introduction to ResNet

- • Motivation behind ResNet

- • Residual Block

Transfer Learning

- • Knowledge acquired while solving one task, can be used to solve related tasks.

- • Example:

- • You know how to ride a Bi-cycle → You can learn how to ride a Motorbike

- • You know how to use a Tablet → You can easily learn how to use a Laptop/desktop

- • Similar to the way humans apply knowledge acquired from one task to solve a new but similar/related task.

- • We learned how to read in Year-1 in literacy class. Reading skills acquired in the literacy classes made it easy to understand Physics in Year-9.

Transfer Learning Benefits

- 1. Less training data required: Don’t have enough data to train a Deep Learning model from scratch. Model trained using a large (similar) dataset can be used.
- 2. Faster training : Training can converge faster, due the use to existing knowledge (weights) to start with rather than from scratch.
- 3. Better model generalization: Model is trained to identify features which can be applied to new contexts.

Source and reference: https://cs231n.github.io/transfer-learning/ https://missinglink.ai/guides/neural-network-concepts/transfer-learning-overview/

![image 2](Week7-Lecture-2025_images/imageFile2.png)

#### Option-1: (VGG-16 considered as an example) Use pre-trained (ImageNet) model for prediction, without any training.

→Useful when your dataset distribution is similar to ImageNet, with small

number of samples.

###### Train

|Freeze|![image 3](Week7-Lecture-2025_images/imageFile3.png)|
|---|---|

#### Option-2: (VGG-16 considered as an example) Train Full-Connected layer, Use CONV layers for feature extraction

→Useful when your dataset distribution is similar to ImageNet (or original dataset), but number of classes are different and your dataset is small.

![image 4](Week7-Lecture-2025_images/imageFile4.png)

Train/Fine-Tune

|Freeze|
|---|

| |
|---|

###### Option-3: (VGG-16 considered as an example)

###### Partially Train CONV layers (usually last layer(s) which have specialised

###### features) + Full Connection (FC) layer (with modifications)

→Useful when your dataset distribution is not similar to ImageNet (or original dataset), number of classes are different and your dataset is small.

|![image 5](Week7-Lecture-2025_images/imageFile5.png)<br><br>Train/Fine-Tune|
|---|

#### Option-4: (VGG-16 considered as an example) Train all the CONV layers + Full Connection (FC) layer (with modifications)

→Useful when your dataset distribution is not similar to ImageNet, number of classes are different, your is dataset large and the task is complex.

### Classic CNN Architectures

- • Similar architecture as LeNet by Yann LeCunn et al. but deeper with more layers

- • Simple architecture:

- • CONV : 5 layers

- • FC: 3 layer

- • Max pooling

- • Dropout

- • Accuracy: top-5 test error rate of 15.3%

- • Winner of ILSVRC 2012!

- • First CNN to be successful on a very big dataset!

![image 6](Week7-Lecture-2025_images/imageFile6.png)

Input: 224x224x3 image

##### CONV1 → CONV2 → CONV3 → CONV4 → CONV5 → FC1 → FC2 → FC3

4096 Neuron

4096 Neuron

Filters: 96 Dim: 11x11 Stride: 4 Pad: 0

Filters: 256 Dim: 5x5 Stride: 1 Pad: 2

Filters: 384 Dim: 3x3 Stride: 1 Pad: 1

Filters: 384 Dim: 3x3 Stride: 1 Pad: 1

Filters: 256 Dim: 3x3 Stride: 1 Pad: 1

1000 Neuron

Activations: Relu after each CONV and FC layer Optimizer: SGD with Momentum Regularization: Dropout in FC1 and FC2 Total Trainable parameter: ~60Million Training settings: 2 X Nvidia GTX 580 3GB GPUs for 5-6days!

- • Accuracy: top-5 test error rate of 6.7%
- • Close to human level performance
- • Winner of ILSVRC 2014!
- • 22 layer Deep CNN
- • Number of trainable parameters: 4 Million (Alexnet ~ 60M), Significantly reduced
- • A novel inception module was introduced.
- • Optimizer: RMSProp

Source and reference: https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5

![image 7](Week7-Lecture-2025_images/imageFile7.png)

![image 8](Week7-Lecture-2025_images/imageFile8.png)

###### Inception Module

![image 9](Week7-Lecture-2025_images/imageFile9.png)

![image 10](Week7-Lecture-2025_images/imageFile10.png)

## Understanding Inception and ResNet

|100|100|100|0|0|0|
|---|---|---|---|---|---|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|

|300|300|300|0|0|0|
|---|---|---|---|---|---|
|300|300|300|0|0|0|
|300|300|300|0|0|0|
|300|300|300|0|0|0|
|300|300|300|0|0|0|
|300|300|300|0|0|0|

###### Is this useful?

|3|
|---|

-

| | |
|---|---|
|Convolution operator| |

1 X 1 filter/Kernel

6 X 6 X 1 dimension volume

6 X 6 X 1 dimension image

![image 11](Week7-Lecture-2025_images/imageFile11.png)

![image 12](Week7-Lecture-2025_images/imageFile12.png)

3 X 3

1 X 1

Images source: https://iamaaditya.github.io/2016/03/one-by-one-convolution/

###### + Relu

-

:

(6 X 6 )

X 32 (# of filters)

6 X 6 X 64

1 X 1 X 64

X 32 (# of filters)

#### So, (6 X 6 X 64) → (6 X 6 X 32) … reduced!

###### Relu

1 X 1 Conv 128 filters

64 X 64 X 256 64 X 64 X 128

![image 13](Week7-Lecture-2025_images/imageFile13.png)

- - Large filter preferred for large objects
- - Small filters for small objects
- - Large variation in object size
- - How to choose the right filter size?

Images source: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

|Designing CNN requires:<br><br>- Deciding filter size and number<br>- Number and type of layers etc.<br>|
|---|

|Inception suggests:<br><br>- Use filters with different size together!<br>- Use different types of layers (CONV, POOL etc.) together Result → Complicated Architecture! & better performance<br>|
|---|

###### 28 X 28 X 64

|1X1|
|---|

|3X3|
|---|

64

| |
|---|

128

28

32 28 32

|5X5|
|---|

28 X 28 X 192 28 X 28 X 256 Max-Pooling

Reference, Images source: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

Computation cost

28 X 28 X 32

|5X5|
|---|

28

32 28

5x5 CONV #Filter: 32

28 X 28 X 32

28 X 28 X 192

###### Computation Cost: 28 X 28 X 32 X 5 X 5 X 192  120M multiplications! Quite expensive !

Reduce Computation cost using 1X1 CONV

|1X1|
|---|

28 X 28 X 32

|5X5|
|---|

28

32 28

1 X 1 X 192 #Filters: 16

5 X 5 X 16 CONV #Filter: 32

28 X 28 X 32

###### 28 X 28 X 16

28 X 28 X 192

Computation Cost: 1X1: 28 X 28 X 16 X 192  2.4M multiplications! 5X5: 28 X 28 X 32 X 5 X 5 X 16  10M multiplications! Total : 12.4M multiplications!  Reduced by 10 times!

## Bottleneck Layer

|192|
|---|

|32|
|---|

|16|
|---|

|Bottleneck|
|---|

![image 14](Week7-Lecture-2025_images/imageFile14.png)

Image adapted from and Reference: http://www.pabloruizruiz10.com/resources/CNNs/1x1Convolutions.pdf

## Inception Module V1

![image 15](Week7-Lecture-2025_images/imageFile15.png)

|![image 16](Week7-Lecture-2025_images/imageFile16.png)|
|---|

![image 17](Week7-Lecture-2025_images/imageFile17.png)

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

![image 18](Week7-Lecture-2025_images/imageFile18.png)

###### GoogleNet(2014): 9 Inception modules stacked together

![image 19](Week7-Lecture-2025_images/imageFile19.png)

| |
|---|

|Deeper Network → Vanishing Gradient<br><br>- Introduced two auxiliary classifier<br>- Applied Softmax to the output<br>- Compute Auxiliary loss/cost<br>- Only used for training<br>|
|---|

| |
|---|

Aux_Loss2

Aux_Loss1

|Auxiliary Classifiers|
|---|

|Total Loss/cost = Real_Loss + 0.3 X Aux_Loss1 + 0.3 X Aux_Loss1|
|---|

![image 20](Week7-Lecture-2025_images/imageFile20.png)

|Authors suggested 3 different modules<br><br>-Factorizing Convolutions: Reducing the number of parameters|
|---|

|1 layer of 5×5 filter, #parameters = 5×5=25<br>2 layers of 3×3 filters, #parameters = 3×3+3×3=18 Number of parameters is reduced by 28%<br>|
|---|

![image 21](Week7-Lecture-2025_images/imageFile21.png)

|3×3 filter, #parameters = 3×3=9<br><br>3×1 and 1×3 filters, #parameters = 3×1+1×3=6<br><br>Number of parameters is reduced by 33%|
|---|

![image 22](Week7-Lecture-2025_images/imageFile22.png)

## Inception V3 Architecture

![image 23](Week7-Lecture-2025_images/imageFile23.png)

Image Source and reference: https://www.jeremyjordan.me/convnet-architectures/

For More details: https://cloud.google.com/tpu/docs/inception-v3-advanced

- - Deep Residual networks (ResNet) → Skip connections
- - Enabled the development of the much deeper networks (100s of layers!)
- - ResNet is composed of Residual Blocks were introduced!
- - Degradation problem: Adding more layers eventually have negative effect on the final performance.

Image Source and reference: https://www.jeremyjordan.me/convnet-architectures/

What wrong with this curves? Overfitting?

![image 24](Week7-Lecture-2025_images/imageFile24.png)

- - 56 layer model is not better than the 20 layers!
- - What happens when we keep add more layers to a plain CNN to make it deeper?

###### In principle deeper model should perform better than shallow CNNs

![image 25](Week7-Lecture-2025_images/imageFile25.png)

![image 26](Week7-Lecture-2025_images/imageFile26.png)

|Residual<br><br>Block|
|---|

|Plain Layers|
|---|

![image 27](Week7-Lecture-2025_images/imageFile27.png)

![image 28](Week7-Lecture-2025_images/imageFile28.png)

## Summary

![image 29](Week7-Lecture-2025_images/imageFile29.png)

Reference and image Source: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf

- • 15+ million labelled high-resolution images
- • 22000 categories
- • ILSVRC (Large Scale Visual Recognition Challenge) used a subset of ImageNet:
- • ~1000 images per category
- • 1000 categories
- • Train: 1.2 million images
- • Validation: 50k images
- • Test : 150k images

Source and reference: http://www.image-net.org/ http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf

![image 30](Week7-Lecture-2025_images/imageFile30.png)

###### Source and reference https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_1k.jpg

ImageNet Dataset Results:

![image 31](Week7-Lecture-2025_images/imageFile31.png)

Image Source and reference: An Analysis of Deep Neural Network Models for Practical Applications, 2017, https://arxiv.org/abs/1605.07678

ImageNet Dataset Results (current):

|![image 32](Week7-Lecture-2025_images/imageFile32.png)|
|---|

Image Source and reference: https://paperswithcode.com/sota/image-classification-on-imagenet
