![image 1](Week5-Lecture-2025_images/imageFile1.png)

42028: Deep Learning and Convolutional Neural Network

## Week-5 Lecture

Convolutional Neural Network (CNN) - 1

Outline

- • Computer Vision tasks Recap
- • CNN Layers
- • Convolution layer (Padding and Stride)
- • Pooling layer
- • Fully Connected Layer
- • CNN Layer Visualization and intuitions.

![image 2](Week5-Lecture-2025_images/imageFile2.png)

![image 3](Week5-Lecture-2025_images/imageFile3.png)

![image 4](Week5-Lecture-2025_images/imageFile4.png)

![image 5](Week5-Lecture-2025_images/imageFile5.png)

Cat

64 X 64 X 3

![image 6](Week5-Lecture-2025_images/imageFile6.png)

Dog

800 X 800 X 3

### Deep-NNs Vs CNNs

|![image 7](Week5-Lecture-2025_images/imageFile7.png)|
|---|

|![image 8](Week5-Lecture-2025_images/imageFile8.png)|
|---|

###### ANN with 3-layers

###### CNN with 3-layers

- - Fully connected
- - Not great for images as input
- - May lead to overfitting
- - Too much of full connectivity
- - Well suited for images with 3 dimensions
- - CNNs have neurons arranged in 3D
- - It is a sequence of layers which transforms input 3D volume to 3D outputs volume

Image Source and reference to read : http://cs231n.github.io/convolutional-networks/

###### CNNs are the foundations of modern state-of-the-art deep

###### learning based computer vision. Layers in a CNN:

Three main type of layers used to build a CNN architecture

- 1. Convolutional Layer (CONV)
- 2. Pooling Layer (POOL)
- 3. Fully Connected layer (FC) These three types of layers are stacked together to form a CNN architecture!

Source and reference to read : http://cs231n.github.io/convolutional-networks/ https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4

##### Sample CNN architecture (LENET-5):

|CONV Layer|
|---|

![image 9](Week5-Lecture-2025_images/imageFile9.png)

|FC Layer|
|---|

|POOL Layer|
|---|

##### Sample CNN architecture:

| |
|---|

| |
|---|

| |
|---|

![image 10](Week5-Lecture-2025_images/imageFile10.png)

Image Source: https://medium.com/analytics-vidhya/ai-ml-dl-whats-what-ecb354967e63

- • CONVolution is the first layer to extract features from an input image
- • Core building block of a CNN
- • Convolutions are basic operation in this layer
- • A number of filters (e.g. edge detectors) are applied to the input image

Convolution Operation

|100|100|100|0|0|0|
|---|---|---|---|---|---|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|

|0|300|300|0|
|---|---|---|---|
|0|300|300|0|
|0|300|300|0|
|0|300|300|0|

|1|0|-1|
|---|---|---|
|1|0|-1|
|1|0|-1|

# \*

| | |
|---|---|
|Convolution operator| |

3 X 3 filter/Kernel

4 X 4 dimension matrix

6 X 6 dimension image

###### Convolution Operation

|(100 X 1 + 100 X 1 + 100 X 1) + (100 X 0 + 100 X 0 + 100 X 0) + (100 X -1 + 100 X -1 + 100 X -1)|
|---|

|1100|0100|-1100|0|0|0|
|---|---|---|---|---|---|
|1100|0100|-1100|0|0|0|
|1100|0100|-1100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|

|0|300|300|0|
|---|---|---|---|
|0|300|300|0|
|0|300|300|0|
|0|300|300|0|

|1|0|-1|
|---|---|---|
|1|0|-1|
|1|0|-1|

# \*

| | |
|---|---|
|Convolution operator| |

3 X 3 filter/Kernel

4 X 4 dimension matrix

6 X 6 dimension image

|Vertical Edge detector|
|---|

###### Convolution Operation

![image 11](Week5-Lecture-2025_images/imageFile11.png)

![image 12](Week5-Lecture-2025_images/imageFile12.png)

![image 13](Week5-Lecture-2025_images/imageFile13.png)

-

# →

3 X 3 Filter

Image Convolved Feature

5 X 5 Image

Image source: https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

|0|0|0|0|0|0|0|0|
|---|---|---|---|---|---|---|---|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|0|0|0|0|0|0|0|

|100|100|100|0|0|0|
|---|---|---|---|---|---|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|
|100|100|100|0|0|0|

###### Padding (p) = 1

6 X 6 dimension image without padding

8 X 8 dimension matrix with padding

|0|0|0|0|0|0|0|0|
|---|---|---|---|---|---|---|---|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|100|100|100|0|0|0|0|
|0|0|0|0|0|0|0|0|

|-200|0|200|200|0|0|
|---|---|---|---|---|---|
|-300|0|300|300|0|0|
|-300|0|300|300|0|0|
|-300|0|300|300|0|0|
|-300|0|300|300|0|0|
|-200|0|200|200|0|0|

|1|0|-1|
|---|---|---|
|1|0|-1|
|| |3 X 3<br><br>1|
|---|---|
|Convolution operator| |
|0|-1|

# \*

| |3 X 3<br><br>1|
|---|---|
|Convolution operator| |

filter/Kernel

###### 6 X 6 dimension matrix 8 X 8 dimension matrix == Input Matrix dimension

||(𝒏 + 𝟐𝒑 − 𝒇 + 𝟏) 𝑿 (𝒏 + 𝟐𝒑 − 𝒇 + 𝟏)|
|---|
<br><br>|(𝒏 𝑿 𝒏) ∗ (𝒇 𝑿 𝒇)|
|---|
<br><br>Input Matrix Dimension : 𝑛 𝑥 𝑛 Filter size: 𝑓 𝑥 𝑓 Padding (𝑝) : 1<br><br>So, ( will produce (<br><br>e.g.: 6 𝑋 6 ∗ 3 𝑋 3 → 6 𝑋 6 Output matrix<br><br>Input Matrix Output Matrix<br><br>|
|---|

Given: Input Matrix Dimension : 𝑛 𝑥 𝑛

###### Filter size: 𝑓 𝑥 𝑓

Required Output Size = 𝒏 + 𝟐𝒑 − 𝒇 + 𝟏 𝑿 𝒏 + 𝟐𝒑 − 𝒇 + 𝟏 Question: What is pad size (𝒑) so that the input and output matrix are of same

sizes?

So, 𝑛 + 2𝑝 − 𝑓 + 1 = 𝑛

𝑝 = (𝑓2−1)

#### Padding (Same and Valid)

|Valid Padding:  No Padding (Padding 𝒑 = 0) So, Output size will be → 𝒏 − 𝒇 + 𝟏 𝑿 𝒏 − 𝒇 + 𝟏<br><br>Same Padding:  Output size and input size is same, this requires<br><br>appropriate padding. Hence use 𝑝 = (𝑓2−1), for calculate the required padding.<br><br>|
|---|

Stride

It is the number of pixels by which we slide the filter over the input

matrix Example:

- 1. Stride(s) = 1: Move the filter by one pixel horizontally and vertically
- 2. Stride(s) = 2: Move the filter by two pixels horizontally and vertically

Source and Reference: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

#### Stride and Padding illustration

Convolution with stride (s)=2,

Convolution with stride (s) =2 padding (p) = 0

Convolution with stride (s) =1 padding (p) = 1

padding (p) = 1

![image 14](Week5-Lecture-2025_images/imageFile14.png)

![image 15](Week5-Lecture-2025_images/imageFile15.png)

![image 16](Week5-Lecture-2025_images/imageFile16.png)

Image source: https://github.com/vdumoulin/conv_arithmetic

Output size with Stride and padding

Given: Input Matrix Dimension : 𝑛 𝑥 𝑛

###### Filter size: 𝑓 𝑥 𝑓

###### Padding:p Stride :s

###### Output Size = 𝑛 +2𝑝𝑠 −𝑓 + 𝟏 𝑿 𝑛 +2𝑝𝑠 −𝑓 + 𝟏

Example:

Input Matrix Dimension : 7 𝑥 7, Filter size: 3 𝑥 3

Padding:0, Stride :2

###### Output Size= 3 X 3

Reference and Source: http://cs230.stanford.edu/files/C4M1.pdf

- • Pooling layer is a down sampling operation which reduces the dimensionality of a matrix.

- • In other words, it reduces the number of parameters for large image, but retain the valuable information.

- • 3 types:

- • Max pooling

- • Average pooling

- • Sum pooling

###### • Max pooling:

|Max(7, 8, 1, 5) = 8|
|---|

|7|8|9|0|
|---|---|---|---|
|1|5|8|3|
|5|9|3|2|
|5|6|6|2|

|8|9|
|---|---|
|9|6|

Max pooling with 2X2 filter and Stride 2

###### • Average pooling:

|(7+8+1+5)/4 = 5.25|
|---|

|7|8|9|0|
|---|---|---|---|
|1|5|8|3|
|5|9|3|2|
|5|6|6|2|

|5.25|5|
|---|---|
|6.25|3.25|

Max pooling with 2X2 filter and Stride 2

###### • In FC layer, the output matrix after convolution layer is flattened and feed into a fully connected layer similar to ANN

![image 17](Week5-Lecture-2025_images/imageFile17.png)

Image source ad reference: https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

- • It is a traditional Multi-layer Perception/Neural Network
- • For multi-class classification, usually Softmax activation is used.
- • Softmax ensures the output
- • Output of the CONV and POOL layers represent a high level features of the Input image
- • FC layer uses this feature to classify the input image into the desired class.

Reference: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

CNN layers visualization and intuition

![image 18](Week5-Lecture-2025_images/imageFile18.png)

###### Example: Face recognition using CNNs

|Uses simple shapes to form higher level features like facial shapes!|
|---|

|Uses edges to detect simple shapes|
|---|

|Low level feature like edges from raw pixels|
|---|

Image Source: http://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf
