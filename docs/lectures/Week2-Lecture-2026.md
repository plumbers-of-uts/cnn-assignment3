![image 1](Week2-Lecture-2026_images/imageFile1.png)

![image 2](Week2-Lecture-2026_images/imageFile2.png)

42028: Deep Learning and Convolutional Neural Network

## Week-2 Lecture

Machine Learning and Image Processing Basics

Outline

- • Types of Machine Learning System
- • Supervised and Un-supervised learning
- • Support Vector Machine (SVM)
- • Evaluation Metrics
- • Image Processing Basics, Types
- • Edge Detection using Convolution

### Machine Learning Basics

Type of Machine Learning Systems

|Supervised Learning Unsupervised Learning Semi-supervised Learning Reinforcement Learning|
|---|

Depending on whether the system is trained with human supervision

|Batch and Online Learning|
|---|

Whether System can learn on the fly

Comparing data points or detect patterns in training data to build a predictive model

|Instance-based and Model-based Learning|
|---|

![image 3](Week2-Lecture-2026_images/imageFile3.png)

?

![image 4](Week2-Lecture-2026_images/imageFile4.png)

Apple

![image 5](Week2-Lecture-2026_images/imageFile5.png)

![image 6](Week2-Lecture-2026_images/imageFile6.png)

Pear

|Predictive Model|
|---|

Supervised Learning

![image 7](Week2-Lecture-2026_images/imageFile7.png)

![image 8](Week2-Lecture-2026_images/imageFile8.png)

![image 9](Week2-Lecture-2026_images/imageFile9.png)

![image 10](Week2-Lecture-2026_images/imageFile10.png)

![image 11](Week2-Lecture-2026_images/imageFile11.png)

![image 12](Week2-Lecture-2026_images/imageFile12.png)

![image 13](Week2-Lecture-2026_images/imageFile13.png)

![image 14](Week2-Lecture-2026_images/imageFile14.png)

![image 15](Week2-Lecture-2026_images/imageFile15.png)

Pear

![image 16](Week2-Lecture-2026_images/imageFile16.png)

Mango

![image 17](Week2-Lecture-2026_images/imageFile17.png)

![image 18](Week2-Lecture-2026_images/imageFile18.png)

|Classification Task|
|---|

Labelled data for training (Object + Desired Output Label)

|0<br><br>500<br><br>1000<br><br>1500<br><br>2000<br><br>2500<br><br>0 200 400 600 800 1000 1200 1400<br><br>PriceinAUD$(in100Ks)<br><br>Size in Sq. ft|
|---|

|House Price prediction Feature:<br><br>- Size of the house To Predict:<br>- Price of the house<br>|
|---|

PriceinAUD$(in10

|Regression Task|
|---|

Important Algorithms:

- • K-Nearest Neighbours
- • Logistic regression
- • Support Vector Machines (SVMs)
- • Neural Networks (\*some of themcan be unsupervised)

Supervised Learning Examples

Image Classification Object Recognition

Predictions

Forecasting

Supervised Learning

CLASSIFICATION CLASSIFICATIONREGRESSION

Fraud Detection

New Insights

Medical Diagnostics

Process Optimization

![image 19](Week2-Lecture-2026_images/imageFile19.png)

Unsupervised Learning

![image 20](Week2-Lecture-2026_images/imageFile20.png)

|![image 21](Week2-Lecture-2026_images/imageFile21.png)|
|---|

![image 22](Week2-Lecture-2026_images/imageFile22.png)

![image 23](Week2-Lecture-2026_images/imageFile23.png)

![image 24](Week2-Lecture-2026_images/imageFile24.png)

![image 25](Week2-Lecture-2026_images/imageFile25.png)

![image 26](Week2-Lecture-2026_images/imageFile26.png)

![image 27](Week2-Lecture-2026_images/imageFile27.png)

|![image 28](Week2-Lecture-2026_images/imageFile28.png)|
|---|

![image 29](Week2-Lecture-2026_images/imageFile29.png)

![image 30](Week2-Lecture-2026_images/imageFile30.png)

![image 31](Week2-Lecture-2026_images/imageFile31.png)

![image 32](Week2-Lecture-2026_images/imageFile32.png)

![image 33](Week2-Lecture-2026_images/imageFile33.png)

![image 34](Week2-Lecture-2026_images/imageFile34.png)

Unsupervised Learning

![image 35](Week2-Lecture-2026_images/imageFile35.png)

![image 36](Week2-Lecture-2026_images/imageFile36.png)

![image 37](Week2-Lecture-2026_images/imageFile37.png)

![image 38](Week2-Lecture-2026_images/imageFile38.png)

![image 39](Week2-Lecture-2026_images/imageFile39.png)

![image 40](Week2-Lecture-2026_images/imageFile40.png)

![image 41](Week2-Lecture-2026_images/imageFile41.png)

|![image 42](Week2-Lecture-2026_images/imageFile42.png)<br><br>![image 43](Week2-Lecture-2026_images/imageFile43.png)<br><br>![image 44](Week2-Lecture-2026_images/imageFile44.png)<br><br>![image 45](Week2-Lecture-2026_images/imageFile45.png)|
|---|

![image 46](Week2-Lecture-2026_images/imageFile46.png)

![image 47](Week2-Lecture-2026_images/imageFile47.png)

![image 48](Week2-Lecture-2026_images/imageFile48.png)

|Clustering Task|
|---|

![image 49](Week2-Lecture-2026_images/imageFile49.png)

![image 50](Week2-Lecture-2026_images/imageFile50.png)

|Groups of similar fruits|
|---|

Unlabelled data for training

Unsupervised Learning

Important Algorithms:

- • k-means

- • Expectation Maximization

- • A Support Vector Machine is a very powerful and versatile Machine Learning model, capable of performing linear or non-linear classification, regression, and also outlier detection.

- • Defined by a separating hyperplane

- • Suitable for small or medium sized datasets

###### Reference and Pre-Reading:

Theory: https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72 Implementation: https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-coding-edd8f1cf8f2d

|Feature dimension: 2|
|---|

Orange Apple

Weight

###### ?

|SVM finds the best line or hyper-plane which will fairly separates the classes|
|---|

![image 51](Week2-Lecture-2026_images/imageFile51.png)

Colour

###### Example: Using sklearn for SVM classification (Partialcodesnippet)

|![image 52](Week2-Lecture-2026_images/imageFile52.png)|
|---|

Reference: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

https://en.wikipedia.org/wiki/Iris_flower_data_set

###### SVM Parameters: Kernel, Gamma, Regularization (C)

| |
|---|

|![image 53](Week2-Lecture-2026_images/imageFile53.png)|
|---|

![image 54](Week2-Lecture-2026_images/imageFile54.png)

Low Regularization value

| |
|---|

|![image 55](Week2-Lecture-2026_images/imageFile55.png)|
|---|

![image 56](Week2-Lecture-2026_images/imageFile56.png)

High Regularization value

Image source and Reference: https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72

###### Example: Using sklearn for SVM classification

![image 57](Week2-Lecture-2026_images/imageFile57.png)

![image 58](Week2-Lecture-2026_images/imageFile58.png)

###### Iris flower data set

![image 59](Week2-Lecture-2026_images/imageFile59.png)

![image 60](Week2-Lecture-2026_images/imageFile60.png)

![image 61](Week2-Lecture-2026_images/imageFile61.png)

![image 62](Week2-Lecture-2026_images/imageFile62.png)

Reference: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

https://en.wikipedia.org/wiki/Iris_flower_data_set

##### • Precision & Recall

###### Precision: TP/Cancer Diagnoses Diagnoses

What are the “correct” cells?

| |No Cancer|Cancer|
|---|---|---|
|No Cancer|TN|FP|
|Cancer|FN|TP|

###### • TN: (Number of True Negatives), i.e., patients who did not have cancer whom we

Truestate

correctly diagnosed as not having cancer.

###### • TP: (Number of True Positives), i.e., patients who did have cancer whom we

correctly diagnosed as having cancer

Recall: TP/Cancer True States

##### • Precision & Recall

###### Precision: TP/Cancer Diagnoses Diagnoses

what are the “error” cells are:

| |No Cancer|Cancer|
|---|---|---|
|No Cancer|TN|FP|
|Cancer|FN|TP|

- • FN: (Number of False Negatives), i.e., patients who did have cancer whom we

incorrectly diagnosed as not having cancer

- • FP: (Number of False Positives), i.e., patients who did not have cancer whom we incorrectly diagnosed as having cancer

Truestate

Recall: TP/Cancer True States

|Precision= (𝑇𝑃)/(𝑇𝑃+𝐹𝑃)|
|---|

|Recall = (𝑇𝑃)/(𝑇𝑃+𝐹𝑁)|
|---|

##### • Intersection over Union (IoU):

![image 63](Week2-Lecture-2026_images/imageFile63.png)

###### Intersection over Union is a metric used for the evaluation of an object detector, i.e. how good is the predicted bounding box for an object detected closely matches

![image 64](Week2-Lecture-2026_images/imageFile64.png)

Reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

### Image Processing Basics

Image Processing Basics

##### What is a digital image?

- - Digital images are made of picture elements called Pixels.
- - It is an array, or a matrix of Pixels arranges in columns and rows.
- - Each Pixel has its own intensity value, or brightness
- - Intensity values in digital images are defined by bits
- - For a standard 8 bits image, a pixel can have 28= 256 (0 – 255) values.
- - Black & White images have a single 8-bits intensity range.

How computer sees Image?

![image 65](Week2-Lecture-2026_images/imageFile65.png)

![image 66](Week2-Lecture-2026_images/imageFile66.png)

|![image 67](Week2-Lecture-2026_images/imageFile67.png)|
|---|

A (24 X 16) Matrix which represents the number ’8’

### Colour Images

![image 68](Week2-Lecture-2026_images/imageFile68.png)

![image 69](Week2-Lecture-2026_images/imageFile69.png)

![image 70](Week2-Lecture-2026_images/imageFile70.png)

![image 71](Week2-Lecture-2026_images/imageFile71.png)

|170|170|55|170|170|
|---|---|---|---|---|
|170|55|170|55|170|
|170|55|170|55|170|
|55|140|140|140|55|
|55|170|170|170|55|

##### Image dimension = 5 X 5 f(2, 3) = 170 (Pixel/intensity value)

Hence, an image may be defined as a 2D function f(x, y) , where, x and y are spatial co-ordinates, and the amplitude of f at (x , y) is the intensity or Gray level of the image at that point/pixel.

###### 5 X 5 Gray scale image (8 bit)

Blue

170 170 55 170 170

Green

###### Image dimension = 5 X 5 X 3 No. of Channels = 3

170 170 55 170 170

170 55 170 55 170

Red

170 170 55 170 170

170 55 170 55 170

170 55 170 55 170

Since, RGB image contains 3 X 8-bits of intensities, they are referred to as 24-bit colour images.

170 55 170 55 170

170 55 170 55 170

55 140 140 140 55

170 55 170 55 170

55 140 140 140 55

55 170 170 170 55

So, 24-bit colour depth

55 140 140 140 55

= 8 X 8 X 8 bits

55 170 170 170 55

= 256 X 256 X 256 colours

55 170 170 170 55

= ~16 million colours

5 X 5 X 3 colour image (24 bit)

Image Processing - Types

- 1. Image Enhancement
- 2. Image Restoration
- 3. Image Segmentation
- 4. Image Recognition & Classification
- 5. Image Compression
- 6. Image Transformation
- 7. Image Filtering
- 8. Morphological Processing
- 9. Colour Image Processing
- 10. 3D Image Processing

### Image Enhancement, Restoration

![image 72](Week2-Lecture-2026_images/imageFile72.png)

![image 73](Week2-Lecture-2026_images/imageFile73.png)

![image 74](Week2-Lecture-2026_images/imageFile74.png)

###### Enhancement Restoration

Source: https://www.mathworks.com/discovery/image-enhancement.html https://en.wikipedia.org/wiki/Image_restoration_by_artificial_intelligence

- - Easiest method for image segmentation!
- - Converts gray-scale image into a binary image If f(x,y) > Threshold, then f(x,y) = 0 else f(x,y) = 255

###### Binary Image (8-bit) has only two possible values of pixel intensity ( 0 and 1, or B & W)

|170|170|55|170|170|
|---|---|---|---|---|
|170|55|170|55|170|
|170|55|170|55|170|
|55|140|140|140|55|
|55|170|170|170|55|

|255|255|0|255|255|
|---|---|---|---|---|
|255|0|255|0|255|
|255|0|255|0|255|
|0|255|255|255|0|
|0|255|255|255|0|

Threshold = 100

|![image 75](Week2-Lecture-2026_images/imageFile75.png)|
|---|

|![image 76](Week2-Lecture-2026_images/imageFile76.png)|
|---|

Thresholding

Original Image Binary Image

Image Source: https://en.wikipedia.org/wiki/Thresholding\_(image_processing)

Image Thresholding methods

![image 77](Week2-Lecture-2026_images/imageFile77.png)

###### - Histogram shape:

Peaks, valleys and curvature of the histogram are analysed.

No.ofPixels

###### - Clustering based:

The 1Otsu method, very good for bimodal distribution

|Threshold|
|---|

###### - Adaptive thresholding:

Gray-scale

Instead of a single threshold, have thresholds for different regions in the image

1Reference: https://en.wikipedia.org/wiki/Otsu%27s_method

![image 78](Week2-Lecture-2026_images/imageFile78.png)

###### Source https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html

![image 79](Week2-Lecture-2026_images/imageFile79.png)

![image 80](Week2-Lecture-2026_images/imageFile80.png)

Edge Detection (Image Filtering)

##### What is an edge?

- - The points/pixels in an image where brightness/intensities changes sharply
- - A simple and fundamental tools in image processing and computer vision, useful in feature detection/extraction

![image 81](Week2-Lecture-2026_images/imageFile81.png)

![image 82](Week2-Lecture-2026_images/imageFile82.png)

![image 83](Week2-Lecture-2026_images/imageFile83.png)

![image 84](Week2-Lecture-2026_images/imageFile84.png)

Colour to Gray

Edge detection

![image 85](Week2-Lecture-2026_images/imageFile85.png)

| |
|---|

Canny Edge detection Sobel Edge detection

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

-

# \*

|100|100|100|100|100|100|
|---|---|---|---|---|---|
|100|100|100|100|100|100|
|100|100|100|100|100|100|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|

|0|0|0|0|
|---|---|---|---|
|300|300|300|300|
|300|300|300|300|
|0|0|0|0|

|1|1|1|
|---|---|---|
|0|0|0|
|-1|-1|-1|

# \*

# \*

### Edge detection filters

||1|0|-1|
|---|---|---|
|1|0|-1|
|1|0|-1|
<br><br>|1|1|1|
|---|---|---|
|0|0|0|
|-1|-1|-1|
<br><br>3 X 3 filter/Kernel For Horizontal edges<br><br>3 X 3 filter/Kernel For Vertical edges<br><br>Prewitt Filters|
|---|

||1|2|1|
|---|---|---|
|0|0|0|
|-1|-2|-1|
<br><br>3 X 3 filter/Kernel For Horizontal edges<br><br>|1|0|-1|
|---|---|---|
|2|0|-2|
|1|0|-1|
<br><br>3 X 3 filter/Kernel For Vertical edges<br><br>Sobel Filters|
|---|

![image 86](Week2-Lecture-2026_images/imageFile86.png)

### Sobel edge detection - Example

![image 87](Week2-Lecture-2026_images/imageFile87.png)

Image Source: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html

Convolutions in CNN

- • Convolutions are very important operation in a Convolutional Neural Networks (CNN)
- • Filters weights are not fixed, but learned during the training operations of a CNN for a specific task!
- • Multiple filters are used in CNNs

![image 88](Week2-Lecture-2026_images/imageFile88.png)

![image 89](Week2-Lecture-2026_images/imageFile89.png)

![image 90](Week2-Lecture-2026_images/imageFile90.png)

![image 91](Week2-Lecture-2026_images/imageFile91.png)

![image 92](Week2-Lecture-2026_images/imageFile92.png)

![image 93](Week2-Lecture-2026_images/imageFile93.png)

![image 94](Week2-Lecture-2026_images/imageFile94.png)

Erosion example

Dilation example
