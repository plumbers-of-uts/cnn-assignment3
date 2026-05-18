![image 1](Week8-Lecture-2026_images/imageFile1.png)

![image 2](Week8-Lecture-2026_images/imageFile2.png)

42028: Deep Learning and Convolutional Neural Network

# Week-8 Lecture

Object Detection -1

Outline

- • Introduction to Object Detection
- • Datasets and Performance metrics
- • Taxonomy
- • Classification and Localization task
- • Object detection as a regression problem
- • Object detection as a classification problem
- • Case study: RCNN family
- • Image annotation

Introduction

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

![image 3](Week8-Lecture-2026_images/imageFile3.png)

![image 4](Week8-Lecture-2026_images/imageFile4.png)

![image 5](Week8-Lecture-2026_images/imageFile5.png)

![image 6](Week8-Lecture-2026_images/imageFile6.png)

| |
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

DOG

![image 7](Week8-Lecture-2026_images/imageFile7.png)

![image 8](Week8-Lecture-2026_images/imageFile8.png)

Single Object

Multiple Object

- • The PASCAL Visual Object Classification (PASCAL VOC) is a popular dataset for object detection, classification and segmentation.

- • 20 categories

- • Link: http://host.robots.ox.ac.uk/pascal/VOC/

- • ImageNet has released an object detection dataset in 2013

- • Train set: 500,000 images, 200 categories.

- • Not very popular due to large number of classes and dataset size!

- • Large number classes complicates the task

Images source: https://towardsdatascience.com/evolution-of-object-detection-and-localization-algorithms-e241021d8bad

![image 9](Week8-Lecture-2026_images/imageFile9.png)

###### Dataset Comparison

Images source: http://image-net.org/challenges/LSVRC/2014/

###### • Intersection over Union (IoU): Intersection over Union is a metric used for the evaluation of an object detector, i.e. how good is the predicted bounding box for an object detected closely matches

![image 10](Week8-Lecture-2026_images/imageFile10.png)

![image 11](Week8-Lecture-2026_images/imageFile11.png)

Reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

## Microsoft COCO Dataset

|![image 12](Week8-Lecture-2026_images/imageFile12.png)|
|---|

![image 13](Week8-Lecture-2026_images/imageFile13.png)

Source: https://cocodataset.org/#home

## Microsoft COCO Evaluation metrics

![image 14](Week8-Lecture-2026_images/imageFile14.png)

Source: https://cocodataset.org/#detection-eval

## Taxonomy of Object detectors

|Object Detection|
|---|

|Network type|
|---|

|Data type|
|---|

|Single stage| |
|---|---|
| | |

|Two Stage| |
|---|---|
| | |

|3D Object Detector|
|---|

|2D Object detectors|
|---|

|Regression/Classification Based|
|---|

|Region Proposal Based| |
|---|---|
| | |

|Monocular Image|
|---|

|Point Cloud|
|---|

|Point Nets|
|---|

|RCNN family|
|---|

|SSD|
|---|

|Yolo|
|---|

Refernce: https://medium.com/@saifhajsalem12/object-detection-state-of-the-art-and-modern-approaches-eaa5e6bfb46b

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

![image 15](Week8-Lecture-2026_images/imageFile15.png)

![image 16](Week8-Lecture-2026_images/imageFile16.png)

![image 17](Week8-Lecture-2026_images/imageFile17.png)

![image 18](Week8-Lecture-2026_images/imageFile18.png)

| |
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

DOG

![image 19](Week8-Lecture-2026_images/imageFile19.png)

![image 20](Week8-Lecture-2026_images/imageFile20.png)

Single Object

Multiple Object

![image 21](Week8-Lecture-2026_images/imageFile21.png)

##### Classification Task:

Input : Image Output: Label Performance Evaluation: Accuracy

##### Output : Dog

![image 22](Week8-Lecture-2026_images/imageFile22.png)

##### Localization Task:

| |
|---|

Input : Image Output: Bounding Box in the image

(x, y, Ht, Wd) or (x, y, x’, y’) Performance Evaluation: IoU

##### Output : (x, y, Ht, Wd)

![image 23](Week8-Lecture-2026_images/imageFile23.png)

###### Output : 4 numbers (x’, y’, Ht’, Wd’)

| |
|---|

###### Calculate Loss L2 Loss

###### CNN

###### Ground Truth: 4 numbers (x, y, Ht, Wd)

Input Image

![image 24](Week8-Lecture-2026_images/imageFile24.png)

![image 25](Week8-Lecture-2026_images/imageFile25.png)

###### Input Image

|We need to modify this CNN pipeline to output Class Label and Bounding Box (4 numbers)|
|---|

![image 26](Week8-Lecture-2026_images/imageFile26.png)

###### Pre-trained model or ImageNet, AlexNet, VGG16, ResNet, etc. Classification Head

Softmax Loss

#### + MultitaskLoss

![image 27](Week8-Lecture-2026_images/imageFile27.png)

Regression Head L2 Loss

Input Image

Image Source and Reference: http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf

Potential locations for Regression head in CNN

###### After CONV Layer, Before the FC layer After Last FC layer

![image 28](Week8-Lecture-2026_images/imageFile28.png)

![image 29](Week8-Lecture-2026_images/imageFile29.png)

Input Image

Task: Object Detection Problem

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

![image 30](Week8-Lecture-2026_images/imageFile30.png)

![image 31](Week8-Lecture-2026_images/imageFile31.png)

![image 32](Week8-Lecture-2026_images/imageFile32.png)

![image 33](Week8-Lecture-2026_images/imageFile33.png)

| |
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

|CAT|
|---|

|DOG|
|---|

DOG

![image 34](Week8-Lecture-2026_images/imageFile34.png)

![image 35](Week8-Lecture-2026_images/imageFile35.png)

Single Object

Multiple Object

Detection as a regression problem

![image 36](Week8-Lecture-2026_images/imageFile36.png)

##### Output : Dog, (x, y, Ht, Wd)

|![image 37](Week8-Lecture-2026_images/imageFile37.png)|
|---|

##### Output : Dog, (x, y, Ht, Wd)

##### Cat, (x, y, Ht, Wd) Cat, (x, y, Ht, Wd)

- 1. Apply Sliding Window technique
- 2. Apply CNN to different Windows and get a prediction

![image 38](Week8-Lecture-2026_images/imageFile38.png)

##### Output : Dog? No Cat? No

|CNN|
|---|

##### Background? Yes

- 1. Apply Sliding Window technique
- 2. Apply CNN to different Windows and get a prediction

![image 39](Week8-Lecture-2026_images/imageFile39.png)

##### Output : Dog? No Cat? Yes Background? No

|CNN|
|---|

- 1. Apply Sliding Window technique
- 2. Apply CNN to different Windows and get a prediction

![image 40](Week8-Lecture-2026_images/imageFile40.png)

##### Output : Dog? No Cat? Yes Background? No

|CNN|
|---|

###### Issue with Sliding Window technique

- 1. Apply CNN on large number of windows
- 2. Multiple scale and locations of windows
- 3. Inaccurate bounding boxes
- 4. Computationally expensive

###### Region Proposal Technique:

- • Find blobs in the image that are most likely to contain objects
- • E.g: Selective search à ~1000-2000 region proposals using CPU!

![image 41](Week8-Lecture-2026_images/imageFile41.png)

![image 42](Week8-Lecture-2026_images/imageFile42.png)

Source and Reference: http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf

Case Study: R-CNN

Linear Regression for bounding box offsets

###### R-CNN: Region based CNN

Classify each region with SVMs

|SVMs|
|---|

|Bbox Reg|
|---|

|SVMs|
|---|

|Bbox Reg|
|---|

- 1. Resized to match the input to CNN requirement.
- 2. mAP: 62.4% for 2007 PASCAL VOC
- 3. Problem: Very Slow!

Pass each region through ConvNet

|SVMs|
|---|

|Bbox Reg|
|---|

|Conv -Net|
|---|

|Conv -Net|
|---|

|Conv -Net|
|---|

Warped image regions

![image 43](Week8-Lecture-2026_images/imageFile43.png)

Region-of-interest (ROI) from proposal method around ~2K

|Linear + Softmax|
|---|

Object category

|Linear|
|---|

Box offset

|CNN|
|---|

Per-Region Network

###### Slow RCNN

![image 44](Week8-Lecture-2026_images/imageFile44.png)

| | | |
|---|---|---|
| | | |

Crop + Resize features

Region of Interest (ROIs) from proposal method

Run whole image through ConvNet

|ConvNet|
|---|

![image 45](Week8-Lecture-2026_images/imageFile45.png)

Source and Reference: https://cs231n.stanford.edu/slides/2022/lecture_9_jiajun.pdf

![image 46](Week8-Lecture-2026_images/imageFile46.png)

- 1. Reduce computation
- 2. ROIs from feature maps using selective search
- 3. mAP: 70% for 2007 PASCAL VOC

Case Study: FASTER- R-CNN

![image 47](Week8-Lecture-2026_images/imageFile47.png)

- 1. Use CNNs to make proposals

- 2. Introduced RPN (Region Proposal Network)

- 3. mAP: 78.8% for 2007 PASCAL VOC

- • RCNN à Look at every patch one by one

- • Fast R-CNN à Look once, and then inspect patches on feature map

- • Faster R-CNN à Propose patches using a neural network (RPN)

|Feature R-CNN Fast R-CNN Fater R-CNN|
|---|
|Region proposal Selective search Selective search RPN (learned)|
|CNN Usage Per region Once per image Once per image|
|Speed Very slow Faster Can work in realtime|
|Training Multi-stage, discreate<br><br>Partially en-to-end Fully end-to-end|
|Accuracy Good Better Best of all three|

## Object Detection Techniques History

![image 48](Week8-Lecture-2026_images/imageFile48.png)

Images source: https://arxiv.org/pdf/1807.05511.pdf

Image Annotation for Object Detection

|![image 49](Week8-Lecture-2026_images/imageFile49.png)<br><br>PASCAL VOC Format|
|---|

![image 50](Week8-Lecture-2026_images/imageFile50.png)

| |
|---|

Image Sources: https://en.wikipedia.org/wiki/Kangaroo#/media/File:RedRoo.JPG https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
