![image 1](Week10-Lecture-2026_images/imageFile1.png)

42028: Deep Learning and Convolutional Neural Network

# Week-10 Lecture

Anchor Free Object Detection, Instance/Semantic Segmentation

Outline

- • Recap: Anchor-based object detection
- • Anchor-free object detection overview
- • Case-Study: YoloX
- • Introduction to Instance Segmentation
- • Techniques and application of Instance Segmentation
- • Case Study: Mask R-CNN
- • Semantic Segmentation Introduction
- • Case Study: U-Net

Recap: Predicting Bounding Boxes

### Training Strategy:

![image 2](Week10-Lecture-2026_images/imageFile2.png)

- - Place a grid over the image
- - Apply image classification and localization to each of the grid cells Input:
- - Image: (ht x wd x 3) Target:
- - Bounding box information for each object
- - Class for each object

Class : {car, bike}

Images source: https://yallacompare.com/car-deals/uae/en/two-cars-one-dnajaguar-xe-300-sport-and-xe-sv-project-8/

Recap: YOLO: You Only Look Once Algorithm

![image 3](Week10-Lecture-2026_images/imageFile3.png)

###### Challenges with overlapping objects

- - Each grid cell detect only one object
- - For multiple overlapping objects, Mid point are on the same grid cell

![image 4](Week10-Lecture-2026_images/imageFile4.png)

###### So, Currently the Target Y = {1, x, y, h, w, C1, C2}, As the mid-points for both the objects are on the same grid cell, only one of the objects will be associated

Anchor Box 1 Anchor Box 2

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

###### Anchor Box 1

###### Associate each object to:

Predicted BB

- 1. A cell which contains its mid-point and
- 2. Anchor box for the cell with highest IoU

Anchor Box 1

Calculate the IoU of Anchor boxes and predicted BB

![image 5](Week10-Lecture-2026_images/imageFile5.png)

Anchor Box 1 Anchor Box 2

Similar Shape

###### So, with Anchor boxes: Target Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2},

![image 6](Week10-Lecture-2026_images/imageFile6.png)

![image 7](Week10-Lecture-2026_images/imageFile7.png)

Anchor Box 1 Anchor Box 2

Recap: YOLO: You Only Look Once Algorithm

| | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |![image 8](Week10-Lecture-2026_images/imageFile8.png)| | |

|Training Set|
|---|

- Anchor Box 1
- Anchor Box 2

|Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2} Cell(1,1) = {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?} : Cell(12,6)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(12,15)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(19,19)= {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?}|
|---|

InputX

Class : {1:car, 2:bike} Y size : (

X 2 X 7 )

|19 X 19|
|---|

|Grid Size|
|---|

|#Anchor Box|
|---|

|5(Po, x,y,h,w) + #Classes(2)|
|---|

Drawbacks of Anchor-based detectors

• Sensitive to:

|4|
|---|

|2|
|---|

|1|
|---|

2

- • Size
- • Aspect ratio
- • No. of Anchor boxes (fixed)
- • To much variation with shape
- • Small object
- • May not generalize due to pre-defined anchor boxes
- • Computation expensive

|3|
|---|

Anchor-free detector

Localize objects without using boxes as proposals Two board categories:

- 1. Key-point based

- 2. Center-based

- • Locates key object parts in an image

- • Detects spatial locations or points unique to an object

- • With human body as an example

- • Key Part of Face: nose, eyebrows etc.

- • Key point of human body: joints, elbows, etc.

- • Object is represented using Key-points

![image 9](Week10-Lecture-2026_images/imageFile9.png)

![image 10](Week10-Lecture-2026_images/imageFile10.png)

![image 11](Week10-Lecture-2026_images/imageFile11.png)

![image 12](Week10-Lecture-2026_images/imageFile12.png)

Anchor-free detector : Center-based

- • Finds positives in the centre
- • Predicts four distances from the positive to the potential object boundary

![image 13](Week10-Lecture-2026_images/imageFile13.png)

Reference and image source: https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/

YOLO Timeline

|Yolo X|
|---|

|Yolo - NAS|
|---|

Yolo v12

Yolo R

|Yolo v8|
|---|

##### Yolo v3

Yolo v1

2016

2020

2026

2022

2025

2023

2015

2018

2021

Yolo v2

Yolo v4 Scaled Yolo v4 Yolo v5

Yolo26

Yolo v6 Yolo v7

Yolo v9

- Yolo v10

2024

- Yolo v11

|Anchor Free|
|---|

- • YoloX: Exceeding the YOLO Series, 2021
- • Anchor-free detector in the Yolo Family
- • Decoupled head used
- • Label assignment using SimOTA
- • Uses YoloV3 SPP with DarkNet53 backbone
- • Uses advanced augmentation such as Mix-up & Mosaic

Source: https://arxiv.org/pdf/1807.05511.pdf

- Backbone Neck Head
- • Every Yolo Architecture consists of three parts: Backbone, neck, head
- • Backbone à Feature extraction
- • Neck à Aggregation of multi-scale feature
- • Headà Localization and Classification scores

## Case Study: YoloX, Decoupled head

![image 14](Week10-Lecture-2026_images/imageFile14.png)

| |
|---|

Source: https://arxiv.org/pdf/2107.08430.pdf

![image 15](Week10-Lecture-2026_images/imageFile15.png)

### Mixup Augmentation

![image 16](Week10-Lecture-2026_images/imageFile16.png)

### Mosaic Augmentation

## Case Study: YoloX, Performace

|![image 17](Week10-Lecture-2026_images/imageFile17.png)|
|---|

Source: https://arxiv.org/pdf/2107.08430.pdf

## Case Study: Yolo State-of-the-art, Performance

![image 18](Week10-Lecture-2026_images/imageFile18.png)

Source: https://docs.ultralytics.com/models/yolo11/#performance-metrics

Yolo26 – The Next Evolution

- • Real-time computer vision model by Ultralytics

- • Supports: Detection, Segmentation, Classification, Pose, Tracking, OBB

- • Available in Nano, Small, Medium, Large, XLarge

- • End-to-end detection pipeline (NMS-free)

- • Designed for edge AI and fast deployment

- Yolo26 – Why is it Faster?

- • NMS-free inference removes post-processing overhead

- • Direct bounding box regression (no DFL)

- • Lower latency and simpler deployment graph

- • CPU-optimized architecture

- • Up to 43% faster on CPUs than YOLO11 (Ultralytics benchmark)

Yolo26 – Key Changes

- • ProgLoss (Progressive Loss Balancing) improves training stability and convergence
- • STAL (Small-Target-Aware Label Assignment) improves small-object detection
- • MuSGD optimizer improves convergence speed
- • Better speed–accuracy trade-off than many previous YOLO models
- • Ideal for robotics, drones, surveillance, and edge devices

![image 19](Week10-Lecture-2026_images/imageFile19.png)

## Yolo26 – High Level Architecture (Inference)

![image 20](Week10-Lecture-2026_images/imageFile20.png)

## Yolo26 – Training Pipeline

![image 21](Week10-Lecture-2026_images/imageFile21.png)

## Yolo26 – Performance

![image 22](Week10-Lecture-2026_images/imageFile22.png)

## Instance Segmentation

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

![image 23](Week10-Lecture-2026_images/imageFile23.png)

![image 24](Week10-Lecture-2026_images/imageFile24.png)

![image 25](Week10-Lecture-2026_images/imageFile25.png)

![image 26](Week10-Lecture-2026_images/imageFile26.png)

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

![image 27](Week10-Lecture-2026_images/imageFile27.png)

![image 28](Week10-Lecture-2026_images/imageFile28.png)

Single Object Multiple Object

Images source: cs224d course

Semantic Segmentation Vs Instance Segmentation:

- • Semantic segmentation classifies object pixels to specific classes/category
- • Instance Segmentation identifies each pixels object instance

![image 29](Week10-Lecture-2026_images/imageFile29.png)

Input Image Semantic Segmentation Instance Segmentation

##### Popular Techniques:

|Semantic Segmentation|Instance Segmentation|
|---|---|
|Conditional Random Field (CRF) Fully Convolutional Network (FCN) U-Net Pyramid Scene Parsing Network (PSPNet) etc.|SegNet, DeepMask, SharpMask, MaskRCNN, etc.|

##### Applications: Autonomous Driving

![image 30](Week10-Lecture-2026_images/imageFile30.png)

Images source: https://slideplayer.com/slide/784090/3/images/2/Dense+CRF+construction.jpg

##### Applications: Scene Understanding

![image 31](Week10-Lecture-2026_images/imageFile31.png)

Images source: https://www.topbots.com/semantic-segmentation-guide/

##### Applications: Aerial Image processing

![image 32](Week10-Lecture-2026_images/imageFile32.png)

Images source: https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/59cbe15b43e6ca172fce40786be68340f50be541/12-Figure1.1-1.png

- • Mask-RCNN à Mask-Region Convolutional Neural Network
- • An addition to the RCNN family, performing instance segmentation
- • Improved over FasterRCNN
- • A Full Convolutional Network (FCN) for predicting Mask for each class/object.
- • 2 Stages:
- • Stage 1: RPN proposes candidate object bounding boxes.
- • Stage 2: Classify the Candidates, refine bounding boxes, and predict mask.

![image 33](Week10-Lecture-2026_images/imageFile33.png)

##### Faster-RCNN

###### Source and Reference: http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf

https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852

![image 34](Week10-Lecture-2026_images/imageFile34.png)

![image 35](Week10-Lecture-2026_images/imageFile35.png)

###### • Sample Results

![image 36](Week10-Lecture-2026_images/imageFile36.png)

![image 37](Week10-Lecture-2026_images/imageFile37.png)

![image 38](Week10-Lecture-2026_images/imageFile38.png)

###### • Sample Results on video:

![image 39](Week10-Lecture-2026_images/imageFile39.png)

Images source: https://github.com/matterport/Mask_RCNN

## Mask R-CNN Limitations

- • Computational Complexity: Training and inference can be computationally intensive, requiring substantial resources, especially for high-resolution images or large datasets.
- • Small-Object Segmentation: may struggle to accurately segment very small objects due to limited pixel information.
- • Data Requirements: Training requires a large amount of annotated data, which can be time-consuming and expensive to acquire.
- • Limited Generalization to Unseen Categories: The model's ability to generalize to unseen object categories is limited.

Images source: https://blog.roboflow.com/mask-rcnn/

## Semantic Segmentation

Introduction to Semantic Segmentation

Semantic segmentation classifies object pixels on specific classes/category

![image 40](Week10-Lecture-2026_images/imageFile40.png)

![image 41](Week10-Lecture-2026_images/imageFile41.png)

###### Input Image Semantic Segmentation Instance Segmentation

Images source: https://datascience.stackexchange.com/questions/52015/what-is-the-difference-between-semantic-segmentation-object-detection-and-insta

![image 42](Week10-Lecture-2026_images/imageFile42.png)

![image 43](Week10-Lecture-2026_images/imageFile43.png)

## Semantic Segmentation: UNet

00000000000001100000000000000000000000 00000000000001110000000011000000000000 00000000000011111111111111000000000000 00000000000011111111111111000000000000 00000000000011111111111110000000000000 00000000000011111111111111100000000000 00000000000011111111111111110000000000 00000000000011111111111111110000000000 00000000000001111111111111110000000000 00000000000000111111111111100000000000 00000000000000011111111111000000000000 00000000000000011111111111000000000000 00000000000000011111111100000000000000

![image 44](Week10-Lecture-2026_images/imageFile44.png)

![image 45](Week10-Lecture-2026_images/imageFile45.png)

###### CAT

![image 46](Week10-Lecture-2026_images/imageFile46.png)

…

{

{

{

![image 47](Week10-Lecture-2026_images/imageFile47.png)

![image 48](Week10-Lecture-2026_images/imageFile48.png)

![image 49](Week10-Lecture-2026_images/imageFile49.png)

## Semantic Segmentation: UNet Architecture

![image 50](Week10-Lecture-2026_images/imageFile50.png)

![image 51](Week10-Lecture-2026_images/imageFile51.png)

![image 52](Week10-Lecture-2026_images/imageFile52.png)

Images source: https://arxiv.org/pdf/1505.04597.pdf

cs224d course
