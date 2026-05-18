![image 1](Week1-Lecture-2026_images/imageFile1.png)

![image 2](Week1-Lecture-2026_images/imageFile2.png)

42028: Deep Learning and Convolutional Neural Network

#### Week 1

Introduction to Machine Learning and Deep Learning

Outline

- •Introduction to AI, ML, CV, & DL
- •Popular use cases
- •The Deep Learning Evolution
- •AI, ML, DL Relationship
- •Features in machine - example
- •ML/DL Pipeline
- •Deep Learning and CNN @ UTS

![image 3](Week1-Lecture-2026_images/imageFile3.png)

![image 4](Week1-Lecture-2026_images/imageFile4.png)

What is Artificial Intelligence?

Human Intelligence exhibited by machine!

What is Artificial Intelligence?

- • A generic term for getting computers to perform human tasks, and the scope is always changing overtime.

- • We don’t have a generic AI system which does multiple human tasks!

- • The systems available today are able to perform one or few well defined tasks, which are at par with the human performance or sometimes better!

- Popular Use Cases

- • Image Classification

- • Object Detection and Recognition

- • Image Captioning

- • Face Detection and Recognition

- • Biometrics (Fingerprint, Retina, Hand Geometry, etc.)

- • Speech Recognition

- • Natural Language Processing (NLP)

- • Language Translations

- • Creative (learn to draw an image in the style of an artist!) :

###### • Speech Recognition

![image 5](Week1-Lecture-2026_images/imageFile5.png)

###### Hey Google… What’s the weather today? …

![image 6](Week1-Lecture-2026_images/imageFile6.png)

![image 7](Week1-Lecture-2026_images/imageFile7.png)

![image 8](Week1-Lecture-2026_images/imageFile8.png)

![image 9](Week1-Lecture-2026_images/imageFile9.png)

Image Source: https://medium.com/@joelgarciajr84/creating-an-application-that-uses-speech-recognition-76117a396b7d

###### • Speech Recognition - Technology Challenges!

![image 10](Week1-Lecture-2026_images/imageFile10.png)

###### • Natural Language Processing

![image 11](Week1-Lecture-2026_images/imageFile11.png)

“Beware though, bots have the illusion of simplicity on the front end but there are many hurdles to overcome to create a great experience. So much work to be done. Analytics, flow optimization, keeping up with ever changing platforms that have no standard. For deeper integrations and real commerce like Assist powers, you have error checking, integrations to APIs, routing and escalation to live human support, understanding NLP, no back buttons, no home button, etc etc. We have to unlearn everything we learned the past 20 years to create an amazing experience in this new browser.” —Shane Mac, CEO of Assist

Reference: https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32

![image 12](Week1-Lecture-2026_images/imageFile12.png)

###### • ChatGPT!

###### • Language translations

![image 13](Week1-Lecture-2026_images/imageFile13.png)

![image 14](Week1-Lecture-2026_images/imageFile14.png)

![image 15](Week1-Lecture-2026_images/imageFile15.png)

What is Machine Learning?

“Machine Leaning is the field of study that gives computer ability to learn without being explicitly programmed”

###### 1Machine Learning is a Science ( and art ) of programming computers so that they can learn from Data!

– Arthur Samuel, 1958

Why and When to use Machine Learning?

Problems for which existing solutions require a lot of hand-tuning or a long list of rules

Complex Problems for which there is no good solution at all using traditional approach

|![image 16](Week1-Lecture-2026_images/imageFile16.png)|
|---|

|![image 17](Week1-Lecture-2026_images/imageFile17.png)|
|---|

Fluctuating environments: Machine Learning systems can adapt on new data

Getting insight about complex problems and a large amount of data

|![image 18](Week1-Lecture-2026_images/imageFile18.png)|
|---|

|![image 19](Week1-Lecture-2026_images/imageFile19.png)|
|---|

- •Insufficient amount of training data

- •Non-representative training data

- •Poor Quality data

- •Irrelevant Features!: Garbage in à Garbage Out!

- •Overfitting the training data

- •Under fitting the training data

- •More of the challenges are around Data!

- •Data or Algorithm, which is more important?

- •Check:

###### 1Unreasonable Effectiveness of data 2Revisiting the Unreasonable Effectiveness of Data

Reference: 1https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf 2https://ai.googleblog.com/2017/07/revisiting-unreasonable-effectiveness.html

###### •Overfitting example

![image 20](Week1-Lecture-2026_images/imageFile20.png)

Image source: https://www.reddit.com/r/ProgrammerHumor/comments/8p96r8/yes/

###### • Overfitting and Underfitting

This is too complex.. Skip!

![image 21](Week1-Lecture-2026_images/imageFile21.png)

![image 22](Week1-Lecture-2026_images/imageFile22.png)

![image 23](Week1-Lecture-2026_images/imageFile23.png)

![image 24](Week1-Lecture-2026_images/imageFile24.png)

![image 25](Week1-Lecture-2026_images/imageFile25.png)

![image 26](Week1-Lecture-2026_images/imageFile26.png)

Not Interested in learning Open Book Exam: 45% Closed Book Exam: 35%

Memorizing everything Open Book Exam: 98% Closed Book Exam: 55%

Learning concept well with examples Open Book Exam: 93% Closed Book Exam: 85%

###### Underfitting/not learning

###### Overfitting

###### Best-Fit

![image 27](Week1-Lecture-2026_images/imageFile27.png)

![image 28](Week1-Lecture-2026_images/imageFile28.png)

Computer Vision

How computers see and understand digital images and videos.

![image 29](Week1-Lecture-2026_images/imageFile29.png)

###### Human Brain

![image 30](Week1-Lecture-2026_images/imageFile30.png)

Apple, Pear, grapes, banana, oranges, basket

###### Human Eye

![image 31](Week1-Lecture-2026_images/imageFile31.png)

![image 32](Week1-Lecture-2026_images/imageFile32.png)

![image 33](Week1-Lecture-2026_images/imageFile33.png)

Apple, Pear, grapes, banana, oranges, basket

###### Input image Output

###### Webcam image sensor

###### Interpreting device Computer

Computer Vision

Computer vision includes all tasks performed by the biological vision system:

- • Eye/Retina à Camera/Webcam
- • Extracting information à Image Processing
- • understanding what is seen à Image Analysis and Understanding/ML

Applications

![image 34](Week1-Lecture-2026_images/imageFile34.png)

![image 35](Week1-Lecture-2026_images/imageFile35.png)

![image 36](Week1-Lecture-2026_images/imageFile36.png)

Assistance to differently abled humans (bionic eye)

Unmanned Surveillance using Drone

###### Image search engines

![image 37](Week1-Lecture-2026_images/imageFile37.png)

![image 38](Week1-Lecture-2026_images/imageFile38.png)

Human machine interaction/ Robotics

Autonomous driving

Computer Vision: Popular tasks

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

![image 39](Week1-Lecture-2026_images/imageFile39.png)

![image 40](Week1-Lecture-2026_images/imageFile40.png)

![image 41](Week1-Lecture-2026_images/imageFile41.png)

![image 42](Week1-Lecture-2026_images/imageFile42.png)

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

![image 43](Week1-Lecture-2026_images/imageFile43.png)

![image 44](Week1-Lecture-2026_images/imageFile44.png)

Single Object Multiple Object

Images source: cs224d course

- • Image Captioning (Computer Vision + NLP)

• Face Detection and Recognition

![image 45](Week1-Lecture-2026_images/imageFile45.png)

![image 46](Week1-Lecture-2026_images/imageFile46.png)

![image 47](Week1-Lecture-2026_images/imageFile47.png)

###### • Biometrics (Fingerprint, Retina, Hand Geometry, etc.) (Computer Vision)

![image 48](Week1-Lecture-2026_images/imageFile48.png)

![image 49](Week1-Lecture-2026_images/imageFile49.png)

![image 50](Week1-Lecture-2026_images/imageFile50.png)

![image 51](Week1-Lecture-2026_images/imageFile51.png)

###### • Creative

###### This are fake images! à Generated using GAN

Image source: https://research.nvidia.com/publication/2017-10_Progressive-Growing-of

Generative AI

Definition:

- •Refers to the use of AI to create new content such as text, images, audio/music, and videos.
- •Examples: LLMs, ChatGPT, Bard etc. are examples of Gen AI designed for conversational purpose, producing human like responses.

Reference: https://cloud.google.com/use-cases/generative-ai

Deep Learning

Definition:

- •It is a class of machine learning algorithms that uses multiple layers to progressively extract higher level of features from the raw input.
- •The word “Deep” in deep learning refers to the number of layers through which data is transformed.

Reference: https://en.wikipedia.org/wiki/Deep_learning

![image 52](Week1-Lecture-2026_images/imageFile52.png)

The Deep Learning Evolution

- - Slow computers
- - Less data

|Deep Learning is a technique for implementing Machine Learning! also know as Deep Neural Networks (DNNs)|
|---|

So, What Changed Overtime?

|Availability of faster computers! Cheap and fast GPUs<br><br>![image 53](Week1-Lecture-2026_images/imageFile53.png)|
|---|

|Very large datasets, Easy to collect and store<br><br>![image 54](Week1-Lecture-2026_images/imageFile54.png)<br><br>![image 55](Week1-Lecture-2026_images/imageFile55.png)|
|---|

Improved libraries, toolboxes, modern architectures!

![image 56](Week1-Lecture-2026_images/imageFile56.png)

![image 57](Week1-Lecture-2026_images/imageFile57.png)

![image 58](Week1-Lecture-2026_images/imageFile58.png)

![image 59](Week1-Lecture-2026_images/imageFile59.png)

![image 60](Week1-Lecture-2026_images/imageFile60.png)

![image 61](Week1-Lecture-2026_images/imageFile61.png)

![image 62](Week1-Lecture-2026_images/imageFile62.png)

###### Keras

![image 63](Week1-Lecture-2026_images/imageFile63.png)

![image 64](Week1-Lecture-2026_images/imageFile64.png)

AI, ML and DL relationship!

Artificial Intelligence

Machine Learning

Deep Learning

explicitprogramming

![image 65](Week1-Lecture-2026_images/imageFile65.png)

![image 66](Week1-Lecture-2026_images/imageFile66.png)

programsintelligent

![image 67](Week1-Lecture-2026_images/imageFile67.png)

Makingmachine&

Learnwithoutany

LearnusingDeep

NeuralNetworks

1950s 1980s 2010s

Orange? Yes-40%

Orange? Yes-85%

No-55% Unsure- 5%

No-14% Unsure- 1%

Weight: 150 grams Colour:Colour: OrangeOrange?

![image 68](Week1-Lecture-2026_images/imageFile68.png)

![image 69](Week1-Lecture-2026_images/imageFile69.png)

![image 70](Week1-Lecture-2026_images/imageFile70.png)

![image 71](Week1-Lecture-2026_images/imageFile71.png)

|Feature dimension: 2|
|---|

###### Orange Apple

Weight

###### ?

|Choosing appropriate and useful features can have a significant impact on the performance of a Machine Learning system!|
|---|

![image 72](Week1-Lecture-2026_images/imageFile72.png)

Colour

Typical Machine Learning Pipeline

|Data/ Features<br><br>![image 73](Week1-Lecture-2026_images/imageFile73.png)| |
|---|---|
| | |

|Launch|
|---|

![image 74](Week1-Lecture-2026_images/imageFile74.png)

|Train ML Algorithm| |
|---|---|
| | |

|Study the Problem| |
|---|---|
| | |

Evaluate Solution

![image 75](Week1-Lecture-2026_images/imageFile75.png)

| |Analyse errors|
|---|---|
| | |

Reference: Hands-On Machine Learning with Scikit-Learn and TensorFlow

Traditional ML Vs DL Pipeline

###### Traditional Machine Learning (ML) pipeline for object detection and classification

![image 76](Week1-Lecture-2026_images/imageFile76.png)

![image 77](Week1-Lecture-2026_images/imageFile77.png)

###### Result

###### Input video

![image 78](Week1-Lecture-2026_images/imageFile78.png)

![image 79](Week1-Lecture-2026_images/imageFile79.png)

###### Result

###### Input video

End-to-End Deep Learning (DL) technique for Object Detection and Classification

Deep Learning Pipeline example

![image 80](Week1-Lecture-2026_images/imageFile80.png)

|More layers that loosely mimic human brain|
|---|

|No explicit feature engineering|
|---|

![image 81](Week1-Lecture-2026_images/imageFile81.png)

##### Deep Learning System Pipeline

| |
|---|

## Student Projects from previous iterations of 42028!

![image 82](Week1-Lecture-2026_images/imageFile82.png)

### KrossConnection

![image 83](Week1-Lecture-2026_images/imageFile83.png)

![image 84](Week1-Lecture-2026_images/imageFile84.png)

![image 85](Week1-Lecture-2026_images/imageFile85.png)

![image 86](Week1-Lecture-2026_images/imageFile86.png)

![image 87](Week1-Lecture-2026_images/imageFile87.png)

![image 88](Week1-Lecture-2026_images/imageFile88.png)

![image 89](Week1-Lecture-2026_images/imageFile89.png)

![image 90](Week1-Lecture-2026_images/imageFile90.png)

![image 91](Week1-Lecture-2026_images/imageFile91.png)

![image 92](Week1-Lecture-2026_images/imageFile92.png)

![image 93](Week1-Lecture-2026_images/imageFile93.png)

![image 99](Week1-Lecture-2026_images/imageFile99.png)

### SignSync

![image 100](Week1-Lecture-2026_images/imageFile100.png)

![image 101](Week1-Lecture-2026_images/imageFile101.png)

![image 102](Week1-Lecture-2026_images/imageFile102.png)

![image 103](Week1-Lecture-2026_images/imageFile103.png)

![image 104](Week1-Lecture-2026_images/imageFile104.png)

![image 105](Week1-Lecture-2026_images/imageFile105.png)

![image 106](Week1-Lecture-2026_images/imageFile106.png)

![image 107](Week1-Lecture-2026_images/imageFile107.png)

![image 108](Week1-Lecture-2026_images/imageFile108.png)

![image 109](Week1-Lecture-2026_images/imageFile109.png)

![image 110](Week1-Lecture-2026_images/imageFile110.png)

![image 116](Week1-Lecture-2026_images/imageFile116.png)

### GestureFly

![image 117](Week1-Lecture-2026_images/imageFile117.png)

![image 118](Week1-Lecture-2026_images/imageFile118.png)

# Deep Learning Projects @UTS!

![image 119](Week1-Lecture-2026_images/imageFile119.png)

Signature and Logo detection

![image 120](Week1-Lecture-2026_images/imageFile120.png)

|![image 121](Week1-Lecture-2026_images/imageFile121.png)|
|---|

| |
|---|

Logo and Signature detection result

##### Drone detection for Security and Surveillance

![image 122](Week1-Lecture-2026_images/imageFile122.png)

![image 123](Week1-Lecture-2026_images/imageFile123.png)

###### The Award winning

![image 124](Week1-Lecture-2026_images/imageFile124.png)
