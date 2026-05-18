![image 1](Week11-Lecture-2026_images/imageFile1.png)

![image 2](Week11-Lecture-2026_images/imageFile2.png)

42028: Deep Learning and Convolutional Neural Network

# Week-11 Lecture

Introduction to Sequence Modelling

Outline

- • Introduction to Sequence Modelling
- • Introduction to RNNs
- • Introduction to Attention mechanism
- • Introduction to Transformers
- • Case Studies:
- • Image Classification using transformer (ViT)
- • Object Detection using Transformer (RF-DETR)
- • Diffusion models

![image 3](Week11-Lecture-2026_images/imageFile3.png)

# ??

![image 4](Week11-Lecture-2026_images/imageFile4.png)

![image 5](Week11-Lecture-2026_images/imageFile5.png)

![image 6](Week11-Lecture-2026_images/imageFile6.png)

![image 7](Week11-Lecture-2026_images/imageFile7.png)

Predict where the ball will go next?

#### “This Sunday I went for a walk”

![image 8](Week11-Lecture-2026_images/imageFile8.png)

![image 9](Week11-Lecture-2026_images/imageFile9.png)

![image 10](Week11-Lecture-2026_images/imageFile10.png)

![image 11](Week11-Lecture-2026_images/imageFile11.png)

Sequence modelling types and applications

Y’

###### X

###### Many to Many Q&A with LLMs, Language translations

###### One to Many Image Captioning

###### One to One Binary classification

###### Many to One Sentiment Analysis

![image 12](Week11-Lecture-2026_images/imageFile12.png)

“Will it rain today?” Yes/No?

“42028 is the best subject so far!”

Me: “Hey Siri what's the weather today ?” Siri: “Its Evening now! Don’t ask boring Qs”

![image 13](Week11-Lecture-2026_images/imageFile13.png)

![image 14](Week11-Lecture-2026_images/imageFile14.png)

“A womenisthrowingfrisbee”

Size #Bedroom #Bathroom Garden Location

Price

###### Y

|House Price prediction|
|---|

![image 15](Week11-Lecture-2026_images/imageFile15.png)

Recurrent Neural Network (RNN) Basics

Output

y’t

y’0

y’1

y’2

h0 h1

###### ht

Recurrent cell

Xt

X0

X1

X2

Input

𝑦 = 𝑓(𝑥 , ℎ )

Output Input Past Memory/ state

Output Function+

y’t

Weights w

ℎ = 𝑓 (𝑥 , ℎ )

ht

RNN

Output Input Past state

###### Recurrence Relation

Xt

𝑦 = 𝑊 ℎ OutputVector

y’t

Output

ht

ℎ = 𝑡𝑎𝑛ℎ (𝑊 ℎ + 𝑊 𝑥 )

RNN

Update Hidden State

𝑥

Xt

Input Vector

Total Loss (L)

Forward Pass Backward Pass

L0 L1 Lt

…

y’0

y’1

y’t

y’t

𝑊

𝑊

𝑊

RNN ht

𝑊 𝑊

≈

RNN

RNN

RNN

𝑊

𝑊

𝑊

…

Xt

Xt

X0

X1

Sequence Modelling: Design Criteria

- • Support for Variable-Length Input

- • Has Temporal Dependency (Long-term, Short-term)

- • Preserve the information order

- • Share parameters across sequence

- RNN Limitations

- • Prone to vanishing and exploding gradient problem

- • Long term memory not supported

- • Slow and no parallelization

Output

y’0

y’1

y’2

y’t

y’t-2

y’t-1

…

#### Features

Input

X0

X1

X2

Xt

Xt-2

Xt-1

𝑦 𝑦 𝑦 𝑦 𝑦 𝑦

Output

…

Features

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

Input

Idea:

- - No Recurrence
- - No Long-term memory
- - Feed Everything into the Dense network
- - Identify and focus on what’s important

Attention Is All you Need!

![image 16](Week11-Lecture-2026_images/imageFile16.png)

- • Identify which parts of the image to focus on
- • Extract features with high attention

Why do we need Attention?

- • RNNs process sequences one step at a time
- • Long sentences lead to Long-term memory loss
- • Important words can be hidden in long dependencies
- • Attention helps to focus on relevant parts of the input

What is Attention? Intuition

- • Mimics human focus mechanism: “Where should I look?”
- • For each output word, attention decides which input word is most important
- • Computes a weighted sum of all input vectors
- • Higher weight = more important

###### Training Query (Q)

![image 18](Week11-Lecture-2026_images/imageFile18.png)

Key (K1)

###### Key (K2)

Compute similarity between Q and K

###### Key (K3)

![image 20](Week11-Lecture-2026_images/imageFile20.png)

###### Training Query (Q)

###### Value (V)

###### Key (K2)

Extract Values based On attention

- • Self-Attention is the foundation for Transformer architecture

- • Entire sequence is processed in parallel

- • Has Encode and a Decoder block

- • Stack of Layers with Self Attention and Feed Forward Neural

- • Self-Attention is the foundation for Transformer architecture

- • Entire sequence is processed in parallel

- • Has Encode and a Decoder block

- • Stack of Layers with Self Attention and Feed Forward Neural

## Case Studies

- • Introduced in 2021: "An Image is Worth 16\*16 Words: Transformers for Image Recognition at Scale," published at ICLR 2021

- • Vision transformer have extensive application in all computer vision tasks

- • ViT is a type of Deep Learning Model that’s looks at Images, like how language model looks at words

- • Images are represented as sequences of patches!

##### Steps:

- 1. Split an image into patches
- 2. Flatten the patches
- 3. Produce lower-dimensional linear embeddings from the flattened patches
- 4. Add positional embeddings
- 5. Feed the sequence as an input to a standard transformer encoder
- 6. Pretrain the model with image labels (fully supervised on a huge dataset)
- 7. Finetune on the downstream dataset for image classification

![image 21](Week11-Lecture-2026_images/imageFile21.png)

![image 22](Week11-Lecture-2026_images/imageFile22.png)

###### Reference: https://github.com/google-research/vision_transformer

###### Comparing CNN and Vision Transformer:

###### Key Aspects CNNs Vision Transformer (ViT) Input Handling

Processes the entire image using filters (kernels)

Splits image into fixed-size patches (like tokens)

Focuses on local patterns first (edges, textures)

Local vs. Global

Uses global self-attention to relate all patches

Hierarchical (convs → pools → deeper features)

Architecture

Flat transformer encoder stack Training Data Need Works well with limited data Needs lots of data or pretraining Computation Efficient with low-res inputs

Computationally heavier, especially on large images

Parallelism Limited; uses sequential feature stacking High; patch processing is highly parallelizable

- • Object Detection techniques using Transformers
- • An improvement over the original DETR (DEtection TRansformer)
- • DETR looks at everything globally but miss small things.
- • RF-DETR looks globally and understands the relationships between things.
- • Real-time Transformer-based object detection architecture
- • Outperforms all object detection models, 60+ mAP achieved on COCO dataset.

![image 23](Week11-Lecture-2026_images/imageFile23.png)

###### Source: ttps://blog.roboflow.com/rf-detr/

![image 24](Week11-Lecture-2026_images/imageFile24.png)

### Case Study: Diffusion Models - Intuition

![image 25](Week11-Lecture-2026_images/imageFile25.png)

Goal: Generate new data samples (e.g. images, audio, text) that is similar to a training dataset by learning to reverse a gradual noise process! A generative model!

##### Generic steps:

- 1. Start with real data
- 2. Add noise step-by-step, until the image becomes pure noise
- 3. Train a model to reverse this process: denoising to recover the original image!
- 4. Once trained, the model can start from pure noise and generate new and realistic samples.

![image 26](Week11-Lecture-2026_images/imageFile26.png)

### Diffusion Models

Example: Given a lot of sprite sample images

![image 27](Week11-Lecture-2026_images/imageFile27.png)

![image 28](Week11-Lecture-2026_images/imageFile28.png)

![image 29](Week11-Lecture-2026_images/imageFile29.png)

![image 30](Week11-Lecture-2026_images/imageFile30.png)

![image 31](Week11-Lecture-2026_images/imageFile31.png)

![image 32](Week11-Lecture-2026_images/imageFile32.png)

###### Training Data

Task: Generate New Sprite images (New Image generation from image input)

#### Example prompt: Butterfly image (Text to Image Generation) Task: Generate image of Butterfly!

![image 33](Week11-Lecture-2026_images/imageFile33.png)

#### • Forward Diffusion:

- • Add noise gradually to the original image for many steps
- • Iterate until the Image becomes pure noise
- • Gaussian noise used
- • No-learning

#### • Reverse Diffusion:

- • Denoising: Model is trained to predict and reverse this noise
- • Use the prediction to denoise the image
- • Given a noisy image, it predicts a slightly less noisy image version
- • After several steps, it reconstructs a clean and new image! From pure noise

Week 12 Guest Lecture

Industry Guest Lecture will be presented by Amazon Web Services (AWS) on Week 12

Topic: Build, Evaluate and Scale Production ready Agents

For more details please check Canvas!
