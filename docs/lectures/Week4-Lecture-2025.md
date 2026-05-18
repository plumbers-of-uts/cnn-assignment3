![image 1](Week4-Lecture-2025_images/imageFile1.png)

42028: Deep Learning and Convolutional Neural Network

# Week-4 Lecture

Neural Network in details

Outline

- • Logistic Regression Recap
- • Back Propagation
- • Gradient Descent and intuitions
- • Optimization techniques: SGD, RMSProp, Adam etc.
- • Activations Functions: Sigmoid, tanh, ReLu, Softmax
- • Logistic Regression with Back Propagation
- • Multi-Layered Neural Network

Logistic Regression – Recap

Function to calculate Loss/error

Mechanism to reduce the loss/error

###### Problem of Binary Classification:

Dog? → 1 Cat ( not Dog)? → 0

|Error/<br><br>Loss|
|---|

Gradient

![image 2](Week4-Lecture-2025_images/imageFile2.png)

### 1

Target

|Model<br><br>ANN Architecture + Parameters|
|---|

# 0

Output (y)

Input (x)

###### Activation function

Problem of Binary Classification → Logistic Regression (Dog ? → 1 | Not Dog? → 0)

|Parameters:<br><br>1. w (weight)<br>2. b (bias)<br>3. Output a= (𝒘𝑻𝒙+𝒃)<br>|
|---|

###### Loss function for Logistic Regression:

|L (a, y) =- 𝑦 log𝑎 + 1 − 𝑦 log(1 − 𝑎)<br><br>|
|---|

Logistic Regression pipeline with the math looks like:

###### X W B

|𝒘𝑻 𝒙 + 𝒃| |
|---|---|
| | |

|a = (𝒘𝑻 𝒙 + 𝒃)|
|---|

|L (a, y)|
|---|

###### L

Logistic Regression pipeline with the math looks like:

|Where,<br><br>W → Weights<br>X → Inputs b → Bias term<br><br><br> → Activation function|
|---|

###### X

###### ŷ

|𝒘𝑻 𝒙 + 𝒃| |
|---|---|
| | |

|a = (𝒘𝑻 𝒙 + 𝒃)| |
|---|---|
| | |

- W b

| |L (a, y)|
|---|---|
| | |

|Parameters:<br><br>1. w (weight)<br>2. b (bias)<br>3. Output a=(𝑤𝑇 𝑥 +𝑏)<br>|
|---|

Activation function

|a= = 1+𝑒1−𝑥<br><br>|
|---|

###### Loss function for Logistic Regression:

|L (a, y) =- 𝑦 log𝑎 + 1 − 𝑦 log(1 − 𝑎)<br><br>|
|---|

Logistic Regression pipeline with the math looks like:

###### X

Activation function

|a= = 1+𝑒1−𝑥<br><br>|
|---|

###### ŷ

|𝒘𝑻 𝒙 + 𝒃| |
|---|---|
| | |

|a = (𝒘𝑻 𝒙 + 𝒃)| |
|---|---|
| | |

- W b

| |L (a, y)|
|---|---|
| | |

If y = 1:L (a, y) =-log a

###### Loss function for Logistic Regression:

If y = 0:L (a, y) =-log (1– a)

|L (a, y) =- 𝑦 log𝑎 + 1 − 𝑦 log(1 − 𝑎)<br><br>|
|---|

Logistic Regression pipeline with the math looks like:

|Where,<br><br>W → Weights<br>X → Inputs b → Bias term<br><br><br> → Activation function|
|---|

###### X

|𝒘𝑻 𝒙 + 𝒃| |
|---|---|
| | |

|a = (𝒘𝑻 𝒙 + 𝒃)||L<br><br>|(a, y)|
|---|---|
| | |
|
|---|---|
| | |

|L<br><br>|(a, y)|
|---|---|
| | |

W b

Forward Pass

###### Back Propagation

|Parameters:<br><br>1. w (weight)<br>2. b (bias)<br>3. Output a=(𝑤𝑇 𝑥 +𝑏)<br>|
|---|

|repeatedly adjust the weights to minimize the difference between actual output and desired output|
|---|

Activation function

|a= = 1+𝑒1−𝑥<br><br>|
|---|

###### Loss function for Logistic Regression:

|L (a, y) =- 𝑦 log𝑎 + 1 − 𝑦 log(1 − 𝑎)<br><br>|
|---|

#### Optimization techniques

|Generic Algorithm:<br><br>Step 1: Initialize w and b<br>Step 2: Perform Forward pass operation/calculations<br>Step 3: Compute Loss/Cost function L (a, y)<br>Step 4: Compute change in w and b (Take the partial derivative of the cost function with<br><br>respect to Weights and bias (dw and db).<br><br>Step 5: Update w and b w := w – dw b := b – db<br>Step 6: Repeat from Step 2 with new values of w and b for ‘n’ number of iterations.<br>|
|---|

###### Gradient Descent for learning parameters: It is an iterative approach for error correction in a machine learning model.

###### Question: Find w and b that will minimize GD(w, b)

Required: Loss/cost function

![image 3](Week4-Lecture-2025_images/imageFile3.png)

###### (L)LossFn

|Example the loss function is:<br><br>L (a, y)=- 𝑦log𝑎 + 1 − 𝑦 log(1 − 𝑎)<br><br>| | |
|---|---|---|
| | → Learning rate| |

GDmin(w)

w

Image Source: https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788397872/1/ch01lvl1sec22/gradient-descent

Source and Reference: http://cs230.stanford.edu/files/C1M2.pdf

###### Gradient Descent for learning parameters: Learning rate() issues:

![image 4](Week4-Lecture-2025_images/imageFile4.png)

(L)LossFn

(L)LossFn

###### - It is a hyper-parameter

Image Source: https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788397872/1/ch01lvl1sec22/gradient-descent

Source and Reference: http://cs230.stanford.edu/files/C1M2.pdf

![image 5](Week4-Lecture-2025_images/imageFile5.png)

###### Learning rate(): more intuitions

Image Source: http://cs231n.github.io/neural-networks-3/

Gradient Descent Types

There are three main types of Gradient Descent Algorithms:

- 1. Batch Gradient Descent (BGD)
- 2. Stochastic Gradient Descent (SGD)
- 3. Mini-Batch Gradient Descent (MBGD)

Batch Gradient Descent (BGD)

###### Issues:

|Generic steps:<br><br>-Process each input sample and find the cost<br>-Find the average cost over all input samples<br>-Update w and b, and<br>-repeat the steps for ‘n’ epochs(iterations)<br>|
|---|

- 1. It uses the complete dataset to calculate the gradients at every steps
- 2. Slow when training set is large
- 3. Difficult to find the learning rate
- 4. Difficult to ascertain the number of epochs(iterations)

###### Advantage:

###### Stochastic → Random

- 1. Computes gradient based on single input sample: memory efficient
- 2. Much faster compared to BGD
- 3. Possible to train on large dataset
- 4. Randomness is a good escape from local minima problem

Due to the random nature, the

Algorithm is much less regular than

BGD

|Generic steps:<br><br>-Process a random input sample and find the cost<br>-Update w and b, and<br>-repeat the steps for ‘n’ iterations on the training samples<br>|
|---|

###### Issues:

1. Might not reach the optimal value,

but very close to it.

Issues: Might not reach the optimal value, but very close to it.

Possible solution: Reduce the learning rate gradually → Stimulated annealing

Create a Learning Schedule to determine the learning at each iteration.

Epoch: One round through the complete training set. Iterations: Process in multiple subsets of the training set, say, ‘m’ iterations

my form 1 epoch

Mini-Batch Gradient Descent (MBGD)

###### Advantage:

|Generic steps:<br><br>-Divide the training set into mini-batches (set of random samples on fixed number)<br>-Process all the samples in a Mini-batch and find the average cost<br>-Update w and b, and<br>-repeat the steps for ‘n’ iterations/epochs on the training samples<br>|
|---|

- 1. Computes gradient based on small sets of input sample
- 2. Much faster compared to BGD
- 3. Possible to train on large dataset
- 4. Performance boost on matrix operations using GPUs!
- 5. Might not reach the optimal value, but very close to it, and possibly better than SGD

###### Issues:

1. It may be harder to escape the local

minima.

## Gradient Descent (SGD) - intuition

![image 6](Week4-Lecture-2025_images/imageFile6.png)

![image 7](Week4-Lecture-2025_images/imageFile7.png)

Image Source: https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3

## Gradient Descent (SGD) – loss function nature

![image 8](Week4-Lecture-2025_images/imageFile8.png)

![image 9](Week4-Lecture-2025_images/imageFile9.png)

- • One of the popular algorithm for smoothing sequential data
- • Also called Moving Average
- • Weight the number of observations and using their average
- • Example: Temperatureover ‘n’ days Days

![image 10](Week4-Lecture-2025_images/imageFile10.png)

Temperature

Vt : Moving average on day ‘t’

![image 11](Week4-Lecture-2025_images/imageFile11.png)

![image 12](Week4-Lecture-2025_images/imageFile12.png)

So, let V0 = 0 V1 = 0.9 V0 + 0.1 1 V2 = 0.9 V1 + 0.1 2 V3 = 0.9 V2 + 0.1 3

Temperature

: : Vt = 0.9 Vt-1 + 0.1 t

Days

Vt = 0.9 Vt-1 + 0.1 t If  = 0.9,

![image 13](Week4-Lecture-2025_images/imageFile13.png)

![image 14](Week4-Lecture-2025_images/imageFile14.png)

Temperature

|Vt =  Vt-1 + (1- ) t|
|---|

This equation gives the moving average

shown by the red line.

Days

|Vt =  Vt-1 + (1- ) t|
|---|

![image 15](Week4-Lecture-2025_images/imageFile15.png)

![image 16](Week4-Lecture-2025_images/imageFile16.png)

Temperature

Vt is approximate average over 1−1

 days

So,  = 0.9 is closer to 10 days temperature  = 0.98 is closer to 50 days temperature  = 0.5 is closer to 2 days temperature

Days

What is Exponentially Weighted Averages doing?

Vt =  Vt-1 + (1- ) t

For, V100= 0.9 V99 + 0.1 100 V99= 0.9 V98 + 0.1 99

Substituting, V99 V100= 0.1 100+ 0.9 (0.9 V98 + 0.1 99) V100= 0.1 100+ 0.9 ( 0.1 99+ 0.9 (0.9 V97+ 0.1 V98)) ..

- • “Compute the Exponentially weighted average of the gradients and use that gradient to update weights” - Andrew NG
- • One of the most popular algorithms
- • Helps to accelerate the gradient vectors in right direction and reduces oscillation
- • Always faster than the SGD

|Algorithm: At iteration t:<br><br>Calculate 𝑑𝑤 𝑎𝑛𝑑 𝑑𝑏 on the current mini-batch<br><br>V𝑑𝑤 =  V𝑑w + (1 - ) 𝑑𝑤 ➔ Vt =  Vt-1 + (1- ) t<br><br>V𝑑𝑏=  V𝑑𝑏 + (1 - ) 𝑑𝑏 Update w and b:<br><br>w = w -  V𝑑𝑤 ,b = b -  V𝑑𝑏 Hyper-parameters: , |
|---|

![image 17](Week4-Lecture-2025_images/imageFile17.png)

![image 18](Week4-Lecture-2025_images/imageFile18.png)

SGD Without Momentum SGD With Momentum

Faster convergence and reduced oscillation

Image Source and reference http://ruder.io/optimizing-gradient-descent/index.html#momentum

- • Root Mean Square Propagation
- • Unpublished adaptive learning method by Geoffrey Hinton
- • RMSProp also reduces oscillation but in a different way than Momentum
- • RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

|Algorithm:<br><br>At iteration t:<br><br>Calculate 𝑑𝑤 𝑎𝑛𝑑 𝑑𝑏 on the current mini-batch S𝑑𝑤 = 2 S𝑑w + (1 - 2) 𝑑𝑤2 S𝑑𝑏= 2 S𝑑𝑏 + (1 - 2) 𝑑𝑏2<br><br>Update w and b:<br><br>w = w -  𝑑𝑤S<br><br>𝑑𝑤<br><br>, b = b -  𝑑𝑏S<br><br>𝑑𝑏<br><br>Squaring the derivatives<br><br>Square root of derivatives|
|---|

![image 19](Week4-Lecture-2025_images/imageFile19.png)

###### Intuition:

###### →Slow

S𝑑𝑤 → Smaller number expected S𝑑𝑏→ Larger number expected

b

###### W

So,

###### Fast →

w = w -  𝑑𝑤S

###### , b = b -  𝑑𝑏S

###### In Practice add ε :

𝑑𝑤

𝑑𝑏

w = w -  S𝑑𝑤

𝑑𝑤+ε , b = b -  S𝑑𝑏

|Smaller number So, w is larger|
|---|

Larger number So, b is small

𝑑𝑏+ ε

ε → small number, 10-8

- • Adam → Adaptive Moment Estimation
- • Combination of RMSProp and Momentum
- • Work well for a wide range of deep learning architecture

|Algorithm:<br><br>Initialize V𝑑𝑤 = 0, V𝑑𝑏= 0, S𝑑𝑤 = 0, S𝑑𝑏 = 0 At iteration t:<br><br>Calculate 𝑑𝑤 𝑎𝑛𝑑 𝑑𝑏 on the current mini-batch V𝑑𝑤 = 1 V𝑑w + (1 - 1) 𝑑𝑤, V𝑑𝑏= 1 V𝑑𝑏 + (1 - 1) 𝑑𝑏  From Momentum, 1 S𝑑𝑤 = 2 S𝑑w + (1 - 2) 𝑑𝑤2, S𝑑𝑏= 2 S𝑑𝑏 + (1 - 2) 𝑑𝑏2  From RMSProp, 2<br><br>Update w and b:<br><br>w = w -  V<br><br>𝑑𝑤<br><br>S𝑑𝑤+ε, b = b -  V<br><br>𝑑𝑏<br><br>S𝑑𝑏+ ε|
|---|

|In practice: Bias correction is required as V𝑑𝑤, V𝑑𝑏, S𝑑𝑤, S𝑑𝑏 are initialized to 0 and are biased towards zero. Hence, a bias correction is required as<br><br>follows:<br><br>V′𝑑𝑤 = V<br><br>𝑑w<br><br>( 1− 1 )<br><br>, V′𝑑b = V<br><br>𝑑b<br><br>(1− 1)<br><br>S′𝑑𝑤 = S<br><br>𝑑w<br><br>(1 − 2)<br><br>, S′𝑑b = S<br><br>𝑑b<br><br>(1 − 2)<br><br>Update w and b:<br><br>w = w -  V<br><br>′<br><br>𝑑𝑤<br><br>S′𝑑𝑤+ε , b = b -  V<br><br>′<br><br>𝑑𝑏<br><br>S′𝑑𝑏+ ε|
|---|

|https://vis.ensmallen.org/<br><br>Hyper parameter guide:<br><br> (Learning rate)→ should be tunned, start with 0.001<br><br>1(Momentum term) → 0.9 (dw)<br>2(moving weighted average) → 0.999 (dw2) ε → 10-8<br><br><br>Optimization Demo: https://vis.ensmallen.org/<br><br>|
|---|

Learning Rate Decay

###### Speed-up the learning algorithm by slowing decreasing the 𝛼 (Learning rate)

#### Activation Functions

Activation Functions: Sigmoid

|= 1+1𝑒−𝑥<br><br>|
|---|

|![image 20](Week4-Lecture-2025_images/imageFile20.png)|
|---|

Sigmoid function:

|Characteristics:<br><br>- Non-linear in nature<br>- Range(0, 1)<br>- Tends to bring the activations to either side of the curve: good for a classifier<br>- Suffers from vanishing gradient problem<br>|
|---|

###### Vanishing Gradient: Towards to the end of the curve, the value of Y change very less to the changes in X values. Hence gradient at the region will be very small. The network will refuse or learning extremely slowly.

Source: https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

Activation Functions: tanh

|![image 21](Week4-Lecture-2025_images/imageFile21.png)|
|---|

|Hyperbolic tangent:<br><br>tanh 𝑥 =<br><br>2 1 + 𝑒−2𝑥<br><br>− 1<br><br>|
|---|

|Characteristics:<br><br>- Non-linear in nature<br>- Range(-1, 1)<br>- Stronger gradient than sigmoid<br>- Also suffers from vanishing gradient problem<br>|
|---|

Activation Functions: ReLu

|![image 22](Week4-Lecture-2025_images/imageFile22.png)|
|---|

|Rectified Linear Unit (ReLu) 𝐴(𝑥) = max(0, x)|
|---|

i.e. : if x < 0, A(x) = 0, if x > 0, A(x) = x

|Characteristics:<br><br>- Non-linear in nature<br>- Range[0, inf]<br>- Stronger gradient than sigmoid<br>- Computationally less expensive than Sigmoid and Tanh<br>- Best used in hidden layers<br>- Dying ReLu problem<br>|
|---|

|Avoids and rectifies vanishing gradient problem|
|---|

Activation Functions: Leaky ReLu

|![image 23](Week4-Lecture-2025_images/imageFile23.png)<br><br>Leak|
|---|

|Leaky Rectified Linear Unit (Leaky ReLu)<br><br>𝐴(𝑥) = max(0.01𝑥,x)|
|---|

###### i.e. : if x < 0, A(x) = 0.01x, if x > 0, A(x) = x

|Characteristics:<br><br>- Non-linear in nature<br>- Range[0, inf]<br>- Leaky ReLUs are one attempt to fix the “dying ReLU” problem<br>|
|---|

Source & Reference : http://cs231n.github.io/neural-networks-1/

https://towardsdatascience.com/activation-functions-in-neural-networks-58115cda9c96

Activation Functions: Softmax

| |Softmax 𝑆 𝑦𝑖 = 𝑒𝑦𝑖 <br><br>𝑗 𝑒𝑦𝑗<br><br>for j = 1, …, K.| |
|---|---|---|
|Characteristics:<br><br>- Non-linear in nature<br>- Turns numbers in probabilities that sum to one.<br>- Useful when we have more than one output<br>- Used for classification in the output layer<br>- Less computationally expensive than Sigmoid and Tanh<br>| |Y|

###### Illustration:

###### = [ 2.0, 1.0, 0.1] Softmax(Y) = [0.7, 0.2, 0.1] (approx.)

Reference: https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d

Logistic Regression with Backpropagation

###### Logistic Regression pipeline with the math looks like:

Average cost over all training ‘m’ samples

X W

###### a = (𝒘𝑻 𝒙 + 𝒃)

| |L (a, y)<br><br>|
|---|---|
| | |

𝒘𝑻 𝒙 + 𝒃

|Avg Loss(J) =<br><br>𝟏<br><br>𝒎<br><br>෍<br><br>𝒊=𝟏<br><br>𝒎<br><br>L(ai,yi)<br><br>|
|---|

b

|Batch GD<br><br>Step 1: Initialize w and b<br>Step 2: Perform Forward pass operation/calculations<br><br><br>Step 2: Compute Loss/Cost function L (a, y)<br>Step 3: Find the average cost over all input samples (Take the partial derivative of the cost function with<br><br>respect to Weights and bias (dw and db).<br><br>Step 4: Update w and b w := w – dw b := b – db<br>Step 5: Repeat from Step 2 with new values of w and b for ‘n’ number of iterations.<br><br><br>|
|---|

|dw = 𝜕𝑤𝜕𝐽 , db = 𝜕𝑏𝜕𝐽<br><br>|
|---|

|𝑤 ≔ 𝑤 − dw<br><br>b := b – db|
|---|

|Size<br><br>#Bedroom<br><br>#Bathroom<br><br>Garden<br><br>Location|
|---|

###### Price

Y

X

Hidden Layer→ Adding more neurons in between input and output layer

![image 24](Week4-Lecture-2025_images/imageFile24.png)

Single layer perceptron 3-layered neural network with 2 hidden layers

Image Source: https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f

![image 25](Week4-Lecture-2025_images/imageFile25.png)

Example: 2-layered architecture for multi-class classification (e.g: Fashion MNIST dataset)

Intuition: In a multi-layer neural network, the first hidden layer will be able to learn some very simple patterns. Each additional hidden layer will somehow be able to learn progressively more complicated patterns.

###### Example: 2-layered architecture for multi-class classification (e.g: MNIST digit dataset) intuition

![image 26](Week4-Lecture-2025_images/imageFile26.png)
