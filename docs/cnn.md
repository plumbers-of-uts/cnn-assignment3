# 42028: Deep Learning and Convolutional Neural Networks — Lecture Notes (Week 1–11)

> Merged from individual weekly lecture PDFs converted via `opendataloader-pdf`.
> Text-only version. Original images live under `docs/lectures/Week<N>-Lecture-YYYY_images/`.

______________________________________________________________________

## Table of Contents

- [Week 1 — Introduction to ML and DL](#week-1--introduction-to-ml-and-dl)
- [Week 2 — Lecture](#week-2--lecture)
- [Week 3 — Lecture](#week-3--lecture)
- [Week 4 — Lecture](#week-4--lecture)
- [Week 5 — Lecture](#week-5--lecture)
- [Week 6 — Lecture](#week-6--lecture)
- [Week 7 — Lecture](#week-7--lecture)
- [Week 8 — Lecture](#week-8--lecture)
- [Week 9 — Lecture](#week-9--lecture)
- [Week 10 — Lecture](#week-10--lecture)
- [Week 11 — Lecture](#week-11--lecture)

______________________________________________________________________

## Week 1 — Introduction to ML and DL

*42028: Deep Learning and Convolutional Neural Network — Introduction to Machine Learning and Deep Learning*

Outline

- Introduction to AI, ML, CV, & DL
- Popular use cases
- The Deep Learning Evolution
- AI, ML, DL Relationship
- Features in machine - example
- ML/DL Pipeline
- Deep Learning and CNN @ UTS

### What is Artificial Intelligence?

Human intelligence exhibited by machines:

- A generic term for getting computers to perform human tasks; the scope changes over time.
- There is no generic AI system that can do all human tasks.
- Today's systems can perform one or a few well-defined tasks at (or beyond) human performance.

### Popular AI/ML use cases

- Image classification
- Object detection and recognition
- Image captioning
- Face detection and recognition
- Biometrics (fingerprint, retina, hand geometry, …)
- Speech recognition (e.g. "Hey Google, what's the weather today?") — challenges include accents, noise, language variation
- Natural language processing (NLP), incl. chatbots and ChatGPT
- Language translation
- Creative AI (learning to paint in the style of an artist)

### What is Machine Learning?

> "Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed."
> — Arthur Samuel, 1958

A more modern phrasing: **Machine Learning is the science (and art) of programming computers so that they can learn from data.**

### When to use Machine Learning

- Problems for which existing solutions require a lot of hand-tuning or a long list of rules
- Complex problems with no good traditional solution
- Fluctuating environments (ML systems adapt to new data)
- Gaining insight from complex problems with large amounts of data

### Main challenges of Machine Learning

- Insufficient training data
- Non-representative training data
- Poor-quality data
- Irrelevant features ("garbage in → garbage out")
- Overfitting / underfitting the training data

Most of these challenges are around the **data**. The classic question — *"data or algorithm, which matters more?"* — is largely answered by *The Unreasonable Effectiveness of Data* and its 2017 revisit by Google Research: more data tends to dominate algorithmic improvements past a point.

### Overfitting vs. underfitting — open-book exam analogy

| Strategy | Open-book | Closed-book | Fit type |
|-----------------------------------------|-----------|-------------|--------------|
| Not interested in learning | 45% | 35% | Underfitting |
| Memorising everything | 98% | 55% | Overfitting |
| Learning the concept well with examples | 93% | 85% | Best fit |

### Computer Vision

How computers see and understand digital images and videos. Computer vision mimics the tasks of the biological vision system:

- Eye / retina → camera / webcam
- Extracting information → image processing
- Understanding what is seen → image analysis & ML

**Applications**: assistance to people with disabilities (bionic eye), drone surveillance, image search engines, human–robot interaction, autonomous driving.

**Popular tasks**: image classification, object detection + localization, instance segmentation, image captioning (CV + NLP), face detection & recognition, biometrics, creative AI (e.g. NVIDIA's GAN-generated portraits are entirely synthetic).

### Generative AI

- Refers to the use of AI to create new content — text, images, audio/music, video.
- Examples: LLMs, ChatGPT, Bard — conversational Gen AI that produces human-like responses.

### Deep Learning

- A class of ML algorithms that uses multiple layers to progressively extract higher-level features from raw input.
- "Deep" refers to the number of layers through which data is transformed.
- Also known as deep neural networks (DNNs); a technique for implementing ML.

### The Deep Learning evolution

DL was held back by slow computers and small datasets. What changed:

- Faster, cheaper GPUs
- Very large datasets, easy to collect and store
- Improved libraries, toolboxes, and modern architectures (e.g. Keras)

### AI / ML / DL relationship

**AI → ML → DL** progression (1950s → 1980s → 2010s): ML *makes machines learn without explicit programming*; DL *learns using deep neural networks*.

### Features in ML — example

Classifying *Orange* vs *Apple*:

- Possible features: weight (e.g. 150 g) and colour.
- Choosing appropriate, useful features has a significant impact on the performance of an ML system.

### Typical ML pipeline

`Study the problem → Collect data / features → Train ML algorithm → Evaluate solution → Analyse errors → Launch`.

### Traditional ML vs. Deep Learning pipeline

- **Traditional ML pipeline** for object detection / classification: hand-crafted feature engineering → classifier → result.
- **End-to-end Deep Learning**: input image/video → deep network learns features and classifier together → result.
- DL stacks many layers that loosely mimic the brain and removes the need for explicit feature engineering.

### Student Projects from previous iterations of 42028

- KrossConnection
- SignSync
- GestureFly

### Deep Learning Projects @ UTS

- Signature and Logo detection
- Drone detection for Security and Surveillance (award winning)

______________________________________________________________________

## Week 2 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Machine Learning and Image Processing Basics*

Outline

- Types of Machine Learning System
- Supervised and Un-supervised learning
- Support Vector Machine (SVM)
- Evaluation Metrics
- Image Processing Basics, Types
- Edge Detection using Convolution

### Machine Learning Basics

**Types of ML system** can be classified along three axes:

- **By supervision**: supervised / unsupervised / semi-supervised / reinforcement learning.
- **By learning mode**: batch (offline) vs online (learns on the fly).
- **By generalisation strategy**: instance-based (compares to known examples) vs model-based (builds a predictive model).

### Supervised learning

Labelled data for training (object + desired output label). Two sub-tasks:

- **Classification** — predict a discrete class (e.g. *Apple* vs *Pear* vs *Mango*).
- **Regression** — predict a continuous value (e.g. house price from size in sq. ft).

Important algorithms: k-Nearest Neighbours, Logistic Regression, Support Vector Machines (SVMs), Neural Networks (some variants are unsupervised).

**Examples**: image classification, object recognition, predictions / forecasting, fraud detection, medical diagnostics, process optimisation.

### Unsupervised learning

Unlabelled data for training; model finds structure or groups in the data.

- Typical task: **clustering** (e.g. group similar fruits).
- Outputs new insights — useful where labels are unavailable or expensive.

Important algorithms: k-means, Expectation–Maximization.

### Support Vector Machine (SVM)

- A powerful, versatile ML model for linear or non-linear classification, regression, and outlier detection.
- Defined by a separating hyperplane that fairly separates the classes.
- Suitable for small or medium-sized datasets.

Hyper-parameters: `kernel`, `gamma`, regularization `C`. A low `C` allows a softer margin (more mis-classifications, simpler model); a high `C` forces a hard margin (fewer mis-classifications, more complex model). Example use: `sklearn.svm.SVC` on the Iris flower dataset.

### Evaluation Metrics — Precision, Recall, IoU

Confusion matrix:

| | No Cancer | Cancer |
|----------|-----------|--------|
| **Pred. No Cancer** | TN | FN |
| **Pred. Cancer** | FP | TP |

- **TN** — patients without cancer, correctly diagnosed as no cancer.

- **TP** — patients with cancer, correctly diagnosed as cancer.

- **FN** — patients with cancer, missed by the model.

- **FP** — patients without cancer, incorrectly flagged as cancer.

- `Precision = TP / (TP + FP)` — of the cases predicted positive, how many are truly positive.

- `Recall = TP / (TP + FN)` — of the truly positive cases, how many the model caught.

**Intersection over Union (IoU)** is the metric for object detectors — it measures how well the predicted bounding box overlaps with the ground-truth box (`IoU = area_intersection / area_union`).

### Image Processing Basics

Image Processing Basics

**What is a digital image?**

- Digital images are made of picture elements called Pixels.
- It is an array, or a matrix of Pixels arranges in columns and rows.
- Each Pixel has its own intensity value, or brightness
- Intensity values in digital images are defined by bits
- For a standard 8-bit image, a pixel can have 2^8 = 256 (0 – 255) values.
- Black & White images have a single 8-bits intensity range.

How does a computer see an image? As a matrix — e.g. the digit '8' on a `24 × 16` grid is just a `24 × 16` matrix of pixel intensities.

#### Colour Images

Example 5×5 grayscale image (8-bit):

|170|170| 55|170|170|
|---|---|---|---|---|
|170| 55|170| 55|170|
| 55|140|140|140| 55|
| 55|170|170|170| 55|

So an image is a 2D function `f(x, y)`, where `x` and `y` are spatial coordinates and `f(x, y)` is the intensity / grey level at that pixel (e.g. `f(2, 3) = 170` in the matrix above).

A colour image adds channels: a `5 × 5 × 3` (24-bit) colour image stores **three** 8-bit channels (R, G, B), each a `5 × 5` matrix of the same shape. RGB has `3 × 8 = 24 bits/pixel` ⇒ `2²⁴ ≈ 16,777,216` (~16M) possible colours per pixel.

Image Processing - Types

1. Image Enhancement
1. Image Restoration
1. Image Segmentation
1. Image Recognition & Classification
1. Image Compression
1. Image Transformation
1. Image Filtering
1. Morphological Processing
1. Colour Image Processing
1. 3D Image Processing

### Image Enhancement and Restoration

- **Image enhancement** improves the appearance of an image (sharpening, denoising, contrast adjustment).
- **Image restoration** recovers an image from degradation (blur, noise) using a model of how it was degraded.

### Image Segmentation — Thresholding

The simplest segmentation method is thresholding: convert a grayscale image to binary,

```text
f(x, y) > T   →   255 (white)
otherwise     →     0 (black)
```

**Example** with `T = 100`:

Original 5×5 grayscale image:

|170|170| 55|170|170|
|---|---|---|---|---|
|170| 55|170| 55|170|
| 55|140|140|140| 55|
| 55|170|170|170| 55|

After thresholding (`T = 100`):

|255|255| 0|255|255|
|---|---|---|---|---|
|255| 0|255| 0|255|
| 0|255|255|255| 0|
| 0|255|255|255| 0|

**Thresholding methods**:

- **Histogram-shape based** — analyse peaks, valleys, and curvature of the histogram.
- **Clustering based** — e.g. the **Otsu method**, very good for bimodal distributions.
- **Adaptive thresholding** — different thresholds for different regions in the image (instead of one global value).

### Edge Detection (Image Filtering)

**What is an edge?** A pixel where brightness/intensity changes sharply. Edge detection is a fundamental tool in image processing and computer vision, useful for feature detection / extraction. Common operators: Canny, Sobel, Prewitt.

**Convolution example** — a `3 × 3` vertical-edge filter `[1, 0, -1]` (replicated in three rows) applied to a `6 × 6` image of a left half block of 100s and right half of 0s. The convolution produces a `4 × 4` response highlighting the vertical boundary:

```text
Input (6×6)            Filter (3×3)          Output (4×4)
[100 100 100  0  0  0]   [ 1  0 -1]          [   0  300  300    0]
[100 100 100  0  0  0]   [ 1  0 -1]   *      [   0  300  300    0]
[100 100 100  0  0  0]   [ 1  0 -1]   →      [   0  300  300    0]
[100 100 100  0  0  0]                       [   0  300  300    0]
[100 100 100  0  0  0]
[100 100 100  0  0  0]
```

For a horizontal edge (top half 100, bottom half 0) with a horizontal-edge filter `[1; 0; -1]` (transposed of above), the response peaks across the horizontal seam.

### Edge detection filters

**Prewitt filters** (3×3):

```text
Horizontal edges      Vertical edges
[  1   1   1 ]        [  1   0  -1 ]
[  0   0   0 ]        [  1   0  -1 ]
[ -1  -1  -1 ]        [  1   0  -1 ]
```

**Sobel filters** (3×3):

```text
Horizontal edges      Vertical edges
[  1   2   1 ]        [  1   0  -1 ]
[  0   0   0 ]        [  2   0  -2 ]
[ -1  -2  -1 ]        [  1   0  -1 ]
```

#### Sobel edge detection - Example

Convolutions in CNN

- Convolutions are very important operation in a Convolutional Neural Networks (CNN)
- Filters weights are not fixed, but learned during the training operations of a CNN for a specific task!
- Multiple filters are used in CNNs

Erosion example

Dilation example

______________________________________________________________________

## Week 3 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Feature Extraction and Neural Network Basics*

Outline

- Image Gradient
- Histogram of Oriented Gradient (HoG)
- Local Binary Pattern
- ANN Basics
- ANN Learning Process
- Logistic Regression using ANN
- Gradient Descent

### Image Gradient

**What is an image gradient?** A directional change in the intensity (or colour) of an image. The gradient captures *where* and *how strongly* the image is changing, in the **x-direction**, the **y-direction**, and combined. It is widely used for edge detection.

### Histogram of Oriented Gradient (HoG)

#### Step 1: Compute the image gradient

Apply horizontal and vertical filters to the image `I`:

- `g_x = f_x ∗ I` — horizontal filter
- `g_y = f_y ∗ I` — vertical filter

#### Step 2: Compute gradient magnitude and direction

For a 3×3 neighbourhood (only the cross is needed):

```text
[ -   100   - ]
[ 70   60  120 ]
[ -    50   - ]
```

- `g_x = |−70 + 120| = 50`
- `g_y = |−100 + 50| = 50`
- Magnitude `g ≈ √(50² + 50²) ≈ 70.7`
- Direction / angle `≈ 45°`

#### Step 3: Build an orientation histogram per cell

- Divide the image into small connected regions called **cells** (an 8×8 pixel patch).
- For each cell, build a histogram from its gradient directions and magnitudes.
- The 64 (= 8×8) gradient vectors in a cell are accumulated into a **9-bin** histogram, where bins are the gradient directions θ (e.g. a pixel with angle 80° and magnitude 2 contributes 2 to the bin around 80°).

#### Step 4: Block normalisation

- Group cells into **16×16-pixel blocks** (i.e. 2×2 cells = 4 histograms per block).
- Normalisation makes the descriptor **scale / multiplication invariant**.
  Worked example: vector `(3, 9)` has L2-norm `√(3² + 9²) ≈ 9.48`, so normalised it becomes `(3/9.48, 9/9.48) ≈ (0.32, 0.95)`. Scaling brightness ×2 gives `(6, 18) → (0.32, 0.95)` — same direction.
- Each block produces a `36 × 1` vector (4 histograms × 9 bins).

#### Step 5: Calculate the final HoG feature vector

Concatenate the `36 × 1` block vectors into one long descriptor.

Example: image `64 × 128` → `8 × 16` cells → `7 × 15` blocks (with 50% overlap) → HoG feature vector of size `7 × 15 × 36 = 3,780`.

### Local Binary Pattern (LBP)

- An efficient **texture operator** that labels each pixel by thresholding its neighbours.
- Powerful feature for texture classification.
- Describes textures using two measures: local spatial patterns and grayscale contrast.

The basic `LBP_{P,R}` operator considers `P` sampling points on a circle of radius `R` around the centre pixel `(x_c, y_c)`:

- `s(x)` — thresholding function (`1` if neighbour ≥ centre, else `0`).
- `g_c` — grey value of the centre pixel.
- `g_p` — grey value of a neighbour at angle `2π p / P` from the centre.

**LBP computation example.** Take a 3×3 neighbourhood with centre value `62`:

| 8 | 1 | 2 |
|---|---|---|
| 7 | 62| 3 |
| 6 | 5 | 4 |

Threshold each neighbour against `62`, read clockwise from pixel 1, and form an 8-bit binary number. Each pixel gives one of `2⁸ = 256` possible values, so the LBP histogram has 256 bins, which becomes the feature vector.

Worked value: `00111110₂ = 0·2⁷ + 0·2⁶ + 1·2⁵ + 1·2⁴ + 1·2³ + 1·2² + 1·2¹ + 0·2⁰ = 62`.

### Neural Network Basics

**What is an Artificial Neural Network (ANN)?**

- Multi-layered, fully-connected networks of artificial neurons.
- Has an **input layer**, one or more **hidden layers**, and an **output layer**.
- CNNs (covered later) are a specialised form of ANN tailored for images.

**Motivating example — house price prediction.** Predict the price `Y` of a house from its size `X` (sq. ft). A simple linear fit gives `y = 1.8537·x − 15.783` (price in AUD$100K). In a richer model the input features become `[size, #bedrooms, #bathrooms, garden, location, …]` and the model learns a single "neuron" that outputs the predicted price.

### ANN learning process — analogy

Think of training as walking towards a target:

- Source position `S = (x, y)`, target `T = (x_T, y_T)`, total distance `D`.
- After one step you've covered `d`; remaining distance is `D − d` — this is the **loss** to minimise.
- Update the position (parameters): `x ← x + dx`, `y ← y + dy`.

Training an ANN follows the same loop:

1. Compute the model's output for the current input.
1. Compute the loss vs. the target.
1. Compute gradients (`dw`, `db`).
1. Update the parameters and repeat.

### Logistic Regression as a one-neuron ANN

Binary classification (e.g. *Shark?* → 1 vs *Not Shark?* → 0). Flatten an input image of dimensions `64 × 128` into a vector `x = [x_1, x_2, …, x_n]` of `n = 8192` pixels (Python: `image.reshape(-1, 1)`):

```text
x_1  x_2  x_3  ...  x_{n-1}  x_n
128   56   89  ...      250  255
```

Linear function of the input followed by a sigmoid:

- `z = wᵀx + b = w_1·x_1 + w_2·x_2 + … + w_n·x_n + b` (weighted sum of inputs)
- `a = σ(z) = 1 / (1 + e^(−z))` — the sigmoid activation maps to (0, 1), so `a` is a probability.
- Decision rule: `a > 0.5` ⇒ class 1 (e.g. *Shark*); otherwise class 0.

> Rule of thumb: for binary classification the sigmoid is the obvious choice for the output-layer activation.

**Parameters**: `w` (weights), `b` (bias). **Output**: `a = σ(wᵀx + b)`.

**Loss function** (logistic / binary cross-entropy):

`L(a, y) = −[y · log(a) + (1 − y) · log(1 − a)]`

- If `y = 1`: `L = −log(a)` — large penalty when `a` is small.
- If `y = 0`: `L = −log(1 − a)` — large penalty when `a` is close to 1.

For `m` training samples the **cost** averages the per-sample loss:

`J(w, b) = (1/m) · Σᵢ₌₁ᵐ L(aᵢ, yᵢ)`

### Gradient descent — first look

Gradient Descent is an iterative approach to error correction. Find `w` and `b` that minimise `J(w, b)`:

- Compute partial derivatives `dw = ∂J/∂w` and `db = ∂J/∂b`.
- Update iteratively with learning rate `α`:

```text
w := w − α · dw
b := b − α · db
```

- `α` is the **learning rate** — a hyper-parameter that controls step size. Too small ⇒ slow; too large ⇒ overshoot / divergence.

______________________________________________________________________

## Week 4 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Neural Network in details*

Outline

- Logistic Regression Recap
- Back Propagation
- Gradient Descent and intuitions
- Optimization techniques: SGD, RMSProp, Adam etc.
- Activations Functions: Sigmoid, tanh, ReLu, Softmax
- Logistic Regression with Back Propagation
- Multi-Layered Neural Network

### Logistic Regression — Recap

Binary classification (e.g. *Dog?* → 1, *Not-Dog?* → 0) trained by gradient descent on the logistic loss:

- Inputs `X`, weights `W`, bias `b`, sigmoid activation `σ`.
- Forward pass: `z = wᵀx + b` → `a = σ(z) = 1 / (1 + e^(−z))` → prediction `ŷ`.
- Loss: `L(a, y) = −[y · log(a) + (1 − y) · log(1 − a)]`
  - If `y = 1`: `L = −log(a)`
  - If `y = 0`: `L = −log(1 − a)`

### Back Propagation

After the forward pass computes the loss, back-propagation walks **backwards** through the computational graph to compute gradients with respect to every parameter, so the optimiser can repeatedly nudge the weights to minimise the difference between actual and desired output.

### Optimization techniques

**Generic gradient-descent algorithm**:

1. Initialise `w` and `b`.
1. Forward pass — compute the model's output for each input.
1. Compute loss / cost `L(a, y)`.
1. Compute the partial derivatives `dw = ∂J/∂w` and `db = ∂J/∂b`.
1. Update parameters: `w := w − α·dw`, `b := b − α·db`.
1. Repeat steps 2–5 for `n` iterations.

The learning rate `α` is a key hyper-parameter — too small means slow convergence, too large means oscillation or divergence.

#### Three flavours of Gradient Descent

| Variant | Per-update sample size | Speed | Memory | Notes |
|----------------------------------|------------------------|---------------|------------|----------------------------------------------------------------|
| Batch Gradient Descent (BGD) | Whole training set | Slow on big data | High | Stable but hard to tune `α` and pick number of epochs. |
| Stochastic Gradient Descent (SGD)| One random sample | Much faster | Very low | Noisy path helps escape local minima; may not hit exact minimum. |
| Mini-Batch Gradient Descent (MBGD)| A mini-batch (e.g. 32–512) | Fast on GPUs | Tunable | Best of both worlds — most practical choice today. |

**SGD detail.** Because each update uses a single random sample, the trajectory is irregular and may oscillate around the optimum without converging exactly. A common remedy is a **learning-rate schedule** — start large and reduce `α` over time (simulated annealing).

> *Epoch*: one full pass through the training set. *Iteration*: one parameter update — there are `m` iterations per epoch when using `m` mini-batches.

**MBGD trade-offs**:

- Computes gradient on small sets of inputs.
- Much faster than BGD; benefits from GPU matrix operations.
- May not reach the exact minimum, but usually closer than SGD.
- Slightly harder than SGD to escape local minima because of less noise.

#### Exponentially weighted average (EWMA) — the building block for Momentum

EWMA is a classic algorithm for smoothing sequential data (a.k.a. *moving average*). It weights recent observations more heavily than older ones.

Example with daily temperature `θ_t` and smoothing factor `β = 0.9`:

```text
V_0 = 0
V_1 = 0.9·V_0 + 0.1·θ_1
V_2 = 0.9·V_1 + 0.1·θ_2
V_3 = 0.9·V_2 + 0.1·θ_3
...
V_t = 0.9·V_{t-1} + 0.1·θ_t
```

In general, with smoothing factor `β`:

`V_t = β · V_{t-1} + (1 − β) · θ_t`

`V_t` is approximately the average over the last `1 / (1 − β)` days:

- `β = 0.5` → ~2-day average
- `β = 0.9` → ~10-day average
- `β = 0.98` → ~50-day average

For, V100= 0.9 V99 + 0.1 θ100 V99= 0.9 V98 + 0.1 θ99

Substituting, V99 V100= 0.1 θ100+ 0.9 (0.9 V98 + 0.1 θ99) V100= 0.1 θ100+ 0.9 ( 0.1 θ99+ 0.9 (0.9 V97+ 0.1 V98)) ..

- “Compute the Exponentially weighted average of the gradients and use that gradient to update weights” - Andrew NG

#### Momentum

- Compute the **exponentially weighted average of the gradients** and use that to update the weights (Andrew Ng).
- One of the most popular optimisers; accelerates the gradient vectors in the right direction and reduces oscillation.
- Generally faster than plain SGD.

Algorithm (Momentum) — at iteration t, on the current mini-batch compute `dw, db`:

```text
V_dw = β·V_dw + (1 − β)·dw          # V_t = β·V_{t−1} + (1 − β)·θ_t
V_db = β·V_db + (1 − β)·db

Update parameters:
  w = w − α · V_dw
  b = b − α · V_db

Hyper-parameters: learning rate α, momentum β (≈ 0.9)
```

With momentum, the optimiser converges faster and oscillates less than plain SGD.

#### RMSProp — Root Mean Square Propagation

- Unpublished adaptive method by Geoffrey Hinton.
- Also reduces oscillation, but in a different way than Momentum.
- Divides the learning rate by an exponentially decaying average of **squared** gradients.

Algorithm (RMSProp) — at iteration t, on the current mini-batch compute `dw, db`:

```text
S_dw = β_2·S_dw + (1 − β_2)·dw^2       # square the derivatives
S_db = β_2·S_db + (1 − β_2)·db^2

Update parameters (ε ≈ 10⁻⁸ for numerical stability):
  w = w − α · dw / (√S_dw + ε)
  b = b − α · db / (√S_db + ε)
```

Intuition: when an oscillating dimension produces a large `S`, the effective step size shrinks; a small `S` keeps the step size large — faster convergence with reduced oscillation.

#### Adam — Adaptive Moment Estimation

- A combination of Momentum and RMSProp.
- Works well across a wide range of deep-learning architectures and is the most common default optimiser.

Algorithm (Adam) — initialize `V_dw = V_db = S_dw = S_db = 0`. At iteration t, on the current mini-batch compute `dw, db`:

```text
# From Momentum (β_1):
V_dw = β_1·V_dw + (1 − β_1)·dw
V_db = β_1·V_db + (1 − β_1)·db

# From RMSProp (β_2):
S_dw = β_2·S_dw + (1 − β_2)·dw^2
S_db = β_2·S_db + (1 − β_2)·db^2

# Bias correction (V, S are initialised to 0 and biased toward 0):
V'_dw = V_dw / (1 − β_1^t),   V'_db = V_db / (1 − β_1^t)
S'_dw = S_dw / (1 − β_2^t),   S'_db = S_db / (1 − β_2^t)

# Update:
w = w − α · V'_dw / (√S'_dw + ε)
b = b − α · V'_db / (√S'_db + ε)
```

Hyper-parameter guide:

- `α` (learning rate): tune; start with 0.001
- `β_1` (momentum term, on `dw`): 0.9
- `β_2` (squared-grad average, on `dw²`): 0.999
- `ε`: 10⁻⁸

### Learning Rate Decay

Speed up training by gradually **decreasing** `α` (the learning rate) as training progresses — large steps early to make rapid progress, small steps later to fine-tune.

### Activation Functions

#### Sigmoid

`σ(x) = 1 / (1 + e^(-x))`

- Non-linear, range `(0, 1)` — useful as a final-layer activation for binary classification.
- Suffers from the **vanishing gradient** problem: toward the ends of the curve `Y` changes very little with `X`, so the gradient is tiny and the network learns slowly.

#### tanh

`tanh(x) = 2 / (1 + e^(−2x)) − 1`

- Non-linear, range `(−1, 1)`.
- Stronger gradient than sigmoid, but still suffers from vanishing gradients at saturation.

#### ReLU (Rectified Linear Unit)

`A(x) = max(0, x)` — `0` for negative inputs, identity for non-negative inputs.

- Non-linear, range `[0, ∞)`.
- Cheaper to compute than sigmoid / tanh; strong gradient.
- Best used in hidden layers; largely fixes the vanishing-gradient problem.
- Can suffer from the **dying ReLU** problem — neurons stuck at 0 stop learning.

#### Leaky ReLU

`A(x) = max(0.01x, x)` — small negative slope instead of a flat zero.

- Range `(−∞, ∞)`.
- A simple fix for the dying-ReLU problem.

#### Softmax

`Softmax(y)_i = e^(y_i) / Σ_j e^(y_j)`, for `j = 1, …, K`.

- Turns a vector of raw scores into a probability distribution that sums to 1.
- Standard final-layer activation for multi-class classification.

Illustration: `Y = [2.0, 1.0, 0.1] → Softmax(Y) ≈ [0.7, 0.2, 0.1]`.

### Logistic Regression with Backpropagation

For a single sample the forward pass and loss are:

```text
z = wᵀx + b     (linear)
a = σ(z)        (sigmoid)
L = −[y·log(a) + (1−y)·log(1−a)]
```

For `m` training samples the **average cost** is `J = (1/m) Σᵢ₌₁ᵐ L(aᵢ, yᵢ)`.

Batch gradient descent on this cost:

1. Initialise `w` and `b`.
1. Forward pass over all samples.
1. Compute average cost `J`.
1. Compute `dw = ∂J/∂w` and `db = ∂J/∂b` via back-propagation.
1. Update: `w := w − α·dw`, `b := b − α·db`.
1. Repeat for `n` iterations.

### Multi-layered Neural Network

Adding a **hidden layer** between input and output gives a multi-layer perceptron (MLP). For example, a 3-layered NN has 2 hidden layers between input and output.

**Intuition**: in a multi-layer NN, the first hidden layer learns very simple patterns (e.g. edges); each subsequent hidden layer composes those into progressively more complex patterns (textures, parts, objects).

**Example**: a 2-layered MLP for multi-class classification on the Fashion-MNIST or MNIST digit datasets.

______________________________________________________________________

## Week 5 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Convolutional Neural Network (CNN) - 1*

Outline

- Computer Vision tasks Recap
- CNN Layers
- Convolution layer (Padding and Stride)
- Pooling layer
- Fully Connected Layer
- CNN Layer Visualization and intuitions.

Cat

64 X 64 X 3

Dog

800 X 800 X 3

### Deep-NNs Vs CNNs

**ANN with 3-layers · CNN with 3-layers**

- Fully connected
- Not great for images as input
- May lead to overfitting
- Too much of full connectivity
- Well suited for images with 3 dimensions
- CNNs have neurons arranged in 3D
- It is a sequence of layers which transforms input 3D volume to 3D outputs volume

**CNNs are the foundation of modern state-of-the-art deep-learning-based computer vision.** Layers in a CNN:
Three main type of layers used to build a CNN architecture

1. Convolutional Layer (CONV)
1. Pooling Layer (POOL)
1. Fully Connected layer (FC) These three types of layers are stacked together to form a CNN architecture!

**Sample CNN architecture (LeNet-5)** — alternating CONV / POOL layers, ending with FC layers.

- CONVolution is the first layer to extract features from an input image
- Core building block of a CNN
- Convolutions are basic operation in this layer
- A number of filters (e.g. edge detectors) are applied to the input image

Convolution Operation

`[ 100, 100, 100, 0, 0, 0 ]`

`[ 0, 300, 300, 0 ]`

`[ 1, 0, -1 ]`

*(convolution)*

3 X 3 filter/Kernel

4 X 4 dimension matrix

6 X 6 dimension image

**Convolution Operation · (100 X 1 + 100 X 1 + 100 X 1) + (100 X 0 + 100 X 0 + 100 X 0) + (100 X -1 + 100 X -1 + 100 X -1)**
|1100|0100|-1100|0|0|0|
|---|---|---|---|---|---|
|100|100|100|0|0|0|

`[ 0, 300, 300, 0 ]`

`[ 1, 0, -1 ]`

*(convolution)*

3 X 3 filter/Kernel

4 X 4 dimension matrix

6 X 6 dimension image

**Vertical Edge detector — Convolution Operation**

5 × 5 image · 3 × 3 filter → convolved feature

|0|0|0|0|0|0|0|0|
|---|---|---|---|---|---|---|---|
|0|100|100|100|0|0|0|0|
|0|0|0|0|0|0|0|0|

`[ 100, 100, 100, 0, 0, 0 ]`

**Padding (p) = 1**
6 X 6 dimension image without padding

8 X 8 dimension matrix with padding

|0|0|0|0|0|0|0|0|
|---|---|---|---|---|---|---|---|
|0|100|100|100|0|0|0|0|
|0|0|0|0|0|0|0|0|

|-200|0|200|200|0|0|
|---|---|---|---|---|---|
|-300|0|300|300|0|0|
|-200|0|200|200|0|0|

Filter / kernel: `[1, 0, -1]`. Input `6 × 6` padded to `8 × 8` (padding `p = 1`).

`(n × n) ∗ (f × f)` with padding `p = 1` produces output of size `(n + 2p − f + 1) × (n + 2p − f + 1)`. Example: `6 × 6 ∗ 3 × 3 → 6 × 6`.

**Same-padding question**: what padding `p` keeps the output size equal to the input?

```
n + 2p − f + 1 = n   →   p = (f − 1) / 2
```

#### Padding (Same and Valid)

**Valid Padding: ≈ No Padding (Padding p = 0) So, Output size will be → n − f + 1 X n − f + 1 Same Padding: ≈ Output size and input size is same, this requires appropriate padding. Hence use p = (f2−1), for calculate the required padding.**
Stride

It is the number of pixels by which we slide the filter over the input

matrix Example:

1. Stride(s) = 1: Move the filter by one pixel horizontally and vertically
1. Stride(s) = 2: Move the filter by two pixels horizontally and vertically

#### Stride and Padding illustration

Convolution with stride (s)=2,

Convolution with stride (s) =2 padding (p) = 0

Convolution with stride (s) =1 padding (p) = 1

padding (p) = 1

Output size with Stride and padding

Given: Input Matrix Dimension : n x n

- Filter size: `f × f`
- Padding: `p`
- Stride: `s`
- Output size: `⌊(n + 2p − f) / s⌋ + 1` along each spatial dimension
  so the output is `(⌊(n + 2p − f)/s⌋ + 1) × (⌊(n + 2p − f)/s⌋ + 1)`
  Example:

Input Matrix Dimension : 7 x 7, Filter size: 3 x 3

Padding:0, Stride :2

**Output Size= 3 X 3**

- Pooling layer is a down sampling operation which reduces the dimensionality of a matrix.

- In other words, it reduces the number of parameters for large image, but retain the valuable information.

- 3 types:

- Max pooling

- Average pooling

- Sum pooling

- Max pooling:

**Max(7, 8, 1, 5) = 8**
|7|8|9|0|
|---|---|---|---|
|1|5|8|3|
|5|9|3|2|
|5|6|6|2|

|8|9|
|---|---|
|9|6|

Max pooling with 2X2 filter and Stride 2

- Average pooling:

**(7+8+1+5)/4 = 5.25**
|7|8|9|0|
|---|---|---|---|
|1|5|8|3|
|5|9|3|2|
|5|6|6|2|

|5.25|5|
|---|---|
|6.25|3.25|

Max pooling with 2X2 filter and Stride 2

- In FC layer, the output matrix after convolution layer is flattened and feed into a fully connected layer similar to ANN

- It is a traditional Multi-layer Perception/Neural Network

- For multi-class classification, usually Softmax activation is used.

- Softmax ensures the output

- Output of the CONV and POOL layers represent a high level features of the Input image

- FC layer uses this feature to classify the input image into the desired class.

CNN layers visualization and intuition

**Example: Face recognition using CNNs · Uses simple shapes to form higher level features like facial shapes! · Uses edges to detect simple shapes · Low level feature like edges from raw pixels**

______________________________________________________________________

## Week 6 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Convolutional Neural Network (CNN) - 2*

Outline

- Dataset preparation

- Bias and Variance

- Understanding Accuracy

- Fixing Bias and Variance issues

- Regularization

- L1 and L2 Regularization

- Dropouts

- Data Augmentation – Simple and advanced

- In case of small dataset (Range : 100 - \<100k)

  - Train set: 60%

  - Validation set: 20%

  - Test set: 20%

**Popular dataset spit choice in non-DL era! Or Small Data era!**
Or,

- Train set: 70%

- Test set: 30%

- In case of Large dataset (Range : 500K - 1M+)

**Example: Total data sample : 1M+ Train: 98% ! Validation: 10,000 samples Test: 10,000 samples · Popular dataset spit choice in DL era! Or BIG Data era! · Train, validation and test set distribution:**
Rule of Thumb: Validation and Test set should come from the same distribution

### Bias and Variance

- It is a value that allows to shift the activation function to left or right, to better fit the data

Compare: without bias `a = σ(wᵀx)` vs with bias `a = σ(wᵀx + b)`.

- Changes in ‘w’ alters the steepness of the curve, keeping the origin at (0,0) or same/unchanged
- Without bias we may get a poor fit to training data

Without bias: `a = σ(wᵀx)`

- Changes in ‘b’ shifts the curve to left or right
- With bias the curve/line will not always pass through origin
- We get a better fit to training data

With bias: `a = σ(wᵀx + b)`
Variance

- It is the change in prediction accuracy of Machine Learning model between training data and test data.
- Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before.
- With high variance, models perform very well on training data but has high error rates on test data.

#### Bias and Variance effect

- Bayesian Optimal Error (BOE):

• Best optimal error that can be achieved

- Human Level performance:

- Humans are very good at a lot of tasks

- Can get labelled data from Humans – helps to improve the ML model performance

- Gain insights from manual error analysis - Why did a human got it right?

Bayesian Optimal Error / Best Possible error

Accuracy

Human-level performance

Time

Medical diagnosis of fractures on arms Consider the performance by these groups:

|A|Untrained human|16 % error|
|---|---|---|
|B|General practitioner (GP)|5 % error|
|C|Orthopedic doctor (Specialist)|2 % error|
|D|Team of experienced doctors|0.4 % error|

X-ray: Stress fracture on arms

What is Human-level error?

- Identify High Bias:

- High training error

- Validation/test error nearly same as train error

- Identify High Variance:

- Low training error

- High validation/test error

- High Bias Low Variance: Models are consistent but inaccurate

- High Bias High Variance: Models are inconsistent and inaccurate

**• Low Bias and Low Variance: Models are consistent and accurate**

- Low Bias and High Variance: Models are somewhat accurate but inconsistent on average

- High Bias: Due to simple ML model and high training error.

- Potential things to try :

- Increase features: this will help in generalizing dataset

- Make ML model more complicated

- Decrease Regularization parameter

- High Variance: Due to a ML model which is fitting most of the training dataset - overfitting.

- Potential things to try :

- Increase dataset size

- Reduce input features

- Increasing Regularization parameter

- Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better.

- Improves the model’s performance on the unseen data as well.

- Popular techniques:

- L2 and L1 regularization

- Dropout

- L2 and L1 regularization are common types and help in reducing the overfitting issue

- Idea: Update the loss/cost function by adding a regularization term

Loss function = Loss + Regularization term (l)

- Due to `λ`, the weight matrices will decrease — and a neural network with smaller weight matrices tends to be simpler.

- In Deep Learning, Regularization penalizes the weight matrices of the nodes

**L2 regularization** (also called weight decay):

`Cost = Loss + (λ / 2m) · Σ w²` — λ is a hyper-parameter. Pushes weights toward zero, but not exactly to zero.

**L1 regularization**:

`Cost = Loss + (λ / 2m) · Σ |w|`

- Penalises the absolute value of `w`

- Weights may reduce exactly to zero

- Useful for model compression

- It produces good results and most popular regularization technique

- At every iteration it randomly selects and drops some nodes and remove all the connections to and from them

- Each iteration has a different set of nodes

Data Augmentation

- Another simple way to reduce overfitting is to increase size of training dataset!
- Increase the size of training data by creating more sample using the existing training set and applying the following simple operations:
- Flip
- Rotate
- Scale
- Crop
- Translate
- Gaussian Noise

Cutout:

Simple regularization technique of randomly masking out square regions of input during training

Key Parameters:

- Patch size: 16X16 to 64X64
- Fill Value: 0(black) or mean
- Patches: 1-3 per image

Original Image After CutOut

Cutout:

Example - Cutout applied to CIFAR-10 dataset

Mixup:

Trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favour simple linear behaviour in-between training examples

Image A (λ=0.55)

Blended Output

Image B (1-λ=0.45)

CutMix:

In CutMix augmentation strategy: patches are cut and pasted among training image; ground truth labels are also mixed proportionally to the area of the patch.

Image A

Pasted Patch

Image B (Patch Donor)

### Overview of Mixup, Cutout and CutMix

### RandAugment:

Example images augment by RandAugment

- Generative Adversarial Networks (GANs):

  - Among the hottest topic is DL
  - Able to generate images which look similar to the original ones
  - Proven to be very effective

Original image from MNIST GAN generated

Data Augmentation

- Advanced data augmentation techniques:

• Neural Style transfer:

- Using CNN to separate style
- transfer style to different image

______________________________________________________________________

## Week 7 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Convolutional Neural Network (CNN) - 3*

Outline

- Transfer Learning strategies

- Classic CNN architectures case studies

- AlexNet

- Inception/GoogleNet

- Understanding Inception and ResNet

- 1X1 Convolution and it’s use

- Motivation behind Inception

- Inception V1, V2 and V3 Modules

- Introduction to ResNet

- Motivation behind ResNet

- Residual Block

Transfer Learning

- Knowledge acquired while solving one task, can be used to solve related tasks.

- Example:

- You know how to ride a Bi-cycle → You can learn how to ride a Motorbike

- You know how to use a Tablet → You can easily learn how to use a Laptop/desktop

- Similar to the way humans apply knowledge acquired from one task to solve a new but similar/related task.

- We learned how to read in Year-1 in literacy class. Reading skills acquired in the literacy classes made it easy to understand Physics in Year-9.

Transfer Learning Benefits

1. Less training data required: Don’t have enough data to train a Deep Learning model from scratch. Model trained using a large (similar) dataset can be used.
1. Faster training : Training can converge faster, due the use to existing knowledge (weights) to start with rather than from scratch.
1. Better model generalization: Model is trained to identify features which can be applied to new contexts.

#### Option-1: (VGG-16 considered as an example) Use pre-trained (ImageNet) model for prediction, without any training.

→Useful when your dataset distribution is similar to ImageNet, with small

number of samples.

#### Option-2: (VGG-16 considered as an example) Train Full-Connected layer, Use CONV layers for feature extraction

→Useful when your dataset distribution is similar to ImageNet (or original dataset), but number of classes are different and your dataset is small.

Train/Fine-Tune

**Option-3: (VGG-16 considered as an example) · Partially Train CONV layers (usually last layer(s) which have specialised · features) + Full Connection (FC) layer (with modifications)**
→Useful when your dataset distribution is not similar to ImageNet (or original dataset), number of classes are different and your dataset is small.

#### Option-4: (VGG-16 considered as an example) Train all the CONV layers + Full Connection (FC) layer (with modifications)

→Useful when your dataset distribution is not similar to ImageNet, number of classes are different, your is dataset large and the task is complex.

### Classic CNN Architectures

- Similar architecture as LeNet by Yann LeCunn et al. but deeper with more layers

- Simple architecture:

- CONV : 5 layers

- FC: 3 layer

- Max pooling

- Dropout

- Accuracy: top-5 test error rate of 15.3%

- Winner of ILSVRC 2012!

- First CNN to be successful on a very big dataset!

Input: 224x224x3 image

**CONV1 → CONV2 → CONV3 → CONV4 → CONV5 → FC1 → FC2 → FC3**
4096 Neuron

4096 Neuron

Filters: 96 Dim: 11x11 Stride: 4 Pad: 0

Filters: 256 Dim: 5x5 Stride: 1 Pad: 2

Filters: 384 Dim: 3x3 Stride: 1 Pad: 1

Filters: 384 Dim: 3x3 Stride: 1 Pad: 1

Filters: 256 Dim: 3x3 Stride: 1 Pad: 1

1000 Neuron

Activations: Relu after each CONV and FC layer Optimizer: SGD with Momentum Regularization: Dropout in FC1 and FC2 Total Trainable parameter: ~60Million Training settings: 2 X Nvidia GTX 580 3GB GPUs for 5-6days!

- Accuracy: top-5 test error rate of 6.7%
- Close to human level performance
- Winner of ILSVRC 2014!
- 22 layer Deep CNN
- Number of trainable parameters: 4 Million (Alexnet ~ 60M), Significantly reduced
- A novel inception module was introduced.
- Optimizer: RMSProp

### Understanding Inception and ResNet

**1×1 Convolution — is this useful?**

Apply a 1×1 filter/kernel over a `6 × 6 × 1` volume → still `6 × 6 × 1`. Pointless on a 1-channel input — but useful when the input has many channels.

Example with `+ ReLU` and 32 filters of size `1 × 1 × 64`:

`(6 × 6 × 64) → (6 × 6 × 32)` — channels reduced!

`64 × 64 × 256` with a `1×1` Conv (128 filters) → `64 × 64 × 128`.

- Large filter preferred for large objects
- Small filters for small objects
- Large variation in object size
- How to choose the right filter size?

**Designing CNN requires: - Deciding filter size and number - Number and type of layers etc. · Inception suggests: - Use filters with different size together! - Use different types of layers (CONV, POOL etc.) together Result → Complicated Architecture! & better performance · 28 X 28 X 64 · 1X1 · 3X3**
64

128

28

32 28 32

28 X 28 X 192 28 X 28 X 256 Max-Pooling

Computation cost

28 X 28 X 32

28

32 28

5x5 CONV #Filter: 32

28 X 28 X 32

28 X 28 X 192

**Computation Cost: 28 X 28 X 32 X 5 X 5 X 192 ≈ 120M multiplications! Quite expensive !**
Reduce Computation cost using 1X1 CONV

28 X 28 X 32

28

32 28

1 X 1 X 192 #Filters: 16

5 X 5 X 16 CONV #Filter: 32

28 X 28 X 32

**28 X 28 X 16**
28 X 28 X 192

Computation Cost: 1X1: 28 X 28 X 16 X 192 ≈ 2.4M multiplications! 5X5: 28 X 28 X 32 X 5 X 5 X 16 ≈ 10M multiplications! Total : 12.4M multiplications! → Reduced by 10 times!

#### Bottleneck Layer

**192 · 32 · 16 · Bottleneck**

#### Inception Module V1

**GoogleNet(2014): 9 Inception modules stacked together**

Deeper networks suffer from the vanishing-gradient problem. GoogLeNet adds two **auxiliary classifiers**:

- Apply Softmax to an intermediate feature map
- Compute an auxiliary loss
- Used only during training (`Total loss = real_loss + 0.3 · Aux_loss_1 + 0.3 · Aux_loss_2`)

Inception V3 introduces three further ideas:

- **Factorising convolutions** to reduce parameters:
  - 1 layer of 5×5 filter → 25 parameters; 2 layers of 3×3 filters → 18 parameters (~28% reduction)
  - 3×3 filter → 9 parameters; 3×1 + 1×3 filters → 6 parameters (~33% reduction)

Aux_Loss2

Aux_Loss1

#### Inception V3 Architecture

- Deep Residual networks (ResNet) → Skip connections
- Enabled the development of the much deeper networks (100s of layers!)
- ResNet is composed of Residual Blocks were introduced!
- Degradation problem: Adding more layers eventually have negative effect on the final performance.

What wrong with this curves? Overfitting?

- 56 layer model is not better than the 20 layers!
- What happens when we keep add more layers to a plain CNN to make it deeper?

**In principle deeper model should perform better than shallow CNNs · Residual Block · Plain Layers**

### Summary

- 15+ million labelled high-resolution images
- 22000 categories
- ILSVRC (Large Scale Visual Recognition Challenge) used a subset of ImageNet:
- ~1000 images per category
- 1000 categories
- Train: 1.2 million images
- Validation: 50k images
- Test : 150k images

**ImageNet dataset results** — see the original lecture slide for the model-vs-mAP scatter plot (CNN architectures vs. their ImageNet top-1/top-5 accuracy).

______________________________________________________________________

## Week 8 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Object Detection -1*

Outline

- Introduction to Object Detection
- Datasets and Performance metrics
- Taxonomy
- Classification and Localization task
- Object detection as a regression problem
- Object detection as a classification problem
- Case study: RCNN family
- Image annotation

Introduction

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

**DOG · CAT**
DOG

Single Object

Multiple Object

- The PASCAL Visual Object Classification (PASCAL VOC) is a popular dataset for object detection, classification and segmentation.

- 20 categories

- Link: http://host.robots.ox.ac.uk/pascal/VOC/

- ImageNet has released an object detection dataset in 2013

- Train set: 500,000 images, 200 categories.

- Not very popular due to large number of classes and dataset size!

- Large number classes complicates the task

- Intersection over Union (IoU): Intersection over Union is a metric used for the evaluation of an object detector, i.e. how good is the predicted bounding box for an object detected closely matches

### Microsoft COCO Dataset

MS COCO is a popular large-scale dataset for object detection, segmentation and captioning. Standard evaluation uses mAP at several IoU thresholds (`AP@.50`, `AP@.75`, `AP@[.50:.05:.95]`).

### Taxonomy of Object detectors

**Object detection taxonomy** is organised by network type (single-stage vs two-stage / region-proposal based) and data type (monocular image vs point cloud).

By data type: **3D Object Detector** (point-cloud / point-nets) and **2D Object detector** (regression / classification based).

Examples by family: **monocular image** input (RCNN family, SSD, YOLO) vs **point-cloud** input (PointNets).

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

**DOG · CAT**
DOG

Single Object

Multiple Object

**Classification Task:**
Input : Image Output: Label Performance Evaluation: Accuracy

**Output : Dog · Localization Task:**

Input : Image Output: Bounding Box in the image

(x, y, Ht, Wd) or (x, y, x’, y’) Performance Evaluation: IoU

**Output : (x, y, Ht, Wd) · Output : 4 numbers (x’, y’, Ht’, Wd’)**

**Calculate Loss L2 Loss · CNN · Ground Truth: 4 numbers (x, y, Ht, Wd)**
Input Image

**Input Image · We need to modify this CNN pipeline to output Class Label and Bounding Box (4 numbers) · Pre-trained model or ImageNet, AlexNet, VGG16, ResNet, etc. Classification Head**
Classification head → Softmax loss, regression head → L2 loss (combined as multi-task loss).

Input Image

Potential locations for Regression head in CNN

**After CONV Layer, Before the FC layer After Last FC layer**
Input Image

Task: Object Detection Problem

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

**DOG · CAT**
DOG

Single Object

Multiple Object

Detection as a regression problem

**Output : Dog, (x, y, Ht, Wd) · Cat, (x, y, Ht, Wd) Cat, (x, y, Ht, Wd)**

1. Apply Sliding Window technique
1. Apply CNN to different Windows and get a prediction

**Output : Dog? No Cat? No · CNN · Background? Yes**

1. Apply Sliding Window technique
1. Apply CNN to different Windows and get a prediction

**Output : Dog? No Cat? Yes Background? No · CNN**

1. Apply Sliding Window technique
1. Apply CNN to different Windows and get a prediction

**Output : Dog? No Cat? Yes Background? No · CNN · Issue with Sliding Window technique**

1. Apply CNN on large number of windows
1. Multiple scale and locations of windows
1. Inaccurate bounding boxes
1. Computationally expensive

**Region Proposal Technique:**

- Find blobs in the image that are most likely to contain objects
- E.g: Selective search → ~1000-2000 region proposals using CPU!

Case Study: R-CNN

Linear Regression for bounding box offsets

Classify each region with SVMs

1. Resized to match the input to CNN requirement.
1. mAP: 62.4% for 2007 PASCAL VOC
1. Problem: Very Slow!

Pass each region through ConvNet

Warped image regions

Region-of-interest (ROI) from proposal method around ~2K

Object category

Box offset

Per-Region Network

Crop + Resize features

Region of Interest (ROIs) from proposal method

Run whole image through ConvNet

1. Reduce computation
1. ROIs from feature maps using selective search
1. mAP: 70% for 2007 PASCAL VOC

Case Study: FASTER- R-CNN

1. Use CNNs to make proposals

1. Introduced RPN (Region Proposal Network)

1. mAP: 78.8% for 2007 PASCAL VOC

- RCNN → Look at every patch one by one

- Fast R-CNN → Look once, and then inspect patches on feature map

- Faster R-CNN → Propose patches using a neural network (RPN)

| Aspect | R-CNN | Fast R-CNN | Faster R-CNN |
|------------------|------------------------|-------------------------|------------------------|
| Region proposal | Selective search | Selective search | RPN (learned) |
| CNN usage | Per region | Once per image | Once per image |
| Speed | Very slow | Faster | Can work in real-time |
| Training | Multi-stage, discrete | Partially end-to-end | Fully end-to-end |
| Accuracy | Good | Better | Best of all three |

### Object Detection Techniques History

See the surveyed timeline (sliding-window → region proposals → R-CNN family → YOLO/SSD → DETR/RF-DETR). Refer to the original slide-deck image for the visual.

### Image annotation for object detection

______________________________________________________________________

## Week 9 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Object Detection -2*

Outline

- Object detection techniques recap
- Strategies for predicting bounding boxes
- Non-Maxima suppression (NMS)
- Anchor boxes
- Case study:
- Yolo (You Look Only Once)
- SSD (Single Shot Detector)
- Object detection state-of-the-art

### Taxonomy of Object detectors

**Object detection taxonomy** is organised by network type (single-stage vs two-stage / region-proposal based) and data type (monocular image vs point cloud).

By data type: **3D Object Detector** (point-cloud / point-nets) and **2D Object detector** (regression / classification based).

Examples by family: **monocular image** input (RCNN family, SSD, YOLO) vs **point-cloud** input (PointNets).

### Object Detection Techniques History

#### Sliding Window technique

- Crop images and classify using CNN
- Try different sizes of the sliding window Issues:
- Slow
- Computationally very expensive
- Less accurate

#### Region Proposals

Currently:

Task:

- Sliding Window

- Selective Search

- Region Proposals

- Predict Bounding boxes from CNN

- Place a grid over the image

- Apply image classification and localization to each of the grid cells

- Input:

  - Image: (ht x wd x 3) Target:
  - Bounding box information for each object
  - Class for each object

**Class : {car, bike} · Target:**
Y = {po, x, y, h, w, c1, c2} for each cell e.g:

- Cell(1,1) = {0, ?, ?, ?, ?, ?, ?} :

- Cell(2,1) = {1, x, y, h, t, 1, 0}

- Cell(2,2) = {0, ?, ?, ?, ?, ?, ?}

- Cell(2,3) = {1, x, y, h, t, 1, 0} :

- Cell(3,3) = {0, ?, ?, ?, ?, ?, ?}

**Class : {car, bike} · Idea: Take the mid-point of the object and Assign it to a grid cell based on its location**

**Target output vector: 3 X 3 X 7 3 X 3: Grid size 7: (5 + Number-of-Classes) · Class : {car, bike}**
3 X 3 X 7

#### Training Strategy:

Target: Y

Input: X

3 X 3 X 7

Class : {car, bike}

In practice: The grid is finer, 19 X 19 instead of 3 X 3 So, Target will be of size: 19 X 19 X 7 Works well for non-overlapping objects

1. Each object has one midpoint
1. Each cells are subjected to object localization + classification
1. Hence, neighbouring cells might assume that it has the mid-point
1. Hence, Multiple detection bounding box

NMS cleans/removes the multiple detection and only keeps the one with very high confidence

1. Check the probabilities of each detection and keep ones with score > Threshold (0.7)

1. For remaining boxes:

- Box with highest score is the detection results.

- Discard any remaining boxes with IoU > 0.5 with final detected box, i.e: overlap with the box with highest score.

YOLO: You Only Look Once Algorithm

**Challenges with overlapping objects**

- Each grid cell detect only one object
- For multiple overlapping objects, Mid point are on the same grid cell

**So, Currently the Target Y = {1, x, y, h, w, C1, C2}, As the mid-points for both the objects are on the same grid cell, only one of the objects will be associated**
Anchor Box 1 Anchor Box 2

Predicted BB

1. A cell which contains its mid-point and
1. Anchor box for the cell with highest IoU

Anchor Box 1

Calculate the IoU of Anchor boxes and predicted BB

Anchor Box 1 Anchor Box 2

Similar Shape

**So, with Anchor boxes: Target Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2},**
Anchor Box 1 Anchor Box 2

- Anchor Box 1
- Anchor Box 2

**Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2} Cell(1,1) = {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?} : Cell(12,6)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(12,15)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(19,19)= {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?}**
InputX

Class : {1:car, 2:bike} Y size : (

X 2 X 7 )

**19 X 19 · Grid Size · #Anchor Box · 5(Po, x,y,h,w) + #Classes(2)**

#### Training:

Target: Y

Input: X

19 X 19 X 2 X 7

Class : {car, bike}

**Input: X · Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2} {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?} : {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?}**

19 X 19 X 2 X 7

**Class : {car, bike} · Input: X · Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2} {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?} : {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?}**

19 X 19 X 2 X 7

**Class : {car, bike}**

- Real-time performance with 45 frames per sec, 0.02 sec per image

- Not suitable for small objects

- Issues with new or multiple aspect ratios and unable to generalize

- Similar to YOLO

- VGG16 base CONV layers

- Take advantage of Anchor boxes with different aspect ratios

- Large number of anchors boxes are chosen

- Not suitable for small objects

- 3 times faster than Faster-RCNN

- With ResNet101 base SSD may be help in detecting small objects with better features from the CONV base

|Method|Train Dataset|mAP|Time in sec/image|Time Frame /sec|
|---|---|---|---|---|
|RCNN (VGG16)|Pascal VOC 2007|66.0|50|-|
|Fast RCNN|VOC 2007+2012|70.0|2|-|
|Faster RCNN (VGG16)|VOC 2007+2012|73.2|0.11|9|
|Faster RCNN (ResNet101)|VOC 2007+2012|83.8|2.24|0.4|
|Yolo|VOC 2007+2012|63.4|0.02|45|
|SSD300|VOC 2007+2012|74.3|0.02|45|
|SSD512|VOC 2007+2012|76.8|0.05|19|

**Common base networks**: VGG16, ResNet101, Inception V2/V3, MobileNet, AlexNet, ZFNet.

**Detection frameworks**: R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN), YOLO family, SSD, R-FCN.

**Summary**:

- Faster R-CNN is more accurate but slower
- YOLO / SSD are faster (real-time) but may be less accurate

______________________________________________________________________

## Week 10 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Anchor Free Object Detection, Instance/Semantic Segmentation*

Outline

- Recap: Anchor-based object detection
- Anchor-free object detection overview
- Case-Study: YoloX
- Introduction to Instance Segmentation
- Techniques and application of Instance Segmentation
- Case Study: Mask R-CNN
- Semantic Segmentation Introduction
- Case Study: U-Net

Recap: Predicting Bounding Boxes

### Training Strategy:

- Place a grid over the image
- Apply image classification and localization to each of the grid cells Input:
- Image: (ht x wd x 3) Target:
- Bounding box information for each object
- Class for each object

Class : {car, bike}

Recap: YOLO: You Only Look Once Algorithm

**Challenges with overlapping objects**

- Each grid cell detect only one object
- For multiple overlapping objects, Mid point are on the same grid cell

**So, Currently the Target Y = {1, x, y, h, w, C1, C2}, As the mid-points for both the objects are on the same grid cell, only one of the objects will be associated**
Anchor Box 1 Anchor Box 2

Predicted BB

1. A cell which contains its mid-point and
1. Anchor box for the cell with highest IoU

Anchor Box 1

Calculate the IoU of Anchor boxes and predicted BB

Anchor Box 1 Anchor Box 2

Similar Shape

**So, with Anchor boxes: Target Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2},**
Anchor Box 1 Anchor Box 2

Recap: YOLO: You Only Look Once Algorithm

- Anchor Box 1
- Anchor Box 2

**Y = {Po, x, y, h, w, C1, C2, Po, x, y, h, w, C1, C2} Cell(1,1) = {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?} : Cell(12,6)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(12,15)= {0, ?, ?, ?, ?, ?, ?, 1, x, y, h, w, 1, 0} : Cell(19,19)= {0, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?}**
InputX

Class : {1:car, 2:bike} Y size : (

X 2 X 7 )

**19 X 19 · Grid Size · #Anchor Box · 5(Po, x,y,h,w) + #Classes(2)**
Drawbacks of Anchor-based detectors

• Sensitive to:

**4 · 2 · 1**
2

- Size
- Aspect ratio
- No. of Anchor boxes (fixed)
- To much variation with shape
- Small object
- May not generalize due to pre-defined anchor boxes
- Computation expensive

Anchor-free detector

Localize objects without using boxes as proposals Two board categories:

1. Key-point based

1. Center-based

- Locates key object parts in an image

- Detects spatial locations or points unique to an object

- With human body as an example

- Key Part of Face: nose, eyebrows etc.

- Key point of human body: joints, elbows, etc.

- Object is represented using Key-points

Anchor-free detector : Center-based

- Finds positives in the centre
- Predicts four distances from the positive to the potential object boundary

YOLO Timeline

**Yolo X · Yolo - NAS**
Yolo v12

Yolo R

**Yolo v8 · Yolo v3**
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

**Anchor Free**

- YoloX: Exceeding the YOLO Series, 2021

- Anchor-free detector in the Yolo Family

- Decoupled head used

- Label assignment using SimOTA

- Uses YoloV3 SPP with DarkNet53 backbone

- Uses advanced augmentation such as Mix-up & Mosaic

- Backbone Neck Head

- Every Yolo Architecture consists of three parts: Backbone, neck, head

- Backbone → Feature extraction

- Neck → Aggregation of multi-scale feature

- Head→ Localization and Classification scores

### Case Study: YoloX, Decoupled head

#### Mixup Augmentation

#### Mosaic Augmentation

#### Case Study: YoloX, Performance

#### Case Study: Yolo State-of-the-art, Performance

### Yolo26 — The Next Evolution

- Real-time computer vision model by Ultralytics

- Supports: Detection, Segmentation, Classification, Pose, Tracking, OBB

- Available in Nano, Small, Medium, Large, XLarge

- End-to-end detection pipeline (NMS-free)

- Designed for edge AI and fast deployment

- Yolo26 – Why is it Faster?

- NMS-free inference removes post-processing overhead

- Direct bounding box regression (no DFL)

- Lower latency and simpler deployment graph

- CPU-optimized architecture

- Up to 43% faster on CPUs than YOLO11 (Ultralytics benchmark)

Yolo26 – Key Changes

- ProgLoss (Progressive Loss Balancing) improves training stability and convergence
- STAL (Small-Target-Aware Label Assignment) improves small-object detection
- MuSGD optimizer improves convergence speed
- Better speed–accuracy trade-off than many previous YOLO models
- Ideal for robotics, drones, surveillance, and edge devices

#### Yolo26 – High Level Architecture (Inference)

#### Yolo26 – Training Pipeline

#### Yolo26 – Performance

### Instance Segmentation

Image Classification

Image Classification Object Detection

- Localization Instance Segmentation

**DOG · CAT**
DOG

Single Object Multiple Object

Semantic Segmentation Vs Instance Segmentation:

- Semantic segmentation classifies object pixels to specific classes/category
- Instance Segmentation identifies each pixels object instance

Input Image Semantic Segmentation Instance Segmentation

**Popular Techniques:**
|Semantic Segmentation|Instance Segmentation|
|---|---|
|Conditional Random Field (CRF) Fully Convolutional Network (FCN) U-Net Pyramid Scene Parsing Network (PSPNet) etc.|SegNet, DeepMask, SharpMask, MaskRCNN, etc.|

**Applications: Autonomous Driving**

**Applications: Scene Understanding**

**Applications: Aerial Image processing**

- Mask-RCNN → Mask-Region Convolutional Neural Network
- An addition to the RCNN family, performing instance segmentation
- Improved over FasterRCNN
- A Full Convolutional Network (FCN) for predicting Mask for each class/object.
- 2 Stages:
- Stage 1: RPN proposes candidate object bounding boxes.
- Stage 2: Classify the Candidates, refine bounding boxes, and predict mask.

**Faster-RCNN · Source and Reference: http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf**

- Sample Results

- Sample Results on video:

#### Mask R-CNN Limitations

- Computational Complexity: Training and inference can be computationally intensive, requiring substantial resources, especially for high-resolution images or large datasets.
- Small-Object Segmentation: may struggle to accurately segment very small objects due to limited pixel information.
- Data Requirements: Training requires a large amount of annotated data, which can be time-consuming and expensive to acquire.
- Limited Generalization to Unseen Categories: The model's ability to generalize to unseen object categories is limited.

### Semantic Segmentation

Introduction to Semantic Segmentation

Semantic segmentation classifies object pixels on specific classes/category

**Input Image Semantic Segmentation Instance Segmentation**

#### Semantic Segmentation: UNet

00000000000001100000000000000000000000 00000000000001110000000011000000000000 00000000000011111111111111000000000000 00000000000011111111111111000000000000 00000000000011111111111110000000000000 00000000000011111111111111100000000000 00000000000011111111111111110000000000 00000000000011111111111111110000000000 00000000000001111111111111110000000000 00000000000000111111111111100000000000 00000000000000011111111111000000000000 00000000000000011111111111000000000000 00000000000000011111111100000000000000

...

{

{

{

#### Semantic Segmentation: UNet Architecture

cs224d course

______________________________________________________________________

## Week 11 — Lecture

*42028: Deep Learning and Convolutional Neural Network — Introduction to Sequence Modelling*

Outline

- Introduction to Sequence Modelling
- Introduction to RNNs
- Introduction to Attention mechanism
- Introduction to Transformers
- Case Studies:
- Image Classification using transformer (ViT)
- Object Detection using Transformer (RF-DETR)
- Diffusion models

Examples to motivate sequence modelling:

- Predict where the ball will go next?
- Complete the sentence: "This Sunday I went for a walk…"

### Sequence modelling types and applications

Y’

**Sequence-modelling task types**:

- **Many to many** — Q&A with LLMs, language translation
- **One to many** — image captioning
- **One to one** — binary classification (e.g. "will it rain today?")
- **Many to one** — sentiment analysis (e.g. "42028 is the best subject so far!")
  “Will it rain today?” Yes/No?

“42028 is the best subject so far!”

Me: “Hey Siri what's the weather today ?” Siri: “Its Evening now! Don’t ask boring Qs”

“A womenisthrowingfrisbee”

Size #Bedroom #Bathroom Garden Location

Price

Recurrent Neural Network (RNN) Basics

Output

y’t

y’0

y’1

y’2

h0 h1

Recurrent cell

X0

X1

X2

Input

Output `y_t = f(x_t, h_t)` — from input `x_t` and past hidden state `h_t`.

Output Function+

y’t

Weights w

Hidden state update `h_t = f(x_t, h_{t-1})` — recurrent cell consumes current input and the previous state.

Xt

Output vector `ŷ_t = W_y · h_t`

y’t

Output

`h_t = tanh(W_hh · h_{t-1} + W_xh · x_t)`

RNN

Update Hidden State

Input Vector

Total Loss (L)

Forward Pass Backward Pass

L0 L1 Lt

...

y’0

y’1

y’t

y’t

RNN ht

W W

≈

RNN

RNN

RNN

...

X0

X1

Sequence Modelling: Design Criteria

- Support for Variable-Length Input

- Has Temporal Dependency (Long-term, Short-term)

- Preserve the information order

- Share parameters across sequence

- RNN Limitations

- Prone to vanishing and exploding gradient problem

- Long term memory not supported

- Slow and no parallelization

Output

y’0

y’1

y’2

y’t

y’t-2

y'\_{t-1}, …

Inputs (features) `X_0, X_1, X_2, …`

Xt-2

Xt-1

y y y y y y

Output

...

Features

x x x x x x

Input

Idea:

- No Recurrence
- No Long-term memory
- Feed Everything into the Dense network
- Identify and focus on what’s important

Attention Is All you Need!

- Identify which parts of the image to focus on
- Extract features with high attention

Why do we need Attention?

- RNNs process sequences one step at a time
- Long sentences lead to Long-term memory loss
- Important words can be hidden in long dependencies
- Attention helps to focus on relevant parts of the input

What is Attention? Intuition

- Mimics human focus mechanism: “Where should I look?”
- For each output word, attention decides which input word is most important
- Computes a weighted sum of all input vectors
- Higher weight = more important

**Training Query (Q)**
Key (K1)

**Key (K2)**
Compute similarity between Q and K

For each attention step: compute similarity between the **Query Q** and each **Key K_i**, then extract **Values V_i** weighted by that similarity.

Extract Values based On attention

- Self-Attention is the foundation for Transformer architecture
- Entire sequence is processed in parallel
- Has Encode and a Decoder block
- Stack of Layers with Self Attention and Feed Forward Neural

### Case Studies

- Introduced in 2021: "An Image is Worth 16\*16 Words: Transformers for Image Recognition at Scale," published at ICLR 2021

- Vision transformer have extensive application in all computer vision tasks

- ViT is a type of Deep Learning Model that’s looks at Images, like how language model looks at words

- Images are represented as sequences of patches!

1. Split an image into patches
1. Flatten the patches
1. Produce lower-dimensional linear embeddings from the flattened patches
1. Add positional embeddings
1. Feed the sequence as an input to a standard transformer encoder
1. Pretrain the model with image labels (fully supervised on a huge dataset)
1. Finetune on the downstream dataset for image classification

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

- Object Detection techniques using Transformers
- An improvement over the original DETR (DEtection TRansformer)
- DETR looks at everything globally but miss small things.
- RF-DETR looks globally and understands the relationships between things.
- Real-time Transformer-based object detection architecture
- Outperforms all object detection models, 60+ mAP achieved on COCO dataset.

#### Case Study: Diffusion Models - Intuition

Goal: Generate new data samples (e.g. images, audio, text) that is similar to a training dataset by learning to reverse a gradual noise process! A generative model!

**Generic steps:**

1. Start with real data
1. Add noise step-by-step, until the image becomes pure noise
1. Train a model to reverse this process: denoising to recover the original image!
1. Once trained, the model can start from pure noise and generate new and realistic samples.

#### Diffusion Models

Example: Given a lot of sprite sample images

**Training Data**
Task: Generate New Sprite images (New Image generation from image input)

**Example prompt** (text-to-image): "Butterfly image" → Task: generate an image of a butterfly.

- Forward Diffusion:

- Add noise gradually to the original image for many steps

- Iterate until the Image becomes pure noise

- Gaussian noise used

- No-learning

- Reverse Diffusion:

- Denoising: Model is trained to predict and reverse this noise

- Use the prediction to denoise the image

- Given a noisy image, it predicts a slightly less noisy image version

- After several steps, it reconstructs a clean and new image! From pure noise

Week 12 Guest Lecture

Industry Guest Lecture will be presented by Amazon Web Services (AWS) on Week 12

Topic: Build, Evaluate and Scale Production ready Agents
