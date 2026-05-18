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

**CNNs are the foundation of modern state-of-the-art deep-learning-based computer vision.** A CNN is built from three main layer types, stacked together:

1. **Convolutional layer (CONV)** — applies learnable filters (e.g. edge detectors) to extract features from the input image.
1. **Pooling layer (POOL)** — down-samples to reduce spatial size.
1. **Fully-connected layer (FC)** — classifies the high-level features into output classes.

Classic example: **LeNet-5** — alternating CONV / POOL layers, ending with FC layers.

### Convolution operation

CONVolution is the first layer to extract features from an input image and is the core building block of a CNN. A 2D convolution slides a small filter (kernel) over the input image and produces a feature map.

**Example** — input `6 × 6` image of two halves (left = 100s, right = 0s), filter `[1, 0, -1]` repeated in three rows (a simple vertical-edge detector):

```text
Input (6×6)           Filter (3×3)         Output (4×4)
[100 100 100  0  0  0]   [ 1  0 -1]        [   0  300  300    0]
[100 100 100  0  0  0]   [ 1  0 -1]   *    [   0  300  300    0]
[100 100 100  0  0  0]   [ 1  0 -1]   →    [   0  300  300    0]
[100 100 100  0  0  0]                     [   0  300  300    0]
[100 100 100  0  0  0]
[100 100 100  0  0  0]
```

The high values (300) in the output map the vertical edge between the two halves.

### Padding (Same vs. Valid)

- **Valid padding** — no padding (`p = 0`); output is `(n − f + 1) × (n − f + 1)`.
- **Same padding** — pad so the output has the same size as the input; use `p = (f − 1) / 2`.

Without padding the output shrinks each layer; with padding the borders are preserved.

### Stride

**Stride `s`** = number of pixels the filter shifts per step.

- `s = 1`: shift by one pixel horizontally and vertically (default).
- `s = 2`: shift by two pixels — output is roughly half the size.

### Output size with padding and stride

For an `n × n` input convolved with an `f × f` filter, padding `p`, stride `s`:

```text
output side = ⌊(n + 2p − f) / s⌋ + 1
```

**Example**: `n = 7`, `f = 3`, `p = 0`, `s = 2` → `⌊(7 − 3)/2⌋ + 1 = 3` → output is `3 × 3`.

### Pooling layer

A down-sampling operation that reduces a feature map's spatial size while keeping the most important information.

Three common variants:

- **Max pooling** — take the maximum in each window.
- **Average pooling** — take the average.
- **Sum pooling** — take the sum (rarely used).

**Max pooling** with a 2×2 filter and stride 2:

Input `4 × 4`:

| 7 | 8 | 9 | 0 |
|---|---|---|---|
| 1 | 5 | 8 | 3 |
| 5 | 9 | 3 | 2 |
| 5 | 6 | 6 | 2 |

Output `2 × 2` (take the max of each 2×2 window, e.g. `max(7, 8, 1, 5) = 8`):

| 8 | 9 |
|---|---|
| 9 | 6 |

**Average pooling** on the same input (e.g. `(7+8+1+5)/4 = 5.25`):

| 5.25 | 5 |
|------|------|
| 6.25 | 3.25 |

### Fully-Connected (FC) layer

After CONV / POOL layers extract high-level features, the FC layer flattens the feature map and feeds it through a traditional multi-layer perceptron. For multi-class classification, the final layer typically uses a **Softmax** activation so the outputs become a probability distribution over classes.

### CNN layer visualisation — intuition

A trained CNN's filters learn increasingly abstract features layer by layer:

- Early layers detect **low-level features** (edges from raw pixels).
- Middle layers compose edges into **simple shapes**.
- Late layers compose shapes into **high-level concepts** (e.g. facial parts) — useful for tasks like face recognition.

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

### Dataset preparation

How you split the data depends on the dataset size:

| Dataset size | Train | Validation | Test |
|-------------------------|-------|------------|----------|
| Small (≤ ~100k) | 60% | 20% | 20% |
| Small (alt. split) | 70% | — | 30% |
| Large (500K – 1M+) | 98% | 10,000 | 10,000 |

The 60/20/20 split is typical for the pre-DL / small-data era. With very large datasets (deep-learning era), a fixed-size validation and test set (~10k each) is enough and the rest goes into training.

> Rule of thumb: validation and test sets must come from the same distribution.

### Bias and Variance

**Bias** is the value that shifts the activation function left/right to better fit the data — without bias the line always passes through the origin, which often leads to a poor fit.

- Compare: without bias `a = σ(wᵀx)` vs. with bias `a = σ(wᵀx + b)`.
- Changes in `w` alter the **steepness** of the curve (keeping the origin fixed).
- Changes in `b` shift the curve **left or right**, giving a better fit to the data.

**Variance** is the change in prediction accuracy between training and test data:

- High variance means the model overfits — it pays too much attention to training data and fails to generalise to unseen data.
- High-variance models perform very well on training data but have high error on test data.

#### Bayesian optimal vs. human-level performance

- **Bayesian Optimal Error (BOE)** — the best possible error any model could achieve on the task.
- **Human-level performance** — humans are very good at many tasks; using human-labelled data and analysing human errors helps improve ML models. As training time grows, model accuracy approaches human-level performance and then plateaus toward BOE.

Example — medical diagnosis of arm fractures from X-rays:

| Group | Annotator | Error rate |
|-------|---------------------------------|-----------:|
| A | Untrained human | 16% |
| B | General practitioner (GP) | 5% |
| C | Orthopaedic specialist | 2% |
| D | Team of experienced doctors | 0.4% |

The error of the strongest human group (D) is the practical human-level error to compare against.

#### Identifying bias / variance issues

- **High bias** — high training error; validation/test error roughly equal to training error.
- **High variance** — low training error; validation/test error much higher than training error.

The four regimes:

| Bias | Variance | Behaviour |
|------|----------|------------------------------------------------------|
| Low | Low | Consistent and accurate (the goal) |
| Low | High | Accurate on average but inconsistent — overfitting |
| High | Low | Consistent but inaccurate — underfitting |
| High | High | Inconsistent and inaccurate |

**Fixing high bias** (model too simple, high training error):

- Add more features or use a more complex model.
- Decrease the regularization parameter.

**Fixing high variance** (model overfits):

- Increase dataset size.
- Reduce input features.
- Increase the regularization parameter.

### Regularization

Regularization slightly modifies the learning algorithm so the model generalises better to unseen data. The basic idea is to update the loss to include a **regularization term**:

`Cost = Loss + λ · R(w)`

Due to `λ`, the weight matrices shrink — and a neural network with smaller weight matrices tends to be simpler.

**L2 regularization** (a.k.a. weight decay):

`Cost = Loss + (λ / 2m) · Σ w²`

- λ is a hyper-parameter.
- Pushes weights toward zero, but not exactly to zero.

**L1 regularization**:

`Cost = Loss + (λ / 2m) · Σ |w|`

- Penalises the absolute value of `w`.
- Weights may reduce exactly to zero.
- Useful for model compression (sparsity).

#### Dropout

- The most popular regularization technique in practice — produces excellent results.
- At every training iteration, randomly select and drop some nodes (and all their connections).
- Each iteration uses a different random subset of nodes.

### Data Augmentation

Increasing the effective training set size by creating new samples from existing ones is a simple and very effective regulariser.

**Simple operations**: flip, rotate, scale, crop, translate, Gaussian noise.

**Cutout** — randomly mask out square regions of the input during training.

- Key parameters: patch size (16×16 to 64×64), fill value (0/black or mean), patches per image (1–3).
- Common test setting: applied to CIFAR-10.

**Mixup** — train on convex combinations of pairs of examples and their labels. The network is encouraged to behave linearly between training examples. For two images `A` and `B` with mixing coefficient `λ`:

`Image_mixed = λ · A + (1 − λ) · B` (e.g. λ = 0.55), and likewise for labels.

**CutMix** — cut a patch from one image and paste it onto another; ground-truth labels are also mixed proportionally to the patch area.

**Overview** — Mixup, Cutout and CutMix are complementary regularisers; many image-classification training pipelines use them in combination.

**RandAugment** — automatically chooses a random sequence of standard augmentations (rotate, shear, colour-jitter, etc.) per image with a single magnitude parameter, removing the need to hand-tune each operation.

### Advanced data augmentation

- **Generative Adversarial Networks (GANs)** — one of the hottest DL topics; can generate new images that look similar to real ones. Sample use: training a GAN on MNIST to produce synthetic digits that augment the original training set.
- **Neural Style Transfer** — use a CNN to separate the *style* and *content* of two images, then re-render the content image in the style of the other.

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

### Transfer Learning

Knowledge acquired while solving one task can be re-used to solve related tasks — just like humans transferring skills:

- Knowing how to ride a bicycle helps you learn to ride a motorbike.
- Knowing how to use a tablet helps you learn a laptop/desktop.
- Reading skills learnt in Year-1 literacy make it easier to understand a Year-9 physics textbook.

**Benefits**:

- Less training data required — start from a model pre-trained on a large (similar) dataset.
- Faster training — converges sooner because existing weights are a good starting point.
- Better generalisation — pre-trained features carry over to new contexts.

#### Four strategies (using VGG-16 on ImageNet as the source model)

| Option | What to train | Use when … |
|--------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 1 | Nothing — use the pre-trained model directly for inference | Your dataset distribution is **similar** to ImageNet and you have very few samples. |
| 2 | Only the FC layer; freeze all CONV layers | Distribution is similar to ImageNet but number of classes differs and dataset is small. |
| 3 | Partially train the last CONV layer(s) + a modified FC head | Distribution is **not similar** to ImageNet, number of classes differs, dataset is still small. |
| 4 | All CONV layers + a modified FC head | Distribution is not similar to ImageNet, classes differ, dataset is **large**, and the task is complex. |

### Classic CNN Architectures

#### AlexNet (ILSVRC 2012 winner)

- Similar architecture to LeNet by Yann LeCun et al., but deeper.
- First CNN to be successful on a very large dataset.
- Top-5 test error: **15.3%**.

```text
Input 224×224×3 → CONV1 → CONV2 → CONV3 → CONV4 → CONV5 → FC1 → FC2 → FC3 (1000 classes)
```

| Layer | Filters | Dim | Stride | Pad |
|-------|--------:|-------|-------:|----:|
| CONV1 | 96 | 11×11 | 4 | 0 |
| CONV2 | 256 | 5×5 | 1 | 2 |
| CONV3 | 384 | 3×3 | 1 | 1 |
| CONV4 | 384 | 3×3 | 1 | 1 |
| CONV5 | 256 | 3×3 | 1 | 1 |
| FC1 | 4096 neurons | — | — | — |
| FC2 | 4096 neurons | — | — | — |
| FC3 | 1000 neurons | — | — | — |

- Activations: ReLU after each CONV and FC layer.
- Optimiser: SGD with Momentum.
- Regularisation: Dropout in FC1 and FC2.
- Total trainable parameters: ~60M.
- Training: 2× Nvidia GTX 580 3GB GPUs for 5–6 days.

#### Inception / GoogLeNet (ILSVRC 2014 winner)

- Top-5 test error: **6.7%** — close to human-level performance.
- 22-layer deep CNN.
- Trainable parameters: ~4M (vs ~60M for AlexNet) — significantly reduced.
- Introduces a novel **inception module** (filters of multiple sizes in parallel).
- Optimiser: RMSProp.

### Understanding Inception and ResNet

#### 1×1 Convolution — what's the point?

A `1×1` filter over a `6 × 6 × 1` volume is pointless — the result is still `6 × 6 × 1`. The trick is that `1×1` is useful when the input has **many channels**, because it can mix or reduce channels at low cost.

Example: `+ ReLU` with 32 filters of size `1 × 1 × 64`:

- `(6 × 6 × 64) → (6 × 6 × 32)` — channels reduced from 64 to 32.
- `64 × 64 × 256 → 64 × 64 × 128` with 128 such `1×1` filters.

#### Motivation behind the Inception module

Choosing the "right" filter size is hard:

- Large filters detect large objects, small filters detect small objects.
- Real images have a large variation in object size.

**Inception's answer**: don't choose — use multiple filter sizes (1×1, 3×3, 5×5) **and** pooling **in parallel** within the same module, and concatenate their outputs. The result is a more complicated architecture but better performance, e.g. taking a `28 × 28 × 192` feature map to `28 × 28 × 256` (= 64 + 128 + 32 + 32 channels from the four parallel branches).

#### Reducing computation with 1×1 convolutions

A naive `5 × 5` CONV with 32 filters on a `28 × 28 × 192` input costs:

`28 × 28 × 32 × 5 × 5 × 192 ≈ 120M multiplications`

That is prohibitively expensive. With a **1×1 bottleneck** that first reduces 192 channels to 16:

- `1×1` step: `28 × 28 × 16 × 192 ≈ 2.4M`
- `5×5` step: `28 × 28 × 32 × 5 × 5 × 16 ≈ 10M`
- **Total ≈ 12.4M** — roughly **10× cheaper** than the naive version.

#### Bottleneck Layer

A **bottleneck** uses a 1×1 conv to squeeze the channel dimension (e.g. `192 → 16`) before the expensive operation, then a follow-up CONV restores it to the target depth (`16 → 32`).

#### Inception module V1 — GoogLeNet

GoogLeNet (2014) stacks **9 Inception modules**. Each module runs 1×1, 3×3, 5×5 convolutions and a max-pooling branch in parallel, concatenating the outputs along the channel axis (with 1×1 bottlenecks for cheaper computation).

Deeper networks suffer from the vanishing-gradient problem, so GoogLeNet adds two **auxiliary classifiers**:

- Apply Softmax to an intermediate feature map.
- Compute an auxiliary loss.
- Used only during training (`Total loss = real_loss + 0.3 · Aux_loss_1 + 0.3 · Aux_loss_2`).

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

### Image classification vs. detection vs. segmentation

| Task | Output | Performance metric |
|-----------------------|-----------------------------------------|-------------------|
| Classification | Label | Accuracy |
| Localization | Bounding box `(x, y, Ht, Wd)` or `(x, y, x', y')` | IoU |
| Object detection | Class + bounding box for **every** object | mAP, IoU |
| Instance segmentation | Pixel mask per object | mAP at mask IoU |

A single-object image (`DOG` only) is image classification; a multi-object image (`DOG`, `CAT`, `DOG`, …) is detection or segmentation.

### Classification + Localization as a multi-task CNN

Modify a standard classification CNN (pre-trained on ImageNet, e.g. AlexNet / VGG-16 / ResNet) by adding **two heads**:

- **Classification head** — Softmax loss for the class label.
- **Regression head** — L2 loss against the ground-truth 4 numbers `(x, y, Ht, Wd)`.

Total loss combines both as a **multi-task loss**. The regression head can sit either right after the last CONV layer (before the FC head) or after the last FC layer.

### Detection as a regression problem

Idea: predict, for each object, a class **and** `(x, y, Ht, Wd)`.

A naive way is a **sliding window**: crop fixed-size windows at multiple positions / scales and run a CNN on each:

- Window at the corner → "Dog? No, Cat? No, Background? Yes".
- Window over the cat → "Cat? Yes".
- Window over the dog → "Dog? Yes".

**Issues with sliding windows**:

1. CNN must be applied to a very large number of windows.
1. Need multiple scales and aspect ratios.
1. Inaccurate bounding boxes.
1. Computationally expensive.

**Better: region proposals.** Find blobs in the image that are most likely to contain objects (e.g. **Selective Search** produces ~1000–2000 region proposals using CPU).

### Case Study: R-CNN family

#### R-CNN (Slow)

- Generate ~2K region proposals via selective search.
- Warp each to the CNN's input size and pass each through the ConvNet.
- Classify each region with **SVMs**; linear regression refines the bounding-box offsets.
- mAP: **62.4%** on PASCAL VOC 2007.
- Problem: very slow — runs the CNN ~2K times per image.

#### Fast R-CNN

- Run the whole image through the ConvNet **once**.
- Take ROIs (from selective search) from the resulting feature map, **crop + resize** features (RoI pooling).
- Per-region head outputs object category (linear + Softmax) and box offset (linear).
- mAP: **70%** on PASCAL VOC 2007.

#### Faster R-CNN

- Replace the external proposal step with a learned **Region Proposal Network (RPN)** that shares features with the detection CNN.
- mAP: **78.8%** on PASCAL VOC 2007.

**Family intuition**:

- R-CNN — look at every patch one by one.
- Fast R-CNN — look once, then inspect patches on the feature map.
- Faster R-CNN — propose patches using a neural network (RPN).

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

### Strategies for predicting bounding boxes

#### Sliding Window

- Crop images at many positions / scales and classify each crop with a CNN.
- **Issues**: slow, very expensive computationally, and produces inaccurate bounding boxes.

#### Region proposals

Use selective search (or similar) to suggest ~1–2K candidate regions, then run the CNN only on those — basis of R-CNN family.

#### Grid + per-cell prediction (the YOLO idea)

Place a grid over the image and predict, for **each grid cell**, whether an object's mid-point lies in that cell along with its bounding box and class. With classes `{car, bike}`, each cell predicts a vector

`Y_cell = {p_o, x, y, h, w, c_1, c_2}`

- `p_o` = 1 if the cell contains an object's mid-point, 0 otherwise.
- `(x, y, h, w)` = bounding-box centre and size relative to the cell.
- `(c_1, c_2)` = one-hot class probabilities.

For a `3 × 3` grid with two classes the target tensor is shaped `3 × 3 × 7` (= 5 + #classes). In practice grids are finer, e.g. `19 × 19 × 7`, which works well for non-overlapping objects.

**Idea**: take the mid-point of each object and assign it to the grid cell that contains it.

### Non-Maxima Suppression (NMS)

Each object has only one mid-point, but neighbouring cells may also predict overlapping boxes. NMS removes duplicates and keeps only the highest-confidence detection:

1. Discard boxes with confidence below a threshold (e.g. 0.7).
1. From the remaining boxes, pick the one with the highest score as a final detection.
1. Discard any remaining box with `IoU > 0.5` against that detection (i.e. heavy overlap).
1. Repeat until no boxes remain.

### Anchor boxes — handling overlapping objects

YOLO's basic form assigns each grid cell to **one** object. For multiple overlapping objects (mid-points in the same cell) only one would be detected.

**Solution**: predefine `k` **anchor boxes** of different shapes per cell. Each anchor is responsible for objects whose shape (aspect ratio / size) best matches it. Targets become:

`Y_cell = {p_o, x, y, h, w, c_1, c_2}_anchor1 + {p_o, x, y, h, w, c_1, c_2}_anchor2 + …`

With 2 anchors per cell on a `19 × 19` grid and 2 classes:

`Y shape = 19 × 19 × 2 × 7`

Assigning an object means selecting **both** the cell containing its mid-point and the anchor box with the highest IoU to its bounding box.

### Case Study: YOLO and SSD

#### YOLO — You Only Look Once

- Real-time: ~45 FPS (0.02 s per image).
- Weakness: not great for small objects; struggles with new or unusual aspect ratios.

#### SSD — Single Shot Detector

- Similar architecture to YOLO; uses VGG-16 base CONV layers.
- Many anchor boxes with different aspect ratios.
- ~3× faster than Faster R-CNN.
- Still not great on small objects; using a ResNet-101 base can help with small-object features.

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

### Recap — anchor-based object detection

YOLO places a grid over the image and predicts, per cell + per anchor, a vector `{p_o, x, y, h, w, c_1, …, c_k}`. With a `19 × 19` grid, 2 anchors and 2 classes the target tensor is `19 × 19 × 2 × 7`.

**Drawbacks of anchor-based detectors**:

- Sensitive to anchor box **size** and **aspect ratio**.
- Number of anchor boxes is fixed.
- Too much variation in object shape; small objects suffer.
- May not generalise — anchors are predefined offline.
- Computationally expensive.

### Anchor-free detectors

Localise objects without using boxes as proposals. Two broad categories:

- **Key-point based** — detect spatial points unique to an object. Examples: facial key points (nose, eyebrows…), human-body joints (elbows, knees…). Each object is represented by its key points.
- **Center-based** — find object centres and predict four distances from the centre to the object's bounding-box edges.

### YOLO timeline

```text
2015 — YOLO v1                       2021 — YOLO v5, YOLO R, YOLO X (anchor-free)
2016 — YOLO v2                       2022 — YOLO v6, YOLO v7
2018 — YOLO v3                       2023 — YOLO v8, YOLO-NAS
2020 — YOLO v4, Scaled YOLO v4       2024 — YOLO v9, v10, v11
                                     2025 — YOLO v12
                                     2026 — YOLO 26
```

### Case Study: YOLOX (anchor-free)

- "YOLOX: Exceeding the YOLO Series", 2021.
- Anchor-free detector in the YOLO family.
- Uses a **decoupled head** (separate branches for classification and regression).
- Label assignment via **SimOTA**.
- Backbone: YOLO v3 SPP with **DarkNet-53**.
- Uses advanced augmentation: **Mixup** and **Mosaic**.

Every YOLO architecture has three parts:

- **Backbone** — feature extraction.
- **Neck** — aggregation of multi-scale features (typically FPN / PANet).
- **Head** — localisation and classification scores.

#### YOLOX augmentations

- **Mixup** — blend two training images and their labels (`λ` and `1 − λ`).
- **Mosaic** — combine four training images into one mosaic per training step; exposes the model to many objects and scales in a single sample.

### YOLO 26 — The Next Evolution

Real-time computer-vision model by Ultralytics:

- Supports detection, segmentation, classification, pose, tracking, OBB (oriented bounding boxes).
- Available in **Nano / Small / Medium / Large / XLarge** sizes.
- End-to-end detection pipeline (**NMS-free** inference).
- Designed for edge AI and fast deployment.

**Why is it faster?**

- NMS-free inference removes post-processing overhead.
- Direct bounding-box regression (no DFL).
- Lower latency and simpler deployment graph.
- CPU-optimised architecture — up to **43% faster on CPUs than YOLO 11** (Ultralytics benchmark).

**Key changes**:

- **ProgLoss** (Progressive Loss Balancing) — improves training stability and convergence.
- **STAL** (Small-Target-Aware Label Assignment) — improves small-object detection.
- **MuSGD** optimiser — faster convergence.
- Better speed–accuracy trade-off than previous YOLO models.
- Ideal for robotics, drones, surveillance, and edge devices.

The Yolo26 release ships three variants — the high-level **inference architecture**, a **training pipeline**, and a published **performance** comparison (Ultralytics' benchmark dashboard).

### Instance Segmentation

**Semantic vs. instance segmentation**:

- **Semantic segmentation** classifies each pixel into a class/category — but does not distinguish individual instances of the same class.
- **Instance segmentation** identifies each pixel **and** the specific object instance it belongs to (e.g. two separate dogs get two separate masks).

| Task | Popular techniques |
|-----------------------|------------------------------------------------------------------------------------------------------|
| Semantic segmentation | Conditional Random Field (CRF), Fully Convolutional Network (FCN), U-Net, Pyramid Scene Parsing Network (PSPNet), … |
| Instance segmentation | SegNet, DeepMask, SharpMask, **Mask R-CNN**, … |

**Applications**: autonomous driving (lane / vehicle / pedestrian masks), scene understanding, aerial image processing.

### Case Study: Mask R-CNN

- **Mask-RCNN** = Mask-Region Convolutional Neural Network.
- Adds an instance-segmentation head to the R-CNN family on top of **Faster R-CNN**.
- Uses a Fully Convolutional Network (FCN) to predict a mask per class / object.

Two stages:

1. **RPN** proposes candidate object bounding boxes.
1. Classify the candidates, refine bounding boxes, and predict the per-instance mask.

#### Mask R-CNN limitations

- **Computational complexity** — training and inference are resource-heavy, especially on high-resolution images or large datasets.
- **Small-object segmentation** — struggles with very small objects (limited pixel information).
- **Data requirements** — needs a lot of annotated data; expensive to acquire.
- **Limited generalisation** to unseen object categories.

### Semantic Segmentation

Semantic segmentation classifies every pixel into a category (e.g. road / car / sky), without distinguishing individual instances.

#### Case Study: U-Net

U-Net is a popular **encoder–decoder** architecture for semantic segmentation:

- Encoder progressively down-samples (CONV + POOL) to capture context.
- Decoder progressively up-samples (transposed CONV / up-conv) to recover spatial resolution.
- **Skip connections** copy feature maps from each encoder stage to the corresponding decoder stage, preserving fine spatial detail.

A predicted mask is a per-pixel binary (or per-class) map, e.g. a `cat` mask:

```text
. . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . X X . . . . . . .
. . . . . . . . . . . . . . . X X X . . . X X .
. . . . . . . . . . . . X X X X X X X X X X X .
. . . . . . . . . . . . X X X X X X X X X X X .
. . . . . . . . . . . . X X X X X X X X X X X .
. . . . . . . . . . . . . X X X X X X X X . . .
. . . . . . . . . . . . . . X X X X X X . . . .
. . . . . . . . . . . . . . . X X X X . . . . .
. . . . . . . . . . . . . . . . X X . . . . . .
```

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

### Sequence modelling — types and applications

| Pattern | Example | Sample task |
|----------------|--------------------------------------|--------------------------------------------------------------------------|
| One to one | "Will it rain today?" → Yes/No | Binary classification |
| Many to one | "42028 is the best subject so far!" | Sentiment analysis |
| One to many | Image → "A woman is throwing frisbee" | Image captioning |
| Many to many | "Hey Siri, what's the weather?" → reply | Q&A with LLMs, language translation |

Compared with a non-sequence task like house-price regression (`features = [size, #bedroom, #bathroom, garden, location] → price`), sequence tasks have variable-length input or output.

### Recurrent Neural Network (RNN) Basics

A recurrent cell consumes the current input `x_t` and the previous hidden state `h_{t-1}`, then emits an output `ŷ_t` and a new hidden state `h_t`. The same parameters are shared across time steps:

```text
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t)
ŷ_t = W_y · h_t
```

Visually:

```text
x_0 → [ RNN ] → ŷ_0
       │
       h_0
       ▼
x_1 → [ RNN ] → ŷ_1
       │
       h_1
       ▼
…
       │
       h_{t-1}
       ▼
x_t → [ RNN ] → ŷ_t
```

Training uses **backpropagation through time (BPTT)**: forward through every time step accumulating losses `L_0, L_1, …, L_t` (total loss `L`), then backward through the unrolled graph to update `W_hh`, `W_xh`, `W_y`.

### Sequence modelling — design criteria

- Support for variable-length input.
- Captures temporal dependency (short-term and long-term).
- Preserves the order of information.
- Shares parameters across time steps.

### RNN limitations

- Prone to **vanishing / exploding gradient** problems.
- Long-term memory not well supported.
- Slow — no parallelisation across time steps.

### Attention mechanism

**Idea**: forget recurrence; feed everything into a dense network and let it decide what to focus on. The 2017 paper title says it all — *Attention Is All You Need*.

**Why do we need attention?**

- RNNs process sequences one step at a time.
- Long sentences lead to long-term memory loss.
- Important words may be hidden in long dependencies.
- Attention focuses on the relevant parts of the input.

**Intuition** (mimicking human focus: "where should I look?"):

- For each output token, attention decides which input tokens are most important.
- Compute a **weighted sum** of all input vectors — higher weight = more relevant.

For each attention step, compute the similarity between the **Query Q** and each **Key K_i**, then extract the **Values V_i** weighted by that similarity. Mathematically (scaled dot-product attention):

`Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`

### Transformers

Self-attention is the foundation of the **Transformer** architecture:

- The entire sequence is processed in parallel.
- Has an **encoder** and a **decoder** block (each is a stack of layers with self-attention and feed-forward sub-layers).

### Case Study: Vision Transformer (ViT)

- Introduced in 2021: *"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,"* ICLR 2021.
- Treats an image like text — represents it as a **sequence of patches**.

**Steps**:

1. Split the image into fixed-size patches.
1. Flatten each patch.
1. Produce lower-dimensional linear embeddings from the flattened patches.
1. Add positional embeddings.
1. Feed the sequence into a standard transformer encoder.
1. Pre-train with image labels on a huge dataset (fully supervised).
1. Fine-tune on the downstream dataset for image classification.

#### CNN vs. Vision Transformer

| Aspect | CNN | Vision Transformer (ViT) |
|---------------------|--------------------------------------------------|------------------------------------------------------|
| Input handling | Processes the entire image with conv filters | Splits image into fixed-size patches (like tokens) |
| Local vs. global | Focuses on local patterns first (edges, textures)| Uses **global self-attention** to relate all patches |
| Architecture | Hierarchical (conv → pool → deeper features) | Flat stack of transformer encoders |
| Training data need | Works well with limited data | Needs lots of data or strong pre-training |
| Computation | Efficient on low-resolution inputs | Heavier, especially on large images |
| Parallelism | Limited — sequential feature stacking | High — patch processing is highly parallelisable |

### Case Study: RF-DETR (object detection with transformers)

- An improvement over the original **DETR** (Detection Transformer).
- DETR looks at everything globally but misses small objects.
- **RF-DETR** looks globally **and** understands the relationships between objects.
- Real-time, transformer-based object-detection architecture.
- Outperforms many object-detection models; 60+ mAP on COCO.

### Case Study: Diffusion Models — Intuition

**Goal**: generate new data samples (images, audio, text) that look like a training dataset by learning to **reverse a gradual noise process** — a generative model.

**Generic steps**:

1. Start with real data.
1. Add noise step-by-step until the image becomes pure noise.
1. Train a model to reverse this process — denoising to recover the original image.
1. Once trained, the model can start from pure noise and generate new, realistic samples.

#### Diffusion in practice

Example training data: many sprite images. Task at inference time: generate a *new* sprite, or with a text-to-image variant generate an image from a text prompt (e.g. "butterfly image" → a butterfly).

- **Forward diffusion** — add Gaussian noise gradually over many steps until the image is pure noise. No learning is required here.
- **Reverse diffusion (denoising)** — a model is trained to predict and reverse the noise. Given a noisy image it predicts a slightly less noisy version; after many steps it reconstructs a clean, new image from pure noise.
