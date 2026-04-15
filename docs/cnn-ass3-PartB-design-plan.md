# 42028: Deep Learning and Convolutional Neural Network Dataset details, GUI design, Implementation Plan

Project Title and number: Plumbing Defect Detection & Pipe Inspection System #37

Team Name: Plumbers of UTS

Team Members:

|Student Name|Student ID|Email ID|
|---|---|---|
|Bo Zhao|25971060|Bo.Zhao@student.uts.edu.au<br><br>|
|Jadyn Braganza|26055044|Jadyn.Braganza@student.uts.edu.au<br><br>|
|Eunkwang Shin|26170530|Eunkwang.Shin@student.uts.edu.au<br><br>|

Dataset Details:

- Pipeline Defect Dataset
  - Link: https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset/data
  - This public dataset features pipeline defect data from videos captured by robot vehicles in industrial environments. It includes 16 types of defects, categorized into structural (e.g., broken, deformation, misalignment, disconnection) and functional (e.g., deposition, obstacle) defects.
- Sewer Defect Detection datasets
  - Link: https://universe.roboflow.com/sewage-defect-detection-s68df/sewage-defect-detection/dataset/1/download
  - Sewage Defect Detection is for detecting defects in sewer pipes, containing 980 images and 7 classes: Crack, Hole, Obstacle, Debris, Buckling, Joint offset, and Utility intrusion. The dataset is publicly available on Roboflow Universe.
- SewerML Dataset
  - Link: https://sciencedata.dk/shared/Large%20AAU%20files/Sewer_ML (PW: SewerML4294lsdaw321)
  - This dataset is a pipeline endoscopic image defect detection and annotation dataset, primarily designed for intelligent inspection and defect recognition tasks in urban underground pipelines.
  - The dataset is organized by image files, with each image corresponding to an annotation record that indicates the types of pipeline defects present in that image. The defect categories include root intrusion (RB), pipe misalignment (FS), attached deposits (AF), valves (VA), leakage (IS), deformation (DE), and a total of 18 defect types, annotated using a multi-label binary format (0 indicates absence, 1 indicates presence). In addition, the dataset records the water level information (WaterLevel) for each image, as well as an overall judgment field (Defect) to indicate whether the image contains any defects.

Task Type: This project is a combination of Object detection, Image Segmentation and Image Classification. This project will implement YOLO to detect and locate multiple defects in an image with the resulting bounding boxes then passed as prompts to the Segment Anything Model (SAM). This generates precise segmentation masks. Segmentation goes further than bounding boxes and enables pixel-level mapping of each defect which allows for accuracy measurement of affected area and severity scoring. It will then use Image Classification to categorize the type of defect (e.g. crack, blockage, rot).

The dataset split will be 60% for training, 20% for testing, 20% for validation.

## GUI Design:

A web application was selected as it requires no installation and can be accessed anywhere. To validate the layout and user flow before development, a prototype was constructed using HTML and CSS and deployed via GitHub Pages. The final implementation will use TypeScript and React to build a maintainable interface.

The prototype consists of four screens: Dashboard for inspection metrics and defect distribution, Detect for uploading images and running YOLO-based detection, History for reviewing past records, and Model Info for model architecture and performance details.

https://plumbers-of-uts.github.io/pipevision-ai/gui-mockup.html

## Implementation plan:

## 1. CNN Architectures & Training Strategies

To meet the project objectives of detecting sewer pipe defects, this project adopts a multimodel hybrid architecture combining object detection, segmentation, and classification.

The project is broken down into the following sections:

1. Object Detection: The first step is to identify the location of the defects. YOLO will serve as the primary detection backbone that localizes the defects across inspection frames. YOLO-based architectures are well-established in pipe inspection literature due to their favourable trade-off between inference speed and mean Average Precision (mAP), making them suitable for processing large volumes of CCTV footage.
2. Segmentation: After the defects are identified, segmentation is used to narrow down the precise location of the defect. The Segment Anything Model (SAM) will be used because of its precision in generating pixel-level segmentation masks from YOLO-predicted bounding boxes. This approach enables precise spatial representation of defect boundaries, supporting more accurate severity assessment beyond coarse bounding-box localisation.
3. Classification: For multi-label defect classification on the datasets, this project will evaluate ResNet-50 and TResNet-L as convolutional backbones. Both architectures have demonstrated strong performance on datasets characterised by complex textures and class imbalance, conditions inherent to sewer inspection imagery. TResNet-L will be prioritised where inference efficiency is critical, given its optimised memory and throughput characteristics.
4. Optimisation Strategy: Model training will employ AdamW for transformer-based and attention-augmented components, leveraging its decoupled weight decay regularisation to improve generalisation. For the YOLO detection stages, Stochastic Gradient Descent (SGD) with momentum will be applied, consistent with findings in the object detection literature showing superior convergence behaviour during the later stages of training on structured visual tasks.

## 2. Testing & Evaluation Plan

Testing and evaluation will be conducted on test splits, with no test data used during model development or hyperparameter tuning. Where dataset sizes permit, k-fold cross-validation will be applied during training to reduce variance in performance estimates. Results will be reported with confidence intervals where applicable, and ablation studies will be conducted to isolate the contribution of each model component.

The performance benchmarks targeted are:

- mAP@0.5: For object detection accuracy.
- F2-CIW Score: Specifically, for the SewerML dataset to account for the economic impact of different defect types.
- Accuracy Expectation: We aim for a mAP of 85-90% on common defects (cracks, roots) and an F1-score of ~90% for identifying normal (non-defective) pipes.

## 3. Error Handling (Overfitting & Under-fitting)

Sewer datasets are often highly imbalanced due to the different types of defects. We will address this through:

- Data Augmentation: Using geometric transforms (rotations, flips) and colour jittering to simulate different lighting conditions inside pipes.
- Regularization: Implementing Dropout (0.2) and L2 Weight Decay (0.01) in the ResNet/YOLO backbones to prevent the model from memorizing specific pipe segments.
- Class Weighting: Applying Class-Importance Weights (CIW) during loss calculation to ensure the model prioritizes rare but critical defects (e.g., severe structural collapses).

## 4. Tentative Implementation Timeline

|Phase|Tasks|Duration|
|---|---|---|
|Data Prep|Clean and unify the Pipeline Defect Dataset, SewerML Dataset, and Sewer Defect Detection Dataset by aligning their label formats and resolving annotation inconsistencies.|Weeks 6|
|Model Dev|Set up the YOLO object detection and SAM segmentation environment; train baseline models on the prepared dataset.|Weeks 7-8|
|Optimization|Perform hyperparameter tuning (optimizers, learning rates, batch sizes) and apply regularization techniques to address overfitting.|Weeks 9-10|
|Final Testing|Benchmark trained models on held-out test sets and evaluate performance using mAP, precision, and recall metrics.|Weeks 9-10|
|UI & Reports|Build the web-based graphical interface and finalize the report.|Weeks 11-12|
