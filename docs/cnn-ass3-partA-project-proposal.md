# 42028: Deep Learning and Convolutional Neural Network Project Proposal

Project Title: Plumbing Defect Detection & Pipe Inspection System

Team Name: Plumbers of UTS

Team Members:

|Student Name|Student ID|Email ID|Tutorial Session|
|---|---|---|---|
|Bo Zhao|25971060|Bo.Zhao@student.uts.edu.au|03|
|Jadyn Braganza|26055044|Jadyn.Braganza@student.uts.edu.au|09|
|Eunkwang Shin|26170530|Eunkwang.Shin@student.uts.edu.au|03|

Abstract (~200 words):

Sydney’s ageing sewer infrastructure loses approximately 114–115 million litres of water each day due to leaks and pipe failures, costing an estimated AUD $228,000–$230,000 daily, as reported by Sydney Water. Current manual inspection methods are time-consuming, hazardous, labour-intensive, and prone to human error. This emphasizes the need for a safer and more efficient detection approach.

This project proposes the development of a software-based sewer defect detection system that automatically identifies cracks, leaks, and blockages in sewer pipe images using deep learning techniques. The system leverages publicly available datasets and annotated images from several sewer image datasets to train a customized YOLO object detection model and use SAM to generate segmentation masks that support annotation and training. The model can accurately classify and localise defects. A user-friendly graphical interface will enable users to upload images or video frames, visualise detected defects, and generate structured reports summarising defect types and locations.

The anticipated outcome is an accurate, efficient, and scalable solution for sewer condition monitoring that reduces reliance on manual inspections, lowers operational costs, and improves safety without requiring specialised hardware.

## Dataset Details (If known, optional):

- Sewer pipe defect datasets on HuggingFace
  - Link: https://huggingface.co/datasets/SRuibo/Sewer-pipe-defects
- Sewer Defect Detection datasets on Roboflow
  - Link: https://universe.roboflow.com/sewage-defect-detection-s68df/sewage-defect-detection/dataset/1/download
- SewerML Dataset (thousands of labelled sewer inspection images, free academic dataset)
  - Link: https://sciencedata.dk/shared/Large%20AAU%20files/Sewer_ML (PW: SewerML4294lsdaw321)

## Additional support required (if any):
