# Warehouse AI System  
## Computer Vision, Machine Learning, and RAG Integration

---

## Overview

This project implements an intelligent Warehouse AI System that integrates:

- Object detection using Computer Vision (OpenCV)
- Object classification using Deep Learning (PyTorch)
- Context-aware instruction generation using Retrieval-Augmented Generation (RAG)

The system runs fully offline using local models and follows a modular design.


---

## Tech Stack

### Computer Vision
- OpenCV
- NumPy

### Machine Learning
- PyTorch
- TorchVision
- MobileNetV2 (Transfer Learning)

### RAG System
- SentenceTransformers (MiniLM)
- FAISS
- HuggingFace Transformers
- DistilGPT2

### Integration
- Python 3.x

---

## System Architecture

Input Image
↓
Computer Vision (OpenCV)
↓
Object Detection
↓
ML Classification
↓
RAG Retrieval
↓
Instruction Generation


---

## Module Description

### detect_objects_v2.py (Computer Vision Module)

Purpose:
Detect warehouse packages and objects from input images.

Main Functions:
- preprocess(): Resize and blur image for noise reduction
- color_segmentation(): Segment box regions in HSV color space
- edge_detection(): Detect edges using Canny algorithm
- clean_mask(): Remove noise using morphological operations
- find_objects(): Extract contours and compute bounding boxes

Output:
- Bounding boxes
- Center coordinates
- Approximate object dimensions

---

### train_model_v2.py (Machine Learning Module)

Purpose:
Classify detected objects into predefined categories.

Approach:
- Transfer learning using MobileNetV2
- Final classification layer modified for 3 classes:
  - Heavy
  - Fragile
  - Hazardous

Training Details:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 5
- Batch Size: 32 (default)
- Pretrained Weights: ImageNet

Performance Metrics:

Training Accuracy (Final Epoch): ~94%

Classification Report:

Heavy:
- Precision: 0.95
- Recall: 0.96
- F1-score: 0.95

Fragile:
- Precision: 0.97
- Recall: 0.95
- F1-score: 0.96

Hazardous:
- Precision: 0.89
- Recall: 0.93
- F1-score: 0.91

Overall Accuracy: ~95%

The trained model is saved and used for inference in the integration module.

---

### rag_system.py (RAG Module)

Purpose:
Provide intelligent handling instructions based on stored warehouse documents.

Pipeline:
1. Load text documents from docs folder
2. Generate embeddings using MiniLM
3. Store embeddings in FAISS index
4. Retrieve relevant documents for each query
5. Generate response using DistilGPT2

Key Features:
- Fully offline execution
- No external APIs
- Fast semantic search using FAISS
- Context-based answer generation

---

### main.py (Integration Module)

Purpose:
Integrate Computer Vision, Machine Learning, and RAG modules into a unified system.

Execution Flow:
1. Load input image
2. Perform object detection (CV module)
3. Crop detected objects
4. Classify each object (ML module)
5. Generate handling instructions (RAG module)
6. Allow optional user queries

This file ensures clean execution order and modular interaction.

---

## How to Run

### Step 1: Clone Repository

```bash
git clone https://github.com/Shreya-812/warehouse-ai-system.git
cd warehouse-ai-system
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Integrated System

```bash
cd integration
python main.py
```

## Results

- Reliable object detection for warehouse packages
- High classification accuracy (~95%)
- Context-aware instruction generation
- Fully modular and extensible pipeline
- Offline execution without paid services
- Screenshots and outputs are available in the outputs directory.

### Author

Shreya Singh
