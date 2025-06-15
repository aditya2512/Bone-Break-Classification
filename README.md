
# Bone Fracture Classification with Deep Learning

This project leverages deep learning and medical imaging to classify various types of bone fractures from X-ray images. It uses a ResNet50-based model and provides an interactive web interface using Streamlit for real-time fracture diagnosis.

---

## Dataset

- **Source:** [Kaggle - Bone Break Classification](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset)
- **Structure:** Folder-based dataset with 10 classes:
  - `Avulsion`, `Comminuted`, `Fracture-Dislocation`, `Greenstick`, `Hairline`, `Impacted`, `Longitudinal`, `Oblique`, `Pathological`, `Spiral`

---

##  Model Architecture

- **Base Model:** ResNet50 (pretrained on ImageNet)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dropout (0.5)
  - Dense (softmax) for 10-class output
- **Training Strategy:**
  - Phase 1: Train classifier head with base frozen
  - Phase 2: Fine-tune top 50 ResNet layers
- **Optimization:** Adam optimizer, categorical crossentropy, early stopping, class weighting

---

##  Preprocessing & Augmentation

- Image resizing to 224×224
- Normalization to [0, 1]
- Augmentations:
  - Rotation, flip, zoom, brightness, shift, shear

---

## Evaluation

- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Validation Split:** 20%
- **Class Imbalance Handling:** Weighted loss using `sklearn.utils.class_weight`

---

## Streamlit Web App

An interactive app is provided via Streamlit:

### Features:
- Selects random images from dataset for prediction
- Displays predicted fracture type and confidence
- Uses trained `bone_fracture_model.h5` for inference

### To Run Locally:


pip install streamlit tensorflow pillow
cd "Bone Break Classification"
streamlit run streamlitapp.py


## Real-World Applications
Assists radiologists in clinical decision-making

Provides quick diagnosis support in remote or emergency settings

Enables mobile and edge-based healthcare tools

##Project Structure

Bone Fracture Classification/
│
├── bone_fracture_model.h5          # Trained Keras model
├── streamlitapp.py                 # Streamlit app
├── README.md                       # This file
└── Bone Fracture Classification/                        # X-ray image dataset
