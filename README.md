# 🧠 NeuroScan AI - Brain Tumor Detection & Neural Explainability

**NeuroScan AI** is an end-to-end deep learning web application for brain tumor classification from MRI scans. It features a fine-tuned **VGG-19** model, **Grad-CAM (Gradient-Weighted Class Activation Mapping)** visualization for explainable AI, and a modern dark-themed medical dashboard.

---

## 📊 Fine-Tuning Performance & Results

| Model Stage | Train Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Before Fine-Tuning** | 84.22% | 86.13% | 84.52% | 0.3130 |
| **After Fine-Tuning** | **95.57%** | **93.23%** | **93.55%** | **0.1785** |

### Fine-Tuning Improvements Applied:
1. **Top-Block Unfreezing**: Unfroze Block 5 layers (`block5_conv1` to `block5_conv4`) of VGG-19 to adapt high-level texture features specifically to brain MRI tissue contrast.
2. **Optimized Learning Rate**: Used `Adam(lr=5e-5)` with `ReduceLROnPlateau` for stable convergence.
3. **Batch Size Tuning**: Reduced batch size from `64` to `32` on 1,445 training samples to double gradient update frequency per epoch.
4. **Regularization**: Increased `Dropout` to `0.3` to eliminate overfitting and boost test set generalizability.

---

## ✨ Features

- ⚡ **Dynamic Model Hot-Reloading**: Flask backend monitors `model.h5` modification timestamp on disk and hot-reloads model weights seamlessly without server downtime.
- 🎯 **Grad-CAM Visual Heatmaps**: Generates neural activation heatmaps from `block5_conv4` overlaid onto brain MRI scans using OpenCV JET colormaps.
- 🎨 **Modern Medical Dashboard**: 
  - **Analysis Tab**: Drag-and-drop MRI scan loading, preview, SVG confidence gauge ring, and Grad-CAM heatmap visualization.
  - **Reports Tab**: Interactive diagnostic log table with search filtering, summary stats (Total Scans, Tumor Count, Avg Confidence), and side-by-side report detail modal.
  - **Settings Tab**: Customizable Grad-CAM target layers, colormaps (`JET`, `VIRIDIS`, `HOT`, `INFERNO`), alpha transparency slider, and manual weight reload triggers.

---

## 🛠️ Project Structure

```
Brain Tumor/
├── app.py                      # Flask Server with Grad-CAM & Weight Hot-Reloading
├── brain_tumor_dataset/        # Training script (train_brain_tumor.py)
├── templates/
│   └── index.html              # NeuroScan AI Dashboard UI
├── static/
│   ├── css/style.css           # Vanilla CSS3 Dark Medical Theme
│   └── js/app.js               # Frontend Controller & REST API Handlers
├── Detection.py                # Preprocessing & evaluation utilities
├── training_performance.png    # Training vs Validation Accuracy/Loss plot
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore specifications
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10+ and TensorFlow are installed:
```bash
pip install tensorflow opencv-python flask numpy matplotlib imutils requests
```

### 2. Launch Backend Application
Run the Flask backend server:
```bash
python app.py
```

### 3. Open Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
Upload any brain MRI scan (JPG/PNG) to analyze and visualize Grad-CAM activation heatmaps!
