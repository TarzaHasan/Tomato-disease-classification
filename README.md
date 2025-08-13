# 🍅 Tomato Disease Classification

This repository contains the source code, dataset, and requirements for the **Tomato Disease Classification** project.  

The proposed deep learning model integrates **residual learning** and a **Channel Attention Layer (CAL)** to achieve high accuracy while keeping computational costs low, making it suitable for **real-time precision agriculture applications**.

---

## 📂 Repository Contents

- **Tomato/** → Augmented dataset for training and validation.
- **DenseNet.ipynb** → Implementation using DenseNet architecture.
- **InceptionV3.ipynb** → Implementation using InceptionV3 architecture.
- **MobileVit.ipynb** → Implementation using MobileViT architecture.
- **NasNetLarge.ipynb** → Implementation using NasNetLarge architecture.
- **Proposed model.ipynb** → Implementation of the proposed Residual + CAL-based architecture.
- **ResNet.ipynb** → Implementation using various ResNet architectures.
- **SEResNet.ipynb** → Implementation using SEResNet architecture.
- **requirements.txt** → Required dependencies for running the code.

---

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TarzaHasan/Tomato-disease-classification.git
   cd Tomato-disease-classification
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
  **requirements**
   ```bash
  tensorflow>=2.13
  scikit-learn
  numpy
  matplotlib
  seaborn
  torch>=2.0
  torchvision
  timm

 ```
## Usage:

### 1. Dataset
The dataset is available in the **Tomato/** directory and contains the augmented training and validation images.
### 2. Run the training scripts
Open and run any `.ipynb` file to train a model.
**Example: Proposed Model**  
```bash
jupyter notebook "Proposed model.ipynb"
 ```
### 3. Modify Parameters
You can adjust batch size, learning rate, and number of epochs inside each notebook.
### 📊Model Overview
- **Architecture:** →  Residual Learning + Channel Attention Layer (CAL)
- **Key Features:**
  - High classification accuracy
  - Low parameter count (0.52M)
  - Reduced GFLOPs and GMACs
  - Low memory usage (1.99 MB)
  - Suitable for real-time and resource-constrained environments
### Contact
- **Email:** tarza.abdullah@su.edu.krd
