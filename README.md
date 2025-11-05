# 🌈 Aftershoot AI Challenge — White Balance Prediction  

---

## 🧠 Overview  
This project was built for the **Aftershoot AI Challenge** hosted by **Aftershoot on HackerEarth**.  
It predicts **Temperature** and **Tint** values for white balance correction using **RAW TIFF images** and their **EXIF metadata**.  

The goal was to design a model that learns how photographers edit their images and automatically applies similar white balance adjustments to new photos.  
I implemented and trained the full pipeline in two environments — **VS Code (local CPU)** and **Google Colab (GPU)** — to analyze performance and convergence differences.  

---

## ⚙️ Model Approach  

I developed a **hybrid deep-learning model** that fuses both **visual** and **numerical** data for consistent white-balance correction.  

- **Image Branch:** Pretrained **ResNet18** (`torchvision.models`) to extract visual embeddings from each 256×256 TIFF image.  
- **Metadata Branch:** Lightweight **MLP (Multi-Layer Perceptron)** to process numerical EXIF parameters such as  
  `grayscale`, `aperture`, `flashFired`, `focalLength`, `isoSpeedRating`, `shutterSpeed`, `currTemp`, and `currTint`.  
- **Fusion Layer:** Combined both image and metadata features to form a single embedding.  
- **Output Layer:** Predicted continuous target values — **Temperature** and **Tint**.  
- **Loss Function:** **Mean Absolute Error (L1Loss)** to directly align with the challenge’s evaluation metric (**MAE**).  
- **Optimizer:** **Adam** with learning rate = `1e-4` and a **StepLR** scheduler (`step_size=5`, `gamma=0.5`).  
- **Training Epochs:** 20 total epochs in both setups.  

---

## 🧮 Data Preprocessing & Feature Engineering  

- Dropped irrelevant columns (`copyCreationTime`, `captureTime`, `touchTime`, etc.).  
- Selected key numeric metadata features available in both training and validation CSVs.  
- Replaced missing values with column-wise mean.  
- Standardized numeric columns using **StandardScaler**.  
- Scaled target labels (**Temperature**, **Tint**) for stable gradient flow.  
- Built a **custom PyTorch Dataset** to load image tensors and corresponding metadata efficiently.  

---

## ⚡ Training Comparison — VS Code (CPU) vs Google Colab (GPU)

| Environment | Device | Training Time (20 Epochs) | Batch Size | Notes |
|--------------|---------|---------------------------|-------------|--------|
| **VS Code (Local)** | CPU (Intel Core i5) | ⏱ ~113 minutes | 16 | Training slower due to CPU-only computation, but stable convergence. |
| **Google Colab (Cloud)** | GPU (Tesla T4, CUDA) | ⚡ ~20 minutes | 32 | Much faster due to GPU acceleration, identical loss curve and results. |

Both environments achieved consistent convergence patterns, proving the model’s reproducibility.  

---

## 📉 Training & Optimization Details  

- Trained for **20 epochs** using **MAE (L1Loss)**.  
- Validation loss stabilized around **0.13 – 0.14 MAE**, showing reliable accuracy.  

**Saved best model checkpoints:**  
- `aftershoot_model.pth` → VS Code (CPU version)  
- `colab_aftershoot_model_v2.pth` → Colab (GPU version)  

---

## 🧾 Inference & Results  

- Loaded the saved model checkpoint for inference.  
- Used **inverse-scaling** to convert predictions back to real-world ranges.  
- Generated outputs for 493 validation images in correct HackerEarth format:  

id_global,Temperature,Tint
EB5BEE31-8D4F-450A-8BDD-27C762C75AA6,5276,13
DE666E1F-0433-4958-AEC0-9A0CC0F81036,5227,14
...


**Temperature Range:** 2000 – 12000 K  
**Tint Range:** −20 to +30  

Predictions closely aligned with human-edited color-tone distributions.  

---

## 📊 Results Summary  

| Version | Environment | Final MAE | Training Duration | Output File |
|----------|--------------|-----------|------------------|-------------|
| **VS Code (CPU)** | Local | 0.15 | 113 min | `submissions/predictions.csv` |
| **Google Colab (GPU)** | Cloud | 0.13 | 20 min | `submissions/Colab_predictions.csv` |

---

## 💡 Key Insights  

- The **hybrid CNN + MLP architecture** successfully fused image and metadata inputs.  
- **GPU-based training** drastically improved speed without changing accuracy.  
- Model reproducibility confirmed by identical MAE trends across environments.  
- Final predictions were realistic, smooth, and within valid Temperature/Tint ranges.  

---

## 🏁 Final Statement  

I developed a **CNN + metadata fusion model** using **PyTorch** for accurate **Temperature** and **Tint** prediction in white-balance correction.  
The model was trained and tested in both **VS Code (CPU)** and **Google Colab (GPU)** environments, achieving consistent **MAE** performance with faster convergence on the GPU setup.  
