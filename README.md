# DeepFake Image Detection using Vision Transformer

A deep learning project for detecting whether a facial image is **REAL** or **FAKE** using a pretrained **Vision Transformer (ViT)**, enhanced with **Test-Time Augmentation (TTA)** and confidence-aware prediction analysis.

---

## 📌 Project Overview

The rapid development of Generative AI has made it possible to generate highly realistic synthetic human faces. These AI-generated images, commonly known as DeepFakes, can be difficult to distinguish from authentic photographs.

This project develops an automated DeepFake image detection system using a **Vision Transformer (ViT)** instead of a traditional Convolutional Neural Network (CNN).

The model is trained to classify facial images into two categories:

- **REAL** – Authentic human facial images
- **FAKE** – AI-generated/synthetic facial images

In addition to the standard ViT classifier, the project introduces a lightweight **Test-Time Augmentation (TTA)** pipeline. Multiple transformed versions of the same image are passed through the model, and their prediction probabilities are averaged to produce the final prediction.

The project also performs **confidence analysis** to study whether prediction confidence can help identify uncertain or difficult samples.

---

## 🎯 Objectives

The main objectives of this project are:

1. Develop a DeepFake image classification system using a Vision Transformer.
2. Use transfer learning with a pretrained ViT model.
3. Perform image preprocessing and augmentation.
4. Train and evaluate the model on real and AI-generated facial images.
5. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
6. Apply Test-Time Augmentation during inference.
7. Analyze prediction confidence and identify potentially uncertain samples.
8. Compare standard ViT performance with ViT + TTA.

---

## 🧠 Why Vision Transformer?

Traditional image classification systems often use CNNs to extract local visual features.

Vision Transformers use a different approach.

The input image is divided into fixed-size patches. These patches are converted into embeddings and processed using Transformer encoder layers containing:

- Multi-Head Self-Attention
- Feed-Forward Neural Networks
- Layer Normalization
- Residual Connections
- Positional Information

This allows the model to learn relationships between different regions of an image.

For this project, the pretrained:

`google/vit-base-patch16-224`

model was used and fine-tuned for binary classification.

---

## 🔬 Proposed Enhancement

The main additional component of this project is a:

### Test-Time Augmentation + Confidence-Aware Inference Pipeline

Instead of making a prediction from only the original image, four versions of each test image are evaluated:

1. Original image
2. Horizontally flipped image
3. Brightness-adjusted image
4. Contrast-adjusted image

The prediction probabilities from all views are averaged.

### Inference Pipeline

```text
                 Input Image
                      |
          +-----------+-----------+
          |           |           |
       Original     Flip      Brightness
          |           |           |
          +-----------+-----------+
                      |
                Contrast View
                      |
                      ↓
              Vision Transformer
                      |
                      ↓
             Prediction Probabilities
                      |
                      ↓
             Probability Averaging
                      |
                      ↓
             Final REAL / FAKE
                      |
                      ↓
              Confidence Analysis
