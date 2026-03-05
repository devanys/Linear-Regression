# 📉 Linear Regression

> A pure Python implementation of Linear Regression — built entirely without NumPy, PyTorch, or any ML library. Covers SGD, K-Fold Cross Validation, Early Stopping, and Learning Rate Schedulers.

---

## 📌 Overview

This project demonstrates the **mathematical foundations of machine learning** by implementing Linear Regression and its training pipeline from first principles. Every component — from gradient computation to cross-validation — is coded by hand using only Python's standard library.

---

## 📐 Mathematical Foundations

### 1. Linear Model

The model predicts output $\hat{y}$ from input $x$ using a single weight $w$ and bias $b$:

$$\hat{y} = w \cdot x + b$$

The goal is to find $w$ and $b$ such that $\hat{y}$ is as close to the true label $y$ as possible.

---

### 2. Loss Function — Mean Squared Error (MSE)

Given $N$ training samples, the loss is defined as:

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2$$

MSE penalizes large errors more heavily due to the squaring, making it sensitive to outliers but easy to differentiate.

---

### 3. Gradient Computation

To minimize the loss, we compute partial derivatives with respect to $w$ and $b$:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} 2(\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} 2(\hat{y}_i - y_i)$$

The factor of 2 is typically absorbed into the learning rate, simplifying to:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

---

### 4. Stochastic Gradient Descent (SGD)

Parameters are updated iteratively in the direction that reduces the loss:

$$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

$$b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

where $\eta$ (eta) is the **learning rate** — a hyperparameter controlling step size.

In **mini-batch SGD**, gradients are computed over a random subset (batch) of $m$ samples per step:

$$\frac{\partial \mathcal{L}}{\partial w} \approx \frac{1}{m} \sum_{i \in \text{batch}} (\hat{y}_i - y_i) \cdot x_i$$

---

### 5. Evaluation Metrics

#### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$

#### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \hat{y}_i - y_i \right|$$

---

## 🔁 K-Fold Cross Validation

The dataset is split into $K$ non-overlapping folds. In each iteration, one fold is used as the **validation set** and the remaining $K-1$ folds are used for **training**.

$$\text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\text{val}}^{(k)}$$

The **standard deviation** across folds measures stability:

$$\sigma_{\text{val}} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left( \mathcal{L}_{\text{val}}^{(k)} - \overline{\mathcal{L}}_{\text{val}} \right)^2}$$

#### Ensemble (Weight Averaging)

After training all $K$ folds, the final model is obtained by averaging their weights:

$$w_{\text{ensemble}} = \frac{1}{K} \sum_{k=1}^{K} w^{(k)}, \qquad b_{\text{ensemble}} = \frac{1}{K} \sum_{k=1}^{K} b^{(k)}$$

---

## 🛑 Early Stopping

Training is halted when the validation loss does not improve for `patience` consecutive epochs:

```
if val_loss < best_val_loss:
    best_val_loss = val_loss
    no_improve = 0
else:
    no_improve += 1
    if no_improve >= patience:
        stop training
```

The **generalization gap** is monitored to detect overfitting:

$$\text{Gap} = \left| \mathcal{L}_{\text{train}} - \mathcal{L}_{\text{val}} \right|$$

| Gap Range | Diagnosis |
|-----------|-----------|
| $< 0.05$ | Good fit ✅ |
| $0.05 - 0.20$ | Slight overfitting ⚠️ |
| $> 0.20$ | Significant overfitting ❌ |

---

## 📉 Learning Rate Schedulers

A fixed learning rate can be suboptimal — too large causes divergence, too small causes slow convergence. Schedulers decay $\eta$ over training.

### Step Decay

The learning rate drops by a factor every `every` epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / \text{every} \rfloor}$$

where $\gamma \in (0, 1)$ is the drop factor (e.g. $\gamma = 0.5$).

### Exponential Decay

The learning rate decays smoothly as an exponential function of epoch $t$:

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

where $\lambda$ controls the decay rate.

### Cosine Annealing

The learning rate follows a cosine curve, gradually decreasing to near zero:

$$\eta_t = \frac{\eta_0}{2} \left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

where $T$ is the total number of epochs.

---
---

## 🧪 Synthetic Dataset

The notebook generates a synthetic dataset following:

$$y = 2x + 1 + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\, 0.4^2)$$

- $N = 100$ samples
- $x \sim \mathcal{U}(-3, 3)$
- Target parameters: $w^* = 2.0$, $b^* = 1.0$

The model is expected to recover these values through training.

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 0.01 – 0.1 | Learning rate |
| `epochs` | 200 – 500 | Training epochs |
| `batch_size` | 1 | Mini-batch size (pure SGD) |
| `K` | 10 | Number of CV folds |
| `patience` | 20 | Early stopping patience |
| `noise` | 0.4 | Dataset noise std |

---

## 📊 Results Summary

| Technique | Final MSE / Val Loss |
|-----------|----------------------|
| Vanilla SGD | ~0.16 |
| K-Fold Best Fold | ~0.15 |
| K-Fold Ensemble | ~0.15 |
| Early Stopping | ~0.15 |
| LR Constant | ~0.16 |
| LR Step Decay | ~0.15 |
| LR Exp Decay | ~0.15 |
| LR Cosine Annealing | ~0.15 |

---

## 📚 Concepts Covered

- ✅ Linear model & forward pass
- ✅ MSE loss function
- ✅ Manual gradient computation (backprop by hand)
- ✅ Stochastic Gradient Descent (SGD)
- ✅ Mini-batch training
- ✅ K-Fold Cross Validation
- ✅ Model ensemble via weight averaging
- ✅ Early stopping with patience
- ✅ Learning rate scheduling (Step, Exp, Cosine)
- ✅ RMSE & MAE evaluation

---
