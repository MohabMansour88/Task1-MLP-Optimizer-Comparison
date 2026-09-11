# 🧠 Neural Network Optimizer Comparison: Mitigating Overfitting in MLPs

## 📌 Project Overview
This project evaluates and compares the performance of five deep learning optimizers (SGD, SGD with Momentum, Adagrad, RMSProp, and Adam) on a multi-class image classification task. The primary objective was to identify the most effective optimizer while actively diagnosing and resolving model overfitting.

## 🏗️ Model Architecture
A custom Multi-Layer Perceptron (MLP) was built using TensorFlow/Keras to process image batches of shape `(128, 128, 3)`. To combat severe memorization of the training data, aggressive regularization was implemented:
* **Input:** Flattened 128x128 RGB images (49,152 features)
* **Hidden Layer 1:** Dense 512 neurons (ReLU) + **50% Dropout**
* **Hidden Layer 2:** Dense 256 neurons (ReLU) + **30% Dropout**
* **Output Layer:** Dense 3 neurons (Softmax) for one-hot encoded classification

## ⚙️ Training Strategy
* **Loss Function:** Categorical Crossentropy
* **Learning Rate:** Tuned to `0.0001` across all optimizers to stabilize validation metrics and smooth out learning curves.
* **Callbacks:** Implemented `EarlyStopping` (patience=3) to halt training automatically when validation accuracy degraded, restoring the best-performing weights.

## 📊 Final Model Accuracy Ranking
| Rank | Optimizer | Train Accuracy | Val Accuracy | Test Accuracy |
|------|-----------|----------------|--------------|---------------|
| 1    | SGD+Momentum | 73.61%       | 86.16%       | 88.24%        |
| 2    | Adagrad   | 77.26%         | 66.09%       | 85.47%        |
| 3    | SGD       | 64.39%         | 61.59%       | 71.97%        |
| 4    | RMSProp   | 38.88%         | 40.83%       | 44.98%        |
| 5    | Adam      | 33.27%         | 33.91%       | 33.91%        |

## 📈 Learning Curves & Analysis
* **SGD + Momentum** emerged as the winning optimizer, achieving an impressive **88.24% test accuracy** with a healthy generalization profile (where validation/test accuracy surpassed training accuracy due to heavy dropout).
* Adjusting the learning rate to `0.0001` alongside structural double-dropout successfully resolved the extreme overfitting seen in baseline models, though it caused adaptive learning rates like Adam and RMSProp to underfit at this scale.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have `tensorflow`, `pandas`, and `matplotlib` installed.
3. Open the Jupyter Notebook and select **Run All** to execute the data pipeline, build the models, and generate the evaluation graphs.
