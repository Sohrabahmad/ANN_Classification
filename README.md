# ANN Classification — Date Fruit Classification Using PyTorch

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** to classify date fruits into seven varieties based on their physical characteristics, shape measurements, and image-derived features.

The model is implemented using **PyTorch** and trained on the Date Fruit Dataset. The workflow includes data exploration, label encoding, feature standardization, train-test splitting, neural network construction, model training, and accuracy evaluation.

The objective is to learn patterns in the dataset that distinguish different date fruit varieties.

## 🎯 Project Objectives

- Explore and analyze the Date Fruit Dataset.
- Separate input features from the target class.
- Encode categorical target labels into numerical values.
- Split the dataset into training and testing sets.
- Standardize numerical features using `StandardScaler`.
- Build a feedforward neural network using PyTorch.
- Train the network using backpropagation and the Adam optimizer.
- Evaluate classification performance using test accuracy.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Data preprocessing and train-test splitting
- **PyTorch** — Neural network implementation and training
- **Matplotlib** — Optional visualization
- **Jupyter Notebook** — Interactive development environment

## 📂 Project Structure

```text
ANN-Classification/
│
├── ANN_Classification.ipynb
├── DateFruit_Dataset.csv
└── README.md
```

## 📊 Dataset Description

The project uses `DateFruit_Dataset.csv`, which contains measurements and image-derived features of date fruits.

| Dataset property | Value |
|---|---:|
| Total samples | 898 |
| Total columns | 35 |
| Input features | 34 |
| Target column | `Class` |
| Number of classes | 7 |
| Missing values | None detected |

### Date Fruit Classes

The dataset contains the following seven date fruit varieties:

| Class | Number of samples |
|---|---:|
| DOKOL | 204 |
| SAFAVI | 199 |
| ROTANA | 166 |
| DEGLET | 98 |
| SOGAY | 94 |
| IRAQI | 72 |
| BERHI | 65 |
| **Total** | **898** |

### Feature Categories

The 34 input features describe several aspects of each fruit.

- **Shape and size:** `AREA`, `PERIMETER`, `MAJOR_AXIS`, `MINOR_AXIS`, `EQDIASQ`
- **Shape characteristics:** `ECCENTRICITY`, `SOLIDITY`, `CONVEX_AREA`, `EXTENT`, `ASPECT_RATIO`, `ROUNDNESS`, `COMPACTNESS`
- **Shape factors:** `SHAPEFACTOR_1`, `SHAPEFACTOR_2`, `SHAPEFACTOR_3`, `SHAPEFACTOR_4`
- **Color statistics:** Mean and standard deviation features for red, green, and blue channels
- **Distribution features:** Skewness and kurtosis measurements for color channels
- **Texture and transform features:** Entropy and `ALLdaub4` features for RGB channels

The target column, `Class`, contains the date fruit variety to predict.

## ⚙️ Project Workflow

### 1. Load the Dataset

The notebook uses Pandas to load the CSV file and inspect its contents.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("DateFruit_Dataset.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())
```

### 2. Separate Features and Target

The `Class` column is separated from the input features.

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

- `X` contains 34 numerical input features.
- `y` contains the seven categorical date fruit classes.

### 3. Encode Target Labels

The notebook uses `LabelEncoder` to convert the categorical class names into integer labels.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
```

This allows PyTorch's `CrossEntropyLoss` to process the target labels.

### 4. Split the Dataset

The notebook uses an 80:20 train-test split with `random_state=42`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates approximately:

- **Training samples:** 718
- **Testing samples:** 180

The notebook does not specify stratification in the split.

### 5. Standardize the Features

The notebook standardizes the training features and applies the fitted scaler to the test features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization helps ensure that features with larger numerical ranges do not dominate the learning process.

### 6. Convert Data into PyTorch Tensors

The features are converted to `float32` tensors, while the encoded class labels are converted to `long` tensors.

```python
import torch

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)
```

### 7. Create DataLoaders

The notebook uses PyTorch's `TensorDataset` and `DataLoader` to prepare batches of 32 samples.

```python
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)
```

### 8. Build the Artificial Neural Network

The ANN is a fully connected feedforward neural network with two hidden layers.

```python
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.model(x)
```

**Model architecture:**

| Layer | Configuration |
|---|---|
| Input layer | 34 features |
| Hidden layer 1 | 64 neurons, ReLU |
| Hidden layer 2 | 64 neurons, ReLU |
| Output layer | 7 neurons, one per class |

The output layer generates seven class scores. The predicted class is the index with the highest score.

### 9. Configure the Loss Function and Optimizer

The notebook uses `CrossEntropyLoss` for multi-class classification and the Adam optimizer to update model parameters.

```python
model = ANN()

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

### 10. Train the Model

The notebook trains the neural network for 100 epochs.

During each epoch, it:

1. Enables training mode.
2. Processes batches of input data.
3. Computes predictions.
4. Calculates classification loss.
5. Performs backpropagation.
6. Updates model parameters using Adam.
7. Prints the epoch's average training loss.

```python
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criteria(outputs, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {train_loss:.4f}"
    )
```

### 11. Evaluate the Model

After training, the notebook evaluates the model on the test dataset and counts the correct predictions.

```python
model.eval()

total = 0
correct = 0

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total * 100

print("Total values:", total)
print("Correct values:", correct)
print("Accuracy score:", accuracy)
```

The test accuracy is calculated as:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Test Samples}}
\times 100
\]

## 📈 Results

The notebook evaluates the trained ANN using test-set classification accuracy.

| Evaluation detail | Value |
|---|---|
| Total samples | 898 |
| Training samples | 718 |
| Testing samples | 180 |
| Input features | 34 |
| Output classes | 7 |
| Hidden layers | 2 |
| Neurons per hidden layer | 64 |
| Training epochs | 100 |
| Batch size | 32 |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Evaluation metric | Accuracy |

**Note:** The notebook contains code to calculate the final accuracy, but no recorded output value is available in the notebook source. Run the notebook to obtain the actual accuracy. No specific performance score is assumed here.

## 🔍 Key Concepts

| Concept | Description |
|---|---|
| Artificial Neural Network | A model composed of interconnected layers that learns patterns from data. |
| ReLU | An activation function that introduces non-linearity. |
| Label Encoding | Converts categorical target classes into integer labels. |
| StandardScaler | Standardizes numerical features. |
| CrossEntropyLoss | A loss function commonly used for multi-class classification. |
| Backpropagation | Computes gradients for updating neural network parameters. |
| Adam Optimizer | An adaptive optimization algorithm used to train the network. |
| DataLoader | Organizes data into batches for model training and evaluation. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ANN-Classification
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy pandas scikit-learn torch jupyter
```

### 3. Verify the Dataset

Ensure `DateFruit_Dataset.csv` is located in the same directory as `ANN_Classification.ipynb`.

### 4. Run the Notebook

```bash
jupyter notebook ANN_Classification.ipynb
```

Execute the cells in order to load the dataset, preprocess the features, train the neural network, and evaluate its accuracy.

## 🚀 Potential Applications

This classification approach can be explored for:

- Automated date fruit variety classification
- Agricultural product sorting
- Food quality analysis
- Agricultural research
- Image-derived feature classification
- Decision-support tools for fruit processing

These are potential applications of the technique; the notebook demonstrates a classification model rather than a deployed sorting system.

## 🔮 Future Improvements

- Add a stratified train-test split to preserve class proportions.
- Report precision, recall, F1-score, and a confusion matrix.
- Plot the training loss across epochs.
- Add validation data and early stopping.
- Experiment with hidden-layer sizes, learning rates, and batch sizes.
- Compare the ANN with Random Forest, SVM, and other classification algorithms.
- Save the trained model and fitted scaler for future predictions.
- Set random seeds for reproducible training.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute the project under a specific open-source license.
