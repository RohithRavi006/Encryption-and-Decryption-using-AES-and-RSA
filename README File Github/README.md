# Smart Meter Anomaly Detection

This project focuses on **anomaly detection** in smart meters' readings using machine learning models across three platforms: **MATLAB**, **Marimo**, and **Weka**.  
The data source is a **CSV file** containing smart meter readings.

## Platforms Used
- **MATLAB**
- **Marimo (Python-based notebook)**
- **Weka**

## Machine Learning Models Used
| Platform | Models |
|:---------|:-------|
| MATLAB   | Random Forest (RF), Naive Bayes (NB), Decision Tree |
| Marimo   | Logistic Regression, Naive Bayes (NB), Decision Tree |
| Weka     | Random Forest (RF), Naive Bayes (NB), Decision Tree |

## Project Workflow
1. **Load** the CSV dataset.
2. **Split** the dataset:  
   - **80%** for training
   - **20%** for testing
3. **Apply classification models** on the training data.
4. **Evaluate** the models' performance on the test data.

---

## Setting Up Marimo Environment

To run the Python/Marimo part of the project, follow these steps:

### 1. Install Miniforge and Mamba
- Install **Miniforge** from [Miniforge GitHub](https://github.com/conda-forge/miniforge).
- Inside the **Miniforge Prompt**, install **Mamba** (if not already installed):
  ```bash
  conda install mamba -n base -c conda-forge
  ```

### 2. Create and Activate a New Environment Using Mamba
```bash
mamba create -n anomaly-env python=3.12
mamba activate anomaly-env
```

### 3. Install Marimo and Required Libraries
```bash
mamba install marimo pandas numpy matplotlib seaborn scikit-learn
```

**Required Libraries:**
- `pandas` (for handling CSV files)
- `numpy` (for numerical operations)
- `matplotlib` (for plotting)
- `seaborn` (for advanced visualizations)
- `scikit-learn` (for implementing ML models)

> **Note:** Marimo is a reactive Python notebook framework, great for lightweight data science experiments.

### 4. Run Marimo
```bash
marimo run smart_model_ui.marimo.py
```

---

## Setting Up MATLAB

- Use MATLAB R2021a or higher for best compatibility.
- Import the dataset using the **Import Data** tool or programmatically using `readtable()`.
- Implement ML models using the **Classification Learner App** or programmatically using built-in functions like `fitctree`, `fitcensemble`, `fitcnb`, etc.

---

## Setting Up Weka

- Download and install Weka from [Weka official site](https://www.cs.waikato.ac.nz/ml/weka/).
- Load the CSV dataset in Weka.
- Use the **Explorer** to apply classification algorithms like Random Forest, Naive Bayes, and Decision Tree.

---

## Dataset

- Input: CSV file containing smart meter readings.
- Target: Detect and classify anomalies based on the readings.

---

## Author
- Project by Yashwanth B and Rohith Ravi.
