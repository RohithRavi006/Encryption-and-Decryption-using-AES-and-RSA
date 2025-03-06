## Overview

This project integrates encryption and decryption mechanisms with AES and RSA algorithms alongside anomaly detection and classification models. It provides secure data handling and network traffic classification capabilities.

## Features ✨

- 🔐 AES Encryption & Decryption
- 🔑 RSA Key Pair Generation
- 🔒 Secure Storage of Encryption Keys
- 📁 Encrypted File Handling
- 🧐 Anomaly Detection using Machine Learning
- 📊 Multiclass Classification of Network Traffic
- 🐱‍💻 WEKA Integration for Data Analysis

## Project Structure 🏗️

```
MFCEOC - Copy/
│-- aes.py                  # Handles AES encryption & decryption
│-- aes_key.key             # AES encryption key file
│-- decrypted_file.csv      # Decrypted output data
│-- decryption.py           # Script for decryption operations
│-- encrypted_aes_key.bin   # Encrypted AES key storage
│-- encrypted_file.csv      # Encrypted data file
│-- rsa.py                  # Handles RSA encryption
│-- rsa_private.pem         # RSA private key file
│-- rsa_public.pem          # RSA public key file
│-- updated_dataset.csv     # Dataset file for testing
│-- anomaly_detection_3.py  # Implements anomaly detection models
│-- multiclass_classifcation_3.py # Performs multiclass classification
│-- oroject_4.py            # Additional classification models
│-- converted_dataset.arff  # WEKA-compatible dataset
```

## Dataset Description 📂

The dataset used in this project is related to **DDoS attack detection in Advanced Metering Infrastructure (AMI)**. It contains various attributes representing network flow characteristics and attack types.

### **Instances and Attributes**

- **Number of Instances:** The dataset contains multiple entries representing network traffic records.
- **Attributes:**
  - **Source & Destination Information:** `src_ip`, `dst_ip`, `src_port`, `dst_port`
  - **Protocol Data:** `protocol`
  - **Traffic Statistics:** `flow_duration`, `flow_byts_s`, `flow_pkts_s`, `tot_fwd_pkts`, `tot_bwd_pkts`, `fwd_pkt_len_max`, `bwd_pkt_len_max`, etc.
  - **Packet Timing Data:** `flow_iat_mean`, `fwd_iat_max`, `bwd_iat_std`, etc.
  - **Flag Indicators:** `syn_flag_cnt`, `ack_flag_cnt`, `fin_flag_cnt`, etc.
  - **Target Label:** `Label` (represents whether a flow is normal or an attack type)

### **Additional Features**

- The dataset is preprocessed by handling missing values and encoding categorical variables.
- **Feature Scaling**: StandardScaler is applied to normalize feature values.
- **SMOTE (Synthetic Minority Over-sampling Technique)** is used to handle class imbalance.

## Machine Learning Classifiers Used 🚀

This project employs the following classifiers for network traffic classification:

1. **Support Vector Machine (SVM) 📈**

   - A powerful classifier that works by finding the best hyperplane to separate classes in high-dimensional space.
   - Effective for both linear and non-linear classification problems.

2. **Random Forest 🌲**

   - An ensemble learning method that builds multiple decision trees and merges their outputs to improve accuracy.
   - Robust against overfitting and provides feature importance scores.

3. **Naive Bayes (NB) 🧠**

   - A probabilistic classifier based on Bayes' theorem, assuming independence between features.
   - Works well for text classification and problems with categorical data.

## WEKA Integration 🐱‍💻

WEKA is a powerful tool for data preprocessing, classification, clustering, and visualization. This project includes a **converted ARFF dataset** for easy use in WEKA.

### **Using the ARFF File in WEKA 🔍**

1. Open WEKA.
2. Click on **"Explorer"**.
3. Load the dataset **`converted_dataset.arff`**.
4. Use WEKA's classification, clustering, or visualization tools to analyze the data.
5. Experiment with different machine learning models like **J48 (Decision Tree) 🌳, Naive Bayes 🧠, SVM 📈, and Random Forest 🌲**.

## Requirements 📌

Ensure you have Python installed along with the required dependencies. You may need the following libraries:

```sh
pip install cryptography pandas scikit-learn imbalanced-learn matplotlib seaborn
```

## Usage ⚡

### Encrypting a File 🔐

1. Run `aes.py` to encrypt a CSV file.
2. The AES key is encrypted using RSA and stored in `encrypted_aes_key.bin`.
3. The encrypted file is saved as `encrypted_file.csv`.

### Decrypting a File 🔓

1. Run `decryption.py` to decrypt the encrypted file.
2. The decrypted file is stored as `decrypted_file.csv`.

### Running Anomaly Detection 📊

1. Execute `anomaly_detection_3.py` to run the anomaly detection pipeline.
2. The script will preprocess the dataset, analyze class distribution, train multiple models, and compare their performance.

### Running Multiclass Classification 🏷️

1. Execute `multiclass_classifcation_3.py` to perform classification on network traffic data.
2. The script will preprocess data, apply SMOTE, train classifiers, and evaluate performance metrics.

## Security Considerations 🔒

- Keep your RSA private key (`rsa_private.pem`) secure.
- Never share encryption keys in public repositories.

##

