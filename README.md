# Healthcare Insurance Fraud Detection

## Project Overview
This project focuses on detecting fraudulent healthcare insurance claims using machine learning and deep learning techniques. The main goal is to identify whether an insurance claim is **fraudulent or genuine** by analyzing claim details and patient information.

Healthcare insurance fraud causes major financial losses, and manual detection is difficult. This project helps automate fraud detection using data-driven models.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- XGBoost
- Matplotlib

---

## Models Used
This project combines three powerful techniques:

### 1. Autoencoder
An Autoencoder is used for **anomaly detection**.  
It learns patterns from normal (genuine) insurance claims and calculates a reconstruction error. Claims with high reconstruction error are considered suspicious.

### 2. K-Nearest Neighbors (KNN)
KNN is used to calculate a **graph-based similarity score**.  
It measures how close a claim is to other claims, helping identify unusual or isolated claims.

### 3. XGBoost
XGBoost is used as the **final classification model**.  
It takes original features along with the Autoencoder error and KNN graph score to predict whether a claim is **Fraud** or **Real**.

---

## Dataset
The dataset is loaded from an online healthcare insurance fraud claims dataset.  
It includes:
- Claim amount
- Patient age
- Patient income
- Claim details
- Claim legitimacy labels

The data is preprocessed using:
- Missing value handling
- One-hot encoding for categorical features
- Feature scaling

---

## How the System Works
1. Data is cleaned and preprocessed
2. Features are extracted and scaled
3. Autoencoder learns normal claim behavior
4. KNN calculates similarity-based graph scores
5. XGBoost combines all features to classify claims
6. The system predicts whether a claim is **Fraud** or **Real**

---

## Output
The model provides:
- Fraud or Real prediction
- Accuracy and AUC score for evaluation

Users can also input a new insurance claim as a Python dictionary to get real-time fraud prediction.

---

## Conclusion
This project demonstrates how combining anomaly detection and classification models improves fraud detection accuracy. The hybrid approach using Autoencoders, KNN, and XGBoost makes the system more reliable for real-world healthcare insurance fraud detection.
