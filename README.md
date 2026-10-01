## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Download `creditcard.csv` from Kaggle and place it in `data/`.
2. Run `notebooks/fraud_detection.ipynb` top to bottom to train and save the model into `models/`.
3. Launch the app:

```bash
streamlit run app.py
```

# 💳 Credit Card Fraud Detection Using Machine Learning

A Machine Learning project that detects **fraudulent credit card transactions** using classification and anomaly detection techniques. The project includes data preprocessing, exploratory data analysis, handling of imbalanced data, model training, evaluation, and a **Streamlit-based web interface** for real-time fraud prediction.

---

## 📌 Project Overview

Credit card fraud is a major challenge in the financial sector, where fraudulent transactions can cause significant financial losses.

This project aims to build an automated **Credit Card Fraud Detection System** that analyzes transaction features and predicts whether a transaction is:

- ✅ **Legitimate Transaction**
- 🚨 **Fraudulent Transaction**

The project demonstrates an end-to-end Machine Learning workflow:

```text
Credit Card Transaction
          ↓
   Data Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Handle Class Imbalance
          ↓
 Feature Engineering
          ↓
   Model Training
          ↓
   Model Evaluation
          ↓
  Trained ML Model
          ↓
    Streamlit App
          ↓
   Fraud / Legitimate
```

---

## 🎯 Objectives

- Understand the application of Machine Learning in fraud detection.
- Analyze and visualize credit card transaction data.
- Identify patterns associated with fraudulent transactions.
- Preprocess and prepare transaction data for Machine Learning.
- Handle the highly imbalanced nature of fraud datasets.
- Train classification models for fraud detection.
- Evaluate the model using appropriate classification metrics.
- Save the trained model for deployment.
- Build an interactive **Streamlit web application** for prediction.

---

## 🛠️ Technologies Used

| Technology              | Purpose                             |
| ----------------------- | ----------------------------------- |
| 🐍 **Python**           | Programming language                |
| 📓 **Jupyter Notebook** | Data analysis and model development |
| 📊 **Pandas**           | Data manipulation and analysis      |
| 🔢 **NumPy**            | Numerical operations                |
| 📈 **Matplotlib**       | Data visualization                  |
| 📊 **Seaborn**          | Statistical visualization           |
| 🤖 **Scikit-learn**     | Machine Learning                    |
| ⚖️ **SMOTE**            | Handling class imbalance            |
| 💾 **Joblib**           | Saving and loading trained models   |
| 🌐 **Streamlit**        | Web-based frontend                  |
| 🐙 **Git / GitHub**     | Version control                     |

---

## 🔄 Project Workflow

### 1. Data Collection

The project uses a credit card transaction dataset containing transaction-related features and a target variable indicating whether a transaction is fraudulent.

The target variable represents:

```text
0 → Legitimate
1 → Fraudulent
```

---

### 2. Exploratory Data Analysis

The dataset is analyzed to understand its structure, distributions, and relationships.

The analysis includes:

- Dataset dimensions
- Data types
- Missing values
- Duplicate records
- Statistical summary
- Class distribution
- Transaction amount distribution
- Transaction time analysis
- Fraud vs legitimate transaction comparison

Visualizations are created using **Matplotlib** and **Seaborn**.

---

### 3. Data Preprocessing

Before training the Machine Learning model, the data is prepared through preprocessing steps.

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Missing Value Check
     ↓
Duplicate Check
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
Processed Dataset
```

Numerical features are appropriately scaled before being passed to the Machine Learning model.

---

### 4. Handling Class Imbalance

Credit card fraud datasets are typically highly imbalanced because legitimate transactions greatly outnumber fraudulent transactions.

For example:

```text
Legitimate Transactions  ███████████████████████████████
Fraudulent Transactions  █
```

Training directly on such data can cause a model to favor the majority class.

To address this problem, **SMOTE (Synthetic Minority Over-sampling Technique)** can be used to generate synthetic samples for the minority class.

```text
Original Dataset
       ↓
Class Imbalance
       ↓
      SMOTE
       ↓
Balanced Training Data
       ↓
Model Training
```

SMOTE is applied only to the training data to avoid data leakage.

---

### 5. Model Training

The processed dataset is divided into training and testing sets.

The Machine Learning model learns patterns from the training data and predicts whether unseen transactions are fraudulent.

The classification process is:

```text
Transaction Features
        ↓
Preprocessing
        ↓
Machine Learning Model
        ↓
   ┌────┴────┐
   ↓         ↓
Legitimate  Fraud
    0         1
```

The trained model is then saved using **Joblib** so that it can be reused by the Streamlit application without retraining.

---

### 6. Model Evaluation

Because fraud detection involves a highly imbalanced dataset, **accuracy alone is not sufficient** for evaluating the model.

The project focuses on:

### Accuracy

Measures the percentage of total predictions that are correct.

### Precision

Measures how many transactions predicted as fraudulent were actually fraudulent.

High precision helps reduce false fraud alerts.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

For fraud detection, recall is particularly important because missing a fraudulent transaction can have significant consequences.

### F1 Score

The F1 score provides a balance between precision and recall.

```text
             Precision × Recall
F1 = 2 × ----------------------------
             Precision + Recall
```

### Confusion Matrix

The confusion matrix provides a detailed view of:

```text
                    Predicted
                 Legit     Fraud
              ┌─────────┬─────────┐
Actual Legit  │   TN    │   FP    │
              ├─────────┼─────────┤
Actual Fraud  │   FN    │   TP    │
              └─────────┴─────────┘
```

Where:

- **TN** → Correctly identified legitimate transactions
- **TP** → Correctly identified fraudulent transactions
- **FP** → Legitimate transactions incorrectly flagged as fraud
- **FN** → Fraudulent transactions incorrectly classified as legitimate

---

## 📊 Model Performance

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

> The final performance values depend on the dataset, preprocessing strategy, sampling technique, model, and hyperparameters used during training.

---

## 🌐 Streamlit Web Application

The trained Machine Learning model is integrated into a **Streamlit frontend**.

The Streamlit application provides an interactive interface through which transaction data can be submitted for fraud detection.

```text
                  Streamlit
                     │
                     ▼
             User Transaction
                     │
                     ▼
              Data Processing
                     │
                     ▼
              Saved ML Model
                     │
                     ▼
                Prediction
                     │
              ┌──────┴──────┐
              ▼             ▼
        🚨 FRAUD       ✅ LEGITIMATE
```

The frontend and Machine Learning development are separated:

```text
notebooks/
    fraud_detection.ipynb
            │
            │ Train
            ▼
     fraud_model.pkl
            │
            │ Load
            ▼
         app.py
            │
            ▼
       Streamlit UI
```

This allows the model to be trained once and subsequently reused by the web application.

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Description

| File / Folder      | Description                                      |
| ------------------ | ------------------------------------------------ |
| `data/`            | Dataset used for training                        |
| `notebooks/`       | EDA, preprocessing, training and evaluation      |
| `models/`          | Saved trained ML model and preprocessing objects |
| `app.py`           | Streamlit frontend                               |
| `requirements.txt` | Required Python libraries                        |
| `README.md`        | Project documentation                            |
| `.gitignore`       | Files excluded from Git                          |

> Large datasets and generated model files can be excluded from GitHub when appropriate using `.gitignore` or Git LFS.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Move into the project directory

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the Model

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then:

1. Load the dataset.
2. Perform data inspection.
3. Analyze the transaction data.
4. Handle missing values and duplicates.
5. Perform exploratory data analysis.
6. Preprocess the features.
7. Handle class imbalance using SMOTE.
8. Train the Machine Learning model.
9. Evaluate the model.
10. Save the trained model using Joblib.

The trained files are stored inside:

```text
models/
```

---

### Run the Streamlit Application

After the model has been trained and saved:

```bash
streamlit run app.py
```

The application will open in the browser and provide the fraud detection interface.

---

## 🧠 Machine Learning Pipeline

The complete Machine Learning pipeline can be represented as:

```text
                 ┌──────────────────────┐
                 │  Credit Card Dataset │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Data Exploration     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Feature Scaling      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ SMOTE Balancing      │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Model Training       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Saved ML Model       │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Streamlit Application│
                 └──────────┬───────────┘
                            ↓
                    FRAUD / LEGITIMATE
```

---

## 💡 Key Learnings

Through this project, I learned how to:

- Work with real-world financial transaction data.
- Perform exploratory data analysis.
- Identify and handle class imbalance.
- Understand the importance of precision and recall in fraud detection.
- Apply SMOTE to minority-class data.
- Build and evaluate a Machine Learning classification model.
- Use confusion matrices and classification reports.
- Save and reuse trained Machine Learning models.
- Separate model development from application deployment.
- Build an interactive frontend using Streamlit.
- Create an end-to-end Machine Learning application.
- Manage a Machine Learning project using Git and GitHub.

---

## 🔮 Future Scope

The project can be extended by:

- Comparing multiple Machine Learning algorithms.
- Applying advanced anomaly detection techniques.
- Implementing hyperparameter optimization.
- Using ensemble learning techniques.
- Adding probability-based fraud risk scores.
- Adding interactive transaction analytics to the Streamlit dashboard.
- Implementing model explainability using SHAP or similar techniques.
- Deploying the Streamlit application to a cloud platform.
- Connecting the system to a real-time transaction stream.
- Adding authentication and transaction history.
- Implementing continuous model retraining with new transaction data.

---

## ⚠️ Limitations

The predictions generated by this system depend on the quality and characteristics of the training dataset.

Important limitations include:

- Highly imbalanced transaction data.
- Fraud patterns can change over time.
- Synthetic oversampling does not represent actual fraudulent transactions.
- Model performance depends on the selected features and algorithm.
- False positives can result in legitimate transactions being flagged.
- False negatives can allow fraudulent transactions to pass undetected.
- A model trained on historical data may not generalize perfectly to new fraud patterns.

Therefore, predictions should be treated as **Machine Learning-based risk classifications rather than definitive financial decisions**.

---

## 📚 Concepts Covered

```text
Credit Card Transaction Data
          ↓
Exploratory Data Analysis
          ↓
Data Preprocessing
          ↓
Feature Scaling
          ↓
Class Imbalance
          ↓
SMOTE
          ↓
Supervised Machine Learning
          ↓
Binary Classification
          ↓
Precision / Recall / F1
          ↓
Confusion Matrix
          ↓
Model Serialization
          ↓
Streamlit Deployment
```

---

## 👨‍💻 Author

**Parthik Patel**

Computer Science & Engineering Student

Interested in:

**Machine Learning • Artificial Intelligence • Data Science • Python • Backend Development**

---

## ⭐ Project

If you find this project useful, feel free to explore the repository and learn from the implementation.

**Building with data, learning through models, and turning predictions into applications. 🚀**
