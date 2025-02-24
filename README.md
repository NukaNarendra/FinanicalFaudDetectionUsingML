# Fraud Detection in Financial Transactions Using Machine Learning

## Project Overview
This project leverages machine learning and anomaly detection to detect fraudulent financial transactions in real-time. It combines supervised learning models, unsupervised anomaly detection, and a Flask-based web application that provides a user-friendly interface for fraud prediction.

## Objectives

### Data Preprocessing & Feature Engineering
- Load transaction data, clean missing values, and encode categorical features.
- Scale numerical values to optimize model performance.

### Supervised Fraud Detection
- Train and evaluate Random Forest and Gradient Boosting classifiers.
- Compare models using accuracy, precision, recall, and F1-score.

### Anomaly Detection
- Use Local Outlier Factor (LOF), One-Class SVM, and Isolation Forest to detect unusual transaction patterns.

### Real-Time Fraud Detection Web Application
- Build a Flask web app to visualize fraudulent transactions.
- Display fraud trends and transaction anomalies using Plotly charts.

## Notice
Due to memory constraints, the dataset is not provided.

## Project Structure

1. **fraud.ipynb (Main Code)**
   - Loads transaction data from `fraud.csv`.
   - Preprocesses and encodes features.
   - Trains supervised classifiers (Random Forest, Gradient Boosting).
   - Implements unsupervised anomaly detection models.

2. **main-2/main.py (Flask Web Application)**
   - Loads the trained fraud detection model.
   - Preprocesses input data and encodes categorical values.
   - Provides a user-friendly bank webpage for fraud prediction.

3. **fraud.csv**
   - Sample dataset containing financial transactions and fraud labels.

4. **Templates/**
   - `form.html`: Web form to enter transaction details.
   - `result.html`: Displays prediction results.
   - `error.html`: Shows errors in case of incorrect inputs.

## Setup & Installation

### Prerequisites
- Python 3.8+
- Required Python Libraries:
  ```sh
  pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib dash plotly flask
  ```

## Running the Project

### 1. Run the Fraud Detection System
```sh
jupyter notebook fraud.ipynb
```
- Loads data and trains fraud detection models.
- Saves the trained model (`fraud_detection.h5`).

### 2. Running the Web Application
- After training the model, navigate to the `main-2` folder.
- Run the Flask web application:
  ```sh
  cd main-2
  python main.py
  ```
- The application will start on `http://127.0.0.1:5800/`.

### 3. Using the Webpage
- Open a browser and visit `http://127.0.0.1:5800/`.
- A user-friendly bank webpage will appear where transaction details can be entered.
- The system will preprocess the input data and use the trained ANN model to predict whether the transaction is fraudulent.
- The prediction result, including the fraud probability, will be displayed.
- Interactive charts and transaction trends are also available for better visualization.

## Expected Deliverables
✔ Fraud detection model using supervised and unsupervised learning.  
✔ Real-time fraud monitoring with an interactive web application.  
✔ Anomaly detection module to flag suspicious transactions.  
✔ Performance reports on fraud classification accuracy.  

## Future Enhancements
- Enhance real-time processing using Apache Kafka.
- Deploy as a cloud-based API for financial institutions.
- Improve model retraining using adaptive learning techniques.

## Contributors
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

