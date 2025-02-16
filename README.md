# Fraud Detection in Financial Transactions Using Machine Learning

##  Project Overview
This project leverages **machine learning and anomaly detection** to detect fraudulent financial transactions in real-time. It combines **supervised learning models**, **unsupervised anomaly detection**, and a **Dash-based visualization dashboard**.

##  Objectives
1. **Data Preprocessing & Feature Engineering**
   - Load transaction data, clean missing values, and encode categorical features.
   - Scale numerical values to optimize model performance.

2. **Supervised Fraud Detection**
   - Train and evaluate **Random Forest** and **Gradient Boosting** classifiers.
   - Compare models using **accuracy, precision, recall, and F1-score**.

3. **Anomaly Detection**
   - Use **Local Outlier Factor (LOF)**, **One-Class SVM**, and **Isolation Forest** to detect unusual transaction patterns.

4. **Real-Time Fraud Detection Dashboard**
   - Build a **Flask + Dash** web app to visualize fraudulent transactions.
   - Display fraud trends and transaction anomalies using **Plotly charts**.

##  Project Structure

### **1. `fraud.ipynb` (Main Code)**
- Loads transaction data from `fraud.csv`.
- Preprocesses and encodes features.
- Trains **supervised classifiers** (Random Forest, Gradient Boosting).
- Implements **unsupervised anomaly detection** models.
- Hosts a **real-time dashboard** using Flask & Dash.

### **2. `fraud.csv`**
- Sample dataset containing financial transactions and fraud labels.

##  Setup & Installation
### **Prerequisites**
- Python 3.8+
- Required Python Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib dash plotly flask
  ```

##  Running the Project
### **1. Run the Fraud Detection System**
```bash
jupyter notebook fraud.ipynb
```
- Loads data and trains **fraud detection models**.
- Starts the **real-time dashboard** at `http://127.0.0.1:8080/`.

### **2. View the Dashboard**
- Open a browser and visit `http://127.0.0.1:8080/`.
- Analyze fraud trends and real-time transaction monitoring.

##  Expected Deliverables
✔ Fraud detection model using **supervised and unsupervised learning**.  
✔ **Real-time fraud monitoring** with an interactive dashboard.  
✔ **Anomaly detection module** to flag suspicious transactions.  
✔ **Performance reports** on fraud classification accuracy.

##  Future Enhancements
- **Enhance real-time processing** using Apache Kafka.
- **Deploy as a cloud-based API** for financial institutions.
- **Improve model retraining** using adaptive learning techniques.

---
###  Contributors
- Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

