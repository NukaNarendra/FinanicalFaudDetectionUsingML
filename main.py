import datetime
import random
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pandas as pd
import joblib

app = Flask(__name__)


# Load the trained model and preprocessors
class ModelLoader:
    def __init__(self):
        self.model = load_model('fraud_detection.h5')
        self.encoders = {
            'category': joblib.load('category_encoder.pkl'),
            'gender': joblib.load('gender_encoder.pkl'),
            'job': joblib.load('job_encoder.pkl'),
            'trans_num': joblib.load('trans_num_encoder.pkl'),
        }

    def encode_feature(self, feature_name, value):
        return self.encoders[feature_name].transform([value])[0]

    def predict(self, data):
        return self.model.predict(data)


model_loader = ModelLoader()


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Generate random values for certain fields
            transaction_data = {
                'unname': random.randint(1000000000000000, 9999999999999999),
                'trans_date_trans_time': datetime.datetime.now(),
                'cc_num': random.randint(698585855, 8525629555262625),
                'zip_code': random.randint(20000, 80000),

                # Form inputs
                'category': model_loader.encode_feature('category', request.form['category']),
                'amt': float(request.form['amount']),
                'gender': model_loader.encode_feature('gender', request.form['gender']),
                'lat': float(request.form['user_lat']),
                'long': float(request.form['user_long']),
                'job': model_loader.encode_feature('job', request.form['job']),
                'trans_num': model_loader.encode_feature('trans_num', request.form['transaction_number']),
                'merch_lat': float(request.form['merch_lat']),
                'merch_long': float(request.form['merch_long']),
                'city_pop': random.randint(48, 85895),
            }

            # Create DataFrame
            data = pd.DataFrame([list(transaction_data.values())],
                                columns=list(transaction_data.keys()))
            data['trans_date_trans_time'] = data['trans_date_trans_time'].astype(np.int64)

            # Make prediction
            prediction = model_loader.predict(data)

            return render_template('result.html',
                                   prediction=float(prediction[0]),
                                   transaction_amount=transaction_data['amt'],
                                   merchant_name=request.form['merchant_name'])

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True, port=5800)