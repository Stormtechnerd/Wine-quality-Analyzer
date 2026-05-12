
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['ph']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]

    arr = np.array(values).reshape(1, -1)
    prediction = model.predict(arr)[0]

    if prediction <= 4:
        quality = "Poor"
    elif prediction <= 6:
        quality = "Average"
    elif prediction <= 7:
        quality = "Good"
    else:
        quality = "Excellent"

    return render_template('index.html', prediction=quality)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
