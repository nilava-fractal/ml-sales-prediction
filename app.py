from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
app = Flask(__name__)
# Load the model
with open('model.pkl', 'rb') as f:
 model = pickle.load(f)

@app.route('/')
def home():
 return jsonify({'message': 'Sales Prediction API is running(Test 2)'})

@app.route('/predict', methods=['POST'])
def predict():
 data = request.get_json()
 features = np.array(data['features']).reshape(1, -1)
 prediction = model.predict(features)
 return jsonify({'predicted_sales': prediction[0]})

if __name__ == '__main__':
 port = int(os.environ.get("PORT", 8080))
 app.run(host='0.0.0.0', port=port)
