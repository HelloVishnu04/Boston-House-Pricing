import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, url_for, render_template
from flask_cors import CORS  # Import CORS

# Initialize App
app = Flask(__name__)

# Enable CORS (This allows your HTML page to talk to this API)
CORS(app)

# Load Model and Scaler
# Ensure these files are in the same folder as app.py
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# API Route (Use this one for your HTML/Tailwind Page)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # FIXED: Use request.json directly (not request.json['data']) 
    # because the HTML sends the data as a direct JSON object.
    data = request.json 
    
    print("Received Data:", data)
    
    # Transform dictionary values to list for the model
    # Note: This assumes the HTML sends keys in the exact order the model expects.
    input_data = np.array(list(data.values())).reshape(1, -1)
    
    # Scale and Predict
    new_data = scalar.transform(input_data)
    output = regmodel.predict(new_data)
    
    print("Prediction:", output[0])
    
    # Return just the prediction value or a full object
    return jsonify({'prediction': output[0]})

# Form Route (Use this if submitting a standard HTML form without JS)
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction="The predicted house price is Rs. {:.2f} lakhs".format(output))

if __name__ == "__main__":
    app.run(debug=True)