import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

# Load the tokenizer from the saved file
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 193

application = Flask(__name__) #Initialize the flask App
# Load your trained model architecture
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.h5")

@application.route('/')
def home():
    return render_template('single-product.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = request.form['review']

    sequence = tokenizer.texts_to_sequences([review])  # Note the conversion to list for the sequence
    sequence = pad_sequences(sequence, maxlen=max_len, padding="pre", truncating="pre")
    prediction = loaded_model.predict(sequence)  # Predict using the loaded model
    # Assuming your model is binary classification and you want a human-readable result
    result = "Fake" if prediction[0][0] < 0.5 else "Real"
    return render_template('single-product.html', prediction=result, entered_review=review)

if __name__ == "__main__":
    application.run(debug=True)

