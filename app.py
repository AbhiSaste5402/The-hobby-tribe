from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('C:\\Users\\Abhishek\Desktop\\a\\audio_classification_model.h5')

# Define the extract_features function
def extract_features(file_path):
    y, sr = librosa.load(file_path)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return mfccs, chroma, zero_crossing_rate, spectral_centroid

# Define the preprocess_audio_for_prediction function
def preprocess_audio_for_prediction(file_path):
    mfccs, chroma, zero_crossing_rate, spectral_centroid = extract_features(file_path)

    # Concatenate the extracted features
    prediction_feature = np.concatenate((mfccs.mean(axis=1), chroma.mean(axis=1),
                                         zero_crossing_rate.mean(axis=1), spectral_centroid.mean(axis=1)))

    # Reshape the input features to match the model's input shape
    prediction_feature = prediction_feature.reshape(1, -1)  # Reshape to (1, number_of_features)
    
    predicted_probabilities = model.predict(prediction_feature)
    predicted_class = int(predicted_probabilities > 0.5)  # Assuming threshold of 0.5

    # Map the predicted class back to the label
    predicted_label = 'The sound you entered is G-Muted' if predicted_class == 0 else 'The sound you entered is G-Correct'
    
    return predicted_label, predicted_probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            audio_path = os.path.join('uploads', file.filename)
            file.save(audio_path)
            predicted_label, predicted_probabilities = preprocess_audio_for_prediction(audio_path)
            return render_template('index.html', predicted_label=predicted_label, predicted_probabilities=predicted_probabilities)
    return render_template('index.html', predicted_label='', predicted_probabilities=None)

if __name__ == '__main__':
    app.run(debug=True)
