from django.conf import settings
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from keras.models import load_model
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = r'D:\path\to\folder\EDM MGC main\EDMMGC'

genre_names = ['ambient', 'big_room_house', 'dnb', 'dubstep',
               'future_garage_wave_trap', 'hardcore', 'hardstyle', 'house',
               'psytrance', 'techno', 'trance']

genre_colors = ['pink', 'orange', 'blue', 'red', 'purple', 'brown',
                'green', 'cyan', 'yellow', 'magenta', 'grey']

def load_models_scalers():
    """
    Loads a pre-trained machine learning model and its associated scaler.

    Returns:
        model: The pre-trained machine learning model loaded from a .h5 file.
        scaler: The scaler used for preprocessing data before prediction, loaded from a .pkl file.
    """
    model_path = os.path.join(BASE_DIR, 'myapp', 'static', 'models', 'model1_80acc.h5')
    scaler_path = os.path.join(BASE_DIR, 'myapp', 'static', 'models', 'scaler (1).pkl')

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler



def process_audio_segment(audio_segment, model, scaler):
    """
    Process a segment of audio data to extract features and make predictions.

    Parameters:
    - audio_segment (numpy.ndarray): A 1-dimensional array containing the audio data for the segment.
    - scaler (sklearn.preprocessing.StandardScaler): A scaler object used to scale the extracted features.

    Returns:
    - prediction (str or int): The predicted genre label for the audio segment.

    This function takes an audio segment represented as a 1-dimensional numpy array and a pre-fitted scaler object.
    It extracts various audio features from the segment, scales them using the provided scaler, and makes a prediction
    using a pre-trained machine learning model.

    The features extracted include:
    - Root Mean Square Energy (RMSE)
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Rolloff Point
    - Zero Crossing Rate
    - Mel-frequency cepstral coefficients (MFCCs)
    - Chroma feature
    - Tonnetz feature
    - Chroma Constant-Q Transform (CQT)
    - Spectral Contrast

    Note: It is assumed that the model variable used for prediction is globally defined.
          Ensure that the model has been trained and loaded before calling this function.
    """
    
    #Load models
    

    
    # Extract features
    rmse = librosa.feature.rms(y=audio_segment)
    spec_cent = librosa.feature.spectral_centroid(y=audio_segment)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_segment)
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment)
    zcr = librosa.feature.zero_crossing_rate(audio_segment)
    mfcc = librosa.feature.mfcc(y=audio_segment, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio_segment)
    tonnetz = librosa.feature.tonnetz(y=audio_segment, chroma=chroma)
    chroma_cq = librosa.feature.chroma_cqt(y=audio_segment)
    spec_contrast = librosa.feature.spectral_contrast(y=audio_segment)

    if any(map(lambda x: x.size == 0, [rmse, spec_cent, spec_bw, rolloff, zcr, mfcc, chroma, tonnetz, chroma_cq, spec_contrast])):
        return "Unknown"
    
    # Concatenate mean and standard deviation for each feature
    to_append = f'{np.mean(rmse)} {np.std(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.std(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)} {np.std(e)}'
    for e in chroma:
        to_append += f' {np.mean(e)} {np.std(e)}'
    for e in tonnetz:
        to_append += f' {np.mean(e)} {np.std(e)}'
    to_append += f' {np.mean(chroma_cq)} {np.std(chroma_cq)} {np.mean(spec_contrast)} {np.std(spec_contrast)}'
    
    # Split the to_append string and convert it to a list of strings
    to_append_list = to_append.split()

    # Convert the list of strings to a NumPy array
    features = np.array(to_append_list, dtype=float)

    # Reshape the array to have a shape of (1, n_features), where n_features is the total number of features
    features = features.reshape(1, -1)

    # Scale the features using the scaler
    scaled_features = scaler.transform(features)

    # Predict genre
    prediction = model.predict(scaled_features)[0]

    return prediction

def get_waveform(y, sr):
    try:
        plt.switch_backend('AGG')  # Set backend to AGG
        #print("Backend is successfully set to AGG")

        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        #print("Plot is created")

        segment_length = 3 * sr
        start = 0
        predictions = []
        model, scaler = load_models_scalers()
        index_list = set()
        #print("Model and scaler are loaded")

        while start + segment_length < len(y):
            audio_segment = y[start:start+segment_length]
            #print("Audio is segmented")
            prediction = process_audio_segment(audio_segment, model, scaler)
            #print("Prediction obtained")
            genre_index = np.argmax(prediction)
            index_list.add(genre_index)
            color = genre_colors[genre_index] if genre_index < len(genre_colors) else 'black'  # Default to black if genre not found

            plt.plot(np.arange(len(audio_segment)) / sr + start / sr, audio_segment, color=color)
            #print("Updated plot")
            start += segment_length

        legend_genres = [genre_names[idx] for idx in index_list]
        legend_colors = [genre_colors[idx] for idx in index_list]
        
        plt.legend(handles=[plt.Line2D([], [], color=color, marker='o', markersize=10, label=genre) for genre, color in zip(legend_genres, legend_colors)],
           title="Genres", title_fontsize='large', facecolor='white', edgecolor='black', framealpha=0.4)

        #print("Waveform plot is computed")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        plt.savefig(buffer, format='png')
        #print("Buffer is saved to png")
        buffer.seek(0)
        img_png = buffer.getvalue()

        wave = base64.b64encode(img_png)
        #print("png is encoded")
        wave = wave.decode('utf-8')
        #print("png is decoded")

        buffer.close()

        #print("Waveform generated")
        return wave

    except Exception as e:
        print("An error occurred:", str(e))
        return None



