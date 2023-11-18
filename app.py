from flask import Flask, request, render_template
from librosa import load
from librosa.feature import mfcc
from tensorflow import keras
import numpy as np
import math

app = Flask(__name__)
CNN = keras.models.load_model("models/CNN.h5")
app.config["UPLOAD_FOLDER"] = "static/styles/"


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.filename = "sound.mp3"
        uploaded_file.save(app.config['UPLOAD_FOLDER'] + "sound.mp3")

    rate = 22050
    duration = 30
    samples_per_track = rate * duration
    data = {
        "mfcc": [],
    }
    
    def save_mfcc(file_path, n_mfcc = 13, n_fft = 2048, hop_len = 512, num_seg = 10):
        num_samples_per_segment = int(samples_per_track / num_seg)
        num_mfcc_per_segment = math.ceil(num_samples_per_segment / hop_len) #we need to round this for duration variability

        signal, sr = load(file_path, sr = rate)
        for s in range(num_seg):
            start_sample = num_samples_per_segment*s
            finish_sample = start_sample + num_samples_per_segment

            mfccV = mfcc(y = signal[start_sample:finish_sample],
                                        sr = sr,
                                        n_fft = n_fft,
                                        n_mfcc = n_mfcc,
                                        hop_length = hop_len)
            mfccV = mfccV.T

            if len(mfccV) == num_mfcc_per_segment:
                data["mfcc"].append(mfccV.tolist())
        
    
    save_mfcc("static/styles/sound.mp3")
    X = np.array(data["mfcc"])
    X = X[..., np.newaxis]

    def infer(X):
        X_d = X[np.newaxis,...]
        CNN_prediction = CNN.predict(X_d)
        return CNN_prediction[0]*100
    
    p = [0,0,0,0,0,0,0,0,0,0]

    for i in range(10):
        pred = infer(X[i])
        p = np.add(p,pred)
    
    p = p / 10

    return render_template('index.html', metal = '{}%'.format(round(p[0])), classical = '{}%'.format(round(p[1])), disco = '{}%'.format(round(p[2])), hiphop = '{}%'.format(round(p[3])), country = '{}%'.format(round(p[4])), blues = '{}%'.format(round(p[5])), reggae = '{}%'.format(round(p[6])), pop = '{}%'.format(round(p[7])), jazz = '{}%'.format(round(p[8])), rock = '{}%'.format(round(p[9])))

    
