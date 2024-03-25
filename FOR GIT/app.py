from flask import Flask, render_template, request
import os
import pickle
import joblib
import torch
from torchvision import transforms
from PIL import Image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask import Flask, jsonify, request, url_for
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)
CORS(app)
dic = { 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'ঁ', 11: 'ং', 12: 'ঃ', 13: 'অ', 14: 'আ', 15: 'ই', 16: 'ঈ', 17: 'উ',
 18: 'ঊ', 19: 'ঋ', 20: 'এ', 21: 'ঐ', 22: 'ও', 23: 'ঔ', 24: 'ক', 25: 'খ', 26: 'গ', 27: 'ঘ', 28: 'ঙ',29: 'চ', 30: 'ছ', 31: 'জ', 32: 'ঝ', 33: 'ঞ',
 34: 'ট', 35: 'ঠ', 36: 'ড', 37: 'ঢ', 38: 'ণ', 39: 'ত', 40: 'থ', 41: 'দ', 42: 'ধ', 43: 'ন', 44: 'প', 45: 'ফ', 46: 'ব', 47: 'ভ', 48: 'ম', 49: 'য', 50: 'র',
 51: 'ল', 52: 'শ', 53: 'ষ', 54: 'স', 55: 'হ', 56: 'ৎ'
}
pkl_path = 'complete_model.pkl'
with open(pkl_path, 'rb') as f:
    model = pickle.load(f)
model.eval() 
def predict_label(img_path):
    image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])   
    image = transformation(image).float()
    image = image.unsqueeze(0) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return dic[predicted.item()]
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")
@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		p = predict_label(img_path)
	return render_template("index.html", prediction = p, img_path = img_path)
@app.route('/upload_wav', methods=['POST'])
def upload_wav():
    if 'wav_file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    wav_file = request.files['wav_file']
    temp_filename = 'temp_voice.wav'
    wav_file.save(temp_filename)
    try:
        y, sr = librosa.load(temp_filename, sr=None)
        os.remove(temp_filename) 
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        spectrogram_image_filename = 'spectrogram.png'
        spectrogram_image_path = os.path.join('static', spectrogram_image_filename)
        plt.savefig(spectrogram_image_path)
        plt.close()
        prediction_label = predict_label('static/' + spectrogram_image_filename)
        return jsonify({
            'prediction': prediction_label,
            'image_path': url_for('static', filename=spectrogram_image_filename)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/submit_voice', methods=['POST'])
def submit_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    filename = 'temp_audio.wav'
    audio_file.save(filename)
    
    # Process audio file to spectrogram
    y, sr = librosa.load(filename, sr=None)
    os.remove(filename)  # Clean up the temporary file
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    spectrogram_path = 'static/spectrogram.png'
    plt.savefig(spectrogram_path)
    plt.close()
    
    # Predict the class of the spectrogram
    prediction = predict_label(spectrogram_path)
    
    return jsonify({
        'prediction': prediction,
        'image_path': spectrogram_path
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)