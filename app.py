from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import string
import os
from pickle import TRUE, dump, load
import tensorflow as tf
from keras.models import load_model
from keras.applications import xception
from keras.applications.xception import Xception 
from keras.applications.vgg16 import VGG16
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from flask_ngrok import run_with_ngrok

#from gevent.pywsgi import WSGIServer

# Define a flask app
UPLOAD_FOLDER = 'D:/Arkision/arkision/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model saved with Keras model.save()
MODEL_PATH = 'D:/Arkision/arkision/model_1.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def extract_features(filename):
	
	model = VGG16()	
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)	
	image = load_img(filename, target_size=(224, 224))	
	image = img_to_array(image)	
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):	
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)		
		yhat = model.predict([photo,sequence], verbose=0)		
		yhat = np.argmax(yhat)		
		word = word_for_id(yhat, tokenizer)	
		if word is None:
			break		
		in_text += ' ' + word		
		if word == 'endseq':
			break
	return in_text

tokenizer = load(open('D:/Arkision/arkision/tokenizer.pkl', 'rb'))
max_length = 34



port = 00

@app.route('/', methods=['GET', 'POST'])
def index():
	description1=''
	file=''
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		file_path= 'D:/Arkision/arkision/uploads'+"/"+filename
		photo = extract_features(file_path)
		description1 = generate_desc(model, tokenizer, photo, max_length)
		description1=description1.split(' ')[1:-1]
		description1 = ' '.join(description1)
	return render_template('index.html',msg=description1)



if __name__ == '__main__':
    app.run(debug=TRUE)