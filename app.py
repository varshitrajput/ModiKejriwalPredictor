from flask import render_template, request,Flask,url_for ,redirect,flash
import pickle
import numpy as numpy
from fastai.tabular import *
from fastai.vision import *
import os
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

path = os.getcwd()
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = path
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '_5#y2L"F4Q8z\n\xec]/'
model = load_learner(path,'export.pkl')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
	return render_template('index.html')


@app.route('/predict',methods = ['POST','GET'])
def predict():
	 if request.method == 'POST':
	 	file = request.files['image']
	 	if file and allowed_file(file.filename):
	 		filename = secure_filename(file.filename)
	 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	 		img = open_image(filename)
	 		pred_class = model.predict(img)
	 		return render_template('index.html',pred='Prediction is {}'.format(pred_class))
	 	return render_template('index.html',pred='Prediction is {}'.format(pred_class))
	
	
if __name__ == '__main__':
	app.run() 