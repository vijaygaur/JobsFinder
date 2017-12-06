import subprocess
import os
import sys
import model
from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['supports_credentials']='true'
app.config['CORS_SUPPORTS_CREDENTIALS']='true'
cors = CORS(app)

mdl = model.Model()

@app.route("/")
def greeting():
    return "Welcome to InsightLake Speech Model Service!!"

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def transcribe():
    if request.method == 'POST':
        docs=request.data
        return mdl.train(docs)
    return '[]'


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        doc=request.data
        return jsonify({'results': mdl.predict(doc)})
    return jsonify({'results': '[]'})


app.run(host= '0.0.0.0')
