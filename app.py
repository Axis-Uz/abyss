from flask import Flask, render_template, jsonify, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import os
import cv2
import io
import numpy as np
import imghdr
from utility import *
# TODO Learn about python environment
# TODO Add app variable for model paths
# TODO Create function for each detection technique
# TODO Create routes to these functions


app = Flask(__name__)
app.config.from_pyfile("config.py")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def upload_file():
    uploaded_file = request.files['img-file']
    filename = secure_filename(uploaded_file.filename)
    store_as_byte(uploaded_file, app.config["BYTE_DIR"], filename)
    return redirect(url_for("df_img", filename=filename))


@app.route("/df_img/<string:filename>")
def df_img(filename: str):
    results = {}
    if filename:
        meso = Meso4()
        meso.load(app.config["MESO"]["DF"])
        fake = detectFakeImg(
            loc_dir=app.config["BYTE_DIR"],
            filename=filename,
            yolo_model=app.config["YOLO_MODEL"],
            caffe_model=app.config["CAFFE_MODEL"],
            model=meso)
    return [fake]


@app.route("/df_vid/<string:filename>")
def df_vid(filename: str):
    pass


if __name__ == "__main__":
    app.run(debug=True)
