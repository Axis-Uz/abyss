from flask import Flask, render_template, jsonify, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from utility import *


app = Flask(__name__)
app.config.from_pyfile("config.py")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    uploaded_file = request.files["img-file"]
    filename = secure_filename(uploaded_file.filename)
    store_as_byte(uploaded_file, app.config["BYTE_DIR"], filename)
    _, ext = os.path.splitext(filename)
    if ext in app.config["IMG_EXT"]:
        return redirect(url_for("df_img", filename=filename))
    elif ext in app.config["VID_EXT"]:
        return redirect(url_for("df_vid", filename=filename))
    return redirect(url_for("index"))


@app.route("/df_img/<string:filename>")
def df_img(filename: str):
    if filename:
        meso = Meso4()
        meso.load(app.config["MESO"]["DF"])
        fake = detectFakeImg(
            loc_dir=app.config["BYTE_DIR"],
            filename=filename,
            yolo_model=app.config["YOLO_MODEL"],
            caffe_model=app.config["CAFFE_MODEL"],
            model=meso,
        )
    return [fake]


@app.route("/df_vid/<string:filename>")
def df_vid(filename: str):
    if filename:
        video_file = video_from_byte(loc_dir=app.config["BYTE_DIR"], filename=filename)
        fake = detectFakeVideo(
            video=video_file, model_path=app.config["VIDEO_DETECT"]["87A_20F"]
        )

    return [fake]


if __name__ == "__main__":
    app.run(debug=True)
