from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utility import *

app = Flask(__name__)
app.config.from_pyfile("config.py")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/index", methods=["POST"])
def upload_file():
    uploaded_file = request.files["media"]
    filename = secure_filename(uploaded_file.filename)
    if filename == "":
        print("filename Missing")
        return redirect(url_for("index"))
    _, ext = os.path.splitext(filename)
    if ext in app.config["IMG_EXT"]:
        store_as_byte(uploaded_file, app.config["IMG_BYTE_DIR"], filename)
        return redirect(url_for("df_img", filename=filename))
    elif ext in app.config["VID_EXT"]:
        store_as_byte(uploaded_file, app.config["VID_BYTE_DIR"], filename)
        return redirect(url_for("df_vid", filename=filename))


@app.route("/df_img/<string:filename>")
def df_img(filename: str):
    if filename:
        meso = Meso4()
        meso.load(app.config["MESO"]["DF"])
        prediction = detectFakeImg(
            loc_dir=app.config["IMG_BYTE_DIR"],
            filename=filename,
            yolo_model=app.config["YOLO_MODEL"],
            caffe_model=app.config["CAFFE_MODEL"],
            model=meso,
        )
        remove_temp_file(loc_dir=app.config["IMG_BYTE_DIR"])
        print(prediction)
        return render_template("result.html", outcome=prediction)
    else:
        return redirect(url_for("index"))


@ app.route("/df_vid/<string:filename>")
def df_vid(filename: str):
    if filename:
        video_file = video_from_byte(
            loc_dir=app.config["VID_BYTE_DIR"], filename=filename)
        prediction = detectFakeVideo(
            video=video_file, model_path=app.config["VIDEO_DETECT"]["87A_20F"]
        )
        remove_temp_file(loc_dir=app.config["VID_BYTE_DIR"])
        print(prediction)
        return render_template("result.html", outcome=prediction)
    else:
        return redirect(url_for("index",))


if __name__ == "__main__":
    app.run(debug=True)
