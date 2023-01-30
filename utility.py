import os
import io
import cv2
import json
import random
import secrets
from string import ascii_lowercase, digits
import numpy as np
import imghdr
import torch
import face_recognition
from torchvision import transforms, models
from PIL import Image
import keras
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    LeakyReLU,
)
from keras.optimizers import adam_v2


# Basic Utility Functions
def store_as_byte(uploaded_file, save_dir: str, filename: str):
    byte_file = io.BytesIO()
    uploaded_file.save(byte_file)
    name, _ = os.path.splitext(filename)
    with open(save_dir + name + ".bin", "wb") as f:
        f.write(byte_file.getvalue())


def video_from_byte(loc_dir: str, filename: str):
    name, ext = os.path.splitext(filename)
    file_path = loc_dir + name
    if validate_path(file_path + ".bin"):
        with open(file_path + ".bin", "rb") as f:
            byte_data = f.read()
        with open(file_path + ext, "wb") as f:
            f.write(byte_data)
        return cv2.VideoCapture(file_path + ext)
    else:
        return "ERROR"


def img_from_byte(loc_dir: str, filename: str):
    name, _ = os.path.splitext(filename)
    file_path = loc_dir + name + ".bin"
    byte_file = io.BytesIO()
    if validate_path(file_path):
        with open(file_path, "rb") as f:
            byte_file = f.read()
            data = np.frombuffer(byte_file, dtype=np.uint8)
        return cv2.imdecode(data, 1)
    else:
        return "ERROR"


def generate_name():
    return "".join(secrets.choice(ascii_lowercase + digits) for _ in range(9))


def validate_all_paths(path_dict: dict):
    for key, path in path_dict.items():
        if os.path.exists(path):
            continue
        else:
            raise FileExistsError(f"{key} Not Exist")
    return True


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    is_img = imghdr.what(None, header)
    if not is_img:
        return None
    return "." + (is_img if is_img != "jpeg" else "jpg")


def validate_path(path: str):
    if os.path.exists(path):
        return True
    else:
        False


# Image Detection Realted


def CaffeExtract(img_file: np.ndarray, caffe_model: dict):
    if img_file.any() != None:
        height, width = img_file.shape[:2]
        frames = []
        shapes = []
        caffe = cv2.dnn.readNetFromCaffe(
            caffe_model["PROTO"], caffe_model["WEIGHTS"])
        blob = cv2.dnn.blobFromImage(
            image=img_file, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
        )
        caffe.setInput(blob)
        modelOutput = caffe.forward()
        for i in range(0, modelOutput.shape[2]):
            box = modelOutput[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            confidence = modelOutput[0, 0, i, 2]
            if confidence > 0.5:
                cropped = img_file[startY:endY, startX:endX]
                shapes.append(cropped.shape)
                frames.append(cropped)
        if len(frames) > 0:
            shapes_count = {
                s: shapes.count(s) for s in shapes if s[0] == 0 or s[1] == 0
            }
            if len(shapes_count) == 0:
                return frames
            else:
                return [img_file]  # Already Cropped
        else:
            return []  # No Face Detected
    else:
        return []


def PersonDetector(img_file: np.ndarray, yolo_model: dict):
    if not validate_all_paths(yolo_model):
        return []
    classes = []
    with open(yolo_model["COCO"], "r") as f:
        classes = f.read().splitlines()
    yolo = cv2.dnn.readNet(
        yolo_model["YOLO3_WEIGHTS"], yolo_model["YOLO3_CONFIG"])
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    blob = cv2.dnn.blobFromImage(
        image=img_file,
        scalefactor=1.0 / 255,
        size=(416, 416),
        mean=0,
        swapRB=True,
        crop=False,
    )
    yolo.setInput(blob)
    ln = yolo.getUnconnectedOutLayersNames()
    modelOutput = yolo.forward(ln)
    class_ids = []
    for output in modelOutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                class_ids.append(class_id)
        classes_dict = {classes[i] for i in class_ids}
        if "person" in classes_dict:
            return True, list(classes_dict)
        else:
            return False, list(classes_dict)


class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = adam_v2.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics={"accuracy"}
        )

    def init_model(self):
        x = Input(shape=(256, 256, 3))

        x1 = Conv2D(8, (3, 3), padding="same", activation="relu")(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding="same")(x1)

        x2 = Conv2D(8, (5, 5), padding="same", activation="relu")(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding="same")(x2)

        x3 = Conv2D(16, (5, 5), padding="same", activation="relu")(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding="same")(x3)

        x4 = Conv2D(16, (5, 5), padding="same", activation="relu")(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding="same")(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation="sigmoid")(y)

        return keras.models.Model(inputs=x, outputs=y)


def detectFakeImg(
    loc_dir: str, filename: str, yolo_model: dict, caffe_model: dict, model: Meso4
):
    img = img_from_byte(loc_dir, filename)
    model_results = {}
    (
        model_results["Person Detected"],
        model_results["Detected Objects"],
    ) = PersonDetector(img, yolo_model)
    if not model_results["Person Detected"]:
        return model_results
    frames = CaffeExtract(img, caffe_model)
    model_results["Number of Frames"] = len(frames)
    if frames == None or len(frames) == 0:
        return model_results
    else:
        model_results["Predictions"] = []
        outcomes = {"real": 0, "fake": 0}
        for index, frame in enumerate(frames):
            model_results["Predictions"].append({})
            frameArr = np.asanyarray(frame)
            reshapedFrame = np.array(
                Image.fromarray(np.uint8(frameArr)).resize((256, 256))
            )
            reshapedFrame = reshapedFrame.reshape((1, 256, 256, 3))
            pred = model.predict(reshapedFrame)[0][0]
            if pred >= 0.7:
                outcomes["real"] += 1
            else:
                outcomes["fake"] += 1

            model_results["Predictions"][index]["Outcome"] = (
                "real" if pred >= 0.7 else "fake"
            )
            model_results["Predictions"][index]["Confidence"] = int(pred * 100)

    model_results["Total Outcome"] = outcomes
    model_results["Confidence"] = {
        "Fake": True if outcomes["fake"] == max(outcomes.values()) else False,
        "Real": True if outcomes["real"] == max(outcomes.values()) else False,
    }
    return model_results


# Video Detection Related
VD_CONSTANTS = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_SIZE": 112,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
}

SM = torch.nn.Softmax(dim=0)
IMG_NORMALIZER = transforms.Normalize(
    mean=-1 * np.divide(VD_CONSTANTS["MEAN"], VD_CONSTANTS["STD"]),
    std=np.divide([1, 1, 1], VD_CONSTANTS["STD"]),
)


VID_TRANSFORMER = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (VD_CONSTANTS["IMG_SIZE"], VD_CONSTANTS["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(VD_CONSTANTS["MEAN"], VD_CONSTANTS["STD"]),
    ]
)


def TensorToImg(tensor: torch.Tensor):
    """Converts a Tensor to Numpy Array"""
    image = tensor.to(device=VD_CONSTANTS["DEVICE"]).clone().detach()
    image = image.squeeze()
    image = IMG_NORMALIZER(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


class Model(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
    ):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(
            latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = torch.nn.LeakyReLU()
        self.dp = torch.nn.Dropout(0.4)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def FrameExtract(vidObj: cv2.VideoCapture):
    hasFrame = 1
    while hasFrame:
        hasFrame, image = vidObj.read()
        if hasFrame:
            yield image


def ProcessVideo(video: cv2.VideoCapture, sequence_length: int = 30, transform=None):
    frames = []
    # first_frame = np.random.randint(0, int(100/sequence_length))
    extracted_frames = FrameExtract(video)
    for index, frame in enumerate(extracted_frames):
        faces = face_recognition.face_locations(frame)
        try:
            top, right, bottom, left = faces[0]
            # startY, endX, endY, startX = faces[0]
            frame = frame[top:bottom, left:right, :]
            # TODO Should I replace with Caffe Extract
        except:
            pass
        frames.append(transform(frame))
        if len(frames) == sequence_length:
            break
    frames_torch = torch.stack(frames)
    frames_torch = frames_torch[:sequence_length]
    return frames_torch.unsqueeze(0)


def makePredictions(model: Model, frames):
    _, logits = model(frames.to(device=VD_CONSTANTS["DEVICE"]))
    # weight_softmax = model.linear1.weight.detach().cpu().numpy()
    # model.linear1.weight.detach().cpu().numpy()
    logits_sm = SM(logits)
    _, prediction = torch.max(logits_sm, 1)
    confidence = logits_sm[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence


def loadModel(model_path: str):
    model = Model(2).to(device=VD_CONSTANTS["DEVICE"])
    model.load_state_dict(torch.load(
        f=model_path, map_location=VD_CONSTANTS["DEVICE"]))
    return model


def detectFakeVideo(video: cv2.VideoCapture, model_path: str):
    model_results = {}
    model = loadModel(model_path)
    frames = ProcessVideo(video, 30, VID_TRANSFORMER)
    model_results["Frames with Faces"] = len(frames)
    if len(frames) > 0:
        output = makePredictions(model, frames)
        model_results["Outcome"] = {
            "real": True if output[0] == 1 else False,
            "fake": True if output[0] == 0 else False,
            "confidence": int(output[1]),
        }
    return model_results
