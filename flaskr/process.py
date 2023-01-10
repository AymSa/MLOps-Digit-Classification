import sys

sys.path.append(".")
sys.path.append("..")

from flask import Blueprint, flash, g, redirect, render_template, request, url_for

from flaskr.auth import login_required
from flaskr.db import get_db
import torch
import cv2
from PIL import Image
import numpy as np
from utils import get_yaml_params, set_mlflow
from model import load_model

from prefect import flow
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bp = Blueprint("predict", __name__)

import re
import base64


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    with open("instance/output.png", "wb") as output:
        output.write(base64.decodebytes(imgstr))


@bp.route("/", methods=["GET", "POST"])
@flow
def predict():
    if request.method == "POST":
        seed_everything(42)
        dict_yaml = get_yaml_params("instance/flask_config.yaml")
        set_mlflow(dict_yaml["mflow_tracking_uri"], dict_yaml["mlflow_experiment_name"])

        parseImage(request.get_data())

        x = Image.open("instance/output.png")
        x = x.resize((28, 28))
        x = np.invert(x)
        x = x[:, :, 0]

        cv2.imwrite("instance/processed_output.jpg", x)
        normalised_img = x / 255.0
        normalised_img[normalised_img > 0.35] = 1.0

        cv2.imwrite("instance/normalised_output.jpg", normalised_img * 255)

        normalised_img = (
            torch.from_numpy(normalised_img)
            .view(1, 1, normalised_img.shape[0], normalised_img.shape[1])
            .to(dtype=torch.float32, device=device)
        )

        model = load_model(dict_yaml["model_name"], dict_yaml["stage"]).to(device)

        preds = model(normalised_img)
        label = preds.argmax(dim=1)

        return str(int(label))

    return render_template("predict/hand_write.html")
