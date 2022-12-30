from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db

import cv2
from PIL import Image
from utils import process_data
import mlflow 

bp = Blueprint('process', __name__)


@bp.route('/', methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        img = request.files["img"]

        error = None

        if not img:
            error = "Digit image is required."

        if error is None:
            pil_img = Image.open(img.stream)
            processed_img = process_data(pil_img)
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            model = mlflow.pytorch.load_model(model_uri=f"models:/{'pytorch-mnist-simple-nn'}/{'Staging'}")
            preds = model(processed_img)
            label = int(preds.argmax(dim=1))
            
        flash(f'The digit predicted is : {label}.')

    return render_template('process/send_img.html')
