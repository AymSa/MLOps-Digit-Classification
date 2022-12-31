from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db
import torch 
import cv2
from PIL import Image
from utils import process_data
import mlflow 
import numpy as np 

bp = Blueprint('process', __name__)


@bp.route('/send', methods=["GET", "POST"])
@login_required
def send_img():
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

import re
import base64

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('instance/output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

@bp.route('/', methods=["GET", "POST"])
#@login_required
def write_img():
    if request.method == "POST":
        parseImage(request.get_data())

        x = Image.open('instance/output.png')
        x = x.resize((28,28))
        x = np.invert(x) 

        cv2.imwrite('instance/processed_output.jpg', x)
       
        processed_img = torch.from_numpy(x[:,:,0]).unsqueeze(0).to(dtype=torch.float32)

        # cv2.imshow('img', processed_img.squeeze(0).numpy())
        # cv2.waitKey(-1)
        # cv2.destroyWindow('img')

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        model = mlflow.pytorch.load_model(model_uri=f"models:/{'pytorch-mnist-simple-nn'}/{'Staging'}")

        preds = model(processed_img)
        label = preds.argmax(dim=1)
        
        return str(int(label))
        
    return render_template('process/write_img.html')