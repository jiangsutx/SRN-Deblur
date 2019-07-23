import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback

from app_utils import blur
from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin

import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
from scipy.misc import imsave

import tensorflow as tf
import models.model as model

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)

class Args:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

args = Args(
        phase='test', 
        model='color',
        batch_size=1,
        gpu=-1,
        height=720,
        width=1280,
        input_path='testing_set', 
        output_path='testing_res',
        datalist='./datalist_gopro.txt',
        epoch=4000,
        learning_rate=1e-4
        )


@app.route("/process", methods=["POST"])
def process():

    input_path = generate_random_filename(upload_directory,"jpg")
    output_path = generate_random_filename(upload_directory,"jpg")

    try:
        url = request.json["url"]

        download(url, input_path)

        deblur.test(args.height, args.width, input_path, output_path)
        
        callback = send_file(output_path, mimetype='image/jpeg')

        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory
    global checkpoint_dir
    global deblur
    global train_dir
    global graph
    global sess

    upload_directory = '/src/upload/'
    create_directory(upload_directory)

    checkpoint_dir = "/src/checkpoints/"

    create_directory(checkpoint_dir)

    url_prefix = 'http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra5.cloud.ovh.net/image/SRN-Deblur/'

    model_zip = "srndeblur_models.zip"

    get_model_bin(url_prefix + model_zip , checkpoint_dir + model_zip)

    os.system("cd " + checkpoint_dir + " && unzip " + model_zip)

    checkpoint_dir = os.path.join(checkpoint_dir, args.model)


    deblur = model.DEBLUR(args)     


    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)
