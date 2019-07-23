import os
import requests
import random
import _thread as thread
from uuid import uuid4

import numpy as np
import skimage
from skimage.filters import gaussian


def blur(image, x0, x1, y0, y1, sigma=1, multichannel=True):
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    im = image.copy()
    sub_im = im[y0:y1,x0:x1].copy()
    blur_sub_im = gaussian(sub_im, sigma=sigma, multichannel=multichannel)
    blur_sub_im = np.round(255 * blur_sub_im)
    im[y0:y1,x0:x1] = blur_sub_im
    return im



def download(url, filename):
    data = requests.get(url).content
    with open(filename, 'wb') as handler:
        handler.write(data)

    return filename


def generate_random_filename(upload_directory, extension):
    filename = str(uuid4())
    filename = os.path.join(upload_directory, filename + "." + extension)
    return filename


def clean_me(filename):
    if os.path.exists(filename):
        os.remove(filename)


def clean_all(files):
    for me in files:
        clean_me(me)


def create_directory(path):
    os.system("mkdir -p %s" % os.path.dirname(path))


def get_model_bin(url, output_path):
    if not os.path.exists(output_path):
        create_directory(output_path)
        cmd = "wget -O %s %s" % (output_path, url)
        os.system(cmd)

    return output_path


#model_list = [(url, output_path), (url, output_path)]
def get_multi_model_bin(model_list):
    for m in model_list:
        thread.start_new_thread(get_model_bin, m)

