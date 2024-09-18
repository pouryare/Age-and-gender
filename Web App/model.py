import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
import cv2
import os

base = '/models'
print("\n\n")
print(tf.__version__)
print("\n\n")
model_gender = tf.keras.models.load_model(f'{base}/model_gender.keras')
model_age = tf.keras.models.load_model(f'{base}/model_age.keras')

def image_pre(path):
    print(f"Processing image: {path}")
    data = np.ndarray(shape=(1, 128, 128, 1), dtype=np.float32)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255-img # creating the negative of the image
    img = cv2.resize(img, (128, 128))
    data = np.array(img).reshape((-1, 128, 128, 1))/255
    return data

def predict(data):
    pred_age = np.round(model_age.predict(data)[0][0])
    pred_gen = np.round(model_gender.predict(data)[0])
    return pred_age, pred_gen
