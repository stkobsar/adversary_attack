import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras_preprocessing import image
from keras import backend as K

iv3 = InceptionV3() #Descargar el modelo ya entrenado
#print(iv3.summary())

absolute_path_every_machine = os.path.abspath(__file__)
adv_dir = os.path.dirname(os.path.dirname(absolute_path_every_machine))
adv_dir_data = os.path.join(adv_dir, "data")
path_image_beer = os.path.join(adv_dir_data, "cerveza.jpeg")
path_image_cat = os.path.join(adv_dir_data, "gato.jpeg")

image_cat = image.load_img(path_image_cat)
image_beer = image.load_img(path_image_beer)


