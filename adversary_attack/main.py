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
path_image_beer = os.path.join(adv_dir, "data/cerveza.jpeg")
path_image_cat = os.path.join(adv_dir, "data/gato.jpeg")

image_cat = image.img_to_array(image.load_img(path_image_cat, target_size=(299, 299)))
image_beer = image.img_to_array(image.load_img(path_image_beer, target_size=(299, 299)))

"""
Image has to be (299, 299, 3) to be interpreted correctly by InceptionV3
Los valores de la matriz resultante son los valores de la intensidad de cada uno de los pixels en cada uno de sus canales.
Los valores originales van de 0 a 255. 
InceptionV3 trabaja con rangos de intensidad de -1 a 1, y por tanto hay que reescalar los valores. 
Si se quiere añadir más de una imagen a la red neuronal, habrá que aumentar las dimensiones de la matriz.
"""
def normalize_matrix_image(matrix_image):
    arr = matrix_image - matrix_image.mean(axis=0)
    matrix_image_norm = arr / np.abs(arr).max(axis=0)
    return matrix_image_norm

image_cat_norm = normalize_matrix_image(image_cat)
#reshape
image_cat_reshape = image_cat_norm.reshape([1, image_cat_norm.shape[0], image_cat_norm.shape[1], image_cat_norm.shape[2]])
print(image_cat_reshape)

#Hasta aqui todo es get image





