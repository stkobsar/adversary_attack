"""
Image has to be (299, 299, 3) to be interpreted correctly by InceptionV3
Los valores de la matriz resultante son los valores de la intensidad de cada uno de los pixels en cada uno de sus canales.
Los valores originales van de 0 a 255.
InceptionV3 trabaja con rangos de intensidad de -1 a 1, y por tanto hay que reescalar los valores.
Si se quiere añadir más de una imagen a la red neuronal, habrá que aumentar las dimensiones de la matriz.
"""

import numpy as np
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras_preprocessing import image

iv3 = InceptionV3() #Descargar el modelo ya entrenado
#print(iv3.summary())



def image_to_array(image_path):
    image_into_array = image.img_to_array(image.load_img(image_path, target_size=(299, 299))) #imagen loded in tensor and reshaped
    return image_into_array

def normalize_matrix_image(matrix_image):
    arr = matrix_image - matrix_image.mean(axis=0)
    matrix_image_norm = arr / np.abs(arr).max(axis=0)
    return matrix_image_norm

def image_reshaped(image):
    """
    Description: reshape to 4 dimension is needed. First dimension (1) can save multiple images in the first dimenson of matrix
    :param image: image converted in array
    :return: image reshaped
    """
    image_reshaped = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
    return image_reshaped



def get_image(image_path):
    image_into_array = image_to_array(image_path)
    image_norm = normalize_matrix_image(image_into_array)
    image_norm_reshaped = image_reshaped(image_norm)
    return image_norm_reshaped