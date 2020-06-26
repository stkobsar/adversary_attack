"""
Image has to be (299, 299, 3) to be interpreted correctly by InceptionV3
Los valores de la matriz resultante son los valores de la intensidad de cada uno de los pixels en cada uno de sus canales.
Los valores originales van de 0 a 255.
InceptionV3 trabaja con rangos de intensidad de -1 a 1, y por tanto hay que reescalar los valores.
Si se quiere añadir más de una imagen a la red neuronal, habrá que aumentar las dimensiones de la matriz.
"""

