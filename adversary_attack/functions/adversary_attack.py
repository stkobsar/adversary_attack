"""
Datos de entrada para confundir a la red neuronal. Generar imagenes adversarias. El proceso se asemeja bastante al proceso
de aprendizaje de la red neuronal. Realizar un proceso de optimización sobre los pixeles de la imagen de entrada.
Pixels son los nuevos parámetros. Maximizar probabilidad para que aparezca una imagen.

Red neuronal es un grafo. Construimos un nuevo grafo para la generación de la imagen.

"""
import numpy as np
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K


iv3 = InceptionV3()

def adversary(image_pred):
    # Definir capa de entrada y capa de salida de los datos
    input_layer = iv3.layers[0].input
    output_layer = iv3.layers[-1].output

    #definir la imagen que queremos que sea

    target_class = 951 #limon

    loss = output_layer[0, target_class] #función de coste que hay que maximizar
    gradient = K.gradients(loss, input_layer)[0] #hace referencia al proceso de calcular el gradiente aumente la prob para ser un limon
    optimaze_gradient = K.function([input_layer, K.learning_phase()], [gradient, loss]) #K.learning_phase necesario para decirle al modelo que esta siendo entrenado

    image_adv_copy = np.copy(image_pred)

    cost = 0.0
    while cost < 0.95:
        gr, cost = optimaze_gradient([image_adv_copy, 0])
        image_adv_copy += gr

    return image_adv_copy
