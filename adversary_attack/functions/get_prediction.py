from keras.applications.inception_v3 import InceptionV3, decode_predictions

iv3 = InceptionV3()

def get_prediction_numeric(image_processed):
    image_pred_matrix = iv3.predict(image_processed)
    return image_pred_matrix

def get_prediction_class(image_processed):

    image_pred_class = decode_predictions(image_processed)
    return image_pred_class