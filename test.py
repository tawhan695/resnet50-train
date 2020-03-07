# import keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import ResNet50
# from keras.applications.resnet50 import preprocess_input
# from keras import Model, layers
# from keras.models import load_model, model_from_json
# from keras.preprocessing import image #*
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2

# def Predict():
#     # load
#     model = load_model('data/models/keras/model.h5')
#     # load
#     model.load_weights('data/models/keras/weights.h5')

#     img_path = 'data/validation/glass/glass582.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     preds = model.predict(x)
#     print(preds)
# # cap = cv2.VideoCapture(0)
# # cap.set(3, 224)
# # cap.set(4, 224)
# # __,img = cap.read()
# # cv2.imshow('show',img)
# Predict()
# # cv2.waitKey(0)


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model, model_from_json
import numpy as np
import cv2
from PIL import Image as PImage
model = load_model('data/models/keras/model.h5')
# load
model.load_weights('data/models/keras/weights.h5')
#

cap  = cv2.VideoCapture(0)
ret, frame = cap.read()
dim = (224, 224)
img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# img = image.load_img(img, target_size = (224, 224))
#X = PImage.fromarray(img) 
#x = pil2tensor(pil_im ,np.float32)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])

