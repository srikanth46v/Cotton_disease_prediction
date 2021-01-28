from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('cotton_disease_prediction.h5')
import numpy as np
import os
img=image.load_img('dis_leaf (7)_iaip.jpg',target_size=(224,224))
x=image.img_to_array(img)
x=x/255
import numpy as np
x=np.expand_dims(x,axis=0)
result=model.predict(x)
a=np.argmax(model.predict(x), axis=1)
a