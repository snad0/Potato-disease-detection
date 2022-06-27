from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
# import uvicorn   
import numpy as np
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator

  

app = Flask(__name__)
  

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('index.html')
    


MODEL = tf.keras.models.load_model("models/3")





CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

        
        

def predict_image(img_path):
    test_image = image.load_img(img_path, target_size=(256,256) )
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image,axis=0)

    predictions = MODEL.predict(test_image)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return predicted_class, confidence

    



   
 
   
@app.route('/', methods=['GET', 'POST'])
def getvalue():
    if request.method == 'POST':
       img = request.files['imgfile']
       img_path = "static/images/" + img.filename
       img.save(img_path)
       output, confidence = predict_image(img_path)
       print(output)

    return render_template('result.html', o=output, confi=confidence,fimage=img.filename)  


# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)