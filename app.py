
from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app=Flask(__name__)
model=load_model("model.h5")
classes=["metal","paper","plastic"]

@app.route("/",methods=["GET","POST"])
def home():
    pred=None
    if request.method=="POST":
        img=request.files["image"]
        img=Image.open(img).resize((224,224))
        img=np.array(img)/255.0
        img=img.reshape(1,224,224,3)
        p=model.predict(img)
        pred=classes[np.argmax(p)]
    return render_template("index.html",prediction=pred)

if __name__=="__main__":
    app.run(debug=True)
