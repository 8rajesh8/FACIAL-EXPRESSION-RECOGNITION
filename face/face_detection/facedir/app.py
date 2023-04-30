import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,flash 
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime as dt
from pandas import DataFrame as df
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
#from pygame import mixer
import random
import os
import matplotlib.pyplot as plt

UPLOAD_FOLDER ='static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png','.jpeg'}
app = Flask(__name__, static_url_path="/static")


# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            process_file(os.path.join(UPLOAD_FOLDER, filename), filename)
            data={
                "processed_img":'static/downloads/'+filename,
                "uploaded_img":'static/uploads/'+filename
            }
            return render_template("index.html",data=data)  
    return render_template('index.html')


def process_file(path, filename):
    detect_object(path, filename)
    

def detect_object(path, filename):    
    image = cv2.imread(path)
    
    face_classifier = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    classifier = tf.keras.models.load_model("./models/model_v6_576.hdf5")

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    #age part
    #ageList=['(0-2)', '(4-6)', '(8-12)', '(13-24)', '(25-32)', '(38-43)', '(44-50)', '(60-100)']
    ageList=['(0-2)', '(4-6)', '(8-12)', '(13-24)', '(25-32)', '(38-43)', '(44-50)', '(60-100)']
    ageProto="./models/deploy_age.prototxt"
    ageModel="./models/age_net.caffemodel"
    ageNet=cv2.dnn.readNet(ageModel,ageProto)


    #Gender part
    genderList=['Female','Male']
    genderProto="./models/deploy_gender.prototxt"
    genderModel="./models/gender_net.caffemodel"
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    i=1
    j=0

    pat=''
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
            #cv2.imshow('emotion', image)
            print('empty')

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0, 102, 102),1)
        roi_gray = gray[y:y+h, x:x+w]
        #reshpaing the gray image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        #converting face to np array
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        #preds = classifier.predict(roi)[0]
        #lable = class_labels[preds.argmax()]
        
        preds = classifier.predict(roi)
        
        max_index=np.argmax(preds[0])
        #print(max_index)
        lable=emotions[max_index]
        
        
        #age part
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        blob=cv2.dnn.blobFromImage(image, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        #print(age)
        
        #Gender
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

    #        print(f'Gender: {gender}')
        
        emotion = ""
        
        
        overlay = image.copy()
        opacity = 0.3
        cv2.rectangle(image,(1,y-25),(90,y+81),(64,64,64),cv2.FILLED)
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)#helps to blend image
        for i in range(len(preds[0])):
            emotion = "%s %s%s" % (emotions[i], round(preds[0][i]*100, 2), '%')
            
            if emotions[i] == lable:
                color = (66, 245, 105)
                cv2.putText(image, emotion, (0, int(y-12+i*15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                color = (245, 239, 66)
                cv2.putText(image, emotion, (0, int(y-12+i*15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
        
        #print(f'Age: {age[1:-1]} years')
        

        
        
        text="{}  {} :  {}y".format(lable.title(),gender, age)
        cv2.putText(image, text, (20,200) , cv2.FONT_HERSHEY_TRIPLEX,.4, (0, 255, 85), 1)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{DOWNLOAD_FOLDER}{filename}",image)
  
# download 
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run()