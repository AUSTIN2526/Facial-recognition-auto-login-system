import cv2
import numpy as np
import datetime
import pandas as pd
import os
from keras.models import load_model
from scipy.spatial import distance

def preProcess(x):
    if x.ndim == 4:
        axis = (1, 2, 3)  
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        pass
    
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    
    whitenImg = (x - mean) / std_adj
    whitenImg = whitenImg[np.newaxis, :]
    
    return whitenImg

def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h <= longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w <= longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        print('the shape is Error') 
    
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))

def normalize(predict_img):
    x = np.concatenate(predict_img)
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-10))
    return output

def face_predict(model, image, embs):
    predict_img = model.predict(image)
    emb = normalize(predict_img)
    name_dict = {name[i]:distance.euclidean(emb, embs[i]) for i in range(len(embs))}
    face_ID = min(name_dict, key=name_dict.get)
    score = name_dict[face_ID]
    return face_ID, score

def init_system(path,parameter):
    try:
        model = load_model('model/facenet_keras.h5', compile = False)
        classfier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
        name,embs = [], []
        for ID in os.listdir('data'):
            name.append(ID[:len(ID)-4:])
            
            img = cv2.imread('data/' + ID)
            img = resize_image(img,160,160)

                
            data = preProcess(img)
            predict_img = model.predict(data)
            embs.append(normalize(predict_img))
        
        
        try:
            df = pd.read_csv(excel_path, index_col="ID")
        except:
            df = pd.DataFrame([[i,'未簽到'] for i in name],columns=['ID', '簽到日期'])
            df.to_csv(excel_path,encoding='utf_8_sig', index=False)
            df = pd.read_csv(excel_path, index_col="ID")
        print('Success')
    except:
        print('Error')

    return df, model, classfier, name, embs, parameter
        
excel_path = 'attend_list/' + datetime.datetime.now().strftime('%Y-%m-%d') + '.csv'   
df, model, classfier, name, embs, parameter  = init_system(path = excel_path, parameter = 0.7)  
cap = cv2.VideoCapture(0)
while(not cap.isOpened()):
    cap = cv2.VideoCapture(0)


while (not cv2.waitKey(1) == ord('q')):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    
    if len(faceRects) > 0:                                    
        for (x, y, w, h) in faceRects:            
            try:
                image = resize_image(frame[y - 10: y + h + 10, x - 10: x + w + 10], 160, 160)
                image = preProcess(image)
                face_ID ,score = face_predict(model, image, embs)
                
                if score <= parameter and df.loc[face_ID]['簽到日期']!='未簽到':
                    text =' Welcome! ' + face_ID
                
                elif score <= parameter:
                    text = face_ID
                    df.loc[face_ID]['簽到日期'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    df.to_csv(excel_path,encoding='utf_8_sig')
                    
                else:
                    text = 'Unknow'
                
            except:
                text  = 'Error'
                

            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
            cv2.putText(frame, text,(x - 30, y - 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
       
            
    cv2.imshow('faceNET', frame)

cap.release()
cv2.destroyAllWindows()


