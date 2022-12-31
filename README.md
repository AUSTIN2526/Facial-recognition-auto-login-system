﻿# Facial-recognition-auto-sign-in-system
This program can recognize faces and create attendance list to help you reduce human resources

## Index

├── data   
│   └── you own picture  
├── model  
│   ├── facenet_keras.h5  
│   └── haarcascade_frontalface_alt2.xml  
├── main.py  
├── attend_list  
│   └── you own csv   
├── requirements  
└── README.md  


## Install
#### 1.Download the following Pre-train model and put into "model" folder

 ```
https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
https://github.com/mitre/biqt-face/blob/master/config/haarcascades/haarcascade_frontalface_alt2.xml
 ```
 
#### 2.Install the python requirements
 ```
pip install -r requirements.txt
 ```
 
 ## How to use
 name the face photo as the employee ID and put into "data" folder , then open main.py to run system. 
 
 example
  ```
 ex:employee_1.png
  ```
 

