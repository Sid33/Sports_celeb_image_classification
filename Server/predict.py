import pywt
import joblib
import json
import numpy as np
import base64
import cv2
from matplotlib import pyplot as plt

d = {
    0 : 'lionel_messi',
    1 : 'maria_sharapova',
    2 : 'roger_federer',
    3 : 'serena_williams',
    4 : 'virat_kohli'
}

### Wavelet Transform of image
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

#### Converting base64 image string to image 
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_images(image):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    # img=cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces
def predict_class(img):
    res = []
    with open('saved_svm_model.pkl' , 'rb') as f:
        model = joblib.load(f)
    if str(type(img)) == "<class 'str'>":                     
        img = get_cv2_image_from_base64_string(img)    ### if recieved an base64 image, converting it to image
    crop_img = get_cropped_images(img)
    print("number of faces : ",len(crop_img))
    for imgg in crop_img:
        # img=cv2.imread(j)
        scalled_raw_img = cv2.resize(imgg, (32, 32))
        img_har = w2d(imgg,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)
        res.append(
            {
                'Name' : d[model.predict(final)[0]],
                'Confidence_Score' : model.predict_proba(final)[0][model.predict(final)[0]] * 100,
                'Confidence_Score_of_all' : np.around(model.predict_proba(final)*100,2).tolist()[0]
            }
        )
    return res

