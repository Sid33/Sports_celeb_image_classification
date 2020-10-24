from flask import Flask, request, jsonify
import predict
import cv2 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def pred():
    # print("hi")
    # image = request.files['image_data']
    # print(image)

    if 'image_data' not in request.files:
        image = request.form['image_data']
        print("recieved base64 image")
        # res = predict.predict_class(image)
    
    else :
        image = request.files['image_data']
        ext = image.filename.split('.')[1]
        print("recieved an image")
        image.save('./image_2_classify/img.'+ ext)
        image = cv2.imread('./image_2_classify/img.'+ ext)

    # print("got image ", image.filename)
    res = predict.predict_class(image)

    # res.headers.add('Access-Control-Allow-Origin', '*')

    
    return jsonify(res)



if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    app.run(port=5000)