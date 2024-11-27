from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging


config = Cfg.load_config_from_name('vgg_seq2seq')
config["weights"] = "seq1seq.pth"
config['cnn']['pretrained'] = False
config['device'] = 'cpu'

recognitor = Predictor(config)
detector = YOLO("vcard_detection.pt")

def select_box(class_names, boxes, confs):
    attributes = {}

    for i, class_name in enumerate(class_names):
        if class_name in ['cccd', 'bhyt', 'gplx']:
            if 'type' not in attributes:
                attributes['type'] = (class_name, i)
            else:
                if confs[i] > confs[attributes['type'][1]]:
                    attributes['type'] = (class_name, i)
        else:
            if class_name not in attributes:
                attributes[class_name] = (boxes[i], i)
            else:
                if confs[i] > confs[attributes[class_name][1]]:
                    attributes[class_name] = (boxes[i], i)

    return attributes

def crop_img(img, attributes, padding):
    
    imgs = {}

    for class_name, value in attributes.items():
        if class_name == 'type':
            imgs['type'] = value[0]
        else:
            box = value[0]
        
            box[0] = box[0] - padding
            box[1] = box[1] - padding
            box[2] = box[2] + padding
            box[3] = box[3] + padding   
        
            x1, y1, x2, y2 = box
    
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]
            crop_img_rgb = Image.fromarray(crop_img)
    
            imgs[class_name] = crop_img_rgb
        
    return imgs

def word_detection(img, detector, padding):
    
    detection = detector(img)
    
    class_indexes = detection[0].boxes.cls.cpu().numpy()
    class_names = [detector.names[int(class_index)] for class_index in class_indexes]
    boxes = detection[0].boxes.xyxy.cpu().numpy()
    confs = detection[0].boxes.conf.cpu().numpy()
    
    attributes = select_box(class_names, boxes, confs)
    
    imgs = crop_img(img, attributes, padding)

    return imgs, attributes

def word_recognition(imgs, recognitor):
    
    texts = {}
    
    for class_name, value in imgs.items():
        if class_name == 'type':
            texts[class_name] = value
        else:
            text = recognitor.predict(value)
            texts[class_name] = text
        
    return texts

def text_manipulation(raw_texts):
    texts = {}

    for class_name, text in raw_texts.items():
        if class_name not in ['current_place1', 'current_place2', 'origin_place1', 'origin_place2', 'iday', 'imonth', 'iyear']:
            texts[class_name] = text

    if 'origin_place1' in raw_texts and 'origin_place2' in raw_texts:
        texts['origin_place'] = f"{raw_texts['origin_place1']}, {raw_texts['origin_place2']}"
    elif 'origin_place1' in raw_texts:
        texts['origin_place'] = f"{raw_texts['origin_place1']}"
    elif 'origin_place2' in raw_texts:
        texts['origin_place'] = f"{raw_texts['origin_place2']}"

    if 'current_place1' in raw_texts and 'current_place2' in raw_texts:
        texts['current_place'] = f"{raw_texts['current_place1']} {raw_texts['current_place2']}"
    elif 'current_place1' in raw_texts:
        texts['current_place'] = f"{raw_texts['current_place1']}"
    elif 'current_place2' in raw_texts:
        texts['current_place'] = f"{raw_texts['current_place2']}"

    if 'iday' in raw_texts and 'imonth' in raw_texts and 'iyear' in raw_texts:
        texts['issue_date'] = f"{raw_texts['iday']}/{raw_texts['imonth']}/{raw_texts['iyear']}"

    return texts

def plot_result(img, attributes, padding):

    for class_name, value in attributes.items():
        if class_name != 'type':
            box = value[0]
            
            box[0] = box[0] - padding
            box[1] = box[1] - padding
            box[2] = box[2] + padding
            box[3] = box[3] + padding   
    
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
            (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,255,0), -1)
            img = cv2.putText(img, class_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    plt.imshow(img)

def predict(recognitor, detector, input_path, plot=False, padding=0):

    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)

    imgs, attributes = word_detection(img, detector, padding)

    raw_texts = word_recognition(imgs, recognitor)

    texts = text_manipulation(raw_texts)

    return texts

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.ERROR)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):   
    os.makedirs(UPLOAD_FOLDER)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Kiểm tra xem có file trong request không
        if 'file' not in request.files:
            logging.error("No file in request")
            return jsonify({"error": "No file in request"}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({"error": "No file selected"}), 400

        print(f"File received: {file.filename}")
        print(f"Received request: {request.method}")
        print(f"Headers: {request.headers}")
        print(f"Files: {request.files}")
        print(f"Form Data: {request.form}")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            texts = predict(recognitor, detector, input_path=filepath, plot=False, padding=0)

            # Trả về kết quả
            return jsonify({"success": True, "text": texts})
    

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)