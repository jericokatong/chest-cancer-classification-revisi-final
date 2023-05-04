from flask import Flask, render_template, request

from keras.models import load_model

from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image


app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/images/temp.png"
    imagefile.save(image_path)

    model = load_model('./model/chest_cancer_model_jerico.h5')

    def chestScanPrediction(path, _model):
        classes_dir = ["Adenocarcinoma","Large cell carcinoma","Normal","Squamous cell carcinoma"]
        # Loading Image
        img = image.load_img(path, target_size=(256,256))
        # Normalizing Image
        norm_img = image.img_to_array(img)/255
        # Converting Image to Numpy Array
        input_arr_img = np.array([norm_img])
        # Getting Predictions
        pred = np.argmax(_model.predict(input_arr_img))
        # Printing Model Prediction
        pred_score = _model.predict(input_arr_img)
        # print(classes_dir)
        # print(pred_score)
        # print(classes_dir[pred])
        pred_score = np.array(pred_score)

        # Normalisasi skor
        total_skor = sum(pred_score[0])
        skor_normal = [s/total_skor for s in pred_score[0]]

        # Konversi skor ke persentase
        skor_persen = [round(s * 100, 2) for s in skor_normal]
        # print('setelah ubah')
        print(skor_persen)
        nilai_terbesar = max(skor_persen)
        

        return [classes_dir, nilai_terbesar, classes_dir[pred]]
    
    path = image_path
    result = chestScanPrediction(path,model)

    return render_template('index.html', label = result[0], skor = result[1], prediction = result[2]);

if __name__ == "__main__":
    app.run(debug=True)

