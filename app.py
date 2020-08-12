from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

folder_dataset = "./static/assets/images/dataset/"
LIST_PATH_IMAGE_DATASET = [
    f for f in os.listdir(folder_dataset) if os.path.isfile(os.path.join(folder_dataset, f))
]

def checkSimilarity(input_img, dataset_img):
    # Mendapatkan jumlah citra yang ada di database
    num_of_database = dataset_img.shape[0]

    # penampung nilai kemiripan
    val_of_sim = np.array([], dtype='float32')

    # perhitungan kemiripan dengan euclidean distance
    for i in range(num_of_database):

        selisih_antar_titik = input_img[0] - dataset_img[i]
        kuadrat_antar_titik = selisih_antar_titik**2
        total_semua_titik = np.sum(kuadrat_antar_titik)
        nilai_kemiripan = total_semua_titik**0.5  # akar pangkat 2
        val_of_sim = np.append(val_of_sim, [nilai_kemiripan])

    val_of_sim_sort = np.sort(val_of_sim)  # pengurutan nilai kemiripan

    indices_val_of_sim = np.argsort(val_of_sim)  # index pengurutan nilai kemiripan

    return indices_val_of_sim

def mainImageRetrieval(image_input):
    ENCODER_MODEL = load_model('./static/assets/model/model.h5')
    DATASET_ARRAY = np.load('./static/assets/array-datasets/dataset_kecil_64x64.npy').astype('float32') / 255
    # print(DATASET_ARRAY[1])

    # mengubah ukuran dari 64x64x3 menjadi 1x64x64x3
    image_input = image_input.reshape(
        1, image_input.shape[0], image_input.shape[1], image_input.shape[2])
    print(image_input[0])

    # predict 
    feature_image_input = ENCODER_MODEL.predict(image_input)
    feature_image_dataset = ENCODER_MODEL.predict(DATASET_ARRAY)
    
    # backend.clear_session()
    indices_val_of_sim = checkSimilarity(feature_image_input, feature_image_dataset)

    return indices_val_of_sim


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['file']

        img = Image.open(file.stream)
        uploaded_img_path = 'static/assets/images/upload/'+ file.filename
        img.save(uploaded_img_path)

        img_input = img.resize((64, 64), Image.ANTIALIAS) # citra diresize ke 64x64
        image_input = np.array(img_input).astype('float32') / 255 # normalisasi from 0-255 to 0-1

        #proses retrieval
        indices_val_of_sim = mainImageRetrieval(image_input=image_input)
        
        indices_val_of_sim_path = [] # untuk menyimpan alamat path citra
        num_of_output = 50
        for i in range(num_of_output):
            indices_val_of_sim_path.append(
                LIST_PATH_IMAGE_DATASET[indices_val_of_sim[i]])

        query_img = file.filename
        result = indices_val_of_sim_path
        return render_template('index.html', query_img=query_img, data=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)