from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import PIL

app = Flask(__name__)

folder = ".\\assets\\images\\dataset\\"
LIST_PATH_IMAGE_DATASET = [
    f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
]

def checkSimilarity(input_img, dataset_img):
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

    return val_of_sim_sort, indices_val_of_sim

def imageRetrieval(class_of_input, input_img, dataset_img, category_dataset):

    # Pengubahan dimensi, misal 10000x64x64x1 menjadi 10000x4096
    num_of_database, row, column, channels = dataset_img.shape
    num_of_input = input_img.shape[0]
    dataset_img = dataset_img.reshape(num_of_database, (row * column),
                                      channels)
    input_img = input_img.reshape(num_of_input, (row * column), channels)

    # Pemanggilan fungsi similaritas
    val_of_sim_sort, indices_val_of_sim = checkSimilarity(
        input_img, dataset_img)  # Mendapatkan nilai dan index
    #     print(val_of_sim_sort.shape)

    category_output = []
    for i in range(len(indices_val_of_sim)):
        cat = indices_val_of_sim[i] // 500
        category_output.append(category_dataset[cat])

        # Akurasi
        precision = np.array([], dtype='float32')
        recall = np.array([], dtype='float32')

        num_of_image_per_class = 500

    relevant = 0
    for i in range(len(indices_val_of_sim)):
        if indices_val_of_sim[i] // num_of_image_per_class == class_of_input:
            relevant += 1
            prec = relevant / (i + 1) * 100
            rec = relevant / 20 * 100
            precision = np.append(precision, [prec])
            recall = np.append(recall, [rec])
        if relevant == 20:
            break
    
    return precision, recall, indices_val_of_sim

def mainImageRetrieval(image_input):
    ENCODER_MODEL = load_model('.\\assets\\model\\model.h5')
    DATASET_FEATURES_ARRAY = np.load('assets\\array-dataset\\dataset_kecil_64x64.npy')

    # Cek apakah citra berukuran 64x64 atau tidak
    if image_input.shape[0] != 64 and image_input.shape[1] != 64:
        image_input = cv2.resize(
            image_input, (64, 64), interpolation=cv2.INTER_AREA)
    # print(image_input)
    # mengubah ukuran dari 64x64x3 menjadi 1x64x64x3
    image_input = image_input.reshape(
        1, image_input.shape[0], image_input.shape[1], image_input.shape[2])

    # normalisasi
    image_input = image_input.astype('float32') / 255
    # print(image_input)
    
    if option_model != "tanpaFeatureLearning":
        # backend.clear_session()
        # mendapatkan fitur dari citra input
        feature_image_input = ENCODER_MODEL.predict(image_input)
        backend.clear_session()
    else:
        feature_image_input = image_input

    precision, recall, indices_val_of_sim, duration = imageRetrieval(
        class_of_input, feature_image_input, DATASET_FEATURES_ARRAY,
        CATEGORY_DATASET)

    return precision, recall, indices_val_of_sim, duration


@app.route('/', methods=['GET', 'POST'])
def index():
    path_hasil = 'static/assets/hasil/'
    if request.method == 'POST':

        file = request.files['query_img']
        img = Image.open(file.stream)
        image_input = np.array(img) // 255

        uploaded_img_path = 'static/assets/upload/'+ file.filename
        result_name = file.filename[:-4]
        img.save(uploaded_img_path)
        img_gray.save(path_hasil + result_name + '.jpg')

        #proses retrieval
        indices_val_of_sim = mainImageRetrieval(image_input=image_input)

        indices_val_of_sim_path = []
        num_of_output = 50
        for i in range(num_of_output):
            indices_val_of_sim_path.append(
                LIST_PATH_IMAGE_DATASET[indices_val_of_sim[i]])

        query_img = file.filename
        result = indices_val_of_sim_path
        return render_template('index.html', query_img=query_img, data=result)
    else:
        return render_template('index.html')

    # if request.method == 'POST':
    #     # check if the post request has the file part
    #     if 'file' not in request.files:
    #         flash('No file part')
    #         return redirect(request.url)
    #     file = request.files['file']
    #     # if user does not select file, browser also
    #     # submit an empty part without filename
    #     if file.filename == '':
    #         flash('No selected file')
    #         return redirect(request.url)
    #     if file and allowed_file(file.filename):
    #         option_model = request.form['optmodel']
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    #         # Mendapatkan kelas dari input
    #         class_of_input = find_class(file.filename)

    #         path_image_input = os.path.join(app.config['UPLOAD_FOLDER'],
    #                                         filename)
    #         image_input = cv2.imread(path_image_input)

    #         precision, recall, indices_val_of_sim, duration = mainImageRetrieval(
    #             class_of_input, image_input, option_model)

    #         indices_val_of_sim_path = []
    #         num_of_output = 50
    #         for i in range(num_of_output):
    #             indices_val_of_sim_path.append(
    #                 LIST_PATH_IMAGE_DATASET[indices_val_of_sim[i]])

    #         query_img = file.filename
    #         result = indices_val_of_sim_path
    #         # return redirect(url_for('upload_file', filename=filename))
    #         return render_template(
    #             'main.html', query_img=query_img, data=result, time_retrieval=duration)
    # else:
    #     return render_template('main.html')