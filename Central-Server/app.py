from flask import Flask, render_template, url_for, request, redirect, send_file, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
import warnings
warnings.simplefilter('ignore')



app = Flask(__name__)
UPLOAD_FOLDER = 'Models/'  # Set the folder where the uploaded files will be saved
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/download/<string:content>')
def download(content):
    file_path = f'AllFiles/{content}.txt'
    return send_file(file_path, as_attachment=True)

def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean/len(scaled_weight_list))
    return avg_grad

@app.route('/upload_model/<int:id>', methods=['POST'])
def upload_model(id):
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'init{id}.h5'))
    folder_path = 'Models/'
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    number_of_files = len(files)

    if number_of_files == 2:#
        scaled_local_weight_list = list()
        for models in os.listdir(folder_path):
            model = tf.keras.models.load_model(os.path.join(folder_path, models))
            scaled_local_weight_list.append(model.get_weights())
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        global_model = tf.keras.models.load_model('GlobalModel/init_global.h5')
        global_model.set_weights(average_weights)
        global_model.save('GlobalModel/init_global.h5')

        # clear Models
        Models_path = 'Models/'
        files = os.listdir(Models_path)
        for models in files:
            curr_path = os.path.join(Models_path, models)
            if os.path.isfile(curr_path):
                os.remove(curr_path)

    return "File uploaded successfully"


@app.route('/get_global/')
def get_global():
    while True:
        folder_path = 'Models/'
        files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        number_of_files = len(files)
        if number_of_files == 0:#
            break

    file_path = f'GlobalModel/init_global.h5'
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    # clear Models
    Models_path = 'Models/'
    files = os.listdir(Models_path)
    for file in files:
        curr_path = os.path.join(Models_path, file)
        if os.path.isfile(curr_path):
            os.remove(curr_path)
    app.run(host='0.0.0.0', port=5001, debug=True)

