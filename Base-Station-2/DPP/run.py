import os
import time
import math
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from loader.load_data import *
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from optimizers.network import *
from optimizers.constrained import constrained_solve
import requests
import tensorflow as tf
import warnings
warnings.simplefilter('ignore')

import sys
import hashlib

BUF_SIZE = 65536

sha256 = hashlib.sha256()

def env_hash():
    with open("./.env", 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

load_dotenv()

Q = int(os.getenv("Q_INIT"))
past = int(os.getenv("PAST"))
V_0 = int(os.getenv("V_0"))
future = int(os.getenv("FUTURE"))
alpha = float(os.getenv("ALPHA"))
NumSeq = int(os.getenv("NUM_SEQ"))
threshold = int(os.getenv("THRESHOLD"))
train_memory = int(os.getenv("TRAIN_MEMORY"))
use_saved = os.getenv("USE_SAVED")=="True"
cost_constraint = int(os.getenv("COST_CONSTRAINT"))
time_limit = float('inf') if os.getenv("TIME_LIMIT")=='inf' else int(os.getenv("TIME_LIMIT"))
path_to_input = os.getenv("PATH_TO_INPUT")
tag = env_hash()[:10]
print("Experiment Tag:", tag)

cache_constraint = int(alpha*threshold)

path = f"./experiments/csv_{NumSeq}/"
try:
    os.makedirs(path)
except FileExistsError:
    pass

our_path = f"./experiments/{tag}/"
try:
    os.makedirs(our_path)
except FileExistsError:
    pass
copyfile("./.env", our_path+"/.env")

data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
# DataLength = len(data)
DataLength = 659963//2
# print(len(data)+10)
# print(int((0+1)*DataLength/NumSeq))


gamma = np.random.normal(0, 1, (threshold,))

class SimpleMLP:
    @staticmethod
    def get_model(past, threshold):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(past, threshold)))
        model.add(RepeatVector(future))
        model.add(LSTM(64, activation='relu'))
        model.add(RepeatVector(future))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(threshold)))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
        return model
    
smlp_global = SimpleMLP()
global_model = smlp_global.get_model(past, threshold)



queue = []
err = []
objective = []
fetching_cost = []
cache_hit = []
prev_demands = []
best_maximum = []
hit_rate = []
download_rate = []


X_t_1 = np.zeros((threshold,))
init_indices = random.sample(range(threshold), cache_constraint)
X_t_1[init_indices] = 1
cachePath = f"../cached_file/"
if os.path.exists(cachePath):
    shutil.rmtree(cachePath)
    
os.makedirs(cachePath)
for i in range(threshold):
    if X_t_1[i]==1:
        url = f"http://10.196.11.11:5001/download/{i}"
        r = requests.get(url, data={'content': f"{i}"}, timeout=60)
        fp = open(f'../TempFilesDpp/{i}.txt', 'wb')
        fp.write(r.content)
        fp.close()
        source_path = f'../TempFilesDpp/{i}.txt'
        destination_path = f'../cached_file/{i}.txt'
        # Copy the file to the destination directory
        shutil.copy(source_path, destination_path)
        os.remove(f'../TempFilesDpp/{i}.txt')
mse = 0
w = 1.0
neita = 1e-2
i=0
with tqdm(total=NumSeq) as pbar:
    while i<NumSeq:
        data = pd.read_csv(path_to_input, sep = ' ')
        data.columns = ['Timestamp', 'File_ID', "File_Size"]
        if len(data) >= int((i+1)*DataLength/NumSeq):

            global_weights = global_model.get_weights()

            V = V_0
            if os.getenv("USE_ROOT_V")=="True": V *= (i+1)**0.5
            next_dem, times = get_demands(i, time_limit, data, DataLength, NumSeq, threshold)
            X_t = np.zeros((threshold,))
            init_indices = random.sample(range(threshold), cache_constraint)
            X_t[init_indices] = 1
            
            if i==past+future:
                model = get_model(prev_demands, global_weights, past, future, threshold, use_saved)
                print(model.summary())
            elif i>past+future:
                to_train = prev_demands[max(0, i-train_memory):]
                model.set_weights(global_weights)
                update_weight(model, to_train, past, future)
                model.save("./models/init.h5")

                pred = predict_demand(model, prev_demands[i-past:])
                pred = np.maximum(pred, np.zeros((pred.size,)))
                pred = np.round(pred)
                np.array(prev_demands).mean(axis=0)
                
                delta_t = get_delta()
                X_t, obj = constrained_solve(pred, cache_constraint, cost_constraint, X_t_1, delta_t, Q, V, threshold)
                objective.append(obj)
                Delta = delta_t*np.linalg.norm(X_t-X_t_1, ord=1)/2
                fetching_cost.append(Delta)
                
                
                e = np.linalg.norm(next_dem-pred, ord=2)/len(pred)
                err.append(e)
                actual_cache_hit = np.dot(next_dem, X_t)
                cache_hit.append(actual_cache_hit)
                
                #
                mseError = np.linalg.norm(next_dem-pred,ord=2)
                mse = mse + math.sqrt(mseError)
                w = w*math.exp(-mse*neita/i)
                
                indices = np.argsort(next_dem)[::-1][:cache_constraint]
                final = np.zeros((threshold,))
                final[indices] = 1
                
                #
                best = np.dot(next_dem, final)
                best_maximum.append(best)
                        
                Q = max(Q + Delta - cost_constraint, 0)
                queue.append(Q)
                
            plt.plot(ma(cache_hit))
            plt.title("Cache Hit vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit")
            plt.savefig(our_path+"Cache_Hit.jpg")
            plt.clf()
            
            plt.plot(ma(err))
            plt.title("Mean Squared Test Error in Demand Prediction vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("MSE")
            plt.savefig(our_path+"NN-MSE.jpg")
            plt.clf()


            plt.plot(ma(queue))
            plt.title("Q vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Q")
            plt.savefig(our_path+"Q.jpg")
            plt.clf()


            plt.plot(ma(objective))
            plt.title("Constrained Objective Function vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Objective Function")
            plt.savefig(our_path+"Obj.jpg")
            plt.clf()


            plt.plot(ma(fetching_cost))
            plt.title("Fetching Cost vs Timeslot")
            plt.axhline(y=cost_constraint, linewidth=2, label='Cost Constraint')
            plt.xlabel("Timeslot")
            plt.ylabel("Cost")
            # plt.legend(loc = 'upper left')
            plt.savefig(our_path+"Cost.jpg")
            plt.clf()


            plt.plot(ma(cache_hit))
            plt.title("Cache Hit vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit")
            plt.savefig(our_path+"Cache_Hit.jpg")
            plt.clf()
            
            hit_rate.append(np.dot(X_t, next_dem)/np.sum(next_dem))
            download_rate.append(np.sum(np.logical_and(X_t==1, X_t_1==0))/np.sum(next_dem))
            
            plt.plot(ma(hit_rate))
            plt.title("Cache Hit Rate vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Cache Hit Rate")
            plt.savefig(our_path+"Cache_Hit_Rate.jpg")
            plt.clf()
            
            plt.plot(ma(download_rate))
            plt.title("Download Rate vs Timeslot")
            plt.xlabel("Timeslot")
            plt.ylabel("Download Rate")
            plt.savefig(our_path+"Download_Rate.jpg")
            plt.clf()

            # max_retries = 20
            # retry_delay = 0.2
            for j in range(threshold):#
                if X_t[j]==1 and X_t_1[j]==0:
                    url = f"http://10.196.11.11:5001/download/{j}"
                    # for retry in range(max_retries):
                        # try:
                    r = requests.get(url, data={'content': f"{j}"}, timeout=60)
                    f = open(f'../TempFilesDpp/{j}.txt', 'wb')
                    f.write(r.content)
                    f.close()
                        #     break
                        # except PermissionError:
                        #     time.sleep(retry_delay)
                        # else:
                        #     print(f"Unable to access the file after {max_retries} retries.")
                    source_path = f'../TempFilesDpp/{j}.txt'
                    destination_path = f'../cached_file/{j}.txt'
                    shutil.copy(source_path, destination_path)
                    os.remove(f'../TempFilesDpp/{j}.txt')

                if X_t[j]==0 and X_t_1[j]==1:
                    destination_path = f'../cached_file/{j}.txt'
                    os.remove(destination_path)
                    
            X_t_1 = X_t
            
            prev_demands.append(next_dem)

            if i >= past+future:
                model_path = 'models/init.h5'
                url = "http://10.196.11.11:5001/upload_model/1"
                weight_data = {'integer': w}
                with open(model_path, 'rb') as file:
                    files = {'file': (model_path, file)}
                    response = requests.post(url, files=files, data=weight_data, timeout=60)

            if i>=past+future:
                url1 = "http://10.196.11.11:5001/get_global/"
                r = requests.get(url1, timeout=60)
                fp = open("models/global_model.h5", "wb")
                fp.write(r.content)
                fp.close()
                downloaded_global_model = tf.keras.models.load_model('models/global_model.h5')
                global_model.set_weights(downloaded_global_model.get_weights())

            # if i>=past+future:
            #     url2 = "http://10.196.11.11:5001/upload_w/1"
            #     data = {'integer': w}
            #     response = requests.post(url2, json=data, timeout=60)

            pbar.update(1)
            i = i+1
        else:
            time.sleep(2)
        

        pd.DataFrame(hit_rate).to_csv(our_path+'hit_rate.csv',index=False)
        pd.DataFrame(download_rate).to_csv(our_path+'download_rate.csv',index=False)
