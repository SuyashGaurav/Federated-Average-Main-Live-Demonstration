import requests
import pandas as pd
from tqdm import tqdm

path_to_input = 'clients_2.txt'
data = pd.read_csv(path_to_input, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
DataLength = len(data)

for i in tqdm(range(DataLength)):
    url = "http://192.168.137.142:5002/"
    r = requests.post(url, data={'content': f"{data['File_ID'][i]}"})
    url1 = f"http://192.168.137.142:5002/download/{data['File_ID'][i]}"
    r1 = requests.get(url1, data={'content': f"{data['File_ID'][i]}"}, timeout=60)
    fp = open(f"downloaded_files/{data['File_ID'][i]}.txt", "wb")
    fp.write(r1.content)
    fp.close()