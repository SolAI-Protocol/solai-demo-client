from flask import Flask, request
from flask_cors import CORS
import time
import subprocess
import multiprocessing
from ParseandTrain import parse, preprocess, Model, train, dataset, test
import torch
import pickle
import hashlib
import os
import boto3
import shutil
import random
import json
from workers import client_worker, aggregate_worker
import hashlib

PUB_KEY = 'hoV4PMWcHc71urq2RY4SSLbfnE2C7FQ1kYpfjTdQAnd'
ASSOCIATED_TOKEN_ADDRESS = 'GKwbknkQFwkAv1fkHnvrwo6nXB6qrfznXnUp5ozmdpuH'
FT_ADDRESS = 'Gsv2927sTBWU8FhT9hEoG84hQ85chd3ErXdtAiFqhTp8'
PUB_KEY2 = '67Q9Uu9CHaifbairNeTLSnmXVsm7n6fCDuGpozaGS2v2'

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers='*')

client_queue = multiprocessing.Queue()
aggregate_queue = multiprocessing.Queue()

p = multiprocessing.Process(target=client_worker, args=(client_queue, aggregate_queue))
p.start()

p = multiprocessing.Process(target=aggregate_worker, args=(aggregate_queue,))
p.start()

@app.route('/', methods=['GET'])
def get():
    return 'Ok'

@app.route('/local-train', methods=['POST'])
def local_train():
    file = request.files.get('file')
    client_address = request.form.get('client_address')
    print('local-train', client_address)
    if file:
        directory = './client_env/' + client_address
        if os.path.exists(directory):
            return 'Already Doing Task'
        os.makedirs(directory)
        filepath = directory + '/' + 'data.zip'
        file.save(filepath)
        client_queue.put({"task": "train", "params": (client_address)})
        return "Train Started"
    else:
        return 'No file received'

@app.route('/check-done', methods=['GET'])
def check_done():
    client_address = request.args.get('client_address')
    if os.path.exists(f'./client_env/{client_address}/success'):
        shutil.rmtree(f'./client_env/{client_address}')
        return "success"
    if os.path.exists(f'./client_env/{client_address}/failed'):
        shutil.rmtree(f'./client_env/{client_address}')
        return "failed"
    return "processing"

@app.route('/local-models', methods=['GET'])
def get_local_models():
    rounds = [int(filename.split('.')[0]) for filename in os.listdir('./scores')]
    current_round = sorted(rounds)[-1]
    with open(f'./scores/{current_round}.json', 'r') as f:
        scores = json.load(f)
    return list(scores.keys())

@app.route('/evaluate-model', methods=['POST'])
def evaluate_model():
    file = request.files.get('file')
    client_address = request.form.get('client_address')
    print('evaluate-model', client_address)
    rounds = [int(filename.split('.')[0]) for filename in os.listdir('./scores')]
    current_round = sorted(rounds)[-1]
    with open(f'./scores/{current_round}.json', 'r') as f:
        scores = json.load(f)
    hash_value = random.choice(list(scores.keys()))
    if file:
        directory = f'./client_env/{client_address}'
        if os.path.exists(directory):
            return 'Already Doing Task'
        os.makedirs(directory)
        filepath = directory + '/' + 'data.zip'
        file.save(filepath)
        client_queue.put({"task": "test", "params": (hash_value, client_address)})
        return "Evaluation Started"
    else:
        return 'No file received'

@app.route('/end-round', methods=['POST'])
def end_round():
    token = request.form.get('token')
    if hashlib.sha3_256(token.encode()) != '952d93d9ea9e4c655f9014c3ca350f4c64b971fbfccb8dd491ba3355dd928600':
        return "Invalid Token"
    aggregate_queue({"task": "update_global"})
    return "End Start"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
