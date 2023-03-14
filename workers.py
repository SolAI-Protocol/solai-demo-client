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
from utils import s3_put_object, nft_mint_call, add_state_dict, divide_state_dict, s3_connection

def client_worker(queue: multiprocessing.Queue, aggregateQueue: multiprocessing.Queue):
    model = Model()
    s3 = s3_connection()
    while True:
        try:
            msg = queue.get()
            rounds = [int(filename.split('.')[0]) for filename in os.listdir('./scores')]
            current_round = sorted(rounds)[-1]
            if msg["task"] == "train":
                client_address = msg["params"]
                directory = f'./client_env/{client_address}'
                start = time.time()
                parsed_data = parse(directory)
                pre_data = preprocess(parsed_data)
                loader_train, loader_test = dataset(pre_data)
                model = train(loader_train, loader_test, model, 200)
                state_dict = model.state_dict()
                state_dict_str = pickle.dumps(state_dict)
                hash_obj = hashlib.sha256(state_dict_str)
                hash_value = hash_obj.hexdigest()
                torch.save(model.state_dict(), directory + '/model.pt')
                result = s3_put_object(s3, 'solai', directory +
                                    '/model.pt', 'local-trained-models/' + hash_value + '.pt')
                print(result)
                metadata = {
                    "name": "SolAI #" + hash_value[-8:],
                    "symbol": "SolAI",
                    "description": "Local Trained AI",
                    "image": "https://raw.githubusercontent.com/Community-Driven-AI/.github/main/solai.png",
                    "properties": {
                        "model_parameters": {
                            "hash": hash_value,
                            "uri": "https://solai.s3.amazonaws.com/local-trained-models/" + hash_value + '.pt'
                        }
                    }
                }
                with open(directory + '/metadata.json', 'w') as f:
                    json.dump(metadata, f)
                result = s3_put_object(s3, 'solai', directory +
                                    '/metadata.json', 'local-trained-models/' + hash_value + '.json')
                result = nft_mint_call(
                    client_address, "https://solai.s3.amazonaws.com/local-trained-models/" + hash_value + '.json')
                if not result:
                    print("Minting Failed")
                
                with open(f'./scores/{current_round}.json', 'r') as f:
                    scores = json.load(f)
                    if hash_value not in scores:
                        scores[hash_value] = []
                with open(f'./scores/{current_round}.json', 'w') as f:
                    json.dump(scores, f)

                end = time.time()
                print(f"{end - start:.5f}sec total")

            elif msg["task"] == "test":
                hash_value, client_address = msg["params"]
                directory = f'./client_env/{client_address}'
                parsed_data = parse(directory)
                pre_data = preprocess(parsed_data)
                loader_train, loader_test = dataset(pre_data)
                # Read the contents of the file
                s3.download_file('solai', 'local-trained-models/' + hash_value + '.pt', directory + '/model.pt')
                state_dict = torch.load(directory + '/model.pt')
                model.load_state_dict(state_dict)
                train_mse = test(loader_train, model)
                test_mse = test(loader_test, model)
                score = 0.8 * train_mse + 0.2 * test_mse
                score = int(score * 10000)

                aggregateQueue.put({"task": "upload_score", "params": (hash_value, score)})

            with open(f'./client_env/{client_address}/success', 'w') as f:
                pass

        except Exception as e:
            print(e)
            with open(f'./client_env/{client_address}/failed', 'w') as f:
                pass


def aggregate_worker(queue: multiprocessing.Queue):
    model = Model()
    s3 = s3_connection()
    while True:
        msg = queue.get()
        rounds = [int(filename.split('.')[0]) for filename in os.listdir('./scores')]
        current_round = sorted(rounds)[-1]
        if msg["task"] == "update_global":
            with open(f'./scores/{current_round}', 'r') as f:
                scores = json.load(f)

            global_state_dict = model.state_dict()
            for key in global_state_dict:
                global_state_dict[key].zero_()
            
            for hash_value in scores.keys():
                s3.download_file('solai', 'local-trained-models/' + hash_value + '.pt',  './aggregator/local_model.pt')
                state_dict = torch.load('./aggregator/local_model.pt')
                global_state_dict = add_state_dict(global_state_dict, state_dict)
            
            global_state_dict = divide_state_dict(global_state_dict, len(list(scores.keys())))

            torch.save(global_state_dict, './aggregator/global_model.pt')
            s3_put_object(s3, 'solai', './aggregator/global_model.pt', f'global-models/{current_round}.pt')
                            
        elif msg["task"] == "upload_score":
            hash_value, score = msg["params"]
            with open(f'./scores/{current_round}.json', 'r') as f:
                scores = json.load(f)
                if hash_value in scores:
                    scores[hash_value].append(score)
            with open(f'./scores/{current_round}.json', 'w') as f:
                json.dump(scores, f)