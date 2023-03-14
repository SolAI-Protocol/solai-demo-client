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

def s3_connection():
    try:
        with open("./credentials/s3_keys.json", 'r') as f:
            keys = json.load(f)
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",  # 자신이 설정한 bucket region
            aws_access_key_id=keys["access_key"],
            aws_secret_access_key=keys["private_key"],
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3

def s3_put_object(s3, bucket, filepath, access_key):
    try:
        s3.upload_file(
            Filename=filepath,
            Bucket=bucket,
            Key=access_key,
        )
    except Exception as e:
        return False
    return True

def add_state_dict(state_dict1, state_dict2):
    new_state_dict = {}
    for key in state_dict1.keys():
        if key in state_dict2:
            new_state_dict[key] = state_dict1[key] + state_dict2[key]
        else:
            new_state_dict[key] = state_dict1[key]
    return new_state_dict

def divide_state_dict(state_dict, alpha):
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = state_dict[key] / alpha
    return new_state_dict

def ft_mint_call(amount, to_address):
    result = subprocess.run(['spl-token', 'transfer', '--fund-recipient', '--allow-unfunded-recipient',
                            FT_ADDRESS, str(amount), to_address], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    return False


def nft_mint_call(to_address, metadata_uri):
    result = subprocess.run(
        ['spl-token', 'create-token', '--decimals', '0'], capture_output=True, text=True)
    if result.returncode == 0:
        address = result.stdout.split()[2]
        result = subprocess.run(
            ['spl-token', 'create-account', address], capture_output=True, text=True)
        if result.returncode == 0:
            account = result.stdout.split()[2]
            time.sleep(60)
            result = subprocess.run(['./metaplex-token-metadata-test-client', 'create_metadata_accounts', '--keypair', './credentials/keypair.json', '--mint', address, '--name',
                                    'SOLAI', '--symbol', 'SAI', '--uri', metadata_uri], capture_output=True, text=True)
            if result.returncode == 0:
                result = subprocess.run(
                    ['spl-token', 'mint', address, '1', account], capture_output=True, text=True)
                if result.returncode == 0:
                    result = subprocess.run(
                        ['spl-token', 'authorize', address, 'mint', '--disable'], capture_output=True, text=True)
                    if result.returncode == 0:
                        result = subprocess.run(['spl-token', 'transfer', '--fund-recipient', '--allow-unfunded-recipient',
                                                address, '1', to_address], capture_output=True, text=True)
                        if result.returncode == 0:
                            return True
    return False