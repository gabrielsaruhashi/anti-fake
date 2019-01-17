# import flask dependencies
from flask import Flask
from flask import jsonify, request, make_response
import tensorflow as tf
from util import *
from webscrape_helper import azureClaimSearch
import time
import random
import pickle
from REP import *
import pandas as pd
from threading import Thread, Lock

from app import *
# from ..rep_model.REP import * 

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

global_res = {}


# class myThread(Thread):
#     def __init__(self, query):
#         Thread.__init__(self)
#         self.query = query

#     def run(self):
#         score, _, _ = runPipeline(self.query)
#         global_res[self.query] = score
        
test_set = pd.read_csv('ben_test.csv')


index = 0
threads = []

for index, row in test_set.iterrows():
    score, _, _ = runPipeline(row['Claim'])
    global_res[row['Claim']] = score

    label = row['Label']
    print(row['Claim'])
    count = 0
    total = 0
    if score > 0 and label == True:
        count += 1
    elif score < 1 and label == False:
        count += 1
    total += 1
print("End")
print(float(count) / total)


# for index, row in test_set.iterrows():
#     claim = row['Claim']
#     label = row['Label']
#     threads.append(myThread(claim))
#     threads[index].start()
#     index += 1
# for thread in threads:
#     thread.join()

df = pd.DataFrame(global_res)
df.to_csv('output')

