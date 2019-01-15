# import flask dependencies
from flask import Flask
from flask import jsonify, request, make_response
import tensorflow as tf
from util import *
from webscrape_helper import azureClaimSearch
import time
import random
import pickle
# from ..rep_model.REP import * 

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# initialize the flask app
app = Flask(__name__)
sess = tf.Session()

# def init():
#     global sess
#     load_model(sess)
#     print("Loaded model...")


# Getting Parameters
def getParameters():
    parameters = []
    # parameters.append(request.args.get('male'))
    return parameters

# Cross origin support
def sendResponse(responseObj):
    response = jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response


def runModel(sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    start_time = time.time()
    print("Now running predictions...")

    # THIS is the info from Henry
    userClaims = "../webscrape/claims.csv"
    userBodies = "../webscrape/bodies.csv"
    # parse that info
    raw_test = FNCData(userClaims, userBodies)
    # need more stuff for this
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    # idk what this does really
    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    # run predictions
    test_pred = sess.run(predict, feed_dict=test_feed_dict)
    # timing
    print("generate test_set--- %s seconds ---" % (time.time() - start_time))
    print("Predictions complete.")
    return test_pred

def trainVectors():
    file_train_instances = "train_stances.csv"
    file_train_bodies = "train_bodies.csv"
    file_test_instances = "test_stances_unlabeled.csv"
    file_test_bodies = "test_bodies.csv"

    # Initialise hyperparameters
    r = random.Random()
    lim_unigram = 5000

    # Load data sets
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)

    # Process data sets
    _, _, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    
    storeVector(bow_vectorizer, "bow.pickle")
    storeVector(tfreq_vectorizer, "tfreq.pickle")
    storeVector(tfidf_vectorizer, "tfidf.pickle")

def calculateScore():
    # call train vectors only at the first time
    # trainVectors()
    bow_vectorizer = loadVector("bow.pickle")
    tfreq_vectorizer = loadVector( "tfreq.pickle")
    tfidf_vectorizer = loadVector("tfidf.pickle")

    # hardcode the number of features
    feature_size = 10001
    target_size = 4
    hidden_size = 100

    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(softmaxed_logits, 1)

    # LOAD MODEL
    sess = tf.Session()
    print("Loading checkpoint")
    load_model(sess)

    
    '''PREDICTION'''
    print("Now running predictions...")

    # THIS is the info from Henry
    userClaims = "claims.csv"
    userBodies = "bodies.csv"
    # parse that info
    raw_test = FNCData(userClaims, userBodies)
    # TODO hotload the vector representations instead of calculating every time
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    # run predictions
    test_pred = sess.run(predict, feed_dict=test_feed_dict)
    # timing
    print("Predictions complete.")
    return test_pred




# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    start_time = time.time()
    # init()
    claim = request.args.get('claim')
    # parameters = getParameters()

    # webscrape
    azureClaimSearch(claim)

    # run model
    stances = calculateScore()
    score = 0
    for stance in stances:
        # agree
        if stance == 0:
            score += 1
        elif stance == 1 or stance == 2:
            score -= 1
    print("Total response time--- %s seconds ---" % (time.time() - start_time))

    return sendResponse({"claim": claim, "score": score,  \
    "sources": ['https://www.washingtonpost.com/news/powerpost/wp/2018/01/11/joe-arpaio-is-back-and-brought-his-undying-obama-birther-theory-with-him/?utm_term=.3c88c56fee34']})


# default route
@app.route('/')
def index():
    return 'Hello world'

# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)
    # fetch action from json

    verdict = "TRUTH"
    number_of_articles = 155

    article1 = "http://amulya.co"
    article2 = "http://amulya.co/article1"

    command = req.get('queryResult').get('parameters').get('Command')
    action = req.get('queryResult').get('intent').get('displayName')

    if action == "echo":
        stance = req.get('queryResult').get('parameters').get('echoText')
        sentence = "Your search was: '" + stance + "' "
        response = {
            "fulfillmentText": sentence,
            "source" : "TruthAI",  
        } 
        return response
         
    if action == "webhook-intent":
        stance = req.get('queryResult').get('parameters').get('stance')
      
        sentence = "Your search was: '" + stance + "' and your command was '" + command + "' | We referenced " + str(number_of_articles)  + " articles and our verdict about your stance is " + verdict + " | Here are a few more articles " + article1 + " and " + article2  + " for more info"

        try:
            stance
        except:
            sentence = "please try again"

        response = {
            "fulfillmentText": sentence,
            "source" : "TruthAI",  
            
            "payload": {
                "google": {
                "expectUserResponse": True,
                "richResponse": {
                    "items": [
                    {
                        "simpleResponse": {
                        "textToSpeech": "Howdy! I can tell you fun facts about almost any number."
                        }
                    },
                    {
                        "simpleResponse": {
                        "textToSpeech": "What number do you have in mind?"
                        }
                    }
                    ],
                    "suggestions": [
                    {
                        "title": "25"
                    },
                    {
                        "title": "45"
                    },
                    {
                        "title": "Never mind"
                    }
                    ],
                    "linkOutSuggestion": {
                    "destinationName": "Website",
                    "url": "https://assistant.google.com"
                    }
                }
                }
            }
        }
        # return a fulfillment response
        return response


# create a route for webhook

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))


# run the app
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    app.run(debug=True, threaded=True)