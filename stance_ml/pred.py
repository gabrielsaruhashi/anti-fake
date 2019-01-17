# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from util import *
import random
import tensorflow as tf
import pandas as pd
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

tf.reset_default_graph()


file_target = 'competition_test_stances.csv'
target = pd.read_csv(file_target)

test_target = target[['Stance']]
test_target_labels = get_labels(test_target)

# Prompt for mode
# mode = input('mode (load / train)? ')


# Set file names
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "test_stances_unlabeled.csv"
file_test_bodies = "test_bodies.csv"
file_predictions = 'predictions_test.csv'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 5


# # Load data sets
# raw_train = FNCData(file_train_instances, file_train_bodies)
# raw_test = FNCData(file_test_instances, file_test_bodies)
# n_train = len(raw_train.instances)


# # Process data sets
# train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
#     pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)

# # train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer  = pipeline_train_cached()
# feature_size = len(train_set[0])
# test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

feature_size = 1001

# Define model

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define conv alyer
# 
print(features_pl.shape)
print(features_pl)
features_reshaped = tf.expand_dims(features_pl)

conv1 = tf.nn.relu(tf.nn.conv1d(features_pl, 3, padding='SAME', stride=1))

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
hidden_layer2 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer, hidden_size * 2)), keep_prob=keep_prob_pl)
hidden_layer3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(hidden_layer2, hidden_size)), keep_prob=keep_prob_pl)

logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer3, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.argmax(softmaxed_logits, 1)

out = pd.read_csv('test_stances_unlabeled.csv')

# Load model
if mode == 'load':
    with tf.Session() as sess:
        load_model(sess, "./my_model/model.checkpoint")
        # get target pred

        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        
        y_hat = np.array(test_pred)
        y = np.array(test_target_labels)
        
        count = (y == y_hat).sum()
        print(float(count) / len(test_target_labels))


# Train model
if mode == 'train':

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss
                print('Testing loss: {}\n'.format(current_loss))

        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        
        y_hat = np.array(test_pred)
        y = np.array(test_target_labels)
        
        count = (y == y_hat).sum()
        print("Training ended. Accuracy is: ".format(float(count) / len(test_target_labels)))

        save_model(sess)


# Save predictions
out = pd.concat([out, pd.DataFrame(test_pred)], axis=1)
print(out)
save_score_predictions(out, file_predictions)
# save_predictions(test_target_labels, 'target_labels.csv')