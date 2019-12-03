# -*- coding: utf-8 -*-
from config import load_config
import sys
import numpy as np
import re
import tensorflow as tf
import BERT_Model as brt
import utils as utl

tf.set_random_seed(42)
np.random.seed(42)

def get_predict(result, output):
    result = np.argmax(result, axis=1)
    max_index = -1
    max_pro = 0.0
    for i in range(len(output)):
        x = output[i][1]
        if x > max_pro:
            max_index = i
            max_pro = x
    r = result[max_index]
    if r == 1:
        return True
    else:
        return False

def get_subtokens_mean(emb_sentence, subtoken_indices):
    ind1, ind2 = subtoken_indices[0], subtoken_indices[1]
    m = []
    for i in range(ind1, ind2 + 1):
        m.append(emb_sentence[i])
    t = (np.mean(m, axis=0))
    return t

def train():
    config = load_config()
    input_sentences, inputs, input_labels = utl.prepare_inputs('train.json')
    test_sentences, tests, test_labels = utl.prepare_inputs('test.json')
    estimator, tokenizer = brt.load_bert_model_tf()


    _, brt_train_sentences = zip(*input_sentences)
    features = brt.create_examples(brt_train_sentences, tokenizer)
    train_embeddings = brt.get_embeddings(features, estimator)

    _, brt_test_sentences = zip(*input_sentences)
    test_features = brt.create_examples(brt_test_sentences, tokenizer)
    test_embeddings = brt.get_embeddings(test_features, estimator)

    ##### Classifier's inputs #####
    # AZP previous and next word embeddings (768*2), candidate embeddings and two features (768 + 2)
    azp_and_cand = tf.placeholder(tf.float32, (None, 1536 + 768 + 2))
    targets = tf.placeholder(tf.float32, (None, 2))
    training = tf.placeholder_with_default(False, shape=(), name='training_bool')

    #hyperparamter settings
    act           = config['activation']
    dropout_rate  = config['dropout_rate']
    learning_rate = config['learning_rate']
    decay_rate = 0.01

    with tf.name_scope('classifier_model'):
        initializer = tf.contrib.layers.xavier_initializer()
        dnn1 = tf.layers.dense(azp_and_cand, config['nhidden1'], activation=None, kernel_initializer=initializer)
        dnn1_act = act(dnn1)
        dnn1_drop = tf.layers.dropout(dnn1_act, dropout_rate, training=training)
        dnn2 = tf.layers.dense(dnn1_drop, config['nhidden2'], activation=None, kernel_initializer=initializer)
        dnn2_act = act(dnn2)
        dnn2_drop = tf.layers.dropout(dnn2_act, dropout_rate, training=training)
        dnn3 = tf.layers.dense(dnn2_drop, config['nhidden3'], activation=None, kernel_initializer=initializer)
        dnn3_act = act(dnn3)
        dnn3_drop = tf.layers.dropout(dnn3_act, dropout_rate, training=training)
        logits = tf.layers.dense(dnn3_drop, config['n_outputs'], name='output_logits')

    with tf.name_scope('loss_fucntion'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_sum(xentropy, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_optimizer')
        if (config['do_clipping']):
            threshold = 5.0
            grads_and_vars = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(capped_gvs)

        softmax_scores = tf.nn.softmax(logits)

    gconfig = tf.ConfigProto()
    gconfig.gpu_options.allow_growth = True

    with tf.Session(config=gconfig) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(config['epochs']):
            for (id, sentence), azp_candidates, labels in zip(input_sentences, inputs, input_labels):
                batch_inputs = []
                batch_labels = []
                print(id)
                print(azp_candidates)

                (azp, _), candidates = azp_candidates

                #we convert sentence tokens into tokenizer subtokens
                wordpiece_indices = brt.get_wordpiece_indices(sentence, tokenizer)

                azp_previous_word = azp - 1
                azp_next_word     = azp + 1

                azp_i = wordpiece_indices[azp_previous_word]  #subtoken first and last indices of AZP previous word
                azp_j = wordpiece_indices[azp_next_word]      #subtoken first and last indices of AZP next word

                for c, l in zip(candidates, labels):
                    c_i = wordpiece_indices[c[0]][0]     #first subtoken index of the candidate noun phrase
                    c_j = wordpiece_indices[c[1]][-1]    #last subtoken index of the candidate noun phrase
                    c_ind = (c_i, c_j)
                    same_sentence_feature = c[2]
                    find_distance_feature = c[3]

                    azp_cand_representation = np.concatenate([get_subtokens_mean(train_embeddings[id], azp_i),
                                                              get_subtokens_mean(train_embeddings[id], azp_j),
                                                              get_subtokens_mean(train_embeddings[id], c_ind),
                                                              [same_sentence_feature], [find_distance_feature]], axis=0)

                    batch_inputs.append(azp_cand_representation)
                    batch_labels.append(l)

                _, loss_, logits_ = sess.run([train_op, loss, logits], feed_dict={azp_and_cand: batch_inputs,
                                                                                  targets: batch_labels,
                                                                                  training: True})
                print(f"epoch:{e}\nlogits:\n{np.argmax(logits_, axis=1)}")
            learning_rate *= decay_rate

        ###############################################
        ##############  Test  ##########################
        ###############################################
        for (id, sentence), azp_candidates, labels in zip(test_sentences, tests, test_labels):
            batch_inputs = []
            batch_labels = []
            (azp, _), candidates = azp_candidates

            #we convert sentence tokens into tokenizer subtokens
            wordpiece_indices = brt.get_wordpiece_indices(sentence, tokenizer)

            azp_previous_word = azp - 1
            azp_next_word = azp + 1
            azp_i = wordpiece_indices[azp_previous_word]
            azp_j = wordpiece_indices[azp_next_word]

            for c, l in zip(candidates, labels):
                c_i = wordpiece_indices[c[0]][0]  # first subtoken index of the candidate noun phrase
                c_j = wordpiece_indices[c[1]][-1]  # last subtoken index of the candidate noun phrase
                c_ind = (c_i, c_j)
                same_sentence_feature = c[2]
                find_distance_feature = c[3]

                azp_cand_representation = np.concatenate([get_subtokens_mean(test_embeddings[id], azp_i),
                                                          get_subtokens_mean(test_embeddings[id], azp_j),
                                                          get_subtokens_mean(test_embeddings[id], c_ind),
                                                          [same_sentence_feature], [find_distance_feature]], axis=0)

                batch_inputs.append(azp_cand_representation)
                batch_labels.append(l)

            logits_, soft_scores = sess.run([logits, softmax_scores], feed_dict={azp_and_cand: batch_inputs})

            print(soft_scores)
            result = get_predict(batch_labels, soft_scores)

            print('TEST:')
            print(result)

train()
