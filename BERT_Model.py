import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
sys.path.insert(1, 'bert_fn')
import modeling, tokenization
import extract_features
import psutil, os


MAXLENGTH = 512

def load_bert_model_tf(layer_ind=10):
    '''
    Function to load BERT feature extraction and its Tokenizer
    :param layer_ind: defines which of BERT layers index to use for extracting embeddings
    :return:
    '''
    BERT_DIR = 'C:\datasets\embeddings/multi_cased_L-12_H-768_A-12/'
    VOCAB_FILE = 'C:\datasets\embeddings\multi_cased_L-12_H-768_A-12/vocab.txt'
    CONFIG_FILE = 'C:\datasets\embeddings\multi_cased_L-12_H-768_A-12/bert_config.json'
    INIT_CHECKPOINT = os.path.join(BERT_DIR, 'bert_model.ckpt')
    BERT_CONFIG = modeling.BertConfig.from_json_file(CONFIG_FILE)

    tokenization.validate_case_matches_checkpoint(False, INIT_CHECKPOINT)
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

    model_fn = extract_features.model_fn_builder(
        bert_config=BERT_CONFIG,
        init_checkpoint=INIT_CHECKPOINT,
        layer_indexes=[layer_ind],
        use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
        use_one_hot_embeddings=False)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=None,
            per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=1)
    return estimator, tokenizer

def get_wordpiece_indices(text, tokenizer):
    '''
    Function to convert from sentence indices to  Wordpiece indices. For example:
        My sweetheart is sleeping          ==>   (My, 0), (sweetheart, 1), (is, 2), (sleeping, 3)
        My sweet ##heart is sleep ##ing    ==>   (My, (0,0)), (sweet ##heart, (1, 2)), (is, (3, 3)), (sleep##ing, (4, 5))
    :param text: the sentence we want to get its wordpiece tokens
    :param tokenizer: tokenizer to help find wordpiece indices
    :return:
    '''
    wordpiece_indices = []
    i_ind = 0
    j_ind = 0

    text_tokens = ['[CLS]'] + text.split() + ['[SEP]']
    tf_sen = []
    for i, x in enumerate(text_tokens):
        tmp = tokenizer.tokenize(x)
        tf_sen.extend(tmp)
        j_ind = i_ind + len(tmp)
        wordpiece_indices.append((i_ind, j_ind - 1))
        i_ind = j_ind
    return wordpiece_indices #, tf_sen


def _create_example(inputs):
    examples = []
    unique_id = 0
    for i, inp in enumerate(inputs):
        examples.append(extract_features.InputExample(unique_id=unique_id, text_a=inp, text_b=None))
        unique_id += 1
    return examples

def create_examples(sents, tokenizer):
    inputs = _create_example(sents)
    features = extract_features.convert_examples_to_features(examples=inputs, seq_length=MAXLENGTH, tokenizer=tokenizer)
    return features


def get_embeddings(features, estimator):
    results = []
    counter = 0
    input_fn = extract_features.input_fn_builder(features=features, seq_length=MAXLENGTH)
    for result in estimator.predict(input_fn, yield_single_examples=True):
        res = result['layer_output_0']
        results.append(res)
        counter += 1
    return results

