import json
import numpy as np


def preprocessing_arabic(text):
    '''
    Function to clean and prepare Arabic textss
    :param text:
    :return:
    '''
    # Typos in OntoNotes
    text = text.replace('ه`ذا', 'هذا')
    text = text.replace('ه`ذه', 'هذه')
    text = text.replace('ه`ذين', 'هذين')
    text = text.replace('الل`ه', 'الله')
    text = text.replace('ذ`لك', 'ذلك')
    text = text.replace('إل`ه', 'إله')
    text = text.replace('{', 'ا')
    text = text.replace('}', 'ا')

    #removing diactrics
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, "", text)
    #normalizing alif
    text = re.sub("[إأٱآا]", "ا", text)
    return text


def convert_to_one_hot_labels(labels):
    '''
    To convert into one hot labels
    :param labels:  labels of one dimension, for examle [0, 1, 0, 0, 1, 0]
    :return: one hot labels
    '''
    one_hot_labels = []
    for l in labels:
        one_hot = np.zeros(2)
        one_hot[l] = 1
        one_hot_labels.append(one_hot)
    return one_hot_labels


def get_candidates_indices_and_features(sent1, azp_ind, start_indices, end_indices):
    '''
    A function to extract candidates indices and features
    :param sent1: the first sentence, we want to extract its length to add it to the second sentence candidate indices
    :param start_indices: candidate start indices
    :param end_indices: candidate end indices
    :return: return (candidate first word index, candidadte last word index, same sentence feature, find distance feature)
    '''
    sent1_indices= zip(start_indices[0], end_indices[0])    #all sentence 1 candidates, their start and end indices
    sent2_indces = zip(start_indices[1], end_indices[1])    #all sentence 2 candidates, their start and end indices
    sent1_len = len(sent1.split())                          #sentence1 length, to make new indices for sentencee 2 candidates, if there is another sentence
    candidates_indices= []

    for i, j in sent1_indices:
        if (i == -1):
            break
        same_sentence = 1 if azp_ind[1]==0 else 0
        i = i + 1                     #accounting for  BERT[CLS] tag
        j = j + 1
        find_distance = azp_ind[0] - i
        candidates_indices.append((i, j, find_distance, same_sentence))

    for i, j in sent2_indces:
        if (i == -1):
            break
        i = i + sent1_len + 1 + 1       #shifting indices wrt to the first sentence length and accounting for [CLS] and [SEP] tags
        j = j + sent1_len + 1 + 1
        same_sentence = 1               #AZPs appear always appear in the second setenece, if there is one
        find_distance =  azp_ind[0] - i
        candidates_indices.append((i, j, same_sentence, find_distance))
    return candidates_indices

def get_azp_index(sent1, azp_ind):
    sentence_id = -1

    if(azp_ind[0] != -1):                      #if AZP in sentence 1
        azp_ind = azp_ind[0] + 1               #accounting for [CLS]
        sentence_id = 0
    else:
        azp_ind = (len(sent1.split())) + azp_ind[1] + 1 + 1 #the new index for the AZP, and accounting for [CLS] + [SEP]
        sentence_id = 1
    return (azp_ind, sentence_id)

def one_sentence_representation(sent1, sent2):
    if(sent2 != ""):
        sent = sent1 +' [SEP] '+sent2
        return sent
    else:
        return sent1

def prepare_inputs(path):
    inputs = []
    labels = []
    input_sentences = []
    ids = 0
    with open(path, encoding='utf-8') as json_file:
        data = json.load(json_file)
        for d in data['data']:
            sent1 = d['sentence1']
            sent2 = d['sentence2']
            sentence_input = one_sentence_representation(sent1, sent2)

            azp_ind = get_azp_index(sent1, d['azp_index'])
            candidates = get_candidates_indices_and_features(sentence_input,
                                                             azp_ind,
                                                             d['candidate_start_indices'],
                                                             d['candidate_end_indices'])

            one_hot_labels = convert_to_one_hot_labels(d['labels'])
            inputs.append([azp_ind, candidates])
            labels.append(one_hot_labels)
            input_sentences.append((ids, sentence_input))
            ids += 1
    return input_sentences, inputs, labels
