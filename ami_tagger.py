from collections import namedtuple, OrderedDict
from nltk.tokenize import word_tokenize
import nltk
import csv
import glob
import os
import sys

dataset = open('sentences.txt', encoding="utf8")
f_data = dataset.readlines()
train_data = f_data[0:40000]
test_data = f_data[40000:]

def get_utterances_from_file(data):
    return [_dict_to_dialog_utterance(du_dict.split('\t')) for du_dict in data]

def get_utterances_from_filename(data):
    return get_utterances_from_file(data)

def get_data(data):
    return get_utterances_from_filename(data)

DialogUtterance = namedtuple("DialogUtterance", ("act_tag", "speaker", "pos", "text"))

PosTag = namedtuple("PosTag", ("token", "pos"))

def _dict_to_dialog_utterance(value):
    speaker = value[0]
    text = value[2]
    act_tag = value[3].replace('\n', '')
    tokernized = word_tokenize(text)
    tags = nltk.pos_tag(tokernized)
    pos = ' '.join([f'{tag[0]}/{tag[1]}' for tag in tags])
    
    du_dict = OrderedDict()
    du_dict['act_tag'] = act_tag
    du_dict['speaker'] = speaker
    du_dict['pos'] = pos
    du_dict['text'] = f'{text} /'
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None

    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)


# ########## my code ###################

import pycrfsuite
from sklearn.metrics import accuracy_score
import string

def my_training(data):
    y_train = []
    X_train = []
    temp=[]
    count=0
    prev=None
    file = data
    temp.append("FIRST_UTTERANCE")
    for utt in file:
        # count+=1
        curr_speaker = utt.speaker
        y_train.append(utt.act_tag)
        if prev!=curr_speaker and prev!=None:
            temp.append("SPEAKER_CHANGED")
        if utt.pos:
            for p in utt.pos:
                token = "TOKEN_" + p[0]
                pos = "POS_" + p[1]
                temp.append(token)
                temp.append(pos)
        else:
            temp.append("NO_WORDS")
        X_train.append(temp)
        prev= curr_speaker
        temp=[]

    return X_train,y_train

def my_testing(data):
    y_train = []
    y_pred=[]
    X_train = []
    temp=[]
    count=0
    prev=None
    i=0
    tagger = pycrfsuite.Tagger()
    file = data
    temp.append("FIRST_UTTERANCE")
    for utt in file:
        # count+=1
        curr_speaker = utt.speaker
        y_train.append(utt.act_tag)
        if prev!=curr_speaker and prev!=None:
            temp.append("SPEAKER_CHANGED")
        if utt.pos:
            for p in utt.pos:
                token = "TOKEN_" + p[0]
                pos = "POS_" + p[1]
                temp.append(token)
                temp.append(pos)
        else:
            temp.append("NO_WORDS")
        X_train.append(temp)
        prev= curr_speaker
        temp=[]

    
    tagger.open('abc.crfsuite')
    y_pred += tagger.tag(X_train)
    y_pred.append("\n")
    i+=1
    X_train = []
    return y_pred, y_train

data_train = get_data(train_data)
X_train,y_train = my_training(data_train)
trainer = pycrfsuite.Trainer(verbose=False)
trainer.append(X_train, y_train)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('abc.crfsuite')

# Testing data
data_test = get_data(test_data)
y_pred, y_train = my_testing(data_test)
y_train_count = 0
correct = 0
f = open('output_ami.txt', "w" )
for item in y_pred:
    if item=="\n":
        f.write(item)
        continue
    f.write(item)
    f.write("\n")
    if item == y_train[y_train_count]:
        correct += 1
    y_train_count += 1

print(correct, y_train_count, (correct / y_train_count) * 100)