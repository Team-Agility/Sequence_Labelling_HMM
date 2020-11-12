from collections import namedtuple
import csv
import glob
import os
import sys

def get_utterances_from_file(dialog_csv_file):
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]

def get_utterances_from_filename(dialog_csv_filename):
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_data(data_dir):
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)

DialogUtterance = namedtuple(
    "DialogUtterance", ("act_tag", "speaker", "pos", "text"))

PosTag = namedtuple("PosTag", ("token", "pos"))


def _dict_to_dialog_utterance(du_dict):
   
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
    for file in data:
        temp.append("FIRST_UTTERANCE")
        for utt in file:
            # count+=1
            curr_speaker = utt.speaker
            y_train.append(utt.act_tag)
            if prev!=curr_speaker and prev!=None:
                temp.append("SPEAKER_CHANGED")
            if utt.pos:
                for p in utt.pos:
                    if p[0]== ",":
                        continue
                    if p[0]== "?":
                        temp.append("QUESTION") 
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
    for file in data:
        temp.append("FIRST_UTTERANCE")
        for utt in file:
            # count+=1
            curr_speaker = utt.speaker
            y_train.append(utt.act_tag)
            if prev!=curr_speaker and prev!=None:
                temp.append("SPEAKER_CHANGED")
            if utt.pos:
                for p in utt.pos:
                    if p[0]== ",":
                        continue
                    if p[0]== "?":
                        temp.append("QUESTION") 
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

location = sys.argv[1]
data_train = get_data(location)

X_train,y_train = my_training(data_train)
trainer = pycrfsuite.Trainer(verbose=False)
trainer.append(X_train, y_train)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('abc.crfsuite')

# Testing data
location = sys.argv[2]
data_test = get_data(location)
y_pred, y_train = my_testing(data_test)
f = open(sys.argv[3], "w" )
y_train_count = 0
correct = 0
for item in y_pred:
    if item=="\n":
        f.write(item)
        continue
    f.write(item)
    f.write("\n")
    if item == y_train[y_train_count]:
        correct += 1
    y_train_count += 1

print(correct, y_train_count)