import sys
import os
import json
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import random
import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB


def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def get_lang_dict():
    lang_dict = {'kn':0,'pt':1,'mr':2,'sv':3,'hi':4,'en':5,'es':6,'ta':7,'fr':8,'ml':9,'bn':10,'de':11,'it':12}
    return lang_dict


def preprocess_text_char(text):
    text = text.lower()
    emoji_pattern = re.compile("["
                             u"\U00002500-\U00002BEF"
                             u"\U0001F300-\U0001F5FF"
                             u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern,'',text)
    text = re.sub('<+[^<]+>+',' ',text)
    text = re.sub('[^\[]+\[+[^\[]+\]+',' ',text)
    text = re.sub(r'URL=\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    #text = re.sub(r'\\[^\\]*', '', text)
    text = text.replace('-', ' ').replace('–',' ')
    text = re.sub(r'\d+', '', text)
    punctuations = string.punctuation
    text = ''.join(char for char in text if char not in punctuations)
    text = ' '.join(text.split())
    text = text.replace(' ','_')
    return text



def get_lang_from_label(label):
    label_to_lang = {0:'kn',1:'pt',2:'mr',3:'sv',4:'hi',5:'en',6:'es',7:'ta',8:'fr',9:'ml',10:'bn',11:'de',12:'it'}
    lang_label = []
    for i in range(len(label)):
        lang_label.append(label_to_lang[label[i]])
    return lang_label

def create_train_dataset(data,path_to_save):
    lang_dict = get_lang_dict()
    text = []
    label = []
    for i in range(len(data)):
        text.append(preprocess_text_char(data[i]['text'])) 
        label.append(data[i]['langid'])
    vectorizer = CountVectorizer(analyzer = 'char',ngram_range = (5,6))
    text = vectorizer.fit_transform(text)
    joblib.dump(vectorizer, os.path.join(path_to_save,'vectorizer.pkl'))
    for i in range(len(label)):
        label[i] = lang_dict[label[i]]
    return text,label

def create_test_dataset(data,path_to_save):
    text = []
    for i in range(len(data)):
        text.append(preprocess_text_char(data[i]['text'])) 
    vectorizer = joblib.load(os.path.join(path_to_save,'vectorizer.pkl'))
    text = vectorizer.transform(text)
    return text



def compute_micro_f1_score(preds, golds):
    TP, FP, FN = 0, 0, 0

    assert len(preds) == len(golds)

    for pp, gg in zip(preds, golds):
        if gg == 'en':
            if pp != gg:
                FP += 1
                FN += 1
        else:
            if pp != gg:
                FP += 1
                FN += 1
            else:
                TP += 1

    prec = TP / (TP + FP)
    rec =  TP / (TP + FN)
    f1 = (2 * prec * rec) / (prec + rec)

    return f1


def compute_macro_f1_score(preds, golds):
    total = 0
    all_langs = list(set(golds))
    all_langs.remove('en')

    for ln in all_langs:
        TP, FP, FN = 0, 0, 0
        for pp, gg in zip(preds, golds):
            if (gg == ln) and (pp == gg):
                TP += 1
            elif (gg == ln) and (pp != gg):
                FN += 1
            elif (pp == ln) and (pp != gg):
                FP += 1

        prec = TP / (TP + FP)
        rec =  TP / (TP + FN)
        f1 = (2 * prec * rec) / (prec + rec)

        total += f1

    return total / (len(all_langs))

key = sys.argv[1]

if key == "train":
    path_to_data = sys.argv[2]
    path_to_save = sys.argv[3]
    train_fp = os.path.join(path_to_data,'train.json')
    valid1_fp = os.path.join(path_to_data,'valid.json')
    valid2_fp = os.path.join(path_to_data,'valid_new.json')
    train_data = read_json(train_fp)
    val1_data = read_json(valid1_fp)
    val2_data = read_json(valid2_fp)
    data = train_data+val1_data+val2_data
    train_text,train_label = create_train_dataset(data,path_to_save)
    model = MultinomialNB(alpha=0.00001)
    model.fit(train_text,train_label)
    joblib.dump(model, os.path.join(path_to_save,'model.joblib'))
    
else:
    path_to_test = sys.argv[3]
    path_to_save = sys.argv[2]
    test_data = read_json(path_to_test)
    test_text = create_test_dataset(test_data,path_to_save)
    model = joblib.load(os.path.join(path_to_save,'model.joblib'))
    test_pred = model.predict(test_text)
    test_pred = get_lang_from_label(test_pred)
    with open("output.txt", "w") as file:
        for i, item in enumerate(test_pred):
            file.write(item)
            if i < len(test_pred) - 1:
                file.write('\n')