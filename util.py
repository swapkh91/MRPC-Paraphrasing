#!/usr/bin/env python
# coding: utf-8

import json
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from pandas import DataFrame
import gensim
import itertools
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

weights = json.load(open("ML_Model/weights.json"))

def text_to_word_list(text):
    # Pre process and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def make_w2v_embeddings(df, embedding_dim=50):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Stopwords
    stops = set(stopwords.words('english'))

    # Load word2vec
    word2vec = gensim.models.word2vec.Word2Vec.load("ML_Model/mrpc.w2v").wv

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both sentences of the row
        for sentence in ['Sentence1', 'Sentence2']:

            s2n = []
            for word in text_to_word_list(row[sentence]):
                # Check for unwanted words
                if word in stops:
                    continue

                # If a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    s2n.append(vocabs_cnt)
                else:
                    s2n.append(vocabs[word])

            # Append sentence as number representation
            df.at[index, sentence + '_n'] = s2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings

def split_and_zero_padding(df, max_seq_length):
    # Split to dicts
    X = {'left': df['Sentence1_n'], 'right': df['Sentence2_n']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

def word_shares(row):
    stops = set(stopwords.words('english'))
    q1 = set(str(row['Sentence1']).lower().split())
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0'

    q2 = set(str(row['Sentence2']).lower().split())
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0'

    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words)) #count share
    R31 = len(q1stops) / len(q1words) #stops in q1
    R32 = len(q2stops) / len(q2words) #stops in q2
    return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)

def get_xgb_features(df):
    x = DataFrame()

    x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
    x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

    x['len_s1'] = df['Sentence1'].apply(lambda x: len(str(x)))
    x['len_s2'] = df['Sentence2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_s1'] - x['len_s2']

    x['len_char_s1'] = df['Sentence1'].apply(lambda x: len(str(x).replace(' ', '')))
    x['len_char_s2'] = df['Sentence2'].apply(lambda x: len(str(x).replace(' ', '')))
    x['diff_len_char'] = x['len_char_s1'] - x['len_char_s2']

    x['len_word_s1'] = df['Sentence1'].apply(lambda x: len(str(x).split()))
    x['len_word_s2'] = df['Sentence2'].apply(lambda x: len(str(x).split()))
    x['diff_len_word'] = x['len_word_s1'] - x['len_word_s2']

    x['avg_world_len1'] = x['len_char_s1'] / x['len_word_s1']
    x['avg_world_len2'] = x['len_char_s2'] / x['len_word_s2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

    x['exactly_same'] = (df['Sentence1'] == df['Sentence2']).astype(int)
    x['duplicated'] = df.duplicated(['Sentence1','Sentence2']).astype(int)
    
    return x
