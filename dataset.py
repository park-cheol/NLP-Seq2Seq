from __future__ import unicode_literals, print_function, division
# 파이썬 2: print 함수를 statement 파이썬3: function으로 간주 이러한 차이
# __future__ 사용하여 불가피하게 구버전 사용 시 import 한 것은 최신 버전을 사용 할 수있도록 함
import spacy
# spacy: 문장의 토큰화, 태깅(tagging)등 전처리 기능을 제공 (but 한국어는 지원안함)
from io import open
# 파일을 열 때 사용
import unicodedata
# 모든 유니코드 문자에 대한 문자 속성을 정의하는 유니코드 문제 데이터베이스(UCD)에 대한 액세스 제공
import string
import re
# 정규 표현식을 지원하기 위해 사용 (regular expression)
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# todo 왜 넣어줘야 하는지
# 기존 pytorch 에서는 0, 1이지만 여기서는 sos=1 : zero reserved를 가지기 위해서 <pad>
SOS_token = 1 # start of sequence 시작을 알리는 신호
EOS_token = 2 # end of sequence 끝을 알리는 신호

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # key: word / value: index {"word": index}
        self.word2count = {} # rare word에 대체할 때 사용할 각 단어의 빈도 수 측정

        self.index2word = {0: "<pad>", SOS_token: "SOS", EOS_token: "EOS"} # key: index / value: word
        self.n_words = 3 # <pad>, SOS, EOS 포함함

    def addSentence(self, sentence):
        for word in sentence.split(' '): # 문장이 들어오면 그 문장을 공백을 기준으로 나눔
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: # word2index에 없는 word일 경우
            self.word2index[word] = self.n_words # {word: n_words} 가 됨
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

        else:
            self.word2count[word] += 1

#######################################
# 파일들은 모두 unicode 이므로 간단히 하기 위해 unicode로 변환
# 단어들을 ASCII로 변환, 영어는 소문자로, 그리고 문장부호들은 자르기

# http://stackoverflow.com/a/518232/2809427
# Unicode -> ASCII
# ASCII(미국정보교환표준부호): 영문 알파벳을 사용하는 대표적인 문자 인코딩
# 유니코드: ASCII도 여기에 포함, 전세계모든 문자를 컴퓨터에서 일관되게 표현하고 다룰 수 있도록 설계된 산업 표준


# word를 입력받아 발음구별기호(액센트), 알파벳으로 분리하고 알파벳만 return
# Mn: 엑센트 카테고리
def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) # word를 발음구별기호와 알파벳으로 분리함 'NFD'
        if unicodedata.category(c) != 'Mn' # Mn이 아닌 즉, 알파벳만 True
    )
# NFD, NFC등이 있는데 윈도우와 맥의 파일 이름 저장 방식

# 소문자, trim and remove non-letter characters
def normalizeString(s):
    s = unicode2Ascii(s.lower().strip()) # 소문자로 만들고 왼쪽 오른쪽 공백 제거
    # r spring: \n 같은것을 문자로 받아들이지 않고 그대로 찍음 (raw spring)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[.!?]", '', s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) : 영어 알파벳 제외 다 없애버림 한국어 이기때문에 삭제
    # re.sub('패턴', 교체함수, '문자열', 바꿀횟수)
    return s

print(normalizeString("ㅁAA"))

################################################33
# 파일을 line으로 나누고 또 line을 나누어서 pair을 맞춤
# reverse: english -> other lang 인걸 other lang -> english
def readLangs(lang1, lang2,auto_encoder=False, reverse=False):
    print("Reading Lines......")

    # 파일을 읽고 라인별로 쪼개기
    # utf-8: 유니크도를 위한 가변 길이 문자 인코딩 방식 중 하나
    lines = open('data/kor.txt', encoding='utf-8').read().strip().split('\n')

    # pair 맞추고 normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines] # \t: tap 간격
    # 각 줄을 받아서 우리 data양식에 따라 \t로 나누고 normalize

    pairs = [[pair[1], pair[0]] for pair in pairs]
    # kor.txt 데이터에 뒤쪽에 불필요한 문자가 있어서 추가적으로 사용
    # 그리고 한국어-> 영어로 바꾸고 싶어서 위치도 바꿔줌

    # AutoEncoder는 output으로 같은 data를 가짐
    if auto_encoder:
        pairs = [[pair[0], pair[0]] for pair in pairs]

    # Revise pairs, make Lang instacnes
    # pairs 반대로
    if reverse:
        pairs = [list(reversed(p)) for p in pairs] # 그냥 pair[0], pair[1] 반대로

        # Lang: word2idx / idx2word 등 선언하고 단어 추가하는 클래스
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)

    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################
""" 불필요하고 안좋아 질 것 같아 임시로 제거(테스트 시 더 좋으면 사용할 예정)
"""

# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#
#
# MAX_LENGTH = 10
#
# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )
#
#
# def filterPair(p, max_input_length):
#     return len(p[0].split(' ')) < max_input_length and \
#            len(p[1].split(' ')) < max_input_length and \
#            p[1].startswith(eng_prefixes)
#
# def filterPairs(pairs, max_input_length):
#     pairs = [pair for pair in pairs if filterPair(pair, max_input_length)]
#     return pairs

########################################################
# 데어터 전치리 과정
# 1. 파일 읽고 -> line 으로 쪼개고 -> pair 맞추기
# 2. normalize 하고 filter는 안함
# 3. pairs 문장으로부터 단어리스트 생성

def prepareData(lang1, lang2, auto_encoder=False, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, auto_encoder, reverse)
    print("총 sentence pairs 개수: %s" % len(pairs))


    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("총 단어들 수:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


########################################################3
# Dataset

# auto encoder란: https://bab2min.tistory.com/628 input과 동일한 출력을 내도록 하는 것

class Dataset():

    def __init__(self, phase, num_embeddings=None, max_input_length=None, transform=None, auto_encoder=False, reverse=False):
        """
        phase: train/test 나눔
        num_embedding: embedding dim
        max_input_length: input의 고정된 길이
        transform: 필요하면 전처리 사용
        auto_encoder: bool 사용 할지 말지
        """

        if auto_encoder:
            lang_in = 'eng'
            lang_out = 'eng'
        else:
            lang_in = 'kor'
            lang_out = 'eng'

        input_lang, output_lang, pairs = prepareData(lang_in, lang_out, auto_encoder, reverse)
        # todo pairs 한번찍어보기

        # random list
        random.shuffle(pairs)

        if phase == 'train':
            selected_pairs = pairs[0:int(0.8 * len(pairs))] # 앞에서 80퍼까지만 train으로 사용

        else:
            selected_pairs = pairs[int(0.8 * len(pairs)):] # 나머지 20퍼는 test

        # Tensor 얻기
        selected_pairs_tensors = [tensorsFromPair(selected_pairs[i], input_lang, output_lang, max_input_length)
                                  for i in range(len(selected_pairs))]
        # todo print
        # 총 selected_pairs 순서대로 넣어줌

        self.transform = transform
        self.num_embeddings = num_embeddings
        self.max_input_length = max_input_length
        self.data = selected_pairs_tensors
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.phase = phase

    def langs(self):
        return self.input_lang, self.output_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        if self.phase == 'test':
            random.shuffle(self.data)

        sample = {'sentence': pair}
        # todo print this

        if self.transform:
            sample = self.transform(sample)


        return sample




























