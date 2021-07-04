from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torch.nn.utils.rnn import pad_sequence
# pad_sequence: 짧은 sentence같은 경우 같은 사이즈 Tensor로 맞추기 위해 사용
# torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08) # paper 그대로

SOS_token = 1
EOS_token = 2

################################
# 학습시키기 위해, 우리는 각각의 pair에서 input tensor(words들의 indexes) target tensor가 필요
# vector를 만드는 과정에서 EOS token을 두 sequences 모두에게 추가

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] # index 반환

def tensorsFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token) # 마지막에 추가
    return torch.tensor(indexes, dtype=torch.long).view(1, -1) # todo print

def tensorsFromPair(pair, input_lang, output_lang, max_input_length):
    input_tensor = tensorsFromSentence(input_lang, pair[0])
    target_tensor = tensorsFromSentence(output_lang, pair[1])

    with torch.no_grad():

        # 고정된 길이를 주기 위해 0으로 pad
        pad_input = nn.ConstantPad1d((0, max_input_length - input_tensor.size(1)), 0) # todo None일떄?
        pad_target = nn.ConstantPad1d((0, max_input_length - target_tensor.size(1)), 0)

        # padding 작업 실행
        input_tensor_padded = pad_input(input_tensor)
        target_tensor_padded = pad_target(target_tensor)
        # todo print shape

    pair_tensor = pad_sequence([input_tensor_padded, target_tensor_padded], batch_first=False, padding_value=0)
    # todo print shape

    return pair_tensor


#####################################################################
def reformat_tensor_mask(tensor):
    tensor = tensor.squeeze(dim=1)
    # squeeze는 1인 차원을 삭제, dim 설정 안해주면 1인 것 다 삭제해버림
    tensor = tensor.transpose(1, 0) # dim 1과 0 바꿈
    mask = tensor != 0 # 같은 shape의 tensor 생성하고 tensor value 중에 0이 아니면 True

    return tensor, mask


#######################################################################
# Tensor 에서 문장으로 바꾸기
def SentenceFromTensor_(lang, tensor):
    indexes = tensor.squeeze()
    indexes = indexes.tolist()
    return [lang.index2word[index] for index in indexes]



















