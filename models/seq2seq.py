import random
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""pack_padded_sequence(input, lengths, batch_first=False, enfoce_sorted=True)
https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
고정된 문장의 길이를 만들기 위해 <pad> 토큰 넣어줌 이는 쓸모없는 연산을 하게 됌
즉, 효율적으로 진행하기 위해 병렬처리를 하려고 함 방법은 정렬 후 하나의 통합된 배치로 만들어줌
: <pod>토큰을 계산 안하기 때문에 더 빠른 연산을 처리 할 수 있다.
"""

""" pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
packed sequence에 pad를 주는 작업
위의 작업에 반대로 되는 것같다. return으로 unpakced seq, unpacked len를 반환(위의 input, length)
"""



class Encoder(nn.Module):

    def __init__(self, args, input_dim, embed_dim, hidden_dim, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # one-hot encoding이 아닌 특정 차원(embed_dim)으로 mapping 함

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # INPUT: input, (h, c) => input(Length, batch, H_in), h(num_layers, batch, H_out), c(layers, batch, H_cell)
        # H = size 의미 hidden dim= 출력이 얼마나 나올지 결정
        # OUTPUT: output, (h,c) => output(length, batch, H_out), h(layer, batch, H_out), c(layers, batch, H_cell)
        # length : cell 을 몇개 할 지 결정

        self.dropout = nn.Dropout(dropout)

    # source를 input으로 받아 context vecotr로 반환
    def forward(self, source, hidden):
        # input = [source length=단어 개수, batch size]

        embedded = self.dropout(self.embedding(source).view(1, 1, -1))
        # todo print this

        output, (hidden, cell) = self.lstm(embedded, hidden) # hidden, cell default로 zero임

        # output [source length, batch size, hidden dim]

        # hidden [num layers, batch size, hidden dim]
        # cell [num layers, batch size, hiden dim]
        # [forward layer 0, forward layer 1, ..., forward layer n]   # todo ????

        # decoder 의 first layer 초기화 하기 위해 hidden cell return # todo ??
        return (hidden, cell)

    def initHidden(self):
        # todo ?? 어떻게 되는지 파악
        # todo 2개인 이유가 hidden , cell state 때문?
        encoder_state = [torch.zeros(self.n_layers, 1, self.hidden_dim).cuda(self.args.gpu),
                         torch.zeros(self.n_layers, 1, self.hidden_dim).cuda(self.args.gpu)]

        return encoder_state


class Decoder(nn.Module):

    def __init__(self, args, output_dim, embed_dim, hidden_dim, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.hidden_dim = hidden_dim
        self.n_layer = n_layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, target, hidden):
        # target = [batch size] # todo target 이 무엇인지, batch size만 있는 이유 다시 프린트
        # hidden = [num layers, batch size, hidden_dim]
        # cell = [num layers, batch size, hidden_dim]

        embedded = self.dropout(self.embedding(target).view(1, 1, -1))
        # [1, batch_size, embedded_dim]

        # 처음에 encoder 마지막 cell, hidden state를 가진 weights 초기화
        # 그다음으로 decoder의 이전 hidden cell state weight 사용
        output, (hidden, cell) = self.lstm(embedded, hidden) # todo 원래 코드랑 좀 다름 되는지 확인

        # output [1, batch_size, hidden_dim]
        # hidden [layers, batch_size, hidden_dim]
        # cell [layers, batch_size, hidden_dim]
        # [forward_layer_0, forward_layer_1, ..., forward_layer_n] # todo ???

        prediction = self.fc(output[0])
        # [batch_size, output_dim]

        # 현재 예측단어, 현재까지의 모든 단어의 정보, 현재까지의 모든단어의 정보 return
        return prediction, (hidden, cell)

    def initHidden(self):

        decoder_state = [torch.zeros(self.n_layer, 1, self.hidden_dim).cuda(self.args.gpu),
                         torch.zeros(self.n_layer, 1, self.hidden_dim).cuda(self.args.gpu)]

        return decoder_state


# encoder에 의해 만들어진 context vector는 decoder의 초기 hidden state로 써야하므로 혹시라도 dim이 맞지 않을 경우
# 이 method를 활용하여 맞게 해주는 작업(hidden 과 cell state)
# mismatch하는경우는 2개: 1) Encoder만 bidirection 일 경우 2) hidden size가 다를 경우
class Linear(nn.Module):
    def __init__(self, hidden_size_encoder, hidden_size_decoder): # todo 인자 무엇인지 파악
        super(Linear, self).__init__()
        num_directions = 1 # 한 방향이 default이고 1임
        self.connection_possibility_status = num_directions * hidden_size_encoder == hidden_size_decoder

        self.linear_connection = nn.Linear(num_directions * hidden_size_encoder, hidden_size_decoder)

    def forward(self, input):

        if self.connection_possibility_status: # Match 될 경우 그냥 return
            return input
        else:
            return self.linear_connection(input)




















