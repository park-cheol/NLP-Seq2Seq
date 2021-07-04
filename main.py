from __future__ import unicode_literals, print_function, division
import argparse
import unicodedata
import string
import re
import random
import os
import warnings
from io import open
import time
import datetime


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchtext.data.metrics import bleu_score

from dataset import Dataset
from utils import *
from models.seq2seq import *

parser = argparse.ArgumentParser(description='Creating Classifier')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs-per-lr-drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

parser.add_argument('--epochs', default=100, type=int, help='Number of epochs for training for training')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--auto-encoder', default=False, type=bool, help='Use auto-encoder model')
parser.add_argument('--MAX-LENGTH', default=10, type=int, help='Maximum length of sentence')
parser.add_argument('--hidden-size-encoder', default=256, type=int, help='Eecoder Hidden Size')
parser.add_argument('--num-layer-encoder', default=1, type=int, help='Number of LSTM layers for encoder')
parser.add_argument('--hidden-size-decoder', default=256, type=int, help='Decoder Hidden Size')
parser.add_argument('--num-layer-decoder', default=1, type=int, help='Number of LSTM layers for decoder')
parser.add_argument('--teacher-forcing', default=True, type=bool, help='If using the teacher frocing in decoder')

# training
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Data 불러오기
    train_dataset = Dataset(phase='train', max_input_length=10, auto_encoder=args.auto_encoder)
    test_dataset = Dataset(phase='test', max_input_length=10, auto_encoder=args.auto_encoder)

    # lang 속성 추출
    input_lang, output_lang = train_dataset.langs()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,

                                              shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    # model 정의
    encoder = Encoder(args, input_lang.n_words, args.hidden_size_encoder, args.hidden_size_encoder, args.num_layer_encoder)
    bridge = Linear(args.hidden_size_encoder, args.hidden_size_decoder)
    decoder = Decoder(args, output_lang.n_words, args.hidden_size_decoder, args.hidden_size_decoder, args.num_layer_decoder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        encoder = encoder.cuda(args.gpu)
        bridge = bridge.cuda(args.gpu)
        decoder = decoder.cuda(args.gpu)

    else:
        encoder = nn.DataParallel(encoder).cuda()
        bridge = nn.DataParallel(bridge).cuda()
        decoder = nn.DataParallel(decoder).cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr)
    bridge_optimizer = torch.optim.SGD(bridge.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.lr)

    # 초기화
    encoder.apply(init_weights)
    bridge.apply(init_weights)
    decoder.apply(init_weights)

    start = time.time()
    for epoch in range(args.epochs):
        print("Training")
        train(args, epoch, encoder, bridge, decoder, criterion, encoder_optimizer, bridge_optimizer,
              decoder_optimizer, train_loader, input_lang, output_lang)

        print("Testing")
        # evaluateRandomly(args, encoder, decoder, bridge, input_lang, output_lang, testset, n=10):
        evaluateRandomly(args, encoder, decoder, bridge, input_lang, output_lang, test_dataset)

        torch.save(encoder.state_dict(), "saved_models/encoder_%d.pth" % (epoch+1))
        torch.save(bridge.state_dict(), "saved_models/bridge_%d.pth" % (epoch+1))
        torch.save(decoder.state_dict(), "saved_models/dncoder_%d.pth" % (epoch+1))


    print("총 걸리시간: ", datetime.timedelta(seconds=time.time() - start))



def train(args, epoch, encoder, bridge, decoder, criterion, encoder_optimizer, bridge_optimizer,
              decoder_optimizer, trainloader, input_lang, output_lang):
    start = time.time()

    for i, data in enumerate(trainloader): # 1은 시작하는 index 번호
        # todo print i ,data
        # get a batch
        training_pair = data

        # input
        input_tensor = training_pair['sentence'][:, :, 0, :].cuda(args.gpu) # todo print 각 자리 의미
        input_tensor, mask_input = reformat_tensor_mask(input_tensor)

        # input의 1인 차원을 삭제하고 dim=0,1 을 transpose
        # mask는 input과 같은 shape이고 bool value로 되어있음 (0이 아니면 True)

        # target
        target_tensor = training_pair['sentence'][:, :, 1, :].cuda(args.gpu)
        target_tensor, mask_target = reformat_tensor_mask(target_tensor)

        input_tensor.cuda(args.gpu, non_blocking=True)
        target_tensor.cuda(args.gpu, non_blocking=True)

        # input sentence 는 encoder에게 주어지고 decoder 첫번째 input으로는 <sos>가 주어짐
        # encoder의 마지막 hidden state,cell은 decoder의 첫번째 hidden state 역할을 함

        # optimizer
        encoder_optimizer.zero_grad()
        bridge_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_hiddens_last = []
        loss = 0

        #############
        # Encoder
        #############
        for step_idx in range(args.batch_size):
            encoder_hidden = encoder.initHidden()
            # decoder_state = [torch.zeros(self.n_layer, 1, self.hidden_dim).cuda(self.args.gpu),
            #                 torch.zeros(self.n_layer, 1, self.hidden_dim).cuda(self.args.gpu)]
            input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]
            # 0은 remove 하라는 의미
            input_length = input_tensor_step.size(0)

            for ei in range(input_length):
                encoder_hidden = encoder(input_tensor_step[ei], encoder_hidden)

            last_hidden, last_cell = encoder_hidden
            hidden_last_layer = last_hidden[-1].view(1, 1, -1)
            cell_last_layer = last_cell[-1].view(1, 1, -1)
            encoder_hidden = [hidden_last_layer, cell_last_layer]

            encoder_hidden = [bridge(item) for item in encoder_hidden] # shape 맞춰주기
            encoder_hiddens_last.append(encoder_hidden)

        #############
        # Decoder
        #############
        decoder_input = torch.tensor([SOS_token]).cuda(args.gpu)
        decoder_hiddens = encoder_hiddens_last

        # teacher_forcing: decoder의 다음 input으로 예측값이 아닌 GT를 대신 넣어줌
        # 초기에 잘못된 예측이 연속적으로 학습이 잘못되어 질 수 있으므로
        if args.teacher_forcing:

            for step_idx in range(args.batch_size):
                target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
                target_length = target_tensor_step.size(0)

                decoder_hidden = decoder_hiddens[step_idx] # decoder 최신꺼 저장

                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden
                    )

                    loss += criterion(decoder_output, target_tensor_step[di].view(1))
                    decoder_input = target_tensor_step[di] # teacher_focing 부분

            loss = loss / args.batch_size

        else:

            for step_idx in range(args.batch_size):
                target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
                target_length = target_tensor_step.size(0)

                decoder_hidden = decoder_hiddens[step_idx]

                for di in range(target_length):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden
                    )

                    top_value, top_index = decoder_output.topk(1)
                    decoder_input = top_index.squeeze().detach()

                    loss += criterion(decoder_output, target_tensor_step[di].view(1))

                    if decoder_input.item() == EOS_token:
                        break

            loss = loss / args.batch_size

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if i % args.print_freq == 0:
            print("[Epoch %d/%d] [Iter %d/%d] [Loss: %f]" % (epoch, args.epochs, i, len(trainloader), loss))
    print("%d epoch finish" % (epoch), "time: ", datetime.timedelta(seconds=time.time() - start))

###################
# Evaluate(test)
###################

def evaluate(args, encoder, decoder, bridge, input_tensor, output_lang):
    max_length = args.MAX_LENGTH # default 10

    with torch.no_grad():
        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        hidden, cell = encoder_hidden # (hidden, cell)
        encoder_hn_last_layer = hidden[-1].view(1, 1, -1)
        encoder_cn_last_layer = cell[-1].view(1, 1, -1)
        encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        decoder_input = torch.tensor([SOS_token]).cuda(args.gpu)  # SOS
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            top_value, top_index = decoder_output.data.topk(1)

            if top_index.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[top_index.item()])

            decoder_input = top_index.squeeze().detach()

        return decoded_words

def evaluateRandomly(args, encoder, decoder, bridge, input_lang, output_lang, testset, n=5):
    for i in range(n):
        pair = testset[i]['sentence']
        input_tensor, mask_input = reformat_tensor_mask(pair[:, 0, :].view(1, 1, -1))
        input_tensor = input_tensor[input_tensor != 0]
        # 0이 아닌 value들만 모이고 [h*w*c]인 1차원 tensor로 변경
        output_tensor, mask_output = reformat_tensor_mask(pair[:,1,:].view(1,1,-1))
        output_tensor = output_tensor[output_tensor != 0]

        input_tensor = input_tensor.cuda(args.gpu)
        output_tensor = output_tensor.cuda(args.gpu)

        input_sentence = ' '.join(SentenceFromTensor_(input_lang, input_tensor))
        # ' '.join(str): 각 str을 모아주고 그 사이마다 ' ' 추가
        output_sentence = ' '.join(SentenceFromTensor_(output_lang, output_tensor))

        print("Input: ", input_sentence)
        print("GT: ", output_sentence)

        output_words = evaluate(args, encoder, decoder, bridge, input_tensor, output_lang)
        # todo print words
        output_sentence = ' '.join(output_words)
        print("Predic: ", output_sentence)
        print(' ')





if __name__ == '__main__':
    main()

















