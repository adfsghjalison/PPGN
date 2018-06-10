import tensorflow as tf
import numpy as np
import sys
import collections

from utils import utils
from lib.ops import *
from lib.encoder import encoder
from lib.decoder import decoder
from lib.discrim import discrim

class sentiment_dialogue():
    
    def __init__(self,args,sess):
    
        # tf.Session
        self.sess = sess

        # test file
        self.test_file = args.test_file
        self.test_file_output = args.test_file_output
        self.utils = utils(args)
        self.encoder = encoder(args,self.sess,self.utils)
        self.decoder = decoder(args,self.sess,self.utils)
        self.discrim = discrim(args,self.sess,self.utils,self.decoder)
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.move_step = 400

    def test(self): 
        count = 0
        sentence_batch = []
        with open(self.test_file, 'r') as input_f:
            sentence = input_f.readline().strip().lower()
            while(sentence):
                count +=1
                sentence_batch.append(sentence) 
                if count == self.batch_size:
                    self.max_activation_batch(sentence_batch)
                    del sentence_batch[:]
                    count = 0
                sentence = input_f.readline().strip().lower()
            if count:
                self.max_activation_batch(sentence_batch)

    def stdin_test(self):
        sentence = 'hi'
        print(sentence)
        while(sentence):
            sentence = sys.stdin.readline().lower()
            sys.stdout.flush()
            sentence_batch = []
            sentence_batch.append(sentence)
            self.max_activation_batch(sentence_batch)

    def max_activation_batch(self,sentence_batch):
        sent_num = len(sentence_batch)
        output_batch = [0]*sent_num
        sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32) 
        for i,sentence in enumerate(sentence_batch):
            input_sent_vec = self.utils.sent2id(sentence)
            sent_vec[i] = input_sent_vec

        encode = self.encoder.encode(sent_vec)
        encoder_state_c = encode["encoder_state_c"]
        sampled_encoder_state_h = encode["sampled_encoder_state_h"]
        original_h_code = sampled_encoder_state_h
        queue = collections.deque()
        reward = self.discrim.decode_get_grad(encoder_state_c,sampled_encoder_state_h,original_h_code)
        queue.append(reward)
        # pred_sent = self.utils.id2sent(reward["pred"][0])

        current_h_code = original_h_code
        for i in range(self.move_step):
            senti_factor = [0.05/np.amax(np.absolute(g_h)) for g_h in queue[0]["grads_sampled_h"]]
            senti_factor_tile = np.tile(np.expand_dims(senti_factor,axis=1),current_h_code[-1].shape[-1])
            current_h_code = current_h_code + senti_factor_tile*queue[0]["grads_sampled_h"] - 10*queue[0]["l2_grads_sampled_h"]
            # print(current_h_code[0])
            # print(senti_factor)
            rewardI = self.discrim.decode_get_grad(encoder_state_c,current_h_code,original_h_code)
            queue.append(rewardI)
            for j in range(sent_num):
                if queue[1]["score"][j]-queue[0]["score"][j] < -0.01 and isinstance(output_batch[j],list) == 0:
                    output_batch[j] = queue[0]["pred"][j]                
            queue.popleft() 

        for k in range(sent_num):
            if isinstance(output_batch[k],list) == 0:
                output_batch[k] = queue[0]["pred"][k]
            pred_sent_origin = self.utils.id2sent(reward["pred"][k])
            pred_sent = self.utils.id2sent(output_batch[k])
            # print(pred_sent_origin)
            # print(reward["score"][k])
            print(pred_sent)
            # print(queue[0]["score"][k])
