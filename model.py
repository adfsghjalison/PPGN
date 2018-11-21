#--coding:utf-8--
import tensorflow as tf
import numpy as np
import os, sys, random
import collections, csv

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
        self.test_file = os.path.join(args.data_dir, 'source_test_chatbot')
        self.test_output = args.out
        self.utils = utils(args)
        self.encoder = encoder(args,self.sess,self.utils)
        self.decoder = decoder(args,self.sess,self.utils)
        self.discrim = discrim(args,self.sess,self.utils,self.decoder)
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.move_step = 50
        self.pos = args.pos
        self.g = args.g
        self.l2 = args.l2

    def test(self):


        print('g: {}    l2: {}    pos: {}'.format(self.g, self.l2, self.pos))

        count = 0
        con_batch = []
        sentence_batch = []
        cf = open(self.test_output, 'w')
        writer = csv.writer(cf, delimiter='|')
        writer.writerow(['context', 'utterance'])
        with open(self.test_file, 'r') as input_f:
            sentence = input_f.readline()
            while(sentence):
                con, sentence = sentence.strip().split(' +++$+++ ')
                count += 1
                con_batch.append(''.join(con.split()))
                sentence_batch.append(sentence)
                if count == self.batch_size:
                    self.max_activation_batch(con_batch, sentence_batch, writer)
                    del con_batch[:]
                    del sentence_batch[:]
                    count = 0
                #if count > 10 :
                #    break
                sentence = input_f.readline()
            if count:
                self.max_activation_batch(con_batch, sentence_batch, writer)
        cf.close()

    def stdin_test(self):
        sentence = 'Hi~'
        print(sentence)
        while(sentence):
            sentence = sys.stdin.readline().lower()
            sys.stdout.flush()
            sentence_batch = []
            sentence_batch.append(sentence)
            self.max_activation_batch(sentence_batch)

    def max_activation_batch(self, con_batch, sentence_batch, w):
        sent_num = len(sentence_batch)
        output_batch = [0]*sent_num
        sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32) 
        for i, sentence in enumerate(sentence_batch):
            input_sent_vec = self.utils.sent2id(sentence)
            sent_vec[i] = input_sent_vec

        encode = self.encoder.encode(sent_vec)
        encoder_state_c = encode["encoder_state_c"]
        sampled_encoder_state_h = encode["sampled_encoder_state_h"]
        original_h_code = sampled_encoder_state_h

        queue = collections.deque()
        reward = self.discrim.decode_get_grad(encoder_state_c,sampled_encoder_state_h,original_h_code)
        queue.append(reward)

        current_h_code = original_h_code
        ori_score = reward['score']
        
        for i in range(self.move_step):
            senti_factor = [0.01/(np.max(np.absolute(g_h)) + 10**-20) for g_h in queue[0]["grads_sampled_h"]]
            senti_factor_tile = np.tile(np.expand_dims(senti_factor,axis=1),current_h_code[-1].shape[-1])
            if self.pos:
              current_h_code = current_h_code + self.g * senti_factor_tile*queue[0]["grads_sampled_h"] - self.l2 * queue[0]["l2_grads_sampled_h"]
            else:
              current_h_code = current_h_code - self.g * senti_factor_tile*queue[0]["grads_sampled_h"] - self.l2 * queue[0]["l2_grads_sampled_h"]
            
            #print(current_h_code[0][:5])

            rewardI = self.discrim.decode_get_grad(encoder_state_c,current_h_code,original_h_code)
            queue.append(rewardI)
            for j in range(sent_num):
                #print ("step {} , {} score : {}".format(i, j, queue[1]["score"][j]))
                if queue[1]["score"][j] > queue[0]["score"][j]:
                    output_batch[j] = queue[1]["pred"][j]                
            queue.popleft() 
        
        for k in range(sent_num):
            if isinstance(output_batch[k],list) == 0:
                output_batch[k] = queue[0]["pred"][k]
            """
            print('original :')
            print(sentence_batch[k])
            print('pred :')
            print(self.utils.id2sent(output_batch[k]))
            print("")
            """
            #print("score : {} -> {}".format(ori_score[k], queue[0]['score'][k]))
            w.writerow([con_batch[k], self.utils.id2sent(output_batch[k]).encode('utf8')])

