import tensorflow as tf
import numpy as np
from lib.ops import *

class encoder():
    
    def __init__(self,args,sess,utils):

        self.sess = sess

        self.model_dir = args.model_dir
        self.sequence_length = args.sequence_length
        self.word_embedding_dim = 300
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.lstm_length = [self.sequence_length+1]*self.batch_size
        self.utils = utils
        self.vocab_size = len(self.utils.word_id_dict)


        self.build_graph()
        self.sample()
        self.saver = tf.train.Saver(var_list={v.op.name : v for v in self.get_var_list()})
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    def get_word_embedding(self):
        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("embedding") as scope:
            word_vector_BOS_EOS = tf.get_variable(
                name="word_vector_BOS_EOS",
                shape=[2, self.word_embedding_dim],
                initializer = init,
                trainable = False)

            word_vector_UNK_DROPOUT = tf.get_variable(
                name="word_vector_UNK_DROPOUT",
                shape=[2, self.word_embedding_dim],
                initializer = init,
                trainable = False)

            pretrained_word_embd  = tf.get_variable(
                name="pretrained_word_embd",
                shape=[self.vocab_size-4, self.word_embedding_dim],
                initializer = init,
                trainable = False)

            # word embedding
            word_embedding_matrix = tf.concat([word_vector_BOS_EOS, word_vector_UNK_DROPOUT, pretrained_word_embd], 0)
            """
            word_embedding_matrix = tf.get_variable(
                    name="word_embedding_matrix",
                    shape=[self.vocab_size, self.word_embedding_dim],
                    initializer = init,
                    trainable = False)
            """
            return word_embedding_matrix

    def build_graph(self):
        word_embedding_matrix = self.get_word_embedding()

        with tf.variable_scope("input") as scope:
            self.inputs = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.sequence_length))
            inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix,self.inputs)

        with tf.variable_scope("encoder") as scope:
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            #bi-lstm encoder
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=self.lstm_length,
                inputs=inputs_embedded,
                time_major=False)
    
            output_fw, output_bw = outputs
            state_fw, state_bw = state
            outputs = tf.concat([output_fw,output_bw],2)      
            state_c = tf.concat((state_fw.c, state_bw.c), 1)
            state_h = tf.concat((state_fw.h, state_bw.h), 1)
            
            self.outputs = outputs
            self.state_c = state_c
            self.state_h = state_h

    def sample(self):
        with tf.variable_scope("sample") as scope:
            w_mean = weight_variable([self.latent_dim*2,self.latent_dim*2],0.1,'Variable')
            b_mean = bias_variable([self.latent_dim*2],'Variable_1')
            w_logvar = weight_variable([self.latent_dim*2,self.latent_dim*2],0.1,'Variable_2')
            b_logvar = bias_variable([self.latent_dim*2],'Variable_3')
            scope.reuse_variables()
            b_mean_matrix = [b_mean] * self.batch_size
            b_logvar_matrix = [b_logvar] * self.batch_size
            
            mean = tf.matmul(self.state_h,w_mean) + b_mean
            logvar = tf.matmul(self.state_h,w_logvar) + b_logvar
            var = tf.exp( 0.5 * logvar)
            noise = tf.random_normal(tf.shape(var))
            self.sampled_state_h = mean #note remove noise  + tf.multiply(var,noise)

    def get_var_list(self):
        embedding_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding')
        encoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        sample_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sample')
        return embedding_var_list + encoder_var_list + sample_var_list

    def encode(self,sent_vec): #batch
        feed_dict = {
            self.inputs:sent_vec
        }
        output_dict = {
            "encoder_state_c":self.state_c, \
            "sampled_encoder_state_h":self.sampled_state_h
        }
        return self.sess.run(output_dict,feed_dict)
