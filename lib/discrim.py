import tensorflow as tf
import numpy as np

class discrim():

    def __init__(self,args,sess,utils,decoder):
        
        self.sess = sess

        self.senti_model_dir = args.senti_model_dir
        self.embedding_dim = args.embedding_dim
        self.max_length = args.sequence_length
        self.unit_size = args.unit_size
        self.batch_size = args.batch_size
        self.utils = utils
        self.vocab_size = len(self.utils.word_id_dict)
        self.decoder = decoder
        self.build_graph(decoder.test_decoder_logits)
        self.gradient()
        self.saver = tf.train.Saver(var_list={v.op.name.replace("sentiment/",""): v for v in self.get_var_list()}) 
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.senti_model_dir))


    def build_graph(self,test_decoder_logits):
        print('starting building graph [sentiment-discriminator]')
        with tf.variable_scope("sentiment") as scope:
            self.inputs = tf.slice(test_decoder_logits,[0,0,0],[self.batch_size,self.max_length,self.vocab_size])
            # variable
            weights = {
                'w2v' : tf.get_variable(initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32),shape = [self.vocab_size, self.embedding_dim], name='w2v'),
                'out_1' : tf.get_variable(initializer = tf.random_normal_initializer(), shape = [self.unit_size*2, 1], name='w_out_1'),
            }
            biases = {
            'out_1' : tf.get_variable(initializer = tf.random_normal_initializer(), shape=[1], name='b_out_1'),
            }
            # structure
            def BiRNN(x):
                x = tf.unstack(x, self.max_length, 1)
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.unit_size, forget_bias=1.0)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.unit_size,forget_bias=1.0)
                outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32 )
                return outputs[-1]

            self.inputs_softmax = tf.nn.softmax(tf.scalar_mul(tf.constant(5.0, shape=[]),self.inputs))
            y_list=[]
            for i in range(self.inputs.get_shape().as_list()[0]):
                y = tf.matmul(self.inputs_softmax[i], weights['w2v'])
                y = tf.reshape(y, [1, self.max_length, self.embedding_dim])
                y_list.append(y)
            embbed_layer = tf.concat(y_list,0)
            layer_1 = BiRNN(embbed_layer)
            pred = tf.matmul(layer_1, weights['out_1']) + biases['out_1'] 
            # get score
            self.score = tf.sigmoid(pred)

    def get_var_list(self):
            return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='sentiment')

    def gradient(self):
        l2_loss = tf.reduce_mean(tf.squared_difference(self.decoder.sampled_encoder_state_h,self.decoder.original_h),axis=1)
        self.grads_sampled_h = tf.gradients(self.score,self.decoder.sampled_encoder_state_h)[0]
        self.l2_grads_sampled_h = tf.gradients(l2_loss,self.decoder.sampled_encoder_state_h)[0]

    def decode_get_grad(self,encoder_state_c,sampled_encoder_state_h,original_h):
        t = np.ones((self.batch_size,self.decoder.sequence_length),dtype=np.int32)
        feed_dict = {
            self.decoder.encoder_state_c:encoder_state_c, \
            self.decoder.sampled_encoder_state_h:sampled_encoder_state_h, \
            self.decoder.train_decoder_sentence:t, \
            self.decoder.original_h:original_h
        }
        output_dict = {
            "pred":self.decoder.test_pred, \
            "score":self.score, \
            "grads_sampled_h":self.grads_sampled_h, \
            "l2_grads_sampled_h":self.l2_grads_sampled_h
        }
        return self.sess.run(output_dict,feed_dict)

