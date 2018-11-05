import tensorflow as tf
from lib.ops import *

class decoder():

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

        self.EOS = self.utils.EOS_id
        self.BOS = self.utils.BOS_id

        self.soft = args.soft
        self.build_graph()

        for i in self.get_var_list():
          print i

        #self.saver = tf.train.Saver(var_list={v.op.name : v for v in self.get_var_list()})
        self.saver = tf.train.Saver(var_list = self.get_var_list())
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        # original h code
        self.original_h = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.latent_dim*2))

    def build_graph(self):
        with tf.variable_scope("embedding") as scope:

            init = tf.contrib.layers.xavier_initializer()

            weight_output = tf.get_variable(
                name="weight_output",
                shape=[self.latent_dim*2, self.vocab_size],
                initializer = init,
                trainable = False)
                
            bias_output = tf.get_variable(
                name="bias_output",
                shape=[self.vocab_size],
                initializer = tf.constant_initializer(value = 0.0),
                trainable = False)
            
        with tf.variable_scope("embedding", reuse = True) as scope:
            # word embedding
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

            word_embedding_matrix = tf.concat([word_vector_BOS_EOS, word_vector_UNK_DROPOUT, pretrained_word_embd], 0)

        #weight_output, bias_output = self.get_output_projection()
        #word_embedding_matrix = self.get_word_embedding()
        self.train_decoder_sentence = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,self.sequence_length))
        BOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.BOS
        train_decoder_sentence = tf.concat([BOS_slice,self.train_decoder_sentence],axis=1)

        self.encoder_state_c = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.latent_dim*2))
        self.sampled_encoder_state_h = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.latent_dim*2))
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c=self.encoder_state_c, h=self.sampled_encoder_state_h)

        decoder_inputs = batch_to_time_major(train_decoder_sentence,self.sequence_length+1)
        #decoder_inputs_embedded = batch_to_time_major(tf.nn.embedding_lookup(word_embedding_matrix,train_decoder_sentence),self.sequence_length+1)

        with tf.variable_scope("decoder") as scope:
            cell = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim*2, state_is_tuple=True)

            def decoder():
                test_decoder_output, test_decoder_state = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(
                    decoder_inputs = decoder_inputs,
                    initial_state = encoder_state,
                    cell = cell,
                    num_symbols = self.vocab_size,
                    embedding_size = self.word_embedding_dim,
                    output_projection = (weight_output, bias_output),
                    feed_previous = True,
                    scope = scope
                )   
                return  test_decoder_output, test_decoder_state

            def decoder_soft():
                def test_decoder_loop(prev,i):
                    factor = tf.constant(5,shape=(),dtype=tf.float32)
                    prev = tf.scalar_mul(factor,tf.add(tf.matmul(prev,weight_output),bias_output))
                    prev_index = tf.nn.softmax(prev) 
                    pred_prev = tf.matmul(prev_index,word_embedding_matrix)
                    next_input = pred_prev
                    return next_input

                test_decoder_output,test_decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs = decoder_inputs_embedded,
                    initial_state = encoder_state,
                    cell = cell,
                    loop_function = test_decoder_loop,
                    scope = scope
                 )
                return  test_decoder_output, test_decoder_state
             
            test_decoder_output,test_decoder_state = decoder_soft() if self.soft else decoder()
            
            for index,time_slice in enumerate(test_decoder_output):
                test_decoder_output[index] = tf.add(tf.matmul(test_decoder_output[index],weight_output),bias_output)

            self.test_decoder_logits = tf.stack(test_decoder_output, axis=1)
            test_pred = tf.argmax(self.test_decoder_logits,axis=-1)
            self.test_pred = tf.to_int32(test_pred,name='ToInt32')

    def get_var_list(self):
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding')
        # decoder/lstm_cell/kernel => decoder/rnn_decoder/lstm_cell/kernel
        if self.soft :
          v_list = {v.op.name.replace('decoder/lstm_cell/', 'decoder/rnn_decoder/lstm_cell/') : v for v in var}
        else:
          v_list = {v.op.name : v for v in var}

        return v_list
