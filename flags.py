import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode','test', 'test / stdin')

# persona-dialogue
tf.app.flags.DEFINE_string('model_dir', '../VRAE/model', 'output_model_dir') 
tf.app.flags.DEFINE_string('data_dir', '../VRAE/data', 'data dir')
tf.app.flags.DEFINE_string('data_name','NLPCC', 'data dir')
tf.app.flags.DEFINE_integer('sequence_length', 15, 'sentence length')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('latent_dim', 500, 'latent_size')

# sentiment-classifier
tf.app.flags.DEFINE_string('senti_model_dir', '../sentiment/model', 'sentiment model dir')
tf.app.flags.DEFINE_integer('embedding_dim', 256, 'embedding_dim')
tf.app.flags.DEFINE_integer('unit_size', 128, 'unit_size')
#tf.app.flags.DEFINE_integer('max_length', 15, 'max_length')

tf.app.flags.DEFINE_string('soft', False, 'whether decode soft')
tf.app.flags.DEFINE_string('out', 'output', 'testing output file')

FLAGS = tf.app.flags.FLAGS

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}'.format(FLAGS.data_name))
FLAGS.senti_model_dir = os.path.join(FLAGS.senti_model_dir, 'model_{}'.format(FLAGS.data_name))
FLAGS.out = os.path.join(FLAGS.out, 'output_{}'.format(FLAGS.data_name))

BOS = 0
EOS = 1
UNK = 2
DROPOUT = 3

