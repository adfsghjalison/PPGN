import argparse
import tensorflow as tf
from model import sentiment_dialogue
from flags import FLAGS

def run():
    
    sess = tf.Session()
    model = sentiment_dialogue(FLAGS, sess)
    if FLAGS.mode == 'stdin':
        model.stdin_test()
    if FLAGS.mode == 'test':
        model.test()    

if __name__ == '__main__':
    run()

