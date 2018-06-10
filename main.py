import argparse
import tensorflow as tf
from model import sentiment_dialogue

def parse():

    # persona-dialogue
    parser = argparse.ArgumentParser(description="VAE+discriminator")
    parser.add_argument('-model_dir','--model_dir',default='model_dir_pure_vae',help='output_model_dir') 
    parser.add_argument('-dict_path','--dict_path',help='dictionary path')
    parser.add_argument('-data_dir','--data_dir',default='data',help='data dir')
    parser.add_argument('-sequence_length','--sequence_length',default=15,type=int,help='sentence length')
    parser.add_argument('-batch_size','--batch_size',default=48,type=int,help='batch size')
    parser.add_argument('-latent_dim','--latent_dim',default=500,type=int,help='latent_size')
    parser.add_argument('-load','--load',action='store_true',help='whether load')
    parser.add_argument('-with_GloVe','--with_GloVe',action='store_true',help='whether use GloVe')

    # sentiment-classifier
    parser.add_argument('-embedding_dim','--embedding_dim',default=128,type=int,help='embedding_dim')
    parser.add_argument('-max_length','--max_length',default=15,type=int,help='max_length')
    parser.add_argument('-unit_size','--unit_size',default=128,type=int,help='unit_size')
    parser.add_argument('-senti_model_dir','--senti_model_dir',default='senti_model', help='sentiment model dir')
    
    parser.add_argument('-test_file','--test_file',default='test.txt', help='test sentences')
    parser.add_argument('-test_file_output','--test_file_output',default='test.txt', help='test sentences')
       
    parser.add_argument('-stdin_test',action='store_true',help='whether stdin test')
    parser.add_argument('-test',action='store_true',help='whether test')
    parser.add_argument('-soft',action='store_true',help='whether decode soft')
    args = parser.parse_args()
    return args

def run(args):
    
    sess = tf.Session()
    model = sentiment_dialogue(args,sess)
    if args.stdin_test:
        model.stdin_test()
    if args.test:
        model.test()    

if __name__ == '__main__':
    args = parse()
    run(args)
