import tensorflow as tf

def batch_to_time_major(inputs,split_size):
    inputs = tf.split(inputs,  num_or_size_splits=split_size ,axis=1)
    inputs = [tf.squeeze(e,axis=1) for e in inputs]
    return inputs
    
def weight_variable(shape,std_dev,name):
    initial = tf.truncated_normal_initializer(stddev = std_dev)
    return tf.get_variable(initializer = initial, shape = shape,name = name)


def bias_variable(shape,name):
    initial = tf.constant_initializer(value = 0.0)
    return tf.get_variable(initializer = initial, shape = shape, name = name)