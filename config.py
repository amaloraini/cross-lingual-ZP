import tensorflow as tf
def load_config():
    config = {}
    config['epochs']        = 10
    config['nhidden1']      = 3300
    config['nhidden2']      = 2200
    config['nhidden3']      = 1200
    config['n_outputs']     = 2
    config['learning_rate'] = 1e-5
    config['decay_rate']    = 0.01
    config['dropout_rate']  = 0.1
    config['activation']    = tf.nn.relu
    config['do_clipping']   = True
    return config