import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def Generator(inputs,is_train = True,reuse=False):
    img_size = FLAGS.output_size
    s2,s4,s8,s16 = int(img_size/2),int(img_size/4),int(img_size/8),int(img_size/16)
    if FLAGS.dataset == "mnist":
        s8+=1;s16+=1
    if FLAGS.dataset == "cifar10" or FLAGS.dataset == "svhn":
        s16+=2
    gf_dim = 64 # will be added to flags
    c_dim = FLAGS.c_dim  # channels
    batch_size = FLAGS.batch_size
    w_init = tf.random_normal_initializer(stddev = 0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)
    with tf.variable_scope("generator",reuse = reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs,name='G/in')
        net_h0 = DenseLayer(net_in,n_units=64*s16*s16,W_init=w_init,\
                            act=tf.identity,name='G/h0/lin')
        net_h0 = ReshapeLayer(net_h0,shape=[-1,s16,s16,64],name='G/h0/reshape')
        net_h0 = BatchNormLayer(net_h0,act=tf.nn.relu,is_train=is_train,\
                                gamma_init=gamma_init,name='G/h0/batch_norm')
        #filter_size = 5 which will be added to FLAGS
        net_h1 = DeConv2d(net_h0,32,(5,5),out_size=(s8,s8),strides=(2,2),\
                          padding='SAME',batch_size=batch_size,act=None,W_init=w_init,name='G/h1/deconv')
        net_h1 = BatchNormLayer(net_h1,act=tf.nn.relu,is_train=is_train,\
                                gamma_init=gamma_init,name='G/h1/batch_norm')
        net_h2 = DeConv2d(net_h1, 16, (5, 5), out_size=(s4, s4), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='G/h2/deconv')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='G/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, 8, (5, 5), out_size=(s2, s2), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='G/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='G/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(img_size, img_size), strides=(2, 2),
                          padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='G/h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
        net_h1_FC = FlattenLayer(net_h1, name='G/h1/flatten')
    return net_h4, net_h1_FC

def Encoder(inputs,is_train=True,reuse=False):
    ef_dim = 64
    c_dim = FLAGS.c_dim  # channels
    batch_size = FLAGS.batch_size
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)
    with tf.variable_scope("encoder",reuse=reuse):
        set_name_reuse(reuse)
        net_in = InputLayer(inputs,name='E/in')
        net_h0 = Conv2d(net_in,8,(5,5),(2,2),act=lambda x:tl.act.lrelu(x,0.2),\
                        padding='SAME',W_init=w_init,name='E/h0/conv2d')

        net_h1 = Conv2d(net_h0, 16, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='E/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='E/h1/batch_norm')

        net_h2 = Conv2d(net_h1, 32, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='E/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='E/h2/batch_norm')

        net_h3 = Conv2d(net_h2, 64, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='E/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='E/h3/batch_norm')
        net_h4_fc = FlattenLayer(net_h3, name='E/h4/flatten')
        net_h4 = DenseLayer(net_h4_fc, n_units=FLAGS.z_dim, act=tf.identity,
                            W_init=w_init, name='E/h4/lin_sigmoid')

        logits = tf.nn.tanh(net_h4.outputs)
        #net_h4.outputs = tf.nn.relu(net_h4.outputs)
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits,net_h4_fc.outputs

def Discriminator(inputs,is_train=True,reuse=False):
    df_dim = 64 #filters dimension in first layer
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)
    with tf.variable_scope("discriminator",reuse=reuse):
        set_name_reuse(reuse)
        net_in = InputLayer(inputs,name='D/in')
        net_h0 = Conv2d(net_in, 8, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='D/h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='D/h0/batch_norm')

        net_h1 = Conv2d(net_h0, 16, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='D/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='D/h1/batch_norm')

        net_h2 = Conv2d(net_h1, 32, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='D/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='D/h2/batch_norm')


        net_h3 = Conv2d(net_h2, 64, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='D/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='D/h3/batch_norm')

        net_h4_fc = FlattenLayer(net_h3, name='D/h4/flatten')
        net_h4 = DenseLayer(net_h4_fc, n_units=1, act=tf.identity,
                            W_init=w_init, name='D/h4/lin_sigmoid')
        logits = net_h4.outputs
        #net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits, net_h4_fc.outputs

def MLP(inputs,is_train = True,reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tf.reshape(inputs, shape = [-1,784])
        network = tl.layers.InputLayer(inputs, name='MLP/input')
        '''network = tl.layers.DenseLayer(network, n_units=256,
                            act = tf.nn.relu, name='MLP_E/relu1')
        network = tl.layers.DenseLayer(network, n_units=128,
                            act = tf.nn.relu, name='MLP_E/relu2')'''
        network = tl.layers.DenseLayer(network, n_units=FLAGS.no_classes,
                            act = tf.identity, name='MLP/output')
        #logits = network.outputs
        #network.outputs = tf.nn.sigmoid(logits)
    return network

def MLP_D(inputs,is_train = True,reuse=False):
    with tf.variable_scope("MLP_D", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(inputs, name='MLP_D/input')
        '''network = tl.layers.DenseLayer(network, n_units=256,
                            act = tf.nn.relu, name='MLP_D/relu1')
        network = tl.layers.DenseLayer(network, n_units=128,
                            act = tf.nn.relu, name='MLP_D/relu2')'''
        network = tl.layers.DenseLayer(network, n_units=FLAGS.no_classes,
                            act = tf.identity, name='MLP_D/output')
    return network

def MLP_D_C(inputs,is_train = True,reuse=False):
    with tf.variable_scope("MLP_D_C", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(inputs, name='MLP_D_C/input')
        '''network = tl.layers.DenseLayer(network, n_units=256,
                            act = tf.nn.relu, name='MLP_D/relu1')
        network = tl.layers.DenseLayer(network, n_units=128,
                            act = tf.nn.relu, name='MLP_D/relu2')'''
        network = tl.layers.DenseLayer(network, n_units=FLAGS.no_classes,
                            act = tf.identity, name='MLP_D_C/output')
    return network

def MLP_E(inputs,is_train = True,reuse=False):
    with tf.variable_scope("MLP_E", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.InputLayer(inputs, name='MLP_E/input')
        '''network = tl.layers.DenseLayer(network, n_units=256,
                            act = tf.nn.relu, name='MLP_E/relu1')
        network = tl.layers.DenseLayer(network, n_units=128,
                            act = tf.nn.relu, name='MLP_E/relu2')'''
        network = tl.layers.DenseLayer(network, n_units=FLAGS.no_classes,
                            act = tf.identity, name='MLP_E/output')
        #logits = network.outputs
        #network.outputs = tf.nn.sigmoid(logits)
    return network