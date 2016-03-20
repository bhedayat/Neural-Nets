
# coding: utf-8

# In[1]:

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
from process import load_data

#x = T.tensor4()


N_HIDDEN = 100

LEARNING_RATE = .001

GRAD_CLIP = 100

NUM_EPOCHS = 20

BATCH_SIZE = 200

vocab_size = 9

inp_t,inp_v,output_t,output_v = load_data()
sli_l = 8
sli = 64

#y = T.ivector()
def gen_data():

    xx = np.zeros((BATCH_SIZE,512,512))
    rng_state = np.random.get_state()
    np.random.shuffle(inp_t)
    np.random.set_state(rng_state)
    np.random.shuffle(output_t)
    y = output_t[0:BATCH_SIZE]
    xx  = inp_t[0:BATCH_SIZE,:,:]
    y_v = output_v

    x_v = np.zeros((936,sli,1,sli_l,512))
    for i in range(len(inp_v)):
        for j in range(0,512,sli_l):
            x_v[i,j:j+sli_l,:,:,:] = inp_v[i,j:j+sli_l,:]

    x = np.zeros((BATCH_SIZE,sli,1,sli_l,512))
    for i in range(len(xx)):
        for j in range(0,512,sli_l):
            x[i,j:j+sli_l,:,:,:] = xx[i,j:j+sli_l,:]
    return x, x_v, y, y_v
#print(xx.shape)

def main(num_epochs=NUM_EPOCHS):

    #l_in = lasagne.layers.InputLayer((BATCH_SIZE,64,1,8,512),x,'input_layer')
    l_in = lasagne.layers.InputLayer((BATCH_SIZE,sli,1,sli_l,512))

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)


    l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)


    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size, W = lasagne.init.GlorotUniform(),nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.ivector('target_output')

    network_output = lasagne.layers.get_output(l_out)

    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)


    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    get_out = theano.function([l_in.input_var],lasagne.layers.get_output(l_out),allow_input_downcast=True)

    probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
    for n in xrange(1000):
        inp_t,inp_v,output_t,output_v = load_data()
        x, x_v, y, y_v = gen_data()
        avg_cost = 0
        avg_cost += train(x,y)
        val_output = get_out(x_v)
        val_predictions = np.argmax(val_output, axis=1)
        #print(val_predictions)
        #print(y_v)
        accuracy = np.mean(val_predictions == y_v)
        print(accuracy)
        print(avg_cost)
if __name__ == '__main__':
    main()

