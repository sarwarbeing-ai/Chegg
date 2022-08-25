from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
import pickle


human_vocab="/content/human_vocab.pickle"
machine_vocab="/content/machine_vocab.pickle"
inv_machine_vocab="/content/inv_machine_vocab.pickle"
model_path="/content/model_date.h5"



with open(human_vocab,"rb") as file:
  human_vocab=pickle.load(file)
with open(machine_vocab,"rb") as file:
  machine_vocab=pickle.load(file)
with open(inv_machine_vocab,"rb") as file:
  inv_machine=pickle.load(file)


Tx=30 # input time-stamp
Ty=10 # output
n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
m=10000 # number of data-points


def string_to_int(string, length, vocab):
    string = string.lower()
    string = string.replace(',','')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep


def softmax(x, axis=1):
    """Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')



repeator = RepeatVector(Tx) # vector will be repeated TX times
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1) # dot product


post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-lstm
output_layer = Dense(len(machine_vocab), activation=softmax) # output layer



def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM(pre-attention-lstm), numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    s_prev = repeator(s_prev) # repeater to repeat s_prev ,so that it shape becomes (m,Tx,n_s)

    concat = concatenator([a,s_prev]) # concatenate on the last axis
    e = densor1(concat) # intermediate energies
    energies = densor2(e) # final energies
    alphas = activator(energies) # attention-weights
    context = dotor([alphas,a]) # dot product between input and attentions,so that each input gets some attention

    return context


def model_date(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0') # initial hidden state for lstm
    c0 = Input(shape=(n_s,), name='c0') # initial cell state
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X) # pre-attention Bi-LSTM

    for t in range(Ty):
        context = one_step_attention(a, s)

        s, _, c = post_activation_LSTM_cell(context,initial_state=[s, c])

        out = output_layer(s)

        outputs.append(out)
    model = Model(inputs=[X, s0, c0],outputs=outputs)
    return model



model=model_date(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.load_weights(model_path)



def translate_date(sentence):
    s00 = np.zeros((1, n_s))
    c00 = np.zeros((1, n_s))
    source = string_to_int(sentence, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s00, c00])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine[int(i)] for i in prediction]
    print("source:", sentence)
    print("output:", ''.join(output),"\n")


if __name__=="__main__":
  example = "4th of july 2001"
  translate_date(example)

 
