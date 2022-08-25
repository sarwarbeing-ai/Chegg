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


Tx=30 # input time-stamp
Ty=10 # output
n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
m=10000 # number of data-points
batch_size=100
lr=0.005
weight_decay=0.01
no_epochs=20


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



dataset_path="/content/dataset.pickle"
human_vocab="/content/human_vocab.pickle"
machine_vocab="/content/machine_vocab.pickle"
inv_machine_vocab="/content/inv_machine_vocab.pickle"


with open(dataset_path,"rb") as file:
  dataset=pickle.load(file)

with open(human_vocab,"rb") as file:
  human_vocab=pickle.load(file)
with open(machine_vocab,"rb") as file:
  machine_vocab=pickle.load(file)
with open(inv_machine_vocab,"rb") as file:
  inv_machine=pickle.load(file)


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    # Tx is the input time-stamp and Ty is the output time stamp
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X))) # one-hot encoding
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh



def string_to_int(string, length, vocab):
    """
    convert the string into a list of integers according to vocab

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- number of time steps
    vocab -- vocabulary, dictionary used to index every character of the string "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    string = string.lower()
    string = string.replace(',','')

    if len(string) > length:
        string = string[:length] # truncate the string

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string)) # if x is not found then unk

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string)) # padded with pad token

    return rep


def int_to_string(ints, inv_vocab):
    """
    convert the list of integers into a list of characters according to inverse vocab
    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters

    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """

    l = [inv_vocab[i] for i in ints]
    return l



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



if __name__=="__main__":
  model=model_date(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
  opt = Adam(learning_rate=lr, decay=weight_decay)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  _,_,Xoh,Yoh=preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
  outputs=list(Yoh.swapaxes(0,1))
  s0 = np.zeros((m, n_s))
  c0 = np.zeros((m, n_s))
  model.fit([Xoh, s0, c0], outputs, epochs=no_epochs, batch_size=batch_size)

  # save the model
  model.save_weights("model_date.h5")
