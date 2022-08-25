import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from tensorflow.keras.utils import to_categorical # for one-hot encoding
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pickle

fake=Faker() # initialize faker object

# the data would be generated in english
# if other language we just need to change it to the correpsonding language
LOCALES = ['en_US']

# setting seed
seed=2021
def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  Faker.seed(seed)
set_seed(seed)

# formats of the data that we will generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']


def load_date():
    """
        Loads some fake dates
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    human_vocab = set() # vocab set
    machine_vocab = set() # machine vocab
    dataset = []

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'],
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}

    return dataset, human, machine, inv_machine


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


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    # Tx is the input time-stamp and Ty is the output time stamp
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X))) # one-hot encoding
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh


if __name__=="__main__":
  m=10000
  dataset,human_vocab,machine_vocab,inv_machine=load_dataset(m)
  Tx = 30 # maximum lenght of the human readable data
  Ty = 10 # length of the output string,"YYYY-MM-DD" is 10 characters long.
  X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

  with open("dataset.pickle","wb") as file:
    pickle.dump((dataset),file)

  with open("human_vocab.pickle","wb") as file:
    pickle.dump(human_vocab,file)

  with open("machine_vocab.pickle","wb") as file:
    pickle.dump(machine_vocab,file)
  with open("inv_machine_vocab.pickle","wb") as file:
    pickle.dump(inv_machine,file)
