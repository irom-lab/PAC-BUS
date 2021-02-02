import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operator import itemgetter
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from text_embedding.vectors import vocab2vecs
import torch
import time

# code based on
# https://github.com/mkhodak/FMRL/blob/master/data.py

dir_path = os.path.dirname(os.path.realpath(__file__))
HOME = dir_path[:-1*dir_path[::-1].find('/')]
DATA = HOME + 'data/miniwiki/'
VECDIR = HOME + 'glove/glove.6B.'
RAW = DATA + 'raw/'


def create_folders(names):
    for name in names:
        if not os.path.exists(name):
            os.system("mkdir " + name)


def word2vec(dim):
    '''returns dict of GloVe embeddings
    Args:
        dim: vector dimension (50, 100, 200, or 300)
    Returns:
        {word: vector} dict
    '''

    # w2v = vocab2vecs(vectorfile=VECDIR + str(dim) + 'd.h5')
    w2v = vocab2vecs(vectorfile=VECDIR + str(dim) + 'd.txt')
    w2v[0] = np.zeros(dim, dtype=np.float32)
    return w2v


def text2cbow(fname, w2v):
    '''returns CBOW text representations from file with "label\ttoken token ... token\n" on each line
    Args:
        fname: file name
        w2v: {word: vector} dict
    Returns:
        numpy data array of shape [number of lines, vector dimension], numpy label array of shape [number of lines,]
    '''

    z = w2v[0]
    with open(fname, 'r') as f:
        labels, texts = zip(*(line.strip().split('\t') for line in f))
        X = np.array([sum((w2v.get(w.lower(), z) for w in text.split()), np.copy(z)) for text in texts])
        nz = norm(X, axis=1) > 0.0
        X[nz] = normalize(X[nz])
        return X, np.array([int(label) for label in labels])


def textfiles(corpus='bal', partition='train', m=32):
    '''returns text file names
    Args:
        corpus: which subcorpus to use ('bal' or 'raw')
        partition: which partition to use ('train', 'dev', or 'test') ; ignored if corpus = 'raw'
        m: number of data points per class (1, 2, 4, ... , or 32) ; ignored if corpus = 'raw'
    Returns:
        list of filenames
    '''
    
    datadir = DATA+corpus+'/'
    if not corpus == 'raw':
        datadir += partition+'/'+str(m)+'/'
    return [datadir+fname for fname, _ in sorted(((fname, int(fname[:-4])) for fname in os.listdir(datadir)), key=itemgetter(1))]


def load_pb_to_device(device, corpus, partition1, partition2, dim, ncls, M, num_tasks1=100, num_tasks2=500, w2v=None):
    w2v, (X_tr1, Y_tr1, X_te1, Y_te1) = load_to_device(device, corpus, partition1, dim, ncls, M, num_tasks=num_tasks1, verbose=False, w2v=w2v)
    w2v, (X_tr2, Y_tr2, X_te2, Y_te2) = load_to_device(device, corpus, partition2, dim, ncls, M, num_tasks=num_tasks2, verbose=False, w2v=w2v)

    X_tr = torch.cat((X_tr1, X_tr2), 0)
    Y_tr = torch.cat((Y_tr1, Y_tr2), 0)
    X_te = torch.cat((X_te1, X_te2), 0)
    Y_te = torch.cat((Y_te1, Y_te2), 0)

    return w2v, (X_tr, Y_tr, X_te, Y_te)


def load_to_device(device, corpus, partition, dim, ncls, M, num_tasks=500, offset=0, verbose=False, w2v=None):
    fnames_train = textfiles(corpus=corpus, partition=partition, m=M)
    fnames_test = textfiles(corpus=corpus, partition=partition, m='test')

    if num_tasks == 0:
        return word2vec(dim), (None, None, None, None)

    if num_tasks + offset > len(fnames_train):
        print("Not enough files for input")
        return False

    if verbose:
        print("Loading word-to-vec dictionary from files")
        st1 = time.time()
    if not w2v:
        w2v = word2vec(dim)
    if verbose:
        print("Dictionary loaded, took", time.time() - st1, "seconds")

    X, Y = text2cbow(fnames_test[0], w2v)
    num_test_examples = len(Y)

    X_tr = torch.empty(num_tasks, ncls*M, dim)
    Y_tr = torch.empty(num_tasks, ncls*M)
    X_te = torch.empty(num_tasks, num_test_examples, dim)
    Y_te = torch.empty(num_tasks, num_test_examples)

    if verbose:
        print("Loading data into tensors")
        st2 = time.time()
    for i in range(num_tasks):
        X_train, Y_train = text2cbow(fnames_train[i+offset], w2v)
        X_tr[i], Y_tr[i] = torch.tensor(X_train), torch.tensor(Y_train)
        X_test, Y_test = text2cbow(fnames_test[i+offset], w2v)
        X_te[i], Y_te[i] = torch.tensor(X_test), torch.tensor(Y_test)

    if verbose:
        print("Data loaded into tensors, took", time.time() - st2, "seconds")
        print("Loading data onto device...")
        st3 = time.time()
    X_tr = X_tr.to(device)
    Y_tr = Y_tr.type('torch.LongTensor').to(device)
    X_te = X_te.to(device)
    Y_te = Y_te.type('torch.LongTensor').to(device)

    if verbose:
        print("Data loaded onto device, took", time.time() - st3, "seconds")

    return w2v, (X_tr, Y_tr, X_te, Y_te)


def clean_text(text):
    # text = text.replace('\'\'', '\"')
    # text = text.replace('\"', '')
    # text = text.replace('*', '')
    if len(text) < 120:
        text = ''
    return text


def create_dataset(corpus, ncls, M, n_pr_tasks, n_tr_tasks, n_te_tasks, verbose=False, data=None, t_num_test=128):
    NUM_FILES = 812
    if verbose: print("Creating folders if they don't exist")
    corpus_folder = DATA + corpus + '/'
    m_tr_folder = corpus_folder + 'train/'
    m_tr_b_tr_folder = m_tr_folder + str(M) + '/'
    m_tr_b_te_folder = m_tr_folder + 'test/'
    m_te_folder = corpus_folder + 'test/'
    m_te_b_tr_folder = m_te_folder + str(M) + '/'
    m_te_b_te_folder = m_te_folder + 'test/'
    p_folder = corpus_folder + 'prior/'
    p_tr_folder = p_folder + str(M) + '/'
    p_te_folder = p_folder + 'test/'
    create_folders((corpus_folder, m_tr_folder, m_tr_b_tr_folder, m_tr_b_te_folder,
                    m_te_folder, m_te_b_tr_folder, m_te_b_te_folder,
                    p_folder, p_tr_folder, p_te_folder))

    # load data into mega array with different label for each class
    if verbose: print("Loading raw data into meta array with different labels for each wiki article")
    if data is None:
        data = {}
        for i, fname in enumerate(textfiles(corpus='raw')):
            with open(fname, 'r') as f:
                labels, texts = zip(*(line.strip().split('\t') for line in f))

            data[4*i+0] = ()
            data[4*i+1] = ()
            data[4*i+2] = ()
            data[4*i+3] = ()
            for label, text in zip(labels, texts):  # will do a deep copy since text is a string not a list
                text = clean_text(text)
                if text != '':
                    data[int(label)+4*i] = (*data[int(label)+4*i], text)

    # shuffling dictionary labels:
    if verbose: print("Shuffling data labels")
    inds = [i for i in range(NUM_FILES)]
    np.random.shuffle(inds)
    shuffled_data = {}
    for i, ind in enumerate(inds):
        for j in range(4):
            shuffled_data[i*4 + j] = list(data[ind*4 + j])
    data = shuffled_data

    n_tasks = n_pr_tasks + n_tr_tasks + n_te_tasks
    split1 = int(n_pr_tasks / n_tasks * len(data)/4)
    pr_labels = [i for i in range(4*split1)]
    m_tr_labels = [i for i in range(4*split1, 4*(len(data)//4))]
    m_te_labels = [i for i in range(4*split1, 4*(len(data)//4))]

    if verbose: print("Creating prior-training files")
    create_testtrain_files(p_tr_folder, p_te_folder, ncls, M, n_pr_tasks, pr_labels, data, t_num_test)
    if verbose: print("Creating meta-train files")
    create_testtrain_files(m_tr_b_tr_folder, m_tr_b_te_folder, ncls, M, n_tr_tasks, m_tr_labels, data, t_num_test)
    if verbose: print("Creating meta-test files")
    create_testtrain_files(m_te_b_tr_folder, m_te_b_te_folder, ncls, M, n_te_tasks, m_te_labels, data, t_num_test)

    return data


def create_testtrain_files(foldertr, folderte, ncls, M, n_tasks, m_labels, data, t_num_test):

    for filenum in range(n_tasks):
        i = np.random.randint(0, len(m_labels)//4)
        m_labs = m_labels[i*4:(i+1)*4]

        fnametr = foldertr + str(filenum) + '.csv'
        fnamete = folderte + str(filenum) + '.csv'

        tr_file_data = []
        te_file_data = []
        for j, label in enumerate(m_labs):
            np.random.shuffle(data[label])
            if j == ncls: break  # only do ncls of these
            b_labels = [k for k in range(len(data[label]))]
            m = 0
            while m < M:
                k = np.random.choice(b_labels)
                tr_file_data.append(str(j) + "\t" + data[label][k])
                m += 1
            m = 0
            while m < int(t_num_test/ncls):
                k = np.random.choice(b_labels)
                te_file_data.append(str(j) + "\t" + data[label][k])
                m += 1

        np.random.shuffle(tr_file_data)
        np.random.shuffle(te_file_data)

        with open(fnametr, 'w') as ftr:
            for line in tr_file_data:
                ftr.write(line + '\n')
        with open(fnamete, 'w') as fte:
            for line in te_file_data:
                fte.write(line + '\n')


if __name__ == '__main__':
    t_num_test = 250
    ncls = 4
    MS = [1, 3, 5]
    n_prior_tasks = 100
    n_train_tasks = 1000
    n_test_tasks = 200

    data = None
    # for trial in ['test']:
    for trial in [1, 2, 3, 4, 5, 'test']:
        for M in MS:
            print('trial:',trial," M:", M)
            corpus = str(trial) + '_ncls' + str(ncls) + 'm' + str(M)
            data = create_dataset(corpus, ncls, M, n_prior_tasks, n_train_tasks, n_test_tasks,
                           data=data, t_num_test=t_num_test, verbose=True)