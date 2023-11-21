import numpy as np
import scipy.io as sio

embeddings_index = {}
f = open('./GloVe/model/glove.6B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vec = []
with open('../datasets/XMediaNet/cateList.txt', 'r') as f:
    for line in f:
        cat = line.split()[0]
        if cat == 'TV':
            cat = 'television'
        cat = cat.split('_')
        vec = sum([embeddings_index[c] for c in cat]) / float(len(cat))

sio.savemat('./embedding/xmedianet-inp-glove6B.mat', {'inp': np.array(vec)})
        