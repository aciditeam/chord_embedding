#!/usr/bin/env python
# -*- coding: utf8 -*-

# SKlearn
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from bhtsne_master.bhtsne import bh_tsne
import numpy as np


def tsne(data):
    print "Compute t-SNE"
    # tsne = TSNE(n_components=2, random_state=0, n_iter=200)
    # data_reduced = tsne.fit_transform(data)
    input_dim = data.shape[1]
    data_reduced = bh_tsne(data, no_dims=2, initial_dims=input_dim, perplexity=50)
    filename = 'TEMP'
    f = open(filename, 'wb')
    for result in bh_tsne(data, no_dims=2, perplexity=50):
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'
        f.write(fmt.format(*result))
    f.close()

    # read from f
    data_reduced = np.loadtxt(filename, delimiter='\t')
    return data_reduced
