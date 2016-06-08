#!/usr/bin/env python
# -*- coding: utf8 -*-

# Bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool
from collections import OrderedDict
from bokeh.plotting import *

# SKlearn
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from bhtsne_master.bhtsne import bh_tsne
# Numpy
import numpy as np
import unicodecsv as csv
import re
import colorsys

COMPUTE = False
highlighted_track = 'Beethoven_PnoCrto2_i(1-16)_ORCH+REDUC+piano.mid'

# Get data
print "Load data"
output_file("../Results/tsne.html")
data_path = "../Results/data.csv"
data = np.transpose(np.loadtxt(data_path, delimiter=','))

# load the change track matrix
time_pointer_path = "../final_data/time_pointer2.csv"
time_pointer = np.loadtxt(time_pointer_path, delimiter=',')

# Beginning of tracks in the new event granularity : tp_ne
event_total_path = "../final_data/event_total2.csv"
event_total = np.loadtxt(event_total_path, delimiter=',')


def get_tp_ne():
    tp_ne = np.zeros((time_pointer.shape[0]-1))  # Last item is useless (end of the last track)
    current_track = 0
    event_counter = 0
    for index in range(event_total.shape[0]):
        if event_total[index]:
            event_counter += 1
        if index == time_pointer[current_track]+1:
            tp_ne[current_track] = event_counter
            current_track += 1
    return tp_ne
tp_ne = get_tp_ne()


def map_color(e, N):
    hue = float(e) / (N+1)
    saturation = float(180 + 90*(float(e) / (N+1))) / 360
    lightness = float(200. / 360)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return '#%02x%02x%02x' % (int(r * 256), int(g * 256), int(b * 256))

# Compute reduction : mask data with event_total
event_total = event_total.astype(np.int)
mask = []
counter = 0
for e in event_total:
    if e == 1:
        mask.append(counter)
    counter += 1
data_ne = data[mask,:]

if __name__ == '__main__':
    # # Compute PCA
    # print "Compute PCA"
    # pca = PCA(n_components=50)
    # data_pca = pca.fit_transform(data)
    # Compute t-SNE
    if COMPUTE:
        print "Compute t-SNE"
        # tsne = TSNE(n_components=2, random_state=0, n_iter=200)
        # data_reduced = tsne.fit_transform(data_ne)
        input_dim = data_ne.shape[1]
        data_reduced = bh_tsne(data_ne, no_dims=2, initial_dims=input_dim, perplexity=50)
        filename = 'TEMP'
        f = open(filename, 'wb')
        for result in bh_tsne(data_ne, no_dims=2, perplexity=50):
            fmt = ''
            for i in range(1, len(result)):
                fmt = fmt + '{}\t'
            fmt = fmt + '{}\n'
            f.write(fmt.format(*result))
        f.close()

        # read from f
        data_reduced = np.loadtxt(filename, delimiter='\t')

        # Labelling and colours
        # Save data
        np.savetxt("tsne.csv", data_reduced, delimiter=',')
    else:
        data_reduced = np.loadtxt("tsne.csv", delimiter=",")

    # Build list name
    file_names = []
    extension = ur"\.mid$"
    def processing_name(name):
        fname_no_extension = re.sub(extension, "", name, re.U)
        fname_no_parenthesis = re.sub(ur"\(.*$", "", fname_no_extension, re.U)
        fname_no_path = re.split(ur"/", fname_no_parenthesis, re.U)[-1]
        return fname_no_path
    with open("../final_data/track_list2.csv", "rb") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            file_names.append(processing_name(row[0]))

    # Colors list
    num_track = len(file_names)
    colors = [map_color(e, num_track) for e in range(len(file_names))]

    # get index of hihlighted track
    import pdb; pdb.set_trace()

    highlighted_ind = file_names.index(processing_name(highlighted_track))
    colors[highlighted_ind] = "#000000"
    # Define a function for labelling the data
    def labelling(time, tp):
        i = 0
        while(tp[i] <= time):
            i += 1
            if i == tp.shape[0]:
                i = -1  # take last index
                break
        return file_names[i], colors[i]

    # Plot t-SNE map
    print "Plot data"
    TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"

    # Plot only one point out of N
    plot_period = 10
    range_plot = range(0, data_reduced.shape[0], plot_period)
    source = ColumnDataSource(
        data=dict(
            x=data_reduced[range_plot,0],
            y=data_reduced[range_plot,1],
            label=[labelling(e, tp_ne)[0] for e in range_plot],
            color=[labelling(e, tp_ne)[1] for e in range_plot]
        )
    )

    p = figure(title="Chord embedding # t-SNE visualization", tools=TOOLS)
    p.circle('x', 'y', color='color', line_width=2, source=source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        # ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("piece", "@label"),
    ])

    show(p)
