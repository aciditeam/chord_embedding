#!/usr/bin/env python
# -*- coding: utf8 -*-

# Bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool
from collections import OrderedDict
from bokeh.plotting import *

# SKlearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# Numpy
import numpy as np
import os
import re
import colorsys

# Get data
print "Load data"
output_file("../Results/tsne.html")
data_path = "../Results/data.csv"
data = np.transpose(np.loadtxt(data_path, delimiter=','))

# Compute PCA
print "Compute PCA"
pca = PCA(n_components=50)
data_pca = pca.fit_transform(data)
# Compute t-SNE
print "Compute t-SNE"
tsne = TSNE(n_components=2, random_state=0, n_iter=200)
data_reduced = tsne.fit_transform(data)

# Labelling and colours
# load the change track matrix
time_pointer_path = "../final_data/time_pointer2.csv"
time_pointer = np.loadtxt(time_pointer_path, delimiter=',')
# Build list name
folder_content = os.listdir("../Midi/Classe/")
file_names = []
for file_name in folder_content:
    extension = r"\.mid$"
    if re.search(extension, file_name):
        fname_no_extension = re.sub(extension, "", file_name)
        fname_no_parenthesis = re.sub(r"\(.*$", "", fname_no_extension)
        file_names.append(fname_no_parenthesis)
# Colors list
num_track = len(file_names)
def map_color(e):
    hue = float(e) / (num_track+1)
    saturation = float(180. / 360)
    lightness = float(200. / 360)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return '#%02x%02x%02x' % (int(r * 256), int(g * 256), int(b * 256))
colors = [map_color(e) for e in range(len(file_names))]

# Define a function for labelling the data
def labelling(time):
    i = 0
    while(time_pointer[i] <= time):
        i += 1
    return file_names[i], colors[i]


# Plot
print "Plot data"
TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"

source = ColumnDataSource(
    data=dict(
        x=data[:,0],
        y=data[:, 1],
        label=[labelling(e)[0] for e in range(data.shape[0])],
        color=[labelling(e)[1] for e in range(data.shape[0])]
    )
)

p = figure(title="Embedding for similarity/dissimilarity (reduced to 2 dim with t-SNE)", tools=TOOLS)
p.circle('x', 'y', color='color', line_width=2, source=source)

hover = p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    # ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ("piece", "@label"),
])

show(p)
