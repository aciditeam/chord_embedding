#!/usr/bin/env python
# -*- coding: utf8 -*-

# Bokeh
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from collections import OrderedDict
from bokeh.plotting import *

# Numpy
import numpy as np
import unicodecsv as csv

import re

import tSNE

track_names = ["Haydn_Symph104_ii(1-8)_ORCH+REDUC+piano.mid", "Haydn_Symph104_ii(1-8)_ORCH+REDUC+piano.mid", "Haydn_Symph104_ii(1-8)_ORCH+REDUC+piano.mid"]
trace_length = 20
ratio_tracks = [0.1, 0.2, 0.4]  # Portion track
# Color list
num_track = len(track_names)
colors = colors = [tSNE.map_color(e, num_track) for e in range(len(track_names))]

# Load t-SNE
print "Load t-SNE"
data = np.loadtxt("tsne.csv", delimiter=",")

# Get files names
file_names = []
with open("../final_data/track_list2.csv", "rb") as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        file_names.append(row[0])

# New track indices
tp_ne = tSNE.get_tp_ne()

# Plot path
print "Plot data"
TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
p = figure(title="Trace of time series in t-SNE space. {} points".format(trace_length), tools=TOOLS)
for track_name, ratio_track, color in zip(track_names, ratio_tracks, colors):
    # Extract t-SNE path for the requested file
    # Get index from file name
    index_file = None
    for num, file_name in enumerate(file_names):
        fname = re.split(ur"/", file_name, flags=re.I|re.U)[-1]
        if fname == track_name:
            index_file = num
            break

    if index_file == 0:
        # First track
        start_track = 0
    else:
        start_track = int(tp_ne[index_file])
    if index_file == len(file_names):
        # Last track
        end_track = data.shape[0]
    else:
        end_track = int(tp_ne[index_file])

    # Extract track
    data_track = data[start_track:end_track,:]
    # Extract path
    N_track = data_track.shape[0]
    start_path = int(ratio_track*N_track)
    end_path = min(start_path+trace_length, N_track)
    path = data_track[start_path:end_path]

    extension = ur"\.mid$"
    fname_no_extension = re.sub(extension, "", row[0], re.U)
    fname_no_parenthesis = re.sub(ur"\(.*$", "", fname_no_extension, re.U)

    source = ColumnDataSource(
        data=dict(
            x=path[:,0],
            y=path[:,1],
            label=[fname_no_parenthesis for t in range(path.shape[0])],
            ratio=[str(ratio_track) for t in range(path.shape[0])],
            time=[t for t in range(path.shape[0])]
        )
    )

    p.line('x', 'y', color=color, line_width=2, source=source)
    p.circle('x', 'y', color=color, line_width=2, source=source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        # ("index", "$index"),
        ("(x,y)", "(@x, @y)"),
        ("label", "@label"),
        ("ratio", "@ratio"),
        ("time", "@time"),
    ])

show(p)
