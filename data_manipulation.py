import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
import mido
import time
import os

# get data from Jsb16thSeparated.npz file
data = np.load('./Jsb16thSeparated.npz', encoding='bytes', allow_pickle=True)

# transpose chorales
all_tracks = []
for x in data.files:
    for y in data[x]:
        for i in range(-6, 6):
            all_tracks.append(y + i)

print(len(all_tracks))

# determine highest and lowest pitches
max_midi_pitch = -np.inf
min_midi_pitch = np.inf
for x in all_tracks:
    if x.max() > max_midi_pitch:
        max_midi_pitch = int(x.max())
    if x.min() < min_midi_pitch:
        min_midi_pitch = int(x.min())
        
print(max_midi_pitch, min_midi_pitch)

# set global variables
I = 4 # number of voices
T = 32 # number of 16th notes per sample
P = max_midi_pitch - min_midi_pitch + 1 # number of distinct pitches

# prepare the training dataset by cutting chorales in 2 measure pieces
# I do not split off a validation set, but you could do so here
train_tracks = []

for track in all_tracks:
    track = track.transpose()
    cut = 0
    while cut < track.shape[1]-T:
        if (track[:, cut:cut+T] > 0).all():
            train_tracks.append(track[:, cut:cut+T] - min_midi_pitch)
        cut += T
        
train_tracks = np.array(train_tracks).astype(int)
np.save('train_tracks.npy', train_tracks)