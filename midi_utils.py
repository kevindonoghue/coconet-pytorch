import numpy as np
import os
from midi2audio import FluidSynth
from IPython.display import Audio, display
from data_manipulation import I, P, T
import matplotlib.pyplot as plt
import mido

def play_midi(path):
    if os.path.exists('test.wav'):
        os.remove('test.wav')
    FluidSynth('font.sf2').midi_to_audio(path, 'test.wav')
    audio = Audio('test.wav')
    display(audio)


# function for converting arrays of shape (T, 4) into midi files
# the input array has entries that are np.nan (representing a rest)
# of an integer between 0 and 127 inclusive
def piano_roll_to_midi(piece):

    piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0)

    bpm = 50
    microseconds_per_beat = 60 * 1000000 / bpm

    mid = mido.MidiFile()
    tracks = {'soprano': mido.MidiTrack(), 'alto': mido.MidiTrack(),
              'tenor': mido.MidiTrack(), 'bass': mido.MidiTrack()}
    past_pitches = {'soprano': np.nan, 'alto': np.nan,
                    'tenor': np.nan, 'bass': np.nan}
    delta_time = {'soprano': 0, 'alto': 0, 'tenor': 0, 'bass': 0}

    # create a track containing tempo data
    metatrack = mido.MidiTrack()
    metatrack.append(mido.MetaMessage('set_tempo',
                                      tempo=int(microseconds_per_beat), time=0))
    mid.tracks.append(metatrack)

    # create the four voice tracks
    for voice in tracks:
        mid.tracks.append(tracks[voice])
        tracks[voice].append(mido.Message(
            'program_change', program=52, time=0))

    # add notes to the four voice tracks
    for i in range(len(piece)):
        pitches = {'soprano': piece[i, 0], 'alto': piece[i, 1],
                   'tenor': piece[i, 2], 'bass': piece[i, 3]}
        for voice in tracks:
            if np.isnan(past_pitches[voice]):
                past_pitches[voice] = None
            if np.isnan(pitches[voice]):
                pitches[voice] = None
            if pitches[voice] != past_pitches[voice]:
                if past_pitches[voice]:
                    tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
                if pitches[voice]:
                    tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
            past_pitches[voice] = pitches[voice]
            # 480 ticks per beat and each line of the array is a 16th note
            delta_time[voice] += 120

    return mid

class Chorale:
    def __init__(self, arr, subtract_30=False):
        # arr is an array of shape (4, 32) with values in range(0, 57)
        self.arr = arr.copy()
        if subtract_30:
            self.arr -= 30
            
        # the one_hot representation of the array
        reshaped = self.arr.reshape(-1)
        self.one_hot = np.zeros((I*T, P))
        r = np.arange(I*T)
        self.one_hot[r, reshaped] = 1
        self.one_hot = self.one_hot.reshape(I, T, P)
        

    def to_image(self):
        # visualize the four tracks as 0-1 heatmaps
        soprano = self.one_hot[0].transpose()
        alto = self.one_hot[1].transpose()
        tenor = self.one_hot[2].transpose()
        bass = self.one_hot[3].transpose()
        
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(np.flip(soprano, axis=0), cmap='hot', interpolation='nearest')
        axs[0].set_title('soprano')
        axs[1].imshow(np.flip(alto, axis=0), cmap='hot', interpolation='nearest')
        axs[1].set_title('alto')
        axs[2].imshow(np.flip(tenor, axis=0), cmap='hot', interpolation='nearest')
        axs[2].set_title('tenor')
        axs[3].imshow(np.flip(bass, axis=0), cmap='hot', interpolation='nearest')
        axs[3].set_title('bass')
        fig.set_figheight(5)
        fig.set_figwidth(15)
        return fig, axs
    
    def play(self, filename='midi_track.mid'):
        # display an in-notebook widget for playing audio
        # saves the midi file as a file named name in base_dir/midi_files
        
        midi_arr = self.arr.transpose().copy()
        midi_arr += 30
        midi = piano_roll_to_midi(midi_arr)
        if not os.path('midi_files').exists():
            os.mkdir('midi_files')
        midi.save('midi_files/' + filename)
        play_midi('midi_files/' + filename)