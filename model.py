import numpy as np
import torch
import torch.nn as nn
from data_manipulation import I, P, T
import os
from midi_utils import piano_roll_to_midi

device = 'cuda:0'

def pad_number(n):
    if n == 0:
        return '00000'
    else:
        digits = int(np.ceil(np.log10(n)))
        pad_zeros = 5 - digits
        return '0'*pad_zeros + str(n)

hidden_size = 32

class Unit(nn.Module):
    def __init__(self):
        super(Unit, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.batchnorm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = y + x
        y = self.relu2(y)
        return y
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.initial_conv = nn.Conv2d(2*I, hidden_size, 3, padding=1)
        self.initial_batchnorm = nn.BatchNorm2d(hidden_size)
        self.initial_relu = nn.ReLU()
        self.unit1 = Unit()
        self.unit2 = Unit()
        self.unit3 = Unit()
        self.unit4 = Unit()
        self.unit5 = Unit()
        self.unit6 = Unit()
        self.unit7 = Unit()
        self.unit8 = Unit()
        self.unit9 = Unit()
        self.unit10 = Unit()
        self.unit11 = Unit()
        self.unit12 = Unit()
        self.unit13 = Unit()
        self.unit14 = Unit()
        self.unit15 = Unit()
        self.unit16 = Unit()
        self.affine = nn.Linear(hidden_size*T*P, I*T*P)
        
    def forward(self, x, C):
        # x is a tensor of shape (N, I, T, P)
        # C is a tensor of 0s and 1s of shape (N, I, T)
        # returns a tensor of shape (N, I, T, P)
        
        # get the number of batches
        N = x.shape[0]
        
        # tile the array C out of a tensor of shape (N, I, T, P)
        tiled_C = C.view(N, I, T, 1)
        tiled_C = tiled_C.repeat(1, 1, 1, P)
        
        # mask x and combine it with the mask to produce a tensor of shape (N, 2*I, T, P)
        y = torch.cat((tiled_C*x, tiled_C), dim=1)
        
        # apply the convolution and relu layers
        y = self.initial_conv(y)
        y = self.initial_batchnorm(y)
        y = self.initial_relu(y)
        y = self.unit1(y)
        y = self.unit2(y)
        y = self.unit3(y)
        y = self.unit4(y)
        y = self.unit5(y)
        y = self.unit6(y)
        y = self.unit7(y)
        y = self.unit8(y)
        y = self.unit9(y)
        y = self.unit10(y)
        y = self.unit11(y)
        y = self.unit12(y)
        y = self.unit13(y)
        y = self.unit14(y)
        y = self.unit15(y)
        y = self.unit16(y)
            
        # reshape before applying the fully connected layer
        y = y.view(N, hidden_size*T*P)
        y = self.affine(y)
        
        # reshape to (N, I, T, P)
        y = y.view(N, I, T, P)
                
        return y
    
    def pred(self, y, C):
        # y is an array of shape (I, T) with integer entries in [0, P)
        # C is an array of shape (I, T) consisting of 0s and 1s
        # the entries of y away from the support of C should be considered 'unknown'
        
        # x is shape (I, T, P) one-hot representation of y
        compressed = y.reshape(-1)
        x = np.zeros((I*T, P))
        r = np.arange(I*T)
        x[r, compressed] = 1
        x = x.reshape(I, T, P)
        
        # prep x and C for the plugging into the model
        x = torch.tensor(x).type(torch.FloatTensor).to(device)
        x = x.view(1, I, T, P)
        C2 = torch.tensor(C).type(torch.FloatTensor).view(1, I, T).to(device)
        
        # plug x and C2 into the model
        with torch.no_grad():
            out = self.forward(x, C2).view(I, T, P).cpu().numpy()
            out = out.transpose(2, 0, 1) # shape (P, I, T)
            probs = np.exp(out) / np.exp(out).sum(axis=0) # shape (P, I, T)
            cum_probs = np.cumsum(probs, axis=0) # shape (P, I, T)
            u = np.random.rand(I, T) # shape (I, T)
            return np.argmax(cum_probs > u, axis=0)         
            
    # harmonize a melody
    # melody is a list of midi pitches (so 60 is middle C, etc) of length T
    # e.g.:
    # melody = [66, 66, 66, 66, 71, 71, 71, 71, 73, 73, 73, 73, 75, 75, 75, 75,
    #         76, 76, 75, 73, 71, 71, 75, 75, 73, 73, 70, 70, 71, 71, 71, 71]
    def harmonize(self, melody):
        x = np.random.randint(P, size=(I, T))
        x[0] = np.array(melody)-30
        C0 = np.ones((1, T)).astype(int)
        C1 = np.zeros((3, T)).astype(int)
        C = np.concatenate([C0, C1], axis=0)
        self.eval()
        with torch.no_grad():
            C2 = C.copy()
            num_steps = int(2*I*T)
            alpha_max = .999
            alpha_min = .001
            eta = 3/4
            for i in range(num_steps):
                p = np.maximum(alpha_min, alpha_max - i*(alpha_max-alpha_min)/(eta*num_steps))
                sampled_binaries = np.random.choice(2, size = C.shape, p=[p, 1-p])
                C2 += sampled_binaries
                C2[C==1] = 1
                x_cache = x
                x = self.pred(x, C2)
                x[C2==1] = x_cache[C2==1]
                C2 = C.copy()
            return x

    def save_midi(melody, id_number):
        prediction = self.harmonize(melody) + 30
        prediction = prediction.transpose().tolist()
        prediction = np.array(prediction)
        midi_output = piano_roll_to_midi(prediction)
        if not os.path.exists('midi_progress'):
            os.mkdir('midi_progress')
        midi_output.save('midi_progress/' + str(pad_number(id_number)) + 'midi.mid')