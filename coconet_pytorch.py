import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from model import Net, device
from data_manipulation import I, T, P


#######################################################################
# prep for displaying heatmaps representing training progress

soprano_probs = []
alto_probs = []
tenor_probs = []
bass_probs = []

def return_probs(y, C):
    compressed = y.reshape(-1)
    x = np.zeros((I*T, P))
    r = np.arange(I*T)
    x[r, compressed] = 1
    x = x.reshape(I, T, P)
    x = torch.tensor(x).type(torch.FloatTensor).to(device)
    x = x.view(1, I, T, P)
    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    model.eval()
    with torch.no_grad():
        out = model.forward(x, C2).view(I, T, P).cpu().numpy().transpose(2, 0, 1)
        probs = np.exp(out)/np.sum(np.exp(out), axis=0)
        return probs.transpose(1, 2, 0)

def store_heatmaps(x, C):
    # stores the heatmaps in the arrays soprano_probs, alto_probs, tenor_probs, bass_probs
    model.eval()
    with torch.no_grad():
        probs = return_probs(x, C)
        soprano_probs.append(probs[0].transpose())
        alto_probs.append(probs[1].transpose())
        tenor_probs.append(probs[2].transpose())
        bass_probs.append(probs[3].transpose())
        
def display_heatmaps():
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np.flip(soprano_probs[-1], axis=0), cmap='hot', interpolation='nearest')
    axs[0].set_title('soprano')
    axs[1].imshow(np.flip(alto_probs[-1], axis=0), cmap='hot', interpolation='nearest')
    axs[1].set_title('alto')
    axs[2].imshow(np.flip(tenor_probs[-1], axis=0), cmap='hot', interpolation='nearest')
    axs[2].set_title('tenor')
    axs[3].imshow(np.flip(bass_probs[-1], axis=0), cmap='hot', interpolation='nearest')
    axs[3].set_title('bass')
    fig.set_figheight(5)
    fig.set_figwidth(15)
    plt.show()

goldberg_like_line = [67, 67, 67, 67, 67, 67, 67, 67, 71, 71, 71, 71, 71, 71, 71, 71,
                      69, 69, 69, 69, 67, 67, 66, 66, 64, 64, 64, 64, 62, 62, 62, 62]    
    
goldberg_like_line_down = [37, 37, 37, 37, 37, 37, 37, 37, 41, 41, 41, 41, 41, 41, 41, 41,
                      39, 39, 39, 39, 37, 37, 36, 36, 34, 34, 34, 34, 32, 32, 32, 32]
#######################################################################





train_tracks = np.load('train_tracks.npy')

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
losses = []

model.train()
batch_size=24 # batch size
N = batch_size

for i in range(30000):
    # tensor of shape (N, I, T)
    C = np.random.randint(2, size=(N, I, T))
      
    # batch is an np array of shape (N, I, T), entries are integers in [0, P)
    indices = np.random.choice(train_tracks.shape[0], size=N)
    batch = train_tracks[indices]    
    
    # targets is of shape (N*I*T)
    targets = batch.reshape(-1)
    targets = torch.tensor(targets).to(device)
    
    # x is of shape (N, I, T, P)
    
    batch = batch.reshape(N*I*T)
    x = np.zeros((N*I*T, P))
    r = np.arange(N*I*T)
    x[r, batch] = 1
    x = x.reshape(N, I, T, P)
    x = torch.tensor(x).type(torch.FloatTensor).to(device)

    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    out = model(x, C2)
    out = out.view(N*I*T, P)
    loss = loss_fn(out, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % 50 == 0:
        print(i)
        print('loss: ', loss.item())
        D0 = np.ones((1, T))
        D1 = np.zeros((3, T))
        D = np.concatenate([D0, D1], axis=0).astype(int)
        y = np.random.randint(P, size=(I, T))
        y[0, :] = np.array(goldberg_like_line_down)
        store_heatmaps(y, D)
        display_heatmaps()
        if i % 500 == 0:
            model.save_midi(goldberg_like_line, i)
        model.train()
        
    # adjust learning rate    
    if i % 5000 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= .75

torch.save(model.state_dict(), 'coconet_model.pt')
plt.plot(losses)