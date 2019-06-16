import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import random
import torch.distributions as torchdist

window_size = 50
batch_size = 10
seq_len = 1000
tot_indivs = 2504
test_indivs = 500
train_indivs = 2004
input_dim = 5
hidden_size = 18
output_dim = 3
masked_indices = []
X_train = np.zeros((seq_len, train_indivs, input_dim))
X_test = np.zeros((seq_len, test_indivs, input_dim))
#Y_train = np.zeros((seq_len, train_indivs, output_dim))
#Y_test = np.zeros((seq_len, test_indivs, output_dim))
Y_train = np.zeros((seq_len, train_indivs))
Y_test = np.zeros((seq_len, test_indivs))
input_encode = np.eye(input_dim)
output_encode = np.eye(output_dim)
input_map = {'0|0':0, '0|1':1, '1|0':2, '1|1':3, '.|.':4}
output_map = {'0|0':0, '0|1':1, '1|0':1, '1|1':2}

try:
    X_train = np.load("xtrain.npy")
    X_test = np.load("xtest.npy")
    Y_train = np.load("ytrain.npy")
    Y_test = np.load("ytest.npy")
except:
    jj = 0
    with open('subset20.vcf', 'r') as s20, open('complete20.vcf') as c20:
        while jj < 253:
            jj += 1
            s20.readline()
            c20.readline()
        jj = 0
        for l in s20:
            line = l.strip().split()
            c = c20.readline()
            if line[-1] == '.|.':
                masked_indices.append(jj)
                cline = c.strip().split()
                for ii in range(train_indivs):
                    X_train[jj][ii] = input_encode[4]
                    #Y_train[jj][ii] = output_encode[output_map[cline[ii + 9]]]
                    Y_train[jj][ii] = output_map[cline[ii + 9]]
                for ii in range(train_indivs, train_indivs + test_indivs):
                    X_test[jj][ii - train_indivs] = input_encode[4]
                    #Y_test[jj][ii - train_indivs] = output_encode[output_map[cline[ii + 9]]]
                    Y_test[jj][ii - train_indivs] = output_map[cline[ii + 9]]
            else:
                for ii in range(train_indivs):
                    X_train[jj][ii] = input_encode[input_map[line[ii + 9]]]
                    #Y_train[jj][ii] = output_encode[output_map[line[ii + 9]]]
                    Y_train[jj][ii] = output_map[line[ii + 9]]
                for ii in range(train_indivs, train_indivs + test_indivs):
                    X_test[jj][ii - train_indivs] = input_encode[input_map[line[ii + 9]]]
                    #Y_test[jj][ii - train_indivs] = output_encode[output_map[line[ii + 9]]]
                    Y_test[jj][ii - train_indivs] = output_map[line[ii + 9]]
            jj += 1
        np.save("xtrain.npy", X_train)
        np.save("xtest.npy", X_test)
        np.save("ytrain.npy", Y_train)
        np.save("ytest.npy", Y_test)
print("Processed input files")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # may experiment with num_layers parameter later
        self.lstm = nn.LSTM(input_size, hidden_size, num_classes, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_classes, batch_first=True)
        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, f):
        n = torchdist.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
        h0 = n.sample((6, x.size(0), self.hidden_size)).squeeze(3)
        c0 = n.sample((6, x.size(0), self.hidden_size)).squeeze(3)
        # change if num_layers changes
        #h0 = torch.zeros(6, x.size(0), self.hidden_size)
        #c0 = torch.zeros(6, x.size(0), self.hidden_size)
        #h0 = torch.zeros(3, x.size(0), self.hidden_size)
        #c0 = torch.zeros(3, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # We'll see if this is a bad idea
        #out = out[:,:, :self.hidden_size] + out[:,:, self.hidden_size: ]
        #print('{} {}'.format(jlen, ilen))
        #out = [[self.fc(out[jj, ii, :]) for ii in range(ilen)] for jj in range(jlen)]
        #out = out.reshape(x.size(0) * x.size(1), -1)
        #print(out.size())
        pred = self.fc(out[f, :, :])
        #print(pred)
        #pred = self.fc(out)
        return pred
        #return out

def load_batch(X, Y, start, end, batch_size=50):
    Xs = np.zeros((end - start, batch_size, input_dim))
    Ys = np.zeros((end - start, batch_size))
    idxs = random.sample(range(len(X[0])), batch_size)
    for ii, idx in enumerate(idxs):
        Xs[:, ii] = X[start:end, idx]
        Ys[:, ii] = Y[start:end, idx]
        #Ys[:, ii] = Y[start+1:end+1, idx]
    Xt = torch.tensor(Xs).float()
    Yt = torch.tensor(Ys).type(torch.LongTensor)
    return Xt, Yt
    
criterion = nn.CrossEntropyLoss()
learning_rate = .01 #01
model = LSTM(input_dim, hidden_size, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''start = 555
end = 555 + window_size'''
masked = []
for jj in range(0, len(X_train)):
    if X_train[jj, 0, 4]:
        masked.append(jj)
        #masked.append(jj - 1)
print(masked)

acc = []
for st in masked:
    tlosses = []
    start = int(st - window_size/2)
    if start < 0:
        start = 0
    if start + window_size >= len(X_train):
        start = len(X_train) - window_size
    end = start + window_size
    print("Window of length {} starting at {}".format(window_size, start))
    for i in range(100):
        xt, yt = load_batch(X_train, Y_train, start, end)
        #outputs = model(xt)
        outputs = model(xt, st - start)
        #loss_sum = []
        #print(outputs[0][0])
        '''for idx, o in enumerate(outputs):
        if (idx + start) == window_masked[0]: #in masked:
            loss_sum.append(criterion(o, yt[idx]))
        else:
            loss_sum.append(.0 * criterion(o, yt[idx]))
        loss = sum(loss_sum)'''
        loss = criterion(outputs, yt[st - start])
        tlosses.append(loss)
        if (i % 24) == 0:
            print("Training loss: {}".format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        sum = 0
        for aa in range(test_indivs):
            ts = torch.tensor(X_test[start:end,aa,:]).float()
            #print(ts.unsqueeze(1).size())
            #op = model(ts.unsqueeze(1))
            op = model(ts.unsqueeze(1), st - start)
            if int(torch.argmax(op)) == int(Y_test[st][aa]):
                sum += 1
            #print(ts)
            #print(op)
            '''
            for ii in masked:
                if ii != masked[0]:
                    continue
                if ii < start or ii >= end:
                    continue
                #print(Y_test[ii][aa])
                #print(int(torch.argmax(op)) == int(Y_test[ii][aa]))
                if int(torch.argmax(op)) == int(Y_test[ii][aa]):
                    sum += 1'''
        print("Test accuracy: {}".format(1.0*sum/test_indivs))
        tp = (st, (1.0*sum/test_indivs))
        #print(tp)
        acc.append(tp)
np.save("accs.npy", np.array(acc))
        #if aa == 99:
        #    print(op)
#print(outputs[0])
#print(predicted)
#lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
#print(lstm(xt))
