import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import random

window_size = 100
batch_size = 10
seq_len = 1000
tot_indivs = 2504
test_indivs = 500
train_indivs = 2004
input_dim = 5
hidden_size = 16
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

    def forward(self, x):
        # change if num_layers changes
        h0 = torch.zeros(6, x.size(0), self.hidden_size)
        c0 = torch.zeros(6, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # We'll see if this is a bad idea
        #out = out[:,:, :self.hidden_size] + out[:,:, self.hidden_size: ]
        #print(out.size())
        jlen = out.size(0)
        ilen = out.size(1)
        print(out.size())
        #print('{} {}'.format(jlen, ilen))
        #out = [[self.fc(out[jj, ii, :]) for ii in range(ilen)] for jj in range(jlen)]
        #out = out.reshape(x.size(0) * x.size(1), -1)
        print(out.size())
        print("HELLO")
        pred = self.fc(out)
        print(pred.size())
        return pred
        #return out

'''
def load_batch(X, Y, window_size=100, batch_size=10):
    Xs = np.zeros((window_size, batch_size, input_dim))
    Ys = np.zeros((window_size, batch_size))
    idxs = random.sample(range(len(Xs[0])), batch_size'''

criterion = nn.CrossEntropyLoss()
learning_rate = .001
model = LSTM(input_dim, hidden_size, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#xx = np.array(X_train[0:50][0:10])
print(X_train.shape)
print(X_train[0:window_size,0:batch_size].shape)
print('-----')
xt = torch.tensor(X_train[0:window_size, 0:batch_size]).float()
print(xt.size())
outputs = model(xt)
print(outputs)
print(outputs.size())
yy = torch.tensor(Y_train[0:window_size, 0:batch_size]).type(torch.LongTensor)
print(yy)
print('SuP')
loss_sum = []
for idx, o in enumerate(outputs):
    loss_sum.append(criterion(o, yy[idx]))
loss = sum(loss_sum)
print(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()

outputs = model(xt)
print(outputs)
print(outputs.size())
yy = torch.tensor(Y_train[0:window_size, 0:batch_size]).type(torch.LongTensor)
print(yy)
print('SuP')
loss_sum = []
for idx, o in enumerate(outputs):
    loss_sum.append(criterion(o, yy[idx]))
loss = sum(loss_sum)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss)
loss = torch.sum(criterion(outputs, yy), dim = 1)
loss_sum = criterion(outputs, torch.tensor(Y_train[0:window_size, 0:batch_size]).type(torch.LongTensor))

for idx0, o in enumerate(outputs):
    #print(o)
    print(o[0].tolist())
    for idx1, p in enumerate(o):
        print(p)
        print(p.data.size())
        print(Y_train[idx0])
        print(Y_train[idx0, idx1])
        yy = torch.tensor([Y_train[idx0, idx1]]).type(torch.LongTensor)
        print(yy.size())
        loss_sum += criterion(p.data, yy) #torch.tensor(Y_train[idx0, idx1]).type(torch.LongTensor))
    #otensor = torch.tensor([x.tolist() for x in o]).float()
    #####otensor = torch.tensor([x[0:output_dim] for x in o])
    #print(otensor.size())
    #print(otensor)
    '''yy = torch.tensor(Y_train[idx0, 0:batch_size]).type(torch.LongTensor)
    print(yy.size())
    print(yy)
    loss_sum += criterion(o, yy)'''
print(loss_sum)
optimizer.zero_grad()
loss_sum.backward()
optimizer.step()
#print(outputs[0:4])
jlen = len(outputs)
ilen = len(outputs[0])
predicted = [[torch.argmax(outputs[j][i].data) for i in range(ilen)] for j in range(jlen)]
print(outputs[0][0])
print(predicted[0])
#loss_sum = 0.0
'''for idx0, pred in enumerate(predicted):
    for idx1, p in enumerate(pred):
        loss = criterion(p, '''
#print(outputs[0])
#print(predicted)
#lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=True)
#print(lstm(xt))
