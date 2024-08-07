import torch
from torch import nn
import pandas as pd
import numpy as np

input_layer_size = 28*28
output_layer_size = 10
n_iterations = 5
n_batches = 10
train_batch_size = 100


class data_loader:
    def __init__(self,batch_pct):
        self.X_train = pd.read_csv('X_train.csv')
        self.X_test = pd.read_csv('X_test.csv')
        self.y_train = pd.read_csv('y_train.csv')
        self.y_test = pd.read_csv('y_test.csv')
        # with open('y_train.csv') as f:
        #     self.y_train = [float(x) for x in f.read().split(',')]
        # with open('y_test.csv') as f:
        #     self.y_test = [float(x) for x in f.read().split(',')]

        self.train_batch_size = int(np.floor(self.X_train.shape[0] * batch_pct))
        self.test_batch_size = int(np.floor(self.X_test.shape[0] * batch_pct))
        self.batch_pct = batch_pct
        
    def get_batch(self,batch):
        if batch < 0:
            return self.X_test, self.y_test
        else:
            return self.X_train[(batch*self.train_batch_size):((batch+1)*self.train_batch_size)],\
                   self.y_train[(batch*self.train_batch_size):((batch+1)*self.train_batch_size)]
                

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs=None, arch=None, n_outputs=None):
        super().__init__()
        self.flatten = nn.Flatten()
        if n_inputs and arch and n_outputs:
            self.make_arch(n_inputs, arch, n_outputs)
                
    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

    def make_arch(self, n_inputs, arch, n_outputs, inner_function='Relu', last_function = None):
        layers = [torch.nn.Linear(n_inputs, arch[0])]
        for a in range(len(arch)-1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(arch[a], arch[a+1]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(arch[-1], n_outputs))
    #    layers.append(torch.nn.Softmax(dim=1))                    # works better without this

        self.stack = torch.nn.Sequential(*layers)

def instance(p):

        loss_fn = torch.nn.CrossEntropyLoss()
        device = 'cuda:'+str(p['thread_id'])
        model = NeuralNetwork(input_layer_size, p['next_points'], n_outputs=output_layer_size)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_history = []
        model.train()

        loader = data_loader(.05)
    
        for i in range(n_iterations):
            for batch in range(n_batches):
                X_train, y_train = loader.get_batch(batch)

                X_train_torch = torch.tensor(X_train.values, dtype=torch.float)
                y_train_torch = torch.tensor(y_train.values, dtype=torch.float)

                X = X_train_torch.to(device)
                y = y_train_torch.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)
        
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_history.append(loss.item())

        X_test, y_test = loader.get_batch(-1)

        model.eval()
        X_test_torch = torch.tensor(X_test.values, dtype=torch.float)
        y_test_torch = torch.tensor(y_test.values, dtype=torch.float)

        X = X_test_torch.to(device)
        y = y_test_torch.to(device)
        pred = model(X)


        return loss.item()#loss.item()#.detatch()









