import torch
from torch import nn
import pandas as pd
import numpy as np



class data_loader:
    def __init__(self, train_test_files, batch_pct):
        self.X_train = torch.tensor(pd.read_csv(train_test_files['x_train']).values, dtype=torch.float)
        self.X_test = torch.tensor(pd.read_csv(train_test_files['x_test']).values, dtype=torch.float)
        self.y_train = torch.tensor(pd.read_csv(train_test_files['y_train']).values, dtype=torch.float)
        self.y_test = torch.tensor(pd.read_csv(train_test_files['y_test']).values, dtype=torch.float)
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
    loss_fn = torch.nn.BCEWithLogitsLoss()
    device = 'cuda:'+str(p['thread_id'])

    # everything in hparameters except the lats one is for architecture
    model = NeuralNetwork(p['input_layer_size'], p['hparameters'][:-1], n_outputs=p['output_layer_size'])
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=p['hparameters'][-1])
    loss_history = []
    model.train()

    loader = data_loader(p['train_test_files'], p['batch_size'])
    
    for i in range(p['train_iterations']):
        for batch in range(p['n_batches']):
            X_train, y_train = loader.get_batch(batch)

            X = X_train.to(device)
            y = y_train.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(loss.item())

    X_test, y_test = loader.get_batch(-1)

    model.eval()

    X = X_test.to(device)
    y = y_test.to(device)
    pred = model(X)

    torch.cuda.empty_cache()

    return 1-loss.item()








