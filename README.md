# pytorch-mdn

This repo contains the code for [mixture density networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf).

Some of helper functions to draw plots are based on: http://edwardlib.org/tutorials/mixture-density-network

## Usage:

```python
import torch.nn as nn
import torch
import torch.optim as optim
import mdn
import numpy as np
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, N)
  r_data = np.random.normal(size=N)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  x_data = x_data.reshape((N, 1))
  return train_test_split(x_data, y_data, random_state=42)

N = 2000
in_feat = 1
h_size = 40
out_feat = 1
n_component = 20
n_epoch = 20
batch_size = 100

# initialize the model
model = nn.Sequential(
    nn.Linear(in_feat, h_size),
    nn.Tanh(),
    nn.Linear(h_size, h_size),
    nn.Tanh(),        
    mdn.MDN(h_size, out_feat, n_component)
)

optimizer = optim.Adam(model.parameters())

X_train, X_test, Y_train, Y_test = build_toy_dataset(N)

X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)

X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)

X_train = X_train.view(-1, batch_size, in_feat)
Y_train = Y_train.view(-1, batch_size, in_feat)

X_test = X_test.view(-1, batch_size, in_feat)
Y_test = Y_test.view(-1, batch_size, in_feat)

print("X, Y train: ", X_train.size(), Y_train.size())

#x = torch.rand(n_batch, batch_size, in_feat)
#y = torch.rand(n_batch, batch_size, out_feat)

# train the model
for epoch in range(0, n_epoch):
	e_loss = 0.0
	for minibatch, labels in zip(X_train, Y_train):
	       

	    minibatch = Variable(minibatch).float()
	    labels = Variable(labels).float()

	    model.zero_grad()

	    pi, sigma, mu = model(minibatch)
	    loss = mdn.mdn_loss(pi, sigma, mu, labels)
	    loss.backward()
	    optimizer.step()
	    e_loss += loss.data[0]

	print("epoch: ", epoch, ": ", e_loss)

# sampleing new points from the trained model
# first predict parameters from MDN
pi, sigma, mu = model(Variable(X_test[0]).float())
#sample 100 instances using parameters
samples_0 = mdn.sample(pi, sigma, mu)

print("samples: ", samples_0.size())
print("done!")
```

