import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

Data = pd.read_csv('PreprocessedDataset.csv')

Data_valid = pd.read_csv('OneSongPreprocessed.csv') 

X = Data.drop(['mark','Unnamed: 0'],axis=1)
X_valid = Data_valid.drop(['id','Unnamed: 0'],axis=1)

y = Data['mark']

y = y.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1234)

X_train = torch.from_numpy(np.array(X_train).astype(np.float32))
X_valid = torch.from_numpy(np.array(X_valid).astype(np.float32))
X_test = torch.from_numpy(np.array(X_test).astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

n_features = X_train.shape[1]

y_train = y_train.view(y_train.shape[0],1) #multiple rows, only 1 column
y_test = y_test.view(y_test.shape[0],1)

print(X_train)

# 1) model

# f = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()

        self.l1 = nn.Linear(n_input_features, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 1)

    def forward(self,x):
        y_predicted1 = torch.sigmoid(self.l1(x))
        y_predicted2 = torch.sigmoid(self.l2(y_predicted1))
        y_predicted3 = torch.sigmoid(self.l3(y_predicted2))

        return torch.sigmoid(y_predicted3)

model = LogisticRegression(n_features)

# 2) Loss and optimizer
criterion  = nn.MSELoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# 3) Train loop

n_iterations = 2000

X_accuracy = []
Y_accuracy = []

def test_accuracy():
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predictedcls = y_predicted.round() #As the y it's a probability, we can round it. In this case, as our model has
        # multiple samples, the y_predicted is a tensor with a multiple rows and in each one has its y result.

        y_testcls = y_test.round()

        acc = y_predictedcls.eq(y_testcls).sum() / y_test.shape[0]

        return acc

for epoch in range(n_iterations):
    y_pred = model.forward(X_train)
    df = pd.DataFrame(y_pred.detach().numpy(), columns=['columna'])
    loss = criterion(y_pred,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 2 == 0:
        X_accuracy.append(epoch)
        Y_accuracy.append(test_accuracy().item())
    
    if epoch % (500) == 0 or epoch==n_iterations:
        print(f'loss = {loss:.8f}')

X_accuracy = np.array(X_accuracy)
Y_accuracy = np.array(Y_accuracy)
plt.plot(X_accuracy,Y_accuracy)
plt.show()

with torch.no_grad():
    y_predicted = model(X_test)
    y_predictedcls = y_predicted.round() #As the y it's a probability, we can round it. In this case, as our model has
    # multiple samples, the y_predicted is a tensor with a multiple rows and in each one has its y result.

    y_testcls = y_test.round()
    acc = y_predictedcls.eq(y_testcls).sum() / y_test.shape[0]

    print(f'accuracy = {acc:.4f}')

with torch.no_grad():
    y_predicted = model(X_valid)

    print('y_valid',y_predicted)