import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,3) #number of features and only one output that is a probability
        self.linear2 = nn.Linear(3,1)

    def forward(self,x):
        y_predicted1 = torch.sigmoid(self.linear(x))
        y_predicted2 = torch.sigmoid(self.linear2(y_predicted1))
        return y_predicted2

class SpotifySongsRecommendationModel:

    def prepareData(self,test_size):
        Data = pd.read_csv('PreprocessedDataset.csv')

        Data_valid = pd.read_csv('OneSongPreprocessed.csv') 

        X = Data.drop(['mark','Unnamed: 0'],axis=1)
        X_valid = Data_valid.drop(['id','Unnamed: 0'],axis=1)

        y = Data['mark']

        y = y.values

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=1234)

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train) #to fit it in mean 0 and unit variance
        X_test = sc.transform(X_test) #uses the same mean and variance previously calculated not to disturb the results
        X_valid = sc.transform(X_valid)

        self.X_train = torch.from_numpy(X_train.astype(np.float32))
        self.X_valid = torch.from_numpy(X_valid.astype(np.float32))
        self.X_test = torch.from_numpy(X_test.astype(np.float32))

        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        self.y_train = y_train.view(y_train.shape[0],1) #multiple rows, only 1 column
        self.y_test = y_test.view(y_test.shape[0],1)

    def trainLoop(self,lr,n_iterations,print_freq,test_freq,model_path):

        n_features = self.X_train.shape[1]
        self.model = LogisticRegression(n_features)

        if model_path != '':
            self.model.load_state_dict(torch.load(model_path))

        self.criterion  = nn.MSELoss()

        optimizer = torch.optim.SGD(self.model.parameters(),lr = lr)

        self.x_axis = []
        self.accuracy_test = []
        self.loss_train = []
        self.loss_test = []

        
        for epoch in range(n_iterations):
            y_pred = self.model.forward(self.X_train)
            loss = self.criterion(y_pred,self.y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % test_freq == 0 or epoch==n_iterations-1:
                self.x_axis.append(epoch)
                self.loss_train.append(loss.item())
                self.accuracy_test.append(self.test_accuracy().item())
                self.loss_test.append(self.test_loss().item())
            
            if epoch % print_freq == 0 or epoch==n_iterations-1:
                [w,b,w,b] = self.model.parameters()
                print(f'epoch {epoch}: loss = {loss:.8f}')


    def showResults(self):
        plt.plot(self.x_axis,self.loss_train)
        plt.show()

        plt.plot(self.x_axis,self.accuracy_test)
        plt.show()

        plt.plot(self.x_axis,self.loss_test)
        plt.show()

        with torch.no_grad():
            y_valid = self.model(self.X_valid)

            print('y_valid',y_valid)

    def test_accuracy(self):
        with torch.no_grad():
            y_predicted = self.model(self.X_test)
            y_predictedcls = y_predicted.round()

            y_testcls = self.y_test.round()

            acc = y_predictedcls.eq(y_testcls).sum() / self.y_test.shape[0]

            return acc

    def test_loss(self):
        with torch.no_grad():
            y_predicted = self.model(self.X_test)

            loss = self.criterion(y_predicted,self.y_test)

            return loss


if __name__ == "__main__":
    SSRM = SpotifySongsRecommendationModel()
    SSRM.prepareData(0.1)
    SSRM.trainLoop(0.1,20000,1000,1000,'model.pt')
    SSRM.showResults()

    model_parameters = SSRM.model.state_dict()
    torch.save(model_parameters, 'model.pt')