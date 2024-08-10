import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self):
        torch.manual_seed(42)
        self.x_axis = []
        self.accuracy_test = []
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []

    def prepareData(self,test_size):
        Data = pd.read_csv('PreprocessedDataset.csv')

        Data_valid = pd.read_csv('FewSongsPreprocessed.csv') 

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

    def trainLoop(self,lr,n_epochs,print_freq,test_freq,model_path,n_batches):
        writer = SummaryWriter('run_ssrm_pt')

        n_features = self.X_train.shape[1]
        self.model = LogisticRegression(n_features)

        if model_path != '':
            self.model.load_state_dict(torch.load(model_path))

        self.criterion  = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(),lr = lr)

        self.createBatches(n_batches)

        for epoch in range(n_epochs):
            print_done = False
            test_done = False

            for batch in range(n_batches):
                X_train_batch = self.X_train_batches[batch]
                y_train_batch = self.y_train_batches[batch]

                y_pred = self.model.forward(X_train_batch)
                loss = self.criterion(y_pred,y_train_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (epoch % test_freq == 0 or epoch==n_epochs-1) and not test_done:
                    self.x_axis.append(epoch)
                    test_accuracy_v = self.test_accuracy().item()
                    test_loss_v = self.test_loss().item()
                    train_loss_v = self.train_loss().item()
                    train_accuracy_v = self.train_accuracy().item()

                    self.accuracy_test.append(test_accuracy_v)
                    self.loss_test.append(test_loss_v)
                    self.loss_train.append(train_loss_v)
                    self.accuracy_train.append(train_accuracy_v)

                    writer.add_scalar('Test/Accuracy',test_accuracy_v,epoch)
                    writer.add_scalar('Test/Loss',test_loss_v,epoch)
                    writer.add_scalar('Train/Loss',train_loss_v,epoch)
                    writer.add_scalar('Train/Accuracy',train_accuracy_v,epoch)

                    test_done=True
                
                if (epoch % print_freq == 0 or epoch==n_epochs-1) and not print_done:
                    print(f'epoch {epoch}: loss = {loss:.8f}')
                    print_done=True

    def createBatches(self,n_batches):
        self.X_train_batches = []
        self.y_train_batches = []
        total_rows = self.X_train.shape[0]
        batch_size = int(total_rows/n_batches)

        for i in range(n_batches):
            index_start = i*batch_size
            index_end = min(i*batch_size + batch_size,total_rows)

            self.X_train_batches.append(self.X_train[index_start:index_end])
            self.y_train_batches.append(self.y_train[index_start:index_end])

    def showResults(self):

        plots = {'Loss train': self.loss_train,
                 'Accuracy train': self.accuracy_train,
                 'Loss test':self.loss_test,
                 'Accuracy test':self.accuracy_test}

        for (name,plot) in plots.items():
            plt.plot(self.x_axis,plot)
            plt.grid()
            plt.title(name)
            plt.show()

        print('\n \n Valid predictions: \n')
        with torch.no_grad():
            y_valid = self.model(self.X_valid)
            results_valid = list(y_valid)

            df_few_songs = pd.read_csv('FewSongs.csv')

            names = list(df_few_songs['name'])

            for i, result in enumerate(results_valid):
                print(names[i]+': '+str(round(float(result[0])*100,2))+'% de coincidencia')

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
    
    def train_loss(self):
        with torch.no_grad():
            y_predicted = self.model(self.X_train)
            loss = self.criterion(y_predicted,self.y_train)

            return loss
    
    def train_accuracy(self):
        with torch.no_grad():
            y_predicted = self.model(self.X_train)
            y_predictedcls = y_predicted.round()
            y_traincls = self.y_train.round()

            acc = y_predictedcls.eq(y_traincls).sum() / self.y_train.shape[0]

            return acc

def addArgs(parser):
    parser.add_argument('-test_size', type=float, default=0.1, required=False, 
                        help='Number between 0 and 1 to determine size of the test when splitting the dataset')
    parser.add_argument('-lr', type=float, default=0.1, required=False, help='Learning rate')
    parser.add_argument('-n_epochs', type=int, default=20000, required=False, help='Number of epochs')
    parser.add_argument('-print_freq', type=int, default=1000, required=False, help='Number of epochs between prints')
    parser.add_argument('-test_freq', type=int, default=2000, required=False, help='Number of epochs between test')
    parser.add_argument('-model_path', type=str, default='', required=False, help='Path of the model to load')
    parser.add_argument('-n_batches', type=int, default=10, required=False, help='Number of batches')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spotify Songs Recommendation Model.')

    addArgs(parser)

    args = parser.parse_args()

    SSRM = SpotifySongsRecommendationModel()
    SSRM.prepareData(test_size=args.test_size)
    SSRM.trainLoop(lr=args.lr,n_epochs=args.n_epochs,print_freq=args.print_freq,test_freq=args.test_freq,
                   model_path=args.model_path,n_batches=args.n_batches)
    SSRM.showResults()

    model_parameters = SSRM.model.state_dict()
    torch.save(model_parameters, 'model.pt')