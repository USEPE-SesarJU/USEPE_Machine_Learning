import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from data_parser import data_import
from data_parser import data_export

class myRNN(nn.Module):
    def __init__(self, input=2, hidden=64, output=2):
        super().__init__()
        self.input_num = input
        self.hidden_num = hidden
        self.n_layers = 1

        self.rnn = nn.RNN(input, hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        inith = self.init_hiddens(x.size(0), x.device)
        x_rnn, hidden = self.rnn(x, inith)
        out = self.fc(x_rnn[:, -1, :])
        return out

    def init_hiddens(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_num).to(device)

N_TIME = 10
SCALER = MinMaxScaler()

class RegDataset(torch.utils.data.Dataset):
    def __init__(self, X, y): # y: lon & lat
        self.prepare_dataset(np.array(X, np.float32), np.array(y, np.float32))

    def prepare_dataset(self, X, y):
        # Min-Max
        y = SCALER.fit_transform(y)
        self.inputs = np.zeros((len(y)-N_TIME, N_TIME, 2))
        self.targets = np.zeros((len(y)-N_TIME, 2))
        for i in range(len(y)-N_TIME):
            self.inputs[i] = y[i:i+N_TIME].reshape(-1, 2)
            self.targets[i] = y[i+N_TIME]

        self.inputs = torch.tensor(self.inputs, dtype=torch.float)
        self.targets = torch.tensor(self.targets, dtype=torch.float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        target_lat = self.targets[index, 0]
        target_lon = self.targets[index, 1]
        return {'input':input, 'target':target,
                'target_lat':target_lat, 'target_lon':target_lon}

def RNN_regressor(result_file, X, y, gpu_no=0, batch_size=256):
    MAX_EPOCH = int(5.0e2)
    # activate gpu
    device = 'cuda:{}'.format(gpu_no) if torch.cuda.is_available() and gpu_no is not None else 'cpu'
    # activate model
    predNN = myRNN().to(device)
    opt = optim.Adam(predNN.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # train
    predNN.train()
    train_RegLoader = torch.utils.data.DataLoader(dataset=RegDataset(X, y), batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(MAX_EPOCH):
        for idx, data in enumerate(train_RegLoader):
            # Get input&target, then input to model
            input = Variable(data['input']).to(device)
            target = Variable(data['target']).to(device)

            predict = predNN(input)
            # calc loss via cost function
            loss = criterion(predict, target)
            # update
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print log
            print('Epoch:{} iter:{}/{} loss:{}'.format(epoch, idx+1, len(train_RegLoader), loss.item()))

    # test
    predNN.eval()
    future_step = 2000
    pred_route = np.zeros((len(y)-N_TIME+future_step, 2))
    # Pred previous lon & lat
    for i in range(len(y)-N_TIME):
        target = np.array(y[i+N_TIME], np.float32)
        input = np.array(y[i:i+N_TIME], np.float32)
        input = (input - SCALER.data_min_) / (SCALER.data_max_ - SCALER.data_min_)
        input = Variable(torch.FloatTensor(input)).to(device)
        input = input.unsqueeze(0)
        with torch.no_grad():
            predict = predNN(input)
        predict_lat, predict_lon = predict[:,0].data.cpu().item(), predict[:,1].data.cpu().item()
        predict = np.array([predict_lat, predict_lon]) * (SCALER.data_max_ - SCALER.data_min_) + SCALER.data_min_
        #predict = np.array([predict_lat, predict_lon])
        pred_route[i] = predict.copy()

        print('Pred with trained route\n')
        print('\t time={} target=({}), predict=({})'.format(i, target, predict))

    ml_NN = pd.DataFrame(pred_route[:len(y)-N_TIME], columns = [' lat', ' lon'])
    ml_NN['Data Type'] = 'Neural Network : RNN (trained route)'
    ml_NN.to_csv(result_file, index=False, mode='a', header=False)

    # Pred future lon & lat
    for i in range(len(y)-N_TIME, len(y)-N_TIME+future_step):
        input = np.array(pred_route[i-N_TIME:i], np.float32)
        input = (input - SCALER.data_min_) / (SCALER.data_max_ - SCALER.data_min_)
        input = Variable(torch.FloatTensor(input)).to(device)
        input = input.unsqueeze(0)
        with torch.no_grad():
            predict = predNN(input)
        predict_lat, predict_lon = predict[:,0].data.cpu().item(), predict[:,1].data.cpu().item()
        predict = np.array([predict_lat, predict_lon]) * (SCALER.data_max_ - SCALER.data_min_) + SCALER.data_min_
        #predict = np.array([predict_lat, predict_lon])
        pred_route[i] = predict.copy()

        print('Pred with future route\n')
        print('\t time={} predict=({})'.format(i, predict))

    ml_NN1 = pd.DataFrame(pred_route[len(y)-N_TIME:len(y)-N_TIME+future_step], columns = [' lat', ' lon'])
    ml_NN1['Data Type'] = 'Neural Network : RNN (future route)'
    ml_NN1.to_csv(result_file, index=False, mode='a', header=False)




file = 'TEST_LOGGER_logger_20220124_20-39-04.log'
result_file_name = (file[:-4] + '.csv')
result_file = data_export(result_file_name)

# write original route
d = data_import(file,1)
d_Lat_Lon = d[[' lat', ' lon']]
d_Lat_Lon[' Data Type'] = 'Original Route'
d_Lat_Lon.to_csv(result_file, index=False)

# Select data columns for ML training
d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]



RNN_regressor(result_file, X, y, gpu_no=0, batch_size=256)

from mapping import on_map

on_map(result_file_name)