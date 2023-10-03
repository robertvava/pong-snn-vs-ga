import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=10, hidden_size2=5, hidden_size3 = 3, output_size=3):  
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


# =============================================== NOT USED IN THE PROJECT =============================================== #

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
    
class PongNet(nn.Module):
    def __init__(self, input_size=6, tcn_output_size=10, hidden_size=10, output_size=3):
        super(PongNet, self).__init__()
        
        self.tcn1 = TCNBlock(input_size, tcn_output_size)
        self.tcn2 = TCNBlock(tcn_output_size, tcn_output_size)
        
        self.lstm = nn.LSTM(tcn_output_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        
        x = self.tcn1(x)
        x = self.tcn2(x)
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        # Use the last LSTM output for decision making
        x = lstm_out[:, -1, :]
        
        x = F.softmax(self.fc(x), dim=1)
        
        return x

