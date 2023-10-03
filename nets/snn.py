import snntorch as snn
import torch.nn as nn
import torch

class SpikingNN(nn.Module):
    def __init__(self, input_size=6, hidden_size1=10, hidden_size2=10, hidden_size3=10, output_size=3):
        super(SpikingNN, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lif1 = snn.Leaky(beta = 0.85)

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta = 0.85)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.lif3 = snn.Leaky(beta = 0.85)

        self.fc_out = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif2.init_leaky()

        spk2_rec = []  
        mem2_rec = []  
        
        for step in range(100):
                cur1 = self.fc1(x.flatten(1))
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
                cur4= self.fc_out(spk3)

                spk2_rec.append(spk2)
                mem2_rec.append(mem2)
        
        return torch.stack(mem2_rec)
