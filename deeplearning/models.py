import torch
import torch.nn as nn

    
class CNNPlus(nn.Module):
    '''
    CNN model to learn gene expression from DNA sequences and ATAC signals.
    '''
    def __init__(self, dropout = 0.5):
        super(CNNPlus, self).__init__()

        self.cnn_1 = nn.Conv1d(4, 16, kernel_size = 4, dilation = 1, padding = 0)
        self.cnn_2 = nn.Conv1d(16, 16, kernel_size = 4, dilation = 2, padding = 3) # L_in = 497, L_out = 497
        self.cnn_3 = nn.Conv1d(16, 16, kernel_size = 4, dilation = 4, padding = 6)
        self.cnn_4 = nn.Conv1d(16, 16, kernel_size = 4, dilation = 8, padding = 12)

        self.cnn_flat = nn.Flatten() # output length = 497 * 16 = 7952
        self.layer1 = nn.Linear(7952, 1000)
        self.layer2 = nn.Linear(1000, 50)

        self.shared1 = nn.Linear(51, 30)
        self.shared2 = nn.Linear(30, 10)
        
        self.output_rna = nn.Linear(10, 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, seq, atac):

        x1 = self.dropout(self.relu(self.cnn_1(seq)))
        x2 = self.relu(self.cnn_2(x1))
        x3 = self.relu(self.cnn_3(x2))
        x4 = self.relu(self.cnn_4(x3))

        x5 = self.dropout(self.cnn_flat(x4))
        x5 = self.dropout(self.relu(self.layer1(x5)))
        x5 = self.dropout(self.relu(self.layer2(x5)))

        x_shared = torch.cat((x5, torch.unsqueeze(atac, 1)), 1)
        x_shared = self.relu(self.shared1(x_shared))
        x_shared = self.relu(self.shared2(x_shared))

        return self.output_rna(x_shared)
    