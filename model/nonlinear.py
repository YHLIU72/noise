import torch
from torch import nn

class NonLinear(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=1, num_sheets=4):
        super(NonLinear, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_sheets)
        self.num_sheets = num_sheets
        self.out_nc = out_nc

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        return output
    

class NonLinearType(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18):
        super(NonLinearType, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)


    def forward(self, inp, type_):
        type_ = torch.LongTensor([type_])
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output
    
class NonLinearBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, num_bins=25):
        super(NonLinearBin, self).__init__()
        self.num_bins = num_bins
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, num_bins)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        return output

class NonLinearTypeBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_bins=51):
        super(NonLinearTypeBin, self).__init__()
        self.num_bins = num_bins
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc*num_bins)

    def forward(self, inp, type_):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output
    
class NonLinearMultiBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=1):
        super(NonLinearMultiBin, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)
        self.out0 = nn.Linear(nc, 18)
        self.out1 = nn.Linear(nc, 6)
        self.out2 = nn.Linear(nc, 4)


    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        output0 = self.out0(x)
        output1 = self.out1(x)
        output2 = self.out2(x)
        return output, output0, output1, output2

class NonLinearTypeBinModel(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_bins=26, num_sheets=4):
        super(NonLinearTypeBinModel, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_bins * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc, self.num_bins)
        return output

class NonLinearTypeBinModelModify(nn.Module):
    def __init__(self, in_nc=6, nc=1600, bowl_idx=2, num_bins=26, num_sheets=4):
        super(NonLinearTypeBinModelModify, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.bowl_idx = bowl_idx
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, bowl_idx * num_bins * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.bowl_idx, self.num_bins)
        return output

class NonLinearBinModel(nn.Module):
    def __init__(self, in_nc=3, nc=1600, num_bins=51, num_sheets=4):
        super(NonLinearBinModel, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, num_sheets * num_bins) 

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.num_bins)
        return output

class NonLinearTypeModel(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_sheets=4):
        super(NonLinearTypeModel, self).__init__()
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc)
        return output

class NonLinearTypeModel_3(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_sheets=4):
        super(NonLinearTypeModel_3, self).__init__()
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden1 = nn.Linear(in_nc, 256)
        self.relu1 = nn.LeakyReLU()
        # self.dropout1 = nn.Dropout(p=0.1)
        self.hidden2 = nn.Linear(256,512)
        self.relu2 = nn.LeakyReLU()
        # self.dropout2 = nn.Dropout(p=0.1)
        self.hidden3 = nn.Linear(512, 256)
        self.relu3 = nn.LeakyReLU()
        # self.dropout3 = nn.Dropout(p=0.1)
        self.out = nn.Linear(256, out_nc * num_sheets)

    def forward(self, inp):
        x = self.relu1(self.hidden1(inp))
        # x = self.dropout1(x)  # 在第一层隐藏层输出后添加 dropout
        x = self.relu2(self.hidden2(x))
        # x = self.dropout2(x)  # 在第二层隐藏层输出后添加 dropout
        x = self.relu3(self.hidden3(x))
        # x = self.dropout3(x)  # 在第三层隐藏层输出后添加 dropout
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc)
        return output

class NonLinearTypeModelModify(nn.Module):
    def __init__(self, in_nc=6, nc=1600, out_nc=4,bowl=2):
        super(NonLinearTypeModelModify, self).__init__()
        self.out_nc = out_nc
        self.bowl = bowl
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc*bowl)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.out_nc,self.bowl)
        return output