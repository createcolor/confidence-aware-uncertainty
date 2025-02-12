import torch
import torch.nn as nn
import torch.nn.functional as F

class AggNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample((128, 128))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 4)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

class AggNet_x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample((128, 128))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 1)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample((128, 128))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 5)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

class AggNet_x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample((128, 128))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 3)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

class AggNet_drop(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((128, 128))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_256(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((256, 256))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(4096, 400)
        self.fc2 = nn.Linear(400, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_256_x1(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((256, 256))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_512(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((512, 512))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(16384, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_512_x1(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((512, 512))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 32, 1)
        self.fc1 = nn.Linear(8192, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = self.pool(F.relu(self.conv5(output)))
        output = F.relu(self.conv6(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_512_x2(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((512, 512))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 16, 1)
        self.fc1 = nn.Linear(8192, 1000)
        self.fc2 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = self.pool(F.relu(self.conv5(output)))
        output = self.pool(F.relu(self.conv6(output)))
        output = F.relu(self.conv7(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_500(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample((500, 500))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv4 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(15376, 400)
        self.fc2 = nn.Linear(400, 1)
        
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

class AggNet_500_drop(nn.Module):
    def __init__(self, p = 0.25):
        super().__init__()
        self.ups = nn.Upsample((500, 500))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv4 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(15376, 400)
        self.fc2 = nn.Linear(400, 1)
        self.drop = nn.Dropout(p = p)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.drop(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

class AggNet_500_drop_v2(nn.Module):
    def __init__(self, p = 0.25):
        super().__init__()
        self.ups = nn.Upsample((500, 500))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 7, padding=3)
        self.conv4 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(15376, 4000)
        self.fc2 = nn.Linear(4000, 400)
        self.fc3 = nn.Linear(400, 1)
        self.drop = nn.Dropout(p = p)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.drop(self.fc1(output)))
        output = F.relu(self.drop(self.fc2(output)))
        output = torch.sigmoid(self.fc3(output))
        return output.double()


class ModelM3(nn.Module):
    '''
    Based on
    code: https://github.com/ansh941/MnistSimpleCNN/blob/master/code/models/modelM3.py
    paper: https://arxiv.org/pdf/2008.10400.pdf 
    '''
    def __init__(self):
        super(ModelM3, self).__init__()
        self.ups = nn.Upsample((512, 512))
        self.conv1 = nn.Conv2d(3, 100, 3, padding=3)       
        self.conv1_bn = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, 3, padding=3)      
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 200, 3, padding=3)      
        self.conv3_bn = nn.BatchNorm2d(200)
        self.conv4 = nn.Conv2d(200, 250, 1)      
        self.conv4_bn = nn.BatchNorm2d(250)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(272250, 5000, bias=False)
        self.fc2 = nn.Linear(5000, 500, bias=False)
        self.fc3 = nn.Linear(500, 1)
        
    def forward(self, x):
        x = x.float()
        x = self.ups(x)
        output = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        output = self.pool(F.relu(self.conv2_bn(self.conv2(output))))
        output = self.pool(F.relu(self.conv3_bn(self.conv3(output))))
        output = self.pool(F.relu(self.conv4_bn(self.conv4(output))))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.sigmoid(output)
        return output.double()
    

class ModelM3_v1(nn.Module):
    '''
    Based on
    code: https://github.com/ansh941/MnistSimpleCNN/blob/master/code/models/modelM3.py
    paper: https://arxiv.org/pdf/2008.10400.pdf 
    '''
    def __init__(self):
        super(ModelM3_v1, self).__init__()
        self.ups = nn.Upsample((512, 512))
        self.conv1 = nn.Conv2d(3, 30, 3, padding=3)       
        self.conv1_bn = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=3)      
        self.conv2_bn = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 70, 3, padding=3)      
        self.conv3_bn = nn.BatchNorm2d(70)
        self.conv4 = nn.Conv2d(70, 90, 1)      
        self.conv4_bn = nn.BatchNorm2d(90)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(98010, 5000, bias=False)
        self.fc2 = nn.Linear(5000, 500, bias=False)
        self.fc3 = nn.Linear(500, 1)
        
    def forward(self, x):
        x = x.float()
        x = self.ups(x)
        output = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        output = self.pool(F.relu(self.conv2_bn(self.conv2(output))))
        output = self.pool(F.relu(self.conv3_bn(self.conv3(output))))
        output = self.pool(F.relu(self.conv4_bn(self.conv4(output))))
        output = torch.flatten(output, 1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.sigmoid(output)
        return output.double()
    

class ModelM3_v2(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super().__init__()
        self.ups = nn.Upsample((512, 512))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.conv5_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16384, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1_bn(self.conv1(output))))
        output = self.pool(F.relu(self.conv2_bn(self.conv2(output))))
        output = self.pool(F.relu(self.conv3_bn(self.conv3(output))))
        output = self.pool(F.relu(self.conv4_bn(self.conv4(output))))
        output = F.relu(self.conv5_bn(self.conv5(output)))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()
    

class AggNet_500_drop_v3(nn.Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.ups = nn.Upsample((500, 500))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 16, 1)
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 1)
        self.drop = nn.Dropout(p = p)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = self.pool(F.relu(self.conv5(output)))
        output = self.pool(F.relu(self.conv6(output)))
        output = F.relu(self.conv7(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.drop(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class ModelM3_v3(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.ups = nn.Upsample((512, 512))
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16384, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1_bn(self.conv1(output))))
        output = self.pool(F.relu(self.conv2_bn(self.conv2(output))))
        output = self.pool(F.relu(self.conv3_bn(self.conv3(output))))
        output = self.pool(F.relu(self.conv4_bn(self.conv4(output))))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()
    

class AggNet_drop_512_v3(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.ups = nn.Upsample((512, 512), mode='bilinear')
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(16384, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()
    


class AggNet_drop_256_v2(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.ups = nn.Upsample((256, 256), mode='bilinear')
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()


class AggNet_drop_128_v2(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.ups = nn.Upsample((128, 128), mode='bilinear')
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 16, 1)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = x.float()
        output = self.ups(x)
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = self.pool(F.relu(self.conv4(output)))
        output = F.relu(self.conv5(output))
        output = torch.flatten(output, 1)
        output = F.relu(self.dropout(self.fc1(output)))
        output = torch.sigmoid(self.fc2(output))
        return output.double()

