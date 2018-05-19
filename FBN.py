import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

# Hyper Parameters
input_size = 52
hidden_size1 = 1024
hidden_size2 = 1024
num_classes = 23
num_epochs = 200
batch_size = 100
learning_rate = 0.001
rate = 0.7 # division rate of data

def read_input():
    file = open('app.txt', 'r')
    result = []
    input_data = []
    for line in file:
        input_data.append(map(float, line.split()))
    input_data = np.array(input_data)
    result.append(input_data[0: int(len(input_data)* rate)])
    result.append(input_data[int(len(input_data)* rate): len(input_data)])
    file.close()
    return torch.Tensor(result[0]), torch.Tensor(result[1]), [len(result[0]), len(result[1])]

def read_output():
    file = open('l1.txt', 'r')
    result = []
    output_data = []
    for line in file:
        output_data.append(map(float, line.split()))
    output_data = np.array(output_data)
    result.append(output_data[0: int(len(output_data) * rate)])
    result.append(output_data[int(len(output_data) * rate): len(output_data)])
    file.close()
    return torch.Tensor(result[0]), torch.Tensor(result[1])

input_traindata, input_testdata, len_list = read_input()
output_traindata, output_testdata = read_output()

train_dataset = Data.TensorDataset(data_tensor = input_traindata, target_tensor = output_traindata)
test_dataset = Data.TensorDataset(data_tensor = input_testdata, target_tensor = output_testdata)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


net = Net(input_size, hidden_size1, hidden_size2, num_classes)

# Loss and Optimizer
#criterion = torch.nn.MSELoss() #200 iterations 88%
criterion = torch.nn.SmoothL1Loss() #200 iterations 88%
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Test the Model
def test():
    correct = 0
    for vectors, labels in test_loader:
        vectors = Variable(vectors)
        labels = Variable(labels)
        a, b = torch.max(labels.data, 1)
        outputs = net(vectors)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == b).sum()
    print('Test_Accuracy of the network: %d %%' % (100 * correct / len_list[1]))

# Train the Model
for epoch in range(num_epochs):
    correct = 0
    for i, (vectors, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        vectors = Variable(vectors)
        labels = Variable(labels)
        a,b = torch.max(labels.data, 1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(vectors)
        #loss = Variable(outputs.data.mm(torch.Tensor(np.transpose((labels.data).numpy()))))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        #print correct
        correct += (predicted == b).sum()
        if (i + 1) % 10 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
    print('Accuracy of the network: %d %%' % (100*correct / len_list[0]))
    test()

# Save the Model
#torch.save(net.state_dict(), 'model.pkl')