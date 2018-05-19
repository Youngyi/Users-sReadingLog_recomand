import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import time
from torchvision import datasets

f = open("iterations.txt", "w+")
# Hyper Parameters
input_size = 52
hidden_size1 = 256
hidden_size2 = 256
num_classes = 23
num_epochs = 500
batch_size = 128
learning_rate = 0.001

def convert_file(filename):
	train = open(filename, 'r+')
	data = []
	for line in train:
		splitted = line.rstrip().split(" ")
		data.append([float(x) for x in splitted])
	data = np.array(data)
	train.close()
	return torch.from_numpy(data).float()

input_data = convert_file("train_app.txt")
output_data = convert_file("train_l1.txt")

test_input = convert_file("val_app.txt")
test_output = convert_file("val_l1.txt")

train_dataset = Data.TensorDataset(data_tensor = input_data, target_tensor = output_data)
test_dataset = Data.TensorDataset(data_tensor = test_input, target_tensor = test_output)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										   batch_size=batch_size,
										   shuffle=False)
dataloders = dict()
dataloders['train'] = train_loader
dataloders['val'] = test_loader

dataset_sizes = {'train': 6000, 'val': 1000}

use_gpu = torch.cuda.is_available()

# Neural Network Model (2 hidden layers)
class Net(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(input_size, hidden_size1),
			nn.Dropout(0.75),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(hidden_size1, hidden_size2),
			nn.Dropout(0.75),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(hidden_size2, num_classes),
			nn.Dropout(0.75))
		#Try doing softmax if possible

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		return out


net = Net(input_size, hidden_size1, hidden_size2, num_classes)

# Loss and Optimizer
criterion = torch.nn.MSELoss() #200 iterations 88%
#criterion = torch.nn.CrossEntropyLoss() #200 iterations 88%
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# # Train the Model
# net.train()
# for epoch in range(num_epochs):
# 	correct = 0
# 	#print('Epoch:', epoch + 1, 'Training...')
# 	for i, (vectors, labels) in enumerate(train_loader):
# 		# Convert torch tensor to Variable
# 		vectors = Variable(vectors)
# 		labels = Variable(labels)
# 		a,b = torch.max(labels.data, 1)
#
# 		# Forward + Backward + Optimize
# 		optimizer.zero_grad()  # zero the gradient buffer
# 		outputs = net(vectors)
# 		#loss = Variable(outputs.data.mm(torch.Tensor(np.transpose((labels.data).numpy()))))
# 		loss = criterion(outputs, labels)
# 		loss.backward()
# 		optimizer.step()
# 		_, predicted = torch.max(outputs.data, 1)
# 		#print correct
# 		correct += (predicted == b).sum()
# 		if (i + 1) % 100 == 0:
# 			print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
# 				   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
# 	print('Accuracy of the network: %d %%' % (100*correct / 6000))

def train_model(model, criterion, optimizer, num_epochs=25):
	since = time.time()

	best_model_wts = model.state_dict()
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for data in dataloders[phase]:
				# get the inputs
				inputs, labels = data

				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)

				_,idx = torch.max(labels.data, 1)

				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()

				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == idx)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			f.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model

model_ft = train_model(net, criterion, optimizer, num_epochs=150)
f.close()
