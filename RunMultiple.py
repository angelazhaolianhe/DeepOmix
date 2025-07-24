from DataLoader import load_data_without, load_pathway
from Train import trainDeepOmixNet_without

import torch
import numpy as np
import os

gpu_id=0
if torch.cuda.is_available():
	device = torch.device('cuda')
	torch.cuda.set_device(gpu_id)
else:
	device = torch.device('cpu')
dtype = torch.FloatTensor

In_Nodes = 5075
Pathway_Nodes = 100 
Hidden_Nodes = 80
Out_Nodes = 30

Initial_Learning_Rate = [0.00001, 0.000075]
L2_Lambda = [0.01, 0.005, 0.001]
num_epochs = 5000
Num_EPOCHS = 20000
Dropout_Rate = [0.7, 0.5]
path='./Data/Multiple/'
pathway_mask = load_pathway(path+"pathway_module_input.csv", dtype)

x_train, ytime_train, yevent_train = load_data_without(path+"train.csv", dtype)
x_valid, ytime_valid, yevent_valid= load_data_without(path+"test.csv", dtype)
x_test, ytime_test, yevent_test= load_data_without(path+"validation.csv", dtype)
opt_l2_loss = 0
opt_lr_loss = 0
opt_loss = torch.Tensor([float("Inf")])
if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()

opt_c_index_va = 0
opt_c_index_tr = 0
for l2 in L2_Lambda:
	for lr in Initial_Learning_Rate:
		loss_train, loss_valid, c_index_tr, c_index_va = trainDeepOmixNet_without(x_train, ytime_train, yevent_train, \
																x_valid,  ytime_valid, yevent_valid, pathway_mask, \
																In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
																lr, l2, num_epochs, Dropout_Rate)
		if loss_valid < opt_loss:
			opt_l2_loss = l2
			opt_lr_loss = lr
			opt_loss = loss_valid
			opt_c_index_tr = c_index_tr
			opt_c_index_va = c_index_va
		print ("the lamda is: ", l2, ", the learning rate is: ", lr,".")
		print ("the Loss in Train data is:",loss_train, " Loss in Validation is: ", loss_valid)
