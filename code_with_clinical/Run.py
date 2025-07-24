from DataLoader import load_data, load_functional_modules
from train import trainDeepMusicsNet
import argparse
import torch
import numpy as np
import os
parser = argparse.ArgumentParser()
args = parser.parse_args()
print("Gene number is :",args.inputG)
print("Functional module or pathways number is :",args.inputF)
torch.cuda.set_device(1)
dtype = torch.FloatTensor
In_Nodes = args.inputG ##gene nodes in need of change for users
functional_Nodes = args.inputF ##functinal modules nodes in need of change for user
Hidden_Nodes = 200 
Out_Nodes = 30 

Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs = 4000 
Num_EPOCHS = 30000 

Dropout_Rate = [0.65, 0.45]
functional_mask = load_functional_modules("~/functional_mask.csv", dtype)

x_train, ytime_train, yevent_train, clinical_train = load_data("~/train.csv", dtype)
x_valid, ytime_valid, yevent_valid, clinical_valid = load_data("~/test.csv", dtype)
x_test, ytime_test, yevent_test, clinical_test = load_data("~/validation.csv", dtype)

opt_l2_loss = 0
opt_lr_loss = 0
opt_loss = torch.Tensor([float("Inf")])

if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()
opt_c_index_va = 0
opt_c_index_tr = 0
for l2 in L2_Lambda:
	for lr in Initial_Learning_Rate:
		loss_train, loss_valid, c_index_tr, c_index_va = trainDeepMusicsNet(x_train, clinical_train, ytime_train, yevent_train, \
																x_valid, clinical_train_valid, ytime_valid, yevent_valid, pathway_mask, \
																In_Nodes, functional_Nodes, Hidden_Nodes, Out_Nodes, \
																lr, l2, num_epochs, Dropout_Rate)
		if loss_valid < opt_loss:
			opt_l2_loss = l2
			opt_lr_loss = lr
			opt_loss = loss_valid
			opt_c_index_tr = c_index_tr
			opt_c_index_va = c_index_va
		print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_v)

loss_train, loss_test, c_index_tr, c_index_te = trainDeepMusicsNet(x_train,  clinical_train, ytime_train, yevent_train, \
							x_test, ytime_test, yevent_test, functional_mask, \
							In_Nodes, functional_Nodes, Hidden_Nodes, Out_Nodes, \
							opt_lr_loss, opt_l2_loss, Num_EPOCHS, Dropout_Rate)
print ("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)
print("C-index in Test: ", c_index_t)
