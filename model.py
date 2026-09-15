import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms

import os
import rasterio

class earth_dataset(Dataset):
	def __init__(self,file_path):
		file_path = file_path
		self.all_files = [os.path.join(file_path,file) for file in os.listdir(file_path) if os.path.isfile(os.path.join(file_path,file))]
		patch_size = 112
		self.net_data = [] #(no_of_images,112,112,9)
		
		for file in self.all_files:
			with rasterio.open(file) as src:
				data = src.read() #(9*area)9,H,W
				data = torch.from_numpy(data)
				"""
				d = []
				for i in range(9):
					da = torch.unsqueeze(data[i],axis=-1)
					d.append(da)
				data2 = torch.concatenate(d,axis=-1) #h,w,9
				"""
				data2 = data.permute(1,2,0)
				#print(data2.shape)
				stride = 20		
				for x in range(0,len(data2)-patch_size+1,stride):
					for y in range(0,len(data2[0])-patch_size+1,stride):
						img_112 = data2[x:x+patch_size,y:y+patch_size,:]
						###checking_density of result/target(deforestation)
						target = img_112[:,:,8:]
						deforestation_density = target.sum(dim=(0,1,2))
						if deforestation_density > 000:
							self.net_data.append(img_112)
		#x = torch.randperm(len(self.net_data))[:50000]
		
		#self.net_data = [self.net_data[i.item()] for i in x ]
						
		#print(len(self.net_data),self.net_data[0].shape)
		
		print(len(self.net_data))				
	
	def __getitem__(self,idx):
		input_data = self.net_data[idx][:,:,:8] #(112,112,8)
		input_data = input_data.permute(2,0,1)
		target_data = self.net_data[idx][:,:,8:] #(112,112,1)			
		#print(torch.min(target_data),torch.max(target_data))
		target_data = target_data > 0
		#print(torch.min(target_data),torch.max(target_data))
		#print(input_data.shape,target_data.shape)
		#print(torch.max(input_data),torch.min(input_data))
		return input_data.float(),target_data.float() #(8,112,112) #(112,112,1)
	
	def __len__(self):
		return len(self.net_data)
		

#dataset = earth_dataset()
#print(len(dataset))
#t,y = dataset[0]
#print('FRESH')

"""
for i in range(8):
	print(i,torch.min(t[i]),torch.max(t[i]))
"""

#dataloader = DataLoader(dataset,shuffle=True,batch_size= 2)

#(I−1)×stride−2×padding+kernel size+output padding

class forestchange_model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.relu_layer = torch.nn.LeakyReLU(negative_slope=1e-4)
		self.sigmoid_layer = torch.nn.Sigmoid()
		
		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=8,out_channels=32,kernel_size=3,padding=1), #112
			torch.nn.BatchNorm2d(32),
			self.relu_layer
		)
		self.layer2 = torch.nn.MaxPool2d(stride=2,kernel_size=2)
		
		self.layer3 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1), #56
			torch.nn.BatchNorm2d(64),
			self.relu_layer
		)
		
		self.layer4 = torch.nn.MaxPool2d(stride=2,kernel_size=2)
		
		self.layer5 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1), #28
			torch.nn.BatchNorm2d(128),
			self.relu_layer
		)
		
		self.layer6 = torch.nn.MaxPool2d(stride=2,kernel_size=2)
		
		###bottleneck(14*14)
		
		self.layer7_up = torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2,padding=0) #28
		self.layer7 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=192,out_channels=128,kernel_size=3,padding=1),
			torch.nn.BatchNorm2d(128),
			self.relu_layer
		)
		
		self.layer8_up = torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2,padding=0) #56
		self.layer8 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
			torch.nn.BatchNorm2d(64),
			self.relu_layer
		)
		
		self.layer9_up = torch.nn.ConvTranspose2d(in_channels=64,out_channels=32,stride=2,kernel_size=2,padding=0) #112
		self.layer9 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1),
			torch.nn.BatchNorm2d(32),
			self.relu_layer
		)
		
		self.layer10 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3,padding=1),
			self.sigmoid_layer
		)
				
		
	def forward(self,input_data):
		#input_data(batch_size,8,256,256)
		self.input_data = input_data
		#i_mean = self.input_data.mean(dim=(0,2,3)) #(8,)
		#i_std = self.input_data.std(dim=(0,2,3))
		
		#i_mean = i_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #1,8,1,1
		#i_std = i_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #1,8,1,1
		
		#print(i_mean.shape)
		#self.input_data = ( self.input_data - i_mean )/ i_std
		#print(self.input_data.shape)
		#print(torch.max(self.input_data),torch.min(self.input_data))

		self.input_data = self.input_data / 255
		
		#####
		layer1 = self.layer1(self.input_data)
		layer2 = self.layer2(layer1)
		layer3 = self.layer3(layer2)
		layer4 = self.layer4(layer3)
		layer5 = self.layer5(layer4)
		layer6 = self.layer6(layer5)
		
		layer7_up = self.layer7_up(layer6)
		layer7_input = torch.cat((layer7_up,layer5),dim=1)
		layer7 = self.layer7(layer7_input)
		
		layer8_up = self.layer8_up(layer7)
		layer8_input = torch.cat((layer8_up,layer3),dim=1)
		layer8 = self.layer8(layer8_input)
		
		layer9_up = self.layer9_up(layer8)
		layer9_input = torch.cat((layer9_up,layer1),dim=1)
		layer9 = self.layer9(layer9_input)
		
		layer10 = self.layer10(layer9)

		"""
		print(layer1.shape)
		print(layer2.shape)
		print(layer3.shape)
		print(layer4.shape)
		print(layer5.shape)
		print(layer6.shape)
		
		print()
		
		print(layer7_up.shape)
		print(layer7_input.shape)
		print(layer7.shape)
		print()
		print(layer8_up.shape)
		print(layer8_input.shape)
		print(layer8.shape)
		print()
		print(layer9_up.shape)
		print(layer9_input.shape)
		print(layer9.shape)
		print()
		print(layer10.shape)
		print()
		layer10.shape.x
		"""
		layer10 = layer10.permute(0,2,3,1)
		return layer10








		






























