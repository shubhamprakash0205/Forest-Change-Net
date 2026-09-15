from model import earth_dataset,forestchange_model
import torch
from torch.utils.data import Dataset,DataLoader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

earth_model = forestchange_model().to(device)
loss_func = torch.nn.BCELoss() 
optimizer = torch.optim.Adam(earth_model.parameters(),lr=1e-4)
epoch_count = 0
checkpoint_path = 'files/results/checkpoint.pth'

if os.path.exists(checkpoint_path):
	checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
	earth_model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch_count = checkpoint['epoch_count']
	loss_net = checkpoint['loss_net']

def train():
	global epoch_count
	while epoch_count < 1e+6:
		file_path = 'files/hansen_dataset/train'
		dataset = earth_dataset(file_path)
		dataset.xx
		dataloader = DataLoader(dataset,batch_size=300,shuffle=True)
		loss_net = 0
		iou_net = 0
		for data in dataloader:
			input_data,target = data
			input_data,target = input_data.to(device),target.to(device)
			output = earth_model(input_data)
			loss = loss_func(output,target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_net = loss.item() * input_data.size(0) + loss_net
			
			pred = output > 0.5
			intersection = (pred * target).sum(dim=(1,2,3))
			union = ((pred + target) > 0).sum(dim=(1,2,3))
			iou = (intersection / (union+ 1e-8)) * 100
			iou_net = iou_net + iou.sum(dim=0) 
			
		iou_avg = iou_net / len(dataset)	
		loss_net = loss_net / len(dataset)	
		epoch_count += 1
		if epoch_count % 7 == 0:
			torch.save({'epoch_count':epoch_count,'iou':iou_avg,'model_state_dict' : earth_model.state_dict() ,'optimizer_state_dict' : optimizer.state_dict(),'loss_net':loss_net},checkpoint_path)
			print('saved..............')
			print(f'epoch_count: {epoch_count} loss: {loss_net}')
			print(f'iou: {iou_avg}')
			 
			
		
train()




































