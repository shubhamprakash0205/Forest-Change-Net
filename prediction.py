import torch
from model import earth_dataset,forestchange_model
import matplotlib.pyplot as plt
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'files/hansen_dataset/test'
checkpoint_path = 'files/results/checkpoint (26).pth'
dataset = earth_dataset(file_path)

earth_model = forestchange_model().to(device)

if os.path.exists(checkpoint_path):
	checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
	earth_model.load_state_dict(checkpoint['model_state_dict'])
	earth_model.eval()
	
def view():
	print('EPOCH: ', checkpoint['epoch_count'])
	random_index = torch.randperm(len(dataset))[:1]
	data_set = dataset[random_index]
	input_band,output_band = data_set
	input_band,output_band = input_band.to(device),output_band.to(device)
	with torch.no_grad():
		prediction = earth_model(torch.stack([input_band])).detach().squeeze(dim=0) > 0.5
		#(8,112,112) #(112,112,1)
		
	input_band = input_band.permute(1,2,0)/255 #112,112,8
	output_band = output_band > 0
	
	first_bands = input_band[:,:,:4]
	last_bands = input_band[:,:,4:8]
	
	first_bands = first_bands[:,:,[0,1,2,3]] #changed 30,40,50,70 to 40,30,50,70 
	last_bands = last_bands[:,:,[0,1,2,3]]
	
	intersection = (output_band * prediction).sum()
	union = (output_band + prediction).sum()
	IOU = (intersection / (union+1e-8) ) * 100
	print(IOU)
	
	images = [ first_bands[:,:,:3], last_bands[:,:,:3], prediction , output_band ]
	titles = ['Year2000 (RED,NIR,SWIR1)', 'Year 2025(RED,NIR,SWIR1)', 'Prediction' , 'Output']
	fig, axes = plt.subplots(2, 2)
	
	for ax, img , title in zip(axes.flat, images, titles):
		ax.imshow(img)
		ax.set_title(title)
		ax.axis("off")
	
	plt.tight_layout()
	plt.show()
		
		
for i in range(10):
	view()	
		
	
	
	
	
	
	
	
	

