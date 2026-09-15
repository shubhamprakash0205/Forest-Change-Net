import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

ee.Authenticate()
ee.Initialize(project = 'deforestation-neural-analysis')


"""
coords = [
    [74.44488445158606, 34.384820559607704],
    [74.50239101286536, 34.384820559607704],
    [74.50239101286536, 34.414281861791316],
    [74.44488445158606, 34.414281861791316],
    [74.44488445158606, 34.384820559607704]
]

# Create the AOI polygon
aoi = ee.Geometry.Polygon(coords)

# Print to verify
print(aoi.getInfo())

dataset = ee.Image(
    "UMD/hansen/global_forest_change_2025_v1_13"
)

target = dataset.select("loss")

first = dataset.select([
    "first_b30",
    "first_b40",
    "first_b50",
    "first_b70"
])


last = dataset.select([
    "last_b30",
    "last_b40",
    "last_b50",
    "last_b70"
])


inputs = first.addBands(last)

print(type(first))
print(type(inputs))

inputs = inputs.clip(aoi)
target = target.clip(aoi)
"""





"""
geemap.ee_export_image(
    inputs,
    filename="kashmir_inputs.tif",
    scale=30,
    region=aoi
)

geemap.ee_export_image(
    target,
    filename="kashmir_loss.tif",
    scale=30,
    region=aoi
)
"""


"""
with rasterio.open("kashmir_inputs.tif") as src:
    image1 = src.read()
with rasterio.open("kashmir_loss.tif") as src:
    image2 = src.read()



print(image1.shape,type(image1))
print(image2.shape,type(image2))
print(np.max(image2))
img2 = np.squeeze(image2,axis=0)
#img2 = np.unsqueeze(img2,axis=-1)
#img2 = np.expand_dims(img2, axis=-1)



#img2 = (image2 == 1)
print(np.sum(img2))

landsat = (
    ee.ImageCollection(
        "LANDSAT/LC08/C02/T1_L2"
    )
)
"""

"""
img1 = image1

for i in range(len(img1)):
	img = img1[i]
	plt.subplot(1,len(img1),i+1)
	plt.imshow(img)
	
#plt.show()
	
img1 = np.expand_dims(img1, axis=-1)
	
img11 = np.concatenate((img1[0],img1[1],img1[2]),axis=-1) 	
print(img11.shape)
plt.subplot(1,3,1)
plt.imshow(img11)

img22 = np.concatenate((img1[4],img1[5],img1[6]),axis=-1) 	
print(img22.shape)
plt.subplot(1,3,2)
plt.imshow(img22)

plt.subplot(1,3,3)
plt.imshow(img2)


#plt.show()



landsat = (
    ee.ImageCollection(
        "LANDSAT/LC08/C02/T1_L2"
    )
)

land_data = landsat.select(["SR_B5","SR_B4","SR_B3"])

img3 = land_data
geemap.ee_export_image(img3,scale=30,filename='img3.tif',region=aoi)

geemap.ee_export_image(
    target,
    filename="kashmir_loss.tif",
    scale=30,
    region=aoi
)

with rasterio.open('img3.tif') as src:
	img3 = src.read()
	
print(img3.shape)

"""


rampur_jk = [
 [74.43861445695109,34.37743362360225],
 [74.5110555580253,34.37743362360225],
 [74.5110555580253,34.41766059498502],
 [74.43861445695109,34.41766059498502],
 [74.43861445695109,34.37743362360225],
]

quilmuqam_jk = [
 [74.53525130133039,34.450911382110576],
[74.63241164068586,34.450911382110576],
[74.63241164068586,34.50355282395067],
[74.53525130133039,34.50355282395067],
[74.53525130133039,34.450911382110576],
]

Basti_kalu_khan_pak = [
[72.92999426111687,34.32987036034728],
[73.01788488611687,34.32987036034728],
[73.01788488611687,34.40921767278144],
[72.92999426111687,34.40921767278144],
[72.92999426111687,34.32987036034728],
]

qumroo_jk = [
[74.51190563377665,33.86545322734403],
[74.56855388817118,33.86545322734403],
[74.56855388817118,33.92045545347432],
[74.51190563377665,33.92045545347432],
[74.51190563377665,33.86545322734403],
]

zampathaer_jk = [
[74.67910617962045,33.66916278541484],
[74.78690952434701,33.66916278541484],
[74.78690952434701,33.77196450125022],
[74.67910617962045,33.77196450125022],
[74.67910617962045,33.66916278541484],
]

bhakra_range_uk = [
[79.10685510748435,29.038741689705375],
[79.50785608404685,29.038741689705375],
[79.50785608404685,29.3312852776826],
[79.10685510748435,29.3312852776826],
[79.10685510748435,29.038741689705375],
]

rudpur_uk = [
[79.5564628148263,28.92665617830835],
[79.76794963123255,28.92665617830835],
[79.76794963123255,29.20393609532604],
[79.5564628148263,29.20393609532604],
[79.5564628148263,28.92665617830835],
]

kachchapal_ch = [
[80.72430735309776,19.14442983390598],
[81.3148224898165,19.14442983390598],
[81.3148224898165,19.95712509581916],
[80.72430735309776,19.95712509581916],
[80.72430735309776,19.14442983390598],
]

tikri_od = [
[83.44546344992594,19.443148599380812],
[84.36831501242594,19.443148599380812],
[84.36831501242594,20.429451737011],
[83.44546344992594,20.429451737011],
[83.44546344992594,19.443148599380812],
]

balliguda_od = [
[82.91811969992594,18.65391579208803],
[83.42349079367594,18.65391579208803],
[83.42349079367594,19.91899513413977],
[82.91811969992594,19.91899513413977],
[82.91811969992594,18.65391579208803],
]

metturu_ap=[
[84.43785662456824,18.720765276367114],
[84.55321306988074,18.720765276367114],
[84.55321306988074,18.876769350421018],
[84.43785662456824,18.876769350421018],
[84.43785662456824,18.720765276367114],
]


####test
khungringFV_as=[
[90.17446987335137,26.583184428895933],
[90.56173793975762,26.583184428895933],
[90.56173793975762,26.968160799213702],
[90.17446987335137,26.968160799213702],
[90.17446987335137,26.583184428895933],
]

mazbat_as = [
[92.14066156012723,26.727166430444548],
[92.60483392340848,26.727166430444548],
[92.60483392340848,26.989348157543883],
[92.14066156012723,26.989348157543883],
[92.14066156012723,26.727166430444548],
]

mukhim_as = [
[92.6028480646843,25.71498154148122],
[92.72163773753586,25.71498154148122],
[92.72163773753586,25.7830127539542],
[92.6028480646843,25.7830127539542],
[92.6028480646843,25.71498154148122],
]

radhanagar_as=[
[92.85167670574153,25.754109514956042],
[92.93510413494074,25.754109514956042],
[92.93510413494074,25.811919477472962],
[92.85167670574153,25.811919477472962],
[92.85167670574153,25.754109514956042],
]

wangti_na=[
[94.79484461226659,26.241751204009706],
[95.37162683882909,26.241751204009706],
[95.37162683882909,26.625421140007617],
[94.79484461226659,26.625421140007617],
[94.79484461226659,26.241751204009706],
]

def dataset_prep(coords,name_file):
	folder_path = f'files/hansen_dataset/test/'
	name = os.path.join(folder_path,f'{name_file}.tif')
	aoi = ee.Geometry.Polygon(coords)
	print(aoi.area().getInfo())
	print()
	dataset = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
	input_bands = dataset.select(["first_b30","first_b40","first_b50","first_b70","last_b30","last_b40","last_b50","last_b70"])            
	target_bands = dataset.select(['lossyear'])
	#print(type(target_bands),'target_band') #ee.image.Image
	
	combine_data = input_bands.addBands(target_bands)
	combine_data = combine_data.clip(aoi)	
	
	band_names = combine_data.bandNames().getInfo()
	num_bands = len(band_names)
	
	# Run a reducer on the first band to count valid pixels
	pixel_stats = combine_data.select(0).reduceRegion(
	    reducer=ee.Reducer.count(),
	    geometry=aoi,
	    scale=30,
	    maxPixels=1e13
	)
	single_band_pixels = list(pixel_stats.getInfo().values())[0]
	total_pixels = single_band_pixels * num_bands
	
	print(f"Total calculated pixels for download: {total_pixels:,}")
	
	# Earth Engine direct download ceiling is roughly 10,000,000 pixels
	DOWNLOAD_LIMIT = 10000000
	grid_ratio = int(total_pixels / DOWNLOAD_LIMIT) + 1
	print('GRID',grid_ratio)
	if total_pixels <= DOWNLOAD_LIMIT:
		print("Safe size! Downloading as a single file...")
		
		f = geemap.ee_export_image(combine_data,filename=name,scale=30,region=aoi)
		with rasterio.open(name) as src:
			x = src.read()
			print(x[0].shape)
			if len(x[0]) < 111 or len(x[0][0]) < 111:
				src.close()
				os.remove(name)
				print('INVALID SHAPE:',"INCREASE AREA")
		
		
	else:
		print(f" Size exceeds limit ({total_pixels:,} > {DOWNLOAD_LIMIT:,}). Automatically slicing into grid...")
		grid = geemap.fishnet(aoi, rows=grid_ratio, cols=grid_ratio)
		grid_features = grid.getInfo()['features']
		
		total_chunks = grid_ratio * grid_ratio
		index = 0
	    
		# Loop over chunks and download each safely
		for feature in grid_features:
			if feature['geometry']['type'] not in ['Polygon', 'MultiPolygon']:
				continue
			
			index += 1
			cell_geometry = ee.Geometry(feature['geometry'])
			cell_image = combine_data.clip(cell_geometry)
			part_filename = os.path.join(folder_path,f'{name_file}_part_{index}.tif')
			print(f" -> Downloading chunk {index} of {total_chunks} to {part_filename}...")
	        
			geemap.ee_export_image(
				cell_image,
				filename=part_filename,
				scale=30,
				region=cell_geometry,
				file_per_band=False
		        )
		        
			with rasterio.open(part_filename) as src:
				x = src.read()
				print(x.shape)
				print(x[0].shape)
				if len(x[0]) < 111 or len(x[0][0]) < 111:
					src.close()
					os.remove(part_filename)
					print('INVALID SHAPE:',"INCREASE AREA",part_filename)

		        
		print("All chunks downloaded successfully!")	
			
				
		#the files will be in shape 9 * (h * w)
	
	
"""
d_list = [
quilmuqam_jk,
Basti_kalu_khan_pak,
qumroo_jk,
zampathaer_jk,
rudpur_uk,
balliguda_od,
tikri_od,
kachchapal_ch,
metturu_ap
]

g_list = [
'quilmuqam_jk',
'Basti_kalu_khan_pak',
'qumroo_jk',
'zampathaer_jk',
'rudpur_uk',
'balliguda_od',
'tikri_od',
'kachchapal_ch',
'metturu_ap'
]
"""

##train
d_list = [
khungringFV_as,
mazbat_as,
mukhim_as,
radhanagar_as,
wangti_na,
]

g_list = [
'khungringFV_as',
'mazbat_as',
'mukhim_as',
'radhanagar_as',
'wangti_na',
]

for i in range(len(d_list)):
	dataset_prep(d_list[i],g_list[i])













