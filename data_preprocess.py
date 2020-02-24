import os, pickle, pdb
import cv2
import numpy as np

dataset = 'data/cub'
imageset = 'images'
imgDir = 'data/cub/images'

if __name__ == '__main__':
	for trainset in ['train', 'test']:
		with open(os.path.join(dataset, trainset, 'filenames.pickle'), 'rb') as f:
			inList = pickle.load(f)
		imgArrayList = []
		for item in inList:
			
			imgPath = os.path.join(imgDir, item + '.jpg')
			if not os.path.exists(imgPath):
				print(item + ' not exists!')
				continue
				
			img = cv2.imread(imgPath)
			resized_img = cv2.resize(img, (224, 224))
			imgArrayList.append(resized_img)
		
		imgArray = np.asarray(imgArrayList)
		with open(dataset + '/' + trainset + '/' + imageset + '_224.pickle', 'wb') as f:
			pickle.dump(imgArray, f)