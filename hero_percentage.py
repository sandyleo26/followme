#By tokyo_adam 4-10-17 
import cv2
import numpy as np
import glob, os

#set to the directory where your masks are saved
img_dir = "../data/train/masks/"

total_files = 0
total_hero = 0

os.chdir(img_dir)
for file in glob.glob("*.png"):
	total_files +=1

	img = cv2.imread(file)
	blue = img[:,:,0]

	if np.any(blue == 255):
            total_hero += 1

percent_hero = 100. * total_hero / total_files
print (percent_hero, "percent of files contain the hero")
