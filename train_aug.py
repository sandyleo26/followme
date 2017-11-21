import os
import glob
from scipy import misc
import numpy as np

## flip training data set.
# shared by minimithi on Slack
# https://medium.com/towards-data-science/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def flip_and_save_images(img_dir, extension):
  os.chdir(img_dir)
  files = glob.glob("*." + extension)
  for i, file in enumerate(files):
    try:
        img = misc.imread(file, flatten=False, mode='RGB')
        flipped_img = np.fliplr(img)
        misc.imsave("flipped_" + file, flipped_img)
    except:
        print(i + " : " + file)

################
flip_and_save_images("/home/ubuntu/robond-followme-project/data/train/masks", "png")
flip_and_save_images("/home/ubuntu/robond-followme-project/data/train/images", "jpeg")
