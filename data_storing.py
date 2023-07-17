import glob
import os
import shutil


train_path = 'data_town4_carla/train/'
val_path = 'data_town4_carla/val/'
# test_path = 'data_/test'


def process(path, dest):
    types = ['images', 'masks']

    for i in types:
        index = 4800
        path1 = path
        dest1 = dest
        path1 = path1 + i
        dest1 = dest1 + i

        print(dest1)

        images = glob.glob(os.path.join(path1, '*'))
  
        for image in images:
            image_dest = dest1 + f'/{index}.png'
            shutil.copy(image, image_dest)
            index += 1
        

    
train_dest = 'data/train/'
val_dest = 'data/val/'
process(val_path, val_dest)