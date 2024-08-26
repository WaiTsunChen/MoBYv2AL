from PIL import Image
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import os
from multiprocessing import Pool

def process_crop_resize_save(row):
    '''
    Function to crop and resize images and save them to a new directory.
    See "preprocessing" chapter.
    '''
    print(f'starting process {os.getpid()}')
    path_input = '/Volumes/data01/waitsun/snapshotSerengeti/'
    path_output = '/Volumes/data01/waitsun/snapshotSerengetiCropedResized/'
    for i in range(len(row)):
        img_name = row.iloc[i]['img_name']
        bbox_im = row.iloc[i]['bbox_pixel_level']
        resized_name = row.iloc[i]['resized_name']
        image = Image.open(path_input + img_name + '.JPG')

        image_croped = T.functional.crop(
            image, int(bbox_im[1]), int(bbox_im[0]), int(bbox_im[3]), int(bbox_im[2]))

        image_croped = T.Resize((64,64))(image_croped)

        save_target_path = os.path.join(path_output, resized_name + '.JPG')
        os.makedirs(os.path.dirname(save_target_path), exist_ok=True)
        image_croped.save(save_target_path)

if __name__ == '__main__':
    print(f'reading in data...')
    df = pd.read_pickle('/Volumes/data01/waitsun/snapshotSerengeti/df_metadata_train.df')
    df['unique_index'] = df['img_name'].index
    df['resized_name'] = df.img_name+'_'+  df['unique_index'].astype(str) 
    df.drop(columns=['unique_index'], inplace=True)
    print(f'...finished reading in data')
    df_small = df.head(100)
    length = len(df_small)
    num_processes = 8
    chunks = [df_small.iloc[i:i + length // num_processes] for i in range(0, length, length // num_processes)]

    # Create a Pool of processes
    with Pool(processes=num_processes) as pool:
        # Map each chunk of the DataFrame to a process
        pool.map(process_crop_resize_save, chunks)