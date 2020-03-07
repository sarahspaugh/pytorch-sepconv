import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
from data_utils import data_import

class DBreader_frame_interpolation(Dataset):
    """
        original implementation:
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    
    for sarah/ben: editing such that database directory contains [a] video file[s]
    and we read the video into frame dictionaries and work from there
    
    
    """

    def __init__(self, db_dir):
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.frame_dict = data_import.load_p_video(db_dir)
        
        self.file_len = len(list(self.frame_dict.keys()))
    
    def __getitem__(self, index):
        
        key_list = self.frame_dict.keys()
        key = key_list[index]
        frame = self.frame_dict[key]
        
        frame0 = self.transform(frame[0,:,:,:])
        frame1 = self.transform(frame[2,:,:,:])
        frame2 = self.transform(frame[1,:,:,:])


        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
