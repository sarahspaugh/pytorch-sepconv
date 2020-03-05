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

    def __init__(self, db_dir, num_frames, frame_start_list, n_frame):
        # added new init variables: num_frames, frame_start_list, n_frame. will need to check other code for consistency
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        """
            changes that need to happen around here:
            need a new load video function that doesn't do any of the fancy segmenting/cropping/sorting; just takes the video into a frame dict with the appropriate structure to output with getitem
            -> as a follow up from the last, need to double check with ben that the result of the new load video file is shaped how i think it is
            need to take out all the redundant variables assuming that whatever ends up in the traindb directory is going to get used all the way (like we aren't going to only use a few of the frames or whatever
        """
        
        self.vid_list = [".".join(f.split(".")[:-1]) for f in listdir(db_dir) if os.path.isfile(f)]
        
        ##### here we need to turn the training video into a dictionary of arrays, preferably indexed by number, with the sets of 3 frames stacked along axis 4
        self.frame_dict = data_import.load_video(vid_list, num_frames, frame_start_list, seed = 1)
        #######d
        
        self.file_len = len(list(self.frame_dict.keys()))

    
    def __getitem__(self, index):
        
        key_list = self.frame_dict.keys()
        key = key_list[index]
        frame = self.frame_dict[key]
        
        frame0 = self.transform(frame[:,:,:,0])
        frame1 = self.transform(frame[:,:,:,2])
        frame2 = self.transform(frame[:,:,:,1])


        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
