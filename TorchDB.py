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

    def __init__(self, db_dir, num_frames, frame_start_list, n_frame, resize=None):
        # added new init variables: num_frames, frame_start_list, n_frame. will need to check other code for consistency
        # just commenting out old code for now, to have as reference
        
        """
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
        """
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # this line now obsolete:
        # self.vid_list = [".".join(f.split(".")[:-1]) for f in listdir(db_dir) if os.path.isfile(f)]
        # this line now obsolete:
        # self.frame_dict = data_import.load_video(vid_list, num_frames, frame_start_list, seed = 1)
        
        for f in listdir(db_dir):
            train_pickle = open(f, 'rb')
            frame_dict = pickle.load(train_pickle)

        self.train_data = create_dataset(frame_dict, resize, verify_movement = True, n_frame, seed = 1, display = False)
        
        self.file_len = num_frames*n_frame

        """
        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])
        self.file_len = len(self.triplet_list)
        """

    # because they treat their database as sets of three frames, it might not make sense to separate the training set into x and y. or we need to replace the getitem with something that makes more sense with the x/y dictionaries
    
    def __getitem__(self, index):
        
        frame = self.train_data[str(index)]
        
        frame0 = self.transform(frame[:,:,:,0])
        frame1 = self.transform(frame[:,:,:,2])
        frame2 = self.transform(frame[:,:,:,1])
        
        """
        frame0 = self.transform(Image.open(self.triplet_list[index] + "/frame0.png"))
        frame1 = self.transform(Image.open(self.triplet_list[index] + "/frame1.png"))
        frame2 = self.transform(Image.open(self.triplet_list[index] + "/frame2.png"))
        """

        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
