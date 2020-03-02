
import cv2
print(cv2.__version__)
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from matplotlib.pyplot import imshow

def load_video(vid_list, num_frames, frame_start_list, seed = 1):
    #num_frames is a list of ints corresponding to the numbers of frames that you want to load from each clip
    #frame_start list is a list of ints corresponding to the first frame you want to load from each clip 
    #vid_list is a list of video file names without extensions (assumed to be mp4)
    frame_dict = {}
    
    for i in range(len(vid_list)):
        f_num = 3
        vidcap = cv2.VideoCapture(vid_list[i]+".mp4")
        
        for j in tqdm(range(frame_start_list[i])):
            success, i_waste = vidcap.read()
        success, i1 = vidcap.read()
        success, i2 = vidcap.read()
        success, i3 = vidcap.read()
        i1, i2, i3 = np.asarray(i1), np.asarray(i2), np.asarray(i3)
        pbar = tqdm(total=(num_frames[i]))
        while(success and f_num<num_frames[i]+3):
            f_id = vid_list[i]+"_f"+str(f_num-3)
            frame_dict[f_id] = np.stack((i1,i3,i2), axis=3)
            i3 = i2
            i2 = i1
            success, i1 = vidcap.read()
            f_num+=1
            pbar.update(1)
        pbar.close()
    return frame_dict

def create_tr_dataset(frame_dict, crop_size, verify_movement = True, n_frame=1, seed=1, display=False):
    #frame_dict is the dictionary from load_video
    #crop_size is the size that you want your dataset to be cropped to (d_y,d_x)
    #split_params is a tuple representing the fractional breakdown of your data into (train, dev, test) (moving to outside function)
    #verify_movement is a boolean that turns optical flow checking on and off
    #n_frame is the number of random cropped subframes that you want generated from each video frame 
    
    np.random.seed(seed)
    train_dict = {}
    #test_X, test_Y, train_X, train_Y, dev_X, dev_Y = {}, {}, {}, {}, {}, {}
    
    key_list = list(frame_dict.keys())
    
    training_key = 0
    
    np.random.shuffle(key_list)
    for i in tqdm(range(len(key_list))):
        frame = frame_dict[key_list[i]]
        
        
        for j in range(n_frame):
            train_dict[str(training_key)] = find_crop(frame, crop_size, display)
            training_key += 1
        
        """
        if(i<split_params[0]*len(key_list)):
            for j in range(n_frame):
                train_X[key_list[i]+"_"+str(j)], train_Y[key_list[i]+"_"+str(j)] = find_crop(frame, crop_size, display)
            
        elif(i<(split_params[0]+split_params[1])*len(key_list)):
             for j in range(n_frame):
                dev_X[key_list[i]+"_"+str(j)], dev_Y[key_list[i]+"_"+str(j)] = find_crop(frame, crop_size, display)
            
        else:
            for j in range(n_frame):
                test_X[key_list[i]+"_"+str(j)], test_Y[key_list[i]+"_"+str(j)] = find_crop(frame, crop_size, display)
                
        """

    return train_dict


def find_crop(frame, crop_size, display, verify_movement=True, attempt_max = 100, threshold = 30000, ):
    #frame is a matrix that should be (f_x, f_y, 3, 3) containing [F1, F3, F2] stacked along axis 4
    #crop_size is the size of the region to be used for training (d_y, d_x)
    #attempt_max is the number of attempts to find a cropped image with sufficient movement between F1 and F3
    #threshold is an arbitrary threshold for defining sufficient movement, looking for something nicer here
    
    attempts = 0
    n_fy, n_fx, n_fc, n_fi = frame.shape
    max_frame_X = []
    max_frame_Y = []
    max_mov = -1
    
    while(attempts<attempt_max):
        c_y, c_x = int(np.random.rand()*(n_fy-crop_size[0])), int(np.random.rand()*n_fx-crop_size[1])
        t_frame_X = frame[c_y:crop_size[0]+c_y,c_x:crop_size[1]+c_x,:,:-1]
        t_frame_Y = frame[c_y:crop_size[0]+c_y,c_x:crop_size[1]+c_x,:,-1]
        
        if(verify_movement==False):
            t_frame = np.stack((t_frame_X, t_frame_Y), axis=3)
            return t_frame
        
        mov = min(check_movement(t_frame_X[:,:,:,0], t_frame_Y[:,:,:]), check_movement(t_frame_X[:,:,:,1], t_frame_Y[:,:,:]))
        
        if(mov>max_mov):
            max_frame_X, max_frame_Y, max_mov = t_frame_X, t_frame_Y, mov       
        
        if(mov>threshold):
            if(display):
                show_frame(t_frame_X, t_frame_Y)
            t_frame = np.stack((t_frame_X, t_frame_Y), axis=3)
            return t_frame
        
        attempts+=1
        if(attempts==attempt_max):
            if(display):
                show_frame(max_frame_X, max_frame_Y)
            t_frame = np.stack((max_frame_X, max_frame_Y), axis=3)
            return t_frame
        
#basic motion detection for now, uses the norm of the difference of the two images
def check_movement(f1, f2):
    return np.linalg.norm(f1-f2)

def show_frame(f_x, f_y):
    plt.figure()
#     plt.imshow(np.concatenate((abs(f_x[:,:,:,0]-f_y[:,:,:]), abs(f_x[:,:,:,1]-f_y[:,:,:])), axis=1))
    plt.imshow(np.concatenate((f_x[:,:,:,0], f_y[:,:,:], f_x[:,:,:,1]), axis=1))
    
    
