import numpy as np
from os import listdir
from os.path import join, isdir
import pickle
from tqdm import tqdm
import ~/data_import

# need to make sure this folder exists with videos whenever data is to be sorted
rawdata_dir = './put_all_video_here'

test_dir = './test_db'
train_dir = './train_db'


def sort_dataset(num_frames, frame_start_list, tr_split):
	# tr_split is the 0<n<1 fraction of data to be used for training (the rest will be for test)
	# num_frames is the number of frames to get from each video
	# 

	vid_list = [".".join(f.split(".")[:-1]) for f in listdir(rawdata_dir) if os.path.isfile(f)]

	master_framedict = data_import.load_video(vid_list, num_frames, frame_start_list, seed = 1)
	train_dict, test_dict = {}, {}
	train_key = 0
	test_key = 0

	master_keys = list(master_framedict.keys())
	np.random.shuffle(master_keys)

	for i in tqdm(range(len(master_keys))):
		frame = master_framedict[master_keys[i]]

		if (i < tr_split*len(master_keys)):
			train_dict[str(train_key)] = frame
			train_key += 1
		else:
			test_dict[str(test_key)] = frame
			test_key += 1

	os.makedirs(test_dir)
	os.makedirs(train_dir)

	pickle_test = open(test_dir+"/db.pickle", "wb")
	pickle.dump(test_dict, pickle_test)
	pickle_test.close()

	pickle_train = open(train_dir+'/db.pickle', "wb")
	pickle.dump(train_dict, pickle_train)
	pickle_test.close()










