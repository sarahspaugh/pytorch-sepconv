from TorchDB import DBreader_frame_interpolation
from datautils import sort_dataset, data_import


# setup to try using various data processing fn's
num_frames = 30
frame_start_list = [300]
db_dir = './train_db'
n_frame = 1

dataset = DBreader_frame_interpolation(db_dir, )