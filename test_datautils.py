from TorchDB import DBreader_frame_interpolation
from datautils import sort_dataset, data_import


# setup to try using various data processing fn's
num_frames = 30
frame_start_list = [300]
db_dir = './train_db'
n_frame = 1

sort_dataset(num_frames, frame_start_list, 0.9)
dataset = DBreader_frame_interpolation(db_dir, num_frames, frame_start_list, n_frame, resize = (128, 128))

