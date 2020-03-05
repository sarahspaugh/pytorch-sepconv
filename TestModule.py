from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os
from data_utils/data_import import load_video


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Middlebury_eval: # hopefully we just never use this? 
    def __init__(self, input_dir):
        self.im_list = ['Army', 'Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Grove', 'Mequon', 'Schefflera', 'Teddy', 'Urban', 'Wooden', 'Yosemite']


class Middlebury_other:
    def __init__(self, input_dir):
        #self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        # i think next three lines are also useless
        # jk they get to 
        # going to turn the 'test' video into these lists of frames
        # need to make sure we're calling along the right index throughout code when we go to pull out a single frame
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []

        self.vid_list = [".".join(f.split(".")[:-1]) for f in listdir(input_dir) if os.path.isfile(f)]
        self.frame_dict = load_video(self.vid_list, input_dir, segment_list)
        # ^^ right now that line is going to break because this function doesn't get segment_list passed in

        self.fr_list = list(self.frame_dict.keys())

        # during this next bit they were doing some kind of "unsqueeze" thing to their images. idk if we need or want this
        for item in self.fr_list:
            frame = self.frame_dict[item]

            first = frame[:,:,:,0]
            second = frame[:,:,:,1]
            groundtr = frame[:,:,:,2] # is this still the right order?

            # I also don't know if we need the "to variable" bit. because we're kind of already doing that.
            # setting axis to 3. hopefully that is the right one to get these back out of the million dim array the right way? 
            self.input0_list.append(to_variable(self.transform(first).unsqueeze(0)), axis=3)
            self.input1_list.append(to_variable(self.transform(second).unsqueeze(0)), axis=3)
            self.gt_list.append(to_variable(self.transform(groundtr).unsqueeze(0)), axis=3)
            # lololol


    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        
        # this part is probably going to be a trainwreck
        for idx in range(len(self.fr_list)):

            # this is where it writes the output file structure
            # need to figure out a better output file strucutre probably, because right now it's doing the stupid "subfolders of pictures" thing
            if not os.path.exists(output_dir + '/' + self.fr_list[idx]):
                os.makedirs(output_dir + '/' + self.fr_list[idx])
            
            # this part should be fine, just pulling frames out of our newly constructed stacked arrays
            frame_out = model(self.input0_list[:,:,:,idx], self.input1_list[:,:,:,idx])
            gt = self.gt_list[:,:,:,idx]

            # checking goodness of interp
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr

            # idk what this is. putting the outputs in the dumb output file tree. also need to fix if we change that
            imwrite(frame_out, output_dir + '/' + self.fr_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.fr_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')

            if logfile is not None:
                logfile.write(msg)

        av_psnr /= len(self.fr_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
