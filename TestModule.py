from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os
from data_import import load_p_video
import numpy as np


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
        self.frame_dict = load_p_video(input_dir)
        

        self.fr_list = list(self.frame_dict.keys())
        # during this next bit they were doing some kind of "unsqueeze" thing to their images. idk if we need or want this
        for item in self.fr_list:
            frame = self.frame_dict[item]

            first = frame[0,:,:,:]
            second = frame[1,:,:,:]
            groundtr = frame[2,:,:,:] # switched to new frame stacking on 1st axis
            
            
            shape1, shape2, shape3 = np.shape(first)[0], np.shape(first)[1], np.shape(first)[2]
            first = np.reshape(first, (1,shape1, shape2, shape3))
            second = np.reshape(second, (1, shape1, shape2, shape3))
            groundtr = np.reshape(groundtr, (1, shape1, shape2, shape3))

            self.fourDshape = np.shape(first)

            # removed from each list item a call of "tovariable(blah blah blah) and .unsqueeze(0), see pre-forked repo for more details"
            # setting axis to 0 to match changes to other functions with stacked frames. idk if it's good
            if (np.ndim(self.input0_list) == 1):
              self.input0_list = first
              self.input1_list = second
              self.gt_list = groundtr
            else:
                # self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))

              self.input0_list = np.concatenate((self.input0_list, first), axis=0)
              self.input1_list = np.concatenate((self.input1_list, second), axis=0)
              self.gt_list = np.concatenate((self.gt_list, groundtr), axis=0)
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

            # old line : frame_out = model(self.input0_list[idx], self.input1_list[idx])

            frame_out = model(to_variable(self.transform(np.reshape(self.input0_list[idx,:,:,:]), self.fourDshape)), to_variable(np.reshape(self.transform(self.input1_list[idx,:,:,:]),  self.fourDshape)
            gt = to_variable(self.transform(np.reshape(self.gt_list[idx,:,:,:],  self.fourDshape)))

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
