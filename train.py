from TorchDB import DBreader_frame_interpolation
from torch.utils.data import DataLoader
from model import SepConvNet
import argparse
from torchvision import transforms
import torch
from torch.autograd import Variable
import os
from TestModule import Middlebury_other

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--video_in', type=str, default='./raw_vid_db')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output_sepconv_pytorch')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--split_vid_out', type=str, default=['./train_db','./dev_db','./test_db'])
parser.add_argument('--splits', type=float, default=(0.8, 0.1, 0.1))
parser.add_argument('--s_list', type=int, default=[(500, 505), (500, 505)])

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():

    # first a bunch of directory manipulation 
    args = parser.parse_args()
    split_vid_out  = args.split_vid_out
    train_db, dev_db, test_db = split_vid_out[0], split_vid_out[1], split_vid_out[2]
    input_dir = args.video_in

    segment_list = args.s_list
    split_params = args.splits


    if not os.path.exists(input_dir):
        raise directoryError("input directory name not specified correctly")

    raw_vid_list = [".".join(f.split(".")[:-1]) for f in os.listdir(input_dir) if os.path.isfile(f)]

    if (len(raw_vid_list) == 0):
        raise directoryError("pls check input directory")

    # split raw video to separate directories for train/dev/test
    # if the directories already exist with old video, clear them out to start fresh
    
    for d in [train_db, test_db, dev_db]:
        if not os.path.exists(d):
            os.makedirs(d)
        else:
            for f in os.listdir(d):
                f_path = os.path.join(d, f)
                try:
                    if os.path.isfile(f_path) or os.path.islink(f_path):
                        os.unlink(f_path)
                    elif os.path.isdir(f_path):
                        shutil.rmtree(f_path)
                except directoryError:
                    print('Failed to clear directories')


    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # import and sort video data into appropriate datasets
    data_import.split_video(raw_vid_list, input_dir, split_vid_out, segment_list, split_params)

    # start log file
    logfile = open(args.out_dir + '/log.txt', 'w')
    logfile.write('batch_size: ' + str(args.batch_size) + '\n')

    total_epoch = args.epochs
    batch_size = args.batch_size

    dataset = DBreader_frame_interpolation(train_db)# leave resize as none here b/c we already did during sort
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    TestDB = Middlebury_other(args.test_input, args.gt)
    test_output_dir = args.out_dir + '/result'

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel
        model = SepConvNet(kernel_size=kernel_size)

    logfile.write('kernel_size: ' + str(kernel_size) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()

    max_step = train_loader.__len__()

    model.eval()
    TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')

    while True:
        if model.epoch.item() == total_epoch:
            break
        model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(train_loader):
            frame0 = to_variable(frame0)
            frame1 = to_variable(frame1)
            frame2 = to_variable(frame2)
            loss = model.train_model(frame0, frame2, frame1)
            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ', '[' + str(model.epoch.item()) + '/' + str(total_epoch) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(max_step) + ']', 'train loss: ', loss.item()))
        model.increase_epoch()
        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckpt_dir + '/model_epoch' + str(model.epoch.item()).zfill(3) + '.pth')
            model.eval()
            TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')
            logfile.write('\n')

    logfile.close()


if __name__ == "__main__":
    main()
