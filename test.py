import argparse
from TestModule import Middlebury_other
from model import SepConvNet
import torch
from pathlib import Path
import json

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--input', type=str, default='./test_db/input')
parser.add_argument('--gt', type=str, default='./test_db/gt')
parser.add_argument('--output', type=str, default='./output_sepconv_pytorch/result')
parser.add_argument('--checkpoint', type=str, default='./output_sepconv_pytorch/checkpoint/model_epoch010.pth')


def main():
    args = parser.parse_args()
    input_dir = args.input
    gt_dir = args.gt
    output_dir = args.output
    ckpt = args.checkpoint

    print("Reading Test DB...")
    TestDB = Middlebury_other(input_dir, gt_dir)
    print("Loading the Model...")
    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']


    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']

    print("Test Start...")
    
    if torch.cuda.is_available():
        model = model.cuda()

    TestDB.Test(model, output_dir)


if __name__ == "__main__":
    main()