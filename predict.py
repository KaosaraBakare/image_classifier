import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms,models
import time
import argparse
from PIL import Image
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str)
    parser.add_argument('checkpoint', type = str)
    parser.add_argument('--category_names', type = str, default='cat_to_name.json')
    parser.add_argument('--top_k', type = int, default=5)
    parser.add_argument('--gpu', action='store_true')
    
    return parser.parse_args()
def print_command_lines(in_arg):
    if in_arg is None:
        print("*Doesn't Check the command line arguments because 'get_input_args' hasn't been defined yet")
    else:
        print("Command Line Arguments:\n image_path=", in_arg.image_path,"\n checkpoint_path=", in_arg.checkpoint,
             "\n category_names=", in_arg.category_names,
             "\n Device=",'cuda' if in_arg.gpu else 'cpu')
def load_checkpoint(filename):
    device = 'cuda:0' if in_args.gpu else 'cpu'
    checkpoint = torch.load(filename, map_location=device)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained =True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
def process_image(image):
    image = Image.open(image)
    image= image.resize((256,256))
    value = 0.5*(256-224)
    image = image.crop((value,value,256-value,256-value))
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229,0.224,0.225])
    image = (image-mean)/std
    return image.transpose(2,0,1)
def prediction(image_path, checkpoint_dir, category_names, topk, devices):
    model = load_checkpoint(checkpoint_dir)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = Variable(image)
    if cuda:
        image = image.cuda()
    output = model.forward(image)
    probabilities = torch.exp(output).data
    probability = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
    label = []
    for i in range(topk):
        label.append(cat_to_name[ind[index[i]]])
        
    print("{:<20} {:<20}".format('Class','Probability'))
    for i in range(topk):
        print("{:<20} {:<20}".format(label[i], probability[i]))
in_args = arguments()
print_command_lines(in_args)
device = 'cuda' if in_args.gpu else 'cpu'
prediction(in_args.image_path,in_args.checkpoint, in_args.category_names,in_args.top_k, device)

    