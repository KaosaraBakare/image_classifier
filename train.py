import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms,models
import time
import argparse
from PIL import Image
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', type = str, default='checkpoint')
    parser.add_argument('--arch', type = str, default='densenet121')
    parser.add_argument('--learning_rate', type = float, default=0.03)
    parser.add_argument('--hidden_units', type = int, default=512)
    parser.add_argument('--epochs', type = int, default=1)
    parser.add_argument('--gpu', action='store_true')
    

    return parser.parse_args()
def print_command_lines(in_arg):
    if in_arg is None:
        print("*Doesn't Check the command line arguments because 'get_input_args' hasn't been defined yet")
    else:
        print("Command Line Arguments:\n data_dir=", in_arg.data_dir,"\n save_dir=", in_arg.save_dir,
             "\n arch=", in_arg.arch,"\n learning_rate =", in_arg.learning_rate, "\n hidden_units=", in_arg.hidden_units, "\n epochs=", in_arg.epochs,
             "\n Device=",'cuda' if in_arg.gpu else 'cpu')
def train(arch, data_dir, save_dir, epochs, learn_rate, hidden_units, device_):
    densenet121 = models.densenet121(pretrained=True)
    vgg19 = models.vgg19(pretrained=True)
    modelss = {'densenet121':densenet121, 'vgg19':vgg19}
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(224),
                                         transforms.ToTensor()])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])

    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    train_image_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size =32, shuffle = True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size =32)


    image_datasets = [train_image_datasets,valid_image_datasets]
    dataloaders = [train_dataloaders,valid_dataloaders]
 

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    cat_label_idx = dict()
    for i in cat_to_name:
        cat_label_idx[cat_to_name[i]] = int(i)
    # print(cat_label_idx)
    model = modelss[arch]
    for each in model.parameters():
        each.require_grad = False
    if(arch == 'densenet121'):
    
        classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.4),
                                  nn.Linear(hidden_units, int(hidden_units/2)),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.4),
                                  nn.Linear(int(hidden_units/2), 102),
                                  nn.LogSoftmax(dim = 1))
    else:
        classifier = nn.Sequential(nn.Linear(25088, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.4),
                                   nn.Linear(1024, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.4),
                                   nn.Linear(hidden_units, int(hidden_units/2)),
                                   nn.ReLU(),
                                   nn.Dropout(p = 0.4),
                                   nn.Linear(int(hidden_units/2), 102),
                                   nn.LogSoftmax(dim = 1))
        
    model.classifier = classifier
    # torch.cuda.empty_cache()
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.03)
    # import gc

    # del variable #delete unnecessary variables
    # gc.collect()

    cuda = torch.cuda.is_available()
    model.to(device)
    if cuda:
        model.cuda()
    else:
        model.cpu()
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    start = time.time()
    for epoch in range(epochs):
        for images, labels in train_dataloaders:
            #         images, labels = images.to(device), labels.to(device)
            if cuda:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            steps += 1
            optimizer.zero_grad()
            logp = model(images)
            loss = criterion(logp, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in valid_dataloaders:
                    if cuda:
                        images, labels = images.cuda(), labels.cuda()
                    else:
                        images, labels = Variable(images), Variable(labels)

                    steps += 1
                    optimizer.zero_grad()
                    logp = model.forward(images)
                    validb_loss = criterion(logp, labels)

                    valid_loss += validb_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logp)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch + 1}/{epochs}..\n ",
                  f"Train loss: {running_loss / len(train_dataloaders):.3f}.. "
                  f"Validation loss: {valid_loss / len(valid_dataloaders):.3f}.. "
                  f"Validation accuracy: {accuracy / len(valid_dataloaders):.3f}")
            loss = 0
            model.train()

    time_elapsed = time.time() - start
    print("\nTotal time : {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    
    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoints = {'arch':arch,
                   'learning_rate':0.03,
                   'batch_size':16,
                   'classifier':classifier,
                   'epochs':10,
                'optimizer':optimizer.state_dict(),
                   'state_dict':model.state_dict(),
                  'class_to_idx':model.class_to_idx
    }
    torch.save(checkpoints,save_dir )

in_args = arguments()
print_command_lines(in_args)
device = 'cuda' if in_args.gpu else 'cpu'
train(in_args.arch,in_args.data_dir,in_args.save_dir,in_args.epochs,in_args.learning_rate,in_args.hidden_units,device)
