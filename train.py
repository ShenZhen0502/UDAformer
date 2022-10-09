import argparse
from my_dataset import MyDataSet as data
from model import UFUformer as net
import cv2
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

ap=argparse.ArgumentParser()
ap.add_argument('-b','--batch',help='the number of batch',type=int,default='4')
ap.add_argument('-e','--epoch',help='the number of training',type=int,default='500')
ap.add_argument('-r','--resume',help='the choice of resume',type=bool,default=False)

args=vars(ap.parse_args())

def log_images(writer, img, out,ll256,gt, it):

    images_array = vutils.make_grid(img).to('cpu')
    out_array = vutils.make_grid(out * 255).to('cpu').detach()
    ll256_array = vutils.make_grid(ll256 * 255).to('cpu').detach()
    gt = vutils.make_grid(gt).to('cpu')

    writer.add_image('input', images_array, it)
    writer.add_image('out', out_array, it)
    writer.add_image('ll256',ll256_array,it)
    writer.add_image('gt',gt,it)


net=net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args['resume']:
    files = '.\\checkpoint\\checkpoint_255_epoch.pkl'
    assert os.path.isfile(files), "{} is not a file.".format(args['resume'])
    state = torch.load(files)
    net.load_state_dict(state['model'])
    iteration = state['epoch'] + 1
    optimizer = state['optimizer']
    print("Checkpoint is loaded at {} | epochs: {}".format(args['resume'], iteration))
else:
    iteration = 0


da=data('D:\\SZ\\UIEB')
dataloder=DataLoader(da,batch_size=args['batch'],shuffle=True)

optimizer=torch.optim.Adam(lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,params=net.parameters())
scheduler = StepLR(optimizer, step_size=150, gamma=0.5)

def proimage(im):
    images = im[image_idx, :, :, :].clone().detach().requires_grad_(False)
    image = torch.transpose(images, 0, 1)
    image = torch.transpose(image, 1, 2).cpu().numpy() * 255
    return image

writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
j = 0
h = 0
for iter in range(iteration,args['epoch']):
    print(iter)
    prob = tqdm(enumerate(dataloder),total=len(dataloder))
    if iter < 1000:
        L1 = nn.MSELoss()
    else:
        L1 = nn.L1Loss()
    for i,data in prob:
        gt=torch.tensor(data[0].numpy(),requires_grad=True,device='cuda')
        raw=torch.tensor(data[1].numpy(),requires_grad=True,device='cuda')
        net.to('cuda')
        a=net(raw)
        L1loss = L1(a,gt)
        loss = L1loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prob.set_postfix(Loss1=L1loss)
        h = h+1
        writer.add_scalar('loss',loss,h)

        c = a
        if i % 100 ==0:
            j += 100
            image_idx =random.randint(0, 0)
            predi=c[image_idx,:,:,:].clone().detach().requires_grad_(False)
            predi=torch.transpose(predi,0,1)
            predi=torch.transpose(predi,1,2).cpu().numpy()*255
            gti=proimage(gt)
            rawi=proimage(raw)
            image=np.concatenate((rawi,predi,gti),axis=1)
            image_name='sample12' + "/out" + str(iter)+'_'+str(i) + ".png"
            cv2.imwrite(image_name,image)

    if (iter + 1) % 1 == 0:
        checkpoint = {"model": net.state_dict(),
                      "optimizer": optimizer,
                      "epoch": iter}

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        path_checkpoint = "checkpoint/checkpoint_{}_epoch.pkl".format(iter)
        torch.save(checkpoint, path_checkpoint)

    scheduler.step()
    print(optimizer.param_groups[0]['lr'])