import torch
import torchvision.transforms as transforms
from model import UFUformer as net
import cv2
import torch.nn
import time
import os

torch.nn.Module.dump_patches = True
nets = net()
nets.to('cuda')
check=torch.load(r'checkpoint\checkpoint_489_epoch.pkl')
nets.load_state_dict(check['model'])
nets.eval()
img_dir=r'.\data\test data\underwater'


# video test
def testvideo():
    videoinpath  = r'.\fish.mp4'
    videooutpath = r'.\videoname_out_fish.avi'
    capture = cv2.VideoCapture(videoinpath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(videooutpath ,fourcc, fps,(640,480))
    if capture.isOpened():
        i=0
        while True:
            ret,img_src=capture.read()
            if not ret:break
            img = cv2.resize(img_src,(640,480))
            transform = transforms.ToTensor()
            imgs = transform(img).float()
            imgs = torch.unsqueeze(imgs, dim=0)
            imgs = torch.tensor(imgs, requires_grad=False, device='cuda')
            begin_time = time.time()
            outimg = nets(imgs)
            end_time = time.time()
            print('mean time per_frame:', (end_time - begin_time))
            outimg = outimg.clone().detach().requires_grad_(False)
            outimg = torch.clamp(outimg, 0, 1)
            outimg = torch.squeeze(outimg)
            out = torch.transpose(outimg, 0, 1)
            out = torch.transpose(out, 1, 2).cpu().numpy() * 255
            writer.write(out.astype('int8'))
            i=i+1
            print(i)
            cv2.imwrite('output24\\'+str(i)+'a.png',out)
    else:
        print('视频打开失败！')
    writer.release()

# image test
def testimage():
    with torch.no_grad():
        for i in os.listdir(img_dir):
            img=cv2.imread(img_dir+'\\'+i)
            img = cv2.resize(img,(256, 256))
            transform=transforms.ToTensor()
            imgs=transform(img).float()
            imgs=torch.unsqueeze(imgs,dim=0)
            imgs=torch.tensor(imgs,requires_grad=False,device='cuda')
            outimg=nets(imgs)
            outimg=outimg.clone().detach().requires_grad_(False)
            outimg=torch.squeeze(outimg)
            out=torch.transpose(outimg,0,1)
            out=torch.transpose(out,1,2).cpu().numpy()*255
            cv2.imwrite(r'D:\SZ\other_img\de_shiftedcolor\\'+i.split('.')[0]+'.'+i.split('.')[-1],out)
            print(i)

if __name__ == '__main__':
    testimage()
