import pandas as pd
import torch

from dataset import dataloader

def infer(img_dir, annotations_path, device, norm_size):
    raw_data = pd.read_csv(annotations_path)

    infer_dataloader = dataloader(img_dir, annotations_path, norm_size=norm_size, n_types=24, batch_size=1)
    net = torch.load("./checkpoint/resnet/resnet_model_300.pth")
    net = net.to(device)

    image_list = list()
    label_list = list()
    net.eval()
    for step, (img, label) in enumerate(infer_dataloader):
        img, label = img.type(torch.float).to(device), label.type(torch.long).to(device)
        pred = torch.argmax(net(img), dim = -1)
        
        image_list.append(raw_data.iloc[step, 0])
        label_list.append(pred.item())

    df_infer = pd.DataFrame({'image': image_list, 'label': label_list})
    df_infer.to_csv(annotations_path, index=None)

if __name__ == '__main__':
    infer(img_dir='./data/test',
          annotations_path='./data/test.csv',
          device='cuda',
          norm_size=(64, 64))