import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import dataloader
import models.resnet as resnet
import models.senet as senet
import visualize

def dataset_random_division_total(data_dir, img_dir, annotations_path, data_rate):
    train_img_dir = os.path.join(data_dir, 'train_tmp')
    train_annotations_path = os.path.join(data_dir, 'train_tmp.csv')
    if os.path.exists(train_img_dir):
        shutil.rmtree(train_img_dir)
    os.mkdir(train_img_dir)
    eval_img_dir = os.path.join(data_dir, 'eval_tmp')
    eval_annotations_path = os.path.join(data_dir, 'eval_tmp.csv')
    if os.path.exists(eval_img_dir):
        shutil.rmtree(eval_img_dir)
    os.mkdir(eval_img_dir)

    raw_data = pd.read_csv(annotations_path)
    raw_data_len = len(raw_data)
    shuffled_idx = np.random.permutation(np.array(range(raw_data_len)))  # len: the number of rows
    split_idx = int(raw_data_len * data_rate)
    
    column_name_list = raw_data.columns.values.tolist()
    img_name_list = list()
    label_list = list()
    for idx in range(split_idx):
        img_name = raw_data.iloc[shuffled_idx[idx], 0]
        shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(train_img_dir, img_name))
        img_name_list.append(img_name)
        label_list.append(raw_data.iloc[shuffled_idx[idx], 1])
    df_train = pd.DataFrame({column_name_list[0]: img_name_list, column_name_list[1]: label_list})
    df_train.to_csv(train_annotations_path, index=None)
    
    img_name_list = list()
    label_list = list()
    for idx in range(split_idx, raw_data_len):
        img_name = raw_data.iloc[shuffled_idx[idx], 0]
        shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(eval_img_dir, img_name))
        img_name_list.append(img_name)
        label_list.append(raw_data.iloc[shuffled_idx[idx], 1])
    df_eval = pd.DataFrame({column_name_list[0]: img_name_list, column_name_list[1]: label_list})
    df_eval.to_csv(eval_annotations_path, index=None)

    return train_img_dir, train_annotations_path, eval_img_dir, eval_annotations_path

def dataset_random_division(data_dir, img_dir, annotations_path, data_rate, n_classes):
    train_img_dir = os.path.join(data_dir, 'train_tmp')
    train_annotations_path = os.path.join(data_dir, 'train_tmp.csv')
    if os.path.exists(train_img_dir):
        shutil.rmtree(train_img_dir)
    os.mkdir(train_img_dir)
    eval_img_dir = os.path.join(data_dir, 'eval_tmp')
    eval_annotations_path = os.path.join(data_dir, 'eval_tmp.csv')
    if os.path.exists(eval_img_dir):
        shutil.rmtree(eval_img_dir)
    os.mkdir(eval_img_dir)

    raw_data = pd.read_csv(annotations_path)
    raw_data_len = len(raw_data)
    shuffled_idx = np.random.permutation(np.array(range(raw_data_len)))  # len: the number of rows

    dict_split_idx = raw_data['label'].value_counts().to_dict()
    for idx in range(n_classes):
        dict_split_idx[idx] = int(dict_split_idx[idx] * data_rate)

    column_name_list = raw_data.columns.values.tolist()
    img_name_list_train = list()
    label_list_train = list()
    img_name_list_eval = list()
    label_list_eval = list()
    for idx in range(raw_data_len):
        img_name = raw_data.iloc[shuffled_idx[idx], 0]
        label = raw_data.iloc[shuffled_idx[idx], 1]
        if label_list_train.count(label) < dict_split_idx[label]:
            shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(train_img_dir, img_name))
            img_name_list_train.append(img_name)
            label_list_train.append(label)
        else:
            shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(eval_img_dir, img_name))
            img_name_list_eval.append(img_name)
            label_list_eval.append(label)

    df_train = pd.DataFrame({column_name_list[0]: img_name_list_train, column_name_list[1]: label_list_train})
    df_train.to_csv(train_annotations_path, index=None)
    df_eval = pd.DataFrame({column_name_list[0]: img_name_list_eval, column_name_list[1]: label_list_eval})
    df_eval.to_csv(eval_annotations_path, index=None)

    return train_img_dir, train_annotations_path, eval_img_dir, eval_annotations_path


def train(train_img_dir, train_annotations_path, eval_img_dir, eval_annotations_path,
          norm_size,
          device, lr, momentum, weight_decay,
          n_epochs, n_epochs_save):
    
    
    train_dataloader = dataloader(train_img_dir, train_annotations_path, norm_size=norm_size, n_types=24, batch_size=32)
    eval_dataloader = dataloader(eval_img_dir, eval_annotations_path, norm_size=norm_size, n_types=24, batch_size=1)

    net = resnet.resnet34(24)
    # net = torch.load("./checkpoint/pre_resnet_model.pth")
    # net = senet.seresnet34(24)
    net = net.to(device)

    loss_func = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    optimizer = optim.AdamW(net.parameters(), lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=lr, 
                                              steps_per_epoch=int(len(train_dataloader)),
                                              epochs=n_epochs,
                                              anneal_strategy='cos')

    print("train begin!")
    loss_list = []
    accuracy_list = []
    loss_eval_list = []
    for epoch in range(n_epochs):
        net.train()
        loss_total = 0.
        
        for step, (img, label) in enumerate(train_dataloader):
            # print(step)
            img, label = img.type(torch.float).to(device), label.type(torch.float).to(device)
            pred = net(img)
            loss = loss_func(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        loss_avg = loss_total / len(train_dataloader)
        loss_list.append(loss_avg)
        print("eval begin!")
        accuracy_eval = eval_once(net, eval_dataloader, device, loss_func, loss_eval_list)
        accuracy_list.append(accuracy_eval)
        print('Epoch {:03d}: loss_train = {:.3f}, loss_val = {:.3f}, accuracy on validation set = {:.3f}'.format(epoch + 1, loss_avg, loss_eval_list[-1], accuracy_eval))

        scheduler.step()

        if (epoch+1) % n_epochs_save == 0:
            torch.save(net, "./checkpoint/resnet/resnet_model_{}.pth".format(epoch+1))
            # torch.save(net, "./checkpoint/senet/senet_model_{}.pth".format(epoch))
            visualize.draw_curve(['train_dataset loss', 'eval_dataset loss', 'eval_dataset accuracy'], 
                                     [loss_list, loss_eval_list, accuracy_list])
    

def eval_once(net, eval_dataloader, device, loss_func, loss_eval_list):
    net.eval()
    n_corrects = 0
    loss_total = 0.

    with torch.no_grad():
        for img, label in eval_dataloader:
            img, label = img.type(torch.float).to(device), label.type(torch.float).to(device)
            pred = net(img)
            loss = loss_func(pred, label)

            pred = torch.argmax(pred, dim=-1)
            label = torch.argmax(label.squeeze(dim=0), dim=-1)
            if pred == label: 
                n_corrects += 1

            loss_total += loss.item()

        loss_avg = loss_total / len(eval_dataloader)
        loss_eval_list.append(loss_avg)
        return float(n_corrects) / float(len(eval_dataloader))

if __name__ == '__main__':
    seed = 1204
    np.random.seed(seed)
    torch.manual_seed(seed)  # set CPU seed
    torch.cuda.manual_seed(seed)  # set present GPU seed
    torch.cuda.manual_seed_all(seed)  # set all GPU seed
    torch.backends.cudnn.deterministic = True  # set network structure

    train_img_dir, train_annotations_path, eval_img_dir, eval_annotations_path = dataset_random_division(
        data_dir='./data', 
        img_dir='./data/train', 
        annotations_path='./data/train.csv', 
        data_rate=0.9,
        n_classes=24,
    )

    train(train_img_dir=train_img_dir,
          train_annotations_path=train_annotations_path,
          eval_img_dir=eval_img_dir,
          eval_annotations_path=eval_annotations_path,
          norm_size=(64, 64),
          device='cuda',
          lr=0.01,
          momentum=0.9,
          weight_decay=0,
          n_epochs=300,
          n_epochs_save=20)
