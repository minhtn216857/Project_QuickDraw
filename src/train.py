import shutil

import torch
import os
import numpy as np
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Mydataset
from model import Module_QuickDraw
from Plot_Confusion_Matrix import plot_confusion_matrix
from torchvision.transforms import Compose, Resize, ToTensor
import argparse
import time
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from config import classes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, default="/home/minhtran/Documents/MDEEP_LEARNING/Project-QuickDraw/data")
    parser.add_argument("--batch_size", "-b", type =int, default=16)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--lr", '-lr', type=float, default=1e-2)
    parser.add_argument("--num_worker", '-n', type=int, default=4)
    parser.add_argument("--log_path", '-l', type=str, default='/home/minhtran/Documents/MDEEP_LEARNING/Project-QuickDraw/src/tensorboard/quickdraw')
    parser.add_argument("--checkpoint_path", '-c', type=str, default='/home/minhtran/Documents/MDEEP_LEARNING/Project-QuickDraw/src/checkpoint/quickdraw')
    parser.add_argument("-f", required=False)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Mydataset(args.data_path)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=True,
        drop_last=True
    )
    test_dataset = Mydataset(args.data_path, mode='test')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=False,
        drop_last=True
    )
    model = Module_QuickDraw()
    model.to(device)

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'last.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)


    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='cyan')
        for id, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch+1, args.epochs, loss.item()))
            writer.add_scalar("Train/loss",loss, id + epoch * len(train_dataloader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_losses = []
        all_predict = []
        all_labels = []
        # Tat dropout, tat batch_norm, backward
        model.eval()
        with torch.no_grad():
            for id, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                all_losses.append(loss.item())
                predict = torch.argmax(output,dim=1)
                all_predict.extend(predict.tolist())
                all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_predict)
        mean_loss = np.mean(all_losses)
        writer.add_scalar("Validation/loss", mean_loss, epoch)
        writer.add_scalar("Validation/acc", acc, epoch)
        conf_matrix = confusion_matrix(all_labels, all_predict)
        plot_confusion_matrix(writer, conf_matrix, classes, epoch)
        print("Epoch: {}/{}. Loss: {:0.4f}. Acc: {:0.4f}".format(epoch+1, args.epochs, mean_loss, acc))

        # save model
        checkpoint = {
            "epoch":epoch,
            "best_acc": acc,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, 'last.pt'))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pt'))
            acc = best_acc


if __name__ == '__main__':
    args = get_args()
    train(args)