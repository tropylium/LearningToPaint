import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.stroke_gen import *

import torch.optim as optim
from tqdm.auto import tqdm
import time


def save_model(net, save_file):
    torch.save(net.state_dict(), save_file)

def load_weights(net, from_file):
    pretrained_dict = torch.load(from_file)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
def gen_data(batch_size):
    # generate synthetic training data using stoke_gen.draw()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 10)
        train_batch.append(f)
        ground_truth.append(draw(f))
    train_batch = torch.tensor(train_batch)
    ground_truth = torch.tensor(ground_truth)
    return train_batch, ground_truth

def train_model(
    file="renderer.pkl", 
    total_steps=500000, 
    batch_size=64, 
    log_file="train_log/"
):
    print("Initializing...")
    writer = TensorBoard(log_file)
    criterion = nn.MSELoss()
    net = FCN()
    load_weights(net, file)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=3e-6)
    generator = FastStrokeGenerator(batch_size, 128, torch.device("cuda"))
    print("Finished init, starting training.")
    
    pbar = tqdm(range(total_steps))
    for step in pbar:
        net.train()

#         start = time.time()
        # generate synthetic training data using stoke_gen.draw()
        train_batch, ground_truth = generator.get_batch() #gen_data(batch_size)
#         train_batch = train_batch.float().cuda()
#         ground_truth = ground_truth.float().cuda()
        finish = time.time()
        tqdm.write(f"Generating data took: {finish - start}")
        start = finish

        # Training boilerplate
        gen = net(train_batch)
        optimizer.zero_grad()
        loss = criterion(gen, ground_truth)
        loss.backward()
        optimizer.step()
#         finish = time.time()
#         tqdm.write(f"Training took: {finish - start}")
#         start = finish

        # decay learning rate according to step
        pbar.set_description(f"loss: {loss.item():3.6f}")
        if step < 200000:
            lr = 1e-4
        elif step < 400000:
            lr = 1e-5
        else:
            lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
#         finish = time.time()
#         tqdm.write(f"Writing took: {finish - start}")
#         start = finish

        # log stuff
        writer.add_scalar("train/loss", loss.item(), step)
#         finish = time.time()
#         tqdm.write(f"Logging took: {finish - start}")
#         start = finish
        if step % 100 == 0:
            net.eval()
            gen = net(train_batch)
            loss = criterion(gen, ground_truth)
            writer.add_scalar("val/loss", loss.item(), step)
            for i in range(32):
                G = gen[i].cpu().data.numpy()
                GT = ground_truth[i].cpu().data.numpy()
                writer.add_image("train/gen{}.png".format(i), G, step)
                writer.add_image("train/ground_truth{}.png".format(i), GT, step)

        # save model every once in a while
        if step % 1000 == 0:
            save_model(net, file)
        
#         finish = time.time()
#         tqdm.write(f"Everything else took: {finish - start}")
#         start = finish

if __name__ == '__main__':
    train_model(
        total_steps=100
    )