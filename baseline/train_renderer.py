import cv2
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.fast_stroke_gen import *

import torch.optim as optim
from tqdm.auto import tqdm
import time

import argparse
import os


def save_model(net, save_file):
    torch.save(net.state_dict(), save_file)

def load_weights(net, from_file):
    pretrained_dict = torch.load(from_file)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

def get_checkpoint_file(file, step):
    root, ext = os.path.splitext(file)
    return f"{root}_{step}{ext}"

def get_criterion():
#     return nn.MSELoss()
    mse_loss = nn.MSELoss()
    def my_loss(output, target):
        batch_size = output.shape[0]
        filled_target = target < (1.0 - 0.01)
        filled_counts = torch.sum(filled_target, dim=(1,2), dtype=torch.float32)
        empty_counts = -(filled_counts - output.numel()/batch_size)
#         print(filled_counts + 1, empty_counts + 1)
#         print(torch.sum(target.masked_fill(~filled_target, 0.0)))

        filled_loss = torch.sum((target.masked_fill(~filled_target, 0.0) - output.masked_fill(~filled_target, 0.0))**2, dim=(1,2))/(filled_counts + 1)  # avoid divide by 0
        
        empty_loss = torch.sum((target.masked_fill(filled_target, 0.0) - output.masked_fill(filled_target, 0.0))**2, dim=(1,2))/(empty_counts + 1)  # avoid divide by 0
        
#         print(filled_loss, empty_loss)
        
#         raise Exception()
        return torch.sum(filled_loss + empty_loss)/batch_size
    return my_loss

def get_renderer(key, batch_size, width=128):
    renderers = {
        "qbc": QbcStroke,
        "squareblur": SquareAndBlurStroke,
        "rectEllipse": RectAndEllipseStroke,
    }
    return renderers[key](batch_size, 128)

def train_model(
    model_dir,
    renderer,
    start,
    steps,
    decay,
    batch_size,
    checkpoint_interval,
    separate_checkpoints,
    log_file,
):
    print("Initializing...")
    
    device = torch.device("cuda")
    num_devices = torch.cuda.device_count()
    print(f"Using {num_devices} gpus.")
    
    generator = get_renderer(renderer, batch_size, 128)
    net = FCN(generator.stroke_size())
    
    model_path = os.path.join(".", "models_renderer", model_dir)
    os.makedirs(model_path, exist_ok=True)
    print(f"Saving at {model_path}")
    
    model_file = os.path.join(model_path, "model.pkl")
    if os.path.exists(model_file):
        load_weights(net, model_file)
    else:
        save_model(net, model_file)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    
    criterion = get_criterion()
    optimizer = optim.Adam(net.parameters(), lr=3e-6*batch_size/64/num_devices)
    
    log_dir = os.path.join(log_file, model_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = TensorBoard(log_dir)
    
    print("Finished init, starting training...")
    
    pbar = tqdm(range(start, steps + start))
    for step in pbar:
        net.train()

        # start = time.time()
        # generate synthetic training data using stoke_gen.draw()
        # strokes = generator.module.generate_strokes()
        strokes = generator.generate_strokes().to(device)
        images = nn.parallel.data_parallel(generator, strokes) #gen_data(batch_size)
        # images = generator(strokes)
        
        # finish = time.time()
        # tqdm.write(f"Generating data took: {finish - start}")
        # start = finish

        train_batch = strokes
        ground_truth = images

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
        if decay:
            if step < 0.4*steps:
                lr = 1e-4*batch_size/64/num_devices
            elif step < 0.8*steps:
                lr = 1e-5*batch_size/64/num_devices
            else:
                lr = 1e-6*batch_size/64/num_devices
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # log stuff
        writer.add_scalar("train/loss", loss.item(), step)
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
        if step % checkpoint_interval == 0:
            if separate_checkpoints:
                save_model(net.module, get_checkpoint_file(model_file, step))
            else:
                save_model(net.module, model_file)
    save_model(net.module, model_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # debug
    parser.add_argument("-debug", "-d", help="Convenience flag for debug configuration.", action="store_true")
    
    # model arguments
    parser.add_argument("model_dir", help="Directory to load model from. If doesn't exist, instead creates this directory and trains a model from scratch.", type=str)
    
    parser.add_argument("-renderer", "-r", help="Renderer selection", type=str, default="qbc")
    
    # optional training arguments
    parser.add_argument("-start", "-st", help="Starting step, for use with checkpointing", type=int, default=0)
    parser.add_argument("-steps", "-s", help="Steps to run for.", type=int, default=25000)
    parser.add_argument("-decay", "-de", help="Decay learning rate over time.", action="store_true")
    parser.add_argument("-batch_size", "-b", help="Batch size at each step.", type=int, default=1024)
    
    # logging/ saving arguments
    parser.add_argument("-checkpoint_interval", "-ci", help="Interval at which to save checkpoints", type=int, default=500)
    parser.add_argument("-separate_checkpoints", "-sc", help="Save separately at each checkpoint, with _{step} appended to filename.", action="store_true")
    parser.add_argument("-log_file", "-l", help="Log directory.", type=str, default="train_log/")
    
    args = parser.parse_args()
    if args.debug:
        args = argparse.Namespace(
            debug=True,
            model_dir=args.model_dir,
            renderer=args.renderer,
            start=0,
            steps=100,
            decay=False,
            batch_size=64,
            checkpoint_interval=50,
            separate_checkpoints=True,
            log_file=args.log_file,
        )
    
    train_model_args = vars(args)
    train_model_args.pop("debug")
#     print(train_model_args)
    train_model(
        **train_model_args
    )
