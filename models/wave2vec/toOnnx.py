import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from module import Wave2Vec
import torch.onnx
import librosa
import numpy as np

@hydra.main(version_base=None, config_path="C:\\Users\\quchenfu\\Documents\\stuttering_detection\\models\\wave2vec\\configs", config_name='config')
def train(config):
    model = Wave2Vec(config)
    checkpoint_path = "C:\\Users\\quchenfu\\Downloads\\config\\lightning_logs\\version_0\\checkpoints\\epoch=0-Val_metrics-Mean_f1=0.0000.ckpt"  # Replace with the path to your checkpoint file
    # # Load the state_dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Use 'cpu' for map_location if necessary
    model.load_state_dict(checkpoint['state_dict'])
    # # Print the keys in the checkpoint
    # print(checkpoint.keys())
    # model.load_from_checkpoint(checkpoint_path, config=config)
    # # Optional: Load other items from the checkpoint
    # # For example, if you have optimizer state or other information, you can load them too
    # # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    array_shape = (1, 48000)
    spec = np.random.rand(*array_shape)
    spec = torch.tensor(spec, dtype=torch.float32)
    onnx_path = "model_wav.onnx"  # Path where you want to save the ONNX file
    torch.onnx.export(model, spec, onnx_path, verbose=True)
# Save the model as a .pt file



if __name__ == '__main__':
    train()
