import gradio as gr
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import model

net = torch.load('mnist.pth')
net.eval()

def predict(img):
    arr = np.array(img) / 255  # Assuming img is in the range [0, 255]
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = torch.from_numpy(arr).float()  # Convert to PyTorch tensor
    output = net(arr)
    topk_values, topk_indices = torch.topk(output, 2)  # Get the top 2 classes
    return [str(k) for k in topk_indices[0].tolist()]


sp = gr.Sketchpad(shape=(28, 28))

gr.Interface(fn=predict,
             inputs=sp,
             outputs=['label','label']).launch()
