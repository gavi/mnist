import gradio as gr
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import NetConv

    
net_conv = torch.load('mnist_conv.pth')
net_conv.eval()

def predict(img):
    arr = np.array(img) / 255  # Assuming img is in the range [0, 255]
    arr.reshape(28,28)
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = torch.from_numpy(arr).float()  # Convert to PyTorch tensor
    output = net_conv(arr)
    topk_values, topk_indices = torch.topk(output, 2)  # Get the top 2 classes
    return [str(k) for k in topk_indices[0].tolist()]

with gr.Blocks() as iface:
    gr.Markdown("# MNIST + Gradio End to End")
    gr.HTML("Shows end to end MNIST training with Gradio interface")
    with gr.Row():
        with gr.Column():
            sp = gr.Sketchpad(shape=(28, 28))
            with gr.Row():
                with gr.Column():
                    pred_button = gr.Button("Predict")
                with gr.Column():
                    clear = gr.Button("Clear")
        with gr.Column():
            label1 = gr.Label(label='1st Pred')
            label2 = gr.Label(label='2nd Pred')

    pred_button.click(predict, inputs=sp, outputs=[label1,label2])
    clear.click(lambda: None, None, sp, queue=False)
iface.launch()