import gradio as gr
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from models import Net,NetConv

net = torch.load('mnist.pth')
net.eval()

net_conv = torch.load('mnist_conv.pth')
net_conv.eval()

def predict(img):
    arr = np.array(img) / 255  # Assuming img is in the range [0, 255]
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = torch.from_numpy(arr).float()  # Convert to PyTorch tensor
    output = net(arr)
    topk_values, topk_indices = torch.topk(output, 2)  # Get the top 2 classes
    return [str(k) for k in topk_indices[0].tolist()]

def predict_conv(img):
    arr = np.array(img) / 255  # Assuming img is in the range [0, 255]
    arr = np.expand_dims(arr, axis=0)  # Conv needs one more dimension
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension
    arr = torch.from_numpy(arr).float()  # Convert to PyTorch tensor
    output = net_conv(arr)
    topk_values, topk_indices = torch.topk(output, 2)  # Get the top 2 classes
    return [str(k) for k in topk_indices[0].tolist()]





with gr.Blocks() as iface:
    gr.Markdown("# MNIST + Gradio End to End")
    gr.HTML("Shows end to end MNIST training with Gradio interface")
    with gr.Tab("Linear Model"):
        with gr.Row():
            with gr.Column():
                sp = gr.Sketchpad(shape=(28, 28))
                with gr.Row():
                    with gr.Column():
                        pred_button = gr.Button("Predict")
                    with gr.Column():
                        clear_button = gr.Button("Clear")
            with gr.Column():
                label1 = gr.Label(label='1st Pred')
                label2 = gr.Label(label='2nd Pred')
    
    with gr.Tab("Convolution Model"):
        with gr.Row():
            with gr.Column():
                sp_conv = gr.Sketchpad(shape=(28, 28))
                with gr.Row():
                    with gr.Column():
                        pred_conv_button = gr.Button("Predict")
                    with gr.Column():
                        clear_button_conv = gr.Button("Clear")
            with gr.Column():
                label1_conv = gr.Label(label='1st Pred')
                label2_conv = gr.Label(label='2nd Pred')
    def clear():
        return ['','',None,'','',None]
    pred_button.click(predict, inputs=sp, outputs=[label1,label2])
    pred_conv_button.click(predict_conv, inputs=sp_conv, outputs=[label1_conv,label2_conv])
    clear_button.click( lambda: ['','',None], None, [label1,label2,sp,], queue=False)
    clear_button_conv.click( lambda: ['','',None], None, [label1_conv,label2_conv, sp_conv], queue=False)


iface.launch()