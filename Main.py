import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import copy

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view() is used to reshape the tensor to match the shape of the input tensor
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Normalize the image
        return (img - self.mean) / self.std

# Function to open a file dialog and return selected file path
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename()  # Show file dialog
    return file_path

# Modify the image_loader function to take device as a parameter
def image_loader(image_path, device, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Function to load pre-trained VGG19 model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    """
    Constructs a style transfer model with content and style losses.

    Args:
        cnn (torch.nn.Module): The pre-trained CNN model.
        normalization_mean (torch.Tensor): The mean values for normalization.
        normalization_std (torch.Tensor): The standard deviation values for normalization.
        style_img (torch.Tensor): The style image.
        content_img (torch.Tensor): The content image.

    Returns:
        tuple: A tuple containing the style transfer model, style losses, and content losses.
    """
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # Just in case we want to use average pooling instead of max pooling
    for i, layer in enumerate(cnn):
        if isinstance(layer, torch.nn.MaxPool2d):
            cnn[i] = torch.nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

    # Collect content/style losses
    content_losses = []
    style_losses = []

    # Assuming cnn is a nn.Sequential
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # Replace in-place ReLU with out-of-place version
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # Add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Function to perform style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    
    # We will use L-BFGS algorithm to optimize. The .backward method dynamically computes gradients
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

# Main function
def main(content_path, style_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Desired size of output image
    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    # Load VGG19
    weights = VGG19_Weights.DEFAULT
    cnn = vgg19(weights=weights).features.to(device).eval()

    # VGG networks are trained on images with each channel normalized
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Load images
    content_img = image_loader(content_path, device)
    style_img = image_loader(style_path, device)

    # Perform style transfer
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

    # Display or save output
    plt.figure()
    imshow(output, title='Output Image')
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("Select the content (target) image")
    content_path = select_file()
    print("Select the style (transferer) image")
    style_path = select_file()

    main(content_path, style_path)
