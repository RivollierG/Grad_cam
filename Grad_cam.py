import streamlit as st

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18, resnet50, swin_v2_b, vgg16, densenet161,  mnasnet1_0

import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

import json

with open("./imagenet_class_index.json", 'r') as j:
    listcat = json.load(j)

inv_list = {v[1]: int(k) for k, v in listcat.items()}

choice = st.selectbox(
    "Choose a model", ["resnet18", "resnet50", "swin_v2_b", "vgg16", "densenet161",  "mnasnet1_0"])

if choice == "resnet50":
    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
if choice == "resnet18":
    model = resnet18(pretrained=True)
    target_layers = [model.layer4[-1]]
elif choice == 'swin_v2_b':
    model = swin_v2_b(pretrained=True)
    # st.text(swin_v2_b.__wrapped__())
    target_layers = [model.features[-1]]
elif choice == "vgg16":
    model = vgg16(pretrained=True)
    target_layers = [model.features[-1]]
elif choice == "densenet161":
    model = densenet161(pretrained=True)
    target_layers = [model.features[-1]]
elif choice == "mnasnet1_0":
    model = mnasnet1_0(pretrained=True)
    target_layers = [model.layers[-1]]


uploaded_file = st.file_uploader("Choose a photo (jpg)", type='jpg')

if uploaded_file is not None:

    img = np.array(Image.open(uploaded_file))
    img = cv2.resize(img, (640, 640))
    img = np.float32(img) / 255
    rgb_img = img.copy()
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0)
    cam = GradCAM(model=model, target_layers=target_layers)

    cat_pred = listcat[str(int(model(input_tensor).max(1).indices[0]))][1]

    st.text(
        f" Seems it's a wonderul {cat_pred}")

    cla = st.selectbox(
        'Choose a category to see what information was used to make this prediction', inv_list.keys(), inv_list[cat_pred])

    clas_output = inv_list[cla]
    targets = [ClassifierOutputTarget(clas_output)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    st.image(Image.fromarray(visualization))
