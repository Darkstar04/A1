import cv2
import PIL
import math
import numpy
import torch
import random
import argparse
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
arguments = parser.parse_args()

image = PIL.Image.open(arguments.input)
new_width = math.ceil(image.size[0] * (512 / image.size[1]))
new_image = image.resize((new_width, 512))

def Process():

    phase = 'X1', 'X2', 'X3', 'X4'

    for phase in phase:

        if phase == 'X1' or phase == 'X2' or phase == 'X4':

            if phase == 'X1': checkpoints = 'checkpoints/cm.lib'
            if phase == 'X2': checkpoints = 'checkpoints/mm.lib'
            if phase == 'X4': checkpoints = 'checkpoints/mn.lib'

            if phase == 'X1': data = torch.utils.data.DataLoader(Dataset(new_image))
            if phase == 'X2': data = torch.utils.data.DataLoader(Dataset(X1))
            if phase == 'X4': data = torch.utils.data.DataLoader(Dataset(X3))

            for data in data:
                generated = Model().inference(data['tensor'], checkpoints)
                im = TensorToImage(generated[0])

            if phase == 'X1':
                X1 = im
                X1.save(os.path.join(arguments.output, 'X1.jpg'))

            if phase == 'X2':
                X2 = im
                X2.save(os.path.join(arguments.output, 'X2.jpg'))

            if phase == 'X4':
                X4 = im.resize((image.size[0], image.size[1]))
                X4.save(os.path.join(arguments.output, 'X4.jpg'))

        if phase == 'X3':
            X3 = PIL.Image.fromarray(X_3(X1, X2))
            X3.save(os.path.join(arguments.output, 'X3.jpg'))

class Dataset:

    def __init__(self, image): self.A = image

    def __getitem__(self, index):
        tensor = {'tensor': torchvision.transforms.ToTensor()(self.A)}
        return tensor

    def __len__(self): return 1

class Model:

    def inference(self, tensor, checkpoints):
        self.Generator = Generator()
        self.Generator.load_state_dict(torch.load(checkpoints))
        with torch.no_grad(): return self.Generator.forward(tensor)

def TensorToImage(tensor):
    tensor = (tensor + 1) / 2
    new_tensor = torchvision.transforms.functional.convert_image_dtype(tensor, torch.uint8)
    pillow_image = torchvision.transforms.ToPILImage()(new_tensor)
    return torchvision.transforms.functional.crop(pillow_image, 0, 0, 512, new_width)

class Generator(torch.nn.Module):

    def __init__(self, norm_layer=torch.nn.InstanceNorm2d, activation=torch.nn.ReLU()):
        super(Generator, self).__init__()
        model = [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(3, 64, kernel_size=7), norm_layer(1), activation]
        for i in range(4): model += [torch.nn.Conv2d(64 * (2**i), 128 * (2**i), kernel_size=3, stride=2, padding=1), norm_layer(1), activation]
        for i in range(9): model += [ResnetBlock(1024, norm_layer, activation)]
        for i in range(4): model += [torch.nn.ConvTranspose2d(64 * (2**(4 - i)), 32 * (2**(4 - i)), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(1), activation]
        model += [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(64, 3, kernel_size=7), torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)

    def forward(self, tensor): return self.model(tensor)

class ResnetBlock(torch.nn.Module):

    def __init__(self, dimension, norm_layer, activation=torch.nn.ReLU()):
        super(ResnetBlock, self).__init__()
        conv_block = [torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, kernel_size=3), norm_layer(1), activation]
        conv_block += [torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(1024, 1024, kernel_size=3), norm_layer(1)]
        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x): return x + self.conv_block(x)

def X_3(X1, X2):
    details = numpy.array(X1)
    X1 = numpy.array(X1)
    X2 = numpy.array(X2)
    for obj in Annotations(X2):
        x = math.ceil(obj.x)
        y = math.ceil(obj.y)
        h = math.ceil(obj.h / 2)
        w = math.ceil(obj.w / 2)
        f1 = numpy.array([0, 0, 0])
        f2 = numpy.array([0, 0, 0])
        mask = cv2.inRange(X2, f1, f2)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(details, contours, -1, 205, cv2.FILLED)
        if obj.name == 'aur': cv2.ellipse(details, (x, y), (h, w), 0, 0, 360, (255, 0, 0), -1)
        if obj.name == 'nip': cv2.ellipse(details, (x, y), (h, w), 0, 0, 360, (255, 255, 255), -1)
        if obj.name == 'bel': cv2.ellipse(details, (x, y), (h, w), 0, 0, 360, (255, 0, 255), -1)
        if obj.name == 'vag': cv2.ellipse(details, (x, y), (h, w), 0, 0, 360, (0, 0, 255), -1)
    f1 = numpy.array([0, 255, 0])
    f2 = numpy.array([0, 255, 0])
    mask = cv2.inRange(X1, f1, f2)
    mask_inv = numpy.invert(cv2.inRange(X1, f1, f2))
    X_3 = cv2.bitwise_and(X1, X1, mask=mask_inv) + cv2.bitwise_and(details, details, mask=mask)
    return X_3

class BodyPart:

    def __init__(self, name, x, y, h, w):
        self.name = name
        self.x = x
        self.y = y
        self.h = h
        self.w = w

def Annotations(X2):
    aur = Part(X2, 'aur')
    vag = Part(X2, 'vag')
    bel = Part(X2, 'bel')
    nip = InferNip(aur)
    return aur + nip + vag + bel

def Part(X2, part):
    X2 = numpy.array(X2)
    bodypart = []
    if part == 'aur':
        f1 = numpy.array([255, 0, 0])
        f2 = numpy.array([255, 0, 0])
        mask = cv2.inRange(X2, f1, f2)
    if part == 'vag':
        f1 = numpy.array([0, 0, 255])
        f2 = numpy.array([0, 0, 255])
        mask = cv2.inRange(X2, f1, f2)
    if part == 'bel':
        f1 = numpy.array([255, 0, 255])
        f2 = numpy.array([255, 0, 255])
        mask = cv2.inRange(X2, f1, f2)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt)>5:
            ellipse = cv2.fitEllipse(cnt)
            x = ellipse[0][0]
            y = ellipse[0][1]
            h = ellipse[1][0]
            w = ellipse[1][1]
            bodypart.append(BodyPart(part, x, y, h, w))
    return bodypart

def InferNip(aur):
    nip = []
    for aur in aur:
        nip_dim = int(5 + aur.w*random.uniform(0.1, 0.1))
        nip.append(BodyPart('nip', aur.x, aur.y, nip_dim, nip_dim))
    return nip

if __name__ == '__main__': Process()