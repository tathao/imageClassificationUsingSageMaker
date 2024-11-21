#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd

from torch.utils.data import DataLoader

import argparse
import os

from PIL import Image, ImageFile
from torchvision.datasets import ImageFolder

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Custom DatasetFolder to skip corrupted images
class SafeImageFolder (ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super(SafeImageFolder, self).__getitem__(index)
            except OSError as e:
                print(f"Skipping corrupted image at index {index}.")
                index = (index + 1) % len(self.samples)


# TODO: Import dependencies for Debugging andd Profiling
from smdebug.pytorch import Hook

def test(model, test_loader, criterion, hook, device):
    '''
    Complete this function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)  # Set hook to EVAL mode
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
            "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset)
            )
        )
    

def train(model, train_loader, criterion, optimizer, epoch, hook, device):
    '''
    Complete this function that can take a model and
    data loaders for training and will get train the model
    Remember to include any debugging/profiling hooks that you might need
    '''

    model.train()
    hook.set_mode(smd.modes.TRAIN)  # Set hook to TRAIN mode
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    epoch,
                    i * len(inputs),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
    return model
    

def net(num_classes, pretrained=True):
    '''
    Complete this function that initializes your model
    Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Dropout(p=0.5),
        nn.BatchNorm1d(num_classes)
    )

    return model


def create_data_loaders(batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset_directory = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    test_dataset_directory = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

    train_image_datasets = SafeImageFolder(root=train_dataset_directory, transform=train_data_transform)
    test_image_datasets = SafeImageFolder(root=test_dataset_directory, transform=test_data_transform)

    # Load train, validation, and test datasets
    train_data_loader = DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader

def main(args):

    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    '''
    Initialize a model by calling the net function
    '''
    model = net(args.num_classes).to(device)

    # Initialize SageMaker Debugger Hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(args.batch_size)
    
    '''
    Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    hook.register_loss(loss_criterion)
    
    
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # Start training the model
    print("Starting training...")
    for epoch in range(1, args.num_epochs+1):
        model = train(model, train_loader, loss_criterion, optimizer, epoch, hook ,device)

        '''
        Test the model to see its accuracy
        '''
        test(model, test_loader, loss_criterion, hook ,device)
    
    
    '''
    Save the trained model
    '''
    # Save the trained model
    print("Saving the trained model...")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "model.pt")
    model = model.to(device)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders")
    parser.add_argument("--num_classes", type=int, default=133, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    
    args=parser.parse_args()
    
    main(args)