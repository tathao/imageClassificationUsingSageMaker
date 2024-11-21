import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Custom DatasetFolder to skip corrupted images
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super(SafeImageFolder, self).__getitem__(index)
            except OSError as e:
                print(f"Skipping corrupted image at index {index}.")
                index = (index + 1) % len(self.samples)

def test(model, test_loader, criterion ,device):
    '''
    This function takes a model and a 
    testing data loader and will get the test accuracy/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0 
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            total += labels.size(0)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, device, epoch):
    '''
    This function trains the model using training data.
    '''
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
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
    Initialize your model
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
    valid_dataset_directory = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    test_dataset_directory = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

    train_image_datasets = SafeImageFolder(root=train_dataset_directory, transform=train_data_transform)
    test_image_datasets = SafeImageFolder(root=test_dataset_directory, transform=test_data_transform)
    valid_image_datasets = SafeImageFolder(root=valid_dataset_directory, transform=test_data_transform)

    train_data_loader = DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(valid_image_datasets, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader, valid_data_loader


def main(args):
    # Check if CUDA (GPU) is available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = args.num_classes
    
    # Initialize the model
    print("Initializing the model...")
    model = net(num_classes).to(device)

    print("Creating data loaders...")
    train_loader, validation_loader, test_loader = create_data_loaders(args.batch_size)
    
    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    # Start training the model
    print("Starting training...")
    for epoch in range(1, args.num_epochs+1):
        model = train(model, train_loader, loss_criterion, optimizer, device, epoch)
        test(model, validation_loader, loss_criterion, device)
    
    # Test the model
    print("Testing the model...")
    test(model, test_loader, loss_criterion, device)
    
    # Save the trained model
    print("Saving the trained model...")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "model.pt")
    model = model.to(device)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(model_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loaders")
    parser.add_argument("--num_classes", type=int, default=133, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")



    args = parser.parse_args()
    
    main(args)
