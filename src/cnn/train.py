import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import net

# =======================================================================================
# Arguments parser
parser = argparse.ArgumentParser()

# number of epochs
parser.add_argument('-e', '--epochs', help='The number of epochs to use for training', type=int)

# momentum
parser.add_argument('-m', '--momentum', help='The momentum coefficient to use for training', type=float)

# original learning rate
parser.add_argument('-lr', '--learningrate', help='The original learning rate to use for training', type=float)

# net summary
parser.add_argument('-s', '--summary', help='Display the summary of the network', action='store_true')

# save dict
parser.add_argument('-sp', '--save', help='Save the dictionary containing tuned parameters under the specified file')

# load dict
parser.add_argument('-ld', '--load', help='Load the dictionary containing tuned parameters from the specified file')
# =======================================================================================


def accuracy(predicted, target):
    """
    Compute the accuracy of the predicted values w.r.t. target
    :param predicted: the predicted values for each categories
    :param target: the ground truth
    :return: the accuracy
    """
    # Takes the predicted class (index of max value) for each sample
    preds = torch.argmax(predicted, dim=1)
    return torch.sum((preds == target)).item()


def get_loaders():
    """
    Instantiate the ImageNet dataset and a DataLoader for it,
    for both training and validation
    :return: dictionary containing : dataloader training, dataloader validation
    """
    batch_size = 128
    transform = transforms.ToTensor()
    dataset_tr = ImageFolder('./data/fruits-360/Training', transform=transform)
    sampler_tr = RandomSampler(dataset_tr)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=False, num_workers=4,
                               pin_memory=True, sampler=sampler_tr)
    dataset_val = ImageFolder('./data/fruits-360/Test', transform=transform)
    sampler_val = RandomSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4,
                                pin_memory=True, sampler=sampler_val)
    return {'train': dataloader_tr, 'validation': dataloader_val}


def validate(model, dataloaders, device, criterion):
    """
    Validate on validation batches. Compute the weighted average accuracy over the different batches
    :param model: the model we use
    :param dataloaders: dictionary containing the dataloaders for training and validation sets
    :param device: the device in use
    :param criterion: the loss function
    :return: the accuracy and average loss of the model on the validation set
    """
    # Switch to evaluation mode
    model.eval()
    # Stop autograd to track history, as we just want to evaluate the model for some images
    with torch.no_grad():
        acc = 0
        loss = 0
        print('Computing accuracy on validation set...')
        for i, (images, targets) in enumerate(dataloaders['validation']):
            # Load the batch to the device
            images, targets = images.to(device), targets.to(device)

            out = model(images)
            tmp = accuracy(out, targets)
            acc += tmp
            loss += criterion(out, targets).item()

        # Average over the number of batches we used
        val_dataset_size = len(dataloaders['validation'].dataset)
        return acc / val_dataset_size, loss / len(dataloaders['validation'])


def train_model(model, optimizer, lr_scheduler, criterion, epochs, dataloaders, device, save, writer):
    """
    Train the model
    :param model: the model to train
    :param optimizer: the optimizer to use
    :param lr_scheduler: the learning rate scheduler to use
    :param criterion: the loss function
    :param epochs: the maximum number of epochs for training
    :param dataloaders: dictionary containing the dataloaders for training and validation sets
    :param device: the device to use
    :param save: if not None, the file to save the parameters to
    :param writer: the writer for tensorboard
    :return:
    """
    train_dataset_size = len(dataloaders['train'].dataset)
    train_dataloader_size = len(dataloaders['train'])
    for ep in range(epochs):
        # Switch to train mode
        model.train()

        running_loss = 0
        running_corrects = 0

        for i, (images, targets) in enumerate(dataloaders['train']):
            # Load the batch to the device
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            out = model(images)

            # Compute loss
            loss = criterion(out, targets)

            # Backward pass
            loss.backward()

            # Update weights of the model, with the optimizer
            optimizer.step()
            print('Batch {}/{}, Loss : {:.4f}'.format(i+1, train_dataloader_size, loss.item()))

            # Add the losses and sum of the corrects predictions
            running_loss += loss.item()
            running_corrects += accuracy(out, targets)

            # Clear out the accumulated gradients
            model.zero_grad()

        # Update learning rate if necessary
        lr_scheduler.step()

        # Validation
        acc, loss = validate(model, dataloaders, device, criterion)
        acc = acc*100
        print('Epoch {}/{}, Accuracy : {:.4f} %'.format(ep+1, epochs, acc))

        # Write loss and accuracy for validation at this epoch
        writer.add_scalars('Loss', {'Validation': loss,
                                    'Training': running_loss / train_dataloader_size}, ep+1)

        writer.add_scalars('Accuracy', {'Validation': acc,
                                        'Training': running_corrects / train_dataset_size * 100}, ep+1)

        writer.flush()

    if save is not None:
        print('Saving parameters under ./{} ...', save)
        torch.save(model.state_dict(), save)
        print('Parameters saved.')


def main():
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else 50
    learning_rate = args.learningrate if args.learningrate is not None else 1e-2
    momentum = args.momentum if args.momentum is not None else 0.9

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    # Writer to tensorboard
    writer = SummaryWriter('runs/dropout_h')

    # Print hyper-parameters
    print('Epochs : {}, learning rate : {:.6f}, momentum : {:.2f}, saving parameters : {},'
          ' loading parameters : {}'.format(epochs, learning_rate, momentum, args.save, args.load))

    # Instantiate the model
    model = net.Net().to(device)

    if args.load is not None:
        print('Loading parameters from {}'.format(args.load))
        model.load_state_dict(torch.load(args.load))

    if args.save is not None:
        print('Parameters will be saved in ./{}'.format(args.save))

    if args.summary:
        # Print the model summary
        summary(model, (3, 100, 100))

    # Define loss function. It combines LogSoftmax and NLL (negative log likelihood) loss
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Add a learning rate scheduler to get a decay in the learning rate
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 8], gamma=0.1)

    # Get the different data loader
    dataloaders = get_loaders()

    # Train the model
    train_model(model, optimizer, lr_scheduler, criterion, epochs, dataloaders, device, args.save, writer)

    writer.close()


if __name__ == '__main__':
    main()
