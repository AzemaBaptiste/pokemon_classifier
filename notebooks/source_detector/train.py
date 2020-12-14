import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.utils.data import RandomSampler
from torchvision import models


def _get_train_data_loader(data_dir, batch_size, classes):
    print("Get train data loader.")

    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transforms = {'train': T.Compose([
        T.RandomResizedCrop(size=64),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)

    ]), 'val': T.Compose([
        T.Resize(size=64),
        T.CenterCrop(size=64),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)

    ]), 'test': T.Compose([
        T.Resize(size=64),
        T.CenterCrop(size=64),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]),
    }

    # Load Pokemon dataset
    pokemon_dataset = torchvision.datasets.ImageFolder('%s/Pokemon' % data_dir, transform=transforms['train'])

    # Set class for each Pokemon to 0
    pokemon_dataset.target_transform = lambda x: classes.index('Pokemon')

    # Split train / test
    pokemon_test_size = round((len(pokemon_dataset)) * .15)
    pokemon_train_size = len(pokemon_dataset) - pokemon_test_size

    pokemon_dataset_train, pokemon_dataset_test = torch.utils.data.random_split(pokemon_dataset,
                                                                                [pokemon_train_size, pokemon_test_size])

    # Load CIFAR 100 dataset
    cifar100_train = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                   transform=transforms['train'])
    cifar100_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                  transform=transforms['test'])

    # Set class for each image to 1
    cifar100_train.target_transform = lambda x: classes.index('Other')
    cifar100_test.target_transform = lambda x: classes.index('Other')

    RandomSampler(cifar100_train, replacement=True, num_samples=5000)
    RandomSampler(cifar100_train, replacement=True, num_samples=1000)

    # Concat Pokemon and cifar100 datasets
    pokemon_detector_train = torch.utils.data.ConcatDataset([pokemon_dataset_train, cifar100_train])
    pokemon_detector_test = torch.utils.data.ConcatDataset([pokemon_dataset_test, cifar100_test])

    pokemon_detector_loader_train = torch.utils.data.DataLoader(pokemon_detector_train,
                                                                batch_size, shuffle=True, drop_last=True)
    pokemon_detector_loader_test = torch.utils.data.DataLoader(pokemon_detector_test,
                                                               batch_size, shuffle=True)

    data_loaders = {
        'train': pokemon_detector_loader_train,
        'val': pokemon_detector_loader_test
    }
    dataset_sizes = {
        'train': len(pokemon_detector_loader_train),
        'val': len(pokemon_detector_loader_test)
    }

    return data_loaders, dataset_sizes


def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def compute_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(3):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        if class_total[i] == 0.0:
            print(pokemon_detector_classes[i])
        else:
            print('Accuracy of %5s : %2d %%' % (
                pokemon_detector_classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    pokemon_detector_classes = ('Pokemon', 'Other')

    # Load the training data.
    data_loaders, dataset_sizes = _get_train_data_loader(args.data_dir, args.batch_size, pokemon_detector_classes)

    model_conv = models.resnet50(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # Trains the model (given line of code, which calls the above training function)
    model_conv = train_model(model_conv, data_loaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler,
                             args.epochs, device)

    # Keep the keys of this dictionary as they are
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model_detector.pth')
    with open(model_path, 'wb') as f:
        torch.save(model_conv.cpu(), f)
