import copy
import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.datasets import load_files
from torchvision import transforms
from tqdm import tqdm
import matplotlib.ticker as mtick


def load_dataset(path):
    data = load_files(path, load_content=False, random_state=42)
    df = pd.merge(pd.DataFrame(data.target, columns=['target']), pd.DataFrame(data.target_names, columns=['label']),
                  how='left', left_on='target', right_index=True)
    df = pd.concat([df, pd.DataFrame(data.filenames, columns=['filename'])], axis=1, sort=False)
    return df


def load_dataset_cifar(cifar100):
    return pd.merge(
        pd.DataFrame(cifar100.targets, columns=['target']),
        pd.DataFrame(cifar100.classes, columns=['label']),
        how='left',
        left_on='target',
        right_index=True,
    )


def imshow(img):
    plt.imshow(img)
    plt.show()


def show_first_img(df):
    fig = plt.figure(figsize=[20, 20])
    for idx, filename in enumerate(df['filename'].values[:10]):
        img = mpimg.imread(filename)
        ax = fig.add_subplot(5, 5, idx + 1)
        plt.imshow(img)
        ax.set_title(df['label'].values[idx])

    plt.show()


def detect_human_face(file, show=True):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # load color (BGR) image
    img = cv2.imread(file)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # print number of faces detected in the image

    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not show:
        return cv_rgb, len(faces)
    # display the image, along with bounding box
    ax = plt.axes()
    ax.set_title(f'Number of faces detected: {len(faces)}')
    plt.imshow(cv_rgb)
    plt.show()


def detect_human_faces(file_list):
    fig = plt.figure(figsize=[20, 20])
    for idx, file in enumerate(file_list):
        cv_rgb, num_faces = detect_human_face(file, show=False)
        ax = fig.add_subplot(5, 5, idx + 1)
        plt.imshow(cv_rgb)
        ax.set_title(f'Num faces detected: {num_faces}')

    plt.show()


def face_detector(img_path):
    """returns "True" if face is detected in image stored at img_path"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def imshow_tensor(inp, std_nums, mean_nums, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std_nums * inp + mean_nums
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_data_loader(data_loader, std_nums, mean_nums, classes_list):
    # Get a batch of training data
    inputs, classes = next(iter(data_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow_tensor(out, std_nums, mean_nums, title=[classes_list[x] for x in classes])


def torch_transformations():
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transform = {'train': transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)

    ]), 'val': transforms.Compose([
        transforms.Resize(size=64),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)

    ]), 'test': transforms.Compose([
        transforms.Resize(size=64),
        transforms.CenterCrop(size=64),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ]),
    }
    return transform, std_nums, mean_nums


def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
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


def visualize_model(model, data_loaders, classes, std_nums, mean_nums, device, num_images=6):
    was_training = model.training
    model.eval()
    plt.figure()

    with torch.no_grad():
        images_so_far = 0
        for i, (inputs, labels) in enumerate(data_loaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, ground truth: {}'
                             .format(classes[preds[j]],
                                     classes[labels[j]]))
                imshow_tensor(inputs.cpu().data[j], std_nums, mean_nums)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def global_accuracy(model, data_loader, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
def accuracy_per_class(model, data_loader, classes):
    class_correct = [0. for i in range(len(classes))]
    class_total = [0. for i in range(len(classes))]
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(3):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] == 0.0:
            yield classes[i], 0
        else:
            yield classes[i], 100 * class_correct[i] / class_total[i]

def test_accuracy(model, data_loader, classes, figsize=(25, 8)):
    global_accuracy(model, data_loader, classes)
    
    res = accuracy_per_class(model, data_loader, classes)
    df = pd.DataFrame(res)

    x = np.arange(len(df))

    width = 0.4

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title('Accuracy per class')
    rects = ax.bar(x - width / 2, df[1], width)
    plt.xticks(x, df[0], rotation=90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = int(rect.get_height())
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height +1),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects)
    plt.show()
