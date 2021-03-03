"""Entry point"""
import copy
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models

from bazema_pokemon import utils, conf


class BazemaPokemon:
    """Main class"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.device = torch.device("cpu")
        self.transform, _, _ = utils.torch_transformations()

        self.run()

    def image_loader(self):
        """load image, returns cuda tensor"""
        image = Image.open(self.image_path)
        image = self.transform['test'](image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image

    def human_detector(self):
        """Use OpenCV to detect faces"""
        return utils.face_detector(self.image_path)

    def detect_pokemon(self):
        """Use pytorch to detect pokemon"""
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'pokemon_detector.pth'
        return predict(self.image_loader(), file, conf.detector_classes, self.device)

    def pokemon_classifier(self):
        """Use pytorch to identify pokemon"""
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'classifier_05_epoch_dict.pth'
        return predict(self.image_loader(), file, conf.pokemons, self.device, True)

    def run(self):
        """Main process"""
        start = datetime.now()
        res_detector = self.detect_pokemon()
        res_classifier = self.pokemon_classifier()

        if utils.face_detector(self.image_path):
            print(f'I guess it\'s a Human, it looks like '
                  f'the Pokemon {res_classifier} !')
        elif res_detector == 'Pokemon':
            print(f'I guess it\'s a Pokemon: {res_classifier} !')
        else:
            print(f'I don \' know what it is but it '
                  f'looks like {res_classifier} !')

        print(f'Took {datetime.now() - start} to predict')


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    # training step
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)

    # validation step
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    # validation epoch end
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Net(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.net = models.resnet50(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 150)

    def forward(self, xb):
        return self.net(xb)


# Due to https://discuss.pytorch.org/t/error-loading-saved-model/8371/6
# we need the model and the predict function at the same level
def predict(image, model_path, classes, device, load_state_dict=False):
    """Load model and run prediction"""
    if load_state_dict:
        net = Net()
        net.load_state_dict(torch.load(model_path))
    # net.load_state_dict(torch.load(model_path))
    else:
        net = torch.load(model_path, map_location=torch.device('cpu'))
    net = net.to(device)
    # image = self.image_loader()
    image = image.to(device)
    output = net(image)
    index = output.data.cpu().numpy().argmax()
    return classes[index]


def main():
    """entrypoint"""
    args = utils.parse_args(args=sys.argv[1:])
    image_path = args.image_path
    BazemaPokemon(image_path)


if __name__ == '__main__':
    main()
