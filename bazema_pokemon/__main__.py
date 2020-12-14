"""Entry point"""

from datetime import datetime
import sys
from pathlib import Path

import matplotlib.image as mpimg
import torch
from PIL import Image
from torch.autograd import Variable

from bazema_pokemon import utils, conf


class BazemaPokemon:
    """Main class"""

    def __init__(self, image_path):
        self.image_path = image_path
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform, _, _ = utils.torch_transformations()

        self.pokemon_identificator()

    def image_loader(self):
        """load image, returns cuda tensor"""
        image = Image.open(self.image_path)
        image = self.transform['test'](image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image  # assumes that you're using GPU

    def predict(self, model_path, classes):
        model_detector = torch.load(model_path)
        image = self.image_loader()
        output = model_detector(image)
        index = output.data.cpu().numpy().argmax()
        return classes[index]

    def human_detector(self):
        return utils.face_detector(self.image_path)

    def detect_pokemon(self):
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'pokemon_detector.pth'
        return self.predict(file, conf.detector_classes)

    def pokemon_classifier(self):
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'pokemon_detector.pth'
        return self.predict(file, conf.pokemons)

    def pokemon_identificator(self):
        start = datetime.now()
        res_detector = self.detect_pokemon()
        res_classifier = self.pokemon_classifier()

        if utils.face_detector(self.image_path):
            print(f'It\'s a Human, it looks like the Pokemon {res_classifier} !')
        elif res_detector == 'Pokemon':
            print(f'It\'s a Pokemon: {res_classifier} !')
        else:
            print(f'It doesn\'t look like a Pokemon nor a Human, but it looks like {res_classifier} !')

        print(f'Took {datetime.now() - start} to predict')


def main():
    args = utils.parse_args(args=sys.argv[1:])
    image_path = args.image_path
    BazemaPokemon(image_path)


if __name__ == '__main__':
    main()
