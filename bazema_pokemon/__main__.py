"""Entry point"""

import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable

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

    def predict(self, model_path, classes):
        """Load model and run prediction"""
        model_conv = torch.load(model_path)
        model_conv = model_conv.to(self.device)
        image = self.image_loader()
        image = image.to(self.device)
        output = model_conv(image)
        index = output.data.cpu().numpy().argmax()
        return classes[index]

    def human_detector(self):
        """Use OpenCV to detect faces"""
        return utils.face_detector(self.image_path)

    def detect_pokemon(self):
        """Use pytorch to detect pokemon"""
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'pokemon_detector.pth'
        return self.predict(file, conf.detector_classes)

    def pokemon_classifier(self):
        """Use pytorch to identify pokemon"""
        current_dir = Path(__file__).parent
        file = current_dir / 'resources' / 'pokemon_detector.pth'
        return self.predict(file, conf.pokemons)

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


def main():
    """entrypoint"""
    args = utils.parse_args(args=sys.argv[1:])
    image_path = args.image_path
    BazemaPokemon(image_path)


if __name__ == '__main__':
    main()
