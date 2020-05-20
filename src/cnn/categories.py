import json
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# ================================================================================
# Script to save the dict of the mapping index -> class name of the dataset
# ================================================================================

# =======================================================================================
# Arguments parser
parser = argparse.ArgumentParser()

# number of epochs
parser.add_argument('-s', '--save', help='The file where to save the mapping')
# =======================================================================================


def main():
    args = parser.parse_args()

    if args.save is not None:
        print('Saving mappings in {}'.format(args.save))
        file = args.save
    else:
        file = 'categories.json'

    transform = transforms.ToTensor()
    dataset = ImageFolder('./data/fruits-360/Training', transform=transform)

    # Revert the mappings, to get (index -> class name) mappings
    mappings = {value: key for key, value in dataset.class_to_idx.items()}
    with open(file, 'w') as f:
        json.dump(mappings, f)

    print(mappings)
    print('Categories saved under {} !'.format(file))


if __name__ == '__main__':
    main()
