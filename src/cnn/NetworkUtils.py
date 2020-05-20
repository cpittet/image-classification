import torch
import json
import os

from . import net


# ================================================================================
# Utils to load parameters and instantiate the network, classify inputs, etc
# ================================================================================

class NetworkUtils:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Evaluating on GPU")
        else:
            device = torch.device("cpu")
            print("Evaluating on CPU")

        # Instantiate the model
        self.model = net.Net().to(device)
        self.model.eval()

        # Instantiate the Softmax class to compute probabilities
        self.softmax = torch.nn.Softmax(dim=1)

        # Load the trained parameters of the network
        #self.model.load_state_dict(torch.load('parameters.pt'))

        # Load the categories mappings
        self.categories = self.load_categories()

    def load_categories(self):
        """
        Loads the categories from the categories file
        :return: the dict corresponding to index -> class name mappings
        """
        path = os.path.join('cnn', 'categories.json')
        with open(path, 'r') as f:
            categories = json.load(f)
        return categories

    def classify(self, image):
        """
        Classify the input image into one of the categories using the network
        :param image: the input image to classify
        :return: the category predicted by the CNN, the probability (confidence)
                 that it is that category
        """
        out = self.model(image)
        # First axis is for the batch size, see application.py -> function classify in Classify
        probas = self.softmax(out)[0]
        # .item() because it returns a tensor
        idx_prediction = torch.argmax(probas).item()
        proba = probas[idx_prediction].item()
        category = self.categories.get(str(idx_prediction))
        return category, proba
