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
            self.device = torch.device("cuda")
            print("Evaluating on GPU")
        else:
            self.device = torch.device("cpu")
            print("Evaluating on CPU")
            # Load the trained parameters of the network

        # Instantiate the model
        self.model = net.Net().to(self.device)
        self.model.eval()

        # Instantiate the Softmax class to compute probabilities
        self.softmax = torch.nn.Softmax(dim=1)

        # Load the trained parameters of the network
        path = os.path.join('cnn', 'param_smaller_modified120_0.1.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        # Load the categories mappings
        self.categories = self.load_categories()

    def load_categories(self):
        """
        Loads the categories from the categories file
        :return: the dict corresponding to index -> class name mappings
        """
        path = os.path.join('cnn', 'mappings120.json')
        with open(path, 'r') as f:
            categories = json.load(f)
        return categories

    def classify(self, image):
        """
        Classify the input image into one of the categories using the network
        :param image: the input image to classify
        :return: the first 3 categories predicted by the CNN, the probability (confidence)
                 that it is that category
        """
        image.to(self.device)

        out = self.model(image)
        # First axis is for the batch size, see application.py -> function classify in Classify
        probas = self.softmax(out)[0].detach()
        print(probas)
        proba, idx_prediction = torch.topk(probas, 3)
        proba = proba.numpy()
        idx_prediction = idx_prediction.numpy()
        categories = [self.categories.get(str(idx)) for idx in idx_prediction]
        print(idx_prediction)
        print(categories)
        return categories, proba
