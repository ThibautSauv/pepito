import torchvision.models as models
import torch.nn as nn
import torch


class PepitoModel(nn.Module):
    """
    Custom model for the Pepito dataset. The model is a simple classifier with a single hidden layer. The input is a
    224x224x3 image and the output is a single class label."
    """

    def __init__(self, num_classes):
        super(PepitoModel, self).__init__()

        self.model = models.vgg11(pretrained=True)

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x
