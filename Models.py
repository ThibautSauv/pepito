import torchvision.models as models
import torch.nn as nn
import torch


class PepitoModel(nn.Module):
    """
    Custom model for the Pepito dataset. The input is a
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


class LightPepitoModel(nn.Module):
    """
    Custom model for the Pepito dataset. Trying to keep the model as light as possible by reducing the number of parameters in FC layers using PCA.
    """

    def __init__(self, num_classes):
        super(LightPepitoModel, self).__init__()

        self.model = models.vgg11(pretrained=True)

        # Feature extraction
        self.features = self.model.features

        # Feature reduction
        self.avgpool = self.model.avgpool

        # Classifier
        in_features = (
            49 * 32
        )  # 49 is the output size of the avgpool layer and 32 is the output size of the PCA
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # torch.Size([B, 512, 7, 7])
        x = self.avgpool(x)  # torch.Size([B, 512, 7, 7])

        # Reduce the output size with PCA
        x = self.apply_pca(x)  # torch.Size([B, 32, 49])

        x = torch.flatten(x, 1)
        x = self.classifier(x)  # torch.Size([B, 2])
        return x

    def apply_pca(self, x):
        # Reshape the tensor by combining H and W and transposing the matrix to have the highest dimension last
        x = x.view(x.size(0), -1, x.size(1))  # torch.Size([B, 49, 512])

        # Apply PCA to a tensor
        q = min(x.size(2), 32)  # Normally q = 32
        u, s, v = torch.pca_lowrank(x, q=q)  # u: [B, 49, q], s: [B, q], v: [B, q, 512]
        x = torch.matmul(x, v)  # torch.Size([B, 49, 32])

        # Swap the dimensions back to the original shape
        x = x.view(x.size(0), -1, x.size(1))
        return x
