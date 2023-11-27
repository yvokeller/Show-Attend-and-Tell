import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19
from torchvision.models import VGG19_Weights

class Encoder(nn.Module):
    """
    Encoder network for image feature extraction, follows section 3.1.1 of the paper
    """
    def __init__(self, network='vgg19'):
        super(Encoder, self).__init__()
        self.network = network
        # Selection of pre-trained CNNs for feature extraction
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            # Removing the final fully connected layers of ResNet152
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048  # Dimension of feature vectors for ResNet152
        elif network == 'densenet161':
            self.net = densenet161(pretrained=True)
            # Removing the final layers of DenseNet161
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920  # Dimension of feature vectors for DenseNet161
        else:
            self.net = vgg19(weights=VGG19_Weights.DEFAULT)
            # Using features from VGG19, excluding the last pooling layer
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512  # Dimension of feature vectors for VGG19

    def forward(self, x):
        x = self.net(x)
        # These steps correspond to the extraction of annotation vectors (a = {a1,...,aL}) as described in Section 3.1.1 of the paper.
        # 1. Change the order from (BS, C, H, W) to (BS, H, W, C) in prep for reshaping
        x = x.permute(0, 2, 3, 1)
        # 2. Reshape to [BS, num_spatial_features, C], the -1 effectively flattens the height and width dimensions into a single dimension
        x = x.view(x.size(0), -1, x.size(-1))
        return x
