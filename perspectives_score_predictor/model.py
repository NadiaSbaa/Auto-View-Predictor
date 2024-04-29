import torch
import torch.nn as nn
from torchvision import transforms, models


class ResnetsRegressionModel(nn.Module):
    """
    A regression model based on ResNets architecture.

    Args:
        backbone (str): The ResNet backbone type, e.g., "resnet18", "resnet34", or "resnet50".
        pretrained (bool): Whether to load pretrained weights. Default is True.
    """
    def __init__(self, backbone, pretrained=True):
        super(ResnetsRegressionModel, self).__init__()
        # Load pre-trained resnets model
        self.backbone = backbone
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)

        # Freeze the parameters so they are not updated during training
        for param in self.model.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer for regression
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class EfficientnetsRegressionModel(nn.Module):
    """
    A regression model based on EfficientNets architecture.

    Args:
        backbone (str): The EfficientNet backbone type, e.g., "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", or "efficientnet_b3".
        pretrained (bool): Whether to load pretrained weights. Default is True.
    """
    def __init__(self, backbone, pretrained=True):
        super(EfficientnetsRegressionModel, self).__init__()
        # Load pre-trained model
        self.backbone = backbone
        if backbone == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
        elif backbone == "efficientnet_b1":
            self.model = models.efficientnet_b1(pretrained=pretrained)
        elif backbone == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif backbone == "efficientnet_b3":
            self.model = models.efficientnet_b3(pretrained=pretrained)

        # Freeze the parameters so they are not updated during training
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the last classifier for regression
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class InceptionRegressionModel(nn.Module):
    """
    A regression model based on InceptionV3 architecture.

    Args:
        pretrained (bool): Whether to load pretrained weights. Default is True.
    """
    def __init__(self, pretrained=True):
        super(InceptionRegressionModel, self).__init__()
        # Load pre-trained model
        self.backbone = "inception_v3"
        self.model = models.googlenet(pretrained=pretrained)
        # Freeze the parameters so they are not updated during training
        for param in self.model.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer for regression
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class VisionTransformersRegressionModel(nn.Module):
    """
    A regression model based on Vision Transformers architecture.

    Args:
        backbone (str): The Vision Transformer backbone type, e.g., "vit_b_16" or "vit_h_14".
        pretrained (bool): Whether to load pretrained weights. Default is True.
    """
    def __init__(self, backbone, pretrained=True):
        super(VisionTransformersRegressionModel, self).__init__()
        # Load pre-trained model
        self.backbone = backbone
        if backbone == "vit_b_16":
            self.model = models.vit_b_16(pretrained=pretrained)

        elif backbone == "vit_h_14":
            self.model = models.vit_h_14(pretrained=pretrained)

        # Freeze the parameters so they are not updated during training
        for param in self.model.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer for regression
        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Output 2 values for regression
            nn.Sigmoid()  # Sigmoid activation to ensure values between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


def get_model(backbone):
    """
    Get the regression model based on the specified backbone.

    Args:
        backbone (str): The backbone type of the model.

    Returns:
        nn.Module: The regression model.
    """
    if backbone in ["resnet18", "resnet34", "resnet50"]:
        return ResnetsRegressionModel(backbone)

    elif backbone in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]:
        return EfficientnetsRegressionModel(backbone)

    elif backbone == "inception_v3":
        return InceptionRegressionModel()

    elif backbone in ["vit_b_16", "vit_h_14"]:
        return VisionTransformersRegressionModel(backbone)

    else:
        return



