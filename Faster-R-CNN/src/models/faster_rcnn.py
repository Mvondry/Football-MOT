import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_faster_rcnn_model(num_classes: int, pretrained: bool = True):
    """
    Create a Faster R-CNN ResNet-50 FPN model with a custom classification head.

    Args:
        num_classes: Number of output classes including the background class.
        pretrained: If True, initializes the model with COCO pretrained weights.

    Returns:
        Faster R-CNN model with modified classification head.
    """
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_faster_rcnn_checkpoint(model, checkpoint_path: str, device: torch.device):
    """
    Load Faster R-CNN model weights from a checkpoint.

    Supports both:
        - plain state_dict
        - checkpoint dictionary containing the key 'model_state_dict'

    Args:
        model: Faster R-CNN model.
        checkpoint_path: Path to checkpoint file.
        device: Device used for loading the checkpoint.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model