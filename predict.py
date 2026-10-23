"""
Loads a trained Pokemon classifier and predicts the species of a single image.

Usage:
    python predict.py --image path/to/image.jpg
"""
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_model(weights_path, class_names_path, device):
    with open(class_names_path) as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def preprocess(image_path):
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return tfms(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image to classify")
    parser.add_argument("--weights", default="pokemon_classifier.pt")
    parser.add_argument("--class_names", default="class_names.json")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.weights, args.class_names, device)

    input_tensor = preprocess(args.image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, args.top_k)

    print(f"\nPredictions for {args.image}:")
    for prob, idx in zip(top_probs, top_idxs):
        print(f"  {class_names[idx.item()]:<20} {prob.item() * 100:.1f}%")


if __name__ == "__main__":
    main()
