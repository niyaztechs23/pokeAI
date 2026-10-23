"""
Pokemon AI Classifier - ResNet18 with Checkpoints
"""

import argparse
import json
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_transforms():

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return train_tfms, val_tfms


def build_model(num_classes):

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def run_epoch(model, loader, criterion, optimizer, device, training):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="pokemon_classifier.pt")

    args = parser.parse_args()


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)


    train_tfms, val_tfms = build_transforms()


    train_ds = datasets.ImageFolder(
        args.data_dir + "/train",
        transform=train_tfms
    )

    val_ds = datasets.ImageFolder(
        args.data_dir + "/val",
        transform=val_tfms
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )


    classes = train_ds.classes


    print(
        f"Found {len(classes)} classes, "
        f"{len(train_ds)} train images, "
        f"{len(val_ds)} val images"
    )


    model = build_model(
        len(classes)
    ).to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=args.lr
    )


    checkpoint_file = "checkpoint.pth"

    start_epoch = 1
    best_accuracy = 0


    # Resume checkpoint
    if os.path.exists(checkpoint_file):

        print("Checkpoint found!")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]

        print(
            f"Resuming from epoch {start_epoch}"
        )


    for epoch in range(start_epoch, args.epochs + 1):

        start = time.time()


        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True
        )


        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False
        )


        print(
            f"Epoch {epoch}/{args.epochs} "
            f"time={time.time()-start:.1f}s "
            f"train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f}"
        )


        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy
            },
            checkpoint_file
        )

        print("Checkpoint saved!")


        # Save best model
        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                args.output
            )

            with open(
                "class_names.json",
                "w"
            ) as f:
                json.dump(classes, f)

            print("New best model saved!")


    print("\nTraining complete!")
    print("Best accuracy:", best_accuracy)


if __name__ == "__main__":
    main()