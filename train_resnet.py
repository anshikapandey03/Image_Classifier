import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from tqdm import tqdm


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


checkpoint = "microsoft/resnet-18"


processor = AutoImageProcessor.from_pretrained(
    checkpoint
)


image_mean = processor.image_mean
image_std = processor.image_std


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=8),
    transforms.ToTensor(),
    transforms.Normalize(
        image_mean,
        image_std
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        image_mean,
        image_std
    )
])


train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)


test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


id2label = {
    i: label
    for i, label in enumerate(classes)
}


label2id = {
    label: i
    for i, label in enumerate(classes)
}


model = ResNetForImageClassification.from_pretrained(
    checkpoint,
    num_labels=10,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


model.to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)


epochs = 5


best_accuracy = 0.0


for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{epochs}"
    )


    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=images,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        accuracy = 100 * correct / total

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{accuracy:.2f}%"
        )


    train_accuracy = 100 * correct / total

    average_loss = total_loss / len(train_loader)


    model.eval()

    test_correct = 0
    test_total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(
                pixel_values=images
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)


    test_accuracy = (
        100 * test_correct / test_total
    )


    print()
    print(f"Epoch: {epoch + 1}")
    print(f"Training Loss: {average_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print()


    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        model.save_pretrained(
            "./resnet_cifar10"
        )

        processor.save_pretrained(
            "./resnet_cifar10"
        )

        print(
            f"Best model saved: {best_accuracy:.2f}%"
        )


print()
print("Training completed.")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print("Model saved in: ./resnet_cifar10")

