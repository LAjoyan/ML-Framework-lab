import torchvision.datasets as datasets

dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
)

for i in range(5):
    img, label = dataset[i]
    img.save(f"cifar_test_{i}.png")

print("5 images saved!")
