"""
Télécharge le dataset MNIST dans data/.
Run : python download_data.py
"""
from torchvision import datasets, transforms

print("Téléchargement MNIST...")
datasets.MNIST(root="data/", train=True,  download=True, transform=transforms.ToTensor())
datasets.MNIST(root="data/", train=False, download=True, transform=transforms.ToTensor())
print("Done — data/MNIST/raw/")
