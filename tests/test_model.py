import torch, torchvision, pytorch_lightning, rasterio, albumentations
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Lightning:", pytorch_lightning.__version__)
print("Rasterio:", rasterio.__version__)