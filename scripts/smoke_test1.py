from src.models import SiameseUNet
import torch

model = SiameseUNet(num_classes=4, pretrained=False)
optical       = torch.randn(2, 3, 512, 512)
sar           = torch.randn(2, 1, 512, 512)
optical_valid = torch.tensor([True, False])

out = model(optical, sar, optical_valid)
print(out.shape)   # → torch.Size([2, 5, 512, 512]) 