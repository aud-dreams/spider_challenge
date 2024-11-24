import torch
from torch.utils.data import DataLoader
from load import load_mha_files
from dataset import SpiderDataset
from unetpp import UNetPlusPlus
#from preprocess import

images_dir = 'images'
masks_dir = 'masks'

# load data
images = load_mha_files(images_dir)
masks = load_mha_files(masks_dir)
print(f"Loaded {len(images)} images of shape {images[0].shape} and {len(masks)} masks of shape {masks[0].shape}")

# preprocess

# dataset
dataset = SpiderDataset(images, masks)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# train Unet++
unetpp = UNetPlusPlus(in_channels=1, out_channels=1)
optimizer = torch.optim.Adam(unetpp.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    unetpp.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = unetpp(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# train SAM
