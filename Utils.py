import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

torch.manual_seed(10000)
# We can use an image folder dataset the way we have it setup.
# Create the dataset
image_size = 64
batch_size = 128
workers = 2
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def loadImages(dataroot):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    print("Number of images: ", len(dataset))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
        :64], padding=2, normalize=True).cpu(), (1, 2, 0)))

    return dataloader, device


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(
        make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def getFakeImage(netG,  fixed_noise):
    
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    return vutils.make_grid(fake, padding=2, normalize=True)


def printStats(epoch, epochs, loss_g, loss_d):
    print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(epoch+1, epochs, loss_g, loss_d))


def imageFlow(img_list):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list[:]]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    return ani
    
    