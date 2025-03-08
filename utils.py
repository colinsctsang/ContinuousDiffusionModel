import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as data
from os.path import join
from os import listdir



def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_gray_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path, cmap='gray')


def save_color_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(args.image_size // args.image_crop_scale),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def test_get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(args.image_test_size // args.image_test_crop_scale),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_test_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=True)
    return dataloader



def setup_logging(run_name):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)  # encoder
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


# get noisyLR-GT pairs for test stage
def simple_transform():
    return torchvision.transforms.Compose([
        # torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# get SIM with no crop
def sim_transform():
    return torchvision.transforms.Compose([
        # torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def test_stage_get_gt(args):
    gt_dir = args.dataset_test_path
    test_dir = args.dataset_ground_truth_path
    sim_gt_dir = args.dataset_sim_path

    return TestingDatasetFromFolder(test_dir, gt_dir, sim_gt_dir, avg400_transform=simple_transform(), noisy_input_transform=simple_transform(), simgt_transform=sim_transform())


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_test_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class TestingDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, gt_dir, sim_gt_dir, avg400_transform=None, noisy_input_transform=None, simgt_transform=simple_transform()):
        super(TestingDatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_gt_filenames = [join(gt_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_sim_filenames = [join(sim_gt_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.avg400_transform = avg400_transform
        self.noisy_input_transform = noisy_input_transform
        self.simgt_transform = simgt_transform


    def __getitem__(self, index):
        # get noise-free image
        input_gt = load_test_img(self.image_gt_filenames[index])
        avg400 = self.avg400_transform(input_gt)
        # get noisy image
        noisy = load_test_img(self.image_filenames[index])
        noisy_input = self.noisy_input_transform(noisy)
        # get sim image
        sim = load_test_img(self.image_sim_filenames[index])
        sim_gt = self.simgt_transform(sim)
        return noisy_input, avg400, sim_gt

    def __len__(self):
        return len(self.image_filenames)



