import copy
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import EMA, UNet
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as nnf
import random
import models
from torch.autograd import Variable

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    # set time step and noise schedule here
    def __init__(self, noise_steps=500, beta_start=5e-5, beta_end=1e-2, img_size=320, warmup_frac=0.5, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.warmup_frac = warmup_frac

        # you may change the noise schedule here
        self.beta = self.reverse_warmup_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # linear noise beta
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # reverse warmup noise beta
    def reverse_warmup_noise_schedule(self):
        betas = self.beta_start * torch.ones(self.noise_steps)
        warmup_time = int(self.noise_steps * self.warmup_frac)
        betas[self.noise_steps - warmup_time:] = torch.linspace(
            self.beta_start, self.beta_end, warmup_time)
        return betas

    # output noisy image based on beta
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    # return a random time step
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


def test_denoiseSR(args, model_unet, mlp, epoch):
    diffusion = Diffusion()
    device = args.device
    img_size = args.image_test_size // args.image_test_crop_scale
    LR_size = args.image_test_size // args.image_test_crop_scale
    test_dataloader = test_get_data(args)

    # starting time step for backward process
    time_step_T = 200
    time_step_T = torch.tensor(time_step_T)

    # make noise schedule
    beta = diffusion.prepare_noise_schedule().to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    # set scale for this SR test
    rand_scale = 2

    # create grid for MLP
    h = img_size * rand_scale
    w = img_size * rand_scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    with torch.no_grad():
        # get test image and time step t
        image_x = next(iter(test_dataloader))[0]
        image_x = image_x.to(device)
        t = time_step_T.to(device)

        # get noisy HR
        noisy_HR = nnf.interpolate(image_x, size=(img_size * rand_scale, img_size * rand_scale), mode='bicubic', align_corners=False)

        # make LR clean image by backward process
        image_clean_LR = image_x
        for j in tqdm(reversed(range(1, t)), position=0):
            # get noise parameters from noise schedule
            j = torch.tensor(j)
            j = j.to(device)
            a = alpha[j]
            b = beta[j]
            a_hat = alpha_hat[j]

            # get predicted noise from U-Net
            image_clean_LR_latent = mlp.gen_feat(image_clean_LR)
            predicted_noise_LR = model_unet(image_clean_LR_latent, j)
            if j > 1:
                noise = torch.randn_like(image_clean_LR)
            else:
                noise = torch.zeros_like(image_clean_LR)

            # use the formula to get predicted clean image of previous time step
            image_clean_LR = 1 / torch.sqrt(a) * (
                        image_clean_LR - ((1 - a) / (torch.sqrt(1 - a_hat))) * predicted_noise_LR) + torch.sqrt(
                b) * noise

        # MLP decoder to make HR output
        clean_HR = torch.zeros(size=(args.test_batch_size, args.n_colors, img_size * rand_scale, img_size * rand_scale), device=device)
        for k in range(args.test_batch_size):
            pred = batched_predict(mlp, (image_clean_LR[k, :, :, :]).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
            pred = pred.view(h, w, 3).permute(2, 0, 1)
            clean_HR[k, :, :, :] = pred

        # save images to results folder
        clean_HR = (clean_HR.clamp(-1, 1) + 1) / 2
        clean_HR = (clean_HR * 255).type(torch.uint8)
        images_x_LR = (image_x.clamp(-1, 1) + 1) / 2
        images_x_LR = (images_x_LR * 255).type(torch.uint8)
        noisy_HR = (noisy_HR.clamp(-1, 1) + 1) / 2
        noisy_HR = (noisy_HR * 255).type(torch.uint8)
        save_gray_images(images_x_LR, os.path.join("results", args.run_name, f"{epoch}_noisyLR_in.jpg"))
        save_color_images(clean_HR, os.path.join("results", args.run_name, f"{epoch}_color_out.jpg"))
        save_color_images(noisy_HR, os.path.join("results", args.run_name, f"{epoch}_noisyHR_out.jpg"))




def train(args, argscont):
    setup_logging(args.run_name)
    device = args.device
    img_size = args.image_size // args.image_crop_scale

    # get data
    dataloader = get_data(args)

    # make U-Net for backward-process
    model_unet = UNet(args).to(device)

    # make decoder MLP and output grid
    mlp = models.make(torch.load(argscont.model)['model'], load_sd=True).to(device)
    h = img_size
    w = img_size
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    optimizer = optim.AdamW(model_unet.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # diffusion function class for forward process
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    # EMA for faster training
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model_unet).eval().requires_grad_(False)

    # Epoch loop
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            # get noisy-HR image
            images_HR = images.to(device)

            # generate random down-sample noisy input image
            rand_scale = random.choice((2, 4))
            LR_size = img_size // rand_scale

            images_LR = nnf.interpolate(images_HR, size=(LR_size, LR_size), mode='bicubic', align_corners=False)

            # forward process to get target LR noise and target HR noise
            t = diffusion.sample_timesteps(images_LR.shape[0]).to(device)  # t are some random integers for time steps
            x_t, target_noise_LR = diffusion.noise_images(images_LR, t)
            _, target_noise_HR = diffusion.noise_images(images_HR, t)

            # apply encoder to input noisy LR latent image
            x_t = mlp.gen_feat(x_t)
            print(x_t.size())

            # U-Net to output predicted noise pixel space, att-Unet need LR_size input
            predicted_noise_LR = model_unet(x_t, t)
            # predicted_noise_LR = model_unet(x_t, t, LR_size)

            # MLP decoder to get predicted noise in HR
            predicted_noise_HR = torch.zeros(size=(args.batch_size, args.n_colors, img_size, img_size), device=device)
            for k in range(args.batch_size):
                pred = batched_predict(mlp, (predicted_noise_LR[k, :, :, :]).cuda().unsqueeze(0),
                                   coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
                pred = pred.view(h, w, 3).permute(2, 0, 1)
                predicted_noise_HR[k, :, :, :] = pred
            predicted_noise_HR = Variable(predicted_noise_HR, requires_grad=True)

            # train with two losses: LR and HR
            loss_1 = mse(target_noise_HR, predicted_noise_HR)
            loss_2 = mse(target_noise_LR, predicted_noise_LR)
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model_unet)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # test the model at some epoch
        if epoch > 500:
            if epoch % 10 == 0:
                model_unet.eval()
                test_denoiseSR(args, model_unet, mlp, epoch)
                model_unet.train()
        else:
            if epoch % 100 == 0:
                model_unet.eval()
                test_denoiseSR(args, model_unet, mlp, epoch)
                model_unet.train()

        # save two models
        torch.save(model_unet.state_dict(), os.path.join("checkpoints", args.run_name, f"ckpt_UNet.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "run001" # put your name for this run here
    args.epochs = 1000
    args.batch_size = 8
    args.test_batch_size = 1
    args.image_size = 512
    args.image_crop_scale = 2 # scale=2 means the img is randomly cropped into a 256*256 image
    args.image_test_size = 512
    args.image_test_crop_scale = 1
    args.in_channels_U_net = 64  # in channels for U-Net
    args.n_colors = 3  # out color channels in U-Net
    args.n_feats = 64  # features in U-Net
    args.dataset_path = os.getcwd() + r"\dataset\train" # put your training set in this folder
    args.dataset_test_path = os.getcwd() + r"\dataset\test" # put your testing set in this folder
    args.device = "cuda"
    args.lr = 1e-4

    # for mlp
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='edsr-avg1-1000.pth')
    parser.add_argument('--gpu', default='0')
    argscont = parser.parse_args()


    train(args, argscont)


if __name__ == '__main__':
    launch()

