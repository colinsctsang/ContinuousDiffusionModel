import torch.nn as nn
from train import Diffusion
import argparse
from modules import UNet
from tqdm import tqdm
from utils import *
import models
import torch.nn.functional as nnf
from torch.autograd import Variable
from math import log10, sqrt
import pytorch_ssim

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.test_batch_size = 1
args.target_image_size = 1024
args.in_channels_U_net = 64  # in channels for U-Net
args.n_colors = 3  # out color channels in U-Net
args.n_feats = 64  # features in U-Net
args.run_name = "run001" # put the name of this run here
args.dataset_test_path = os.getcwd() + r"\dataset\train" # put your testing images in this folder
args.dataset_ground_truth_path = os.getcwd() + r"\dataset\test" # put the noise-free LR images in this folder
args.dataset_sim_path = os.getcwd() + r"\dataset\test" # put the SIM images (i.e., ground truth) in this folder
args.device = "cuda"
args.lr = 1e-4

# input LR size
LR_size = 512

# starting time step
time_step_T = 200  # starting time step for backward process
print("The starting time step is", time_step_T)
time_step_T = torch.tensor(time_step_T)

# parameters
diffusion = Diffusion()
noise_steps = 500
beta_start = 5e-5
beta_end = 1e-2
beta = diffusion.prepare_noise_schedule().to(args.device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# make output file
os.makedirs(os.path.join("test_results", args.run_name), exist_ok=True)

# prepare the mlp
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='edsr-avg1-1000.pth')
parser.add_argument('--gpu', default='0')
argscont = parser.parse_args()

# get test data
test_set = test_stage_get_gt(args)
testing_data_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False)
print('the total number of testing images is ', len(test_set))


# load UNet
model = UNet(args).to(args.device)
ckpt1 = torch.load("./checkpoints/run001/ckpt_UNet.pt") # the location of your trained check pts

model.load_state_dict(ckpt1)

# make decoder MLP and output grid
mlp = models.make(torch.load(argscont.model)['model'], load_sd=True).to(args.device)
h = args.target_image_size
w = args.target_image_size
coord = make_coord((h, w)).cuda()
cell = torch.ones_like(coord)
cell[:, 0] *= 2 / h
cell[:, 1] *= 2 / w

# initialize PSNR, RMSE, and SSIM
avg_psnr_nf, avg_rmse_nf, avg_ssim_nf = 0, 0, 0
sd_psnr_nf, sd_rmse_nf, sd_ssim_nf = 0, 0, 0
avg_psnr_sim, avg_rmse_sim, avg_ssim_sim = 0, 0, 0
sd_psnr_sim, sd_rmse_sim, sd_ssim_sim = 0, 0, 0
all_ssim_nf = []
all_rmse_nf = []
all_ssim_sim = []
all_rmse_sim = []
k = 0
mse = nn.MSELoss()

# start testing
model.eval()
mlp.eval()
with torch.no_grad():
    pbar = tqdm(testing_data_loader)
    for i, images in enumerate(pbar):
        print("working on image", i)

        # get noisy LR and gt LR image
        gt, noisy_LR, sim = Variable(images[0]), Variable(images[1]), Variable(images[2])
        image_x = noisy_LR.to(args.device)
        gt = gt.to(args.device)
        sim = sim.to(args.device)

        # get time step t
        t = time_step_T.to(args.device)

        # interpolate LR to input size (original size is 512 )
        image_x = nnf.interpolate(image_x, size=(LR_size, LR_size), mode='bicubic', align_corners=False)

        # get target image, and interpolate it to final output size (original size is 1024 for sim and 512 for noise-free)
        gt = nnf.interpolate(gt, size=(args.target_image_size, args.target_image_size), mode='bicubic', align_corners=False)
        noisy_HR = nnf.interpolate(image_x, size=(args.target_image_size, args.target_image_size), mode='bicubic', align_corners=False)
        sim = nnf.interpolate(sim, size=(args.target_image_size, args.target_image_size), mode='bicubic', align_corners=False)

        # backward process
        image_clean = image_x
        for j in tqdm(reversed(range(1, t)), position=0):
            # print("the current time step is", j)
            j = torch.tensor(j)
            j = j.to(args.device)
            a = alpha[j]
            b = beta[j]
            a_hat = alpha_hat[j]

            # get predicted noise from U-Net
            image_clean_latent = mlp.gen_feat(image_clean)
            predicted_noise = model(image_clean_latent, j)
            if j > 1:
                noise = torch.randn_like(image_clean)
            else:
                noise = torch.zeros_like(image_clean)
            image_clean = 1 / torch.sqrt(a) * (image_clean - ((1 - a) / (torch.sqrt(1 - a_hat))) * predicted_noise) + torch.sqrt(
                b) * noise


        # MLP decoder to make HR output
        clean_HR = batched_predict(mlp, (image_clean[0, :, :, :]).cuda().unsqueeze(0),
                                       coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        clean_HR = clean_HR.view(h, w, 3).permute(2, 0, 1)


        # calculate PSNR, RMSE, and SSIM
        clean_HR_cal = clean_HR[None, :, :, :]
        mse_nf = torch.sqrt(mse(clean_HR_cal, gt))
        mse_sim = mse(clean_HR_cal, sim)
        k = k + 1
        old_avg_psnr_nf = avg_psnr_nf
        old_avg_rmse_nf = avg_rmse_nf
        old_avg_ssim_nf = avg_ssim_nf

        print(clean_HR_cal.size())

        all_rmse_nf.append(mse_nf.item())
        all_rmse_sim.append(mse_sim.item())
        all_ssim_nf.append(pytorch_ssim.ssim(clean_HR_cal, gt).item())
        all_ssim_sim.append(pytorch_ssim.ssim(clean_HR_cal, sim).item())
        print("all rmse nf", all_rmse_nf)
        print("all rmse sim", all_rmse_sim)
        print("all ssim nf", all_ssim_nf)
        print("all ssim sim", all_ssim_sim)


        avg_psnr_nf += (10 * log10(1 / mse_nf.data) - avg_psnr_nf) / k
        avg_rmse_nf += (sqrt(mse_nf) - avg_rmse_nf) / k
        avg_ssim_nf += (pytorch_ssim.ssim(clean_HR_cal, gt) - avg_ssim_nf) / k
        sd_psnr_nf += (10 * log10(1 / mse_nf.data) - avg_psnr_nf) * (10 * log10(1 / mse_nf.data) - old_avg_psnr_nf)
        sd_rmse_nf += (sqrt(mse_nf) - avg_rmse_nf) * (sqrt(mse_nf) - old_avg_rmse_nf)
        sd_ssim_nf += (pytorch_ssim.ssim(clean_HR_cal, gt) - avg_ssim_nf) * (
                    pytorch_ssim.ssim(clean_HR_cal, gt) - old_avg_ssim_nf)
        print("===> Avg. PSNR_nf: {:.4f} dB".format(avg_psnr_nf),
              ",===> S.D: {:.4f} dB".format(sd_psnr_nf / len(testing_data_loader)))
        print("===> Avg. RMSE_nf: {:.4f} dB".format(avg_rmse_nf),
              ",===> S.D: {:.4f} dB".format(sd_rmse_nf / len(testing_data_loader)))
        print("===> Avg. SSIM_nf: {:.4f} dB".format(avg_ssim_nf),
              ",===> S.D: {:.4f} dB".format(sd_ssim_nf / len(testing_data_loader)))
        old_avg_psnr_sim = avg_psnr_sim
        old_avg_rmse_sim = avg_rmse_sim
        old_avg_ssim_sim = avg_ssim_sim
        avg_psnr_sim += (10 * log10(1 / mse_sim.data) - avg_psnr_sim) / k
        avg_rmse_sim += (sqrt(mse_sim) - avg_rmse_sim) / k
        avg_ssim_sim += (pytorch_ssim.ssim(clean_HR_cal, sim) - avg_ssim_sim) / k
        sd_psnr_sim += (10 * log10(1 / mse_sim.data) - avg_psnr_sim) * (10 * log10(1 / mse_sim.data) - old_avg_psnr_sim)
        sd_rmse_sim += (sqrt(mse_sim) - avg_rmse_sim) * (sqrt(mse_sim) - old_avg_rmse_sim)
        sd_ssim_sim += (pytorch_ssim.ssim(clean_HR_cal, sim) - avg_ssim_sim) * (
                pytorch_ssim.ssim(clean_HR_cal, sim) - old_avg_ssim_sim)
        print("===> Avg. PSNR_sim: {:.4f} dB".format(avg_psnr_sim),
              ",===> S.D: {:.4f} dB".format(sd_psnr_sim / len(testing_data_loader)))
        print("===> Avg. RMSE_sim: {:.4f} dB".format(avg_rmse_sim),
              ",===> S.D: {:.4f} dB".format(sd_rmse_sim / len(testing_data_loader)))
        print("===> Avg. SSIM_sim: {:.4f} dB".format(avg_ssim_sim),
              ",===> S.D: {:.4f} dB".format(sd_ssim_sim / len(testing_data_loader)))


        # save images
        clean_HR = (clean_HR.clamp(-1, 1) + 1) / 2
        clean_HR = (clean_HR * 255).type(torch.uint8)
        noisy_HR = (noisy_HR.clamp(-1, 1) + 1) / 2
        noisy_HR = (noisy_HR * 255).type(torch.uint8)
        image_x = (image_x.clamp(-1, 1) + 1) / 2
        image_x = (image_x * 255).type(torch.uint8)
        gt = (gt.clamp(-1, 1) + 1) / 2
        gt = (gt * 255).type(torch.uint8)
        sim = (sim.clamp(-1, 1) + 1) / 2
        sim = (sim * 255).type(torch.uint8)



        save_color_images(image_x, os.path.join("test_results", args.run_name, f"{i}_in.jpg"))
        save_color_images(noisy_HR, os.path.join("test_results", args.run_name, f"{i}_noisyHR.jpg"))
        save_color_images(clean_HR, os.path.join("test_results", args.run_name, f"{i}_out.jpg"))
        save_color_images(gt, os.path.join("test_results", args.run_name, f"{i}_zgt1noisefree.jpg"))
        save_color_images(sim, os.path.join("test_results", args.run_name, f"{i}_zgt0sim.jpg"))







