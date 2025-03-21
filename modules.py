import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, LR_size):
        x = x.view(-1, self.channels, LR_size * LR_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, LR_size, LR_size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, args, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.c_in = args.in_channels_U_net
        self.c_out = args.n_colors
        self.features = args.n_feats

        self.inc = DoubleConv(self.c_in, self.features)
        self.down1 = Down(self.features, self.features * 2)
        self.down2 = Down(self.features * 2, self.features * 4)
        self.down3 = Down(self.features * 4, self.features * 4)

        self.bot1 = DoubleConv(self.features * 4, self.features * 8)
        self.bot2 = DoubleConv(self.features * 8, self.features * 8)
        self.bot3 = DoubleConv(self.features * 8, self.features * 4)

        self.up1 = Up(self.features * 8, self.features * 2)
        self.up2 = Up(self.features * 4, self.features)
        self.up3 = Up(self.features * 2, self.features)
        self.outc = nn.Conv2d(self.features, self.c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

class Att_UNet(nn.Module):
    def __init__(self, args, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.c_in = args.in_channels_U_net
        self.c_out = args.n_colors
        self.features = args.n_feats
        self.inc = DoubleConv(self.c_in, self.features)
        self.down1 = Down(self.features, self.features * 2)
        self.sa1 = SelfAttention(self.features * 2)
        self.down2 = Down(self.features * 2, self.features * 4)
        self.sa2 = SelfAttention(self.features * 4)
        self.down3 = Down(self.features * 4, self.features * 4)
        self.sa3 = SelfAttention(self.features * 4)

        self.bot1 = DoubleConv(self.features * 4, self.features * 8)
        self.bot2 = DoubleConv(self.features * 8, self.features * 8)
        self.bot3 = DoubleConv(self.features * 8, self.features * 4)

        self.up1 = Up(self.features * 8, self.features * 2)
        self.sa4 = SelfAttention(self.features * 2)
        self.up2 = Up(self.features * 4, self.features)
        self.sa5 = SelfAttention(self.features)
        self.up3 = Up(self.features * 2, self.features)
        self.sa6 = SelfAttention(self.features)
        self.outc = nn.Conv2d(self.features, self.c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, LR_size):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2, LR_size // 2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3, LR_size // 4)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4, LR_size // 8)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x, LR_size // 4)
        x = self.up2(x, x2, t)
        x = self.sa5(x, LR_size // 2)
        x = self.up3(x, x1, t)
        x = self.sa6(x, LR_size)
        output = self.outc(x)
        return output
