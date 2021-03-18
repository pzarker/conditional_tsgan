import os
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers, out_dim=1):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim  # num features in original time series
        self.num_layers = num_layers
        self.lstm = nn.LSTM(z_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        """ Generate 'fake' time series from noise

        :param z: noise from latent dim of shape batch_size x seq_len x z_dim
        :return: generated time series of shape batch_size x seq_len x out_dim
        """
        batch_size = z.shape[0]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        out, _ = self.lstm(z, (h0, c0))
        #out = self.ln(out)  # layer norm
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        out = self.tanh(out)
        out = out.view(batch_size, -1, self.out_dim)
        return out


def get_noise(n_samples, seq_len, z_dim, device='cpu'):
    return torch.randn(n_samples, seq_len, z_dim, device=device)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(Critic, self).__init__()
        self.in_dim = in_dim  # num features in original time series
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ Classifies each time step as real or fake

        :param x: time series of shape batch_size x seq_len x in_dim
        :return: classification of shape batch_size x seq_len
        """
        x = x.float()
        batch_size = x.shape[0]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        #out = self.ln(out)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)
        #out = self.sigmoid(out)
        out = out.reshape(batch_size, -1)
        return out


class GeneratorLoss(nn.Module):
    """Implementations of various losses for the generator"""

    def __init__(self, lambda_adversarial=1.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_adversarial = lambda_adversarial

    def bce_adversarial_loss(self, fake, disc):
        disc_fake_pred = disc(fake)
        gen_adv_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_adv_loss

    def lsgan_adversarial_loss(self, fake, disc):
        """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
        disc_fake_pred = disc(fake)
        gen_adv_loss = torch.mean((disc_fake_pred - 1.) ** 2)
        return gen_adv_loss

    def hinge_adversarial_loss(self, fake, disc):
        """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
        disc_fake_pred = disc(fake)
        gen_adv_loss = torch.mean(-disc_fake_pred)

    def forward(self, fake, disc):
        #gen_loss = self.bce_adversarial_loss(fake, disc)
        gen_loss = self.lsgan_adversarial_loss(fake, disc)
        return gen_loss


class DiscriminatorLoss(nn.Module):
    """Implementations of various losses for the discriminator"""

    def __init__(self, lambda_adversarial=1.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_adversarial = lambda_adversarial

    def bce_adversarial_loss(self, real, fake, disc):
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_adv_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    def lsgan_adversarial_loss(self, real, fake, disc):
        """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = torch.mean(disc_fake_pred ** 2)
        disc_real_pred = disc(real)
        disc_real_adv_loss = torch.mean((disc_real_pred - 1.) ** 2)
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    def hinge_adversarial_loss(self, real, fake, disc):
        """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = torch.mean(F.relu(1 - disc_fake_pred))
        disc_real_pred = disc(real)
        disc_fake_adv_loss = torch.mean(F.relu(1 + disc_fake_pred))
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    def forward(self, real, fake, disc):
        #disc_loss = self.bce_adversarial_loss(real, fake, disc)
        disc_loss = self.lsgan_adversarial_loss(real, fake, disc)
        return disc_loss


def save_ckpt(epoch, model, model_name, optimizer, ckpt_dir, device='cpu'):
    """
    Save model checkpoint to disk.

    Args:
        epoch (int): Current epoch
        model : Model to save
        model_name (str): Name of model to save
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        ckpt_dir (str): Directory to save the checkpoint
        device (str): Device where the model/optimizer parameters belong
    """
    try:
        model_class = model.module.__class__.__name__
        model_state = model.to('cpu').module.state_dict()
        print("Unwrapped DataParallel module.")
    except AttributeError:
        model_class = model.__class__.__name__
        model_state = model.to('cpu').state_dict()

    ckpt_dict = {
        'ckpt_info': {'epoch': epoch},
        'model_class': model_class,
        'model_state': model_state,
        'optimizer': optimizer.state_dict(),
    }

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch}.pth.tar")
    torch.save(ckpt_dict, ckpt_path)
    model.to(device)
    print(f"Saved {model_name} at epoch {epoch} to {ckpt_path}.")


def load_ckpt(ckpt_path, model, optimizer=None, scheduler=None):
    """
    Function that loads model, optimizer, and scheduler state-dicts from a ckpt.

    Args:
        model (nn.Module): Initialized model objects
        ckpt_path (str): Path to saved model ckpt
        optimizer (torch.optim.Optimizer): Initialized optimizer object
        scheduler (torch.optim.lr_scheduler): Initilazied scheduler object
    """
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    print(f"Loaded {checkpoint['model_class']} from {ckpt_path} at {checkpoint['ckpt_info']['epoch']}")

