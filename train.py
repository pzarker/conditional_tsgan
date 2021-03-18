import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchnet import meter

import os
import time
import random
import numpy as np
from tqdm.auto import tqdm

# user packages
from data_utils import get_data
from model import Generator, Critic, get_noise
from model import GeneratorLoss, DiscriminatorLoss
from model import save_ckpt, load_ckpt
from plotting import save_plot_sample

exp_no = "8"
exp_dir = f"./experiments/exp{exp_no}"
os.makedirs(exp_dir, exist_ok=True)
print(f"Experiment {exp_no}")
print(f"PyTorch {torch.__version__}")

ckpt_dir = os.path.join(exp_dir, "ckpts")
os.makedirs(ckpt_dir, exist_ok=True)

plot_dir = os.path.join(exp_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

model_dir = os.path.join(exp_dir, "pretrained_models")
os.makedirs(model_dir, exist_ok=True)

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

'''
TODO (for consideration):
*  normalize data
*  use batch mean
*  initialize weights for generator and discriminator
*  learning rate scheduler
'''


# LOAD DATA
data_type = 'sine'
options = {'seq_length': 20,
           'num_samples': 16000,
           'num_signals': 1,
           'freq_low': 1,
           'freq_high': 5,
           'amplitude_low': 0.3,
           'amplitude_high': 0.9}
data, _, _ = get_data(data_type, options)

print(data.shape)

batch_size = 8
shuffle = True
dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)


# INITIALIZE GEN AND DISC
z_dim = 10
device = 'cpu'  # GPU
hidden_dim_g = 200
hidden_dim_d = 100
in_dim = options['num_signals']

num_layers = 2
gen = Generator(z_dim, hidden_dim_g, num_layers).to(device)
crit = Critic(in_dim, hidden_dim_d, num_layers).to(device)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


gen = gen.apply(weights_init)
crit = crit.apply(weights_init)


# OPTIMIZER
lr = 0.0002  # 0.0002
beta_1 = 0.9  # 0.5
beta_2 = 0.999
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))


# LOADING CKPTS
'''
gen_ckpt_path = "/Users/parkerzhao/Desktop/Projects/gans_panel_data/code/" \
                "conditional_tsgan/experiments/exp6/ckpts/generator_epoch_80.pth.tar"
disc_ckpt_path = "/Users/parkerzhao/Desktop/Projects/gans_panel_data/code/" \
                 "conditional_tsgan/experiments/exp6/ckpts/critic_epoch_80.pth.tar"
'''
gen_ckpt_path = None
disc_ckpt_path = None

if gen_ckpt_path:
    load_ckpt(gen_ckpt_path, gen, gen_opt)
if disc_ckpt_path:
    load_ckpt(disc_ckpt_path, crit, crit_opt)

# TRAINING
n_epochs = 50
n_rounds_g_per_d = 5  # >1 => train generator more than discriminator
clip = 0.25

cur_step = 0
display_step = 5000
epochs_per_save = 20

# Initialize average meters
gen_loss_meter = meter.AverageValueMeter()
crit_loss_meter = meter.AverageValueMeter()

# Initialize loss functions
gen_loss_fn = GeneratorLoss()
crit_loss_fn = DiscriminatorLoss()

gen_loss_history = []
crit_loss_history = []

n_plot_samples = 6

start = time.time()
for epoch in range(n_epochs):
    for real in tqdm(dataloader):
        cur_batch_size, seq_len, _ = real.shape
        real = real.to(device)

        fake_noise = get_noise(cur_batch_size, seq_len, z_dim, device=device)
        fake = gen(fake_noise)

        # Update discriminator
        if cur_step % n_rounds_g_per_d == 0:
            crit_opt.zero_grad()
            crit_loss = crit_loss_fn(real, fake, crit)
            crit_loss.backward(retain_graph=True)
            crit_opt.step()
            crit_loss_meter.add(crit_loss.item())

        # Update generator
        gen_opt.zero_grad()
        gen_loss = gen_loss_fn(fake, crit)
        gen_loss.backward()
        ''' Vanishing gradient problem?
        torch.nn.utils.clip_grad_norm_(gen.parameters(), clip)
        '''
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Gradient: {gen.lstm.weight_ih_l0.grad}")

        gen_opt.step()
        gen_loss_meter.add(gen_loss.item())

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {gen_loss_meter.value()[0]}, "
                  f"discriminator loss: {crit_loss_meter.value()[0]}")
            gen_loss_history.append(gen_loss_meter.value()[0])
            crit_loss_history.append(crit_loss_meter.value()[0])
            # Reset average meters
            gen_loss_meter.reset()
            crit_loss_meter.reset()
            save_plot_sample(fake, f"Fakes at Step {cur_step}", plot_dir, f"fake_step_{cur_step}", n_samples=n_plot_samples, ncol=3)
            save_plot_sample(real, f"Reals at Step {cur_step}", plot_dir, f"real_step_{cur_step}", n_samples=n_plot_samples, ncol=3)

        cur_step += 1

    if epoch % epochs_per_save == 0:
        save_ckpt(epoch, gen, 'generator', gen_opt, ckpt_dir, device)
        save_ckpt(epoch, crit, 'critic', crit_opt, ckpt_dir, device)
        time_elapsed = time.time() - start
        print(f'Time elapsed: {time_elapsed:.2f} seconds')

# save generator for inference
model_name = 'generator_final'
save_ckpt(n_epochs, gen, model_name, gen_opt, model_dir, device)

end = time.time()

# EVALUATE MODEL
gen_to_eval = gen
gen_to_eval.eval()

n_samples_eval = 100
seq_len = options['seq_length']
noise = get_noise(n_samples_eval, seq_len, z_dim, device=device)
fake = gen_to_eval(noise)

save_plot_sample(fake, f"Fakes with Final Generator", exp_dir, f"real_step_{cur_step}", n_samples=n_plot_samples, ncol=3)

# KEEP RECORD OF EXPERIMENT RUN
record_name = os.path.join(exp_dir, "record.txt")
with open(record_name, 'a') as out:
    out.write("RESULTS\n")
    out.write("Evaluation metric:\n")
    out.write(f"Final generator loss: {gen_loss_history[-1]}\n"
              f"Final critic loss: {crit_loss_history[-1]}\n"
              f"Training time: {end-start:.2f} seconds")
    out.write("\n")
    out.write("DATA\n")
    out.write(f"{data_type}\n")
    out.write(f"sequence length: {seq_len}\n"
              f"num_samples: {options['num_samples']}\n"
              f"num_signals: {options['num_signals']}\n"
              f"freq_range: {options['freq_low']} to {options['freq_high']}\n"
              f"amp_range: {options['amplitude_low']} to {options['amplitude_high']}\n")
    out.write("\n")
    out.write("MODEL PARAMETERS\n")
    out.write(f"z dim: {z_dim}\n"
              f"hidden gen dim: {hidden_dim_g}\n"
              f"hidden crit dim: {hidden_dim_d}\n"
              f"number of layers: {num_layers}\n"
              f"number of generator updates per critic update: {n_rounds_g_per_d}\n")
    out.write("\n")
    out.write("HYPERPARAMETERS\n")
    out.write(f"Number of epochs: {n_epochs}\n"
              f"batch size: {batch_size}\n")
    out.write("\n")
    out.write("OPTIMIZER\n")
    out.write(f"learning rate: {lr}, beta1: {beta_1}, beta2: {beta_2}")
    out.write(f"Xavier weight initialization")
    #out.write("Added layer norm")
    #out.write(f"Gradient clip: {clip}")
    out.write("Added Dropout 0.2")
    out.write("Least squares loss for generator and critic")

