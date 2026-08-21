import os
import numpy as np
import torch
from metageformer_torch.tokenizer import MetAgeFormer_Tokenizer
from metageformer_torch.models import MetAgeFormer_ForPreTrain
from metageformer_torch.dataset import load_dataset_from_dir_NMR
from metageformer_torch.mask_utils import Mask_Schedule
from typing import Optional
import math
import json
import argparse

from utils import *


class WarmupLR:
    def __init__(self, optimizer, max_lr, num_warm, num_allsteps, decay_type='linear') -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = max_lr
        self.num_step = 0
        self.num_allsteps = num_allsteps
        self.decay_type = decay_type

    def __compute(self, lr) -> float:
        if self.num_step <= self.num_warm:
            initial_lr = lr * 0.1
            return initial_lr + (lr - initial_lr) * (self.num_step / self.num_warm)
        else:
            if self.decay_type == 'linear':
                return lr * (1 - ((self.num_step - self.num_warm) / (self.num_allsteps - self.num_warm)))
            elif self.decay_type == 'cosine':
                return lr * 0.5 * (1 + math.cos(math.pi * (self.num_step - self.num_warm) / (self.num_allsteps - self.num_warm)))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]

    def get_lr(self):
        return self.lr


def train(
        model,
        dataloader,
        n_train,
        val_dataloader,
        lr: float = 0.0001,
        n_epoch: int = 20,
        save_dir: str = './',
        max_mr: float = .5,
        n_toler: Optional[int] = None,
        f_loss: str = 'MAE',
        f_gamma: str = 'cosine',
    ):

    mask_scheduler = Mask_Schedule(mode='train', max_mask_ratio=max_mr, gamma_func=f_gamma)

    if f_loss == 'MAE':
        criterion = torch.nn.L1Loss(reduction='mean')
    elif f_loss == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif f_loss == 'SmoothL1Loss':
        criterion = torch.nn.SmoothL1Loss(reduction='mean')
    else:
        raise TypeError("Loss function type is not supported.")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-07)

    n_step_epoch = math.ceil(n_train / batch_size)
    n_total_steps = n_epoch * n_step_epoch
    n_warm_steps = int(n_total_steps / 10)
    scheduler = WarmupLR(
        optimizer=optim,
        max_lr=[lr],
        num_warm=n_warm_steps,
        num_allsteps=n_total_steps,
        decay_type='cosine',
    )

    train_loss_epoch = []
    train_conc_loss_epoch = []
    val_loss_epoch = []
    val_conc_loss_epoch = []

    best_val_loss = 999999
    watchdog = 0
    step_count = 0

    for epoch in range(n_epoch):
        model.train()

        step_loss_collect = []
        step_conc_loss_collect = []

        for data in dataloader:
            optim.zero_grad()

            mask_ratios = mask_scheduler.get_ratio(len(data))
            inputs, label = Tokenizer.tokenize_from_anndata(
                data,
                padding='longest',
                masking='random',
                data_layer='Z-score normalized',
                masking_ratios=mask_ratios,
                return_tensor=True,
                device=device,
            )

            outputs = model(inputs)

            idx_masked = torch.where(inputs['masking_mask'] == 1)
            predicted = outputs['logit_conc'][:, 1:]
            loss = criterion(predicted[idx_masked], label[idx_masked])
            step_conc_loss_collect.append(loss.data.cpu().detach().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optim.step()

            step_loss_collect.append(loss.data.cpu().detach().numpy())
            scheduler.step()
            step_count += 1

            train_info = {
                'learning_rate': optim.param_groups[0]['lr'],
                'loss': step_loss_collect[-1],
                'conc': step_conc_loss_collect[-1],
            }
            print(f"step {step_count} loss:\t{step_loss_collect[-1]}\tconc:\t{step_conc_loss_collect[-1]}")

            if wandb_monitor == True:
                wandb.log(train_info)

        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))
        train_conc_loss_epoch.append(np.mean(np.array(step_conc_loss_collect)))

        save_txt_single_column(train_loss_epoch, save_dir=save_dir, filename='train_loss.txt')
        save_txt_single_column(train_conc_loss_epoch, save_dir=save_dir, filename='train_conc_loss.txt')

        val_loss, val_loss_conc = validation(model, val_dataloader, mask_scheduler, criterion)
        val_loss_epoch.append(val_loss)
        val_conc_loss_epoch.append(val_loss_conc)
        print(f'epoch:\t{str(epoch)}, val loss:\t{str(val_loss_epoch[-1])}')
        save_txt_single_column(val_loss_epoch, save_dir=save_dir, filename='val_loss.txt')
        save_txt_single_column(val_conc_loss_epoch, save_dir=save_dir, filename='val_conc_loss.txt')

        if isinstance(n_toler, int):
            watchdog += 1
            if val_loss < best_val_loss:
                watchdog = 0
                best_val_loss = val_loss
                model.save_pretrained(save_path=os.path.join(save_dir, 'model_weights.pth'))
            if watchdog >= n_toler:
                break
        else:
            model.save_pretrained(save_path=os.path.join(save_dir, 'model_weights.pth'))

        torch.cuda.empty_cache()


def validation(model, val_dataloader, mask_scheduler, criterion):
    model.eval()

    step_loss_collect = []
    conc_loss_collect = []

    for data in val_dataloader:
        mask_ratios = mask_scheduler.get_ratio(len(data))
        inputs, label = Tokenizer.tokenize_from_anndata(
            data,
            padding='longest',
            masking='random',
            data_layer='Z-score normalized',
            masking_ratios=mask_ratios,
            return_tensor=True,
            device=device,
        )

        with torch.no_grad():
            outputs = model(inputs)

        idx_masked = torch.where(inputs['masking_mask'] == 1)
        predicted = outputs['logit_conc'][:, 1:]
        loss = criterion(predicted[idx_masked], label[idx_masked])
        conc_loss_collect.append(loss.data.cpu().detach().numpy())
        step_loss_collect.append(loss.data.cpu().detach().numpy())

    val_loss = np.mean(np.array(step_loss_collect))
    val_loss_conc = np.mean(np.array(conc_loss_collect))
    val_info = {'val_loss': val_loss, 'val_loss_conc': val_loss_conc}

    if wandb_monitor == True:
        wandb.log(val_info)

    return val_loss, val_loss_conc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--train_config', type=str, help='.')
    args = parser.parse_args()

    with open(args.train_config, 'r') as file:
        train_settings = json.load(file)

    lr = train_settings['lr']
    batch_size = train_settings['batch_size']
    n_epoch = train_settings['n_epoch']
    n_toler = train_settings['n_toler']
    max_mr = train_settings['max_mr']
    f_loss = train_settings['f_loss']
    data_path = train_settings['data_path']
    f_gamma = train_settings['f_gamma']
    wandb_monitor = train_settings['wandb_monitor']

    drop_out = train_settings['drop_out']
    attn_mode = train_settings['attn_mode']
    n_heads = train_settings['n_heads']
    n_blocks = train_settings['n_blocks']
    d_ff = train_settings['d_ff']
    d_model = train_settings['d_model']
    f_act = train_settings['f_act']

    save_note = train_settings['save_note']
    save_dir = os.path.join(train_settings['save_dir'], f'{save_note}')

    create_directory(save_dir)

    set_seeds(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if wandb_monitor == True:
        import wandb  # optional dependency; only required when wandb_monitor is true
        wandb.login()  # uses WANDB_API_KEY env var or local login credentials
        wandb.init(project=os.environ.get("WANDB_PROJECT", "MetAgeFormer"))

    train_loader, val_loader, train_data, _, = load_dataset_from_dir_NMR(
        data_path,
        batch_size=batch_size,
        device=device,
    )
    num_train = len(train_loader.dataset)
    metabo_id = train_data.var_names.values.tolist()

    Tokenizer = MetAgeFormer_Tokenizer(VOCAB_Identifiers=metabo_id)

    EmbeddingModule_conf = {
        "n_vocabs": {'identifier': Tokenizer.vocab_size_identifiers}
    }

    save_tokenizer(Tokenizer, save_dir=save_dir)

    Model_conf = {
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "d_ff": d_ff,
        "d_model": d_model,
        "dropout": drop_out,
        "activation": f_act,
        "need_weights": True,
        "average_attn_weights": True,
        "attn_mode": attn_mode,
    }

    model = MetAgeFormer_ForPreTrain(EmbeddingModule_conf, Model_conf)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters:\t{total_params}')

    Model_conf_all = Model_conf
    Model_conf_all['n_params'] = total_params
    save_dict_2_json(Model_conf_all, filename='config.json', save_dir=save_dir)

    model.to(device)

    train(
        model,
        train_loader,
        num_train,
        val_loader,
        lr,
        n_epoch,
        save_dir,
        max_mr,
        n_toler,
        f_loss,
        f_gamma,
    )

    if wandb_monitor == True:
        wandb.finish()
