import os
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from face2sketch import get_face2sketch_loader
from models import Discriminator, Generator
from utils import make_dirs, get_lr_scheduler, set_requires_grad, sample_images, plot_losses

# cuDNN Configuration for performance and reproducibility
cudnn.deterministic = True
cudnn.benchmark = False

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------- Multi-Stage Training Configuration ----------------------
# Strictly aligned with the proposed 3-stage cognitive framework in the paper.
STAGE1_EPOCH = 50  # Stage 1: Contour extraction (Train G1+D1, a <-> b1)
STAGE2_EPOCH = 150  # Stage 2: Detail refinement (Train G2+D2, a+fake_b1 <-> b2)
STAGE3_EPOCH = 250  # Stage 3: Texture synthesis (Train G3+D3, a+fake_b2 <-> b3)


def train():
    # Set random seeds for reproducibility
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Create output directories
    paths = [config.samples_path, config.weights_path, config.plots_path, config.checkpoints_path]
    [make_dirs(path) for path in paths]

    # ---------------------- Data Loading ----------------------
    # Load training datasets
    train_face_loader, train_sketch1_loader, train_sketch2_loader, train_sketch3_loader = get_face2sketch_loader(
        purpose='train', batch_size=config.batch_size if hasattr(config, 'batch_size') else 4)

    # Load testing datasets for validation sampling
    test_face_loader, test_sketch_loader1, test_sketch_loader2, test_sketch_loader3 = get_face2sketch_loader(
        purpose='test', batch_size=config.val_batch_size if hasattr(config, 'val_batch_size') else 2)

    # Calculate total iterations per epoch
    total_batch = min(len(train_face_loader), len(train_sketch1_loader),
                      len(train_sketch2_loader), len(train_sketch3_loader))

    # ===================== Selective Weight Initialization =====================
    def safe_init_weights(model):
        """
        Initializes the decoder and adaptation layers using normal distribution,
        but strictly preserves the pre-trained weights of the VGG encoder
        (layers starting with 'enc').
        """
        print(
            f"[*] Applying selective initialization to {model.__class__.__name__} (Preserving VGG Encoder weights)...")
        for name, module in model.named_modules():
            if 'enc' in name:
                continue  # Skip pre-trained VGG feature extraction layers

            # Initialize other layers (Decoder, CBAM, Adapter)
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight.data, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

    # ---------------------- Model Initialization ----------------------
    # Discriminators: All take 3-channel inputs (Real/Fake evaluations)
    D1_0 = Discriminator(input_nc=3).to(device)  # Discriminates face 'a'
    D1_1 = Discriminator(input_nc=3).to(device)  # Discriminates sketch 'b1'
    D2_0 = Discriminator(input_nc=3).to(device)  # Discriminates face 'a'
    D2_2 = Discriminator(input_nc=3).to(device)  # Discriminates sketch 'b2'
    D3_0 = Discriminator(input_nc=3).to(device)  # Discriminates face 'a'
    D3_3 = Discriminator(input_nc=3).to(device)  # Discriminates sketch 'b3'

    # Generators: Input channels dynamically adapted within the model architecture
    vgg_path = './pretrained_weight/vgg19-dcbb9e9d.pth'

    # G1: Stage 1 (Face <-> Contour)
    G1_A2B = Generator(pretrained_path=vgg_path).to(device)
    G1_B2A = Generator(pretrained_path=vgg_path).to(device)

    # G2: Stage 2 (Face + Contour <-> Detail)
    G2_A2B = Generator(pretrained_path=vgg_path).to(device)
    G2_B2A = Generator(pretrained_path=vgg_path).to(device)

    # G3: Stage 3 (Face + Detail <-> Texture)
    G3_A2B = Generator(pretrained_path=vgg_path).to(device)
    G3_B2A = Generator(pretrained_path=vgg_path).to(device)

    # Model Grouping for easy iteration and freezing
    discriminators = [D1_0, D1_1, D2_0, D2_2, D3_0, D3_3]
    generators = [G1_A2B, G1_B2A, G2_A2B, G2_B2A, G3_A2B, G3_B2A]
    G1_group = [G1_A2B, G1_B2A]
    G2_group = [G2_A2B, G2_B2A]
    G3_group = [G3_A2B, G3_B2A]

    # ---------------------- Loss Functions ----------------------
    criterion_Adversarial = nn.MSELoss()  # Adversarial Loss (LSGAN)
    criterion_Cycle = nn.L1Loss()  # Cycle-consistency Loss
    criterion_Identity = nn.L1Loss()  # Identity Preservation Loss
    criterion_trans = nn.L1Loss()  # Transition Loss (constrains progressive sketching)

    # Loss Weights
    lambda_adv = config.lambda_adv if hasattr(config, 'lambda_adv') else 1.0
    lambda_cycle = config.lambda_cycle if hasattr(config, 'lambda_cycle') else 10.0
    lambda_id = config.lambda_identity if hasattr(config, 'lambda_identity') else 5.0

    # ---------------------- Optimizers ----------------------
    lr_base = config.lr_base if hasattr(config, 'lr_base') else 2e-4

    # Stage 1: G1 + D1
    G1_optim = torch.optim.Adam(chain(G1_A2B.parameters(), G1_B2A.parameters()), lr=lr_base, betas=(0.5, 0.999))
    D1_optim = torch.optim.Adam(chain(D1_0.parameters(), D1_1.parameters()), lr=lr_base * 0.25, betas=(0.5, 0.999))

    # Stage 2: G2 + D2
    G2_optim = torch.optim.Adam(chain(G2_A2B.parameters(), G2_B2A.parameters()), lr=lr_base, betas=(0.5, 0.999))
    D2_optim = torch.optim.Adam(chain(D2_0.parameters(), D2_2.parameters()), lr=lr_base, betas=(0.5, 0.999))

    # Stage 3: G3 + D3
    G3_optim = torch.optim.Adam(chain(G3_A2B.parameters(), G3_B2A.parameters()), lr=lr_base, betas=(0.5, 0.999))
    D3_optim = torch.optim.Adam(chain(D3_0.parameters(), D3_3.parameters()), lr=lr_base, betas=(0.5, 0.999))

    # Learning Rate Schedulers
    G1_scheduler = get_lr_scheduler(G1_optim)
    D1_scheduler = get_lr_scheduler(D1_optim)
    G2_scheduler = get_lr_scheduler(G2_optim)
    D2_scheduler = get_lr_scheduler(D2_optim)
    G3_scheduler = get_lr_scheduler(G3_optim)
    D3_scheduler = get_lr_scheduler(D3_optim)

    # Resume Training (Optional)
    resume = False
    start_epoch = config.starting_epoch if hasattr(config, 'starting_epoch') else 0
    if resume:
        path_checkpoint = "./results/checkpoints/model_030.pt"
        checkpoint = torch.load(path_checkpoint)
        for g in generators:
            g_key = g.__class__.__name__ + '_' + g._get_name()
            if g_key in checkpoint: g.load_state_dict(checkpoint[g_key])
        for d in discriminators:
            d_key = d.__class__.__name__ + '_' + d._get_name()
            if d_key in checkpoint: d.load_state_dict(checkpoint[d_key])
        G1_optim.load_state_dict(checkpoint['G1_optim'])
        D1_optim.load_state_dict(checkpoint['D1_optim'])

    # Loss Logging Dictionary
    loss_log = {
        'D1_B': [], 'D2_B': [], 'D3_B': [], 'D1_A': [], 'D2_A': [], 'D3_A': [],
        'G1': [], 'G2': [], 'G3': []
    }

    # ---------------------- Core Training Loop ----------------------
    total_epochs = config.total_epochs if hasattr(config, 'total_epochs') else STAGE3_EPOCH

    print(f"Training Initialized! Total Epochs: {total_epochs}")
    print(
        f"Stage 1: 0-{STAGE1_EPOCH - 1} (Train G1+D1: a <-> b1) | Batch Size: {config.batch_size if hasattr(config, 'batch_size') else 4}")
    print(
        f"Stage 2: {STAGE1_EPOCH}-{STAGE2_EPOCH - 1} (Train G2+D2: a+fake_b1 <-> b2) | Batch Size: {config.batch_size if hasattr(config, 'batch_size') else 4}")
    print(
        f"Stage 3: {STAGE2_EPOCH}-{STAGE3_EPOCH - 1} (Train G3+D3: a+fake_b2 <-> b3) | Batch Size: {config.batch_size if hasattr(config, 'batch_size') else 4}")

    for epoch in range(start_epoch, total_epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Determine current stage based on the epoch number
        if epoch < STAGE1_EPOCH:
            current_stage = 1
        elif epoch < STAGE2_EPOCH:
            current_stage = 2
        elif epoch < STAGE3_EPOCH:
            current_stage = 3
        else:
            break

        # Iterate through batched data
        data_iter = zip(train_face_loader, train_sketch1_loader, train_sketch2_loader, train_sketch3_loader)

        for i, (face, sketch1, sketch2, sketch3) in enumerate(data_iter):
            real_A = face.to(device)
            real_B1 = sketch1.to(device)
            real_B2 = sketch2.to(device)
            real_B3 = sketch3.to(device)

            # ---------------------- Stage 1: Train G1+D1 (a <-> b1) ----------------------
            if current_stage == 1:
                # Freeze other modules, activate G1 and D1
                set_requires_grad(generators, requires_grad=False)
                set_requires_grad(discriminators, requires_grad=False)
                set_requires_grad(G1_group, requires_grad=True)
                set_requires_grad([D1_0, D1_1], requires_grad=True)

                for _ in range(config.num_train_gen if hasattr(config, 'num_train_gen') else 1):
                    G1_optim.zero_grad()
                    fake_B1 = G1_A2B(real_A)
                    fake_A = G1_B2A(real_B1)

                    # Adversarial Loss
                    loss_adv_G1_A2B = criterion_Adversarial(D1_1(fake_B1), torch.ones_like(D1_1(fake_B1)).to(device))
                    loss_adv_G1_B2A = criterion_Adversarial(D1_0(fake_A), torch.ones_like(D1_0(fake_A)).to(device))

                    # Cycle Loss
                    recon_A = G1_B2A(fake_B1)
                    recon_B1 = G1_A2B(fake_A)
                    loss_cycle_G1_ABA = criterion_Cycle(recon_A, real_A)
                    loss_cycle_G1_BAB = criterion_Cycle(recon_B1, real_B1)

                    # Identity Loss
                    loss_id_G1 = criterion_Identity(G1_A2B(real_B1), real_B1)

                    # Total G1 Loss
                    G1_loss = lambda_adv * (loss_adv_G1_A2B + loss_adv_G1_B2A) + \
                              lambda_cycle * (loss_cycle_G1_ABA + loss_cycle_G1_BAB) + \
                              lambda_id * loss_id_G1
                    G1_loss.backward()
                    G1_optim.step()

                # Train D1
                D1_optim.zero_grad()
                loss_D1_real_B1 = criterion_Adversarial(D1_1(real_B1), torch.ones_like(D1_1(real_B1)).to(device))
                loss_D1_fake_B1 = criterion_Adversarial(D1_1(fake_B1.detach()),
                                                        torch.zeros_like(D1_1(fake_B1)).to(device))
                loss_D1_B = (loss_D1_real_B1 + loss_D1_fake_B1) * 0.5

                loss_D1_real_A = criterion_Adversarial(D1_0(real_A), torch.ones_like(D1_0(real_A)).to(device))
                loss_D1_fake_A = criterion_Adversarial(D1_0(fake_A.detach()), torch.zeros_like(D1_0(fake_A)).to(device))
                loss_D1_A = (loss_D1_real_A + loss_D1_fake_A) * 0.5

                D1_loss = loss_D1_A + loss_D1_B
                D1_loss.backward()
                D1_optim.step()

                loss_log['G1'].append(G1_loss.item())
                loss_log['D1_A'].append(loss_D1_A.item())
                loss_log['D1_B'].append(loss_D1_B.item())
                for k in ['G2', 'G3', 'D2_A', 'D2_B', 'D3_A', 'D3_B']:
                    loss_log[k].append(0.0)

            # ---------------------- Stage 2: Train G2+D2 (a+fake_b1 <-> b2) ----------------------
            elif current_stage == 2:
                # Freeze G1/D1 and G3/D3, activate G2 and D2
                set_requires_grad(generators, requires_grad=False)
                set_requires_grad(discriminators, requires_grad=False)
                set_requires_grad(G2_group, requires_grad=True)
                set_requires_grad([D2_0, D2_2], requires_grad=True)

                with torch.no_grad():
                    fake_B1 = G1_A2B(real_A)

                # Concatenate along channel dimension to form 6-channel input
                input_A2 = torch.cat([real_A, fake_B1], dim=1)
                input_B2 = torch.cat([real_B2, fake_B1], dim=1)

                for _ in range(config.num_train_gen if hasattr(config, 'num_train_gen') else 1):
                    G2_optim.zero_grad()
                    fake_B2 = G2_A2B(input_A2)
                    fake_A = G2_B2A(input_B2)

                    # Adversarial Loss
                    loss_adv_G2_A2B = criterion_Adversarial(D2_2(fake_B2), torch.ones_like(D2_2(fake_B2)).to(device))
                    loss_adv_G2_B2A = criterion_Adversarial(D2_0(fake_A), torch.ones_like(D2_0(fake_A)).to(device))

                    # Cycle Loss
                    recon_input_A2 = torch.cat([fake_A, fake_B1], dim=1)
                    recon_B2 = G2_A2B(recon_input_A2)
                    recon_input_B2 = torch.cat([fake_B2, fake_B1], dim=1)
                    recon_A = G2_B2A(recon_input_B2)

                    loss_cycle_G2_ABA = criterion_Cycle(recon_A, real_A)
                    loss_cycle_G2_BAB = criterion_Cycle(recon_B2, real_B2)

                    # Identity & Transition Loss
                    loss_id_G2 = criterion_Identity(fake_B2, real_B2)
                    loss_trans_G2 = criterion_trans(fake_B2, fake_B1)

                    # Total G2 Loss
                    G2_loss = lambda_adv * (loss_adv_G2_A2B + loss_adv_G2_B2A) + \
                              lambda_cycle * (loss_cycle_G2_ABA + loss_cycle_G2_BAB) + \
                              lambda_id * loss_id_G2 + 0.5 * lambda_id * loss_trans_G2
                    G2_loss.backward()
                    G2_optim.step()

                # Train D2
                D2_optim.zero_grad()
                loss_D2_real_B2 = criterion_Adversarial(D2_2(real_B2), torch.ones_like(D2_2(real_B2)).to(device))
                loss_D2_fake_B2 = criterion_Adversarial(D2_2(fake_B2.detach()),
                                                        torch.zeros_like(D2_2(fake_B2)).to(device))
                loss_D2_B = (loss_D2_real_B2 + loss_D2_fake_B2) * 0.5

                loss_D2_real_A = criterion_Adversarial(D2_0(real_A), torch.ones_like(D2_0(real_A)).to(device))
                loss_D2_fake_A = criterion_Adversarial(D2_0(fake_A.detach()), torch.zeros_like(D2_0(fake_A)).to(device))
                loss_D2_A = (loss_D2_real_A + loss_D2_fake_A) * 0.5

                D2_loss = loss_D2_A + loss_D2_B
                D2_loss.backward()
                D2_optim.step()

                loss_log['G2'].append(G2_loss.item())
                loss_log['D2_A'].append(loss_D2_A.item())
                loss_log['D2_B'].append(loss_D2_B.item())
                loss_log['G1'].append(loss_log['G1'][-1] if loss_log['G1'] else 0.0)
                loss_log['D1_A'].append(loss_log['D1_A'][-1] if loss_log['D1_A'] else 0.0)
                loss_log['D1_B'].append(loss_log['D1_B'][-1] if loss_log['D1_B'] else 0.0)
                for k in ['G3', 'D3_A', 'D3_B']:
                    loss_log[k].append(0.0)

            # ---------------------- Stage 3: Train G3+D3 (a+fake_b2 <-> b3) ----------------------
            elif current_stage == 3:
                # Freeze G1/G2 and D1/D2, activate G3 and D3
                set_requires_grad(generators, requires_grad=False)
                set_requires_grad(discriminators, requires_grad=False)
                set_requires_grad(G3_group, requires_grad=True)
                set_requires_grad([D3_0, D3_3], requires_grad=True)

                with torch.no_grad():
                    fake_B1 = G1_A2B(real_A)
                    input_G2 = torch.cat([real_A, fake_B1], dim=1)
                    fake_B2 = G2_A2B(input_G2)

                input_A3 = torch.cat([real_A, fake_B2], dim=1)
                input_B3 = torch.cat([real_B2, fake_B2], dim=1)

                for _ in range(config.num_train_gen if hasattr(config, 'num_train_gen') else 1):
                    G3_optim.zero_grad()
                    fake_B3 = G3_A2B(input_A3)
                    fake_A = G3_B2A(input_B3)

                    # Adversarial Loss
                    loss_adv_G3_A2B = criterion_Adversarial(D3_3(fake_B3), torch.ones_like(D3_3(fake_B3)).to(device))
                    loss_adv_G3_B2A = criterion_Adversarial(D3_0(fake_A), torch.ones_like(D3_0(fake_A)).to(device))

                    # Cycle Loss
                    recon_input_A3 = torch.cat([fake_A, fake_B2], dim=1)
                    recon_B3 = G3_A2B(recon_input_A3)
                    recon_input_B3 = torch.cat([fake_A, fake_B2], dim=1)
                    recon_A = G3_B2A(recon_input_B3)

                    loss_cycle_G3_ABA = criterion_Cycle(recon_A, real_A)
                    loss_cycle_G3_BAB = criterion_Cycle(recon_B3, real_B3)

                    # Identity & Transition Loss
                    loss_id_G3 = criterion_Identity(fake_B3, real_B3)
                    loss_trans_G3 = criterion_trans(fake_B3, fake_B2)

                    # Total G3 Loss
                    G3_loss = lambda_adv * (loss_adv_G3_A2B + loss_adv_G3_B2A) + \
                              lambda_cycle * (loss_cycle_G3_ABA + loss_cycle_G3_BAB) + \
                              lambda_id * loss_id_G3 + 0.5 * lambda_id * loss_trans_G3
                    G3_loss.backward()
                    G3_optim.step()

                # Train D3
                D3_optim.zero_grad()
                loss_D3_real_B3 = criterion_Adversarial(D3_3(real_B3), torch.ones_like(D3_3(real_B3)).to(device))
                loss_D3_fake_B3 = criterion_Adversarial(D3_3(fake_B3.detach()),
                                                        torch.zeros_like(D3_3(fake_B3)).to(device))
                loss_D3_B = (loss_D3_real_B3 + loss_D3_fake_B3) * 0.5

                loss_D3_real_A = criterion_Adversarial(D3_0(real_A), torch.ones_like(D3_0(real_A)).to(device))
                loss_D3_fake_A = criterion_Adversarial(D3_0(fake_A.detach()), torch.zeros_like(D3_0(fake_A)).to(device))
                loss_D3_A = (loss_D3_real_A + loss_D3_fake_A) * 0.5

                D3_loss = loss_D3_A + loss_D3_B
                D3_loss.backward()
                D3_optim.step()

                loss_log['G3'].append(G3_loss.item())
                loss_log['D3_A'].append(loss_D3_A.item())
                loss_log['D3_B'].append(loss_D3_B.item())
                for k in ['G1', 'G2', 'D1_A', 'D1_B', 'D2_A', 'D2_B']:
                    loss_log[k].append(loss_log[k][-1] if loss_log[k] else 0.0)

            # ---------------------- Print Logs ----------------------
            if (i + 1) % (config.print_every if hasattr(config, 'print_every') else 10) == 0:
                print(f"Stage {current_stage} | Epoch [{epoch + 1}/{total_epochs}] | Iter [{i + 1}/{total_batch}]")
                if current_stage == 1:
                    print(f" D1_B: {np.mean(loss_log['D1_B'][-10:]):.4f} | G1: {np.mean(loss_log['G1'][-10:]):.4f}")
                elif current_stage == 2:
                    print(f" D2_B: {np.mean(loss_log['D2_B'][-10:]):.4f} | G2: {np.mean(loss_log['G2'][-10:]):.4f}")
                elif current_stage == 3:
                    print(f" D3_B: {np.mean(loss_log['D3_B'][-10:]):.4f} | G3: {np.mean(loss_log['G3'][-10:]):.4f}")

        # ---------------------- Sample Visualization ----------------------
        save_img_epoch = config.save_every_image if hasattr(config, 'save_every_image') else 10
        if (epoch + 1) % save_img_epoch == 0:
            if current_stage == 1:
                sample_images(test_face_loader, test_sketch_loader1, None, None,
                              G1_A2B, None, None, G1_B2A, None, None,
                              epoch, config.samples_path, stage=1)
            elif current_stage == 2:
                sample_images(test_face_loader, test_sketch_loader1, test_sketch_loader2, None,
                              G1_A2B, G2_A2B, None, G1_B2A, G2_B2A, None,
                              epoch, config.samples_path, stage=2)
            elif current_stage == 3:
                sample_images(test_face_loader, test_sketch_loader1, test_sketch_loader2, test_sketch_loader3,
                              G1_A2B, G2_A2B, G3_A2B, G1_B2A, G2_B2A, G3_B2A,
                              epoch, config.samples_path, stage=3)

        # ---------------------- Learning Rate Scheduling ----------------------
        if current_stage == 1:
            G1_scheduler.step()
            D1_scheduler.step()
        elif current_stage == 2:
            G2_scheduler.step()
            D2_scheduler.step()
        elif current_stage == 3:
            G3_scheduler.step()
            D3_scheduler.step()

        # ---------------------- Save Models ----------------------
        val_every = config.val_every if hasattr(config, 'val_every') else 20
        if (epoch + 1) % val_every == 0:
            if current_stage == 1:
                torch.save(G1_A2B.state_dict(), os.path.join(config.weights_path, f'G1_A2B_{epoch + 1}.pkl'))
                torch.save(G1_B2A.state_dict(), os.path.join(config.weights_path, f'G1_B2A_{epoch + 1}.pkl'))
                torch.save(D1_0.state_dict(), os.path.join(config.weights_path, f'D1_0_{epoch + 1}.pkl'))
                torch.save(D1_1.state_dict(), os.path.join(config.weights_path, f'D1_1_{epoch + 1}.pkl'))
            elif current_stage == 2:
                torch.save(G2_A2B.state_dict(), os.path.join(config.weights_path, f'G2_A2B_{epoch + 1}.pkl'))
                torch.save(G2_B2A.state_dict(), os.path.join(config.weights_path, f'G2_B2A_{epoch + 1}.pkl'))
                torch.save(D2_0.state_dict(), os.path.join(config.weights_path, f'D2_0_{epoch + 1}.pkl'))
                torch.save(D2_2.state_dict(), os.path.join(config.weights_path, f'D2_2_{epoch + 1}.pkl'))
            elif current_stage == 3:
                torch.save(G3_A2B.state_dict(), os.path.join(config.weights_path, f'G3_A2B_{epoch + 1}.pkl'))
                torch.save(G3_B2A.state_dict(), os.path.join(config.weights_path, f'G3_B2A_{epoch + 1}.pkl'))
                torch.save(D3_0.state_dict(), os.path.join(config.weights_path, f'D3_0_{epoch + 1}.pkl'))
                torch.save(D3_3.state_dict(), os.path.join(config.weights_path, f'D3_3_{epoch + 1}.pkl'))

                # In the final epoch, save an extra set of weights named 'final' for easy inference
                if (epoch + 1) == STAGE3_EPOCH:
                    torch.save(G1_A2B.state_dict(), os.path.join(config.weights_path, 'G1_A2B_final.pkl'))
                    torch.save(G2_A2B.state_dict(), os.path.join(config.weights_path, 'G2_A2B_final.pkl'))
                    torch.save(G3_A2B.state_dict(), os.path.join(config.weights_path, 'G3_A2B_final.pkl'))

            # Plot loss curves
            plot_losses(current_stage, loss_log, epoch + 1, config.plots_path)

        # ---------------------- Save Checkpoints ----------------------
        model_save_every = config.model_save_every if hasattr(config, 'model_save_every') else 50
        if (epoch + 1) % model_save_every == 0:
            checkpoint = {
                'G1_A2B': G1_A2B.state_dict(), 'G1_B2A': G1_B2A.state_dict(),
                'G2_A2B': G2_A2B.state_dict(), 'G2_B2A': G2_B2A.state_dict(),
                'G3_A2B': G3_A2B.state_dict(), 'G3_B2A': G3_B2A.state_dict(),
                'D1_0': D1_0.state_dict(), 'D1_1': D1_1.state_dict(),
                'D2_0': D2_0.state_dict(), 'D2_2': D2_2.state_dict(),
                'D3_0': D3_0.state_dict(), 'D3_3': D3_3.state_dict(),
                'G1_optim': G1_optim.state_dict(), 'D1_optim': D1_optim.state_dict(),
                'G2_optim': G2_optim.state_dict(), 'D2_optim': D2_optim.state_dict(),
                'G3_optim': G3_optim.state_dict(), 'D3_optim': D3_optim.state_dict(),
                'G1_scheduler': G1_scheduler.state_dict(), 'D1_scheduler': D1_scheduler.state_dict(),
                'G2_scheduler': G2_scheduler.state_dict(), 'D2_scheduler': D2_scheduler.state_dict(),
                'G3_scheduler': G3_scheduler.state_dict(), 'D3_scheduler': D3_scheduler.state_dict(),
                'current_stage': current_stage, 'epoch': epoch + 1,
            }
            torch.save(checkpoint, os.path.join(config.checkpoints_path, f'model_{epoch + 1:03d}.pt'))

    print("Training successfully finished!")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()