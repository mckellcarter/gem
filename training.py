'''Implements a generic training loop.
'''

import loss_functions
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from collections import defaultdict
import torch.distributed as dist
from utils import get_mgrid
import device_utils
from datetime import datetime
from pathlib import Path
import glob
import json


def dict_to_gpu(ob, device=None):
    """Legacy function name for backward compatibility. Use dict_to_device instead."""
    if device is None:
        device = device_utils.get_device()
    return device_utils.dict_to_device(ob, device)


# ============================================================================
# Checkpoint Management Utilities
# ============================================================================

def save_checkpoint(checkpoint_dir, epoch, total_steps, model, ema_model, optimizers,
                    train_losses, config=None, val_loss=None, is_best=False):
    """
    Save a comprehensive training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        total_steps: Total training steps completed
        model: Model to save
        ema_model: EMA model (can be None)
        optimizers: List of optimizers
        train_losses: List of training losses
        config: Optional training configuration dict
        val_loss: Optional validation loss
        is_best: Whether this is the best checkpoint so far

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'total_steps': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'train_losses': train_losses,
        'timestamp': datetime.now().isoformat(),
    }

    # Add EMA model state if available
    if ema_model is not None:
        shadow_params = ema_model.shadow_params
        ema_dict = {}
        named_model_params = list(model.named_parameters())
        for (k, v), param in zip(named_model_params, shadow_params):
            ema_dict[k] = param
        checkpoint['ema_state_dict'] = ema_dict

    # Add config if provided
    if config is not None:
        checkpoint['config'] = config

    # Add validation loss if provided
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss

    # Save checkpoint with step number in filename
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}_step_{total_steps:06d}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save as latest checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)

    # Save as best checkpoint if applicable
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)

    # Update checkpoint manifest
    _update_checkpoint_manifest(checkpoint_dir, checkpoint_path, epoch, total_steps, val_loss)

    print(f"[Checkpoint] Saved checkpoint at epoch {epoch}, step {total_steps} to {checkpoint_path}")

    return str(checkpoint_path)


def load_checkpoint(checkpoint_path, model, ema_model=None, optimizers=None, device=None):
    """
    Load a checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        ema_model: Optional EMA model
        optimizers: Optional list of optimizers to restore
        device: Device to load checkpoint to

    Returns:
        Dictionary containing: epoch, total_steps, train_losses, config
    """
    if device is None:
        device = device_utils.get_device()

    print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}")

    # Load checkpoint (weights_only=False is needed for loading optimizer states and complex objects)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect old checkpoint format (just model weights)
    if 'model_state_dict' not in checkpoint:
        print("[Checkpoint] Detected old checkpoint format (model weights only)")

        # Old format may have 'model_dict' key or be the state dict directly
        if 'model_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_dict'])
        else:
            model.load_state_dict(checkpoint)

        return {
            'epoch': 0,
            'total_steps': 0,
            'train_losses': [],
            'config': None
        }

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Checkpoint] Loaded model state from epoch {checkpoint['epoch']}, step {checkpoint['total_steps']}")

    # Load EMA model state if available
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_dict = checkpoint['ema_state_dict']
        # Restore EMA shadow params
        shadow_params = []
        for k, v in model.named_parameters():
            if k in ema_dict:
                shadow_params.append(ema_dict[k])
            else:
                shadow_params.append(v.clone().detach())
        ema_model.shadow_params = shadow_params
        print("[Checkpoint] Loaded EMA model state")

    # Load optimizer states if available
    if optimizers is not None and 'optimizer_states' in checkpoint:
        optimizer_states = checkpoint['optimizer_states']
        for i, (opt, state) in enumerate(zip(optimizers, optimizer_states)):
            opt.load_state_dict(state)
        print(f"[Checkpoint] Loaded {len(optimizers)} optimizer states")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'total_steps': checkpoint.get('total_steps', 0),
        'train_losses': checkpoint.get('train_losses', []),
        'config': checkpoint.get('config', None),
        'val_loss': checkpoint.get('val_loss', None),
    }


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the most recent checkpoint in a directory by step number.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    # First, find checkpoint with highest step number from numbered checkpoints
    checkpoint_pattern = str(checkpoint_dir / 'checkpoint_epoch_*_step_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)

    if checkpoints:
        # Extract step number from filename and find checkpoint with highest step
        def extract_step_number(path):
            try:
                # Filename format: checkpoint_epoch_XXXX_step_XXXXXX.pth
                filename = Path(path).stem
                step_part = filename.split('_step_')[-1]
                return int(step_part)
            except:
                return 0

        latest = max(checkpoints, key=extract_step_number)
        print(f"[Checkpoint] Found checkpoint with highest step number: {Path(latest).name}")
        return latest

    # If no numbered checkpoints, check for checkpoint_latest.pth
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    if latest_path.exists():
        print(f"[Checkpoint] Using checkpoint_latest.pth")
        return str(latest_path)

    # Finally, check for old format checkpoints
    old_pattern = str(checkpoint_dir / 'model_*.pth')
    old_checkpoints = glob.glob(old_pattern)
    if old_checkpoints:
        # Extract step numbers and find latest
        def extract_step(path):
            try:
                return int(Path(path).stem.split('_')[-1])
            except:
                return 0
        latest = max(old_checkpoints, key=extract_step)
        print(f"[Checkpoint] Using old format checkpoint: {Path(latest).name}")
        return latest

    return None


def list_checkpoints(checkpoint_dir):
    """
    List all checkpoints in a directory with metadata.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of dictionaries with checkpoint information
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / 'checkpoint_manifest.json'

    # If manifest exists, read it
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest.get('checkpoints', [])
        except:
            pass

    # Otherwise scan directory
    checkpoint_pattern = str(checkpoint_dir / 'checkpoint_epoch_*_step_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)

    checkpoint_info = []
    for ckpt_path in sorted(checkpoints):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            info = {
                'path': ckpt_path,
                'epoch': ckpt.get('epoch', 0),
                'total_steps': ckpt.get('total_steps', 0),
                'timestamp': ckpt.get('timestamp', ''),
                'val_loss': ckpt.get('val_loss', None),
            }
            checkpoint_info.append(info)
        except:
            pass

    return checkpoint_info


def _update_checkpoint_manifest(checkpoint_dir, checkpoint_path, epoch, total_steps, val_loss=None):
    """
    Update the checkpoint manifest file.

    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_path: Path to the new checkpoint
        epoch: Epoch number
        total_steps: Total steps
        val_loss: Optional validation loss
    """
    manifest_path = Path(checkpoint_dir) / 'checkpoint_manifest.json'

    # Load existing manifest or create new one
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except:
            manifest = {'checkpoints': []}
    else:
        manifest = {'checkpoints': []}

    # Add new checkpoint entry
    checkpoint_entry = {
        'path': str(checkpoint_path),
        'epoch': epoch,
        'total_steps': total_steps,
        'timestamp': datetime.now().isoformat(),
    }

    if val_loss is not None:
        checkpoint_entry['val_loss'] = float(val_loss)

    manifest['checkpoints'].append(checkpoint_entry)

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def multiscale_training(model, ema_model, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
                        dataloader_callback, dataloader_iters, dataloader_params,
                        val_loss_fn=None, summary_fn=None, iters_til_checkpoint=None, clip_grad=False,
                        overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, device=None,
                        resume_from_checkpoint=None, auto_resume=True):

    if device is None:
        device = device_utils.get_device()

    model_dir_base = model_dir
    for i in range(1000):
        for params, max_steps in zip(dataloader_params, dataloader_iters):
            train_dataloader, val_dataloader = dataloader_callback(*params)
            model_dir = os.path.join(model_dir_base, '_'.join(map(str, params)))

            model, optimizers = train(model, ema_model, train_dataloader, epochs=10000, lr=lr, steps_til_summary=steps_til_summary,
                                      val_dataloader=val_dataloader, epochs_til_checkpoint=epochs_til_checkpoint, model_dir=model_dir, loss_fn=loss_fn,
                                      val_loss_fn=val_loss_fn, summary_fn=summary_fn, iters_til_checkpoint=iters_til_checkpoint,
                                      clip_grad=clip_grad, overwrite=overwrite, optimizers=optimizers, batches_per_validation=batches_per_validation,
                                      gpus=gpus, rank=rank, max_steps=max_steps, device=device,
                                      resume_from_checkpoint=resume_from_checkpoint, auto_resume=auto_resume)


def train(model, ema_model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None, device=None,
          resume_from_checkpoint=None, auto_resume=True):

    if device is None:
        device = device_utils.get_device()

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]
    # schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200) for optimizer in optimizers]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        # if os.path.exists(model_dir):
        #     if overwrite:
        #         shutil.rmtree(model_dir)
        #     else:
        #         val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        #         if val == 'y' or overwrite:
        #             shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    # Checkpoint resuming logic
    start_epoch = 0
    total_steps = 0
    train_losses = []

    if rank == 0:
        # Auto-resume from latest checkpoint if requested
        if auto_resume and resume_from_checkpoint is None:
            resume_from_checkpoint = find_latest_checkpoint(checkpoints_dir)

        # Load checkpoint if available
        if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
            checkpoint_info = load_checkpoint(
                resume_from_checkpoint,
                model=model,
                ema_model=ema_model,
                optimizers=optimizers,
                device=device
            )
            start_epoch = checkpoint_info['epoch']
            total_steps = checkpoint_info['total_steps']
            train_losses = checkpoint_info['train_losses']

            print(f"[Training] Resuming from epoch {start_epoch}, step {total_steps}")
        elif resume_from_checkpoint is not None:
            print(f"[Training] Warning: Checkpoint {resume_from_checkpoint} not found, starting fresh")

    # Broadcast start_epoch and total_steps to all ranks for multi-GPU
    if gpus > 1:
        # Convert to tensor for broadcast
        start_info = torch.tensor([start_epoch, total_steps], dtype=torch.long).to(device)
        dist.broadcast(start_info, src=0)
        start_epoch = int(start_info[0].item())
        total_steps = int(start_info[1].item())

    print("len data loader: ", len(train_dataloader), " w/ epochs: ", len(train_dataloader) * epochs)

    # Track validation loss for checkpointing
    current_val_loss = None

    # Calculate initial progress for tqdm based on total_steps already completed
    # This makes the progress bar start from where we left off when resuming
    initial_progress = total_steps if total_steps > 0 else (start_epoch * len(train_dataloader))

    with tqdm(total=len(train_dataloader) * epochs, initial=initial_progress) as pbar:
        for epoch in range(start_epoch, epochs):

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_gpu(model_input, device)
                gt = dict_to_gpu(gt, device)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    # torch.save(model.state_dict(),
                    #            os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                ema_model.update(model.parameters())

                # for scheduler in schedulers:
                #     scheduler.step()

                if rank == 0:
                    pbar.update(1)

                # if total_steps % 500 == 0:
                #     optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    # Run validation if available
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = dict_to_gpu(model_input, device)
                                gt = dict_to_gpu(gt, device)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                                # Track validation loss for checkpointing
                                if loss_name == 'img_loss' or 'total' in loss_name.lower():
                                    current_val_loss = single_loss

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    # Save comprehensive checkpoint using new checkpoint system
                    save_checkpoint(
                        checkpoint_dir=checkpoints_dir,
                        epoch=epoch,
                        total_steps=total_steps,
                        model=model,
                        ema_model=ema_model,
                        optimizers=optimizers,
                        train_losses=train_losses,
                        val_loss=current_val_loss
                    )

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        # if rank == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(checkpoints_dir, 'model_final.pth'))
        #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #                np.array(train_losses))

        return model, optimizers


def train_autodecoder_gan(model, discriminator, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None, device=None):

    if device is None:
        device = device_utils.get_device()

    model_optim, disc_optim = [torch.optim.Adam(lr=lr, params=model.parameters()),
            torch.optim.Adam(lr=lr, params=discriminator.parameters(), betas=(0.0, 0.9))]

    # schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200) for optimizer in optimizers]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    print("len data loader: ", len(train_dataloader), " w/ epochs: ", len(train_dataloader) * epochs)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_gpu(model_input, device)
                gt = dict_to_gpu(gt, device)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    # torch.save(model.state_dict(),
                    #            os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)


                # Fake forward pass
                fake_model_output = model(model_input, prior_sample=True, render=False, manifold_model=False)

                # Real forward pass
                real_model_output = model(model_input, render=False, manifold_model=False)

                pred_fake = discriminator(fake_model_output, detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(real_model_output, detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)

                if rank == 0:
                    writer.add_scalar('disc_fake', loss_d_fake, total_steps)
                    writer.add_scalar('disc_real', loss_d_real, total_steps)
                    writer.add_scalar('discriminator_loss', disc_loss, total_steps)

                discriminator.requires_grad_(False)
                pred_fake = discriminator(fake_model_output, detach=False)
                discriminator.requires_grad_(True)
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)

                if rank == 0:
                    writer.add_scalar('gan_gen', loss_g_gan_fake, total_steps)

                disc_loss = disc_loss + 0.01 * loss_g_gan_fake

                batch_size = real_model_output['representation'].shape[0]
                alpha = torch.rand(batch_size, 1).to(device)
                interpolated = alpha * real_model_output['representation'].data + (1 - alpha) * fake_model_output['representation'].data
                interpolated = interpolated.requires_grad_(True)

                # Calculate probability of interpolated examples
                prob_interpolated = discriminator({'representation':interpolated})
                input_list = [interpolated]

                # Calculate gradients of probabilities with respect to examples
                gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                                create_graph=True, retain_graph=True, allow_unused=True)

                gradients = [g for g in gradients if g is not None]
                gradients = torch.cat(gradients, dim=-1)
                # Gradients have shape (batch_size, num_channels, img_width, img_height),
                # so flatten to easily take norm per example in batch
                gradients = gradients.view(batch_size, -1)

                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                gradient_penalty = ((gradients_norm - 1) ** 2).mean()

                disc_loss += gradient_penalty

                if rank == 0:
                    writer.add_scalar('gradient_penalty', gradient_penalty, total_steps)

                model_optim.zero_grad()
                disc_optim.zero_grad()

                disc_loss.backward()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)
                    average_gradients(discriminator)

                disc_optim.step()
                model_optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = dict_to_gpu(model_input, device)
                                gt = dict_to_gpu(gt, device)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save({'model_dict': model.state_dict(), 'disc_dict': discriminator.state_dict()},
                               os.path.join(checkpoints_dir, 'model_{}.pth'.format(total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        # if rank == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(checkpoints_dir, 'model_final.pth'))
        #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #                np.array(train_losses))

        return model, optimizers


def train_latent_gan(model, discriminator, data_loader, epochs, lr, steps_til_summary, model_dir,
                     loss_fn, summary_fn=None, iters_til_checkpoint=None, overwrite=True, optimizers=None, val_loader=None,
                     gt_model=None, gradient_penalty=False, r1_loss=True, real_reconstruction_loss=True, gpus=1, rank=0, val_dataset=None,
                     multimodal=False, device=None):

    if device is None:
        device = device_utils.get_device()

    if optimizers is None:
        model_optim, disc_optim = [torch.optim.Adam(lr=lr, params=model.generator.parameters(), betas=(0.0, 0.9)),
                                   torch.optim.Adam(lr=lr, params=discriminator.parameters(), betas=(0.0, 0.9))]

    if gt_model is None:
        gt_model = model


    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0

    size = 64
    mgrid = utils.get_mgrid((size, size), dim=2)
    print("pre img mgrid: ", mgrid.shape)
    mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    print("post img mgrid: ", mgrid.shape)

    if multimodal:
        img_mgrid = data_loader.dataset.real_dataset.dataset.img_mgrid
        img_coords = np.concatenate([img_mgrid, torch.zeros(img_mgrid.shape[0], 1)], axis=1)

        audio_mgrid = data_loader.dataset.real_dataset.dataset.audio_mgrid
        audio_coords = np.concatenate([torch.zeros(audio_mgrid.shape[0],2), audio_mgrid], axis=1) * 100  # account for diff in frequency

        mgrid = torch.from_numpy(np.vstack([img_coords, audio_coords]))

        print("mgrid: ", mgrid.shape)

        mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    else:
        mgrid = utils.get_mgrid((size, size), dim=2)
        mgrid = mgrid[None, :, :].repeat(16, 1, 1)
    mgrid = mgrid.to(device)

    with tqdm(total=len(data_loader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, ((fake_model_input, fake_gt), (real_model_input, real_gt)) in enumerate(data_loader):
                if ((total_steps % 10000) == 0) and rank == 0:
                    from metrics import compute_fid_nn
                    if not multimodal:
                        fid_score, panel_im, panel_im_latent = compute_fid_nn(val_dataset, model, rank)
                        writer.add_scalar('fid_score', fid_score, total_steps)
                        writer.add_image('nn', panel_im,
                                         global_step=total_steps, dataformats="HWC")
                        writer.add_image('nn_latent', panel_im_latent,
                                         global_step=total_steps, dataformats="HWC")
                model_loss = 0.

                fake_model_input = dict_to_gpu(fake_model_input, device)
                real_model_input = dict_to_gpu(real_model_input, device)
                real_gt = dict_to_gpu(real_gt, device)
                fake_gt = dict_to_gpu(fake_gt, device)

                # Fake forward pass
                fake_model_output = model(fake_model_input, prior_sample=True, render=False, manifold_model=False)

                # Real forward pass
                real_model_output = gt_model(real_model_input, render=real_reconstruction_loss, manifold_model=False)

                if real_reconstruction_loss:
                    real_losses = loss_fn(real_model_output, real_gt)

                    for loss_name, loss in real_losses.items():
                        single_loss = loss.mean()

                        writer.add_scalar("real_" + loss_name, single_loss, total_steps)
                        model_loss += single_loss

                # Discriminator forward passes
                # Fake forward step
                pred_fake = discriminator(fake_model_output, detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(real_model_output, detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)

                # Gradient penalty
                if gradient_penalty:
                    # Calculate interpolation
                    batch_size = real_model_output['representation'].shape[0]
                    alpha = torch.rand(batch_size, 1).to(device)
                    interpolated = alpha * real_model_output['representation'].data + (1 - alpha) * fake_model_output['representation'].data
                    interpolated = interpolated.requires_grad_(True)

                    representations = [alpha * rep.clone().detach() + (1-alpha) * rep_neg.clone().detach() for rep, rep_neg in zip(real_model_output['representations'], fake_model_output['representations'])]
                    representations = [rep.requires_grad_(True) for rep in representations]

                    # Calculate probability of interpolated examples
                    prob_interpolated = discriminator({'representation':interpolated, 'representations': representations})
                    input_list = [interpolated] + representations

                    # Calculate gradients of probabilities with respect to examples
                    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                                    create_graph=True, retain_graph=True, allow_unused=True)

                    gradients = [g for g in gradients if g is not None]
                    gradients = torch.cat(gradients, dim=-1)
                    # Gradients have shape (batch_size, num_channels, img_width, img_height),
                    # so flatten to easily take norm per example in batch
                    gradients = gradients.view(batch_size, -1)

                    # Derivatives of the gradient close to 0 can cause problems because of
                    # the square root, so manually calculate norm and add epsilon
                    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

                    disc_loss += gradient_penalty

                    if rank == 0:
                        writer.add_scalar('gradient_penalty', gradient_penalty, total_steps)

                if r1_loss:
                    # Calculate interpolation
                    batch_size = real_model_output['representation'].shape[0]
                    alpha = torch.rand(batch_size, 1).to(device)
                    interpolated = real_model_output['representation'].clone().detach().requires_grad_()
                    representations = [rep.clone().detach().requires_grad_() for rep in real_model_output['representations']]

                    # Calculate probability of interpolated examples
                    prob_interpolated = discriminator({'representation':interpolated, 'representations': representations})
                    input_list = [interpolated] + representations

                    # Calculate gradients of probabilities with respect to examples
                    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=input_list,
                                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                                    create_graph=True, retain_graph=True)

                    gradients = torch.cat(gradients, dim=-1)

                    # Gradients have shape (batch_size, num_channels, img_width, img_height),
                    # so flatten to easily take norm per example in batch
                    gradients = gradients.view(batch_size, -1)
                    r1_loss = gradients.pow(2).sum(dim=1).mean()

                    disc_loss += r1_loss

                    if rank == 0:
                        writer.add_scalar('rl_loss', r1_loss, total_steps)

                if rank == 0:
                    writer.add_scalar('disc_fake', loss_d_fake, total_steps)
                    writer.add_scalar('disc_real', loss_d_real, total_steps)
                    writer.add_scalar('discriminator_loss', disc_loss, total_steps)


                disc_loss.backward()

                if gpus > 1:
                    average_gradients(discriminator)
                #     # average_gradients(discriminator)

                disc_optim.step()
                disc_optim.zero_grad()

                # Generator forward pass
                # Try to fake discriminator
                pred_fake = discriminator(fake_model_output, detach=False)
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)

                if rank == 0:
                    writer.add_scalar('generator_loss_fake', loss_g_gan_fake, total_steps)

                model_loss += loss_g_gan_fake

                model_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                model_optim.step()
                model_optim.zero_grad()

                if not total_steps % steps_til_summary and rank == 0:
                    with torch.no_grad():
                        fake_model_input['context']['x'] = mgrid
                        fake_model_input['context']['idx'] = fake_model_input['context']['idx'][:16]
                        real_model_input['context']['x'] = mgrid
                        real_model_input['context']['idx'] = real_model_input['context']['idx'][:16]
                        fake_model_output = model(fake_model_input, prior_sample=True, render=True)
                        real_model_output = gt_model(real_model_input, render=True)
                        torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, fake_model_input, fake_gt, fake_model_output, writer, total_steps, prefix='fake_')

                    writer.add_histogram('fake_representation', fake_model_output['representation'], total_steps)
                    writer.add_histogram('real_representation', real_model_output['representation'], total_steps)

                    summary_fn(model, real_model_input, real_gt, real_model_output, writer, total_steps, prefix='real_')

                if rank == 0:
                    pbar.update(1)

                print("done with step")

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f" % (epoch, model_loss+disc_loss))

                    if val_loader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_loader):
                                model_input = dict_to_gpu(model_input)
                                gt = dict_to_gpu(gt)

                                model_output = model(model_input, prior_sample=True)
                                val_loss = loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())
                                break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    # pass
                    torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                               os.path.join(checkpoints_dir, 'model_latest.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1

        if rank == 0:
            torch.save({'model_state_dict': model.state_dict(), 'discriminator_state_dict': discriminator.state_dict},
                       os.path.join(checkpoints_dir, 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                       np.array(train_losses))


def train_conv_gan(model, discriminator, data_loader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
                   summary_fn=None, iters_til_checkpoint=None, overwrite=True, optimizers=None, val_loader=None, device=None):

    if device is None:
        device = device_utils.get_device()

    if optimizers is None:
        model_optim, disc_optim = [torch.optim.Adam(lr=lr, betas=(0., 0.9), params=model.parameters()),
                                   torch.optim.Adam(lr=lr, betas=(0., 0.9), params=discriminator.parameters())]

    if os.path.exists(model_dir):
        if overwrite:
            shutil.rmtree(model_dir)
        else:
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
            if val == 'y' or overwrite:
                shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(data_loader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(data_loader):
                model_loss = 0.

                # Fake forward pass
                model_input = dict_to_gpu(model_input, device)
                gt = dict_to_gpu(gt, device)

                fake_model_output = model(model_input)

                # Discriminator forward passes
                # Fake forward step
                pred_fake = discriminator(fake_model_output['rgb'], detach=True)  # Detach to make sure no gradients go into generator
                loss_d_fake = loss_functions.gan_loss(pred_fake, False)

                # Real forward step
                pred_real = discriminator(gt['rgb'], detach=True)
                loss_d_real = loss_functions.gan_loss(pred_real, True)

                disc_loss = 0.5 * (loss_d_real + loss_d_fake)
                writer.add_scalar('discriminator_loss', disc_loss, total_steps)

                disc_loss.backward()
                disc_optim.step()
                disc_optim.zero_grad()

                # Generator forward pass
                # Try to fake discriminator
                # fake_model_output_det_z = model(fake_model_input, real=False, detach_z=True, render=False)
                pred_fake = discriminator(fake_model_output['rgb'])
                loss_g_gan_fake = loss_functions.gan_loss(pred_fake, True)
                writer.add_scalar('generator_loss_fake', loss_g_gan_fake, total_steps)

                generator_loss = loss_g_gan_fake

                generator_loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                disc_optim.zero_grad()

                writer.add_scalar("gen_loss", generator_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, fake_model_output, writer, total_steps, prefix='fake_')

                pbar.update(1)

                print("done with step")

                if not total_steps % steps_til_summary:
                    print("Epoch %d, Total loss %0.6f" % (epoch, model_loss+disc_loss))

                    if val_loader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_loader):
                                model_input = dict_to_gpu(model_input, device)
                                gt = dict_to_gpu(gt, device)

                                model_output = model(model_input, prior_sample=True, real=False)
                                break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint):
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
