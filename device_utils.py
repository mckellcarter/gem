"""
Device utilities for supporting CUDA, MPS, and CPU devices.

This module provides utilities to automatically detect and use the best available
device (CUDA GPU, Apple Silicon MPS, or CPU).
"""

import torch
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


def get_device(gpu_id=0):
    """
    Get the best available device.

    Priority: CUDA > MPS > CPU

    Args:
        gpu_id: GPU ID to use (only relevant for CUDA)

    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't support device IDs like CUDA
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_device(gpu_id=0):
    """
    Set the device for the current process.

    For CUDA, this sets the active CUDA device.
    For MPS and CPU, this is a no-op.

    Args:
        gpu_id: GPU ID to use (only relevant for CUDA)

    Returns:
        torch.device: The device that was set
    """
    device = get_device(gpu_id)
    if device.type == 'cuda':
        torch.cuda.set_device(gpu_id)
    return device


def dict_to_device(ob, device):
    """
    Recursively move a dictionary of tensors to the specified device.

    Args:
        ob: Object to move (can be dict or tensor)
        device: Target device

    Returns:
        Object moved to device
    """
    if isinstance(ob, Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def is_distributed_available():
    """
    Check if distributed training is available.

    Returns:
        bool: True if distributed training can be used
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def print_device_info():
    """Print information about available devices."""
    print("=" * 60)
    print("Device Information:")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
    else:
        print("CUDA Available: No")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) Available: Yes")
    else:
        print("MPS (Apple Silicon) Available: No")

    device = get_device()
    print(f"\nUsing Device: {device}")
    print("=" * 60)
