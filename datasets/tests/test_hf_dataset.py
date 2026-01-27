"""Test script for loading Parquet waveform data as HuggingFace dataset."""

import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


def test_local_parquet(parquet_path: str):
    """Test loading a local parquet file."""
    print("=" * 60)
    print("Loading Parquet Dataset")
    print("=" * 60)

    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return None

    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"Samples: {len(ds)}")
    print(f"Columns: {list(ds.features.keys())}")
    return ds


def test_sample_access(ds):
    """Test accessing individual samples."""
    print("\n" + "=" * 60)
    print("Sample Access")
    print("=" * 60)

    sample = ds[0]

    print(f"Pick: {sample['phase_type']} at {sample['phase_time']}")
    print(f"Event: {sample['event_id']} (M{sample['event_magnitude']})")
    print(f"Station: {sample['network']}.{sample['station']}")
    print(f"SNR: {sample['snr']}")

    waveform = np.array(sample['waveform'], dtype=np.float32)
    print(f"Waveform: {waveform.shape}, range [{waveform.min():.1f}, {waveform.max():.1f}]")

    return sample, waveform


def test_filtering(ds):
    """Test filtering the dataset."""
    print("\n" + "=" * 60)
    print("Filtering")
    print("=" * 60)

    p_picks = ds.filter(lambda x: x['phase_type'] == 'P')
    s_picks = ds.filter(lambda x: x['phase_type'] == 'S')
    print(f"P picks: {len(p_picks)}, S picks: {len(s_picks)}")

    high_snr = ds.filter(lambda x: x['snr'] and x['snr'] > 10.0)
    print(f"SNR > 10: {len(high_snr)}")

    if any(ds['event_magnitude']):
        large = ds.filter(lambda x: x['event_magnitude'] and x['event_magnitude'] > 3.0)
        print(f"Magnitude > 3: {len(large)}")


def test_streaming(parquet_path: str):
    """Test streaming mode."""
    print("\n" + "=" * 60)
    print("Streaming Mode")
    print("=" * 60)

    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    for i, sample in enumerate(ds):
        if i >= 3:
            break
        print(f"  {sample['phase_type']}: {sample['event_id']} @ {sample['network']}.{sample['station']}")


def test_batch_loading(ds, batch_size: int = 16):
    """Test batch loading for ML training."""
    print("\n" + "=" * 60)
    print("Batch Loading")
    print("=" * 60)

    batch = ds[:batch_size]
    waveforms = np.array(batch['waveform'], dtype=np.float32)
    print(f"Batch shape: {waveforms.shape}")

    phase_types = batch['phase_type']
    phase_indices = np.array(batch['phase_index'])
    print(f"Phase types: {phase_types}")
    print(f"Phase indices: {phase_indices}")


def plot_waveform(sample, waveform, save_path: str = None):
    """Plot a sample waveform."""
    print("\n" + "=" * 60)
    print("Plotting")
    print("=" * 60)

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    t = np.arange(waveform.shape[1]) / 100.0  # 100 Hz

    for i, (ax, comp) in enumerate(zip(axes, ['E', 'N', 'Z'])):
        ax.plot(t, waveform[i], 'k', lw=0.5)
        ax.set_ylabel(comp)
        ax.grid(True, alpha=0.3)

    # Mark phase pick (always at phase_index)
    phase_time = sample['phase_index'] / 100.0
    color = 'b' if sample['phase_type'] == 'P' else 'r'
    for ax in axes:
        ax.axvline(phase_time, color=color, lw=1.5, label=sample['phase_type'])

    axes[0].legend(loc='upper right')
    axes[0].set_title(f"{sample['phase_type']} | Event: {sample['event_id']} | "
                      f"Station: {sample['network']}.{sample['station']} | SNR: {sample['snr']:.1f}")
    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parquet_path = "./SCEDC/dataset/2025/001.parquet"

    ds = test_local_parquet(parquet_path)
    if ds is None:
        exit(1)

    sample, waveform = test_sample_access(ds)
    test_filtering(ds)
    test_streaming(parquet_path)
    test_batch_loading(ds)
    plot_waveform(sample, waveform, save_path="test_waveform.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
