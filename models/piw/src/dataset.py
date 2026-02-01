"""
Dataset utilities for PIW training.

Supports LibriSpeech for pretraining and Helsinki Prosody for finetuning.
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset for self-supervised pretraining.
    
    Args:
        root (str): Path to LibriSpeech root directory
        subset (str): Subset name (e.g., 'train-clean-100')
        max_hours (float): Maximum hours of data to use
        sample_rate (int): Target sample rate
        max_length (float): Maximum audio length in seconds
        min_length (float): Minimum audio length in seconds
    """
    
    def __init__(
        self,
        root: str,
        subset: str = "train-clean-100",
        max_hours: Optional[float] = None,
        sample_rate: int = 16000,
        max_length: float = 10.0,
        min_length: float = 1.0,
    ):
        self.root = Path(root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)
        self.min_samples = int(min_length * sample_rate)
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        
        # Limit by hours if specified
        if max_hours is not None:
            self.audio_files = self._limit_by_hours(max_hours)
        
        print(f"Loaded {len(self.audio_files)} files from {subset}")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all FLAC files in the subset directory."""
        subset_dir = self.root / self.subset
        if not subset_dir.exists():
            # Try HuggingFace datasets cache format
            return self._find_hf_files()
        
        files = list(subset_dir.rglob("*.flac"))
        return sorted(files)
    
    def _find_hf_files(self) -> List[Path]:
        """Placeholder for HuggingFace datasets integration."""
        # Will be populated when using datasets library
        return []
    
    def _limit_by_hours(self, max_hours: float) -> List[Path]:
        """Limit dataset to specified number of hours."""
        max_seconds = max_hours * 3600
        total_seconds = 0
        selected = []
        
        # Shuffle for random selection
        shuffled = random.sample(self.audio_files, len(self.audio_files))
        
        for f in shuffled:
            try:
                info = torchaudio.info(str(f))
                duration = info.num_frames / info.sample_rate
                
                if total_seconds + duration <= max_seconds:
                    selected.append(f)
                    total_seconds += duration
                
                if total_seconds >= max_seconds:
                    break
            except Exception:
                continue
        
        print(f"Selected {len(selected)} files ({total_seconds/3600:.2f}h)")
        return selected
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)  # (T,)
        
        # Truncate or pad
        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        elif waveform.shape[0] < self.min_samples:
            # Skip too short samples
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            "waveform": waveform,
            "length": waveform.shape[0],
        }


class HelsinkiProsodyDataset(Dataset):
    """
    Helsinki Prosody Corpus for prominence detection finetuning.
    
    Args:
        root (str): Path to dataset root
        split (str): 'train', 'val', or 'test'
        sample_rate (int): Target sample rate
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        
        # Load file list
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self) -> List[Dict]:
        """Load samples with audio paths and labels."""
        samples = []
        
        # Expected format: root/split/audio/*.wav, root/split/labels/*.txt
        audio_dir = self.root / self.split / "audio"
        label_dir = self.root / self.split / "labels"
        
        if not audio_dir.exists():
            print(f"Warning: {audio_dir} not found")
            return samples
        
        for audio_file in audio_dir.glob("*.wav"):
            label_file = label_dir / f"{audio_file.stem}.txt"
            
            if label_file.exists():
                samples.append({
                    "audio": audio_file,
                    "label": label_file,
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(str(sample["audio"]))
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        # Load labels (frame-level binary)
        with open(sample["label"], "r") as f:
            labels = [int(x.strip()) for x in f.readlines()]
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            "waveform": waveform,
            "labels": labels,
            "length": waveform.shape[0],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length audio."""
    max_len = max(item["waveform"].shape[0] for item in batch)
    
    waveforms = []
    lengths = []
    attention_masks = []
    
    for item in batch:
        wav = item["waveform"]
        length = wav.shape[0]
        
        # Pad to max length
        if length < max_len:
            wav = torch.nn.functional.pad(wav, (0, max_len - length))
        
        waveforms.append(wav)
        lengths.append(length)
        
        # Create attention mask
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[length:] = False
        attention_masks.append(mask)
    
    result = {
        "waveform": torch.stack(waveforms),
        "lengths": torch.tensor(lengths),
        "attention_mask": torch.stack(attention_masks),
    }
    
    # Add labels if present
    if "labels" in batch[0]:
        max_label_len = max(item["labels"].shape[0] for item in batch)
        labels = []
        for item in batch:
            lbl = item["labels"]
            if lbl.shape[0] < max_label_len:
                lbl = torch.nn.functional.pad(lbl, (0, max_label_len - lbl.shape[0]))
            labels.append(lbl)
        result["labels"] = torch.stack(labels)
    
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader with proper collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
