"""
ACDC Cine MRI Dataset Loader
==========================================
Handles:
  - Loading 4D NIfTI volumes (H x W x D x T)
  - Odd-frame selection from diastole → peak systole
  - Myocardium mask extraction (label=2 in ACDC ground truth)
  - Per-patient normalization

ACDC Class Labels (from Info.cfg):
  NOR  = Normal (healthy)
  DCM  = Dilated Cardiomyopathy
  HCM  = Hypertrophic Cardiomyopathy
  MINF = Myocardial Infarction
  RV   = Abnormal Right Ventricle
"""

import os
import re
import configparser
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import warnings

# ── Label mappings ──────────────────────────────────────────────────────────
CLASS_TO_IDX = {
    "NOR":  0,
    "DCM":  1,
    "HCM":  2,
    "MINF": 3,
    "RV":   4,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

MYOCARDIUM_LABEL = 2  # label value in _gt.nii.gz files


# ── Utility: load NIfTI ─────────────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return its data as a numpy array."""
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


# ── Utility: parse Info.cfg ─────────────────────────────────────────────────

def parse_patient_info(patient_dir: str) -> Dict:
    """
    Parse the Info.cfg file in a patient directory.

    Returns a dict with keys:
        group       : str  — NOR / DCM / HCM / MINF / RV
        ed_frame    : int  — End-diastolic frame index (0-based after we convert)
        es_frame    : int  — End-systolic frame index  (0-based after we convert)
        height_mm   : float
        weight_kg   : float
        nb_frames   : int
    """
    cfg_path = os.path.join(patient_dir, "Info.cfg")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Info.cfg not found in {patient_dir}")

    info = {}
    with open(cfg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()

    return {
        "group":      info.get("Group", "UNK"),
        # ACDC frames are 1-indexed in Info.cfg → convert to 0-indexed
        "ed_frame":   int(info.get("ED", 1)) - 1,
        "es_frame":   int(info.get("ES", 1)) - 1,
        "height_mm":  float(info.get("Height", 0.0)),
        "weight_kg":  float(info.get("Weight", 0.0)),
        "nb_frames":  int(info.get("NbFrame", 0)),
    }


# ── Utility: find 4D file ───────────────────────────────────────────────────

def find_4d_file(patient_dir: str, patient_id: str) -> str:
    """
    Find the 4D cine MRI NIfTI file for a patient.
    ACDC naming: patientXXX_4d.nii.gz
    """
    candidate = os.path.join(patient_dir, f"{patient_id}_4d.nii.gz")
    if os.path.exists(candidate):
        return candidate
    # Fallback: search for any *_4d.nii.gz
    for fname in os.listdir(patient_dir):
        if "_4d.nii.gz" in fname:
            return os.path.join(patient_dir, fname)
    raise FileNotFoundError(f"No 4D NIfTI found in {patient_dir}")


def find_gt_file(patient_dir: str, patient_id: str, frame_idx: int) -> Optional[str]:
    """
    Find ground truth segmentation for a specific frame (0-indexed).
    ACDC naming: patientXXX_frameYY_gt.nii.gz  (YY is 1-indexed, zero-padded)
    """
    frame_str = str(frame_idx + 1).zfill(2)  # 0-indexed → 1-indexed, zero-padded
    candidate = os.path.join(patient_dir, f"{patient_id}_frame{frame_str}_gt.nii.gz")
    if os.path.exists(candidate):
        return candidate
    return None


# ── Core: odd-frame selection ────────────────────────────────────────────────

def select_odd_frames(
    total_frames: int,
    ed_frame: int,
    es_frame: int,
) -> List[int]:
    """
    Select odd-indexed frames from ED (diastole) to ES (systole), inclusive.

    ACDC stores frames as a full cycle. We only want the contraction phase:
        ED → ES  (diastole → peak systole)

    We then take every other frame (odd indices within this sub-sequence)
    to reduce temporal redundancy.

    Args:
        total_frames : total number of frames in the 4D volume
        ed_frame     : 0-indexed end-diastolic frame (usually frame 0)
        es_frame     : 0-indexed end-systolic frame  (usually ~frame 13-15)

    Returns:
        List of 0-indexed frame indices to use, ordered ED → ES
    """
    # Build the forward sequence from ED to ES
    # Handle wrap-around (rare but possible if ED > ES numerically)
    if ed_frame <= es_frame:
        sequence = list(range(ed_frame, es_frame + 1))
    else:
        # Wrap around: e.g., ED=28, ES=10 in a 30-frame sequence
        sequence = list(range(ed_frame, total_frames)) + list(range(0, es_frame + 1))

    # Take odd positions within the subsequence (index 0, 2, 4, ... of the sequence)
    # This means: sequence[0], sequence[2], sequence[4], ...
    # i.e., every other frame starting from the first
    odd_frames = [sequence[i] for i in range(0, len(sequence), 2)]

    # Ensure we always have at least 2 frames (needed for at least 1 frame pair)
    if len(odd_frames) < 2:
        # Fall back to the full sequence if too short
        odd_frames = sequence[:min(4, len(sequence))]

    return odd_frames


# ── Core: extract myocardium mask from segmentation ─────────────────────────

def extract_myocardium_mask(
    gt_volume: np.ndarray,
    slice_idx: int
) -> np.ndarray:
    """
    Extract binary myocardium mask from a 3D ground truth volume for one slice.

    Args:
        gt_volume  : (H, W, D) segmentation volume
        slice_idx  : which depth slice to extract

    Returns:
        Binary mask (H, W) where 1 = myocardium
    """
    slice_gt = gt_volume[:, :, slice_idx]
    return (slice_gt == MYOCARDIUM_LABEL).astype(np.float32)


# ── Core: per-patient normalization ─────────────────────────────────────────

def normalize_volume(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-patient min-max normalization to [0, 1].
    Applied across the entire 4D volume so intensities are comparable
    across time frames.
    """
    v_min = volume.min()
    v_max = volume.max()
    return (volume - v_min) / (v_max - v_min + eps)


# ── Core: resize frames to a fixed spatial size ──────────────────────────────

def resize_2d(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a 2D image to (target_h, target_w) using nearest-neighbor for masks,
    bilinear for images. We use torch for this to stay in-framework.
    """
    t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    t = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return t.squeeze().numpy()


def resize_mask_2d(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize binary mask using nearest-neighbor to avoid fractional values."""
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode="nearest"
    )
    return t.squeeze().numpy()


# ── Main Dataset class ───────────────────────────────────────────────────────

class ACDCSliceDataset(Dataset):
    """
    Slice-level ACDC dataset.

    Each sample is a single 2D short-axis slice trajectory:
        frames : (N_frames, H, W)  — odd-selected frames for that slice
        masks  : (N_frames, H, W)  — myocardium masks for each frame
        label  : int               — patient-level class label
        meta   : dict              — patient id, slice index, frame indices

    The ED-frame segmentation mask is used for all time frames
    (ACDC only provides ground truth at ED and ES; we use ED as the
    reference since it is our starting frame and typically has the
    clearest myocardium boundary).

    Args:
        root_dir    : path to ACDC training or testing directory
        patient_ids : list of patient folder names to include (e.g. ["patient001"])
                      If None, all patients in root_dir are used.
        target_h    : spatial height to resize all frames to (default 128)
        target_w    : spatial width  to resize all frames to (default 128)
        min_myo_pixels : minimum number of myocardium pixels in a slice
                         for it to be included (filters out empty slices)
    """

    def __init__(
        self,
        root_dir: str,
        patient_ids: Optional[List[str]] = None,
        target_h: int = 128,
        target_w: int = 128,
        min_myo_pixels: int = 50,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.target_h = target_h
        self.target_w = target_w
        self.min_myo_pixels = min_myo_pixels

        # Discover patient directories
        if patient_ids is None:
            patient_ids = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
                and d.startswith("patient")
            ])
        self.patient_ids = patient_ids

        # Build the flat list of (patient_id, slice_idx, frame_indices, label)
        # This is done at init time so __len__ and __getitem__ are fast
        self.samples = []          # list of dicts, one per valid slice
        self.patient_info = {}     # patient_id → info dict

        print(f"[ACDCSliceDataset] Scanning {len(self.patient_ids)} patients...")
        self._build_sample_list()
        print(f"[ACDCSliceDataset] Total valid slices: {len(self.samples)}")

    # ── Build sample list ────────────────────────────────────────────────────

    def _build_sample_list(self):
        skipped_patients = []
        skipped_slices = 0

        for pid in self.patient_ids:
            patient_dir = os.path.join(self.root_dir, pid)

            try:
                info = parse_patient_info(patient_dir)
            except FileNotFoundError as e:
                warnings.warn(f"Skipping {pid}: {e}")
                skipped_patients.append(pid)
                continue

            if info["group"] not in CLASS_TO_IDX:
                warnings.warn(f"Skipping {pid}: unknown group '{info['group']}'")
                skipped_patients.append(pid)
                continue

            label = CLASS_TO_IDX[info["group"]]
            self.patient_info[pid] = info

            # Load 4D volume to get shape — we do NOT store it in memory
            try:
                nifti_path = find_4d_file(patient_dir, pid)
            except FileNotFoundError as e:
                warnings.warn(f"Skipping {pid}: {e}")
                skipped_patients.append(pid)
                continue

            vol_4d = load_nifti(nifti_path)  # (H, W, D, T)
            H, W, D, T = vol_4d.shape
            del vol_4d  # free memory immediately — we reload per __getitem__

            # Select odd frames
            ed_frame = info["ed_frame"]
            es_frame = info["es_frame"]
            odd_frames = select_odd_frames(T, ed_frame, es_frame)

            # Find ED ground truth segmentation (used as mask reference)
            gt_path = find_gt_file(patient_dir, pid, ed_frame)
            if gt_path is None:
                # Try ES frame as fallback
                gt_path = find_gt_file(patient_dir, pid, es_frame)
            if gt_path is None:
                warnings.warn(f"Skipping {pid}: no ground truth segmentation found")
                skipped_patients.append(pid)
                continue

            gt_vol = load_nifti(gt_path)  # (H, W, D)

            # Register each valid slice
            for slice_idx in range(D):
                myo_mask = extract_myocardium_mask(gt_vol, slice_idx)
                n_myo = int(myo_mask.sum())

                if n_myo < self.min_myo_pixels:
                    skipped_slices += 1
                    continue  # this slice has no meaningful myocardium

                self.samples.append({
                    "patient_id":  pid,
                    "patient_dir": patient_dir,
                    "nifti_path":  nifti_path,
                    "gt_path":     gt_path,
                    "slice_idx":   slice_idx,
                    "frame_indices": odd_frames,
                    "label":       label,
                    "depth":       D,
                    "n_myo_pixels": n_myo,
                })

        if skipped_patients:
            print(f"  [!] Skipped patients: {skipped_patients}")
        if skipped_slices:
            print(f"  [!] Skipped {skipped_slices} slices with < {self.min_myo_pixels} myocardium pixels")

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dict:
            frames  : FloatTensor (N_frames, 1, H, W)  — normalized MRI frames
            masks   : FloatTensor (N_frames, 1, H, W)  — binary myocardium masks
            label   : LongTensor  scalar
            times   : FloatTensor (N_frames,)           — normalized time points in [0,1]
            meta    : dict with patient_id, slice_idx, frame_indices
        """
        sample = self.samples[idx]

        # Load 4D volume (reload from disk — avoids keeping all data in RAM)
        vol_4d = load_nifti(sample["nifti_path"])   # (H, W, D, T)
        vol_4d = normalize_volume(vol_4d)            # normalize over full patient volume

        # Load segmentation for mask
        gt_vol = load_nifti(sample["gt_path"])       # (H, W, D)

        slice_idx    = sample["slice_idx"]
        frame_idxs   = sample["frame_indices"]
        N_frames     = len(frame_idxs)

        # Extract 2D frames for this slice across selected time points
        frames_list = []
        for t in frame_idxs:
            frame_2d = vol_4d[:, :, slice_idx, t]          # (H, W)
            frame_2d = resize_2d(frame_2d, self.target_h, self.target_w)
            frames_list.append(frame_2d)

        # Extract myocardium mask (using ED frame segmentation)
        myo_mask_2d = extract_myocardium_mask(gt_vol, slice_idx)  # (H, W)
        myo_mask_2d = resize_mask_2d(myo_mask_2d, self.target_h, self.target_w)

        # Stack frames: (N_frames, H, W)
        frames_arr = np.stack(frames_list, axis=0)

        # Replicate mask for each time frame: (N_frames, H, W)
        # NOTE: we use the static ED mask for all frames as the ROI guide.
        # The deformation field will track how the myocardium moves away from this.
        masks_arr = np.stack([myo_mask_2d] * N_frames, axis=0)

        # Normalized time points: [0, ..., 1]
        times = np.linspace(0.0, 1.0, N_frames).astype(np.float32)

        # Convert to tensors, add channel dimension
        frames_tensor = torch.from_numpy(frames_arr).unsqueeze(1).float()  # (N, 1, H, W)
        masks_tensor  = torch.from_numpy(masks_arr).unsqueeze(1).float()   # (N, 1, H, W)
        times_tensor  = torch.from_numpy(times)                             # (N,)
        label_tensor  = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "frames": frames_tensor,
            "masks":  masks_tensor,
            "label":  label_tensor,
            "times":  times_tensor,
            "meta": {
                "patient_id":    sample["patient_id"],
                "slice_idx":     slice_idx,
                "frame_indices": frame_idxs,
                "n_myo_pixels":  sample["n_myo_pixels"],
            }
        }


# ── Patient-level dataset wrapper ────────────────────────────────────────────

class ACDCPatientDataset(Dataset):
    """
    Patient-level dataset — each sample contains ALL slices for one patient.
    Useful for evaluation where you need a single prediction per patient.

    Returns:
        frames : (D, N_frames, 1, H, W)
        masks  : (D, N_frames, 1, H, W)
        label  : scalar
        times  : (N_frames,)
        meta   : dict
    """

    def __init__(
        self,
        root_dir: str,
        patient_ids: Optional[List[str]] = None,
        target_h: int = 128,
        target_w: int = 128,
        min_myo_pixels: int = 50,
    ):
        # Re-use the slice dataset internally
        self._slice_ds = ACDCSliceDataset(
            root_dir, patient_ids, target_h, target_w, min_myo_pixels
        )

        # Group slice indices by patient
        from collections import defaultdict
        self._patient_to_slices = defaultdict(list)
        for i, s in enumerate(self._slice_ds.samples):
            self._patient_to_slices[s["patient_id"]].append(i)

        self._patient_list = sorted(self._patient_to_slices.keys())

    def __len__(self):
        return len(self._patient_list)

    def __getitem__(self, idx):
        pid = self._patient_list[idx]
        slice_indices = self._patient_to_slices[pid]

        slices_data = [self._slice_ds[i] for i in slice_indices]

        # Stack along depth dimension
        frames = torch.stack([s["frames"] for s in slices_data], dim=0)  # (D, N, 1, H, W)
        masks  = torch.stack([s["masks"]  for s in slices_data], dim=0)  # (D, N, 1, H, W)
        label  = slices_data[0]["label"]
        times  = slices_data[0]["times"]

        return {
            "frames": frames,
            "masks":  masks,
            "label":  label,
            "times":  times,
            "meta":   {"patient_id": pid, "n_slices": len(slice_indices)},
        }


# ── Collate function for variable N_frames ───────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate to handle variable N_frames across patients.
    Pads along the time dimension to the max N_frames in the batch.
    """
    max_frames = max(b["frames"].shape[0] for b in batch)

    frames_list, masks_list, labels, times_list = [], [], [], []

    for b in batch:
        N = b["frames"].shape[0]
        pad = max_frames - N

        if pad > 0:
            # Repeat last frame to pad (better than zero-padding for registration)
            frames_pad = b["frames"][-1:].repeat(pad, 1, 1, 1)
            masks_pad  = b["masks"][-1:].repeat(pad, 1, 1, 1)
            frames_padded = torch.cat([b["frames"], frames_pad], dim=0)
            masks_padded  = torch.cat([b["masks"],  masks_pad],  dim=0)
            times_padded  = torch.cat([
                b["times"],
                torch.linspace(b["times"][-1], 1.0, pad + 1)[1:]
            ], dim=0)
        else:
            frames_padded = b["frames"]
            masks_padded  = b["masks"]
            times_padded  = b["times"]

        frames_list.append(frames_padded)
        masks_list.append(masks_padded)
        labels.append(b["label"])
        times_list.append(times_padded)

    return {
        "frames": torch.stack(frames_list, dim=0),  # (B, N, 1, H, W)
        "masks":  torch.stack(masks_list,  dim=0),  # (B, N, 1, H, W)
        "label":  torch.stack(labels,      dim=0),  # (B,)
        "times":  torch.stack(times_list,  dim=0),  # (B, N)
        "meta":   [b["meta"] for b in batch],
    }


# ── Quick dataset split utility ──────────────────────────────────────────────

def get_patient_splits(
    root_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patients into train/val/test by patient (not by slice),
    stratified by class label.

    Returns train_ids, val_ids, test_ids
    """
    from collections import defaultdict
    import random

    rng = random.Random(seed)

    all_patients = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("patient")
    ])

    # Group by class
    class_to_patients = defaultdict(list)
    for pid in all_patients:
        try:
            info = parse_patient_info(os.path.join(root_dir, pid))
            class_to_patients[info["group"]].append(pid)
        except Exception:
            continue

    train_ids, val_ids, test_ids = [], [], []

    for group, pids in class_to_patients.items():
        rng.shuffle(pids)
        n = len(pids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_ids.extend(pids[:n_train])
        val_ids.extend(pids[n_train:n_train + n_val])
        test_ids.extend(pids[n_train + n_val:])

    return train_ids, val_ids, test_ids


# ── Quick sanity check ───────────────────────────────────────────────────────

def inspect_dataset(root_dir: str, n_patients: int = 5):
    """
    Print a summary of the first n_patients in the dataset.
    Run this to verify everything loads correctly before training.
    """
    patient_ids = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("patient")
    ])[:n_patients]

    print(f"\n{'='*60}")
    print(f"  ACDC Dataset Inspection ({n_patients} patients)")
    print(f"{'='*60}")

    for pid in patient_ids:
        pdir = os.path.join(root_dir, pid)
        try:
            info = parse_patient_info(pdir)
            nifti_path = find_4d_file(pdir, pid)
            vol = load_nifti(nifti_path)
            H, W, D, T = vol.shape
            odd_frames = select_odd_frames(T, info["ed_frame"], info["es_frame"])
            print(f"\n  {pid}")
            print(f"    Group      : {info['group']} (label={CLASS_TO_IDX.get(info['group'], '?')})")
            print(f"    Volume     : {H}x{W}x{D} spatial, {T} frames total")
            print(f"    ED/ES      : frame {info['ed_frame']} / frame {info['es_frame']}")
            print(f"    Odd frames : {odd_frames} ({len(odd_frames)} selected)")
        except Exception as e:
            print(f"\n  {pid} — ERROR: {e}")

    print(f"\n{'='*60}\n")


# ── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: str,
    patient_ids: Optional[List[str]] = None,
    batch_size: int = 8,
    target_h: int = 128,
    target_w: int = 128,
    num_workers: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function: build train/val/test DataLoaders.

    If patient_ids is provided, splits those patients.
    Otherwise uses all patients in root_dir.
    """
    if patient_ids is None:
        train_ids, val_ids, test_ids = get_patient_splits(
            root_dir, train_ratio, val_ratio, seed
        )
    else:
        # Manual split of provided patients (useful for small experiments)
        rng = __import__("random").Random(seed)
        ids = list(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_ids = ids[:n_train]
        val_ids   = ids[n_train:n_train + n_val]
        test_ids  = ids[n_train + n_val:]

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients")

    train_ds = ACDCSliceDataset(root_dir, train_ids, target_h, target_w)
    val_ds   = ACDCSliceDataset(root_dir, val_ids,   target_h, target_w)
    test_ds  = ACDCSliceDataset(root_dir, test_ids,  target_h, target_w)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py /path/to/acdc/training")
        sys.exit(1)

    root = sys.argv[1]

    # 1. Inspect first 5 patients
    inspect_dataset(root, n_patients=5)

    # 2. Build a small dataset from first 10 patients
    all_pids = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("patient")
    ])[:10]

    ds = ACDCSliceDataset(root, patient_ids=all_pids, target_h=128, target_w=128)
    print(f"Dataset size: {len(ds)} slices from {len(all_pids)} patients")

    # 3. Test __getitem__
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  frames shape : {sample['frames'].shape}")   # (N_frames, 1, 128, 128)
    print(f"  masks  shape : {sample['masks'].shape}")    # (N_frames, 1, 128, 128)
    print(f"  label        : {sample['label']} ({IDX_TO_CLASS[sample['label'].item()]})")
    print(f"  times        : {sample['times']}")
    print(f"  meta         : {sample['meta']}")

    # 4. Test DataLoader with collate
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  frames : {batch['frames'].shape}")   # (4, N_frames, 1, 128, 128)
    print(f"  masks  : {batch['masks'].shape}")
    print(f"  labels : {batch['label']}")
    print(f"  times  : {batch['times'].shape}")
    print("\n[dataset.py] All checks passed.")