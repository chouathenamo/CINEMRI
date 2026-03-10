"""
ACDC Cine MRI Dataset Loader
==========================================
Handles:
  - Loading 4D NIfTI volumes (H x W x D x T)
  - Odd-frame selection from diastole → peak systole
  - Myocardium mask extraction (label=2 in ACDC ground truth)
  - Per-patient normalization

New data layout (March 9 restructure):
  march9_dir/
    Images/   patientXXX_frameYY.nii.gz        ← individual MRI frames
    Masks/    patientXXX_frameYY_gt.nii.gz      ← segmentation masks

  training_dir/
    patientXXX/
      Info.cfg                                  ← group label, ED/ES indices
      patientXXX_4d.nii.gz                      ← full 4D cine volume

ACDC Class Labels (from Info.cfg):
  NOR  = Normal (healthy)
  DCM  = Dilated Cardiomyopathy
  HCM  = Hypertrophic Cardiomyopathy
  MINF = Myocardial Infarction
  RV   = Abnormal Right Ventricle
"""

import os
import re
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

def parse_patient_info(training_dir: str, patient_id: str) -> Dict:
    """
    Parse the Info.cfg file from the training directory for a patient.

    Args:
        training_dir : root of /path/to/ACDC/training
        patient_id   : e.g. "patient001"

    Returns a dict with keys:
        group       : str  — NOR / DCM / HCM / MINF / RV
        ed_frame    : int  — End-diastolic frame index (0-based)
        es_frame    : int  — End-systolic frame index  (0-based)
        height_mm   : float
        weight_kg   : float
        nb_frames   : int
    """
    patient_dir = os.path.join(training_dir, patient_id)
    cfg_path = os.path.join(patient_dir, "Info.cfg")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Info.cfg not found at {cfg_path}")

    info = {}
    with open(cfg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()

    return {
        "group":     info.get("Group", "UNK"),
        # ACDC frames are 1-indexed in Info.cfg → convert to 0-indexed
        "ed_frame":  int(info.get("ED", 1)) - 1,
        "es_frame":  int(info.get("ES", 1)) - 1,
        "height_mm": float(info.get("Height", 0.0)),
        "weight_kg": float(info.get("Weight", 0.0)),
        "nb_frames": int(info.get("NbFrame", 0)),
    }


# ── Utility: find 4D file in training dir ───────────────────────────────────

def find_4d_file(training_dir: str, patient_id: str) -> str:
    """
    Find the 4D cine MRI NIfTI file from the training directory.
    ACDC naming: training_dir/patientXXX/patientXXX_4d.nii.gz
    """
    patient_dir = os.path.join(training_dir, patient_id)
    candidate = os.path.join(patient_dir, f"{patient_id}_4d.nii.gz")
    if os.path.exists(candidate):
        return candidate
    # Fallback: search for any *_4d.nii.gz in the patient folder
    if os.path.isdir(patient_dir):
        for fname in os.listdir(patient_dir):
            if "_4d.nii.gz" in fname:
                return os.path.join(patient_dir, fname)
    raise FileNotFoundError(f"No 4D NIfTI found for {patient_id} in {patient_dir}")


# ── Utility: find mask file in March9Data/Masks ──────────────────────────────

def find_gt_file(march9_dir: str, patient_id: str, frame_idx: int) -> Optional[str]:
    """
    Find the ground-truth segmentation mask from March9Data/Masks/.

    File naming (1-indexed, zero-padded frame number):
        Masks/patientXXX_frameYY_gt.nii.gz

    Args:
        march9_dir : root of March9Data directory
        patient_id : e.g. "patient001"
        frame_idx  : 0-indexed frame number (converted internally to 1-indexed)

    Returns the full path if found, else None.
    """
    masks_dir = os.path.join(march9_dir, "Masks")
    frame_str = str(frame_idx + 1).zfill(2)   # 0-indexed → 1-indexed, zero-padded
    candidate = os.path.join(masks_dir, f"{patient_id}_frame{frame_str}_gt.nii.gz")
    if os.path.exists(candidate):
        return candidate
    return None


# ── Utility: discover patients from March9Data/Images ───────────────────────

def discover_patients_from_images(march9_dir: str) -> List[str]:
    """
    Scan March9Data/Images/ to find all unique patient IDs present.
    Files are named patientXXX_frameYY.nii.gz.

    Returns a sorted list of patient IDs (e.g. ["patient001", "patient003"]).
    """
    images_dir = os.path.join(march9_dir, "Images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    patient_ids = set()
    pattern = re.compile(r"^(patient\d{3})_frame\d{2}\.nii\.gz$")
    for fname in os.listdir(images_dir):
        m = pattern.match(fname)
        if m:
            patient_ids.add(m.group(1))

    return sorted(patient_ids)


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
    if ed_frame <= es_frame:
        sequence = list(range(ed_frame, es_frame + 1))
    else:
        # Wrap around: e.g., ED=28, ES=10 in a 30-frame sequence
        sequence = list(range(ed_frame, total_frames)) + list(range(0, es_frame + 1))

    odd_frames = [sequence[i] for i in range(0, len(sequence), 2)]

    # Ensure at least 2 frames (needed for at least 1 registration pair)
    if len(odd_frames) < 2:
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
    """Resize a 2D image to (target_h, target_w) using bilinear interpolation."""
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
    Slice-level ACDC dataset — new split-directory layout.

    Data sources:
      march9_dir   : March9Data root — contains Images/ and Masks/ subdirs
      training_dir : ACDC training root — contains patientXXX/ folders with
                     Info.cfg and patientXXX_4d.nii.gz

    Each sample is a single 2D short-axis slice trajectory:
        frames : (N_frames, 1, H, W)  — odd-selected frames for that slice
        masks  : (N_frames, 1, H, W)  — myocardium masks for each frame
        label  : int                  — patient-level class label
        times  : (N_frames,)          — normalized time points in [0, 1]
        meta   : dict

    The ED-frame segmentation mask is used as the ROI reference for all time
    frames (ACDC provides segmentation at ED and ES; ED is preferred since it
    is the starting frame and has the clearest myocardium boundary).

    Args:
        march9_dir     : path to March9Data directory
        training_dir   : path to ACDC training directory
        patient_ids    : list of patient IDs to include (e.g. ["patient001"]);
                         if None, all patients found in Images/ are used
        target_h       : height to resize frames to (default 128)
        target_w       : width  to resize frames to (default 128)
        min_myo_pixels : minimum myocardium pixels to include a slice (default 50)
    """

    def __init__(
        self,
        march9_dir: str,
        training_dir: str,
        patient_ids: Optional[List[str]] = None,
        target_h: int = 128,
        target_w: int = 128,
        min_myo_pixels: int = 50,
    ):
        super().__init__()
        self.march9_dir   = march9_dir
        self.training_dir = training_dir
        self.target_h     = target_h
        self.target_w     = target_w
        self.min_myo_pixels = min_myo_pixels

        # Discover patient IDs from the Images folder if not provided
        if patient_ids is None:
            patient_ids = discover_patients_from_images(march9_dir)
        self.patient_ids = patient_ids

        self.samples      = []   # list of dicts, one per valid slice
        self.patient_info = {}   # patient_id → info dict

        print(f"[ACDCSliceDataset] Scanning {len(self.patient_ids)} patients...")
        self._build_sample_list()
        print(f"[ACDCSliceDataset] Total valid slices: {len(self.samples)}")

    # ── Build sample list ────────────────────────────────────────────────────

    def _build_sample_list(self):
        skipped_patients = []
        skipped_slices   = 0

        for pid in self.patient_ids:

            # 1. Parse Info.cfg from training dir
            try:
                info = parse_patient_info(self.training_dir, pid)
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

            # 2. Load 4D volume shape from training dir (then free memory)
            try:
                nifti_path = find_4d_file(self.training_dir, pid)
            except FileNotFoundError as e:
                warnings.warn(f"Skipping {pid}: {e}")
                skipped_patients.append(pid)
                continue

            vol_4d = load_nifti(nifti_path)   # (H, W, D, T)
            H, W, D, T = vol_4d.shape
            del vol_4d   # free memory; we reload per __getitem__

            # 3. Select odd frames from the ED→ES contraction phase
            ed_frame   = info["ed_frame"]
            es_frame   = info["es_frame"]
            odd_frames = select_odd_frames(T, ed_frame, es_frame)

            # 4. Find ED mask from March9Data/Masks (ES as fallback)
            gt_path = find_gt_file(self.march9_dir, pid, ed_frame)
            if gt_path is None:
                gt_path = find_gt_file(self.march9_dir, pid, es_frame)
            if gt_path is None:
                warnings.warn(
                    f"Skipping {pid}: no mask found in "
                    f"{os.path.join(self.march9_dir, 'Masks')} "
                    f"for frame {ed_frame} (ED) or {es_frame} (ES)"
                )
                skipped_patients.append(pid)
                continue

            gt_vol = load_nifti(gt_path)   # (H, W, D)

            # 5. Register each valid slice
            for slice_idx in range(D):
                myo_mask = extract_myocardium_mask(gt_vol, slice_idx)
                n_myo    = int(myo_mask.sum())

                if n_myo < self.min_myo_pixels:
                    skipped_slices += 1
                    continue

                self.samples.append({
                    "patient_id":    pid,
                    "nifti_path":    nifti_path,
                    "gt_path":       gt_path,
                    "slice_idx":     slice_idx,
                    "frame_indices": odd_frames,
                    "label":         label,
                    "depth":         D,
                    "n_myo_pixels":  n_myo,
                })

        if skipped_patients:
            print(f"  [!] Skipped patients: {skipped_patients}")
        if skipped_slices:
            print(f"  [!] Skipped {skipped_slices} slices "
                  f"with < {self.min_myo_pixels} myocardium pixels")

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dict:
            frames  : FloatTensor (N_frames, 1, H, W)  — normalized MRI frames
            masks   : FloatTensor (N_frames, 1, H, W)  — binary myocardium masks
            label   : LongTensor  scalar
            times   : FloatTensor (N_frames,)           — normalized time in [0,1]
            meta    : dict with patient_id, slice_idx, frame_indices
        """
        sample = self.samples[idx]

        # Load & normalize 4D volume from training dir
        vol_4d = load_nifti(sample["nifti_path"])   # (H, W, D, T)
        vol_4d = normalize_volume(vol_4d)

        # Load segmentation mask from March9Data/Masks
        gt_vol = load_nifti(sample["gt_path"])       # (H, W, D)

        slice_idx  = sample["slice_idx"]
        frame_idxs = sample["frame_indices"]
        N_frames   = len(frame_idxs)

        # Extract 2D frames for this slice across selected time points
        frames_list = []
        for t in frame_idxs:
            frame_2d = vol_4d[:, :, slice_idx, t]           # (H, W)
            frame_2d = resize_2d(frame_2d, self.target_h, self.target_w)
            frames_list.append(frame_2d)

        # Extract myocardium mask (static ED mask used as ROI guide)
        myo_mask_2d = extract_myocardium_mask(gt_vol, slice_idx)   # (H, W)
        myo_mask_2d = resize_mask_2d(myo_mask_2d, self.target_h, self.target_w)

        frames_arr = np.stack(frames_list, axis=0)                     # (N, H, W)
        masks_arr  = np.stack([myo_mask_2d] * N_frames, axis=0)       # (N, H, W)
        times      = np.linspace(0.0, 1.0, N_frames).astype(np.float32)

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
            },
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
        march9_dir: str,
        training_dir: str,
        patient_ids: Optional[List[str]] = None,
        target_h: int = 128,
        target_w: int = 128,
        min_myo_pixels: int = 50,
    ):
        self._slice_ds = ACDCSliceDataset(
            march9_dir, training_dir, patient_ids, target_h, target_w, min_myo_pixels
        )

        from collections import defaultdict
        self._patient_to_slices = defaultdict(list)
        for i, s in enumerate(self._slice_ds.samples):
            self._patient_to_slices[s["patient_id"]].append(i)

        self._patient_list = sorted(self._patient_to_slices.keys())

    def __len__(self):
        return len(self._patient_list)

    def __getitem__(self, idx):
        pid          = self._patient_list[idx]
        slice_indices = self._patient_to_slices[pid]
        slices_data  = [self._slice_ds[i] for i in slice_indices]

        frames = torch.stack([s["frames"] for s in slices_data], dim=0)  # (D, N, 1, H, W)
        masks  = torch.stack([s["masks"]  for s in slices_data], dim=0)  # (D, N, 1, H, W)
        label  = slices_data[0]["label"]
        times  = slices_data[0]["times"]

        return {
            "frames": frames,
            "masks":  masks,
            "label":  label,
            "times":  times,
            "meta": {
                "patient_id": pid,
                "n_slices":   len(slice_indices),
            },
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
        N   = b["frames"].shape[0]
        pad = max_frames - N

        if pad > 0:
            frames_pad    = b["frames"][-1:].repeat(pad, 1, 1, 1)
            masks_pad     = b["masks"][-1:].repeat(pad, 1, 1, 1)
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
        "frames": torch.stack(frames_list, dim=0),   # (B, N, 1, H, W)
        "masks":  torch.stack(masks_list,  dim=0),   # (B, N, 1, H, W)
        "label":  torch.stack(labels,      dim=0),   # (B,)
        "times":  torch.stack(times_list,  dim=0),   # (B, N)
        "meta":   [b["meta"] for b in batch],
    }


# ── Patient split utility ────────────────────────────────────────────────────

def get_patient_splits(
    march9_dir: str,
    training_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patients into train/val/test by patient (not by slice),
    stratified by class label.

    Discovers patient IDs from March9Data/Images/ and reads labels from
    training_dir/patientXXX/Info.cfg.

    Returns train_ids, val_ids, test_ids.
    """
    from collections import defaultdict
    import random

    rng          = random.Random(seed)
    all_patients = discover_patients_from_images(march9_dir)

    class_to_patients = defaultdict(list)
    for pid in all_patients:
        try:
            info = parse_patient_info(training_dir, pid)
            class_to_patients[info["group"]].append(pid)
        except Exception:
            continue

    train_ids, val_ids, test_ids = [], [], []
    for group, pids in class_to_patients.items():
        rng.shuffle(pids)
        n       = len(pids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_ids.extend(pids[:n_train])
        val_ids.extend(pids[n_train:n_train + n_val])
        test_ids.extend(pids[n_train + n_val:])

    return train_ids, val_ids, test_ids


# ── Inspection utility ───────────────────────────────────────────────────────

def inspect_dataset(march9_dir: str, training_dir: str, n_patients: int = 5):
    """
    Print a summary of the first n_patients discovered in Images/.
    Run this to verify everything loads correctly before training.
    """
    patient_ids = discover_patients_from_images(march9_dir)[:n_patients]

    print(f"\n{'='*60}")
    print(f"  ACDC Dataset Inspection ({n_patients} patients)")
    print(f"  march9_dir   : {march9_dir}")
    print(f"  training_dir : {training_dir}")
    print(f"{'='*60}")

    for pid in patient_ids:
        try:
            info       = parse_patient_info(training_dir, pid)
            nifti_path = find_4d_file(training_dir, pid)
            vol        = load_nifti(nifti_path)
            H, W, D, T = vol.shape
            odd_frames = select_odd_frames(T, info["ed_frame"], info["es_frame"])
            gt_path    = find_gt_file(march9_dir, pid, info["ed_frame"])
            print(f"\n  {pid}")
            print(f"    Group      : {info['group']} "
                  f"(label={CLASS_TO_IDX.get(info['group'], '?')})")
            print(f"    Volume     : {H}x{W}x{D} spatial, {T} frames total")
            print(f"    ED/ES      : frame {info['ed_frame']} / frame {info['es_frame']}")
            print(f"    Odd frames : {odd_frames} ({len(odd_frames)} selected)")
            print(f"    Mask found : {'YES  ' + gt_path if gt_path else 'NO  (will try ES)'}")
        except Exception as e:
            print(f"\n  {pid} — ERROR: {e}")

    print(f"\n{'='*60}\n")


# ── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    march9_dir: str,
    training_dir: str,
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

    Args:
        march9_dir   : path to March9Data/ (contains Images/ and Masks/)
        training_dir : path to ACDC training/ (contains patientXXX/ folders)
        patient_ids  : if provided, split these patients; otherwise use all
                       patients discovered in Images/
    """
    if patient_ids is None:
        train_ids, val_ids, test_ids = get_patient_splits(
            march9_dir, training_dir, train_ratio, val_ratio, seed
        )
    else:
        import random
        rng = random.Random(seed)
        ids = list(patient_ids)
        rng.shuffle(ids)
        n       = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_ids = ids[:n_train]
        val_ids   = ids[n_train:n_train + n_val]
        test_ids  = ids[n_train + n_val:]

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test patients")

    train_ds = ACDCSliceDataset(
        march9_dir, training_dir, train_ids, target_h, target_w)
    val_ds   = ACDCSliceDataset(
        march9_dir, training_dir, val_ids,   target_h, target_w)
    test_ds  = ACDCSliceDataset(
        march9_dir, training_dir, test_ids,  target_h, target_w)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <march9_dir> <training_dir>")
        print("  e.g. python dataset.py "
              "/home/amo/CINEMRI/data/ACDC/March9Data "
              "/home/amo/CINEMRI/data/ACDC/training")
        sys.exit(1)

    march9_dir   = sys.argv[1]
    training_dir = sys.argv[2]

    # 1. Inspect first 5 patients
    inspect_dataset(march9_dir, training_dir, n_patients=5)

    # 2. Discover all patients and build a small dataset
    all_pids  = discover_patients_from_images(march9_dir)
    test_pids = all_pids[:10]

    ds = ACDCSliceDataset(
        march9_dir, training_dir,
        patient_ids=test_pids,
        target_h=128, target_w=128,
    )
    print(f"Dataset size: {len(ds)} slices from {len(test_pids)} patients")

    # 3. Test __getitem__
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  frames shape : {sample['frames'].shape}")   # (N_frames, 1, 128, 128)
    print(f"  masks  shape : {sample['masks'].shape}")    # (N_frames, 1, 128, 128)
    print(f"  label        : {sample['label']} "
          f"({IDX_TO_CLASS[sample['label'].item()]})")
    print(f"  times        : {sample['times']}")
    print(f"  meta         : {sample['meta']}")

    # 4. Test DataLoader with collate
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  frames : {batch['frames'].shape}")   # (4, N_frames, 1, 128, 128)
    print(f"  masks  : {batch['masks'].shape}")
    print(f"  labels : {batch['label']}")
    print(f"  times  : {batch['times'].shape}")
    print("\n[dataset.py] All checks passed.")