"""
ACDC Cine MRI Dataset Loader  (v3)
===================================
Changes from v2:
  - Computes 8 patient-level clinical features at scan time
      EF, LV-EDV, LV-ESV, myocardial volume, RV volume,
      RV/LV ratio, height, weight  (all normalised to ~[0,1])
  - Loads pixel spacing from NIfTI header → volumes in mL
  - Both ED and ES masks loaded to compute EF
  - augment=True flag for training: random flip, rotation ±15°,
    intensity jitter ±15% (uses cv2, already a dependency)
  - n_myo_pixels already in meta (for weighted slice aggregation in
    val_epoch)
  - collate_fn now pads and stacks clinical_feat (B, 8) tensor

Data layout (unchanged):
  march9_dir/
    Images/   patientXXX_frameYY.nii.gz
    Masks/    patientXXX_frameYY_gt.nii.gz
  training_dir/
    patientXXX/
      Info.cfg
      patientXXX_4d.nii.gz

ACDC labels in gt files:
  0 = background
  1 = RV cavity
  2 = myocardium (LV wall)
  3 = LV cavity
"""

import os
import re
import random
import numpy as np
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import warnings

# ── Label mappings ──────────────────────────────────────────────────────────
CLASS_TO_IDX = {"NOR": 0, "DCM": 1, "HCM": 2, "MINF": 3, "RV": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

MYOCARDIUM_LABEL = 2
LV_LABEL         = 3
RV_LABEL         = 1

N_CLINICAL_FEATURES = 8   # exported so model.py can read it


# ── NIfTI helpers ───────────────────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def load_nifti_with_zooms(path: str):
    """Returns (data_float32, (sx, sy, sz)) where spacing is in mm."""
    img   = nib.load(path)
    data  = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()
    return data, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


# ── Info.cfg parser ─────────────────────────────────────────────────────────

def parse_patient_info(training_dir: str, patient_id: str) -> Dict:
    cfg_path = os.path.join(training_dir, patient_id, "Info.cfg")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Info.cfg not found: {cfg_path}")
    info = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return {
        "group":     info.get("Group", "UNK"),
        "ed_frame":  int(info.get("ED", 1)) - 1,
        "es_frame":  int(info.get("ES", 1)) - 1,
        "height_mm": float(info.get("Height", 0.0)),
        "weight_kg": float(info.get("Weight", 0.0)),
        "nb_frames": int(info.get("NbFrame", 0)),
    }


# ── File finders ────────────────────────────────────────────────────────────

def find_4d_file(training_dir: str, patient_id: str) -> str:
    patient_dir = os.path.join(training_dir, patient_id)
    candidate   = os.path.join(patient_dir, f"{patient_id}_4d.nii.gz")
    if os.path.exists(candidate):
        return candidate
    if os.path.isdir(patient_dir):
        for fname in os.listdir(patient_dir):
            if "_4d.nii.gz" in fname:
                return os.path.join(patient_dir, fname)
    raise FileNotFoundError(f"No 4D NIfTI for {patient_id} in {patient_dir}")

def find_gt_file(march9_dir: str, patient_id: str, frame_idx: int) -> Optional[str]:
    masks_dir  = os.path.join(march9_dir, "Masks")
    frame_str  = str(frame_idx + 1).zfill(2)
    candidate  = os.path.join(masks_dir, f"{patient_id}_frame{frame_str}_gt.nii.gz")
    return candidate if os.path.exists(candidate) else None

def discover_patients_from_images(march9_dir: str) -> List[str]:
    images_dir = os.path.join(march9_dir, "Images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images/ not found: {images_dir}")
    pids     = set()
    pattern  = re.compile(r"^(patient\d{3})_frame\d{2}\.nii\.gz$")
    for fname in os.listdir(images_dir):
        m = pattern.match(fname)
        if m:
            pids.add(m.group(1))
    return sorted(pids)


# ── Frame selection ─────────────────────────────────────────────────────────

def select_odd_frames(total_frames: int, ed_frame: int, es_frame: int) -> List[int]:
    if ed_frame <= es_frame:
        sequence = list(range(ed_frame, es_frame + 1))
    else:
        sequence = list(range(ed_frame, total_frames)) + list(range(0, es_frame + 1))
    odd_frames = [sequence[i] for i in range(0, len(sequence), 2)]
    if len(odd_frames) < 2:
        odd_frames = sequence[:min(4, len(sequence))]
    return odd_frames


# ── Mask / volume helpers ───────────────────────────────────────────────────

def extract_myocardium_mask(gt_volume: np.ndarray, slice_idx: int) -> np.ndarray:
    return (np.round(gt_volume[:, :, slice_idx]) == MYOCARDIUM_LABEL).astype(np.float32)

def normalize_volume(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v_min, v_max = volume.min(), volume.max()
    return (volume - v_min) / (v_max - v_min + eps)

def resize_2d(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()

def resize_mask_2d(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode="nearest")
    return t.squeeze().numpy()


# ── Clinical feature computation ─────────────────────────────────────────────

def compute_clinical_features(
    gt_ed: Optional[np.ndarray],
    gt_es: Optional[np.ndarray],
    voxel_vol_mm3: float,
    height_mm: float,
    weight_kg: float,
) -> np.ndarray:
    """
    Compute 8 normalised scalar features from ED and ES segmentation volumes.
    All features are clipped to [0, 2] to suppress outliers.

    Features:
        0  EF               Ejection fraction (LV cavity) [0,1]
        1  lv_edv_norm      LV EDV / 300 mL
        2  lv_esv_norm      LV ESV / 200 mL
        3  myo_vol_norm     Myocardial volume / 200 mL
        4  rv_vol_norm      RV EDV / 250 mL
        5  rv_lv_ratio      RV / LV EDV ratio / 2
        6  height_norm      Height / 200 cm
        7  weight_norm      Weight / 120 kg

    Clinical relevance:
        DCM  → high EDV/ESV, low EF
        HCM  → high myo_vol, normal/high EF
        MINF → reduced EF, regional, moderate EDV
        RV   → high rv_lv_ratio
        NOR  → all normal ranges
    """
    def _vol(gt, label):
        if gt is None:
            return 0.0
        return float((np.round(gt) == label).sum()) * voxel_vol_mm3 / 1000.0  # mL

    lv_ed  = _vol(gt_ed, LV_LABEL)
    lv_es  = _vol(gt_es, LV_LABEL)
    myo_ed = _vol(gt_ed, MYOCARDIUM_LABEL)
    rv_ed  = _vol(gt_ed, RV_LABEL)

    ef           = (lv_ed - lv_es) / (lv_ed + 1e-6)
    rv_lv_ratio  = rv_ed / (lv_ed + 1e-6)
    height_cm    = height_mm / 10.0
    weight       = weight_kg

    features = np.array([
        np.clip(ef, 0, 1),
        lv_ed  / 300.0,
        lv_es  / 200.0,
        myo_ed / 200.0,
        rv_ed  / 250.0,
        np.clip(rv_lv_ratio / 2.0, 0, 2.0),
        np.clip(height_cm   / 200.0, 0, 1.0),
        np.clip(weight      / 120.0, 0, 1.0),
    ], dtype=np.float32)

    # Final clip to suppress extreme outliers
    return np.clip(features, 0.0, 2.0)


# ── Main Dataset class ───────────────────────────────────────────────────────

class ACDCSliceDataset(Dataset):
    """
    Slice-level ACDC dataset (v3).

    Same interface as v2 with two additions:
      - Returns 'clinical_feat': FloatTensor (N_CLINICAL_FEATURES,) per sample
      - Supports augmentation (augment=True for training splits)
    """

    def __init__(
        self,
        march9_dir:     str,
        training_dir:   str,
        patient_ids:    Optional[List[str]] = None,
        target_h:       int   = 128,
        target_w:       int   = 128,
        min_myo_pixels: int   = 50,
        augment:        bool  = False,
    ):
        super().__init__()
        self.march9_dir     = march9_dir
        self.training_dir   = training_dir
        self.target_h       = target_h
        self.target_w       = target_w
        self.min_myo_pixels = min_myo_pixels
        self.augment        = augment

        if patient_ids is None:
            patient_ids = discover_patients_from_images(march9_dir)
        self.patient_ids  = patient_ids
        self.samples      = []
        self.patient_info = {}   # pid → info + clinical_features

        print(f"[ACDCSliceDataset] Scanning {len(self.patient_ids)} patients "
              f"(augment={augment})...", flush=True)
        self._build_sample_list()
        print(f"[ACDCSliceDataset] Total valid slices: {len(self.samples)}", flush=True)

    # ── Build sample list ────────────────────────────────────────────────────

    def _build_sample_list(self):
        skipped_patients = []
        skipped_slices   = 0

        for pid in self.patient_ids:

            # 1. Parse Info.cfg
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

            label    = CLASS_TO_IDX[info["group"]]
            ed_frame = info["ed_frame"]
            es_frame = info["es_frame"]

            # 2. Load 4D volume to get shape + pixel spacing
            try:
                nifti_path = find_4d_file(self.training_dir, pid)
            except FileNotFoundError as e:
                warnings.warn(f"Skipping {pid}: {e}")
                skipped_patients.append(pid)
                continue

            vol_4d, (sx, sy, sz) = load_nifti_with_zooms(nifti_path)
            H, W, D, T           = vol_4d.shape
            voxel_vol_mm3        = sx * sy * sz
            del vol_4d

            # 3. Frame selection
            odd_frames = select_odd_frames(T, ed_frame, es_frame)

            # 4. Find ED mask (required for myo filter + clinical features)
            gt_ed_path = find_gt_file(self.march9_dir, pid, ed_frame)
            gt_es_path = find_gt_file(self.march9_dir, pid, es_frame)

            # Choose the mask file used as ROI reference for slices
            gt_roi_path = gt_ed_path or gt_es_path
            if gt_roi_path is None:
                warnings.warn(f"Skipping {pid}: no mask found in "
                              f"{os.path.join(self.march9_dir, 'Masks')}")
                skipped_patients.append(pid)
                continue

            gt_vol = load_nifti(gt_roi_path)   # (H, W, D) — used for myo filter

            # 5. Clinical features (patient-level)
            gt_ed_arr = load_nifti(gt_ed_path) if gt_ed_path else None
            gt_es_arr = load_nifti(gt_es_path) if gt_es_path else None

            clinical_feat = compute_clinical_features(
                gt_ed_arr, gt_es_arr, voxel_vol_mm3,
                info["height_mm"], info["weight_kg"],
            )
            info["clinical_features"] = clinical_feat
            self.patient_info[pid]    = info

            # 6. Register valid slices
            for slice_idx in range(D):
                myo_mask = extract_myocardium_mask(gt_vol, slice_idx)
                n_myo    = int(myo_mask.sum())

                if n_myo < self.min_myo_pixels:
                    skipped_slices += 1
                    continue

                self.samples.append({
                    "patient_id":    pid,
                    "nifti_path":    nifti_path,
                    "gt_path":       gt_roi_path,
                    "slice_idx":     slice_idx,
                    "frame_indices": odd_frames,
                    "label":         label,
                    "depth":         D,
                    "n_myo_pixels":  n_myo,
                })

        if skipped_patients:
            print(f"  [!] Skipped patients: {skipped_patients}", flush=True)
        if skipped_slices:
            print(f"  [!] Skipped {skipped_slices} slices "
                  f"with < {self.min_myo_pixels} myocardium pixels", flush=True)

    # ── Augmentation helpers ─────────────────────────────────────────────────

    def _augment(self, frames_arr: np.ndarray, masks_arr: np.ndarray):
        """
        In-place-safe augmentation applied consistently to all frames and masks.
        frames_arr : (N, H, W) float32 in [0,1]
        masks_arr  : (N, H, W) binary float32
        Returns augmented copies.
        """
        frames_arr = frames_arr.copy()
        masks_arr  = masks_arr.copy()
        N, H, W    = frames_arr.shape

        # Random horizontal flip
        if random.random() > 0.5:
            frames_arr = frames_arr[:, :, ::-1].copy()
            masks_arr  = masks_arr[:, :, ::-1].copy()

        # Random vertical flip
        if random.random() > 0.5:
            frames_arr = frames_arr[:, ::-1, :].copy()
            masks_arr  = masks_arr[:, ::-1, :].copy()

        # Random rotation ±15 degrees
        angle  = random.uniform(-15.0, 15.0)
        center = (W / 2.0, H / 2.0)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        for i in range(N):
            frames_arr[i] = cv2.warpAffine(
                frames_arr[i], M, (W, H), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            masks_arr[i] = cv2.warpAffine(
                masks_arr[i], M, (W, H), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        # Intensity jitter: frames only (masks are binary)
        scale         = random.uniform(0.85, 1.15)
        frames_arr    = np.clip(frames_arr * scale, 0.0, 1.0)

        return frames_arr, masks_arr

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load & normalize 4D volume
        vol_4d = load_nifti(sample["nifti_path"])
        vol_4d = normalize_volume(vol_4d)

        # Load segmentation mask
        gt_vol = load_nifti(sample["gt_path"])

        slice_idx  = sample["slice_idx"]
        frame_idxs = sample["frame_indices"]
        N_frames   = len(frame_idxs)

        # Extract 2D frames across selected time points
        frames_list = []
        for t in frame_idxs:
            frame_2d = vol_4d[:, :, slice_idx, t]
            frame_2d = resize_2d(frame_2d, self.target_h, self.target_w)
            frames_list.append(frame_2d)

        myo_mask_2d = extract_myocardium_mask(gt_vol, slice_idx)
        myo_mask_2d = resize_mask_2d(myo_mask_2d, self.target_h, self.target_w)

        frames_arr = np.stack(frames_list, axis=0)              # (N, H, W)
        masks_arr  = np.stack([myo_mask_2d] * N_frames, axis=0) # (N, H, W)

        # Augmentation (training only)
        if self.augment:
            frames_arr, masks_arr = self._augment(frames_arr, masks_arr)

        times = np.linspace(0.0, 1.0, N_frames).astype(np.float32)

        frames_tensor = torch.from_numpy(frames_arr).unsqueeze(1).float()  # (N, 1, H, W)
        masks_tensor  = torch.from_numpy(masks_arr).unsqueeze(1).float()   # (N, 1, H, W)
        times_tensor  = torch.from_numpy(times)
        label_tensor  = torch.tensor(sample["label"], dtype=torch.long)

        # Clinical features for this patient
        pid            = sample["patient_id"]
        clinical_feat  = torch.from_numpy(
            self.patient_info[pid]["clinical_features"]
        ).float()   # (N_CLINICAL_FEATURES,)

        return {
            "frames":       frames_tensor,
            "masks":        masks_tensor,
            "label":        label_tensor,
            "times":        times_tensor,
            "clinical_feat": clinical_feat,
            "meta": {
                "patient_id":   pid,
                "slice_idx":    slice_idx,
                "frame_indices": frame_idxs,
                "n_myo_pixels": sample["n_myo_pixels"],
            },
        }


# ── Collate function ─────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pad variable-length N_frames to the max in the batch.
    Stacks clinical_feat directly (no padding needed — fixed size).
    """
    max_frames = max(b["frames"].shape[0] for b in batch)

    frames_list, masks_list, labels, times_list, clinical_list = [], [], [], [], []

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
        clinical_list.append(b["clinical_feat"])

    return {
        "frames":        torch.stack(frames_list,   dim=0),   # (B, N, 1, H, W)
        "masks":         torch.stack(masks_list,    dim=0),   # (B, N, 1, H, W)
        "label":         torch.stack(labels,        dim=0),   # (B,)
        "times":         torch.stack(times_list,    dim=0),   # (B, N)
        "clinical_feat": torch.stack(clinical_list, dim=0),   # (B, 8)
        "meta":          [b["meta"] for b in batch],
    }


# ── Patient split utility ────────────────────────────────────────────────────

def get_patient_splits(
    march9_dir:   str,
    training_dir: str,
    train_ratio:  float = 0.7,
    val_ratio:    float = 0.15,
    seed:         int   = 42,
) -> Tuple[List[str], List[str], List[str]]:
    from collections import defaultdict
    rng          = random.Random(seed)
    all_patients = discover_patients_from_images(march9_dir)
    class_to_pats = defaultdict(list)
    for pid in all_patients:
        try:
            info = parse_patient_info(training_dir, pid)
            class_to_pats[info["group"]].append(pid)
        except Exception:
            continue

    train_ids, val_ids, test_ids = [], [], []
    for group, pids in class_to_pats.items():
        rng.shuffle(pids)
        n       = len(pids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        train_ids.extend(pids[:n_train])
        val_ids.extend(pids[n_train:n_train + n_val])
        test_ids.extend(pids[n_train + n_val:])

    return train_ids, val_ids, test_ids


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dataset_v3.py <march9_dir> <training_dir>")
        sys.exit(1)

    march9_dir   = sys.argv[1]
    training_dir = sys.argv[2]

    pids = discover_patients_from_images(march9_dir)[:6]
    print(f"\nTesting with patients: {pids}")

    ds = ACDCSliceDataset(march9_dir, training_dir, patient_ids=pids,
                          target_h=128, target_w=128, augment=False)
    print(f"\nDataset size: {len(ds)} slices")

    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  frames shape    : {sample['frames'].shape}")
    print(f"  masks  shape    : {sample['masks'].shape}")
    print(f"  label           : {sample['label']} ({IDX_TO_CLASS[sample['label'].item()]})")
    print(f"  clinical_feat   : {sample['clinical_feat'].numpy()}")
    print(f"    EF={sample['clinical_feat'][0]:.3f}  "
          f"LV-EDV={sample['clinical_feat'][1]*300:.0f}mL  "
          f"EF%={sample['clinical_feat'][0]*100:.1f}%")

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  frames        : {batch['frames'].shape}")
    print(f"  clinical_feat : {batch['clinical_feat'].shape}")
    print(f"  labels        : {batch['label']}")
    print("\n[dataset_v3.py] All checks passed.")
