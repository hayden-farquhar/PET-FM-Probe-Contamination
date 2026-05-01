"""Core preprocessing utilities for PET-FM-Bench.

Shared across all task-specific preprocessing notebooks.
All parameters match the pre-registration (Section 5.1).
"""

import numpy as np
import SimpleITK as sitk


# Registration-specified parameters
SPACING_ISO = (2.0, 2.0, 2.0)
PATCH_SIZE_3D = (96, 96, 96)
MIP_SIZE_2D = (224, 224)
CT_HU_MIN = -1024
CT_HU_MAX = 3071
SUV_RESEG_MIN = 0.5  # pyradiomics resegmentation lower bound


def resample_to_isotropic(img_sitk, target_spacing=SPACING_ISO, interpolator=sitk.sitkLinear):
    """Resample a SimpleITK image to isotropic spacing."""
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(img_sitk)


def extract_patches_grid(volume, patch_size=PATCH_SIZE_3D, stride_fraction=1.0):
    """Extract patches on a regular grid from a 3D volume.

    Args:
        volume: numpy array (Z, Y, X)
        patch_size: tuple (pZ, pY, pX)
        stride_fraction: 1.0 = non-overlapping, 0.5 = 50% overlap

    Returns:
        patches: (N, pZ, pY, pX) array
        positions: list of (z, y, x) start coordinates
    """
    pz, py, px = patch_size
    sz, sy, sx = volume.shape
    stride_z = max(1, int(pz * stride_fraction))
    stride_y = max(1, int(py * stride_fraction))
    stride_x = max(1, int(px * stride_fraction))

    patches = []
    positions = []

    for z in range(0, max(1, sz - pz + 1), stride_z):
        for y in range(0, max(1, sy - py + 1), stride_y):
            for x in range(0, max(1, sx - px + 1), stride_x):
                patch = volume[z:z+pz, y:y+py, x:x+px]
                if patch.shape != (pz, py, px):
                    padded = np.zeros((pz, py, px), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((z, y, x))

    return np.stack(patches), positions


def extract_lesion_patch(volume, centroid_zyx, patch_size=PATCH_SIZE_3D):
    """Extract a single patch centred on a lesion centroid.

    Args:
        volume: numpy array (Z, Y, X)
        centroid_zyx: (z, y, x) centre coordinates
        patch_size: tuple (pZ, pY, pX)

    Returns:
        patch: (pZ, pY, pX) array, zero-padded at boundaries
    """
    pz, py, px = patch_size
    cz, cy, cx = [int(round(c)) for c in centroid_zyx]

    # Compute start/end with boundary handling
    z0 = max(0, cz - pz // 2)
    y0 = max(0, cy - py // 2)
    x0 = max(0, cx - px // 2)
    z1 = min(volume.shape[0], z0 + pz)
    y1 = min(volume.shape[1], y0 + py)
    x1 = min(volume.shape[2], x0 + px)

    patch = np.zeros((pz, py, px), dtype=volume.dtype)
    patch[:z1-z0, :y1-y0, :x1-x0] = volume[z0:z1, y0:y1, x0:x1]
    return patch


def compute_mips(volume):
    """Compute maximum intensity projections along three axes.

    Args:
        volume: numpy array (Z, Y, X)

    Returns:
        dict with keys 'coronal' (Z,X), 'axial' (Y,X), 'sagittal' (Z,Y)
    """
    return {
        "coronal": volume.max(axis=1),   # (Z, X)
        "axial": volume.max(axis=0),     # (Y, X)
        "sagittal": volume.max(axis=2),  # (Z, Y)
    }


def resize_2d(arr_2d, target_size=MIP_SIZE_2D):
    """Resize a 2D array to target size."""
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((target_size[1], target_size[0]))
    scale_x = arr_2d.shape[1] / target_size[1]
    scale_y = arr_2d.shape[0] / target_size[0]
    resampler.SetOutputSpacing((scale_x, scale_y))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(img)
    return sitk.GetArrayFromImage(resampled)


def clip_ct(ct_arr):
    """Clip CT to HU window per registration."""
    return np.clip(ct_arr, CT_HU_MIN, CT_HU_MAX)
