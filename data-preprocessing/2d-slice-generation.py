import glob

import nibabel as nib
import numpy as np
import torch


def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range

    return data_normalized


# =============================================================================
# Slicing (Dict: ct, mask) je 2d Bild:
# =============================================================================
# ein Dict für jede Schicht von jeden Bild ()
def save(mask, z, image):
    save_folder = "/home/mfmezger/data/COVID/"

    # Nifti -> Numpy
    mask_new = nib.load(mask)
    mask_new = mask_new.get_fdata()
    mask_new = mask_new.transpose(2, 0, 1)

    img_new = nib.load(image)
    img_new = img_new.get_fdata()
    img_new = img_new.transpose(2, 0, 1)

    img_new = img_new.astype(np.float32)
    img_new = min_max_normalization(img_new, 0.001)
    # Save

    for i in range(img_new.shape[0]):
        # check if mask is empty.

        values, counts = np.unique(mask_new[i, ...], return_counts=True)

        if len(values) != 1 and counts[1] > (0.02 * 512 * 512):
            # Numpy -> Torch
            mask_t = torch.from_numpy(mask_new[i, ...])
            mask_t = mask_t.to(torch.float16)

            img_t = torch.from_numpy(img_new[i, ...])
            img_t = img_t.to(torch.float16)
            path = save_folder + "/" + str(z) + "_" + str(i) + ".pt"
            torch.save({"vol": img_t, "mask": mask_t, "name": image.split(".")[0]}, path)


def main():
    i = 2
    z = 0
    Ordner = sorted(glob.glob(
        "/home/mfmezger/Downloads/COVID-19-20_v2/Train/volume-covid19-A-*.nii.gz"))  # Ordner: Alle Pfade aus dem Ordner Train sortiert (die Pfade unterscheiden sich nur durch *)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner Train
        if (i % 2) == 0:  # ct und mask kommen immer abwchselnd: bei i%2 == 0 -> immer das ct Bild
            hold = fileA  # hold = ct Bild
        if (i % 2) != 0:  # i%2 != 0 -> immer mask (aber das ct Bild von davor ist in hold gespeichert)
            save(fileA, z, hold)  # fileA = mask, hold = ct
            z = z + 1
            print(fileA.split(".")[0].split("/")[-1])
        i = i + 1


if __name__ == '__main__':
    main()
