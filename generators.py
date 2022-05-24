import numpy as np
import nibabel as nib
import os
import glob


def load_volfile(filename, np_var='vol'):
    if filename.endswith(('.nii', '.nii.gz', '.mgz')):
        img = nib.load(filename)
        vol = img.get_data().squeeze()
    elif filename.endswith('.npy'):
        vol = np.load(filename)
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
    else:
        raise ValueError('unknown filetype for %s' % filename)
    return vol


def volgen(
        vol_names,
        batch_size=1,
        return_segs=False,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.

    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var)
        imgs = [load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        invols = [scan, atlas]
        yield invols


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]
        invols = [scan1, scan2]

        yield invols


def get_volfile_list(filepath_list):
    img_list = []
    for filepath in filepath_list:
        img_list.append(load_volfile(filepath))
    return img_list


def patient_in_scan_to_scan(patient_dir_list, batch_size=1):
    pass
    while True:
        [index] = np.random.randint(len(patient_dir_list), size=batch_size)
        time_list = os.listdir(patient_dir_list[index])

        vol_names = []
        for tm in time_list:
            vol_names += glob.glob(patient_dir_list[index] + tm + "/vols/*.nii.gz")
        gen = volgen(vol_names, batch_size=batch_size)

        scan1 = next(gen)[0]
        scan2 = next(gen)[0]
        invols = [scan1, scan2]

        yield invols




