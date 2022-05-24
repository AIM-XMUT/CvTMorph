import os, glob, numpy as np, torch
import nibabel as nib
import utils
import models


def prepare_load():
    mov_vol = "./mov_vol.nii.gz"
    mov_seg = "./mov_seg.nii.gz"
    fix_vol = "./fix_vol.nii.gz"
    fix_seg = "./fix_seg.nii.gz"

    mov_vol = nib.load(mov_vol).get_fdata()
    mov_seg = nib.load(mov_seg).get_fdata()

    fix_vol = nib.load(fix_vol).get_fdata()
    fix_seg = nib.load(fix_seg).get_fdata()

    mov_vol = mov_vol[np.newaxis, np.newaxis, ...]
    mov_seg = mov_seg[np.newaxis, np.newaxis, ...]
    fix_vol = fix_vol[np.newaxis, np.newaxis, ...]
    fix_seg = fix_seg[np.newaxis, np.newaxis, ...]

    return [mov_vol, mov_seg], [fix_vol, fix_seg]


def test():
    labels = [1, 2]  # lung and airways
    landmark_flag = False
    load_model_pt = "./save_model_pt/0500.pt"

    [mov_vol, mov_seg], [fix_vol, fix_seg] = prepare_load()
    reg_model = models.register_model((160, 160, 160), 'nearest')
    model = torch.load(load_model_pt)
    model.eval()
    model.cuda()

    mov_vol = torch.from_numpy(mov_vol).cuda().float()
    mov_seg = torch.from_numpy(mov_seg).cuda().float()
    fix_vol = torch.from_numpy(fix_vol).cuda().float()

    x_in = torch.cat((mov_vol, fix_vol), dim=1)
    _, flow = model(x_in)

    warped_seg = reg_model((mov_seg, flow))

    mov_seg = mov_seg.cuda().data.cpu().numpy()
    before_dsc = utils.dice(mov_seg, fix_seg, labels)
    warped_seg = warped_seg.cuda().data.cpu().numpy()
    after_dsc = utils.dice(warped_seg, fix_seg, labels)

    flow = flow.squeeze().permute(1, 2, 3, 0)
    jac = utils.jacobian_determinant(flow.cuda().data.cpu().numpy())
    jac = len(jac[jac < 0].flatten())

    print("before  dice score:", before_dsc, "after dice score:", after_dsc, "neg jac:", jac)

    if landmark_flag:  # if have landmark points
        """
        Refer to the following:
        https://github.com/voxelmorph/voxelmorph/issues/84
        https://colab.research.google.com/drive/1V0CutSIfmtgDJg1XIkEnGteJuw0u7qT-#scrollTo=sw2TPiWNaH9b
        """
        mov_point = "./mov_point.npy"  # mov_point.shape: (300, 3)
        fix_point = "./fix_point.npy"  # fix_point.shape: (300, 3)
        mov_point = np.load(mov_point)
        fix_point = np.load(fix_point)

        sum_difference1 = []
        before_distance = fix_point - mov_point
        for ii in range(len(before_distance)):
            sum_difference1 += [np.linalg.norm(before_distance[ii])]

        fix_point = fix_point[np.newaxis, ...]
        data_in = [torch.from_numpy(fix_point).cuda(), flow]
        warp_point = utils.point_spatial_transformer(data_in)

        sum_difference2 = []
        after_different = warp_point.cuda().data.cpu().numpy() - mov_point
        after_different = after_different.squeeze()
        for ii in range(len(after_different)):
            sum_difference2 += [np.linalg.norm(after_different[ii])]

        print("before distance:", np.mean(sum_difference1), np.std(sum_difference1),
              "after distance:", np.mean(sum_difference2), np.std(sum_difference2))


if __name__ == '__main__':
    '''
    GPU configuration
    '''

    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    with torch.no_grad():
        test()
