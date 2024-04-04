import torch
import numpy as np
import pickle as pkl
from pathlib import Path
import subprocess
import random
from nvdiff_render import util
import torchvision
import torchvision.transforms as T

downsampler_512 = T.Resize((512, 512))
tensor_to_img = T.ToPILImage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_view_obj(n_view, cam_radius, res=[512, 512], cam_near_far=[0.1, 1000.0], spp=1, is_face=False):
    iter_res = res
    fovy = np.deg2rad(45)
    proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Random rotation/translation matrix for optimization.
    mv_list, mvp_list, campos_list, direction_list = [], [], [], []
    for view_i in range(n_view):
        if view_i == 0:
            angle_x = 0.0  # elevation
            angle_y = 0.0  # azimuth
        else:
            angle_x = np.random.uniform(-np.pi / 3, np.pi / 3)
            angle_y = np.random.uniform(0, 2 * np.pi)

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = overhead
        if angle_x < -np.pi / 4:
            direction = 3
        else:
            if 0 <= angle_y <= np.pi / 4 or angle_y > 7 * np.pi / 4:
                direction = 0
            elif np.pi / 4 < angle_y <= 3 * np.pi / 4:
                direction = 1
            elif 3 * np.pi / 4 < angle_y <= 5 * np.pi / 4:
                direction = 2
            elif 5 * np.pi / 4 < angle_y <= 7 * np.pi / 4:
                direction = 1

        # for object, hard to tell front, back. so, perform prompt augment for only overhead view
        # If the results do not look good, you may use this direction prompts.
        # if angle_x < -np.pi / 4:
        #     direction = 1
        # else:
        #     direction = 0

        mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv[None, ...].cuda())
        mvp_list.append(mvp[None, ...].cuda())
        campos_list.append(campos[None, ...].cuda())
        direction_list.append(direction)

    cam = {
        'mv': torch.cat(mv_list, dim=0),
        'mvp': torch.cat(mvp_list, dim=0),
        'campos': torch.cat(campos_list, dim=0),
        'direction': np.array(direction_list, dtype=np.int32),
        'resolution': iter_res,
        'spp': spp
    }
    return cam


def sample_circle_view(n_view, elev, cam_radius, res=[512, 512], cam_near_far=[0.1, 1000.0], spp=1):
    iter_res = res
    fovy = np.deg2rad(45)
    proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Random rotation/translation matrix for optimization.
    mv_list, mvp_list, campos_list, direction_list = [], [], [], []
    angles_y = np.linspace(0.0, 2 * np.pi, n_view)
    for view_i in range(n_view):
        angle_x = elev
        angle_y = angles_y[view_i]
        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi / 8 or angle_y > 15 * np.pi / 8:
            direction = 0
        elif angle_y > np.pi / 8 and angle_y <= 7 * np.pi / 8:
            direction = 1
        elif angle_y > 7 * np.pi / 8 and angle_y <= 9 * np.pi / 8:
            direction = 2
        elif angle_y > 9 * np.pi / 8 and angle_y <= 15 * np.pi / 8:
            direction = 3

        mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv[None, ...].cuda())
        mvp_list.append(mvp[None, ...].cuda())
        campos_list.append(campos[None, ...].cuda())
        direction_list.append(direction)

    cam = {
        'mv': torch.cat(mv_list, dim=0),
        'mvp': torch.cat(mvp_list, dim=0),
        'campos': torch.cat(campos_list, dim=0),
        'direction': np.array(direction_list, dtype=np.int32),
        'resolution': iter_res,
        'spp': spp
    }
    return cam


def create_video(img_path, out_path, fps=60):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    command = ['/usr/bin/ffmpeg', '-y', '-r', str(fps), '-i', img_path, \
               '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    subprocess.run(command)



def imgcat(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[imgcat] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
#

def load_obj_uv(obj_path, device):
    vert, face, aux = load_obj(obj_path, device=device)
    vt = aux.verts_uvs
    ft = face.textures_idx
    vt = torch.cat((vt[:, [0]], 1.0 - vt[:, [1]]), dim=1)
    return ft, vt, face.verts_idx, vert



