import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import torch
import numpy as np
import random
import math
import copy
import argparse
import torch.nn.functional as F
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import time
from nvdiff_render.mesh import *
from nvdiff_render.render import *
from nvdiff_render.texture import *
from nvdiff_render.material import *
from nvdiff_render.obj import *
from utils import *
from dc_pbr import skip
from sd import StableDiffusion

glctx = dr.RasterizeCudaContext()
OBJAVERSE_PATH = './data'


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--decay', type=float, default=0)  # weight decay
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--decay_step', type=int, default=100)

    # training
    parser.add_argument('--sd_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--n_iter', type=int, default=1500)  # can be increased
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--sd_min', type=float, default=0.2)
    parser.add_argument('--sd_max', type=float, default=0.98)
    parser.add_argument('--sd_min_l', type=float, default=0.2)
    parser.add_argument('--sd_min_r', type=float, default=0.3)
    parser.add_argument('--sd_max_l', type=float, default=0.5)
    parser.add_argument('--sd_max_r', type=float, default=0.98)
    parser.add_argument('--bg', type=float, default=0.25)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sd_minmax_anneal', type=eval, default=True, choices=[True, False])
    parser.add_argument('--n_view', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--env_scale', type=float, default=2.0)
    parser.add_argument('--envmap', type=str, default='data/irrmaps/mud_road_puresky_4k.hdr')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--gd_scale', type=int, default=100)

    args = parser.parse_args()
    args.kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
    args.kd_max = [1.0, 1.0, 1.0, 1.0]
    args.ks_min = [0.0, 0.08, 0.0]  # Limits for ks
    args.ks_max = [1.0, 1.0, 1.0]
    args.nrm_min = [-0.1, -0.1, 0.0]  # Limits for normal map
    args.nrm_max = [0.1, 0.1, 1.0]
    return args


def seed_all(args):
    # Constrain all sources of randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_model(args):
    # MLP Settings
    input_depth = 3
    net = skip(input_depth, 9,
               num_channels_down=[128] * 5,
               num_channels_up=[128] * 5,
               num_channels_skip=[128] * 5,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(torch.cuda.FloatTensor)

    params = list(net.parameters())

    lgt = light.load_env(args.envmap, scale=args.env_scale)
    for p in lgt.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(params, args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)

    return net, lgt, optim, activate_scheduler, lr_scheduler


def report_process(i, loss, exp_name):
    full_loss = 0
    log_message = f'[{exp_name}] iter: {i} '
    for loss_type, loss_val in loss.items():
        full_loss += loss_val
        log_message += f'{loss_type}: {"%.3f" % loss_val} '
    loss['L_all'] = full_loss
    print(log_message)


def get_template_normal(h=512, w=512):
    return torch.cat([torch.zeros((h, w, 1), device=device), torch.zeros((h, w, 1), device=device),
                      torch.ones((h, w, 1), device=device)], dim=-1)[None, ...]


def compute_sd_step(min, max, iter_frac):
    step = (max - (max - min) * math.sqrt(iter_frac))
    return step


def main(args, guidance):
    exp_name = time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name
    output_dir = os.path.join('./logs', exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # seed_all(args)

    # Get text prompt and tokenize it
    sd_prompt = ", ".join(
        (f"a DSLR photo of {args.identity}", "best quality, high quality, extremely detailed, good geometry"))

    # load obj and read uv information
    args.obj_path = os.path.join(OBJAVERSE_PATH, args.objaverse_id, 'mesh.obj')
    obj_f_uv, obj_v_uv, obj_f, obj_v = load_obj_uv(obj_path=args.obj_path, device=device)

    # initialize template mesh
    mesh_t = Mesh(obj_v, obj_f, v_tex=obj_v_uv, t_tex_idx=obj_f_uv)
    mesh_t = unit_size(mesh_t)
    mesh_t = auto_normals(mesh_t)
    mesh_t = compute_tangents(mesh_t)

    input_uv_ = torch.randn((3, 512, 512), device=device)

    # scale input
    input_uv = (input_uv_ - torch.mean(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)) / torch.std(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)

    network_input = copy.deepcopy(input_uv.unsqueeze(0))

    # get model and optimizer
    net, lgt, optim, activate_scheduler, lr_scheduler = get_model(args)

    # get text embedding
    neg_prompt = 'deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke'
    text_z = []

    for d in ['front', 'side', 'back', 'overhead']:
        # construct dir-encoded text
        text_z.append(guidance.get_text_embeds([f"{sd_prompt}, {d} view"], [neg_prompt], 1))
    text_z = torch.stack(text_z, dim=0)

    kd_min, kd_max = torch.tensor(args.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(args.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(args.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(args.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(args.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(args.nrm_max, dtype=torch.float32, device='cuda')
    nrm_t = get_template_normal()  # (512, 512, 3)

    # Main training loop
    for step in tqdm(range(args.n_iter + 1)):
        cur_iter_frac = step / args.n_iter
        losses = {}
        optim.zero_grad()
        # build mips
        lgt.build_mips()
        with torch.no_grad():
            mesh = copy.deepcopy(mesh_t)

        net_output = net(network_input)  # [B, 9, H, W]
        pred_tex = net_output.permute(0, 2, 3, 1)
        pred_kd = pred_tex[..., :-6]
        pred_ks = pred_tex[..., -6:-3]
        pred_n = F.normalize((pred_tex[..., -3:] * 2.0 - 1.0) + nrm_t, dim=-1)

        pred_material = Material({
            'bsdf': 'pbr',
            'kd': Texture2D(pred_kd, min_max=[kd_min, kd_max]),
            'ks': Texture2D(pred_ks, min_max=[ks_min, ks_max]),
            'normal': Texture2D(pred_n, min_max=[nrm_min, nrm_max])
        })
        pred_material['kd'].clamp_()
        pred_material['ks'].clamp_()
        pred_material['normal'].clamp_()

        mesh.material = pred_material

        cam = sample_view_obj(args.n_view, cam_radius=3.25)
        buffers = render_mesh(glctx, mesh, cam['mvp'], cam['campos'], lgt, cam['resolution'],
                              spp=cam['spp'], msaa=True, background=None, bsdf='pbr')
        pred_obj_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
        pred_obj_ws = buffers['shaded'][..., 3].unsqueeze(1)  # [B, 1, H, W]
        obj_image = pred_obj_rgb * pred_obj_ws + (1 - pred_obj_ws) * args.bg  # white bg

        # SDS losses
        all_pos, all_neg = [], []
        #
        text_z_iter = text_z[cam['direction']]
        #
        #
        for emb in text_z_iter:  # list of [2, S, -1]
            pos, neg = emb.chunk(2)  # [1, S, -1]
            all_pos.append(pos)
            all_neg.append(neg)
        text_embedding = torch.cat(all_pos + all_neg, dim=0)  # [2b, S, -1]

        sd_min_step = compute_sd_step(args.sd_min_l, args.sd_min_r, cur_iter_frac)
        sd_max_step = compute_sd_step(args.sd_max_l, args.sd_max_r, cur_iter_frac)

        # # compute sds_loss for the body
        sd_loss = guidance.batch_train_step(text_embedding, obj_image,
                                            guidance_scale=args.gd_scale,
                                            min_step=sd_min_step,
                                            max_step=sd_max_step)

        total_loss = sd_loss
        losses['L_sds'] = sd_loss.item()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.sd_max_grad_norm)

        optim.step()
        lr_scheduler.step()

        if step % args.log_freq == 0 and args.logging:
            with torch.no_grad():
                report_process(step, losses, exp_name)
                mtl_file = os.path.join(output_dir, 'mesh.mtl')
                save_mtl(mtl_file, mesh.material, step=step)
                torchvision.utils.save_image(obj_image[0], os.path.join(output_dir, f'obj_{step:04}.jpg'))

    with torch.no_grad():
        #
        vis_mesh = copy.deepcopy(mesh_t)
        final_pred = net(network_input)
        final_tex = final_pred.permute(0, 2, 3, 1).contiguous()

        final_kd = final_tex[..., :-6]
        final_ks = final_tex[..., -6:-3]
        final_n = F.normalize((final_tex[..., -3:] * 2.0 - 1.0) + nrm_t, dim=-1)
        circle_n_view = 120

        for elev in [-np.pi / 4, 0.0]:
            final_cam = sample_circle_view(n_view=circle_n_view, elev=elev, cam_radius=3.25)

            final_material = Material({
                'bsdf': 'pbr',
                'kd': Texture2D(final_kd, min_max=[kd_min, kd_max]),
                'ks': Texture2D(final_ks, min_max=[ks_min, ks_max]),
                'normal': Texture2D(final_n, min_max=[nrm_min, nrm_max])
            })
            final_material['kd'].clamp_()
            final_material['ks'].clamp_()
            final_material['normal'].clamp_()
            vis_mesh.material = final_material

            write_obj(output_dir, vis_mesh)

            final_lgt = lgt
            final_buffers = render_mesh(glctx, vis_mesh, final_cam['mvp'], final_cam['campos'], final_lgt,
                                        final_cam['resolution'], spp=final_cam['spp'], msaa=True, background=None,
                                        bsdf='pbr')

            final_obj_rgb = final_buffers['shaded'].permute(0, 3, 1, 2).contiguous()
            final_obj_ws = final_buffers['shaded'][..., 3].unsqueeze(1)  # [B, 1, H, W]
            vis_mesh_img = final_obj_rgb * final_obj_ws + (1 - final_obj_ws) * 1  # white bg, float32, [B, 3, H, W]

            # # save final front body image
            if elev == 0.0:
                os.makedirs(os.path.join(output_dir, 'view_front'), exist_ok=True)
            else:
                os.makedirs(os.path.join(output_dir, 'view_top'), exist_ok=True)
            for idx in range(circle_n_view):
                if idx == 0:
                    if elev == 0.0:
                        torchvision.utils.save_image(final_obj_rgb[idx], os.path.join(output_dir, "final_front.png"))
                    else:
                        torchvision.utils.save_image(final_obj_rgb[idx], os.path.join(output_dir, "final_top.png"))
                if elev == 0.0:
                    torchvision.utils.save_image(vis_mesh_img[idx], os.path.join(output_dir, 'view_front', f'{idx:04}.png'))
                else:
                    torchvision.utils.save_image(vis_mesh_img[idx], os.path.join(output_dir, 'view_top', f'{idx:04}.png'))


if __name__ == '__main__':
    args = parse_args()

    mesh_dicts = {
        '9ce8ab24383c4c93b4c1c7c3848abc52': 'a pretzel',
    }

    # load stable-diffusion model
    guidance = StableDiffusion(device, min=args.sd_min, max=args.sd_max)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False

    # iterate through the renderpeople items
    for obj_id, caption in mesh_dicts.items():
        args.exp_name = '_'.join((caption.split(' ')[1:] + [obj_id[:6]]))
        args.objaverse_id = obj_id
        args.identity = caption
        main(args, guidance)
