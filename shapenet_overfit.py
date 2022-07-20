# learn single object in shapenet as a normal nerf training process
# check if dataset is correct

# TODO: test if table dataset can be overfit 
from pathlib import Path
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.shapenet import build_shapenet
from models.nerf import build_nerf
from utils.shape_video import create_360_video
from models.rendering import get_rays_shapenet, sample_points, volume_render

DEBUG=False

def overfit(args, model, optim, imgs, poses, hwf, bound):
    """
    test-time-optimize the meta trained model on available views
    """
    pixels = imgs.reshape(-1, 3)

    rays_o, rays_d = get_rays_shapenet(hwf, poses)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    num_rays = rays_d.shape[0]
    for step in tqdm(range(args.tto_steps), desc="Overfit steps:"):
        indices = torch.randint(num_rays, size=[args.tto_batchsize])
        raybatch_o, raybatch_d = rays_o[indices], rays_d[indices]
        pixelbatch = pixels[indices] 
        t_vals, xyz = sample_points(raybatch_o, raybatch_d, bound[0], bound[1],
                                    args.num_samples, perturb=True)
        
        optim.zero_grad()
        rgbs, sigmas = model(xyz)
        colors = volume_render(rgbs, sigmas, t_vals, white_bkgd=True)
        loss = F.mse_loss(colors, pixelbatch)
        loss.backward()
        optim.step()


def report_result(args, model, imgs, poses, hwf, bound):
    """
    report view-synthesis result on heldout views
    """
    ray_origins, ray_directions = get_rays_shapenet(hwf, poses)

    view_psnrs = []
    for img, rays_o, rays_d in zip(imgs, ray_origins, ray_directions):
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        t_vals, xyz = sample_points(rays_o, rays_d, bound[0], bound[1],
                                    args.num_samples, perturb=False)
        
        synth = []
        num_rays = rays_d.shape[0]
        with torch.no_grad():
            for i in range(0, num_rays, args.test_batchsize):
                rgbs_batch, sigmas_batch = model(xyz[i:i+args.test_batchsize])
                color_batch = volume_render(rgbs_batch, sigmas_batch,
                                            t_vals[i:i+args.test_batchsize],
                                            white_bkgd=True)
                synth.append(color_batch)
            synth = torch.cat(synth, dim=0).reshape_as(img)
            if DEBUG:
                # import numpy as np 
                # import matplotlib.pyplot as plt
                # img = all_imgs.cpu().numpy()[3,:,:,:] 
                fig = plt.figure(figsize=(10,5))
                fig.add_subplot(1,2,1)
                vis_img = img.cpu().numpy()
                plt.imshow(vis_img)
                fig.add_subplot(1,2,2)
                vis_synth = synth.detach().cpu().numpy()
                plt.imshow(vis_synth)
                plt.show()
            
            error = F.mse_loss(img, synth)
            psnr = -10*torch.log10(error)
            view_psnrs.append(psnr)
    
    scene_psnr = torch.stack(view_psnrs).mean()
    return scene_psnr


def train():
    parser = argparse.ArgumentParser(description='shapenet train on one object')
    parser.add_argument('--config', type=str, required=True,
                    help='config file for the shape class (cars, chairs or lamps)')    
    # parser.add_argument('--weight-path', type=str, required=True,
    #                     help='path to the meta-trained weight file')
    parser.add_argument('--object_idx', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_shapenet(args, image_set="train", dataset_root=args.dataset_root, splits_path=args.splits_path, 
                               num_views=args.train_views)
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    
    model = build_nerf(args)
    model.to(device)

    savedir = Path(args.savedir)
    savedir.mkdir(exist_ok=True)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # for idx, (imgs, poses, hwf, bound) in enumerate(test_loader):
    imgs, poses, hwf, bound = train_set[args.object_idx]
    imgs, poses, hwf, bound = imgs.to(device), poses.to(device), hwf.to(device), bound.to(device)
    imgs, poses, hwf, bound = imgs.squeeze(), poses.squeeze(), hwf.squeeze(), bound.squeeze()

    optim = torch.optim.SGD(model.parameters(), args.tto_lr)

    overfit(args, model, optim, imgs, poses, hwf, bound)
    scene_psnr = report_result(args, model, imgs, poses, hwf, bound)
    scene_id = str(train_set.all_folders[args.object_idx]).split("/")[-1]
    create_360_video(args, model, hwf, bound, device, scene_id, savedir)
    
    print(f"scene {scene_id}, psnr:{scene_psnr:.3f}, video created")
    object_class = args.dataset_root.split("/")[-2]
    torch.save(
        {
            'train_step': args.tto_steps,
            'meta_model_state_dict': model.state_dict(),
            'meta_optim_state_dict': optim.state_dict(),
        }, 
        os.path.join(args.outdir, f'{object_class}_{scene_id}_{args.tto_steps}.pth')
    )
    
if __name__ == '__main__':
    train()