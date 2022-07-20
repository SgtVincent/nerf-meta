import re
from imageio import save
import numpy as np 
import os 
import math
from PIL import Image 

def get_intrinsics(W, H):
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    f = W / 2.0 / math.tan(math.radians(hfov / 2.0))
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0
    return f, cx, cy

def create_frame(frame_id, out_id, label, pose, replica_dir, save_dir):
    
    rgb_img_path = os.path.join(replica_dir, "rgb", f"rgb_{frame_id}.png")
    depth_img_path = os.path.join(replica_dir, "depth", f"depth_{frame_id}.png")
    sem_img_path = os.path.join(replica_dir, "semantic_class", f"semantic_class_{frame_id}.png")
    rgb_img = np.asarray(Image.open(rgb_img_path))
    depth_img = np.asarray(Image.open(depth_img_path))
    sem_img = np.asarray(Image.open(sem_img_path))
    
    # TODO: only select one instance in segmentation image
    save_rgb_dir = os.path.join(save_dir, "rgb")
    if not os.path.exists(save_rgb_dir):
        os.makedirs(save_rgb_dir)
    save_depth_dir = os.path.join(save_dir, "depth")
    if not os.path.exists(save_depth_dir):
        os.makedirs(save_depth_dir)
    save_pose_dir = os.path.join(save_dir, "pose")
    if not os.path.exists(save_pose_dir):
        os.makedirs(save_pose_dir)
        
    mask_rgb_img = np.zeros_like(rgb_img)
    mask_rgb_img[sem_img == label] = rgb_img[sem_img == label]
    mask_depth_img = np.zeros_like(depth_img)
    mask_depth_img[sem_img == label] = depth_img[sem_img == label]

    Image.fromarray(mask_rgb_img).save(
        os.path.join(save_rgb_dir, f"{str(out_id).zfill(6)}.png")
    ) 
    Image.fromarray(mask_depth_img).save(
        os.path.join(save_depth_dir, f"{str(out_id).zfill(6)}.png") 
    )
    with open(os.path.join(save_pose_dir, f"{str(out_id).zfill(6)}.txt"), "w") as f:
        f.write(pose)
        
    return
    

if __name__ == "__main__":
    
    # data source 
    replica_datadir = "/media/junting/SSD_data/sem_nerf/room_0/Sequence_1"
    pose_file = f"{replica_datadir}/traj_w_c.txt"
    semantic_file = "/media/junting/SSD_data/sem_nerf/semantic_info/room_0/info_semantic.json"

    # target dir 
    frames_file = "/media/junting/SSD_data/meta_nerf/shapenet/replica_table/frames.txt"
    out_dir = "/media/junting/SSD_data/meta_nerf/shapenet/replica_table"
    
    class_id = "table"
    class_label = 80
    W=640
    H=480
    
    # save intrinsics.txt file 
    f, cx, cy = get_intrinsics(W, H)
    with open(os.path.join(out_dir, 'intrinsics.txt'),'w') as intrinsics_file:
        intrinsics_file.write('%f %f %f 0.\n'%(f, cx, cy))
        intrinsics_file.write('0. 0. 0.\n')
        intrinsics_file.write('1.\n')
        intrinsics_file.write('%d %d\n'%(W, H))
    
    with open(pose_file) as f:
        poses = [line.strip() for line in f]
    
    # read selected frames 
    with open(frames_file, "r") as f:
        for i, line in enumerate(f):
            frame_id = int(line.strip()) 
            pose = poses[frame_id]
            create_frame(frame_id, i, class_label, pose, replica_datadir, out_dir)
            
            
    