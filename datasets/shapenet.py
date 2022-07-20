from pathlib import Path
import os
import json
import imageio
import torch
from torch.utils.data import Dataset
import numpy as np

class ShapenetDataset(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """
    def __init__(self, all_folders, num_views):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains indiviual scene info
            num_views (int): number of views to return for each scene
        """
        super().__init__()
        self.all_folders = all_folders
        self.num_views = num_views

    def __getitem__(self, idx):
        folderpath = self.all_folders[idx]
        meta_path = folderpath.joinpath("transforms.json")
        with open(meta_path, "r") as meta_file:
            meta_data = json.load(meta_file)
        
        all_imgs = []
        all_poses = []
        for frame_idx in range(self.num_views):
            frame = meta_data["frames"][frame_idx]

            img_name = f"{Path(frame['file_path']).stem}.png"
            img_path = folderpath.joinpath(img_name)
            img = imageio.imread(img_path)
            all_imgs.append(torch.as_tensor(img, dtype=torch.float))

            pose = frame["transform_matrix"]
            all_poses.append(torch.as_tensor(pose, dtype=torch.float))

        all_imgs = torch.stack(all_imgs, dim=0) / 255.
        # composite the images to a white background
        all_imgs = all_imgs[...,:3] * all_imgs[...,-1:] + 1-all_imgs[...,-1:]

        all_poses = torch.stack(all_poses, dim=0)

        # all images of a scene has the same camera intrinsics
        H, W = all_imgs[0].shape[:2]
        camera_angle_x = meta_data["camera_angle_x"]
        camera_angle_x = torch.as_tensor(camera_angle_x, dtype=torch.float)
        
        # camera angle equation: tan(angle/2) = (W/2)/focal
        focal = 0.5 * W / torch.tan(0.5 * camera_angle_x)
        hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

        # all shapenet scenes are bounded between 2. and 6.
        near = 2.
        far = 6.
        bound = torch.as_tensor([near, far], dtype=torch.float)

        return all_imgs, all_poses, hwf, bound
    
    def __len__(self):
        return len(self.all_folders)

class ShapenetRendered(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """
    def __init__(self, args, all_folders, num_views):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains indiviual scene info
            num_views (int): number of views to return for each scene
        """
        super().__init__()
        
        self.all_folders = all_folders
        self.num_views = num_views
        self.cam_frame = args.cam_frame
        self.near = args.near
        self.far = args.far
        
    def __getitem__(self, idx):
        folderpath = self.all_folders[idx]
        rgb_path = folderpath.joinpath("rgb")
        pose_path = folderpath.joinpath("pose")
        intrinsics_path = folderpath.joinpath("intrinsics.txt")
        
        with open(intrinsics_path, "r") as f:
            lines = f.readlines()
            # NOTE: since the nerf-meta also uses rendered shapenet as training data, 
            # the camera center in rendering is set to (H/2, W/2), which seems to be 
            # default setting of blender? In custom dataset, xc=yc=H/2=W/2
            focal, xc, yc, _ = [float(i) for i in lines[0].strip().split()]
        
        all_imgs = []
        all_poses = []
        for frame_idx in range(self.num_views):
            frame = str(frame_idx).zfill(6)
            img_path = rgb_path.joinpath(f"{frame}.png")
            img = imageio.imread(img_path)
            all_imgs.append(torch.as_tensor(img, dtype=torch.float))
            
            extrinsics_path = pose_path.joinpath(f"{frame}.txt")
            pose = np.loadtxt(extrinsics_path).reshape((4,4))
            # shapenet frame to z-axis upwards world frame
            pose = np.array([[-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]]) @ pose
            # convert camera frame from opencv convention to blender convention
            if self.cam_frame == "blender":
                pose = pose @ np.array(
                    [[1, 0,  0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0,  0, 1]]
                )
            all_poses.append(torch.as_tensor(pose, dtype=torch.float))

        all_imgs = torch.stack(all_imgs, dim=0) / 255.
        # composite the images to a white background
        all_imgs = all_imgs[...,:3] * all_imgs[...,-1:] + 1-all_imgs[...,-1:]

        all_poses = torch.stack(all_poses, dim=0)

        # all images of a scene has the same camera intrinsics
        H, W = all_imgs[0].shape[:2]
        hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

        bound = torch.as_tensor([self.near, self.far], dtype=torch.float)

        return all_imgs, all_poses, hwf, bound
    
    def __len__(self):
        return len(self.all_folders)

class ReplicaTest(Dataset):
    """
    returns the images, poses and instrinsics of a partiucular scene
    """
    def __init__(self, args, root_path, num_test=1):
        """
        Args:
            all_folders (list): list of folder paths. each folder contains indiviual scene info
            num_views (int): number of views to return for each scene
        """
        super().__init__()
        self.root_path = root_path
        self.images = os.listdir(os.path.join(self.root_path, "rgb"))
        self.num_test = num_test
        self.near = args.near
        self.far = args.far
        
        
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.root_path, "rgb")
        pose_path = os.path.join(self.root_path, "pose")
        intrinsics_path = os.path.join(self.root_path, "intrinsics.txt")
        
        with open(intrinsics_path, "r") as f:
            lines = f.readlines()
            # NOTE: since the nerf-meta also uses rendered shapenet as training data, 
            # the camera center in rendering is set to (H/2, W/2), which seems to be 
            # default setting of blender? In custom dataset, xc=yc=H/2=W/2
            focal, xc, yc, _ = [float(i) for i in lines[0].strip().split()]
        
        all_imgs = []
        all_poses = []
        
        # return test train image + test eval image 
        train_ids = list(range(self.num_test * idx, self.num_test * (idx+1)))
        eval_ids = list(set(range(len(self.images))) - set(train_ids))
        frame_ids = train_ids + eval_ids
        print(f"cross fold eval:{idx} train {train_ids}; eval {eval_ids}")
        
        for frame_idx in frame_ids:
            frame = str(frame_idx).zfill(6)
            img_path = os.path.join(rgb_path, f"{frame}.png")
            img = imageio.imread(img_path)
            all_imgs.append(torch.as_tensor(img, dtype=torch.float))
            
            extrinsics_path = os.path.join(pose_path, f"{frame}.txt")
            pose = np.loadtxt(extrinsics_path).reshape((4,4))
            all_poses.append(torch.as_tensor(pose, dtype=torch.float))

        all_imgs = torch.stack(all_imgs, dim=0) / 255.
        # composite the images to a white background
        all_imgs = all_imgs[...,:3] * all_imgs[...,-1:] + 1-all_imgs[...,-1:]

        all_poses = torch.stack(all_poses, dim=0)

        # all images of a scene has the same camera intrinsics
        H, W = all_imgs[0].shape[:2]
        hwf = torch.as_tensor([H, W, focal], dtype=torch.float)

        # all shapenet scenes are bounded between 2. and 6.
        bound = torch.as_tensor([self.near, self.far], dtype=torch.float)

        return all_imgs, all_poses, hwf, bound
    
    def __len__(self):
        return int(np.floor(len(self.images) / self.num_test))

def build_shapenet(args, image_set, dataset_root, splits_path, num_views):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: num of views to return from a single scene
    """
    root_path = Path(dataset_root)
    splits_path = Path(splits_path)
    with open(splits_path, "r") as splits_file:
        splits = json.load(splits_file)
    
    
    dataset_source = args.dataset_source
    if dataset_source == "learnit":
        all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[image_set])]
        if hasattr(args, 'train_objects_max'):
            max_obj = args.train_objects_max
            all_folders = all_folders[:max_obj]
        dataset = ShapenetDataset(all_folders, num_views)   
        
    elif dataset_source == "render":
        all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[image_set])]
        if hasattr(args, 'train_objects_max'):
            max_obj = args.train_objects_max
            all_folders = all_folders[:max_obj]
        dataset = ShapenetRendered(args, all_folders, num_views)
        
    elif dataset_source == "replica_test":
        dataset = ReplicaTest(args, root_path, num_test=1)
        
    else:
        print(f"dataset source {dataset_source} not supported")
        raise NotImplementedError

    return dataset