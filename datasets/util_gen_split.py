import os 
import numpy as np
import random 
import json

if __name__ == "__main__":
    
    seed = 0
    num_train = -1
    num_val = 50
    num_test = 100
    class_name = "table"
    class_id = "04379243"
    data_dir = "/media/junting/SSD_data/meta_nerf/shapenet"
    splits_file = f"/media/junting/SSD_data/meta_nerf/shapenet/{class_name}_splits.json"
    
    
    class_dir = os.path.join(data_dir, class_name, class_id)
    inst_ids = os.listdir(class_dir)
    
    random.seed(seed)
    random.shuffle(inst_ids)
    
    val_ids = inst_ids[:num_val]
    test_ids = inst_ids[num_val: num_val+num_test]
    train_ids = inst_ids[num_val+num_test:]
    
    split_dict = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids 
    }
    
    with open(splits_file, "w") as f:
        json.dump(split_dict, f)