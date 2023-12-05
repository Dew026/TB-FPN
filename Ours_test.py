import torch
from Ours_net import OurModel
import os
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from tools import *
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    model = OurModel()  # 创建模型实例
    
    # 加载模型权重
    model = torch.load('Our_teacher.pth')
    
    # 设置模型为评估模式（可选）
    model.eval()
    
    
    DATA_DIR = './dataset'
    
    x_test_dir = os.path.join(DATA_DIR, 'train')
    y_test_dir = os.path.join(DATA_DIR, 'trainannot')
    
    # create segmentation model with pretrained encoder
    
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    
    test_dataset_vis = Dataset(
        x_test_dir, 
        y_test_dir, 
    )
    
    DEVICE = 'cuda'
    ids = test_dataset_vis.ids
    
    for i in tqdm(range(3)):
        
        image_vis = test_dataset_vis[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        

        img = cv2.imread(os.path.join(x_test_dir, ids[i]))
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask, pre_boundary = pr_mask[0], pr_mask[1]
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pre_boundary = (pre_boundary.squeeze().cpu().numpy())
        
        mask_path = os.path.join(y_test_dir, ids[i])
        pr_mask = np.uint8(pr_mask*255)
        
        plt.imshow(pre_boundary)
    cv2.imwrite('./train_2p.png', pr_mask)
            
        # visualize(
        #     img=img,
        #     image=image_vis, 
        #     ground_truth_mask=gt_mask[0], 
        #     predicted_mask=pr_mask,
        #     predicted_boundary=pre_boundary,
        # )
