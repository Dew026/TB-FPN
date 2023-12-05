from tools import *
import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from Ours_net import OurModel
from tqdm import tqdm

def Train(epoch = 40):
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid' 
    # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    DATA_DIR = './dataset'
    
    x_train_dir = os.path.join(DATA_DIR, 'train_student')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot_student')
    
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')
    
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')
    
    # create segmentation model with pretrained encoder
    model = torch.load("Our_student.pth")
    
    for name, param in model.named_parameters():
        if "FPN1" in name:
            param.requires_grad = False

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )


    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    loss = [smp.utils.losses.DiceLoss(), smp.utils.losses.MSELoss()]

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001),
    ])

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    max_score = 0
    
    for i in range(0, epoch):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

if __name__ == "__main__":
    Train()
# if __name__ == "__main__":
#     model = torch.load("Our_teacher.pth")
    
#     for name, param in model.named_parameters():
#         if "FPN1" in name:
#             param.requires_grad = False