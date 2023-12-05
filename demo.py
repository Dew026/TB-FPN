from tools import *
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

DATA_DIR = './dataset'
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
    
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

image, mask = train_dataset[7]

plt.subplot(121)
plt.imshow(image[0,:,:])
plt.subplot(122)
plt.imshow(mask[0,:,:])