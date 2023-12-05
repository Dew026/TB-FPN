import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
import torch

class OurModel(SegmentationModel):

    def __init__(self):
        super().__init__()
        # se_resnext50_32x4d
        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid' 
        
        self.FPN1 = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            activation=ACTIVATION,
        )
        
        self.FPN2 = smp.FPN(
            in_channels=4,
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            activation=ACTIVATION,
        )


    def forward(self, image):
        boundary = self.FPN1(image)
        temp = torch.cat((image, boundary), dim=1)
        mask = self.FPN2(temp)
        return mask, boundary

if __name__ == "__main__":
    input_data = torch.randn(8, 3, 320, 640)
    model = OurModel()
    
    out = model(input_data)