import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
from deformationcytometer.detection.includes.regionprops import fit_ellipses_regionprops


class Segmentation():

    def __init__(self, img_shape=None, **kwargs):
        self.unet = UNet().create_model((img_shape[0], img_shape[1], 1), 1, d=8)
        network_path ="/home/user/Desktop/2020_Deformation_Cytometer/models_local/weights_andy_spatial/andyUnet_andy_spatial_weighting_w2_10_w3_1_n200__20201022-140138.h5"
        self.unet.load_weights(network_path)

    def segmentation(self, img):
            if len(img.shape) == 3:
                img = img[:,:,0]
            img = (img - np.mean(img)) / np.std(img).astype(np.float32)
            prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
            ellipses = fit_ellipses_regionprops(prediction_mask)
            return prediction_mask, ellipses
'''
     self.db.setMask(image=self.cp.getImage(), data=prediction_mask.astype(np.uint8))
self.db.setEllipse(image=im, x=x, y=y, width=region.major_axis_length, height=region.minor_axis_length,
                               angle=ellipse_angle, type=self.marker_type_cell2)
self.cp.reloadMask()
'''