import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
from deformationcytometer.detection.includes.UNETmodels_andy import UNet_gl
from deformationcytometer.detection.includes.regionprops import fit_ellipses_regionprops



class Segmentation():

    def __init__(self, img_shape=None, **kwargs):
        self.unet = UNet_gl().create_model((img_shape[0], img_shape[1], 1), 1, d=8)
        network_path ="/home/user/Downloads/andyUnet_andy_bens_network_long_n300_20201027-082115_checkpoint.h5"
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