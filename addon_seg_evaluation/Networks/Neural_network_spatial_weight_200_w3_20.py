
import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
from deformationcytometer.detection.includes.regionprops import mask_to_cells


class Segmentation():

    def __init__(self, img_shape=None, pixel_size=None, r_min=None, frame_data=None, edge_dist=15, channel_width=0,
                 **kwargs):
        # rmin in µm
        # channel_width in pixel ?
        # pixel_size in m / not in mµ!!
        self.unet = UNet().create_model((img_shape[0], img_shape[1], 1), 1, d=8)
        network_path ="/home/user/Desktop/2020_Deformation_Cytometer/models_local/weights_andy_spatial/andyUnet_andy_spatial_weighting_w2_10_w3_20_n200__20201022-120838.h5"
        self.unet.load_weights(network_path)

        self.pixel_size = pixel_size
        self.r_min = r_min
        self.frame_data = frame_data if frame_data is not frame_data else {}
        self.edge_dist = edge_dist
        self.config = {}
        self.config["channel_width_px"] = channel_width
        self.config["pixel_size_m"] = pixel_size

    def segmentation(self, img):
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = (img - np.mean(img)) / np.std(img).astype(np.float32)
        prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
        cells = mask_to_cells(prediction_mask, img, self.config, self.r_min, self.frame_data, self.edge_dist)
        return prediction_mask, cells
