import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
import clickpoints
from matplotlib.path import Path
from deformationcytometer.detection.includes.regionprops import mask_to_cells


class Segmentation():

    def __init__(self, img_shape=None, pixel_size=None, r_min=None, frame_data=None, edge_dist=15, channel_width=0, **kwargs):
        # rmin in µm
        # channel_width in pixel ?
        # pixel_size in m / not in mµ!!
        self.unet = UNet().create_model((img_shape[0], img_shape[1], 1), 1, d=8)
        network_path ="/home/user/Desktop/2020_Deformation_Cytometer/models_local/weights_andy_transfer_learning1/andyUnet_andy_transfer_long_n200__20201006-155443.h5"
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
                img = img[:,:,0]
            img = (img - np.mean(img)) / np.std(img).astype(np.float32)
            prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
            cells = mask_to_cells(prediction_mask, img, self.config, self.r_min, self.frame_data, self.edge_dist)
            return prediction_mask, cells


class FromOtherDB():
    def __init__(self, db_path=None, **kwargs):
        if db_path is not None:
            self.db = clickpoints.DataFile(db_path)

        im_shape =  self.db.getImages()[0].getShape()
        self.nx, self.ny = im_shape[1], im_shape[0]
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x, y = x.flatten(), y.flatten()
        self.points = np.vstack((x, y)).T

    def getMaskEllipse(self, frame):
        try:
            mask = self.db.getMask(frame=frame).data
            im = self.db.getImages(frame=frame)[0]
        except AttributeError:
            mask_shape = (self.ny, self.nx)
            mask = np.zeros(mask_shape, dtype=np.uint8)
            img_o = self.db.getImage(frame=frame)
            q_polys = self.db.getPolygons(image=img_o)
            for pol in q_polys:
                if np.shape(pol)[0] != 0:
                    polygon = np.array([[pol.points]])
                    if np.sum(polygon.shape) > 7:  # single point polygon can happen on accident when clicking
                        path = Path(polygon.squeeze())
                        grid = path.contains_points(self.points)
                        grid = grid.reshape(mask_shape)
                        mask += grid

        cells = mask_to_cells(mask, im.data, self.config, self.r_min, self.frame_data, self.edge_dist)
        return mask, cells
