import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
from deformationcytometer.detection.includes.regionprops import fit_ellipses_regionprops
import clickpoints
from matplotlib.path import Path


class Segmentation():

    def __init__(self, img_shape=None, **kwargs):
        self.unet = UNet().create_model((img_shape[0], img_shape[1], 1), 1, d=8)
        network_path ="/home/user/Desktop/2020_Deformation_Cytometer/models_local/Unet_0-0-5_fl_RAdam_20200610-141144.h5"
        self.unet.load_weights(network_path)

    def segmentation(self, img):
            if len(img.shape) == 3:
                img = img[:,:,0]
            img = (img - np.mean(img)) / np.std(img).astype(np.float32)
            prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
            ellipses = fit_ellipses_regionprops(prediction_mask)
            return prediction_mask, ellipses

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

        ellipses = fit_ellipses_regionprops(mask)
        return mask, ellipses