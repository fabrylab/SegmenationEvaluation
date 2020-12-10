import numpy as np
from addon_seg_evaluation.UNETmodel import UNet
import clickpoints
from matplotlib.path import Path
from deformationcytometer.detection.includes.regionprops import mask_to_cells
from skimage.morphology import binary_closing, remove_small_holes, label
from skimage.measure import regionprops
import fill_voids
from deformationcytometer.detection.includes.andy_data_handling import preprocess_batch

def mask_to_cells_edge(prediction_mask, im, config, r_min, frame_data, edge_dist=15, return_mask=False):
    r_min_pix = r_min / config["pixel_size_m"] / 1e6
    edge_dist_pix = edge_dist / config["pixel_size_m"] / 1e6
    cells = []
    # TDOD: consider first applying binary closing operations to avoid impact of very small gaps in the cell border
    filled = fill_voids.fill(prediction_mask)
    # iterate over all detected regions
    for region in regionprops(label(filled), im):  # region props are based on the original image
        # checking if the anything was filled up by extracting the region form the original image
        # if no significant region was filled, we skip this object
        yc, xc = np.split(region.coords, 2, axis=1)
        if np.sum(~prediction_mask[yc.flatten(), xc.flatten()]) < 10:
            if return_mask:
                prediction_mask[yc.flatten(), xc.flatten()] = False
            continue
        elif return_mask:
            prediction_mask[yc.flatten(), xc.flatten()] = True

        a = region.major_axis_length / 2
        b = region.minor_axis_length / 2
        r = np.sqrt(a * b)

        if region.orientation > 0:
            ellipse_angle = np.pi / 2 - region.orientation
        else:
            ellipse_angle = -np.pi / 2 - region.orientation

        Amin_pixels = np.pi * (r_min_pix) ** 2  # minimum region area based on minimum radius
        # filtering cells close to left and right image edge
        # usually cells do not come close to upper and lower image edge
        x_pos = region.centroid[1]
        dist_to_edge =  np.min([x_pos, prediction_mask.shape[1] - x_pos])

        if region.area >= Amin_pixels and dist_to_edge > edge_dist_pix:  # analyze only regions larger than 100 pixels,
            # and only of the canny filtered band-passed image returend an object

            # the circumference of the ellipse
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

            # %% compute radial intensity profile around each ellipse
            theta = np.arange(0, 2 * np.pi, np.pi / 8)

            i_r = np.zeros(int(3 * r))
            for d in range(0, int(3 * r)):
                # get points on the circumference of the ellipse
                x = d / r * a * np.cos(theta)
                y = d / r * b * np.sin(theta)
                # rotate the points by the angle fo the ellipse
                t = ellipse_angle
                xrot = (x * np.cos(t) - y * np.sin(t) + region.centroid[1]).astype(int)
                yrot = (x * np.sin(t) + y * np.cos(t) + region.centroid[0]).astype(int)
                # crop for points inside the iamge
                index = (xrot < 0) | (xrot >= im.shape[1]) | (yrot < 0) | (yrot >= im.shape[0])
                x = xrot[~index]
                y = yrot[~index]
                # average over all these points
                i_r[d] = np.mean(im[y, x])

            # define a sharpness value
            sharp = (i_r[int(r + 2)] - i_r[int(r - 2)]) / 5 / np.std(i_r)

            # %% store the cells
            yy = region.centroid[0] - config["channel_width_px"] / 2
            yy = yy * config["pixel_size_m"] * 1e6

            data = {}
            data.update(frame_data)
            data.update({
                          "x_pos": region.centroid[1],  # x_pos
                          "y_pos": region.centroid[0],  # y_pos
                          "radial_pos": yy,                  # RadialPos
                          "long_axis": float(format(region.major_axis_length)) * config["pixel_size_m"] * 1e6,  # LongAxis
                          "short_axis": float(format(region.minor_axis_length)) * config["pixel_size_m"] * 1e6,  # ShortAxis
                          "angle": np.rad2deg(ellipse_angle),  # angle
                          "irregularity": region.perimeter / circum,  # irregularity
                          "solidity": region.solidity,  # solidity
                          "sharpness": sharp,  # sharpness
            })
            cells.append(data)
    if return_mask:
        return cells, prediction_mask
    else:
        return cells
class Segmentation():

    def __init__(self,network_path=None, img_shape=None, pixel_size=None, r_min=None, frame_data=None, edge_dist=15, channel_width=0, edge_only=False, return_mask=True,d=8, **kwargs):
        # rmin in µm
        # channel_width in pixel ?
        # pixel_size in m / not in mµ!!
        self.unet = UNet().create_model((img_shape[0], img_shape[1], 1), 1, d=d)
        #network_path ="/home/user/Desktop/2020_Deformation_Cytometer/models_local/weights_andy_transfer_learning1/andyUnet_andy_transfer_long_n200__20201006-155443.h5"
        self.unet.load_weights(network_path)

        self.pixel_size = pixel_size
        self.r_min = r_min
        self.frame_data = frame_data if frame_data is not frame_data else {}
        self.edge_dist = edge_dist
        self.config = {}
        self.config["channel_width_px"] = channel_width
        self.config["pixel_size_m"] = pixel_size
        self.edge_only = edge_only
        self.return_mask = return_mask



    def search_cells(self, prediction_mask, img):

        if self.edge_only:
            if self.return_mask:
                cells, prediction_mask = mask_to_cells_edge(prediction_mask, img, self.config, self.r_min,
                                                            self.frame_data, self.edge_dist,
                                                            return_mask=self.return_mask)
            else:
                cells = mask_to_cells_edge(prediction_mask, img, self.config, self.r_min, self.frame_data,
                                           self.edge_dist, return_mask=self.return_mask)
        else:
            cells = mask_to_cells(prediction_mask, img, self.config, self.r_min, self.frame_data, self.edge_dist)

        return prediction_mask, cells


    def segmentation(self, img):
        # image batch
        if len(img.shape) == 4:
            img = preprocess_batch(img)
            prediction_mask = self.unet.predict(img) > 0.5
            cells = []
            for i in range(prediction_mask.shape[0]):
                _ , cells_ = self.search_cells(prediction_mask[i,:,:,0], img[i,:,:,0])
                cells.extend(cells_)
            prediction_mask = None
        # single image
        elif len(img.shape) == 2:

            img = (img - np.mean(img)) / np.std(img).astype(np.float32)
            prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
            prediction_mask, cells = self.search_cells(prediction_mask, img)
        else:
            raise Exception("incorrect image shape: img.shape == " + str(img.shape))
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
