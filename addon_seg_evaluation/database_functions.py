import clickpoints
from localAlignement.parameters_and_strings import line_color, line_name, mask_name, mask_color
from localAlignement.lines_masks import *
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt









def setup_masks(db):
    # deleteing all eyisting masks

    if not all([mask_name in m.name for m in db.getMaskTypes()]):
        db.deleteMaskTypes()
        db.deleteMasks()
    db.setMaskType(name=mask_name, index=1, color=mask_color)


def setup_lines(db):
    # setting up new marker type (mode is "line")
    if not any(["SingleFibre" == m.name for m in db.getMarkerTypes()]):
        db.setMarkerType(name=line_name, mode=clickpoints.DataFile.TYPE_Line, color=line_color)


def add_roi(db, dist_threshold, mask_name, line_name, frame, all_frames=False):
    if all_frames:
        frame = list(range(db.getImageCount()))
    else:
        frame = [frame]
    for f in frame:
        line_vecs = np.array([[[l.y1, l.x1], [l.y2, l.x2]] for l in db.getLines(frame=f, type=line_name)])
        if line_vecs.size == 0:
            continue
        im_shape = db.getImages(frame=f)[0].getShape()

        # addding all line points to an array
        mask = np.zeros(im_shape)
        for l in tqdm(line_vecs):
            line_points = np.array(get_line(start=np.round(l[0]).astype(int), end=np.round(l[1]).astype(int)))
            mask[line_points[:, 0], line_points[:, 1]] = 1
        # performing distance transform and selecting area close to lines
        mask_out = distance_transform_edt(distance_transform_edt(~mask.astype(bool))) < dist_threshold
        # writing to clickpoints
        mask_ind = db.getMaskType(name=mask_name).index
        mask_out[mask_out] = mask_ind
        mask_out = mask_out.astype("uint8")
        db.setMask(frame=f, data=mask_out)
    print("##### done")


def check_empty_vector(x, y, frame):
    if len(x) == 0 or len(y) == 0:
        print("no vectors found in frame %s" % str(frame))
        return False
    else:
        return True


def display_ROI_force():
    pass


def add_line_numbers(db, frame):
    for l in db.getLines(frame=frame, type=line_name):
        l.text = str(l.id)
        l.save()


def export_local_alignement(vecs_dict, db,file_name):
    # interpolation factor is size of mask (original image size) devided by size of vector field
    full_data = {}
    for frame in range(db.getImageCount()):
        if len(db.getMasks(frame=frame)) == 0 or len(db.getLines(frame=frame)) == 0:
            continue
        if len(vecs_dict[frame]) == 0:
            continue
        x, y = vecs_dict[frame][0], vecs_dict[frame][1]
        if not check_empty_vector(x, y, frame):
            continue
        labels, distances, mid_points, line_vecs, line_ids, interpolation_factor = split_areas(frame, db, x.shape)
        indices = get_index_dictionary(labels)
        indices = translate_lables_dict(labels, line_ids, mid_points, indices)
        if len(indices.keys()) != len(line_ids):
            print("number of identified mask blobs and number of lines is unequal: ")
            print("number of lines: ", len(line_ids))
            print("number of mask blobs: ", len(indices.keys()))
            print(indices.keys(), line_ids)
            raise Exception

        for obj_id, lvs, mps in tqdm(zip(line_ids, line_vecs, mid_points)):
            inds = indices[obj_id]  # line ids is unique over all frames
            vec_x = x[inds[:, 0], inds[:, 1]]
            vec_y = y[inds[:, 0], inds[:, 1]]
            # position in the coordinate system of the original image
            x_pos = inds[:, 1] * interpolation_factor
            y_pos = inds[:, 0] * interpolation_factor
            dists = distances[inds[:, 0], inds[:, 1]] * interpolation_factor
            lvs = lvs * interpolation_factor
            mps = mps * interpolation_factor
            vec = lvs[1] - lvs[0]
            # lvs is [[y1,x1],[y2,x2]]
            # mps is [y,x]
            full_data[obj_id] = [frame, x_pos, y_pos, vec_x, vec_y, dists, lvs, mps, vec]
        # id=[projected forces, projected forces weighted by distance to line, projected forces normalized by local area,
        # weighted projeced forces normalized by area] --> normalization is per pixel of original image
        # TODO: check if this makes sense
        # TODO : export all frames
    processed_data, total_forces, total_areas = process_data(full_data, interpolation_factor)
    # TODO add output file name and location option
    write_file(file_name, full_data, processed_data,
               total_forces, total_areas)


def write_file(f_name, full_data, processed_data, total_forces, total_areas):
    print("writing to %s" % f_name)
    with open(f_name, "w") as f:
        f.write("total force, total area\n")
        f.write("%.2f, %.2f\n" % (total_forces, total_areas))
        for id in full_data.keys():
            frame, x_pos, y_pos, vec_x, vec_y, dists, lvs, mps, vec = full_data[id]
            f.write("frame, id, middle_x, middle_y, vecx, vecy, p1x, p1y, p2x, p2y\n")
            f.write("%d, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
            frame, id, mps[1], mps[0], vec[1], vec[0], lvs[0][1], lvs[0][0], lvs[1][1], lvs[1][0]))
            f.write(
                "projected force, projected force weighted distance, projected force norm, projected force weighted distance norm\n")
            f.write("%.2f, %.2f, %.6f, %.8f\n" % tuple(processed_data[id]))  # find better solution to writ this
            f.write("pos_x, pos_y, vec_x, vec_y, dist_to_line\n")
            for i1, i2, i3, i4, i5 in zip(x_pos, y_pos, vec_x, vec_y, dists):
                f.write("%.2f, %.2f, %.2f, %.2f, %.2f\n" % (i1, i2, i3, i4, i5))

## miscelleanous
# plt.figure();plt.imshow(np.abs(dist))
# plt.figure();
# plt.imshow(dist)
# plt.plot(l[0][1],l[0][0],"o")
# plt.plot(l[1][1],l[1][0],"o")
# plt.plot(mid_point[1], mid_point[0], "o",color="red")
# plt.arrow(l[0][1],l[0][0], v1[1], v1[0],head_width=20)
# plt.arrow(l[0][1], l[0][0], v1_n[1], v1_n[0], head_width=20)
# plt.plot(pn1[1],pn1[0],"o")
# plt.plot(pn2[1],pn2[0],"o")
