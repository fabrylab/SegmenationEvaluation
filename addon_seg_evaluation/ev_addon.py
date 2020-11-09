#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CellDetector.py

# Copyright (c) 2015-2016, Richard Gerum, Sebastian Richter
#
# This file is part of ClickPoints.
#
# ClickPoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ClickPoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ClickPoints. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import qtawesome as qta
from qtpy import QtCore, QtWidgets
from functools import partial
import clickpoints
from PyQt5.QtCore import QSettings
from clickpoints.includes.QtShortCuts import AddQSpinBox, AddQOpenFileChoose
from clickpoints.includes import QtShortCuts

from inspect import getdoc
import traceback
import os
import sys
from importlib import import_module, reload
import configparser
from skimage.measure import label, regionprops
from addon_seg_evaluation.UNETmodel import UNet
from pathlib import Path
from PyQt5.QtCore import pyqtSignal
import numpy as np
from clickpoints.includes.matplotlibwidget import MatplotlibWidget, NavigationToolbar
from matplotlib import pyplot as plt
import time
from pathlib import Path
print(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import importlib.util
from PIL import Image
from tqdm import tqdm
from addon_seg_evaluation.base_segmentation_classes import Segmentation, FromOtherDB



class SetFile(QtWidgets.QHBoxLayout):

    fileSeleted = pyqtSignal(bool)
    def __init__(self, file=None, type=QtWidgets.QFileDialog.ExistingFile):
        super().__init__() # activating QVboxLayout
        if file is None:
            self.files = ""
        else:
            self.files = file
        self.type = type
        #self.folder = os.getcwd()
        # line edit holding the currently selected folder 1
        self.line_edit_folder = QtWidgets.QLineEdit(self.files)
        self.addWidget(self.line_edit_folder, stretch=4)

        # button to browse folders
        self.open_folder_button = QtWidgets.QPushButton("choose files")
        self.open_folder_button.clicked.connect(self.file_dialog)
        self.addWidget(self.open_folder_button, stretch=2)


    def file_dialog(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(self.type)
        dialog.setDirectory(os.path.split(self.files)[0])
        if dialog.exec_():
            if self.type == QtWidgets.QFileDialog.ExistingFile:
                self.files = dialog.selectedFiles()[0]
            else:
                self.files = dialog.selectedFiles()
            self.line_edit_folder.setText(str(self.files)) # This wrong?
            self.fileSeleted.emit(True)

def set_up_additional_databases(ev_addon, db_name,  illustration=False):

    folder = os.path.split(db_name)[0]
    os.makedirs(folder,exist_ok=True)
    export_db_path = os.path.join(folder,db_name)
    notes_txt = open(export_db_path[:-4] + "_notes.txt", "+a")
    if os.path.exists(export_db_path):
        exp_db = clickpoints.DataFile(export_db_path, "r")
    else:
        exp_db = clickpoints.DataFile(export_db_path, "w")
        exp_db.deletePaths()
        exp_db.setPath(folder)
        if illustration:
            mt1 = exp_db.setMaskType(name=ev_addon.net1_db_name, color=ev_addon.net1_db_color, index=1)
            mt2 = exp_db.setMaskType(name=ev_addon.net2_db_name, color=ev_addon.net2_db_color, index=2)
            mt_ov = exp_db.setMaskType(name=ev_addon.overlap_mask, color=ev_addon.overlap_mask_color, index=3)
            elt1 = exp_db.setMarkerType(name=ev_addon.net1_db_name, color=ev_addon.net1_db_color,
                                             mode=clickpoints.DataFile.TYPE_Ellipse)
            elt2 = exp_db.setMarkerType(name=ev_addon.net2_db_name, color=ev_addon.net2_db_color,
                                             mode=clickpoints.DataFile.TYPE_Ellipse)
    return exp_db, notes_txt


class Addon(clickpoints.Addon):

    signal_update_plot = QtCore.Signal()
    signal_plot_finished = QtCore.Signal()

    net1_db_name = "segmentation 1"
    net1_db_color = "#0a2eff" # TODO maye choose better colors
    net2_db_name = "segmentation 2"
    net2_db_color =  "#Fa2eff"
    overlap_mask = "overlapp"
    overlap_mask_color = "#fff014"

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        # database setup
        '''
        if not all(["ROI" in m.name for m in self.db.getMaskTypes()]):
            choice = QtWidgets.QMessageBox.question(self, 'continue',
                                                "All exisiting masks will be deleted",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.No:
            return
        else:
            setup_masks(self.db)
            self.cp.reloadMaskTypes()
            self.cp.reloadMaskTypes()
        '''
        self.exp_db =  None
        self.exp_db_mask = None
        self.export_db_path = "/home/user/Desktop/2020_Deformation_Cytometer/network_problems/network_problems.cdb"
        self.export_db_path_mask = "/home/user/Desktop/2020_Deformation_Cytometer/network_problems_illustration2/network_problems_mask.cdb"
        #self.setup_additonal_databases_wrapper(db_name=self.export_db_path, db_name_illustration=self.export_db_path_mask)

        self.current_positions = []
        self.curr_pos_index = 0


        # TODO load this from config // at least try to
        self.magnification = 40
        self.coupler = 0.5
        self.pixel_size_camera = 6.9
        self.pixel_size = self.pixel_size_camera/(self.magnification * self.coupler)
        self.pixel_size  *= 1e-6  #conversion to meter
        self.r_min = 6 # in µm
        self.edge_dist = 15 # in µm
        self.channel_width = 0

        self.solidity_threshold = 0.96
        self.irregularity_threshold = 1.06

        self.note_filtered =  True # write irregularity and solidtiy down for cells that would have been filtered out by these thresholds

        self.db.deleteMaskTypes()
        self.mt1 = self.db.setMaskType(name=self.net1_db_name, color=self.net1_db_color, index=1)
        self.mt2 = self.db.setMaskType(name=self.net2_db_name, color=self.net2_db_color, index=2)
        self.mt_ov = self.db.setMaskType(name=self.overlap_mask, color=self.overlap_mask_color, index=3)

        self.elt1 = self.db.setMarkerType(name=self.net1_db_name, color=self.net1_db_color, mode=clickpoints.DataFile.TYPE_Ellipse)
        self.elt2 = self.db.setMarkerType(name=self.net2_db_name, color=self.net2_db_color,
                                              mode=clickpoints.DataFile.TYPE_Ellipse)
        self.cp.reloadMaskTypes()
        self.cp.reloadTypes()
        self.cp.save()

        self.folder = os.getcwd()
        self.file1 = ""
        self.file2 = ""
        self.settings = QSettings("clickpoints_segmentation_evaluation", "clickpoints_segmentation_evaluation")
        if "file1" in self.settings.allKeys():
            self.file1 = self.settings.value("file1")
        if "file2" in self.settings.allKeys():
            self.file2 = self.settings.value("file2")

        self.Segmentation1 = None
        self.Segmentation2 = None

        """ GUI Widgets"""
        self.setWindowTitle("DeformationCytometer - ClickPoints")
        # layouts
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(10, 20, 10, 20)

        self.layout_find_files1 =  SetFile(self.file1)
        self.layout_find_files1.fileSeleted.connect(partial(self.files_selected, obj=[1]))
        self.layout_find_files2 = SetFile(self.file2)
        self.layout_find_files2.fileSeleted.connect(partial(self.files_selected, obj=[2]))
        self.layout_evaluate = QtWidgets.QHBoxLayout()
        # trying to load the Segmentation functions right now
        self.files_selected(obj=[1,2])

        # path to export db
        self.layout_export_1 = QtWidgets.QHBoxLayout()
        self.line_edit_export = QtWidgets.QLineEdit(self.export_db_path)
        # note: editing finshed is emit when you pressed enter
        self.line_edit_export.editingFinished.connect(partial(self.setup_additonal_databases_wrapper, db_name="auto"))
        self.layout_export_1.addWidget(self.line_edit_export, stretch=4)

        self.export_button = QtWidgets.QPushButton("export")
        self.export_button.clicked.connect(partial(self.export, filename=self.export_db_path, illustration=False))
        self.layout_export_1.addWidget(self.export_button, stretch=2)

        '''
        # the selected files
        self.scroll = QtWidgets.QScrollArea()
        self.add_file()
        '''

        # accept and decline buttons plus counters
        self.accept_layout = QtWidgets.QVBoxLayout()
        self.accept_button = QtWidgets.QPushButton("error network 1")
        self.accept_counter = QtWidgets.QLabel(str(0))
        self.accept_button.clicked.connect(partial(self.decide_network, self.accept_counter, "error network 1"))
        self.accept_layout.addWidget(self.accept_button,stretch=1)
        self.accept_layout.addWidget(self.accept_counter,stretch=1)
        self.accept_layout.addStretch(stretch=3)
        self.layout_evaluate.addLayout(self.accept_layout)

        self.reject_layout = QtWidgets.QVBoxLayout()
        self.reject_button = QtWidgets.QPushButton("error network 2")
        self.reject_counter = QtWidgets.QLabel(str(0))
        self.reject_button.clicked.connect(partial(self.decide_network,self.reject_counter, "error network 2"))
        self.reject_layout.addWidget(self.reject_button,stretch=1)
        self.reject_layout.addWidget(self.reject_counter,stretch=1)
        self.reject_layout.addStretch(stretch=3)
        self.layout_evaluate.addLayout(self.reject_layout)

        self.next_layout = QtWidgets.QVBoxLayout()
        self.next_button = QtWidgets.QPushButton("next")
        self.next_button.clicked.connect(self.next_position)
        self.next_layout.addWidget(self.next_button)
        self.next_layout.addStretch()
        self.layout_evaluate.addLayout(self.next_layout)

        #self.skip_layout = QtWidgets.QVBoxLayout()
        #self.skip_button = QtWidgets.QPushButton("skip")
        #self.skip_button.clicked.connect(partial(self.decide_network, None, None))
        #self.skip_layout.addWidget(self.skip_button)
        #self.skip_layout.addStretch()
        #self.layout_evaluate.addLayout(self.skip_layout)
        self.export_layout = QtWidgets.QVBoxLayout()
        self.export_mask_button = QtWidgets.QPushButton("export illustration")
        self.export_mask_button.clicked.connect(partial(self.export, filename=self.export_db_path_mask, illustration=True))
        self.all_frames_button = QtWidgets.QPushButton("predict all frames")
        self.all_frames_button.clicked.connect(partial(self.start_thread, run_function=self.predict_all))

        self.export_layout.addWidget(self.export_mask_button)
        self.export_layout.addWidget(self.all_frames_button)
        self.layout_evaluate.addLayout(self.export_layout)

        self.layout.addLayout(self.layout_find_files1)
        self.layout.addLayout(self.layout_find_files2)
        self.layout.addLayout(self.layout_export_1)
        self.layout.addLayout(self.layout_evaluate)
        #self.layout.addLayout(self.layout_layers)
        self.show() #TODO do I really need this??


    def setup_additonal_databases_wrapper(self, db_name=None, db_name_illustration=None):

        if db_name == "auto":
            db_name = self.line_edit_export.text()
        if not db_name is None:
            if  db_name.endswith(".cdb"):
                print("writing to " + db_name)
                if self.exp_db is not None:
                    self.exp_db.db.close()
                    self.notes_txt.close()
                self.exp_db, self.notes_txt = set_up_additional_databases(self, db_name)
            else:
                print("invalid path: " + db_name)

        if not db_name_illustration is None:
            print("writing to " + db_name_illustration)
            if self.exp_db_mask is not None:
                    self.exp_db_mask.db.close()
                    self.notes_txt_mask.close()
            self.exp_db_mask, self.notes_txt_mask = set_up_additional_databases(self, db_name_illustration, illustration=True)

    def files_selected(self, obj=[0]):
        self.file1 =  self.layout_find_files1.files
        self.file2 =  self.layout_find_files2.files

        try:
            frame = self.cp.getCurrentFrame()
        except AttributeError:
            frame = 0
        img_shape = self.db.getImage(frame=frame).getShape()
        if 1 in obj and not self.file1 == "":
            self.file1 = self.layout_find_files1.files
            self.settings.setValue("file1", self.file1)
            if self.file1.endswith(".cdb"):
                self.Seg1 = FromOtherDB(db_path=self.file1)
            else:
                self.Segmentation1 = self.import_seg_function(self.file1)
                if not self.Segmentation1 is None:
                    try:
                        self.Seg1 = self.Segmentation1(
                            img_shape=img_shape, pixel_size=self.pixel_size, r_min=self.r_min, frame_data=None, edge_dist=self.edge_dist, channel_width=self.channel_width)
                        print("succesfully loaded %s"%self.file1)
                    except OSError as e:
                        print(e)

        if 2 in obj and not self.file2 == "":
            self.file2 = self.layout_find_files2.files
            self.settings.setValue("file2", self.file2)
            if self.file2.endswith(".cdb"):
                self.Seg2 = FromOtherDB(db_path=self.file2)
            else:
                self.Segmentation2 = self.import_seg_function(self.file2)
                if not self.Segmentation2 is None:
                    try:
                        self.Seg2 = self.Segmentation2(
                            img_shape=img_shape, pixel_size=self.pixel_size, r_min=self.r_min, frame_data=None, edge_dist=self.edge_dist, channel_width=self.channel_width)
                        print("succesfully loaded %s"%self.file2)
                    except OSError as e:
                        print(e)

        self.settings.sync()


    def decide_network(self, counter, text):

        if len(self.current_positions) > 0 and not counter is None:
            ellipse = self.find_closest_ellipse(self.current_positions[self.curr_pos_index])
            ellipse.text = text
            ellipse.save()
            self.cp.reloadMarker(self.cp.getCurrentFrame())
            counter.setText(str(int(counter.text()) + 1))
            #self.next_position()
        else:
            #self.next_position()
            pass

    def find_closest_ellipse(self, pos):

        frame = self.cp.getCurrentFrame()
        all_ellipses = self.db.getEllipses(frame=frame)
        dists = [np.sqrt((pos[0] - e.y) ** 2 + (pos[1] - e.x) ** 2) for e in all_ellipses]
        min_ellipse = all_ellipses[np.argmin(dists)]
        return min_ellipse


    def next_frame(self):
        self.predict(frame=self.cp.getCurrentFrame() + 1)
        self.cp.jumpFramesWait(1)
        print(self.cp.getCurrentFrame())
        self.curr_pos_index = 0


    def next_position(self):
        if len(self.current_positions) <= self.curr_pos_index + 1:
            self.next_frame()
            while len(self.current_positions) == 0:
                self.next_frame()
            self.cp.centerOn(self.current_positions[self.curr_pos_index][1], self.current_positions[self.curr_pos_index][0])

        else:
            self.curr_pos_index +=1
            self.cp.centerOn(self.current_positions[self.curr_pos_index][1], self.current_positions[self.curr_pos_index][0])

    def previous_position(self):
        if self.curr_pos_index > 0:
            self.curr_pos_index -=1
            self.cp.centerOn(self.current_positions[self.curr_pos_index][1], self.current_positions[self.curr_pos_index][0])


    def predict_all(self):
        curr_frame = self.cp.getCurrentFrame()
        total_frames =self.db.getImageCount()
        for f in tqdm(range(curr_frame, total_frames,1)):
            self.predict(f)

    def export(self, filename="", illustration=False):
        if not os.path.exists(filename):
            set_up_additional_databases(self, filename, illustration=illustration)

        folder = os.path.split(filename)[0]
        frame = self.cp.getCurrentFrame()
        db_im = self.db.getImage(frame=frame)
        name = "frame" + str(frame) + "_" + os.path.split(db_im.filename)[1]
        data = db_im.data
        im = Image.fromarray(data)
        im.save(os.path.join(folder, name))
        if illustration:
            self.notes_txt_mask.write(filename + "\t" + name + "\n")
            self.save_to_other_db(self.exp_db_mask, frame, name, save_mask=illustration)
        else:
            self.notes_txt.write(filename + "\t" + name + "\n")
            self.save_to_other_db(self.exp_db, frame, name)


    def save_to_other_db(self, db, frame, fname, save_mask=0):
        try:
            s_id = db.getImageCount()
        except TypeError:
            s_id = 0
        try:
            im = db.setImage(sort_index=s_id, filename=fname,path=1)

            if save_mask:
                try:
                    mask_data = self.db.getMask(frame=frame).data
                    db.setMask(frame=s_id, data=mask_data)
                except AttributeError:
                    pass
                for ellipse in self.db.getEllipses(frame=frame):
                    db.setEllipse(type=ellipse.type.name, image=im, x=ellipse.x, y=ellipse.y, width=ellipse.width, height=ellipse.height, angle=ellipse.angle)
        except Exception as e:
            print(e)

    def keyPressEvent(self, event):

        # error network 1
        if event.key() == QtCore.Qt.Key_1:
            self.decide_network(self.accept_counter, "error network 1")
        # error network 1
        if event.key() == QtCore.Qt.Key_2:
            self.decide_network(self.reject_counter, "error network 2")
        # go to next position:
        if event.key() == QtCore.Qt.Key_3:
            self.next_position()
        # go to previous position:
        if event.key() == QtCore.Qt.Key_4:
            self.next_position()

        # export to improve network
        if event.key() == QtCore.Qt.Key_E:
            self.export(filename=self.export_db_path, illustration=False)
        # export to illustrate errors
        if event.key() == QtCore.Qt.Key_U:
            self.export(filename=self.export_db_path_mask, illustration=True)

    def import_seg_function(self,path):
        try:
            imp = importlib.util.spec_from_file_location("module.name", path)
            module = importlib.util.module_from_spec(imp)
            imp.loader.exec_module(module)
            return module.Segmentation
        except Exception as e:
            print("import failed")
            print(e)
            return None

    def predict(self, frame):


        # loading the image
        db_im = self.db.getImage(frame=frame)
        self.db.deleteEllipses(frame=frame,type="segmentation 1")
        self.db.deleteEllipses(frame=frame,type="segmentation 2")
        image = db_im.data

        # making the predicition
        if not isinstance(self.Seg1, FromOtherDB):
            pred_mask1, ellipses1 = self.Seg1.segmentation(image)
        else:
            pred_mask1, ellipses1 = self.Seg1.getMaskEllipse(frame)
        if not isinstance(self.Seg2, FromOtherDB):
            pred_mask2, ellipses2 = self.Seg2.segmentation(image)
        else:
            pred_mask2, ellipses2 = self.Seg2.getMaskEllipse(frame)



        # writing prediction to database
        mask_data = pred_mask2.astype(np.uint8) * 2
        mask_data += pred_mask1.astype(np.uint8)
        self.find_unique_objects( mask_data)

        self.db.setMask(frame=frame, data=mask_data)
        for el in ellipses1:
            self.set_ellipse(db_im, self.net1_db_name, el)
        for el in ellipses2:
            self.set_ellipse(db_im, self.net2_db_name, el)
        self.cp.reloadMarker(frame=frame)
        self.cp.reloadMask()


    def find_unique_objects(self, mask_pred):
        labeled = label(mask_pred > 0) # this merges overlapping areas and stuff
        self.current_positions = [r.centroid for r in regionprops(labeled) if r.area > 100]


    def set_ellipse(self, im, mtype, ellipse):
        pix_size = self.pixel_size*1e6
        text = "irregularity:%s\nsolidity%s"%(str(np.round(ellipse["irregularity"], 2)), str(np.round(ellipse["solidity"], 2)))
        filtered = ~((ellipse["solidity"] > 0.96) & (ellipse["irregularity"] < 1.06))
        if filtered and self.note_filtered:
            el = self.db.setEllipse(image=im, type=mtype, x=ellipse["x_pos"], y=ellipse["y_pos"],
                                    width=0, height=0, angle=0, text=text)
        elif not filtered:
            el = self.db.setEllipse(image=im, type=mtype, x=ellipse["x_pos"], y=ellipse["y_pos"],
                               width=ellipse["long_axis"]/pix_size, height=ellipse["short_axis"]/pix_size, angle=ellipse["angle"], text=text)
        else:
            return


        #self.cp.centerOn(self.data.x[nearest_point], self.data.y[nearest_point])
    # run in a separate thread to keep clickpoints gui responsive // now using QThread and stuff

    def start_thread(self, run_function=None):
        self.thread = Worker(self, run_function=run_function)
        self.thread.start()  # starting thread
        #self.thread.finished.connect(self.reload_all)  # connecting function on thread finish

class Worker(QtCore.QThread):
    output = pyqtSignal()

    def __init__(self, main, parent=None, run_function=None):
        QtCore.QThread.__init__(self, parent)
        self.main = main
        if run_function is None:
            self.run_function = self.main.start
        else:
            self.run_function = run_function

    def run(self):
        self.run_function()

    '''
    def add_file(self):

        self.scrollAreaWidgetContents = QtWidgets.QWidget()  # adding grid layout to extra widget to allow for scrolling
        self.disp_file_layout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.disp_file_layout.setRowStretch(10, 3)
        self.file_fields = {}
        for i, file in enumerate(self.files):
            edit = QtWidgets.QLineEdit(file)
            remove_button = QtWidgets.QPushButton("remove")
            remove_button.clicked.connect(partial(self.remove_file_field,i))
            self.disp_file_layout.addWidget(edit, i, 0)
            self.disp_file_layout.addWidget(remove_button, i, 1)
            self.file_fields[i] = [edit, remove_button]
        self.scroll.setWidget(self.scrollAreaWidgetContents)
        self.settings.setValue("files",self.files)
    '''

    '''
    def remove_file_field(self,i=0):
        self.files.pop(i)
        self.add_file()

    '''

    def load_script(self):
        pass

