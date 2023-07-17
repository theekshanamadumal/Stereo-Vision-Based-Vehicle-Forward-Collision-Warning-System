
from IPython.display import clear_output
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from visualDet3D.utils.utils import  draw_3D_box

import config.pipeline_config as configData

class View:
    def __init__(self):
        print("Starting View.")
        self._read_params()

    def _read_params(self):
        print("Reading View params.")
        self.x0 = configData.x0_danger_zone
        self.x1 = configData.x1_danger_zone
        self.y0 = configData.y0_danger_zone
        self.y1 = configData.y1_danger_zone
        self.w = configData.w_danger_zone

    def draw_bbox2d(self, rgb_image, bboxes2d, color=(255, 0, 255)):
        for box2d in bboxes2d:
            cv2.rectangle(rgb_image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color, 3)
    
    def draw_3D_bbox(self,rgb_image,bbox_3d_corner_homo):
        for box in bbox_3d_corner_homo:
                box = box.cpu().numpy().T
                rgb_image = draw_3D_box(rgb_image, box)

    def draw_track_ID(self, rgb_image, track_bbs_ids_list):
         for track_bbs_ids in track_bbs_ids_list:
            coords = track_bbs_ids[0]
            x1,y1,x2,y2=int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
            trackID=int(track_bbs_ids[-1])
            name="ID: {}".format(str(trackID))

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (x1, y1-10)
            # fontScale
            fontScale = 0.9
            # Line thickness of 2 px
            thickness = 2
            # Blue color in BGR
            color=(0, 255, 255)
            text_color_bg=(0, 0, 0)
            text_size, _ = cv2.getTextSize(name, font, fontScale, thickness)
            text_w, text_h = text_size
            cv2.rectangle(rgb_image, (x1 , y1-10 - text_h), (x1 + text_w, y1-10), text_color_bg, -1)
            
            # Using cv2.putText() method
            cv2.putText(rgb_image,name, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def show_zone_warning(self,image_c,id_list):

        message= "Warning! ID:"+ str(id_list)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        text_color_bg=(255, 255, 255)

      
        warning_bg = (0,0,255)
        text_size, _ = cv2.getTextSize(message, font, fontScale, thickness)
        text_w, text_h = text_size
        x1 = int(620-text_w/2)
        y1 = 60
        cv2.rectangle(image_c, (x1 , y1-10 - text_h), (x1 + text_w, y1+10), text_color_bg, -1)
        cv2.putText(image_c,message, (x1,y1), font, fontScale, warning_bg, thickness, cv2.LINE_AA)
#         image_c = cv2.copyMakeBorder(
#                     image_c,
#                     top=50,
#                     bottom=50,
#                     left=50,
#                     right=50,
#                     borderType=cv2.BORDER_CONSTANT,
#                     value=[255, 0, 0]
# )
       
        #warning sound
        os.system("echo -n '\a';sleep 0.1;" * 1)


    def show_future_warning(self,image_c,id_list):

        message= "Future Warning! ID:"+ str(id_list)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        color=(0,0,255)
        text_color_bg=(255, 255, 255)
        text_size, _ = cv2.getTextSize(message, font, fontScale, thickness)
        text_w, text_h = text_size
        x1 = int(620-text_w/2)
        y1 = 30
        cv2.rectangle(image_c, (x1 , y1-text_h-10), (x1 + text_w, y1+10), text_color_bg, -1)
        cv2.putText(image_c,message, (x1,y1), font, fontScale, color, thickness, cv2.LINE_AA)
        
        #warning sound
        os.system("echo -n '\a';sleep 0.1;" * 1)
        
    def draw_danger_zone(self,rgb_image):
        thickness=3
        color=text_color_bg=(0, 125, 255)
        pts=np.array([[self.x0,self.y0],[self.x0+self.w,self.y1],[self.x1-self.w,self.y1],[self.x1,self.y0]],np.int32)
        isClosed=True
        cv2.polylines(rgb_image, [pts],isClosed, color, thickness)
    #     cv2.rectangle(rgb_image, (x0 , y0), (x1, y1), text_color_bg, 0)    

    def show_image_on_same_window(self,image):
        fig = plt.figure(figsize=(16,9))
        clear_output(wait=True) #to show in same window
        rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()
    

    def show_image_on_different_window(self,image):
        fig = plt.figure(figsize=(16,9))
        rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()


#warning sound
# beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
# beep(3)