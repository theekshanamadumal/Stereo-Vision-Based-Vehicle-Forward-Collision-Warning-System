
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from visualDet3D.yolo3d_node import Yolo3DNode
from trajectory.trajectory import Trajectory
from visualize.visualize import View
import config.pipeline_config as configData

class Pipeline:
    def __init__(self,max_age=1, min_hits=3, iou_threshold=0.3, no_of_future_frames=5, no_of_past_frames=5):
        print("Starting Pipeline.")
        self.max_age=max_age
        self.min_hits=min_hits
        self.iou_threshold=iou_threshold
        self.no_of_future_frames=no_of_future_frames
        self.no_of_past_frames=no_of_past_frames

        self._read_params()
        self._init_model()

    def _read_params(self):
        print("Reading Pipeline params.")
        self.x0_danger_zone = configData.x0_danger_zone
        self.x1_danger_zone = configData.x1_danger_zone     
        self.y0_danger_zone = configData.y0_danger_zone
        self.y1_danger_zone = configData.y1_danger_zone
        self.w_danger_zone = configData.w_danger_zone
        
    def _init_model(self):
        print("Loading Pipeline.")   
        #detection model
        self.detector = Yolo3DNode()
        #create instance of Trajectory
        self.mot_tracker = Trajectory(self.max_age,self.min_hits,self.iou_threshold,self.no_of_future_frames,self.no_of_past_frames) 
        #create instance of View
        self.view = View() 

    def assignTrackID_without_view(self,left_image,right_image):
        # get detections
        rgb_image, scores, bbox_2d, obj_names, bbox_3d_corner_homo = self.detector.predict(left_image, right_image)
        
        if len(scores) > 0:
            """
                NOTE: In 3D bbox_3d_corner_homo 0,1,2,7 are the coordinates of the front face of the 3D bounding box
            """
            detectionsList=[]
            for i in range(len(scores)):
                #for 2d bounding box
                # detection=np.append((bbox_2d[i]).cpu(),scores[i].cpu())
                
                #for 3D bounding box converting to 2D
                coords_3d = bbox_3d_corner_homo[i].cpu()
                x1,y1 = coords_3d[1][0], coords_3d[1][1]
                x2,y2 = coords_3d[7][0], coords_3d[7][1]
                bb=[x1,y1,x2,y2]
                detection=np.append(bb,scores[i].cpu())
                
                detectionsList.append(detection)
            detectionsNumPyArray = np.asarray(detectionsList)
            track_bbs_ids_list = self.mot_tracker.update(detectionsNumPyArray)
        else:
            track_bbs_ids_list = self.mot_tracker.update(np.empty((0, 5)))    

        return track_bbs_ids_list

    def assignTrackID(self,left_image,right_image):
        # get detections
        rgb_image, scores, bbox_2d, obj_names, bbox_3d_corner_homo = self.detector.predict(left_image, right_image)
        
        if len(scores) > 0:
            """
                NOTE: In 3D bbox_3d_corner_homo 0,1,2,7 are the coordinates of the front face of the 3D bounding box
            """
            detectionsList=[]
            for i in range(len(scores)):
                #for 2d bounding box
                # detection=np.append((bbox_2d[i]).cpu(),scores[i].cpu())
                
                # #for 3D bounding box converting to 2D
                coords_3d = bbox_3d_corner_homo[i].cpu()
                x1,y1 = coords_3d[1][0], coords_3d[1][1]
                x2,y2 = coords_3d[7][0], coords_3d[7][1]
                bb=[x1,y1,x2,y2]
                detection=np.append(bb,scores[i].cpu())
                
                detectionsList.append(detection)
            detectionsNumPyArray = np.asarray(detectionsList)
            track_bbs_ids_list = self.mot_tracker.update(detectionsNumPyArray)
            
            #draw_bbox2d
            # self.view.draw_bbox2d(rgb_image,bbox_2d)
            #draw_3D_bbox
            self.view.draw_3D_bbox(rgb_image,bbox_3d_corner_homo)
            
            #draw_track_ID
            track_bbs_ids_list=track_bbs_ids_list.tolist()
            self.view.draw_track_ID(rgb_image, track_bbs_ids_list)
        else:
            track_bbs_ids_list = self.mot_tracker.update(np.empty((0, 5)))  
                
        return np.clip(rgb_image, 0, 255), track_bbs_ids_list

    def is_in_zone(self,x,y):
        if self.x0_danger_zone<x<self.x1_danger_zone and self.y0_danger_zone>y>self.y1_danger_zone:
            return True
        return False
    
    def is_in_zone_bb(self,bb):
        x0,y0,x1,y1=bb
        if self.is_in_zone(x0,y0):
            return True
        if self.is_in_zone(x1,y1):
            return True
        return False

    def is_in_trapezoid_zone(self, testx, testy):
        x0 = self.x0_danger_zone
        y0 = 300 - self.y0_danger_zone
        x1 = self.x1_danger_zone
        y1 = 300 - self.y1_danger_zone
        w = self.w_danger_zone

        vertx = [x0, x0+w, x1-w, x1]
        verty = [y0, y1, y1, y0]

        testy = 300 - testy

        c = False
        j = len(vertx)-1

        for i in range(len(vertx)):
            if ( ((verty[i]>testy) != (verty[j]>testy)) and (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
                c = not c

            j = i

        return c
    
    def is_in_trapezoid_zone_bb(self, bb):
        x0,y0,x1,y1=bb
        if self.is_in_trapezoid_zone(x0, y0):
            return True
        if self.is_in_trapezoid_zone(x0,y1):
            return True
        if self.is_in_trapezoid_zone(x1, y0):
            return True
        if self.is_in_trapezoid_zone(x1, y1):
            return True
        return False

    def get_zone_check_point(self, bb):
        x0,y0,x1,y1=bb
        if x0<self.x1_danger_zone:
            x=x1
        else:
            x=x0
        y=max(y0,y1)
        return x,y

    def collision_warning_without_view(self,left_image,right_image):
        track_bbs_ids=self.assignTrackID_without_view(left_image,right_image)

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
        
            x,y=self.get_zone_check_point(bb)
            if self.is_in_zone(x,y):
                print('Warning! vehicle ID:',i)
                
            for i,future_bb in enumerate (future_bbs):
                x,y=self.get_zone_check_point(future_bb)
                if self.is_in_zone(x,y):
                    print('Warning! vehicle ID:',vehicle_ID)
                    break #to avoid count ff bb of same vehicle 
                
    def collision_warning(self,left_image,right_image):
        image, track_bbs_ids=self.assignTrackID(left_image,right_image)

        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
        
            x,y=self.get_zone_check_point(bb)
            if self.is_in_zone(x,y):
                id_list.append(vehicle_ID)
                print('Warning! vehicle ID:',i)
                
            for i,future_bb in enumerate (future_bbs):
                x,y=self.get_zone_check_point(future_bb)
                if self.is_in_zone(x,y):
                    future_id_list.append(vehicle_ID)
                    print('Warning!','Future Frame: ',i,'vehicle ID:',vehicle_ID)
                    break #to avoid count ff bb of same vehicle 
                    
        rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        self.view.draw_danger_zone(rgb_image)
        if len(id_list)>0:
            self.view.show_zone_warning(rgb_image,id_list)
        if len(future_id_list)>0:
            self.view.show_future_warning(rgb_image,future_id_list)    

        self.view.show_image_on_same_window(image)    

    def collision_warning_trapezoid(self,left_image,right_image):
        image, track_bbs_ids=self.assignTrackID(left_image,right_image)

        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
        
            if self.is_in_trapezoid_zone_bb(bb):
                id_list.append(vehicle_ID)
                
            for i,future_bb in enumerate (future_bbs):
                if self.is_in_trapezoid_zone_bb(future_bb):
                    future_id_list.append(vehicle_ID)
                    break #to avoid count ff bb of same vehicle 
                    
        # rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        self.view.draw_danger_zone(image)
        if len(id_list)>0:
            self.view.show_zone_warning(image,id_list)
        if len(future_id_list)>0:
            self.view.show_future_warning(image,future_id_list)    

        return image
        # self.view.show_image_on_same_window(image)    
        
    def save_collision_warning_images(self,left_image,right_image,out_path,index):
        rgb_image, track_bbs_ids=self.assignTrackID(left_image,right_image)

        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
        
            x,y=self.get_zone_check_point(bb)
            if self.is_in_zone(x,y):
                id_list.append(vehicle_ID)
                
            for i,future_bb in enumerate (future_bbs):
                x,y=self.get_zone_check_point(future_bb)
                if self.is_in_zone(x,y):
                    future_id_list.append(vehicle_ID)
                break
                        
        self.view.draw_danger_zone(rgb_image)
        if len(id_list)>0:
            self.view.show_zone_warning(rgb_image,id_list)
        if len(future_id_list)>0:
            self.view.show_future_warning(rgb_image,future_id_list)    

        file_name = "/%06d" % index+".png"
        out_path = out_path + file_name

        print(index,cv2.imwrite(out_path, rgb_image))