import os, sys
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IPython.display import clear_output
import time

from pipeline import Pipeline
from visualize.visualize import View
from visualDet3D.yolo3d_node import Yolo3DNode
from trajectory.trajectory import Trajectory
from visualize.visualize import View
import config.pipeline_config as configData

detector = Yolo3DNode()
#create instance of Trajectory
#mot_tracker = Trajectory()
pipeline = Pipeline()
view = View()

def test_image_reading():
    print('\n-------------------- test_image_reading --------------------\n')
    #left_image = cv2.imread("test/images/l.png")
    #right_image = cv2.imread("test/images/r.png")
    
    left_image = cv2.imread("./data/"+path+"_left/%010d" % 0+".png")
    right_image = cv2.imread("./data/"+path+"_right/%010d" % 0+".png")
    return left_image, right_image

def test_image_showing():
    print('\n-------------------- test_image_showing --------------------\n')

    left_image, right_image = test_image_reading()
    view.show_image_on_different_window(left_image)
    view.show_image_on_different_window(right_image)

def test_image_detection():
    print('\n-------------------- test_image_detection --------------------\n')

    left_image, right_image = test_image_reading()
    rgb_image, scores, bbox_2d, obj_names, bbox_3d_corner_homo = detector.predict(left_image, right_image)
    if len(scores) > 0:
#         rgb_image = draw_bbox2d_to_image(rgb_image, bbox_2d.cpu().numpy())
        for box in bbox_3d_corner_homo:
            box = box.cpu().numpy().T
            view.draw_3D_bbox(rgb_image,bbox_3d_corner_homo)
    view.show_image_on_different_window(rgb_image)

def test_sequence_image_detection(length):
    print('\n-------------------- test_sequence_image_detection --------------------\n')
    start_track = time.time()

    for i in range (length):

        left_image = cv2.imread("./data/"+path+"_left/%06d" % i+".png")
        right_image = cv2.imread("./data/"+path+"_right/%06d" % i+".png")

        rgb_image, scores, bbox_2d, obj_names, bbox_3d_corner_homo = detector.predict(left_image, right_image)

        for box in bbox_3d_corner_homo:
            box = box.cpu().numpy().T
            view.draw_3D_bbox(rgb_image,bbox_3d_corner_homo)
        print(i)
        #cv2.imshow('image',rgb_image)
        view.show_image_on_same_window(rgb_image)

    end_track = time.time()
    # Time elapsed
    seconds_track = end_track - start_track
    print ("Time taken : {0} seconds".format(seconds_track))
    # Calculate frames per second
    fps_track  = length / seconds_track
    print("Estimated frames per second : {0}".format(fps_track))

def test_trajectory():
    print('\n-------------------- test_trajectory --------------------\n')
    pipeline = Pipeline()
    left_image = cv2.imread("./data/"+path+"_left/%06d" %0+".png")
    right_image = cv2.imread("./data/"+path+"_right/%06d" %0+".png")

    track_bbs_ids=pipeline.assignTrackID_without_view(left_image,right_image)

    # print(track_bbs_ids)
    for bbs in track_bbs_ids: #for each vehicle in a image frame
        bb=bbs[0] #current bounding box
        future_bb=bbs[1] #future bounding boxes
        vehicle_id=bbs[-1] #vehicle id
        print('vehicle_id',vehicle_id)
        print('bb coords',bb)
        print('trackID',bbs[-1])
        print('future coords',future_bb,'\n')

def test_sequence_trajectory_without_view(length):    
    print('\n-------------------- test_sequence_trajectory_without_view --------------------\n')
    start_track = time.time()
    pipeline = Pipeline()
    for frame in range (length):
        left_image = cv2.imread("./data/"+path+"_left/%06d" % frame+".png")
        right_image = cv2.imread("./data/"+path+"_right/%06d" % frame+".png")

        track_bbs_ids=pipeline.assignTrackID_without_view(left_image,right_image)
        print('\n-------------------- {} --------------------'.format(frame))

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id

            x,y=pipeline.get_zone_check_point(bb)
            if pipeline.is_in_zone(x,y):
                print('Warning!','Frame:',frame,'vehicle ID:',vehicle_ID)

            for i,future_bb in enumerate (future_bbs):
                x,y=pipeline.get_zone_check_point(future_bb)
                if pipeline.is_in_zone(x,y):
                    print('Warning!','Future Frame: ',i,'vehicle ID:',vehicle_ID)

    end_track = time.time()
    # Time elapsed
    seconds_track = end_track - start_track
    print ("Time taken : {0} seconds".format(seconds_track))

    # Calculate frames per second
    fps_track  = length / seconds_track
    print("Estimated frames per second : {0}".format(fps_track))

def test_sequence_trajectory(length):
    print('\n-------------------- test_sequence_trajectory --------------------\n')
    start_track = time.time()
    pipeline = Pipeline()
    for frame in range (length):
        left_image = cv2.imread("./data/"+path+"_left/%010d" % frame+".png")
        right_image = cv2.imread("./data/"+path+"_right/%010d" % frame+".png")

        rgb_image, track_bbs_ids=pipeline.assignTrackID(left_image,right_image)
        print('\n-------------------- {} --------------------'.format(frame))
        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id

            x,y=pipeline.get_zone_check_point(bb)
            if pipeline.is_in_zone(x,y):
                id_list.append(vehicle_ID)
                print('Warning!','Frame:',frame,'vehicle ID:',vehicle_ID)

            for i,future_bb in enumerate (future_bbs):
                x,y=pipeline.get_zone_check_point(future_bb)
                if pipeline.is_in_zone(x,y):
                    future_id_list.append(vehicle_ID)
                    print('Warning!','Future Frame: ',i,'vehicle ID:',vehicle_ID)

        view.draw_danger_zone(rgb_image)
        if len(id_list)>0:
            view.show_zone_warning(rgb_image,id_list)
        if len(future_id_list)>0:
            view.show_future_warning(rgb_image,future_id_list)

        view.show_image_on_same_window(rgb_image)

    end_track = time.time()
    # Time elapsed
    seconds_track = end_track - start_track
    print ("Time taken : {0} seconds".format(seconds_track))

    # Calculate frames per second
    fps_track  = length / seconds_track
    print("Estimated frames per second : {0}".format(fps_track))

def save_sequence_trajectory_images(length):
    print('\n-------------------- save_sequence_trajectory_images --------------------\n')
    start_track = time.time()
    for frame in range (length):
        left_image = cv2.imread("./data/"+path+"_left/%010d" % frame+".png")
        right_image = cv2.imread("./data/"+path+"_right/%010d" % frame+".png")

        rgb_image, track_bbs_ids=pipeline.assignTrackID(left_image,right_image)
        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
            
            x,y=pipeline.get_zone_check_point(bb)
            if pipeline.is_in_zone_bb(bb) or pipeline.is_in_zone(x,y):
                id_list.append(vehicle_ID)
                x0,y0,x1,y1=bb
                print('x0,y0,x1,y1,', x0,y0,x1,y1)
                print('is_in_zone,',bb)


            # x,y=pipeline.get_zone_check_point(bb)
            # if pipeline.is_in_zone(x,y):
            #     id_list.append(vehicle_ID)
            #     x0,y0,x1,y1=bb
            #     print('x0,y0,x1,y1,', x0,y0,x1,y1)
            #     print('is_in_zone,',x,y)

            for i,future_bb in enumerate (future_bbs):
                x,y=pipeline.get_zone_check_point(future_bb)
                if pipeline.is_in_zone_bb(future_bb) or pipeline.is_in_zone(x,y):
                    future_id_list.append(vehicle_ID)
                    print('is_in_zone future_bb,',x,y)
                    print('is_in_zone,',bb)


            # for i,future_bb in enumerate (future_bbs):
            #     x,y=pipeline.get_zone_check_point(future_bb)
            #     if pipeline.is_in_zone(x,y):
            #         future_id_list.append(vehicle_ID)
            #         print('is_in_zone future_bb,',x,y)

        view.draw_danger_zone(rgb_image)
        if len(id_list)>0:
            view.show_zone_warning(rgb_image,id_list)
        if len(future_id_list)>0:
            view.show_future_warning(rgb_image,future_id_list)

        out_path = "./data/trajectory/%06d" % frame+".png"
        print(frame,cv2.imwrite(out_path, rgb_image))

    end_track = time.time()
    # Time elapsed
    seconds_track = end_track - start_track
    print ("Time taken : {0} seconds".format(seconds_track))

    # Calculate frames per second
    fps_track  = length / seconds_track
    print("Estimated frames per second : {0}".format(fps_track))


def save_sequence_trajectory_images_with_trapezoid_danger_zone(length):
    print('\n-------------------- save_sequence_trajectory_images --------------------\n')
    start_track = time.time()
    for frame in range (length):
        left_image = cv2.imread("./data/"+path+"_left/%06d" % frame+".png")
        right_image = cv2.imread("./data/"+path+"_right/%06d" % frame+".png")

        rgb_image, track_bbs_ids=pipeline.assignTrackID(left_image,right_image)
        id_list=[]
        future_id_list=[]

        for bbs in track_bbs_ids: #for each vehicle in a image frame
            bb=bbs[0] #current bounding box
            future_bbs=bbs[1] #future bounding boxes
            vehicle_ID=bbs[-1] #vehicle id
            
            if pipeline.is_in_trapezoid_zone_bb(bb):
                id_list.append(vehicle_ID)
                print('is_in_zone,',bb)

            for i,future_bb in enumerate (future_bbs):
                if pipeline.is_in_trapezoid_zone_bb(future_bb):
                    if vehicle_ID not in (future_id_list):
                        future_id_list.append(vehicle_ID)
                    print('is_in_zone,',bb)



        view.draw_danger_zone(rgb_image)
        if len(id_list)>0:
            view.show_zone_warning(rgb_image,id_list)
        if len(future_id_list)>0:
            view.show_future_warning(rgb_image,future_id_list)

        out_path = "./data/trajectory/%06d" % frame+".png"
        print(frame,cv2.imwrite(out_path, rgb_image))

    end_track = time.time()
    # Time elapsed
    seconds_track = end_track - start_track
    print ("Time taken : {0} seconds".format(seconds_track))

    # Calculate frames per second
    fps_track  = length / seconds_track
    print("Estimated frames per second : {0}".format(fps_track))

# test image folder


# path = 'kitti_s1' #836
# path = 'kitti_s2' #77
# path = 'kitti_s3' #446
# path = 'kitti_s4' #313
# path = 'kitti_s5' #21
# path = 'kitti_s6' #432

# path = 'carla_s1' #767
# path = 'carla_s2' #277
# path = 'carla_s3' #224
# path = 'carla_s4' #158

# path = 'local_s1' #7757
# path = 'local_s2'
path = 'local_s3'


pipeline = Pipeline(max_age=15, min_hits=1, iou_threshold=0.01, no_of_future_frames=8, no_of_past_frames=5)

# test_image_reading()
# test_image_showing()
# test_image_detection()
# test_sequence_image_detection(10)
# test_trajectory()
# test_sequence_trajectory_without_view(10)
# test_sequence_trajectory(100)
# save_sequence_trajectory_images(433)
save_sequence_trajectory_images_with_trapezoid_danger_zone(4000)