#!/usr/bin/env python3
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import config.pipeline_config as configData

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collate_fn(batch):
    left_images = np.array([item["image"][0] for item in batch])#[batch, H, W, 3]
    left_images = left_images.transpose([0, 3, 1, 2])

    right_images = np.array([item["image"][1] for item in batch])#[batch, H, W, 3]
    right_images = right_images.transpose([0, 3, 1, 2])  
   
    return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float()
#     return torch.from_numpy(left_images).float(), torch.from_numpy(right_images).float()

class Yolo3DNode:
    def __init__(self):
        print("Starting Yolo3DNode.")
        self._read_params()
        self._init_model()

    def _read_params(self):
        print("Reading params.")
        from visualDet3D.utils.utils import cfg_from_file
        from visualDet3D.networks.utils import BBox3dProjector, BackProjection

        self.cfg = cfg_from_file("config/config.py")
        self.cfg.detector.backbone.pretrained=False

        checkpoint_name = "Stereo3D_latest.pth"
        self.weight_path = os.path.join(self.cfg.path.checkpoint_path, checkpoint_name)

        # self.inference_w   = 1280 #int(rospy.get_param("~INFERENCE_W",  1280))
        # self.inference_h   = 288 #int(rospy.get_param("~INFERENCE_H",  288))
        # self.crop_top      = 100 #int(rospy.get_param("~CROP_TOP", 100))
        self.inference_scale = 1 #float(rospy.get_param("~INFERENCE_SCALE", 1.0))
        # self.cfg.data.test_augmentation[1].keywords.crop_top_index = self.crop_top 
        # self.cfg.data.test_augmentation[2].keywords.size = (self.inference_h, self.inference_w)
        
        self.P2 = configData.P2
        self.P3 = configData.P3

        # Load projector and backprojector
        self.projector = BBox3dProjector().cuda()
        self.backprojector = BackProjection().cuda()

    def _init_model(self):
        print("Loading model.")
        from visualDet3D.networks.utils.registry import DETECTOR_DICT, PIPELINE_DICT
        from visualDet3D.data.pipeline import build_augmentator

         # Build a detector network
        detector = DETECTOR_DICT[self.cfg.detector.name](self.cfg.detector)
        self.detector = detector.cuda()
        # Tensor load by GPU
        state_dict = torch.load(
            self.weight_path, map_location='cuda:{}'.format(self.cfg.trainer.gpu)
        )
        self.detector.load_state_dict(state_dict, strict=False)
        self.detector.eval()
        
        # self.ort_session = ort.InferenceSession(self.onnx_path)

        self.transform = build_augmentator(self.cfg.data.test_augmentation)
        self.test_func = PIPELINE_DICT[self.cfg.trainer.test_func]
        print("Done loading model.")

    def denorm(self,image):
        new_image = np.array((image * self.cfg.data.augmentation.rgb_std +  self.cfg.data.augmentation.rgb_mean) * 255, dtype=np.uint8)
        return new_image
    
    def predict(self, left_image, right_image):
        transformed_left_image, transformed_P2 = self.transform(left_image.copy(), p2=self.P2.copy())
        transformed_right_image, transformed_P3 = self.transform(right_image.copy(), p3=self.P3.copy())
        data = {'image': [transformed_left_image,transformed_right_image]}
        data = collate_fn([data])
#         data = data + (transformed_P2, transformed_P3)
#         print('data',data)
         
        with torch.no_grad():
            left_images, right_images = data[0], data[1]
            
            transformed_P2=[transformed_P2]
            transformed_P3=[transformed_P3]
            transformed_P2=torch.tensor(transformed_P2).float()
            transformed_P3=torch.tensor(transformed_P3).float()
            scores, bbox, obj_names = self.detector([left_images.cuda().float().contiguous(),
                                          right_images.cuda().float().contiguous(),
                                          transformed_P2.cuda().float(),
                                          transformed_P3.cuda().float()])
#             scores, bbox, obj_names = self.test_func(data, self.detector, None, cfg=self.cfg)
            transformed_P2 = transformed_P2[0]
            bbox_2d = bbox[:, 0:4]
            bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha]
            bbox_3d_state[:, 2] *= self.inference_scale
            bbox_3d_state_3d = self.backprojector(bbox_3d_state, transformed_P2.cuda()) #[x, y, z, w,h ,l, alpha]
#             abs_bbox, bbox_3d_corner_homo, thetas = self.projector(bbox_3d_state_3d, bbox_3d_state_3d.new(transformed_P2))
            abs_bbox, bbox_3d_corner_homo, thetas = self.projector(bbox_3d_state_3d, transformed_P2.cuda())
        
#             rgb_image = self.denorm(left_images[0].transpose([1, 2, 0]))
            rgb_image = self.denorm(left_images[0].cpu().numpy().transpose([1, 2, 0]))
        
            return rgb_image, scores, bbox_2d, obj_names, bbox_3d_corner_homo
    
    
