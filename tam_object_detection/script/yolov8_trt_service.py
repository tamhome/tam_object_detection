#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
from typing import Any

import roslib
import rospy
from tam_object_detection.msg import ObjectDetection
from tam_object_detection.srv import (
    ObjectDetectionService,
    ObjectDetectionServiceResponse,
)
from yolov8_trt_node import YOLOv8TensorRT

sys.path.append(
    osp.join(
        roslib.packages.get_pkg_dir("tam_object_detection"),
        "../third_party/YOLOv8-TensorRT/",
    )
)
import infer as yolov8_trt
from models import TRTModule


class YOLOv8TensorRTService(YOLOv8TensorRT):
    def __init__(self) -> None:
        super().__init__()

        # Service
        self.srv_detection = rospy.Service(
            f"{self.p_action_name}/service", ObjectDetectionService, self.run
        )
        self.loginfo("ObjectDetection Service is ready.")

        self.p_confidence_th_default = self.p_confidence_th
        self.p_iou_th_default = self.p_iou_th
        self.p_max_distance_default = self.p_max_distance

    def set_params(self, req: Any):
        if req.confidence_th <= 0:
            # self.logwarn("confidence_th must be positive")
            # self.loginfo(
            #     f"use the default value [{self.p_confidence_th}] for confidence_th"
            # )
            self.p_confidence_th = self.p_confidence_th_default
        else:
            self.p_confidence_th = req.confidence_th
        if req.iou_th <= 0:
            # self.logwarn("iou_th must be positive")
            # self.loginfo(f"use the default value [{self.p_iou_th}] for iou_th")
            self.p_iou_th = self.p_iou_th_default
        else:
            self.p_iou_th = req.iou_th
        self.p_use_latest_image = req.use_latest_image
        if req.max_distance != 0:
            self.p_max_distance = req.max_distance
        else:
            self.p_max_distance = self.p_max_distance_default
        self.p_specific_id = req.specific_id

    def return_no_objects(self, bgr=None):
        if bgr is not None:
            msg_rgb = self.bridge.cv2_to_imgmsg(bgr)
            self.pub.result_image.publish(msg_rgb)
        msg = ObjectDetection()
        msg.is_detected = False
        return msg

    def run(self, req) -> ObjectDetectionServiceResponse:
        try:
            self.set_params(req)

            # RGB画像取得
            while not rospy.is_shutdown():
                if not hasattr(self, "cv_bgr"):
                    continue
                if self.p_use_latest_image:
                    self.wait_for_message("msg_rgb")
                cv_bgr = self.cv_bgr
                msg_rgb = self.msg_rgb
                break

            # 推論
            try:
                bboxes, scores, labels, masks = self.inference(cv_bgr)
            except Exception:
                self.logwarn("No objects detected.")
                return self.return_no_objects(cv_bgr)

            # size validation
            bboxes, scores, labels, masks = self.filter_elements_by_area_ratio(
                self.p_max_area_raito, bboxes, scores, labels, masks
            )

            # top k
            bboxes, scores, labels, masks = self.filter_elements_by_topk(
                self.p_topk, bboxes, scores, labels, masks
            )

            # 特定ラベルのみ抽出
            if self.p_specific_id != "":
                bboxes, scores, labels, masks = self.filter_elements_by_id(
                    self.p_specific_id, bboxes, scores, labels, masks
                )
            if len(bboxes) == 0:
                return self.return_no_objects(cv_bgr)

            # Depth画像処理
            if self.p_use_depth:
                if not hasattr(self, "cv_depth"):
                    return
                cv_depth = self.cv_depth
                msg_depth = self.msg_depth

                # 3次元座標の取得
                poses = self.get_3d_poses(cv_depth, bboxes)

                # 座標の無い結果の削除
                poses, bboxes, scores, labels, masks = self.filter_elements_by_pose(
                    poses, bboxes, scores, labels, masks
                )
                if len(bboxes) == 0:
                    return self.return_no_objects(cv_bgr)

                # TF配信
                if self.p_show_tf and poses != []:
                    self.show_tf(poses, labels)
            else:
                msg_depth = None
                poses = None

            # 可視化
            cv_result = self.visualize(cv_bgr, bboxes, labels, masks)
            msg_result = self.bridge.cv2_to_imgmsg(cv_result)
            self.pub.result_image.publish(msg_result)

            # 推論結果
            result = ObjectDetectionServiceResponse()
            result.detections = self.create_object_detection_msg(
                msg_rgb, bboxes, scores, labels, msg_depth, poses
            )
            return result

        except Exception as e:
            self.logerr(e)
            return self.return_no_objects()


def main():
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    p_loop_rate = rospy.get_param(rospy.get_name() + "/loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)

    cls = YOLOv8TensorRTService()
    rospy.on_shutdown(cls.delete)
    while not rospy.is_shutdown():
        loop_wait.sleep()


if __name__ == "__main__":
    main()
