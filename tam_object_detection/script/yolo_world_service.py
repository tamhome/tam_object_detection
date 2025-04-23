#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import random
import numpy as np
import os.path as osp
from torch import Tensor
import supervision as sv
from tamlib.tf import Transform
from tamlib.open3d import Open3D
from PIL import Image as PILImage
import groundingdino.datasets.transforms as T

from torchvision.ops import nms
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast

from typing import Any, Dict, List, Optional, Tuple
from tam_object_detection.msg import BBox, ObjectDetection, Pose3D
from groundingdino.util.inference import load_model, load_image, predict, annotate

import rospy
import roslib
from std_msgs.msg import Header
from tamlib.node_template import Node
from tamlib.cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tam_object_detection.msg import ObjectDetection
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from tam_object_detection.srv import LangSamObjectDetectionService
from tam_object_detection.srv import LangSamObjectDetectionServiceResponse


class YoloWorldService(Node):
    def __init__(self) -> None:
        super().__init__()

        # Parameters
        self.p_action_name = rospy.get_param("~action_name", "yolo_world_object_detection")
        p_config_path = rospy.get_param(
            "~config_path",
            osp.join(
                roslib.packages.get_pkg_dir("tam_object_detection"),
                "io/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
        )
        p_weight_path = rospy.get_param(
            "~weight_path",
            osp.join(
                roslib.packages.get_pkg_dir("tam_object_detection"),
                "io/groundingdino/weights/groundingdino_swint_ogc.pth",
            ),
        )
        self.p_device = rospy.get_param("~device", "cuda:0")
        self.p_confidence_th = rospy.get_param("~confidence_th", 0.3)
        self.p_iou_th = rospy.get_param("~iou_th", 0.65)
        self.p_max_area_raito = rospy.get_param("~max_area_ratio", 0.3)
        self.p_topk = rospy.get_param("~topk", 100)

        self.p_camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/rgb/camera_info"
        )
        self.p_rgb_topic = rospy.get_param("~rgb_topic", "/camera/rgb/image_raw")
        self.p_use_depth = rospy.get_param("~use_depth", False)
        self.p_depth_topic = rospy.get_param(
            "~depth_topic", "/camera/depth_registered/image_raw"
        )

        self.p_use_segment = rospy.get_param("~use_segment", True)
        self.p_use_latest_image = rospy.get_param("~use_latest_image", False)
        self.p_get_pose = rospy.get_param("~get_pose", False)
        self.p_show_tf = rospy.get_param("~show_tf", False)
        self.p_max_distance = rospy.get_param("~max_distance", -1)
        self.p_specific_id = rospy.get_param("~specific_id", "")

        # Service
        self.pkg_dir = roslib.packages.get_pkg_dir("tam_object_detection")
        self.gdino_model = load_model(p_config_path, p_weight_path)
        self.srv_detection = rospy.Service(
            f"{self.p_action_name}/service", LangSamObjectDetectionService, self.run
        )
        self.p_confidence_th_default = self.p_confidence_th
        self.p_iou_th_default = self.p_iou_th
        self.p_max_distance_default = self.p_max_distance
        self.p_prompt = None

        self.bridge = CvBridge()
        self.tamtf = Transform()
        self.open3d = Open3D()

        self.logdebug("set publisher node")
        # Publisher
        self.pub_register("result_image", "object_detection/image", Image)
        self.pub_register("result", f"{self.p_action_name}/detection", ObjectDetection)

        self.logdebug("create camera model")
        self.camera_info = rospy.wait_for_message(self.p_camera_info_topic, CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.camera_frame_id = self.camera_info.header.frame_id

        # self.loginfo("Language Segment Anything object detection model loading...")
        # self.lang_sam_model = LangSAM()
        # self.logsuccess("Language Segment Anything object detection service is ready.")

        self.loginfo("yolo-worldの初期化")
        cfg = Config.fromfile(p_weight_path)
        cfg.work_dir = "."
        cfg.load_from = p_config_path
        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        runner.pipeline = Compose(pipeline)
        runner.model.eval()

        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.class_names = ("all")

    def save_img(self, cv_img: np.ndarray) -> str:
        """画像を保存し，そのパスを返してくれる関数
        """
        cv2.imwrite(cv_img, f"{self.pkg_dir}/io/temp.png")
        return f"{self.pkg_dir}/io/temp.png"

    def run_image(self, cv_img: np.ndarray, max_num_boxes=100, score_thr=0.05, nms_thr=0.5, output_image="output.png"):
        image_path = self.save_img(cv_img=cv_img)
        texts = [[t.strip()] for t in self.class_names.split(",")] + [[" "]]
        data_info = self.runner.pipeline(dict(img_id=0, img_path=image_path, texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        pred_instances = pred_instances.cpu().numpy()
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores']
        )

        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        image = PILImage.open(image_path)
        svimage = np.array(image)
        svimage = self.bounding_box_annotator.annotate(svimage, detections)
        svimage = self.label_annotator.annotate(svimage, detections, labels)
        return svimage[:, :, ::-1]

    def set_params(self, req: Any):
        if req.confidence_th <= 0:
            self.logwarn("confidence_th must be positive")
            self.loginfo(
                f"use the default value [{self.p_confidence_th}] for confidence_th"
            )
            self.p_confidence_th = self.p_confidence_th_default
        else:
            self.p_confidence_th = req.confidence_th
        if req.iou_th <= 0:
            self.logwarn("iou_th must be positive")
            self.loginfo(f"use the default value [{self.p_iou_th}] for iou_th")
            self.p_iou_th = self.p_iou_th_default
        else:
            self.p_iou_th = req.iou_th
        self.p_use_latest_image = req.use_latest_image
        if req.max_distance != 0:
            self.p_max_distance = req.max_distance
        else:
            self.p_max_distance = self.p_max_distance_default
        self.p_specific_id = req.specific_id
        self.p_prompt = req.prompt

    def return_no_objects(self, bgr=None):
        if bgr is not None:
            msg_rgb = self.bridge.cv2_to_imgmsg(bgr)
            self.pub.result_image.publish(msg_rgb)
        msg = ObjectDetection()
        msg.is_detected = False
        return msg

    def cv2_to_pil(self, image) -> PILImage:
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = PILImage.fromarray(new_image)
        return new_image

    def pil_to_transformed_image(self, pil_rgb: PILImage):
        # cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(pil_rgb, None)
        return image_transformed

    def show_tf(self, poses: List[Optional[Pose]], labels: Tensor) -> None:
        """TFを配信する

        Args:
            poses (List[Optional[Pose]]): 3次元座標リスト．
            labels (Tensor): ラベル情報．
        """
        label_dict: Dict[str, int] = {}
        index = 0
        for pose, label in zip(poses, labels):
            # label = int(label)
            if pose is None:
                continue
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
            self.tamtf.send_transform(
                label + str(index),
                self.camera_frame_id,
                pose,
            )
            index += 1

    def visualize(
        self,
        image: np.ndarray,
        bboxes: Tensor,
        labels: Tensor,
        masks: Optional[List[Tensor]],
    ) -> np.ndarray:
        """結果の可視化

        Args:
            image (np.ndarray): 入力画像．
            bboxes (Tensor): BBox情報．
            labels (Tensor): ラベル情報．
            masks (Optional[List[Tensor]], optional): マスク情報. Defaults to None.

        Returns:
            np.ndarray: 描画画像．
        """
        result_image = image.copy()

        # if masks is not None:
        #     self.seg_img = torch.asarray(
        #         self.seg_img[
        #             self.dh : self.H - self.dh, self.dw : self.W - self.dw, [2, 1, 0]
        #         ],
        #         device=self.p_device,
        #     )
        #     self.mask, mask_color = [
        #         m[:, self.dh : self.H - self.dh, self.dw : self.W - self.dw, :]
        #         for m in masks
        #     ]
        #     inv_alph_masks = (1 - self.mask * 0.5).cumprod(0)
        #     mcs = (mask_color * inv_alph_masks).sum(0) * 2
        #     self.seg_img = (self.seg_img * inv_alph_masks[-1] + mcs) * 255
        #     result_image = cv2.resize(
        #         self.seg_img.cpu().numpy().astype(np.uint8),
        #         result_image.shape[:2][::-1],
        #     )

        for (bbox, label) in zip(bboxes, labels):
            # bbox = bbox.round().int().tolist()
            # cls = self.class_names[int(label)]
            # color = self.colors[cls]
            random.seed(hash(label))
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (b, g, r)
            cv2.rectangle(result_image, bbox[:2], bbox[2:], color, 2)
            cv2.putText(
                result_image,
                label,
                (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                [0, 0, 0],
                # [255, 255, 255],
                thickness=2,
            )

        return result_image

    def create_segment_msg(self, num: int) -> CompressedImage:
        """segmentメッセージを作成する

        Args:
            num (int): 検出数．

        Returns:
            List[CompressedImage]: segmentメッセージリスト．
        """
        # mask = self.mask.to("cpu").detach().numpy().copy() * 255
        # mask_split = self.masks
        masks = self.masks
        seg_msgs = []
        for i in range(num):
            m = np.array(masks[i])
            seg_msg = self.bridge.cv2_to_compressed_imgmsg(m)
            seg_msgs.append(seg_msg)
        return seg_msgs

    def create_object_detection_msg(
        self,
        msg_rgb: CompressedImage,
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        msg_depth: Optional[np.ndarray] = None,
        poses: Optional[List] = None,
    ) -> ObjectDetection:
        """ObjectDetection.msgを作成する

        Args:
            msg_rgb (CompressedImage): RGBメッセージ．
            bboxes (Tensor): BBox情報．
            scores (Tensor): スコア情報．
            labels (Tensor): ラベル情報．
            msg_depth (Optional[CompressedImage, optional): Depthメッセージ. Defaults to None.
            poses (Optional[List], optional): 3次元座標リスト. Defaults to None.

        Returns:
            ObjectDetection: メッセージ．
        """
        msg = ObjectDetection()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.camera_frame_id
        msg.is_detected = True
        msg.camera_info = self.camera_info
        msg.rgb = msg_rgb
        if self.p_use_segment:
            msg.segments = self.create_segment_msg(len(bboxes))
        index = 0
        for bbox, score, label in zip(bboxes, scores, labels):
            msg.bbox.append(BBox())
            msg.bbox[-1].id = index
            msg.bbox[-1].name = label
            msg.bbox[-1].score = float(score)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = bbox[0] + w / 2, bbox[1] + h / 2
            msg.bbox[-1].x = int(x)
            msg.bbox[-1].y = int(y)
            msg.bbox[-1].w = int(w)
            msg.bbox[-1].h = int(h)
            index += 1
        if msg_depth is not None and poses is not None:
            msg.depth = msg_depth
            for pose in poses:
                msg.pose.append(Pose3D())
                if pose is None:
                    msg.pose[-1].is_valid = False
                else:
                    msg.pose[-1].is_valid = True
                    msg.pose[-1].frame_id = self.camera_frame_id
                    msg.pose[-1].position = pose.position
                    msg.pose[-1].orientation = pose.orientation
        return msg

    def get_3d_poses(self, depth: np.ndarray, bboxes: Tensor) -> List[Optional[Pose]]:
        """カメラ座標系での三次元座標を取得する（物体中心）

        Args:
            depth (np.ndarray): Depth画像．
            bboxes (Tensor): BBox情報．

        Returns:
            List[Optional[Pose]]: 各オブジェクトの3次元座標．
                計算できなかった場合，Noneが格納される．
        """
        poses: List[Optional[Pose]] = []
        for box in bboxes:
            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = int(box[0] + w / 2), int(box[1] + h / 2)
            crop_depth = depth[cy - 2 : cy + 3, cx - 2 : cx + 3] * 0.001
            flat_depth = crop_depth[crop_depth != 0].flatten()
            if len(flat_depth) == 0:
                poses.append(None)
                continue
            mean_depth = np.mean(flat_depth)
            uv = list(self.camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * mean_depth for x in uv]
            if self.p_max_distance < 0 or (
                self.p_max_distance > 0 and self.p_max_distance > uv[2]
            ):
                poses.append(Pose(Point(*uv), Quaternion(0, 0, 0, 1)))
            else:
                poses.append(None)
        return poses

    def flatten_nested_list(self, nested_lists):
        flat_lists = []
        for nested_list in nested_lists:
            flat_list = []
            for sub_list in nested_list:
                for item in sub_list:
                    flat_list.append(item[0][0])
                    flat_list.append(item[0][1])
            flat_lists.append(flat_list)
        return flat_lists

    def run(self, req) -> LangSamObjectDetectionServiceResponse:
        try:
            self.set_params(req)
            self.logdebug("get request")

            # RGB画像取得
            while not rospy.is_shutdown():
                if self.p_use_latest_image:
                    self.logdebug("wait for new message")
                    self.msg_rgb = rospy.wait_for_message(self.p_rgb_topic, CompressedImage)
                    self.msg_depth = rospy.wait_for_message(self.p_depth_topic, CompressedImage)
                    self.cv_bgr = self.bridge.compressed_imgmsg_to_cv2(self.msg_rgb)
                    self.cv_depth = self.bridge.compressed_imgmsg_to_depth(self.msg_depth)

                cv_bgr = self.cv_bgr
                msg_rgb = self.msg_rgb
                pil_rgb = self.cv2_to_pil(cv_bgr)
                image_transformed = self.pil_to_transformed_image(pil_rgb)
                break
            self.loginfo("get latest image")

            # 推論
            try:
                # bboxes, scores, labels, masks = self.inference(cv_bgr)
                # masks, boxes, phrases, logits = predict(
                tensor_masks, tensor_bboxes, labels, scores = self.lang_sam_model.predict(
                    pil_rgb, self.p_prompt
                )
                # maskの整形
                masks_np = [mask.squeeze().cpu().numpy().astype(np.uint8)*255 for mask in tensor_masks]
                masks = []
                for mask_np in masks_np:
                    # Contoursを検出
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    masks.append(contours)
                self.masks = self.flatten_nested_list(masks)

                # bboxesの整形
                bboxes = tensor_bboxes.to('cpu').detach().numpy().copy().astype(np.uint16)

            except Exception as e:
                self.logerr(e)
                self.logwarn("No objects detected.")
                return self.return_no_objects(cv_bgr)

            # # size validation
            # bboxes, scores, labels, masks = yolov8_utils.filter_elements_by_area_ratio(
            #     self.p_max_area_raito, bboxes, scores, labels, masks
            # )

            # # top k
            # bboxes, scores, labels, masks = yolov8_utils.filter_elements_by_topk(
            #     self.p_topk, bboxes, scores, labels, masks
            # )

            # # 特定ラベルのみ抽出
            # if self.p_specific_id != "":
            #     bboxes, scores, labels, masks = yolov8_utils.filter_elements_by_id(
            #         self.p_specific_id, bboxes, scores, labels, masks
            #     )
            # if len(bboxes) == 0:
            #     return self.return_no_objects(cv_bgr)

            # Depth画像処理
            if self.p_use_depth:
                if not hasattr(self, "cv_depth"):
                    self.logdebug("cv_depth has not attr")
                    return
                cv_depth = self.cv_depth
                msg_depth = self.msg_depth

                # 3次元座標の取得
                poses = self.get_3d_poses(cv_depth, bboxes)

                # 座標の無い結果の削除
                # poses, bboxes, scores, labels, masks = yolov8_utils.filter_elements_by_pose(
                #     poses, bboxes, scores, labels, masks
                # )
                if len(bboxes) == 0:
                    self.logdebug("return no objects")
                    return self.return_no_objects(cv_bgr)

                # TF配信
                if self.p_show_tf and poses != []:
                    self.show_tf(poses, labels)
            else:
                msg_depth = None
                poses = None

            # 可視化
            cv_result = self.visualize(cv_bgr, bboxes, labels, None)
            msg_result = self.bridge.cv2_to_imgmsg(cv_result)
            self.pub.result_image.publish(msg_result)

            # 推論結果
            result = LangSamObjectDetectionServiceResponse()
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

    cls = YoloWorldService()
    rospy.on_shutdown(cls.delete)
    while not rospy.is_shutdown():
        loop_wait.sleep()


if __name__ == "__main__":
    main()
