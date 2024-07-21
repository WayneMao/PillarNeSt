# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply

from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D, BboxOverlaps3D
from .centerpoint_head import CenterHead


@HEADS.register_module()
class CenterPlusHead(CenterHead):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_iou_reg=None,  # 'IoU', 'DIoU'
                 iou_reg=dict(type='BboxOverlaps3D', coordinate='lidar'),
                 iou_reg_weight=0.25,
                 iou_score=dict(type='BboxOverlaps3D', coordinate='lidar'),
                 loss_iou_score=dict(
                     type='L1Loss', reduction='none', loss_weight=1.0),
                 iou_score_weight=1.0,
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterPlusHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            num_heatmap_convs=num_heatmap_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            norm_bbox=norm_bbox,
            init_cfg=init_cfg
        )

        self.with_iou_score = 'iou' in common_heads
        if self.with_iou_score:
            self.iou_score = build_iou_calculator(iou_score)
            self.iou_score_beta = test_cfg.get('iou_score_beta', 0.5)
            self.loss_iou_score = build_loss(loss_iou_score)
            self.iou_score_weight = iou_score_weight

        if loss_iou_reg is not None:
            self.with_iou_reg = True
            self.iou_reg = build_iou_calculator(iou_reg)
            self.loss_iou_reg = loss_iou_reg
            self.loss_reg_weight = iou_reg_weight
        else:
            self.with_iou_reg = False

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function. 6个task

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.with_iou_score:
                preds_dict[0]['iou'] = clip_sigmoid(preds_dict[0]['iou'])
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel'], preds_dict[0]['iou']),
                    dim=1)
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel']),
                    dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)

            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            if self.with_iou_score or self.with_iou_reg:
                dim = 7  # xs, ys, z, dx, dy, dz, yh
                batch_size = pred.size(0)
                real_pred_bbox = self.get_really_bboxes(pred, ind).view(-1, dim)
                real_target_bbox = self.get_really_bboxes(target_box, ind).view(-1, dim)

                if self.with_iou_score:
                    # get IoU target
                    iou_score = self.iou_score(real_target_bbox, real_pred_bbox)
                    iou_score = torch.diag(iou_score).view(batch_size, -1, 1).detach()

                    iou_score_pred = pred[:, :, -1:]
                    iou_mask = masks[task_id].unsqueeze(2).expand_as(iou_score).float()
                    isnotnan = (~torch.isnan(iou_score)).float()
                    iou_mask *= isnotnan
                    loss_iou_score = self.loss_iou_score(iou_score_pred, iou_score, iou_mask, avg_factor=(num + 1e-4))
                    loss_iou_score *= self.iou_score_weight

                    loss_dict[f'task{task_id}.loss_iou_score'] = loss_iou_score

                    pred = pred[:, :, :10]

                if self.with_iou_reg:
                    iou = self.iou_reg(real_target_bbox, real_pred_bbox)
                    iou = torch.diag(iou).view(batch_size, -1)
                    if self.loss_iou_reg in ['IoU', 'iou']:
                        loss_iou_reg = ((1. - iou) * masks[task_id]).sum() / (masks[task_id].sum() + 1e-4)
                        loss_iou_reg *= self.loss_reg_weight
                    loss_dict[f'task{task_id}.loss_iou_reg'] = loss_iou_reg

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def get_really_bboxes(self, pred, ind):
        out_size_factor = self.train_cfg['out_size_factor']
        voxel_size = self.train_cfg['voxel_size']
        pc_range = self.train_cfg['point_cloud_range']

        # get really bboxes
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        feature_map_size = grid_size[:2] // out_size_factor
        xs = ind % feature_map_size[1]
        ys = ind // feature_map_size[1]

        xs = xs[:, :, None] + pred[:, :, 0:1]
        ys = ys[:, :, None] + pred[:, :, 1:2]

        xs = xs * out_size_factor * voxel_size[0] + pc_range[0]
        ys = ys * out_size_factor * voxel_size[1] + pc_range[1]

        # get rotation value
        rot_sin = pred[:, :, 6:7]
        rot_cos = pred[:, :, 7:8]
        pred_rot = torch.atan2(rot_sin, rot_cos)

        z = pred[:, :, 2:3]
        dim = torch.exp(torch.clamp(pred[:, :, 3:6].clone(), min=-5, max=5))

        pred_bboxes = torch.cat([xs, ys, z, dim, pred_rot], dim=2)

        return pred_bboxes

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):  # 6 task
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.with_iou_score:
                batch_iou = preds_dict[0]['iou'].sigmoid()
            else:
                batch_iou = None

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                iou_scores=batch_iou,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]

            if self.with_iou_score:
                batch_cls_preds = [torch.pow(box['scores'], 1 - self.iou_score_beta) *
                                   torch.pow(box['iou_scores'], self.iou_score_beta) for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    if self.with_iou_score:
                        iou_scores = temp[i]['iou_scores']
                        scores = torch.pow(scores, 1 - self.iou_score_beta) * torch.pow(iou_scores, self.iou_score_beta)
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list
