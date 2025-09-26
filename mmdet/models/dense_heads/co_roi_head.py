import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class CoStandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            assign_results = []
            sampling_results = []
            for i in range(num_imgs):
                # 对每张图片进行 bbox 正负样本属性分配
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # 然后进行正负样本采样
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                assign_results.append(assign_result)
                #print("assign_results",assign_results) [<AssignResult(num_gts=3, gt_inds.shape=(1003,), max_overlaps.shape=(1003,), labels.shape=(1003,)) at 0x7f028b7015d0>]
                #print("sampling_results", sampling_result)
                sampling_results.append(sampling_result)
                #<SamplingResult({'neg_bboxes': torch.Size([482, 4]),'neg_inds': tensor([,*482], device='cuda:0'),
                # 'num_gts': 5,
                # 'pos_assigned_gt_inds': tensor([0, 1, 2, 3, 4, 0, 1, 1, 0, 1], device='cuda:0'),
                #'pos_bboxes': torch.Size([10, 4]),
                #'pos_inds': tensor([  0,   1,   2,   3,   4, 215, 296, 593, 603, 910], device='cuda:0'),
                #'pos_is_gt': tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.uint8)})>

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, assign_results, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            # bbox_results为一个字典
            # cls_score: Tensor shape为[num_proposal_all_img, num_cls+1]
            # bbox_pred: Tensor shape为[num_proposal_all_img, num_cls*4]
            # bbox_feats Tensor shape为[num_proposals_all_img, C, 7, 7]
            # loss_bbox： Dict[str, *] 分別记录了loss_cls, acc, loss_bbox
            losses.update(bbox_results['loss_bbox'])

            #对每张图像的采样结果进行处理，限制每张图像的最大建议框数量，并提取相应的特征和目标，最终将这些信息整合起来更新模型的损失函数，以便进行后续的训练步骤
            bbox_targets = bbox_results['bbox_targets']
            num_imgs = len(img_metas)
            max_proposal = 2000
            for res in sampling_results:
                max_proposal =  min(max_proposal, res.bboxes.shape[0])
            ori_coords = bbox2roi([res.bboxes for res in sampling_results])#将不同图片的建议框列表转换成一个统一的坐标表示
            ori_proposals, ori_labels, ori_bbox_targets, ori_bbox_feats = [], [], [], []
            for i in range(num_imgs):
                idx = (ori_coords[:,0]==i).nonzero().squeeze(1)
                idx = idx[:max_proposal]
                ori_proposal = ori_coords[idx][:, 1:].unsqueeze(0)
                ori_label = bbox_targets[0][idx].unsqueeze(0)
                ori_bbox_target = bbox_targets[2][idx].unsqueeze(0)
                ori_bbox_feat = bbox_results['bbox_feats'].mean(-1).mean(-1)#所有建议框的特征，代码中对其进行了平均池化操作
                ori_bbox_feat = ori_bbox_feat[idx].unsqueeze(0)
                ori_proposals.append(ori_proposal) 
                ori_labels.append(ori_label)
                ori_bbox_targets.append(ori_bbox_target)
                ori_bbox_feats.append(ori_bbox_feat)
            ori_coords = torch.cat(ori_proposals, dim=0)
            ori_labels = torch.cat(ori_labels, dim=0)
            ori_bbox_targets = torch.cat(ori_bbox_targets, dim=0)
            ori_bbox_feats = torch.cat(ori_bbox_feats, dim=0)
            pos_coords = (ori_coords, ori_labels, ori_bbox_targets, ori_bbox_feats, 'rcnn')#包含了建议框、标签、边界框回归目标和边界框特征的元组
            losses.update(pos_coords=pos_coords)

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, assign_results, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # rois： Tensor  [all_proposals_all_img, 5] 将B张图像的采样得到proposals拼接在了一起, 第一列标记了proposal来自于哪张图像

        bbox_results = self._bbox_forward(x, rois)
        # bbox_results为一个字典, 内部有3个量，cls_score, bbox_pred, bbox_feats
        # cls_score: Tensor shape为[num_proposal_all_img, num_cls+1]
        # bbox_pred: Tensor shape为[num_proposal_all_img, num_cls*4]
        # bbox_feats Tensor shape为[num_proposals_all_img, C, 7, 7]

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # bbox_targets： Tuple[Tensor]
        # labels, label_weights, bbox_targets, bbox_weights4个Tensor
        # labels: 正样本样本位置为真实label, 负样本位置为类别个数， 这里coco为80  [num_proposal_all_img,]
        # label_weights:  正样本样本位置为可设置weight, 负样本位置为1.0， 这里均为1  [num_proposal_all_img,]
        # bbox_targets: 正样本样本位置为真实回归target, 负样本位置为0  [num_proposal_all_img,4]
        # bbox_weights: 正样本样本位置为1, 负样本位置为0  [num_proposal,4]

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        # 解析过，唯一的区别在于由于reg_class_agnostic=False，在计算bbox损失时会略有不同
        # bbox_pred shape为[num_proposal_all_img, num_cls*4],仅选取其匹配到的gt对应label的预测回归参数与真实回归参数进行损失计算
        # 此外由配置文件可知，这里bbox的损失使用的是L1 Loss
        # loss_bbox Dict[str, *]  分別记录了loss_cls, acc, loss_bbox

        num_proposals = [torch.sum(rois[:, 0] == i) for i in range(len(img_metas))]#num_proposals [tensor(512, device='cuda:0')]
        #print("num_proposals",num_proposals)
        cls_scores = bbox_results['cls_score'].clone().split(num_proposals)
        #for i in cls_scores:
            #print("cls_scores", i.shape)#torch.Size([512, 10])
        bbox_feats = bbox_results['bbox_feats'].clone().split(num_proposals)
        #for i in bbox_feats:
            #print("bbox_feats", i.shape)#torch.Size([512, 256, 7, 7])
        #comp_feats = bbox_results['comp_feats'].clone().split(num_proposals)  # [bs, num_proposals, 256, 1, 1]
        #print("comp_feats", comp_feats)
        proposal_labels = bbox_targets[0].clone().split(num_proposals)
        #for i in proposal_labels:
            #print("proposal_labels", i.shape)#torch.Size([512])

        con_losses = cls_scores[0].new_zeros(1)
        print("con_losses", con_losses)

        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
