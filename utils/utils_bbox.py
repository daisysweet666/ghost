"""对anchor_box进行调整：解码过程"""
import torch
# import torch.nn as nn
from torchvision.ops import nms
import numpy as np


class DecodeBox:
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors             # 9个anchor
        self.num_classes = num_classes     # 1类
        self.bbox_attrs = 5 + num_classes  # x,y,w,h,obj_conf,conf_cls
        '''obj_conf：预测框预测的是物体的概率
           conf_cls：预测框已经预测了是一个物体，这个物体属于某一种类的概率'''
        self.input_shape = input_shape     # [416,416]
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        """解码"""
        outputs = []
        for i, input in enumerate(inputs):  # input是网络预测的outputs：[bs,c,w,h]
            batch_size = input.size(0)      # 一共有bs张图片
            input_width = input.size(2)     # 获取图片的高、宽
            input_height = input.size(3)
            stride_w = self.input_shape[0] / input_width   # 计算步长：416/13=32
            stride_h = self.input_shape[1] / input_height  # 416/26=16 416/52=8...

            '''
            Q：为什么要除以步长呢？
            A：因为 当前的anchor模板是对应原始图像(416*416)，而网络预测是在三个特征层(13*13,26*26,52*52...)上进行的，
            以(13*13)为例：下采样了32倍，也就是步长为32；
            将416*416的图片分成13*13个grid cell，每个grid cell代表着原图上的32*32的区域，13*13特征层的感受野就是32*32；
            在yolo_v3中将grid cell归一化为1*1大小的方格，也就是缩小了32倍，对应原图就需要用416/32。
            '''
            # 对应特征层的scaled_anchors为[w/32, h/32]
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                              for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]
            # 将bbox_attrs(x,y,w,h,obj,conf_cls)=6，转到最后一个通道 prediction：(bs,3,6,13,13) -> (bs,3,13,13,6)
            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width)\
                .permute(0, 1, 3, 4, 2).contiguous()

            # 先验框的调整参数：中心位置、宽高
            x = torch.sigmoid(prediction[..., 0])          # 抽取bbox_attrs的x列展开
            y = torch.sigmoid(prediction[..., 1])          # 抽取bbox_attrs的y列展开
            w = prediction[..., 2]                         # 抽取bbox_attrs的w列展开
            h = prediction[..., 3]                         # 抽取bbox_attrs的h列展开
            # 获得种类置信度image_pred，种类置信度pred_cls
            conf = torch.sigmoid(prediction[..., 4])       # 抽取bbox_attrs的obj列展开
            #  bs,3,13,13,conf_cls(因为我只有一个类别，所以可以直接prediction[..., 5])
            pred_cls = torch.sigmoid(prediction[..., 5:])  # 抽取bbox_attrs的conf_cls列展开

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor  # 默认生成32位浮点数
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成网格，先验框中心（网格的左上角）
            # grid_x维度为x.shape:[bs,3,13,13],grid_y维度为y.shape:[bs,3,13,13]
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).\
                repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            # torch.linspace().repeat().t()只能转置2维
            '''Q：这里的t()转置有什么用？
            h.repeat(w,1)不就是y.shape的13*13了吗？'''
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).\
                t().repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # 先验框的宽高 scaled_anchors为3个[w/32,h/32]，3个[w/16, h/16]，3个[w/8, h/8]
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))  # 获取第1个维度且索引号为0的张量子集
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            # 维度：[bs,3,13,13]
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # 利用预测结果对先验框进行调整，调整先验框的中心、宽高
            pred_boxes = FloatTensor(prediction[..., :4].shape)  # prediction的最后一个维度的前4个序号：(bs,3,13,13,6)
            pred_boxes[..., 0] = x.data + grid_x  # 最终预测框的坐标 中心点
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # 将输出结果归一化成小数的形式 output=[bs,3,13,13,6(x,y,w,h,image_pred,pred_cls)]
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            # _scale将特征层上预测框的大小又恢复到原始图片(416*416)上的大小
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale, conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs  # outputs就是实际框

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """去掉resize后图片的上下灰框"""
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]   # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))  # new_shape指的是宽高缩放情况
            offset = (input_shape - new_shape)/2./input_shape   # offset是图像有效区域相对于图像左上角的偏移情况
            scale = input_shape/new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                                box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape,
                            letterbox_image, conf_thres=0.5, nms_thres=0.4):
        """
        非极大抑制：
        1、根据分类器的类别分类概率，对所有预测框预测为某一物体的概率做排序
        假设有5个预测框，预测为person的概率为A(0.7) B(0.5) C(0.4) D(0.3) E(0.2)排好序
        2、选定最大概率的A，分别判断B、C、D、E与A的重叠度IOU是否大于设定的阈值，超过阈值丢弃
        假设IOU(A,B)超过阈值，丢弃B，保留A框；剩余C、D、E框；
          再从C、D、E框中选出大概率的C框，判断D、E与C的重叠度IOU是否大于设定的阈值，超过则丢弃
        假设IOU(C,D)超过阈值，丢弃D，保留C框；剩余E框保留。
        """
        # 将预测结果的中心宽高格式转换成左上角右下角的格式
        # prediction:[bs,3,13,13,6]  6=x,y,w,h,obj_conf,pred_cls
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # X1
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # Y1
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # X2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # Y2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]

        # 因为image是有bs维度的，所以需要进行循环遍历batch_size里面的image
        for i, image_pred in enumerate(prediction):
            # 取出prediction中第i个预测结果image_pred:[num_classes,6]  6=x1,y1,x2,y2,obj_conf,pred_cls
            # 对预测结果第2维 第5个序号之后的内容(pred_cls)取max
            # 取max后的结果为：当前预测框最大的种类置信度class_conf[num_anchors,?]，当前预测框所属种类class_pred
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # 利用置信度进行第一轮筛选（将包含物体的预测框与包含物体概率最大的那个相乘得到的IOU判断阈值）
            # IOU = 是否包含物体conf + 包含物体属于的种类conf
            # image_pred[:, 4]  预测框内部包含一个物体的置信度（obj_conf）
            # class_conf[:, 0]  当前预测框最大的种类置信度
            '''为什么class_conf有[:, 0],它是2维张量吗？'''
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            # 根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]  # 预测结果
            class_conf = class_conf[conf_mask]  # 当前预测框最大的种类置信度
            class_pred = class_pred[conf_mask]  # 当前预测框所属于的种类
            if not image_pred.size(0):  # 如果没有框，直接进行下一张图片的处理
                continue
            # prediction：[bs,3,13,13,6]    6的内容为：x,y,w,h,obj_conf,pred_cls
            # detections：[num_anchors,7]   7的内容为: x1,y1,x2,y2,obj_conf,class_conf,class_pred
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 使用官方自带的非极大抑制
            for c in unique_labels:
                # 只对包含的种类进行循环，获得某一类得分筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]
                keep = nms(detections_class[:, :4],
                           detections_class[:, 4] * detections_class[:, 5],
                           nms_thres)
                max_detections = detections_class[keep]
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            # 去掉resize的灰条
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
