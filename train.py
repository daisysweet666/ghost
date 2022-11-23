#  对数据集进行训练
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

'''训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，按默认参数训练完会有100个权值
损失值的大小用于判断是否收敛，训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中'''
if __name__ == "__main__":
    Cuda = True
    classes_path = '/content/gdrive/MyDrive/Model/ghost/model_data/classes.txt'
    anchors_path = '/content/gdrive/MyDrive/Model/ghost/model_data/9anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    # 同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    model_path = ''
    input_shape = [416, 416]
    pretrained = False

    """冻结阶段训练参数"""
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    """解冻阶段训练参数"""
    UnFreeze_Epoch = 400
    Unfreeze_batch_size = 16
    Freeze_Train = False

    Init_lr = 1e-2           # 模型的最大学习率
    Min_lr = Init_lr * 0.01  # 模型的最小学习率，默认为最大学习率的0.01
    optimizer_type = "sgd"   # 使用到的优化器种类，可选的有adam(Init_lr=1e-3)、sgd(Init_lr=1e-2)
    momentum = 0.937         # 优化器内部使用到的momentum参数
    weight_decay = 5e-4      # 权值衰减，可防止过拟合
    lr_decay_type = "cos"    # 使用到的学习率下降方式，可选的有step、cos
    save_period = 50          # 多少个epoch保存一次权值，默认每个世代都保存
    num_workers = 2          # 设置是否使用多线程读取数据

    train_annotation_path = '/content/gdrive/MyDrive/Model/ghost/model_data/my_train.txt'
    val_annotation_path = '/content/gdrive/MyDrive/Model/ghost/model_data/my_val.txt'                 # 获得图片路径和标签
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)   # 获取classes和anchor

    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained)  # 创建yolo模型
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/", model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #   读取数据集对应的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 主干特征提取网络特征通用，冻结训练可以加快训练速度
    if True:
        UnFreeze_flag = False
        #   冻结一定部分训练
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        #   判断当前batch_size与64的差别，自适应调整学习率
        nbs = 64
        Init_lr = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr = max(batch_size / nbs * Min_lr, 1e-6)
        #   根据optimizer_type选择优化器
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)  # 获得学习率下降的公式
        #   判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        #   构建数据集加载器
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        #   开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period)

        loss_history.writer.close()
