# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo_salience_swin_finalMonoatt_noSeghead_noKAN import YoloBody_Salience_Swin
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory
from utils.dataloader_salience import YoloDataset_Salience, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, get_lr
from utils.utils_fit_salience_autol_metabalance import fit_one_epoch_salience
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn.functional as F
from auto_lambda_for_me import AutoLambda

class Salience_Loss(nn.Module):
    def __init__(self):
        super(Salience_Loss, self).__init__()

    def forward(self, x, label):
        dice_loss = DiceLoss(mode='binary', from_logits=False, smooth=1e-5)
        loss = F.binary_cross_entropy(x, label) + dice_loss(x, label)
        # F.binary_cross_entropy_with_logits()+(1-pytorch_ssim.ssim(label, x))
        # loss = F.binary_cross_entropy(x, label)
        return loss


'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''
if __name__ == "__main__":

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True

    # 选择梯度处理方法：'graddrop'、'pcgrad'、'cagrad'
    grad_method = 'graddrop'

    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = 'model_data/small.txt'
    # ---------------------------------------------------------------------#
    #   anchors_path    代表先验框对应的txt文件，一般不修改。
    #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # model_path = 'logs/yolov5s_416_salience_swid/ep100-loss0.039-val_loss0.032.pth'
    model_path = ''

    # ------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是16的倍数
    # ------------------------------------------------------#
    # input_shape     = [608, 608]

    ######################################
    input_shape = [512, 512]
    ######################################

    # ------------------------------------------------------#
    #   phi             所使用的YoloV5的版本。s、m、l、x
    # ------------------------------------------------------#
    phi = 'x'
    # ------------------------------------------------------------------#
    #   mosaic          马赛克数据增强
    #                   参考YoloX，由于Mosaic生成的训练图片，
    #                   远远脱离自然图片的真实分布。
    #                   本代码会在训练结束前的N个epoch自动关掉Mosaic
    #                   100个世代会关闭30个世代（比例可在dataloader.py调整）
    #   label_smoothing 标签平滑。一般0.01以下。如0.01、0.005
    # ------------------------------------------------------------------#
    mosaic = False

    label_smoothing = 0

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练：
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True（默认参数）
    #       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#

    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01

    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 1e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs/demo'
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 4

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    train_annotation_path = 'dataset/IRSTD-1k/trainval.txt'
    # val_annotation_path = "dataset/IHAST/val.txt"

    # ------------------------------------------------------#
    #   获取classes和anchor
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = YoloBody_Salience_Swin(anchors_mask, num_classes)
    weights_init(model)
    # if model_path != '':
    #     # ------------------------------------------------------#
    #     #   权值文件请看README，百度网盘下载
    #     # ------------------------------------------------------#
    #     print('Load weights {}.'.format(model_path))
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(model_path, map_location=device)
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        filted_dict = {}
        filted_dict.update(
            {key: value for key, value in torch.load(model_path, map_location=device).items() if
             key in model_dict.keys() and model_dict[key].size() == value.size()})
        model_dict.update(filted_dict)
        model.load_state_dict(model_dict)
        print('该模型共有{}层，其中有{}层是公共的。'.format(len(model_dict.keys()), len(filted_dict.keys())))
        print('完成预训练权重的更新！')

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    salience_loss = Salience_Loss()

    if Cuda:
        # model_train = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
        model = model.cuda()
    else:
        model = model.cpu()

    # auto-λ参数，autol_init第一个值是目标检测损失权重，第二个是显著性损失权重
    # 这是根据代码逻辑确定的，如果要修改顺序，可能也要修改对应处的代码！！！
    autol_init = [0.1, 0.1]
    autol_lr = 1e-3
    # auto-Balance初始化
    autol = AutoLambda(model, torch.device('cuda'), task_num=2, weight_init=autol_init)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=autol_lr)

    # apply gradient methods
    grad_dims = []
    rng = np.random.default_rng()
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(autol_init)).to('cuda')


    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#

    # ------------------------这里将下面两个with open中的encoding的utf-8改成了gbk--------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    # with open(val_annotation_path, encoding='utf-8') as f:
    #     val_lines = f.readlines()
    # np.random.seed(10321)
    # np.random.shuffle(train_lines)
    # np.random.shuffle(val_lines)
    num_train = len(train_lines)
    # num_val = len(val_lines)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(Unfreeze_batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(Unfreeze_batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                                 weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        # epoch_step_val = num_val // batch_size

        # if epoch_step == 0 or epoch_step_val == 0:
        #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        train_dataset = YoloDataset_Salience(train_lines, input_shape, epoch_length=UnFreeze_Epoch, mosaic=False,
                                             train=True)
        # val_dataset = YoloDataset_Salience(val_lines, input_shape, epoch_length=UnFreeze_Epoch, mosaic=False,
        #                                    train=False)
        gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=False, collate_fn=yolo_dataset_collate)
        # gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
        #                      pin_memory=True,
        #                      drop_last=True, collate_fn=yolo_dataset_collate)

        meta_dataset = YoloDataset_Salience(train_lines, input_shape, epoch_length=UnFreeze_Epoch, mosaic=False,
                                            train=True)
        meta_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=False, collate_fn=yolo_dataset_collate)

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        best_map = 0
        best_iou = 0

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size与64的差别，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                # epoch_step_val = num_val // batch_size

                # if epoch_step == 0 or epoch_step_val == 0:
                #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=False, collate_fn=yolo_dataset_collate)
                # gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                #                      pin_memory=True,
                #                      drop_last=True, collate_fn=yolo_dataset_collate)
                meta_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=False, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            # gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # every_epoch_map,every_epoch_iou, every_epoch_niou = fit_one_epoch_salience(model, yolo_loss, salience_loss, loss_history,
            #                                          optimizer, epoch, epoch_step, epoch_step_val,
            #                                          gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir,
            #                                          best_map,best_iou,best_niou)

            every_epoch_map, every_epoch_iou = fit_one_epoch_salience(model, yolo_loss, salience_loss,
                                                                       optimizer, epoch, epoch_step,
                                                                       gen, UnFreeze_Epoch, Cuda, save_dir, best_map,
                                                                       best_iou, autol, meta_optimizer, meta_loader,
                                                                       get_lr(optimizer),grads,grad_dims,rng,grad_method)
            best_map = every_epoch_map
            best_iou = every_epoch_iou
            # best_niou = every_epoch_niou