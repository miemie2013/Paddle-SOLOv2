# SOLOv2

## 概述

请前往AIStudio体验：[实时实例分割SOLOv2来了！](https://aistudio.baidu.com/aistudio/projectdetail/1266474)

我之前自己使用PaddleDetection写过一版SOLOv2 [实时实例分割SOLOv2来了！](https://aistudio.baidu.com/aistudio/projectdetail/985880)  ，奈何无法复现论文中训练成果。几天前发现PaddleDetection已经发布了复现的SOLOv2模型。恰巧飞桨也发布了2.0版本，主推动态图。故炒此冷饭，也算是自己给SOLOv2算法的学习画个句号。

我之前写的静态图版SOLOv2对Loss处稍有改动，主要是为了解决某些输出层没有正样本的问题，因为静态图里写条件判断太难了，不像pytorch一样可以使用python原生控制流if、for等。静态图里必须要使用一些套路编程技巧才能实现pytorch里的if、for那样的效果。这次用飞桨2.0版本复现SOLOv2，我也仔细看了PaddleDetection官方实现的OLOv2的代码，贵团队真的是高手如云，我也是基于贵团队的代码搬运。我用动态图重新写了PaddleDetection中的Resnet、FPN、SOLOv2Head等部分的代码，干货满满，非常适合学习动态图的小伙伴阅读。也非常适合学习SOLOv2算法的小伙伴阅读，你可以把代码下载下来，在PyCharm中逐行调试，了解该算法的实现。咩酱也加有很多中文注释，可以帮助你理解该算法。

本仓库支持的模型：

| 模型 | 配置文件 | 骨干网络 | 图片输入大小 | box AP(val2017) | mask AP(val2017) | FPS  |
|:------------:|:--------:|:--------:|:----:|:-------:|:-------:|:---------:|
| SOLOv2_R50_3x.pth    | solov2_r50_fpn_8gpu_3x.py | ResNet50-vb | (target_size=800, max_size=1333)  | 0.404  | 0.376  | 4.6 |
| SOLOv2_LIGHT_448_R50_3x.pth    | solov2_light_448_r50_fpn_8gpu_3x.py | ResNet50-vb | (target_size=448, max_size=768)  | 0.362  | 0.337  | 8.5 |
| solov2_r50_fpn_3x.pdparams    | solov2_r50_fpn_8gpu_3x.py | ResNet50-vb | (target_size=800, max_size=1333)  | 0.408  | 0.380  | 4.6 |
| solov2_light_r50_vd_fpn_dcn_512_3x.pdparams    | solov2_light_r50_vd_fpn_dcn_512_3x.py | ResNet50-vd | (target_size=512, max_size=852)  | 0.420  | 0.388  | 7.6 |

**注意:**

- 以上4个模型均需要对应的4个1_*2paddle.py 脚本转换成本仓库可使用的模型。SOLOv2_R50_3x.pth和SOLOv2_LIGHT_448_R50_3x.pth来自原版SOLO仓库，solov2_r50_fpn_3x.pdparams和asolov2_light_r50_vd_fpn_dcn_512_3x.pdparams来自PaddleDetection。SOLOv2_R50_3x.pth与solov2_r50_fpn_3x.pdparams使用了同一个配置文件configs/solov2_r50_fpn_8gpu_3x.py。
比如你若需要转换solov2_light_r50_vd_fpn_dcn_512_3x.pdparams模型，将solov2_light_r50_vd_fpn_dcn_512_3x.pdparams放在与1_paddle_solov2_light_r50_vd_fpn_dcn_512_3x2paddle.py同级目录下，然后运行1_paddle_solov2_light_r50_vd_fpn_dcn_512_3x2paddle.py脚本，就在同级目录下生成一个dygraph_solov2_light_r50_vd_fpn_dcn_512_3x.pdparams模型文件，这是本仓库支持的模型文件。
- 测速环境为：  ubuntu18.04, i5-9400F, 8GB RAM, GTX1660Ti(6GB), cuda10.2。FPS由demo.py测得，包括预处理和后处理时间（均使用多线程优化）。预测50张图片，预测之前会有一个热身(warm up)阶段使速度稳定。若使用AIStudio上的V100测速，结果和1660Ti差不多，这对咩酱来说一直是个未解之谜。
- 据咩酱观测，飞桨2.0动态图直接预测的速度和静态图导出后的速度差不多。
- 据咩酱观测，solov2_light_r50_vd_fpn_dcn_512_3x.pdparams的预测速度约为PPYOLO_2x(608x608) （来自我的另一个精品项目[Paddle2.0动态图版PPYOLO的简单实现](https://aistudio.baidu.com/aistudio/projectdetail/1156231)）的一半，PPYOLO_2x(608x608)在ubuntu18.04、GTX1660Ti上约15.6FPS。solov2_light_r50_vd_fpn_dcn_512_3x.pdparams官方宣称的V100 FP32(FPS)=38.6，PPYOLO_2x(608x608)官方宣称的V100 FP32(FPS)=72.9。咩酱的假设成立。
- 咩酱实现了除多卡训练外的全部细节。逐一编码实现了原配置文件的功能要点。





## 快速开始
(1)安装一些依赖

```
%cd ~/work
! pip install pycocotools
! pip install shapely
```
(2)获取预训练模型
```
%cd ~/work
! wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_3x.pdparams
! python 1_paddle_solov2_r50_fpn_8gpu_3x2paddle.py
! rm -f solov2_r50_fpn_3x.pdparams
! wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
! python 1_paddle_solov2_light_r50_vd_fpn_dcn_512_3x2paddle.py
! rm -f solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
```

(3)使用模型预测图片、获取FPS（预测images/test/里的图片，结果保存在images/res/）

```
! cd ~/work; python demo.py --config=2
```

进入images/res/即可看到预测后的图片。
--config=2表示的是使用配置文件solov2_light_r50_vd_fpn_dcn_512_3x.py；--config=1表示的是使用配置文件solov2_light_448_r50_fpn_8gpu_3x.py；--config=0表示的是使用配置文件solov2_r50_fpn_8gpu_3x.py；读者可以按需调整。train.py、eval.py、test_dev.py中也同理，读者可以按需调整。


## 训练

以复现solov2_light_r50_vd_fpn_dcn_512_3x.pdparams在COCO上的精度为例。
如果你需要训练COCO2017数据集，那么需要先解压数据集

```
! pip install pycocotools
! cd ~/data/data7122/; unzip ann*.zip
! cd ~/data/data7122/; unzip val*.zip
! cd ~/data/data7122/; unzip tes*.zip
! cd ~/data/data7122/; unzip image_info*.zip
! cd ~/data/data7122/; unzip train*.zip
```

获取预训练的resnet50vd_ssld

```
%cd ~/work
! wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
! tar -xf ResNet50_vd_ssld_pretrained.tar
! python 1_paddle_r50vd_ssld_2paddle.py
! rm -f ResNet50_vd_ssld_pretrained.tar
! rm -rf ResNet50_vd_ssld_pretrained
```

再输入以下命令训练（所有的配置都在config/solov2_light_r50_vd_fpn_dcn_512_3x.py里，请查看代码注释做相应修改。如果你抢到32GB的V100，可以开batch_size=12，否则请调小batch_size。使用的预训练模型是config/solov2_light_r50_vd_fpn_dcn_512_3x.py里self.train_cfg -> model_path指定的模型）

```
! cd ~/work; python train.py --config=2
```

咩酱亲测，仅训练30000步时mask AP就可以到0.30左右了(批大小12)，至于能否到达0.388还需要看接下来几天的训练。


## 训练其它数据集
支持coco注解风格的数据集，比如DeepFashion2。
在config/solov2_light_r50_vd_fpn_dcn_512_3x.py里修改train_path、val_path、classes_path（类别名文件，需要自己写一下，一行对应一个类别名）、train_pre_path、val_pre_path、num_classes这6个变量使其指向该数据集，以及修改test_path、test_pre_path（即该数据集的test集相关的配置，没有test集的话可以不填）,就可以开始训练自己的数据集了。


## 评估
运行以下命令。评测的模型是config/solov2_light_r50_vd_fpn_dcn_512_3x.py里self.eval_cfg -> model_path指定的模型

```
! cd ~/work; python eval.py --config=2
```

该mAP是val集的结果。

## 预测
运行以下命令。使用的模型是config/solov2_light_r50_vd_fpn_dcn_512_3x.py里self.test_cfg -> model_path指定的模型

```
! cd ~/work; python demo.py --config=2
```

喜欢的话点个喜欢或者关注我哦~
为了更方便地查看代码、克隆仓库、跟踪更新情况，该仓库不久之后也会登陆我的GitHub账户，对源码感兴趣的朋友可以提前关注我的GitHub账户鸭（求粉）~

AIStudio: [asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

GitHub: [miemie2013](https://github.com/miemie2013)


## 传送门

咩酱重写过很多算法，比如PPYOLO、SOLOv2、FCOS、YOLOv4等，而且在多个深度学习框架（tensorflow、pytorch、paddlepaddle等）上都实现了一遍，你可以进我的GitHub主页看看，看到喜欢的仓库可以点个star呀！

cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或AIStudio上关注我（求粉）~

## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓
![Example 0](weixin/sk.png)

咩酱爱你哟！另外，有偿接私活，可联系微信wer186259，金主快点来吧！
