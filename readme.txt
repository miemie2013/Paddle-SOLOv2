


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*


pip install --user paddlepaddle-gpu==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple



# 解压预训练模型
nvidia-smi
cd ~/w*
cp ../data/data61361/dygraph_solov2_r50_fpn_8gpu_3x.pdparams ./dygraph_solov2_r50_fpn_8gpu_3x.pdparams
cp ../data/data61361/dygraph_solov2_light_448_r50_fpn_8gpu_3x.pdparams ./dygraph_solov2_light_448_r50_fpn_8gpu_3x.pdparams



下载预训练模型solov2_r50_fpn_3x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_3x.pdparams
python 1_paddle_solov2_r50_fpn_8gpu_3x2paddle.py
rm -f solov2_r50_fpn_3x.pdparams



下载预训练模型solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
python 1_paddle_solov2_light_r50_vd_fpn_dcn_512_3x2paddle.py
rm -f solov2_light_r50_vd_fpn_dcn_512_3x.pdparams



下载预训练模型ResNet50_vd_ssld_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
tar -xf ResNet50_vd_ssld_pretrained.tar
python 1_paddle_r50vd_ssld_2paddle.py
rm -f ResNet50_vd_ssld_pretrained.tar
rm -rf ResNet50_vd_ssld_pretrained






# 安装依赖、解压COCO2017数据集
nvidia-smi
cd ~
pip install pycocotools
pip install shapely
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*



# 安装依赖、解压voc数据集
nvidia-smi
cd ~
pip install pycocotools
cd data
cd data4379
unzip pascalvoc.zip
cd ~/w*







-------------------------------- SOLOv2 --------------------------------
parser.add_argument('-c', '--config', type=int, default=2,
                    choices=[0, 1, 2],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- solov2_r50_fpn_8gpu_3x.py
                    1 -- solov2_light_448_r50_fpn_8gpu_3x.py
                    2 -- solov2_light_r50_vd_fpn_dcn_512_3x.py'''))

训练
cd ~/w*
python train.py --config=0

cd ~/w*
python train.py --config=1

cd ~/w*
python train.py --config=2

cd ~/w*
python train.py --config=3




预测
cd ~/w*
python demo.py --config=0

cd ~/w*
python demo.py --config=1

cd ~/w*
python demo.py --config=2

cd ~/w*
python demo.py --config=3



预测并打包图片
cd ~/w*
python demo.py --config=2
rm -f out.zip
zip -r out.zip images/res/*.jpg




验证
cd ~/w*
python eval.py --config=0

cd ~/w*
python eval.py --config=1

cd ~/w*
python eval.py --config=2

cd ~/w*
python eval.py --config=3




跑test_dev
cd ~/w*
python test_dev.py --config=0

cd ~/w*
python test_dev.py --config=1

cd ~/w*
python test_dev.py --config=2

cd ~/w*
python test_dev.py --config=3












2020-11-22 14:59:23,395-INFO: Test iter 4900
2020-11-22 14:59:34,284-INFO: Test Done.
2020-11-22 14:59:34,284-INFO: total time: 959.001371s
2020-11-22 14:59:34,284-INFO: Speed: 0.193659s per image,  5.2 FPS.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719




2020-11-22 15:32:24,914-INFO: Test iter 4900
2020-11-22 15:32:34,286-INFO: Test Done.
2020-11-22 15:32:34,286-INFO: total time: 805.920212s
2020-11-22 15:32:34,286-INFO: Speed: 0.162746s per image,  6.1 FPS.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.541
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.765

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707














========================================= Paddle的模型 ======================================================
R50_3x

2020-11-24 09:18:08,345-INFO: Test iter 4900
2020-11-24 09:18:16,802-INFO: Test Done.
2020-11-24 09:18:16,802-INFO: total time: 782.208325s
2020-11-24 09:18:16,802-INFO: Speed: 0.157958s per image,  6.3 FPS.
loading annotations into memory...
Done (t=0.92s)
creating index...
index created!
2020-11-24 09:18:17,777-INFO: Generating json file...
2020-11-24 09:18:22,612-INFO: Start evaluate...
Loading and preparing results...
DONE (t=2.78s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=41.69s).
Accumulating evaluation results...
DONE (t=5.84s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.210
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
loading annotations into memory...
Done (t=0.68s)
creating index...
index created!
2020-11-24 09:19:14,770-INFO: Generating json file...
2020-11-24 09:19:22,213-INFO: Start evaluate...
Loading and preparing results...
DONE (t=7.53s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=51.51s).
Accumulating evaluation results...
DONE (t=5.82s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.590
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707




solov2_light_r50_vd_fpn_dcn_512_3x

2020-11-24 09:01:29,885-INFO: Test iter 4900
2020-11-24 09:01:38,332-INFO: Test Done.
2020-11-24 09:01:38,332-INFO: total time: 779.739039s
2020-11-24 09:01:38,332-INFO: Speed: 0.157459s per image,  6.4 FPS.
loading annotations into memory...
Done (t=0.67s)
creating index...
index created!
2020-11-24 09:01:39,057-INFO: Generating json file...
2020-11-24 09:01:44,299-INFO: Start evaluate...
Loading and preparing results...
DONE (t=2.96s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=51.85s).
Accumulating evaluation results...
DONE (t=5.89s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.611
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.632
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.796
loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
2020-11-24 09:02:46,594-INFO: Generating json file...
2020-11-24 09:02:52,113-INFO: Start evaluate...
Loading and preparing results...
DONE (t=5.86s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=51.75s).
Accumulating evaluation results...
DONE (t=6.04s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.603
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.742
















./weights/21000.pdparams


2020-11-24 22:58:24,235-INFO: Test iter 4900
2020-11-24 22:58:33,509-INFO: Test Done.
2020-11-24 22:58:33,510-INFO: total time: 875.109538s
2020-11-24 22:58:33,510-INFO: Speed: 0.176718s per image,  5.7 FPS.
loading annotations into memory...
Done (t=0.78s)
creating index...
index created!
2020-11-24 22:58:34,319-INFO: Generating json file...
2020-11-24 22:58:55,579-INFO: Start evaluate...
Loading and preparing results...
DONE (t=1.92s)
creating index...
index created!
<string>:6: DeprecationWarning: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=40.30s).
Accumulating evaluation results...
DONE (t=6.38s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
loading annotations into memory...
Done (t=0.56s)
creating index...
index created!
2020-11-24 22:59:45,627-INFO: Generating json file...
2020-11-24 23:01:14,311-INFO: Start evaluate...
Loading and preparing results...
DONE (t=6.72s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=49.04s).
Accumulating evaluation results...
DONE (t=6.33s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641







