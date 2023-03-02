from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = '/mmsegmentation/work_dirs/zzj_apple_start/zzj_apple_v01.py'
checkpoint_file = '/mmsegmentation/work_dirs/zzj_apple_start/iter_13500.pth'
# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# # 对单张图片进行推理并展示结果
# img = 'img_2.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # 在新窗口中可视化推理结果
# model.show_result(img, result, show=True)
# # 或将可视化结果存储在文件中
# # 你可以修改 opacity 在 (0,1] 之间的取值来改变绘制好的分割图的透明度
# model.show_result(img, result, out_file='pics/result_2.jpg', opacity=0.5)
# 对视频进行推理并展示结果
# 将video读取为frame序列
video = mmcv.VideoReader('/mmsegmentation/tools/ZZJtools/pics/test1.mp4')
for frame in video:
    result = inference_segmentor(model, frame)
    # out_file暂时不能工作
    model.show_result(frame, result, out_file='pics/resultM.jpg', wait_time=1)

