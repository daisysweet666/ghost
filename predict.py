# predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能整合到了一个py文件中，通过指定mode进行模式的修改。
"""
1、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
2、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
在原图上利用矩阵的方式进行截取。
3、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
比如判断if
predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
"""
import time
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    mode = "dir_predict"
    crop = False                # 是否在单张图片预测后对目标进行截取
    video_path = 0              # 指定视频的路径，此时检测摄像头
    video_save_path = ""        # 视频保存的路径，此时表示不保存
    video_fps = 25.0            # 保存的视频的fps
    test_interval = 10          # 指定测量fps的时候，图片检测的次数
    dir_origin_path = "F:/05-pycharm/test/"    # 用于检测的图片的文件夹路径
    dir_save_path = "F:/05-pycharm/img1/"  # 检测完图片的保存路径

    if mode == "predict":
        """
        预测单张照片模式
        """
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop)
                r_image.show()

    elif mode == "video":
        """
        video模式
        """
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()  # 读取某一帧
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
            frame = Image.fromarray(np.uint8(frame))        # 转变成Image
            frame = np.array(yolo.detect_image(frame))      # 进行检测
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBtoBGR满足opencv显示格式

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('F:/05-pycharm/test/2.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
