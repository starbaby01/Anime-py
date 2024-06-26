# Anime-py
一、先下载人脸特征点所需的模型文件：shape_predictor_68_face_landmarks.dat.bz2
地址：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2；
在windows的WSL环境，或Linux中解压命令：bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2；
得到解压后文件：shape_predictor_68_face_landmarks.dat；

二、下载漫画风格GAN模型：face_paint_512_v2_0.pt
地址：https://huggingface.co/akhaliq/AnimeGANv2-pytorch/tree/main

三、修改anime_test.py代码对应目录地址和源图片地址后，直接Pycharm运行即可。
最新可运行时间：2024/6月

依赖环境列表：Python 3.11 CPU环境即可运行；
    torch=2.3.1
    dlib=19.24.1
    其他随意。



        
