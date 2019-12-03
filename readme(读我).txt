1、运行该程序前先要确保计算机中已经安装pytorch1.1开发环境，程序运行环境为linux操作系统（推荐使用Ubuntu16.04）
pytorch1.1的安装方法可参考https://pytorch.org/get-started/previous-versions/（conda安装方式conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch）
2、在命令行运行eval_txt.py脚本文件
该脚本接受四个路径，1、图片名称txt文件 2、标注文件夹路径 3、图片文件夹路径 4、本次评估模型路径 
示例：python eval_txt.py -N battery_sub_test_txt -A annotation -I image -M bestmodel.pt
为了方便运行，我将数据集按照要求排列在了文件夹中，搭好环境后直接运行上述命令即可，在5500张图像上的mAp表现目前为0.98，预计该模型的mAp浮动范围为0.87~0.93

