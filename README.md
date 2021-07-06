# 快速使用
**main.py为程序的主窗口，通过运行mian.py进行窗口启动\n**
**所有结果文件保存在test_result -> result 中\n**
**可以通过 pip install -r requirements.txt 快速搭建运行环境\n** 
\n
\n
##train_crnn
*该文件夹下为CRNN训练、模型及识别的包*
\n
## train_ctpn
*该文件下为CTPN训练、模型及预测的包*
\n
##window
*该包下为窗口界面所需要的py*

##使用流程
1.通过点击 main.py运行程序\n
2.点击打开文件后默认打开test_images文件夹，可自己选择
3.选择文件后选择切割方式，cut_e_invoice.py 和 cut_p_invoice.py 分别为切割电子发票和切割发票照片
4.在右下角状态栏显示切割完成后，点击识别发票，会调用 recognition.py 中的识别方法
5.结果显示是通过读取 test_result -> result 下的文件进行读取


