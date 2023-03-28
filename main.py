'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-03-25 20:37:44
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-03-28 09:14:21
FilePath: \cv_1\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import math
import numpy as np
import canny_mine as cm
import hough_mine as hm
import hough_circle_mine as hcm
#设置参数
Path = "pic.jpg"                                            #图像路径
Save_Path = "result/"                                       #处理后图像保存路径
Reduced_ratio = 2                                           #缩小图像的比例

"""
第一步：读取图像并缩小图像
"""
img_gray = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)           #读取图像
# print("img_gray:\n", img_gray)                  
img_RGB = cv2.imread(Path)                                      
# print("img_RGB:\n", img_RGB)
y, x = img_gray.shape[0:2]                                  #获取图像大小
img_gray = cv2.resize(img_gray,                             #缩小图像
                      (int(x / Reduced_ratio),              
                       int(y / Reduced_ratio)))     
img_RGB = cv2.resize(img_RGB,                               #缩小图像
                     (int(x / Reduced_ratio), 
                      int(y / Reduced_ratio)))
img_RGB_cv2 = img_RGB.copy()
"""
第二步：检测边缘
"""
canny = cm.Canny(img_gray)                                      #自写的canny实例化
edges = canny.canny()                                           #检测边缘
cv2.imwrite(Save_Path + "canny_result_mine.jpg", edges)         #生成检测边缘后的图像
edges_cv2 = cv2.Canny(img_gray, 120, 180)  
"""
第三步：检测圆
"""
Hough = hm.Hough_transform(edges_cv2, canny.angle, step=5, threshold=25)          #自写的hough实例化
circles = Hough.Calculate()           

# circles=cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 
#                          y/8, param1= 100,param2= 40,
#                          minRadius= 50, maxRadius= 150)                                 #检测圆 
print("circles_mine:")


"""
第四步：画圆，保存图像
"""         
# 画自写的hough检测圆
if len(circles):                                         #如果检测到圆        
    circles = np.uint16(np.around(circles))                     #四舍五入取整
    for circle in circles[0, :]:                                #遍历每个圆 
        print("圆心坐标：", circle[0], circle[1], "半径：", circle[2]) # 打印圆心坐标和半径
        cv2.circle(img_RGB, (circle[0], circle[1]),             #画圆
                   circle[2], (132, 135, 239), 2)
    cv2.imwrite(Save_Path + "hough_result_mine.jpg", img_RGB)       #保存检测圆后的图像


"""
第五步：与cv2自带的canny检测边缘和hough检测圆对比
"""
use_cv2 = True
if use_cv2:
    # cv2自带的canny检测边缘
    edges_cv2 = cv2.Canny(img_gray, 120, 180)                   #检测边缘
    cv2.imwrite(Save_Path + "canny_result_cv2.jpg", edges_cv2)  #生成检测边缘后的图像

    # cv2自带的hough检测圆
    circles_cv2=cv2.HoughCircles(edges_cv2, cv2.HOUGH_GRADIENT, 1, 
                         y/8, param1= 100,param2= 60,
                         minRadius= 10, maxRadius= 150)         #检测圆
    
    # 画cv2自带的hough检测圆
    if circles_cv2 is not None:                                 #如果检测到圆
        circles_cv2 = np.uint16(np.around(circles_cv2))         #四舍五入取整
        print("circles_cv2:")
        for circle in circles_cv2[0, :]:                        #遍历每个圆 
            print("圆心坐标：", circle[0], circle[1], "半径：", circle[2])
            cv2.circle(img_RGB_cv2, (circle[0], circle[1]),     #画圆
                       circle[2], (132, 135, 239), 2)       
            
    cv2.imwrite(Save_Path + "hough_result_cv2.jpg", img_RGB_cv2)   
    

"""
已完成
"""
print('已完成！')
