import numpy as np
import cv2
class Canny:

    def __init__(self, img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100):
        """
        img:输入图像
        sigma:高斯核标准差
        kernel_size:高斯核大小
        lowthreshold:低阈值
        highthreshold:高阈值
        weak_pixel:弱边缘像素值
        """
        self.img = img
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowthreshold = lowthreshold
        self.highthreshold = highthreshold
        self.weak_pixel = weak_pixel
        self.angle = np.zeros_like(img)                              #梯度方向矩阵

    def gaussian_kernel(self, size, sigma): 
        """
        生成高斯核
        size: 高斯核大小
        sigma: 高斯核标准差
        """
        size = int(size) // 2                                       #取整
        x, y = np.mgrid[-size:size+1, -size:size+1]                 #生成网格
        normal = 1 / (2.0 * np.pi * sigma**2)                       #正态分布
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal     #高斯函数
        return g
    
    def gaussian_filter(self, img, kernel_size, sigma):
        """
        高斯滤波
        img: 输入图像
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
        """
        kernel = self.gaussian_kernel(kernel_size, sigma)         #生成高斯核
        return cv2.filter2D(img, -1, kernel)                      #卷积
    
    def sobel_filter(self, img, kernel_size):
        """
        sobel滤波
        img: 输入图像
        kernel_size: sobel核大小
        """
        if len(img.shape) == 3:                                           #如果是彩色图像
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                   #转换为灰度图像
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)      #x方向梯度  
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)      #y方向梯度
        sobel = np.hypot(sobelx, sobely)                                  #梯度幅值
        sobel = sobel / sobel.max() * 255                                 #归一化
        return sobel.astype(np.uint8)                                     #返回梯度幅值
    

    def non_max_suppression(self, img, D):
        """
        非极大值抑制
        img: input image
        D:   gradient directions
        """
        M, N = img.shape #
        Z = np.zeros((M,N), dtype=np.int32)             #M行N列的零矩阵
        angle = D * 180. / np.pi                        #角度
        angle[angle < 0] += 180                         #角度小于0的加180
        for i in range(1,M-1):                          #遍历每个像素
            for j in range(1,N-1):
                try:                                    #防止越界
                    q = 255                             #q为梯度幅值
                    r = 255                             #r为梯度幅值
                    # angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):    
                        q = img[i, j+1]                 #右边像素
                        r = img[i, j-1]                 #左边像素
                    # angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]               #右上像素
                        r = img[i-1, j+1]               #左下像素
                    # angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]                 #下边像素
                        r = img[i-1, j]                 #上边像素
                    # angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]               #左上像素
                        r = img[i+1, j+1]               #右下像素
                    if (img[i,j] >= q) and (img[i,j] >= r):     #如果当前像素大于左右像素
                        Z[i,j] = img[i,j]               #保留当前像素
                    else:
                        Z[i,j] = 0                      #否则置零
                except IndexError as e:                 #防止越界
                    pass
        return Z                                        #返回非极大值抑制后的图像


    def threshold(self, img, weak_pixel, lowthreshold, highthreshold):
        """
        双阈值
        img: 输入图像
        weak_pixel: 弱像素值
        lowthreshold: 低阈值
        highthreshold: 高阈值
        """
        M, N = img.shape                                #图像大小
        res = np.zeros((M,N), dtype=np.uint8)           #M行N列的零矩阵
        weak = np.int32(weak_pixel)                     #弱像素值
        strong = np.int32(255)                          #强像素值
        low = img.max() * lowthreshold                  #低阈值
        high = img.max() * highthreshold                #高阈值
        strong_i, strong_j = np.where(img >= high)      #高于高阈值的像素
        zeros_i, zeros_j = np.where(img < low)          #低于低阈值的像素
        weak_i, weak_j = np.where((img <= high) & (img >= low))     #介于两者之间的像素
        res[strong_i, strong_j] = strong                #强像素
        res[weak_i, weak_j] = weak                      #弱像素
        return res, weak, strong                        #返回二值图像，弱像素值，强像素值
    
    def hysteresis(self, img, weak_pixel, strong_pixel):
        """
        连接弱边缘
        img: 输入图像
        weak_pixel: 弱像素值
        strong_pixel: 强像素值
        """
        M, N = img.shape                                        #图像大小   
        for i in range(1,M-1):                                  #遍历每个像素
            for j in range(1,N-1):
                if (img[i,j] == weak_pixel):                    #如果是弱像素
                    try:
                        if ((img[i+1, j-1] == strong_pixel)     
                            or (img[i+1, j] == strong_pixel) 
                            or (img[i+1, j+1] == strong_pixel)
                            or (img[i, j-1] == strong_pixel) 
                            or (img[i, j+1] == strong_pixel)
                            or (img[i-1, j-1] == strong_pixel) 
                            or (img[i-1, j] == strong_pixel) 
                            or (img[i-1, j+1] == strong_pixel)):#如果周围有强像素
                            img[i,j] = strong_pixel             #则将该像素设为强像素
                        else:
                            img[i,j] = 0                        #否则设为0
                    except IndexError as e:                     #防止越界
                        pass
        return img

    def canny(self):
        self.img = self.gaussian_filter(self.img, 
                                   self.kernel_size, self.sigma) # 高斯滤波
        self.img = self.sobel_filter(self.img, self.kernel_size)           # sobel滤波
        
        theta = np.arctan2(cv2.Sobel(self.img, cv2.CV_64F, 0, 1, 
                                     ksize=self.kernel_size),
                           cv2.Sobel(self.img, cv2.CV_64F, 1, 0, 
                                     ksize=self.kernel_size))    # 计算梯度方向                                # 保存梯度方向
        # print("theta:",theta)
        self.img = self.non_max_suppression(self.img, theta)               # 非极大值抑制
        self.img, weak, strong = self.threshold(self.img, self.weak_pixel, self.lowthreshold, self.highthreshold) # 双阈值
        self.img = self.hysteresis(self.img, weak, strong)                 # 连接边缘
        return self.img
    

