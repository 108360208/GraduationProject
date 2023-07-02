from email.errors import FirstHeaderLineIsContinuationDefect
from re import L
from turtle import color
from xml.etree.ElementTree import PI
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import linear_model
import math
from sympy import * 
import sympy
from vpython import *
import time 
#相機id 1為測量速度以及仰角，2為方向角
class path:
    frame_longx=3
    frame_longy=3
    def __init__(self):
        self.vel=0
        self.angle1=0
        self.angle2=0    
    def cal_path(self,camera1,camera2):
        ret=path.cal_vel(self,camera1)
        if(not ret):
            print("Cal vel failed")
            return 0
        path.cal_angle(self,camera2)
        if (ret):
            print("Calculate is successful")
            return 1
        else :
            print("Calculate is failed")
            return 0
        #return totel_vel,angle1,angle2
    #df = pd.read_csv("C:\\Users\\Steven\\Desktop\\test_golf\\022041", sep='\t')
    #判斷id來區分為哪來相機的資料
    def separatedata(df):
        camera1=pd.DataFrame()
        camera2=pd.DataFrame()
        counter=0
        for i in df.id:
            if (i==1):
                camera1=camera1.append(df.iloc[[counter]])
            else:
                camera2=camera2.append(df.iloc[[counter]])
            counter+=1
        return camera1,camera2
    #利用線性回歸求出軌跡方程式
    def path_predcit(df):

        X=np.array(list(df.X))
        Y=np.array(list(df.Y))
        X = X.reshape(-1, 1)  # 輸入轉換為 n行 1列（多元迴歸則為多列）的二維陣列
        Y = Y.reshape(-1, 1) 
        model=make_pipeline(PolynomialFeatures(3),linear_model.LinearRegression())
        model.fit(X,Y)
        # plt.scatter(X,Y)
        # test=np.array(list(range(1,700))).reshape(-1,1)
        # plt.plot(test,model.predict(test),color='red')
        # print((model))
        # plt.show()
        return X,Y,model
    #利用id1相機求出x方向速度
    def cal_vel(self,df):
        #重設時間序列
        per_pixel_longy=float(path.frame_longy)/543
        per_pixel_longx=float(path.frame_longx)/724  #frame寬度

        for (colname,colval) in df.iteritems():
            if (colname=="time"):
                time_data=colval.values-(int(colval.values[0]))   
     
        X=np.array(list(time_data))
        Y=per_pixel_longy*np.array(list(df.Y))
           
        X = X.reshape(-1, 1)  # 輸入轉換為 n行 1列（多元迴歸則為多列）的二維陣列
        Y = Y.reshape(-1, 1)
            
        # model=make_pipeline(PolynomialFeatures(3,include_bias=True),linear_model.LinearRegression(fit_intercept=False))
        # model.fit(X,Y)
        # function_coef=model.named_steps['linearregression'].coef_[0]
        # x=sympy.Symbol('x')
        # function=function_coef[0]+function_coef[1]*x+function_coef[2]*x*x+function_coef[3]*x*x*x
        # totel_vel=diff(function).subs(x,0)
        #求出x方向速度
            
        x_distance=(df.X.iloc[-1]-df.X.iloc[0])*per_pixel_longx           
        x_time=df.time.iloc[-1]-df.time.iloc[0]      
        x_vel=x_distance/x_time
        # if (math.pow(totel_vel,2)-math.pow(x_vel,2)<0):
        #     print("When calculate vel the val is negative")
        #     print("total_vel : "+str(totel_vel))
        #     print("x_vel : "+str(x_vel))
        #     return 0
        # else:
        #     y_vel=math.sqrt(math.pow(totel_vel,2)-math.pow(x_vel,2))  #y=根號r^2-x^2
        #     angle=math.degrees(math.atan(y_vel/x_vel))
        #         #print("totel vel:"+str(totel_vel)+"m/s")
        #         # print("y's vel"+str(y_vel)+"m/s")
        #         # print("x's vel:"+str(x_vel)+"m/s")
        #         #print("Angle is:"+str(angle))
        #     self.vel=totel_vel
        #     self.angle1=angle
        X=np.array(list(df.X))
        Y=np.array(list(df.Y))
        X = X.reshape(-1, 1)  # 輸入轉換為 n行 1列（多元迴歸則為多列）的二維陣列
        Y = Y.reshape(-1, 1) 
        model=make_pipeline(PolynomialFeatures(1),linear_model.LinearRegression())
        model.fit(X,Y)
        coef=model.named_steps['linearregression'].coef_[0]
        dy=coef[1]
        angle=math.degrees(math.atan(dy))
        y_vel=x_vel*((math.tan(math.radians(angle)))/per_pixel_longx)*(per_pixel_longy)
        totel_vel=math.sqrt(math.pow(y_vel,2)+math.pow(x_vel,2))
        self.vel=totel_vel
        self.angle1=angle        
        return 1

    def cal_angle(self,df):  

        X=np.array(list(df.X))
        Y=np.array(list(df.Y))
        X = X.reshape(-1, 1)  # 輸入轉換為 n行 1列（多元迴歸則為多列）的二維陣列
        Y = Y.reshape(-1, 1) 
 
        model=make_pipeline(PolynomialFeatures(1),linear_model.LinearRegression())
        model.fit(X,Y)
        coef=model.named_steps['linearregression'].coef_[0]
        dy=coef[1]

        angle=math.degrees(math.atan(dy))
            #print("Angle is(方位):"+str(angle))
        self.angle2=angle
        return 1


    def draw(totel_vel,angle1,angle2):
        size = 1              # 小球半徑
        v0 = totel_vel              # 小球初速
        theta = radians(angle1)   # 小球抛射仰角, 用 radians 將單位轉為弧度
        L = 100              # 地板長度
        g = 9.8               # 重力加速度 9.8 m/s^2
        t = 0                 # 時間
        dt = 0.001            # 時間間隔

        scene = canvas(title="Projection", width=800, height=400, x=0, y=0,
                    center=vec(0, 5, 0), background=vec(0, 0.6, 0.6))
        floor = box(pos=vec(0, -size, 0), size=vec(L, 0.01, 10), texture=textures.metal)
        ball = sphere(pos=vec(-L/2, 0, 0), radius=size, color=color.red, make_trail=True,
                    v=vec(v0*cos(theta), v0*sin(theta), v0*cos(theta)*tan(radians(angle2))), a=vec(0, -g, 0))

        while ball.pos.y - floor.pos.y >= size:
            rate(1000)
            ball.v += ball.a*dt
            ball.pos += ball.v*dt
            t += dt
    def printpath(self):
        print(self.vel ,self.angle1 ,self.angle2)
# fig, axs = plt.subplots(2, 1) 

# test=np.array(list(range(1,700))).reshape(-1,1)
# t=time.time()
# camera1,camera2=separatedata(df)
# totel_vel,angle1=cal_vel(camera1)
# angle2=cal_angle(camera2)

#draw(totel_vel,angle1,angle2)
# print(cal_vel(camera1))
# X,Y,camera1_path=path_predcit(camera1)
# axs[0].scatter(X, Y) 
# axs[0].plot(test,camera1_path.predict(test),color='red')
# X,Y,camera2_path=path_predcit(camera2)
# axs[1].scatter(X, Y) 
# axs[1].plot(test,camera2_path.predict(test),color='red')

# plt.scatter(X,Y)
# X,Y,camera2_path=path_predcit(camera2)

# X = X.reshape(-1, 1)  # 輸入轉換為 n行 1列（多元迴歸則為多列）的二維陣列
# Y = Y.reshape(-1, 1) 
# modelRegL = LinearRegression()  # 建立線性迴歸模型
# modelRegL.fit(X, Y)  # 模型訓練：資料擬合
# yFit = modelRegL.predict(X)  # 用迴歸模型來預測輸出

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(X, Y, 'o', label="data")  # 原始資料
# ax.plot(X, yFit, 'r-', label="OLS")  # 擬合資料

# ax.legend(loc='best')  # 顯示圖例
# plt.title('Linear regression by SKlearn (Youcans)')
# plt.show()  # YouCans, XUPT
# axes = plt.axes()
# axes.set_xlim([0, 1448/2])
# axes.set_ylim([-1086/2, 0])
# plt.xlabel("X-Axis")
# plt.ylabel("Y-Axis")
# plt.scatter(df.X, -df.Y)
# plt.show()
