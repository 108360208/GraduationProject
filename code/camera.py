import ctypes
from socket import timeout
from pyueye import ueye
import numpy as np
import cv2
import pandas as pd 
import time
from asyncio.windows_events import NULL
from ctypes import c_double, c_uint, c_void_p
#backsub=cv2.createBackgroundSubtractorMOG2(history=2000,varThreshold=10,detectShadows=True)
class trackbar:
    def __init__(self,camara_id) :
        self.id=camara_id
        cv2.namedWindow('camera'+str(camara_id))
        cv2.createTrackbar('HMin','camera'+str(camara_id),0,179,self.nothing) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','camera'+str(camara_id),44,255,self.nothing)
        cv2.createTrackbar('VMin','camera'+str(camara_id),210,255,self.nothing)
        cv2.createTrackbar('HMax','camera'+str(camara_id),33,179,self.nothing)
        cv2.createTrackbar('SMax','camera'+str(camara_id),139,255,self.nothing)
        cv2.createTrackbar('VMax','camera'+str(camara_id),255,255,self.nothing)
        self.hMin = cv2.setTrackbarPos('HMin','camera'+str(camara_id),0)
        self.sMin = cv2.setTrackbarPos('SMin','camera'+str(camara_id),0)
        self.vMin = cv2.setTrackbarPos('VMin','camera'+str(camara_id),150)
        self.hMax = cv2.setTrackbarPos('HMax','camera'+str(camara_id),33)
        self.sMax = cv2.setTrackbarPos('SMax','camera'+str(camara_id),139)
        self.vMax = cv2.setTrackbarPos('VMax','camera'+str(camara_id),255)
    def nothing(self,x):
        pass
    def update(self):
        self.hMin = cv2.getTrackbarPos('HMin','camera'+str(self.id))
        self.sMin = cv2.getTrackbarPos('SMin','camera'+str(self.id))
        self.vMin = cv2.getTrackbarPos('VMin','camera'+str(self.id))
        self.hMax = cv2.getTrackbarPos('HMax','camera'+str(self.id))
        self.sMax = cv2.getTrackbarPos('SMax','camera'+str(self.id))
        self.vMax = cv2.getTrackbarPos('VMax','camera'+str(self.id))
        return np.array([self.hMin, self.sMin, self.vMin]),np.array([self.hMax, self.sMax, self.vMax])  

       
class position:
    def __init__(self):
        self.golf_X= []
        self.golf_Y=[]
        self.golf_time=[]   
        self.id=[]
    def add(self,x,y,time,camera_id):
        self.id.append(camera_id)
        self.golf_X.append(x)
        self.golf_Y.append(y)
        self.golf_time.append(time)
    def convertdata(self):
        df = pd.DataFrame({'id':self.id,'X':self.golf_X,'Y':self.golf_Y,'time':self.golf_time})
        return df
class camera:
    nBitsPerPixel = ueye.INT(24)  
    channels = 3                  #3: channels for color mode(RGB); take 1 channel for monochrome	# Y8/RGB16/RGB24/REG32
    bytes_per_pixel = int(nBitsPerPixel / 8)
    
    def __init__(self,camera_id) :
        self.hCam = ueye.HIDS(camera_id)  
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.m_nColorMode = ueye.INT()
        self.width =0
        self.heigth=0
        self.position=position()
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=60,detectShadows=False)     

    def OK(self,color_lower,color_upper,min_dist,param1,param2,minRadius,maxRadius):
        #framerate=(c_double())


        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")
        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")
        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, camera.nBitsPerPixel, self.m_nColorMode)
            camera.bytes_per_pixel = int(camera.nBitsPerPixel / 8)
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")
        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t",self.width)
        print("Maximum image height:\t", self.height)
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, camera.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)
            nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
            if nRet != ueye.IS_SUCCESS:
                print("is_CaptureVideo ERROR")
            # Enables the queue mode for existing image memory sequences
            nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height, self.nBitsPerPixel, self.pitch)
            if nRet != ueye.IS_SUCCESS:
                print("is_InquireImageMem ERROR")
            else:
                print("Press q to leave the programm")
            nRet = ueye.is_ParameterSet(self.hCam, ueye.IS_PARAMETERSET_CMD_LOAD_FILE, c_void_p(NULL), c_uint(NULL))
            if nRet != ueye.IS_SUCCESS:
                print("is_ParameterSet ERROR")
           
            while(nRet == ueye.IS_SUCCESS):

                
                lower,upper=color_lower,color_upper 
                #lower,upper=self.trackbar.update() 

                array = ueye.get_data(self.pcImageMemory, self.width, self.height, camera.nBitsPerPixel, self.pitch, copy=False)
                frame = cv2.UMat(np.reshape(array,(self.height.value,self.width.value, camera.bytes_per_pixel)))
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                copy=frame
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                # ...resize the image by a half
                frame=self.equalize_clahe_color_hsv(frame)
                fgmask=cv2.medianBlur(frame,3)
                fgmask=cv2.medianBlur(frame,3)
                # fgmask= cv2.GaussianBlur(frame , (11,11), 0)
                fgmask = self.backSub.apply(fgmask)
                mask = cv2.erode(fgmask, None, iterations=2)
                mask = cv2.dilate(fgmask, None, iterations=2)
                # #ret,threshold = cv2.threshold(mask,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                bitwise=cv2.bitwise_and(frame,frame,mask=mask)
                # # #bitwise=modify_contrast_and_brightness2(bitwise)
                # # #mask= equalize_clahe_color_hsv(bitwise)
                bitwise=cv2.inRange( bitwise,lower,upper)
                circle1 = cv2.HoughCircles(bitwise, cv2.HOUGH_GRADIENT, 2, min_dist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius) 
              
    
                if (circle1.get() is not None):
                    circles = circle1.get()[0, :, :]  # 提取為二維
                    circles = np.uint16(np.around(circles))  # 四捨五入，取整
                    for i in circles[:]:
                        #cv2.circle(copy, (i[0], i[1]), i[2], (255, 0, 0), 5) 
                        cv2.rectangle(copy,(int(i[0])-50,int(i[1])+50),(int(i[0])+50,int(i[1])-50),  (0, 255, 0), 2) # 畫圓
                        cv2.circle(copy, (i[0], i[1]), 2, (255, 0, 0), 10)  # 畫圓心
                        self.position.add(int(i[0]),int(i[1]),time.time(),int(self.hCam))

                cv2.imshow("camera"+str(self.hCam),copy)
                #cv2.imshow("a"+str(self.hCam),mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
              
               
            
    def equalizeHist(self,frame):

        v = cv2.equalizeHist(frame)

        return v
    def imshow_components(labels):
        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeled_img[label_hue==0] = 0
  

        return labeled_img
    def equalize_clahe_color_hsv(self,img):
            cla = cv2.createCLAHE(clipLimit=5)
            
            eq_V = cla.apply(img)
            
            return eq_V