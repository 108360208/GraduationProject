import threading
from apscheduler.schedulers.background import BackgroundScheduler
import path
import camera
golf_path=path.path()
camera1=camera.camera(1)
camera2=camera.camera(2)
def job():

    if (len(camera1.position.golf_X)>1 and len(camera2.position.golf_X)>1): 
        golf_path.cal_vel(camera2.position.convertdata())
        golf_path.cal_angle(camera1.position.convertdata())
        print(camera1.position.convertdata())
        print(camera2.position.convertdata())
        golf_path.printpath()     
        #httppost.httppost(golf_path.vel,golf_path.angle1,golf_path.angle2)
    else:
        print("data is few")
    camera1.position.__init__() 
    camera2.position.__init__() 
sched = BackgroundScheduler(timezone='MST')
sched.add_job(job, 'interval', id='2_second_job', seconds=1)
sched.start() 
def find1(camera):
    camera.OK(204,255,64,29,29,5,40)
def find2(camera):
    camera.OK(204,255,64,32,32,5,20)
if __name__ == '__main__':
    p1= threading.Thread(target = find1,args=(camera1,))
    p2= threading.Thread(target = find2,args=(camera2,))
    #p3=threading.Thread(target = hstack)
    p1.start() 
    p2.start() 
    





    
