import cv2
from time import time
import os
import msvcrt
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np

#ark_background
plt.style.use('dark_background')

face_cascade = cv2.CascadeClassifier('Additional/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Additional/haarcascade_eye.xml')#_tree_eyeglasses.xml')
upper_coefficient = 0.25
lower_coefficient = 0.6
h_percentage = 0.2
w_percentage = 0.2

def eyeDetection(faces, img, gray):
    t = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y+int(h*upper_coefficient):y+int(h*lower_coefficient), x:x+w]
        roi_color = img[y+int(h*upper_coefficient):y+int(h*lower_coefficient), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.6, 5)
    eyes = eyes[:2]
    if len(eyes) < 1:
        t = time()
    for (ex, ey, ew, eh) in eyes:
        cv2.circle(roi_color, (ex+ew/2, ey+eh/2), 5, (0, 255, 0), thickness=1)
    return t

def faceDetection(img):
    t = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cv2.rectangle(img, (0, 0), (len(img[0]), len(img)), (0, 0, 255), 2)
    number = len(faces)
    if number == 0:
        print "\a"
    else:
        t = eyeDetection(faces, img, gray)
#        print "Face detection time was: %.3f segundos"%(time()-detect_time)
    return img, t, faces

def show_webcam(path, vivo = True, camara = 0):
    cam = cv2.VideoCapture(camara)
#    cam.set(cv2.cv.CV_CAP_PROP_FPS, 30)
    f = open(path+'/Blink.csv', 'wb')
    wr = csv.writer(f)#, quoting=csv.QUOTE_ALL)
    wr.writerow(["Time (s)", "Difference (s)"])
    wr.writerow([0.0,0.0])
    f.close()
    eyes_time = [0]
    diff_time = [0]
    start_time = time()
    i = 0
    count_closes = 1
    ret_val, img = cam.read()
    if ret_val:
        height = len(img[:,0])# - 1
        weight = len(img[0,:])# - 1
        x = 0
        w = weight
        y = 0
        h = height
        
        #### Plots
    #    increase = 60
        fig, ax = plt.subplots(1,1, figsize = (8, 4.5)) 
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 20)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Time difference (s)")
        ax.hold(True)
        plt.grid()
        fig.canvas.manager.window.move(0, 0)
        fig.show(False)
        plt.draw()
        points = ax.plot(eyes_time, diff_time, 'o', label = "Data")[0]#, animated=True)#[0]
        line = ax.plot([], [], "-", color="red", label= "Average")[0]
        tempC = ax.plot([], [], "-", lw=10, color="green", label="Standard deviation")[0]
        plt.legend()
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
        background = fig.canvas.copy_from_bbox(ax.get_figure().bbox)
        text = ax.text(0, 19, r"Average difference is %.3f $\pm$"%0.0 + "%.3f s"%0.0)
        text_Count = ax.text(0, 18, "Count %d"%0)
        print "Press Esc to exit"
        while cam.isOpened():
            ret_val, img = cam.read()
            img_crop, t, faces = faceDetection(img[y:y+h, x:x+w])
            if msvcrt.kbhit():
                if ord(msvcrt.getch()) == 27:
                    break
            if i == 0:
                cv2.imwrite(path+"/0-First.jpg", img_crop)
            if t != 0:
                cv2.putText(img, "Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                dif = t-start_time
                tempH = int(h*upper_coefficient)
                cv2.putText(img_crop, "%.4f s"%dif, (20, tempH+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                if dif-eyes_time[-1] > 1:
                    temp = dif-eyes_time[-1]
                    
                    cv2.imwrite(path+"/%.0f-close.jpg"%dif, img_crop[tempH:int(h*lower_coefficient), :])
                    f = open(path+'/Blink.csv', 'a')
                    wr = csv.writer(f)
                    wr.writerow([dif, temp])
                    f.close()
                    
                    diff_time.append(temp)
                    eyes_time.append(dif)
                    
                    averageDif = np.array([np.sum(diff_time)/len(diff_time) for a in [0,1]])
                    std = np.std(diff_time)
                    #### Plots
                    for col in (ax.collections):
                        ax.collections.remove(col)
                        
                    # restore background
                    plotMin, plotMax = ax.get_xlim()
                    if eyes_time[-1] >= plotMax or count_closes == 1:
                        line.set_data([], [])
                        points.set_data([], [])
                        text.set_text("")
                        text_Count.set_text("")
                        ax.set_xlim(0, plotMax*2)
                        fig.canvas.draw()
                        background = fig.canvas.copy_from_bbox(ax.get_figure().bbox)
    
                    # update data
                    temp_eyes = np.array([eyes_time[0], eyes_time[-1]])
                    ax.fill_between(temp_eyes, averageDif+std, averageDif-std, alpha=0.5, color="green")
                    line.set_data([eyes_time[0], eyes_time[-1]], averageDif)                
                    points.set_data(eyes_time, diff_time)
                    text.set_text(r"Average Difference is %.3f $\pm$"%averageDif[0] + "%.3f s"%std)                 
                    text_Count.set_text("Count: %d"%count_closes)                
                    fig.canvas.draw()
                    # update plot
                    if eyes_time[-1] >= plotMax:
                        fig.canvas.draw()
                    else:
                        fig.canvas.restore_region(background)
                        ax.draw_artist(ax.get_children()[0])
    #                    ax.draw_artist(ax.collections)
                        ax.draw_artist(points)
                        ax.draw_artist(line)
                        ax.draw_artist(text)
                        ax.draw_artist(text_Count)
                        fig.canvas.blit(ax.clipbox)#bbox)
                    if not plt.get_fignums():
                        fig.show(False)
                    count_closes += 1
            if vivo:
                cv2.imshow("Stream", img)
                if cv2.waitKey(1) == 27:
                    break
            if len(faces) == 0:
                y = 0
                h = height
                x = 0
                w = weight
            else:
                x, y, w, h = limitsHandler(faces, x, y)
            i += 1
        print "Total closes: %d"%count_closes
        plt.close(fig)
        cam.release()
        cv2.destroyAllWindows()            
        return eyes_time, diff_time, averageDif, std
    return None, None, None, None

def limitsHandler(faces, x, y):
    for face in faces:
        # relative to past
        x0, y0, w, h = face
        # actual
        rX = x0 + x
        rY = y0 + y
        tempH = int(h*h_percentage)
        tempW = int(w*w_percentage)
        x = rX - tempW
        y = rY - tempH
        h += 2*tempH
        w += 2*tempW
        if x < 0:
            x = 0
        if y < 0:
            y = 0
    return x, y, w, h

def differences(times):
    temp = [0]
    for i in range(1, len(times)):
        temp.append(times[i]-times[i-1])
    return temp            

def main(path, vivo, camara):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        temp = glob.glob(path+'/*.jpg')
        for item in temp:
            os.remove(item)
#    try:
        return show_webcam(path, vivo, camara)	
#    except Exception:
        print "No camara attached"