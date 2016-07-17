import EyeDetection as eye
import matplotlib
import numpy as np
matplotlib.use('Qt4Agg') 
import matplotlib.pyplot as plt

path = "Results"
camara = 0
time, dif, averageDif, std = eye.main(path, True, camara)

plt.style.use('classic')
if time != None:    
    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.subplot(111)
    temp = [time[0], time[-1]]
    ax.fill_between(temp, averageDif+std, averageDif-std, alpha=0.5, color="green")
    plt.plot(time, dif, "o", label="Data")
    plt.plot(temp, averageDif, "-", color="red", label="Average")
    tempC = ax.plot([], [], "-", lw=10, color="green", label="Standard deviation")
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='12')
    plt.xlabel("Time (s)")
    plt.ylabel("Time difference(s)")
    plt.grid()
    plt.ylim(0, 20)
    plt.xlim(0, 10*np.ceil(time[-1]/10.0))
    plt.text(0, 19, r"Average difference is %.3f$\pm$"%averageDif[0] + "%.3f s"%std)
    plt.savefig(path+"/Results.pdf", transparent=True)
    plt.show()    
