import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_data_from_file(filename,lable=""):
    data = pd.read_csv(filename,sep=" ")
    data = data.values
    X = data[:, 0]
    Y = data[:, 1]
    if lable != "":
        plt.plot(X, Y, label=lable)
        plt.legend()
    else:
        plt.plot(X, Y)
    x_max = np.max(X)
    x_min = np.min(X)
    return [x_min,x_max]

def plot_data(func,intrervall):
    x = np.linspace(intrervall[0],intrervall[1],1000)
    y = func(x)
    plt.plot(x,y)
    
    
def plot_data_from_file_and_function(filename,func):
    intrervall = plot_data_from_file(filename)
    plot_data(func,intrervall)
    

filenname_1 = "f_pred.dat"
filenname_2 = "f_real.dat"
plot_data_from_file(filenname_1,"f_pred")
plot_data_from_file(filenname_2,"f_real")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("sin.png")
plt.show()