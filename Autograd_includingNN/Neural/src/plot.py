import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_data_from_file(filename,lable="",scatter=True,Color="blue"):
    data = pd.read_csv(filename,sep=" ",header=None)
    data = data.values
    X = data[:, 0]
    Y = data[:, 1]
    print(X,Y)
    if lable != "":
        if scatter==True:
         plt.scatter(X, Y, label=lable,color=Color)
        else:
         plt.plot(X,Y,label=lable,color=Color)
        plt.legend()
    else:
        plt.scatter(X, Y)
    x_max = np.max(X)+0.2
    x_min = np.min(X)-0.2
    return [x_min,x_max]

def plot_data(func,intrervall):
    x = np.linspace(intrervall[0],intrervall[1],1000)
    y = func(x)
    plt.plot(x,y)
    
    
def plot_data_from_file_and_function(filename,func):
    intrervall = plot_data_from_file(filename)
    plot_data(func,intrervall)
    

filenname_1 = "train.dat"
filenname_2 = "func.dat"
filename_3="test.dat"
plot_data_from_file(filenname_1,"f_pred",Color="black")
plot_data_from_file(filenname_2,"f_real",scatter=False,Color="green")
plot_data_from_file(filename_3,"f_test")

plt.xlabel("x")
plt.ylabel("y")

plt.savefig("sin.png")
plt.show()