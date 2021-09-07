import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

class LogReg:
    def __init__(self, json_path):
        data = open(json_path)
        self.raw_data = json.load(data)
        
    def iter_mean(iterable_arr):
        return np.sum(iterable_arr)/len(iterable_arr)
    
    def iter_variance(iterable_arr, mean):
        return np.sum((iterable_arr - mean)**2)
    
    def iter_cov(iter_x, mean_x, iter_y, mean_y):
        result = (iter_x - mean_x)*(iter_y - mean_y)
        return np.sum(result)
    
    def get_coef(self):
        self.a = (sum((self.raw_data['x']-np.mean(self.raw_data['x']))*(self.raw_data['is_adult']-np.mean(self.raw_data['is_adult']))))/sum((self.raw_data['x']-np.mean(self.raw_data['x'])))
        self.b = np.mean(self.raw_data['is_adult']) - np.mean(self.raw_data['x'])*self.a
        
    def get_plot(self):
        x = self.raw_data['x']
        y  = math.exp(self.a + self.b)
        y1 = self.raw_data['is_adult']

        #PLOT 
        fig, ax = plt.subplots()
        ax.plot(x,y,'-',x,y1,'o')
        plt.show()


weight = LogReg('/Users/Zhou/Documents/GitHub/590-jz708/HW1.1/HW1.1/weight.json')
weight.get_coef()
weight.get_plot()

    
