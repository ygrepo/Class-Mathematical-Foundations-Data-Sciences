import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import setp
from sklearn.model_selection import train_test_split



def split_and_plot(a, max_temp_train, rain_train, max_temp_val, rain_val, grid): 
	
	grid_less_a = grid[grid<=a]
	grid_greater_a = grid[grid>=a]
	
	
	# fill in code to assign values for the variables below.

	linear_fit_rain_less_a = None #prediction values on relevant points on grid using linear fit with points max_temp < a
	linear_fit_rain_greater_a =  None #prediction values on relevant points on grid using linear fit with points max_temp > a
	
	fig = plt.figure(figsize = (9,6)) 
	plt.scatter(max_temp_val, rain_val, s=5, c="dodgerblue", marker='o', edgecolor="skyblue")

	plt.plot(grid_less_a,linear_fit_rain_less_a,'-o',lw=3,color='purple',label="<a")
	plt.plot(grid_greater_a,linear_fit_rain_greater_a,'-o',lw=3,color='crimson',label=">a")
	plt.axvline(a, ymin = 0, ymax = np.max(max_temp_train), color = 'k', linestyle = 'dotted')
	plt.ylabel("Rain", fontsize=15,labelpad=10)
	plt.xlabel("Maximum temperature", fontsize=15,labelpad=10)
	plt.legend(fontsize=18)
	plt.xticks(fontsize=18) 
	plt.yticks(fontsize=18)
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.gcf().subplots_adjust(left=0.15)
	
	# use a relevant error metric below. make sure you normalize correctly to take into account number of datapoints.
	val_error = #error of your fit on validation data points
	train_error = #errpr of your fit on your train data points
	plt.title('a = '+str(a) + ' val error: ' + str(round(val_error, 3)) + ' train error: ' + str(round(train_error, 3)), 
			 fontsize=18)
	plt.savefig('a='+str(a)+'.pdf')
	
	
	
def main():
	
	dataset = np.loadtxt("oxford_temperatures.txt")

	max_temp = np.array(dataset[:,2])
	rain = np.array(dataset[:,5])

	max_temp_train, max_temp_test, rain_train, rain_test = train_test_split(max_temp, rain, test_size=0.30, random_state=42)
	max_temp_val, max_temp_test, rain_val, rain_test = train_test_split(max_temp_test, rain_test, test_size=0.50, random_state=42)

	width_bin = 0.5
	max_val = np.max(max_temp)
	grid = np.arange(0, max_val + 1,width_bin)
	
	for a in np.arange(4, 25, 4):
		split_and_plot(a, max_temp_train, rain_train, max_temp_val, rain_val)
		
	### selecting the best model according to val accuracy
	a = #best value of a from grid search above
	
	grid_less_a = grid[grid<=a]
	grid_greater_a = grid[grid>=a]
   
	linear_fit_rain = #regular linear regression on the entire train dataset
	linear_fit_rain_less_a = #fit from your best model/best a
	linear_fit_rain_greater_a = #fit from your best model/best a

	fig = plt.figure(figsize = (9,6)) 
	plt.scatter(max_temp_test,rain_test, s=5, c="dodgerblue", marker='o', edgecolor="skyblue")

	plt.plot(grid,linear_fit_rain,'--',lw=3,color='purple',label="Linear regression")
	plt.plot(grid_less_a,linear_fit_rain_less_a,'-o',lw=3,color='crimson')
	plt.plot(grid_greater_a,linear_fit_rain_greater_a,'-o',lw=3,color='crimson',label="a="+str(a))
	plt.axvline(a, ymin = 0, ymax = np.max(max_temp), color = 'k', linestyle = 'dotted')
	plt.ylabel("Rain", fontsize=15,labelpad=10)
	plt.xlabel("Maximum temperature", fontsize=15,labelpad=10)
	plt.legend(fontsize=18)
	plt.xticks(fontsize=18) 
	plt.yticks(fontsize=18)
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.gcf().subplots_adjust(left=0.15)
	
	a_error = #error of your best model
	lr_error = #error of regular linear regression
	plt.title('a='+str(a) + '_error: ' + str(round(a_error, 3)) + ' LR_error: ' + str(round(lr_error, 3)), 
			 fontsize=18)
	plt.savefig('test_comparison.pdf')

if __name__ == "__main__": 
	main()
	

