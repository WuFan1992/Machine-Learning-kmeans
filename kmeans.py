#!/usr/bin/env python

import numpy as np
import random


def LoadData(filename):
	dataMat = []
	infile = open(filename, 'r')
	for line in infile.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	dataMat = np.mat(dataMat)
	return dataMat



def Distance(VecA, VecB):
	return np.sqrt((np.power(VecA - VecB, 2)).sum())


#This funtion is used to create the cluster centroid
#dataSet is like this [[2.1,4.5],[3.6,9.8]]
def randCent(dataSet,k):
	# dimension of vector
	dim = np.shape(dataSet)[1]
	
	# create list of centroid
	centroid = np.mat(np.zeros((k,dim)))
	#centroid = np.zeros((k,dim)) this is fault

	# loop to find the biggest and smallest value
	for j in range(dim):
		Max_value = dataSet[:,j].max()
		Min_value = dataSet[:,j].min()
		Diff = Max_value - Min_value
		centroid[:,j] = Min_value + Diff*np.random.rand(k,1)

	return centroid
	
def Kmeans(dataSet,k):
	data_quantity = np.shape(dataSet)[0] # the total number of dataSet
	ClusterAssment = np.mat(np.zeros((data_quantity,2))) # this list with 2 coloum is used to save , for each data , it is in which cluster		
	centroid = randCent(dataSet,k)   # initialization of centroid, with values in random
	clusterchange = True
	# begin to loop
	while clusterchange:
		clusterchange = False
		for i in range(data_quantity):
			cluster_num = 0
			Min_Distance = float("inf")
			for j in range(k):
				dis = Distance(dataSet[i,:],centroid[j,:])
				if dis < Min_Distance :
					Min_Distance = dis
					cluster_num = j
					if ClusterAssment[i,0] != cluster_num:
						clusterchange = True
					ClusterAssment[i,:] = cluster_num, Min_Distance
	# update the centroid
	
	for cent in range(k):
		temp = dataSet[np.nonzeros(ClusterAssment[:,0] == cent)[0]]
		centroid[cent,:] = np.means(temp,axis = 0)
	return centroid, ClusterAssment
		


if __name__ == '__main__':
	dataSet = LoadData('testSet.txt')
	centroid,ClusterAssment = Kmeans(dataSet,10)
	print (centroid)



