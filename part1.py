'''All input files must go in input1 folder in current directory'''

from hmmlearn.hmm import GMMHMM
import copy
import numpy as np
n=3
trainList=[]
trainList.append(["single_1_mfcc.txt","single_3_mfcc.txt","single_5_mfcc.txt","single_6_mfcc.txt"])	#Training Files For Single Speaker
trainList.append(["TWO_1_mfcc.txt","TWO_2_mfcc.txt","TWO_3_mfcc.txt","TWO_4_mfcc.txt"])	#Training Files For Two Speakers
trainList.append(["3_1_mfcc.txt","3_2_mfcc.txt","3_3_mfcc.txt","3_4_mfcc.txt"])	#Training Files For Three Speakers
testList=[]
testList.append(["single_7_mfcc.txt","single_9_mfcc.txt","single_10_mfcc.txt"])	#Testing Files For Single Speaker
testList.append(["TWO_6_mfcc.txt","TWO_12_mfcc.txt","TWO_13_mfcc.txt","TWO_14_mfcc.txt"])	#Testing Files For Two Speakers
testList.append(["3_6_mfcc.txt","3_7_mfcc.txt","3_8_mfcc.txt"])	#Testing Files For Three Speakers
GMMList=[]
for i in range(n):
	GMMList.append(GMMHMM(n_components=3, n_mix=1, covariance_type='diag'))	#One GMM For Single, Two And Three Speakers

def testSingle(obs,cls):
	''' Takes each observation list and the class, and determines to which class the observation belongs to.
		Also returns whether it is the expected class
	'''


	maxVal=GMMList[0].score(obs)
	maxCls=0
	tmp=GMMList[1].score(obs)
	if(tmp>maxVal):
		maxVal=tmp
		maxCls=1
	tmp=GMMList[2].score(obs)
	if(tmp>maxVal):
		maxVal=tmp
		maxCls=2
	if(cls==maxCls):
		return 1
	else:
		return 0

def train():
	'''Reads the training files and trains GMMs accordingly '''

	for i in range(n):
		lst=[]
		for f in trainList[i]:
			fp=open("input1/"+f,"r")
			temp=[]
			for ln in fp.read().split("\n")[:-1]:
				temp.append([float(k) for k in ln.split(",")])
			lst.append(copy.deepcopy(temp))
			fp.close()
		lst= [np.array(k) for k in lst]
		GMMList[i].fit(np.array(lst))

def test():
	'''Reads the testing files and calls testSingle on each file.
		Prints accuracy '''

	rCount=0
	tCount=0
	lst=[]
	classList=[]
	for i in range(n):
		for f in testList[i]:
			fp=open("input1/"+f,"r")
			temp=[]
			for ln in fp.read().split("\n")[:-1]:
				temp.append([float(k) for k in ln.split(",")])
			lst.append(copy.deepcopy(temp))
			classList.append(i)
			fp.close()
	for i in range(len(lst)):
		tCount+=1
		rCount+=testSingle(lst[i],classList[i])
	print "Accuracy Is",(float(rCount)/float(tCount)*100.0),"%"

train()
test()
