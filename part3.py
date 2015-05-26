'''All input files must go in input1 folder in current directory'''

from hmmlearn.hmm import GMMHMM
import copy
import numpy as np
n=4
trainList=[]

#Training files for Mister

trainList.append(["001 (1)_mfcc.txt", "001 (2)_mfcc.txt", "001 (3)_mfcc.txt", "001 (4)_mfcc.txt", "001 (5)_mfcc.txt", "001 (6)_mfcc.txt",
 "001 (7)_mfcc.txt", "001 (8)_mfcc.txt", "001 (9)_mfcc.txt", "001 (10)_mfcc.txt", "001 (11)_mfcc.txt", "001 (12)_mfcc.txt", "001 (13)_mfcc.txt",
 "001 (14)_mfcc.txt", "001 (15)_mfcc.txt", "001 (16)_mfcc.txt", "001 (17)_mfcc.txt", "001 (18)_mfcc.txt", "001 (19)_mfcc.txt"])

#Training files for One Second
trainList.append(["002 (1)_mfcc.txt", "002 (2)_mfcc.txt", "002 (3)_mfcc.txt", "002 (4)_mfcc.txt", "002 (5)_mfcc.txt", "002 (6)_mfcc.txt",
 "002 (7)_mfcc.txt", "002 (8)_mfcc.txt", "002 (9)_mfcc.txt", "002 (10)_mfcc.txt", "002 (11)_mfcc.txt", "002 (12)_mfcc.txt", "002 (13)_mfcc.txt",
 "002 (14)_mfcc.txt", "002 (15)_mfcc.txt", "002 (16)_mfcc.txt", "002 (17)_mfcc.txt", "002 (18)_mfcc.txt", "002 (19)_mfcc.txt",
 "002 (20)_mfcc.txt", "002 (21)_mfcc.txt"])

#Training files for Thank You
trainList.append(["64_mfcc.txt","145_mfcc.txt","13025_mfcc.txt"])

#Training files for Others
trainList.append(["000 (1)_mfcc.txt", "000 (2)_mfcc.txt", "000 (3)_mfcc.txt", "000 (4)_mfcc.txt", "000 (5)_mfcc.txt", "000 (6)_mfcc.txt",
 "000 (7)_mfcc.txt", "000 (8)_mfcc.txt", "000 (9)_mfcc.txt", "000 (10)_mfcc.txt", "000 (11)_mfcc.txt", "000 (12)_mfcc.txt", "000 (13)_mfcc.txt",
 "000 (14)_mfcc.txt", "000 (15)_mfcc.txt", "000 (16)_mfcc.txt", "000 (17)_mfcc.txt"])

testList= []

#Testing files for Mister
testList.append(["001 (20)_mfcc.txt", "001 (21)_mfcc.txt", "001 (22)_mfcc.txt", "001 (23)_mfcc.txt", "001 (24)_mfcc.txt"])

#Testing files for One Second
testList.append(["002 (22)_mfcc.txt", "002 (23)_mfcc.txt", "002 (24)_mfcc.txt", "002 (25)_mfcc.txt", "002 (26)_mfcc.txt", "002 (27)_mfcc.txt"])

#Testing files for Thank You
testList.append(["26195_mfcc.txt"])

#Testing files for Others
testList.append(["000 (18)_mfcc.txt", "000 (19)_mfcc.txt", "000 (20)_mfcc.txt", "000 (21)_mfcc.txt", "000 (22)_mfcc.txt"])
GMMList=[]

#testFile= raw_input("Enter the audio file name\n")

#testFile= "input3/"+ testFile.split('.')[0]+ "_mfcc.txt";	

for i in range(n):
	GMMList.append(GMMHMM(n_components=1, n_mix=3, covariance_type='diag'))

def testSingle(obs, cls):
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
	tmp=GMMList[3].score(obs)
	if(tmp>maxVal):
		maxVal=tmp
		maxCls=3
	
	global clss
	clss= ""
	if(maxCls== 0):
		clss= "Mister"
	elif(maxCls== 1):
		clss= "One Second"
	elif(maxCls== 2):
		clss= "Thank You"
	else:
		clss= "Others"

	if(cls==maxCls):
		return 1
	else:
		return 0

def train():
	'''Reads the training files and trains GMMs accordingly '''
	
	for i in range(n):
		lst=[]
		for f in trainList[i]:
			fp=open("input3/"+f,"r")
			tmp= []
			for ln in fp.read().split("\n")[:-1]:
				tmp.append([float(k) for k in ln.split(",")])
			lst.append(copy.deepcopy(tmp))
			fp.close()
		GMMList[i].fit(np.array(lst))

def test():
	rcount= 0
	tcount= 0
	lst=[]
	classList= []
	for i in range(n):
		for f in testList[i]:
			fp=open("input3/"+f,"r")
			temp=[]
			for ln in fp.read().split("\n")[:-1]:
				temp.append([float(k) for k in ln.split(",")])
			lst.append(copy.deepcopy(temp))
			classList.append(i)
			fp.close()

	for i in range(len(lst)):
		tcount+= 1
		rcount+= testSingle(lst[i], classList[i])


	print "Accuracy= ", (float(rcount)/tcount)*100.0, "%"
train()
test()
