'''All input files must go in input1 folder in current directory'''

from hmmlearn.hmm import GMMHMM
import copy
import numpy as np
n1=3
n2=2
trainList1=[]
trainList1.append(["arnab_1_mfcc.txt", "arnab_2_mfcc.txt", "arnab_5_mfcc.txt", "arnab_6_mfcc.txt"]) #Training Files For Arnab
trainList1.append(["arvind_1_mfcc.txt", "arvind_2_mfcc.txt", "arvind_3_mfcc.txt", "arvind_4_mfcc.txt"]) #Training Files For Arvind
trainList1.append(["host1_mfcc.txt", "host2_mfcc.txt", "host3_mfcc.txt", "host4_mfcc.txt"]) #Training Files For Ravish

GMMList1=[]

trainList2=[]
trainList2.append(["single_1_mfcc.txt","single_3_mfcc.txt","single_5_mfcc.txt","single_6_mfcc.txt"]) #Training Files For Single Speaker
trainList2.append(["MULTI_1_mfcc.txt", "MULTI_2_mfcc.txt", "MULTI_3_mfcc.txt", "MULTI_4_mfcc.txt"]) #Training Files For Multiple Speakers

testFile= raw_input("Enter the audio mfcc file name\n")

testFile= "input2/"+ testFile

GMMList2=[]

classification1= []
classification2= []

#Counts for speakers

arnabc=0
arvindc=0
ravishc=0

for i in range(n1):
	GMMList1.append(GMMHMM(n_components=3, n_mix=1, covariance_type='diag'))

for i in range(n2):
	GMMList2.append(GMMHMM(n_components=3, n_mix=1, covariance_type='diag'))

def testSingle(obs):

	maxVal=GMMList2[0].score(obs)
	maxCls=0
	tmp=GMMList2[1].score(obs)
	if(tmp>maxVal):
		maxVal=tmp
		maxCls=1
	if(maxCls==1):
		classification1.append(-1)
		classification2.append(1)
	else:
		classification2.append(0)
		maxVal=GMMList1[0].score(obs)
		maxCls= 0
		tmp=GMMList1[1].score(obs)
		if(tmp>maxVal):
			maxVal=tmp
			maxCls=1
		tmp=GMMList1[2].score(obs)
		if(tmp>maxVal):
			maxVal=tmp
			maxCls=2

		classification1.append(maxCls)

	
def train():
	for i in range(n1):
		lst=[]
		for f in trainList1[i]:
			fp=open("input2/"+f,"r")
			j=0
			temp=[]
			for ln in fp.read().split("\n")[:-1]:
				temp.append([float(k) for k in ln.split(",")])
				j=j+1
				if(j==10):
					lst.append(copy.deepcopy(temp))
					j=0
					temp=[]
			fp.close()
		lst= [np.array(k) for k in lst]
		GMMList1[i].fit(np.array(lst))

	for i in range(n2):
		lst=[]
		for f in trainList2[i]:
			fp=open("input2/"+f,"r")
			j=0
			temp=[]
			for ln in fp.read().split("\n")[:-1]:
				temp.append([float(k) for k in ln.split(",")])
				j=j+1
				if(j==10):
					lst.append(copy.deepcopy(temp))
					j=0
					temp=[]
			fp.close()
		GMMList2[i].fit(np.array(lst))

def writeToFile():

	'''Writes to tempFiles, the timelines of different speakers '''

	fp1= open("timeLine1.txt", "w")
	fp2= open("timeLine2.txt", "w")
	global arnabc
	global arvindc
	global ravishc

	j= 0
	arn=0
	arv= 0
	rav= 0
	mul =0
	for i in range(len(classification1)):
		if(i%10== 0 and i>0):
			arn= classification1[j:i].count(0)
			arv= classification1[j:i].count(1)
			rav= classification1[j:i].count(2)
			mul= classification1[j:i].count(-1)
			if(max(arn, arv, rav, mul)== arn):
				speaker= "Arnab"
				arnabc+= 1
			elif(max(arn, arv, rav, mul)== arv):
				speaker= "Arvind"
				arvindc+= 1
			elif(max(arn, arv, rav, mul)== rav):
				speaker= "Ravish"
				ravishc+= 1
			else:
				speaker= "Multiple"

			fp1.write("For "+str((i-10)/20.0)+ " to "+str(i/20.0)+ " seconds, "+speaker)
			fp1.write("\n")

			j= i

	sin= 0
	mul= 0
	j= 0
	for i in range(len(classification2)):
		if(i%10== 0 and i >0):
			sin= classification2[j:i].count(0)
			mul= classification2[j:i].count(1)
			if(sin>= mul):
				cla= "Single"
			else:
				cla= "Multiple"

			fp2.write("For "+str((i-10)/20.0)+ " to " +str(i/20.0)+" seconds, there is/are "+cla+" speaker(s).")
			fp2.write("\n")
			j= i

	fp1.close()
	fp2.close()


def test():
	mfccCount=0
	lst=[]
	fp=open(testFile,"r")
	j=0
	temp=[]
	for ln in fp.read().split("\n")[:-1]:
		mfccCount+= 1
		temp.append([float(k) for k in ln.split(",")])
		j=j+1
		if(j==10):
			lst.append(copy.deepcopy(temp))
			j=0
			temp=[]
	fp.close()
	for i in range(len(lst)):
		testSingle(lst[i])

	writeToFile()

train()
test()

print "Arnab spoke for ", arnabc/2.0," seconds."
print "Arvind spoke for ", arvindc/2.0, " seconds."
print "Ravish spoke for ", ravishc/2.0, " seconds."
