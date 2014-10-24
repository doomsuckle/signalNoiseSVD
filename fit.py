#!/usr/env python

from sklearn import svm
from random import gauss
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from ROOT import * 


def gen_signal(): 
	return (gauss(-1,3), 1)


def gen_background(): 
	return (gauss(5,3), 0)


def gen_sim(nsig, nbkg):
	ret = []
	ret.extend([gen_signal() for _ in range(nsig)])
	ret.extend([gen_background() for _ in range(nbkg)])
	return ret

def map_data(inlist):
	x, y= zip(*inlist)
	x = map(lambda a: [a], x)
	return x, y

def fill_hists(hists, dat):
	for val, target in dat:
		hists[target].Fill(val)

# Show confusion matrix in a separate window
# non-blocking
def draw_confusion(cm):
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show(block=False)


def get_purity(cm):
	nacc = sum(zip(*cm)[1])
	return cm[1][1] * 1./nacc


if __name__ == '__main__':
	

	can = TCanvas()
	hists = {}
	hists[1] = TH1F('a','a', 100, -20, 20)
	hists[0] = TH1F('b','b', 100, -20, 20)
	
	hists[1].SetLineWidth(3)
	hists[0].SetLineWidth(3)
	hists[0].SetLineColor(kRed)
	hists[1].SetLineColor(kBlue)


# get training data
	train = gen_sim(100, 1000)
	xtrain, ytrain = map_data(train)

	test = gen_sim(100, 1000) 
	xtest, ytest = map_data(test)

	fill_hists(hists,train)

	clf = svm.SVC(kernel='linear')

	model = clf.fit(xtrain, ytrain)
#	score = model.score(xtest,ytest)
#	print score 

	
	ypred = clf.predict(xtest)
	print accuracy_score(ytest, ypred)
	cm = confusion_matrix(ytest, ypred)
	print get_purity(cm)

	hists[1].Draw()
	hists[0].Draw('same')
	can.Update()

	draw_confusion(cm)

