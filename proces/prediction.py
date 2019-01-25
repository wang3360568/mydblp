# -*- coding: utf-8 -*-
import os
import json
import sys
import matplotlib.pyplot as plt
import util
import re
import csv
import time
import random
import HTMLParser
import cPickle as pickle
import collections
import numpy as np
import math
sys.path.append('.')
from myclass.myobj import Paper,Person,Author
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import spectral_clustering,KMeans
from networkTool import NetworkTool
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

class Prediction():
    # number of embedding dimensions
    m=32
    # number of sparse embedding dimensions
    p=128
    # number of clusters
    k=12

    d=7


    re=226
    t = 100000000.



    def __init__(self,edgeDict,nodeIndexDict,attributeslist,refexFeature,authorObjDict,isBinary=True):
        if isBinary:
            self.edgeDict=edgeDict
            self.nodeIndexDict=nodeIndexDict
            self.attributeslist=attributeslist
            self.refexFeature=refexFeature
            self.authorObjDict=authorObjDict
        else:
            self.edgeDict=pickle.load(open(edgeDict, "rb"))
            self.nodeIndexDict=pickle.load(open(nodeIndexDict, "rb"))
            self.attributeslist=pickle.load(open(attributeslist, "rb"))
            self.refexFeature=pickle.load(open(refexFeature, "rb"))
            self.authorObjDict=pickle.load(open(authorObjDict, "rb"))

        # total number of nodes of all timestamps
        self.q=0
        for key in self.nodeIndexDict:
            self.q+=len(nodeIndexDict[key])

    def getTrainAndTest_ours(self,samplingRate,f_list,gtNum):
        i=0
        xtrainList=[]
        xtestList=[]
        ytrainList=[]
        ytestList=[]
        for key in self.edgeDict:
            F=f_list[i]
            A=self.attributeslist[i]
            Y=MinMaxScaler().fit_transform(A)
            y=Y[:,gtNum]
            xtrain,xtest,ytrain,ytest=self.getTrainAndTest(samplingRate,F,y)
            xtrainList.append(xtrain)
            xtestList.append(xtest)
            ytrainList.append(ytrain)
            ytestList.append(ytest)

            i+=1

        return xtrainList,xtestList,ytrainList,ytestList

    def getTrainAndTest(self,samplingRate,y):
        xtrainIndex=[]
        xtestIndex=[]
        ytrain=[]
        ytest=[]
        for j in range(y.shape[0]):
            if random.random()<samplingRate:
                xtrainIndex.append(j)
                ytrain.append(y[j])
            else:
                xtestIndex.append(j)
                ytest.append(y[j])

        return xtrainIndex,xtestIndex,ytrain,ytest

    def getTrainAndTest_Onestep(self,samplingRate,y,key):
        indexdict=self.getNeiborghood(key)
        xtrainIndex=[]
        xtestIndex=[]
        ytrain=[]
        ytest=[]
        for j in range(y.shape[0]):
            if j not in indexdict: continue
            if random.random()<samplingRate:
                xtrainIndex.append(j)
                ytrain.append(y[indexdict[j]])
            else:
                xtestIndex.append(j)
                ytest.append(y[indexdict[j]])

        return xtrainIndex,xtestIndex,ytrain,ytest

    def getNeiborghood(self,key):
        current=self.nodeIndexDict[key]
        last=self.nodeIndexDict[key-1]
        index=0
        indexDict=dict()
        for keyId in last:
            if keyId in current:
                indexDict[index]=current.index(keyId)
            index+=1
        return indexDict
    
    def predicting_useFeature(self,f_list,gtNum,samplingRate):

        xtrainList,xtestList,ytrainList,ytestList=self.getTrainAndTest_ours(samplingRate,f_list,gtNum)

        for i in range(len(f_list)):
            yTest=self.gbrt(xtrainList[i],ytrainList[i],xtestList[i])
            self.results(yTest,ytestList[i])

    def perPrediction(self,embeddings,aNum,gtNum,samplingRate):
        A=self.attributeslist[aNum]
        Y=MinMaxScaler().fit_transform(A)
        y=Y[:,gtNum]
        xtrainIndex,xtestIndex,ytrain,yGround=self.getTrainAndTest(samplingRate,y)
        xtrain=embeddings[xtrainIndex]
        xtest=embeddings[xtestIndex]
        yTest=self.gbrt(xtrain,ytrain,xtest)
        self.results(yTest,yGround)

    def gbrt(self,xTrain,yTrain,xTest,n_es=50,max_dep=3,min_samples_split=2,learning_rate=0.01):
        params = {'n_estimators': n_es, 'max_depth': max_dep, 'min_samples_split': min_samples_split,'learning_rate': learning_rate, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(xTrain, yTrain)
        yTest= clf.predict(xTest)
        return yTest

    def adaboost(self,xTrain,yTrain,xTest,n_es=1000,max_dep=3,rng=np.random.RandomState(1)):
        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_dep), n_estimators=n_es, random_state=rng)
        clf.fit(xTrain, yTrain)
        yTest= clf.predict(xTest)
        return yTest

    def svr(self,xTrain,yTrain,xTest,c=100,ep=0.01,vb=True):
        clf = SVR(C=c, epsilon=ep,verbose=vb)
        clf.fit(xTrain, yTrain)
        yTest= clf.predict(xTest)
        return yTest

    def results(self,yTest,yGround):
        mse = mean_squared_error(yGround,yTest)
        print("MSE: %.4f" % mse)
        mae = mean_absolute_error(yGround,yTest)
        print("MAE: %.4f" % mae)
        return mse,mae

    def readLineAndDW(self,filepath,year):
        contentDict=dict()
        fp=open(filepath,'r')
        while 1:
            line = fp.readline()
            if not line:
                break
            content=line.strip().split(' ')
            numericContent= map(float, content[1:])
            contentDict[content[0]]=numericContent
        keylist=self.nodeIndexDict[year]
        outputlist=[]
        for key in keylist:
            outputlist.append(contentDict[key])
        farray=np.array(outputlist)
        return farray
    
    def output(self,keyList,labels,name):
        outputList=[]
        length=len(keyList)
        for i in range(length):
            clu=labels[i]
            theKey=keyList[i]
            outputList.append([theKey,self.authorObjDict[theKey].name,clu])

        util.write_csv_inlist('./authorCluster_'+name+'.csv',outputList)
    
    def draw(self,nodeIndex,nt,labels,name):
        labelDict=dict()
        for i in range(len(nodeIndex)):
            labelDict[nodeIndex[i]]=labels[i]
        nt.drawGraph(labelDict,nt.myGraph,name)

    def counter(self,labels,clusterNum):
        counterlist=[]
        for i in range(clusterNum):
            counterlist.append(0)

        for val in labels:
            counterlist[val]+=1

        return counterlist

    def concatenateMatrixInList(self,matrixList,dim,axis):
        if axis==0:
            E=np.empty([0,dim])
        elif axis==1:
            E=np.empty([dim,0])
        for i in range(len(matrixList)):
            E=np.concatenate((E, matrixList[i]), axis=axis)

        return E
    
    def updateMatrixInList(self,E,matrixList):
        globalLen=0
        for i in range(len(matrixList)):
            shape_i=matrixList[i].shape[0]
            matrixList[i]=E[globalLen:globalLen+shape_i]
            globalLen=globalLen+shape_i
        return matrixList

    def initSim(self,X,Y):
        sim_local=cosine_similarity(X)
        sim_global=cosine_similarity(Y)
        sim=(99*sim_local+sim_global)/100.0
        return sim

    def initX(self,Ajen):
        firstOrder=Ajen/Ajen.max()
        secondOrder=np.dot(Ajen,Ajen)
        thirdOrder=np.dot(Ajen,secondOrder)
        secondOrder=secondOrder/secondOrder.max()  
        thirdOrder=thirdOrder/thirdOrder.max()
        simM=(firstOrder+secondOrder+thirdOrder)/3.0
        return simM

    def eigenVectorAndEigenValue(self,M,k):
        val,vec=np.linalg.eig(M)
        sorted_indices = np.argsort(val)
        topk_evecs = vec[:,sorted_indices[:k:1]]
        topN=0.0
        counter=0
        for v in sorted_indices:
            if counter==k: break
            topN=topN+val[v]
            counter+=1

        return [topN,topk_evecs]


    def getLaplacian(self,W):
        D = np.zeros((W.shape[0], W.shape[1]))
        L = np.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[1]):
            D[i][i]=np.sum(W[:,i])
        L=D-W
        return [D,L]

    def NMI(self,perd,gtrues,clusterNum):
        total=len(perd)
        counter_p=[]
        for i in range(clusterNum):
            counter_p.append(0)
        counter_g=[0,0,0,0]
        mat=np.zeros((clusterNum,4))
        for i in range(total):
            counter_p[perd[i]]+=1
            counter_g[gtrues[i]]+=1
            mat[perd[i],gtrues[i]]+=1
        
        low_p1=0.0
        for i in range(len(counter_p)):
            val=counter_p[i]/float(total)
            low_p1+=val*math.log(val,2)
        
        low_p2=0.0
        for i in range(len(counter_g)):
            val=counter_g[i]/float(total)
            low_p2+=val*math.log(val,2)

        low=-(low_p1+low_p2)

        upp=0.0
        for i in range(len(counter_p)):
            pcVal=counter_p[i]/float(total)
            pval=0.0
            for j in range(len(counter_g)):
                if mat[i,j]==0: continue
                val=float(mat[i,j])/(counter_p[i])
                pval+=val*math.log(val,2)
            upp+=pval*pcVal
        upp=-upp
        nmi=2*(-low_p2-upp)/low
        return nmi



if __name__ == "__main__":

    label='0.001'
    edgeDict=pickle.load(open('edgeDict_'+label+'.dat', "rb"))
    nodeIndexDict=pickle.load(open('nodeIndexDict_'+label+'.dat', "rb"))
    attributeslist=pickle.load(open('attributeslist_'+label+'.dat', "rb"))
    refexFeature=pickle.load(open('refexFeature_'+label+'.dat', "rb"))
    authorDict_obj=pickle.load(open('authorDict_obj.dat', "rb"))

    pred=Prediction(edgeDict,nodeIndexDict,attributeslist,refexFeature,authorDict_obj)
    gtNum=4
    sampleRate=0.8
    A=pred.attributeslist[-1]
    Y=MinMaxScaler().fit_transform(A)
    y=Y[:,gtNum]
    xtrainIndex,xtestIndex,ytrain,yGround=pred.getTrainAndTest(sampleRate,y)
    
    f_list=pickle.load(open('P_list_sharedSpace_mf_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-1]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)

    f_list=pickle.load(open('P_list_sharedSpace_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-1]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)

    f_list=pickle.load(open('P_list_sharedSpace_mfdw_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-1]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    
    # pred.predicting_useFeature(f_list,gtNum,sampleRate)
    lineEmbedding=pred.readLineAndDW('D:\\code\\LINE\\windows\\2013_output_1.txt',2013)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\LINE\\windows\\2013_output.txt',2013)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\deepwalk\\2013_dw_output.txt',2013)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\graphwave\\graph_wave_embeddings.txt',2013)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)
    
    # clu.clustering_useFeature(f_list)
    print '-------------------------------------------' 
    xtrainIndex,xtestIndex,ytrain,yGround=pred.getTrainAndTest_Onestep(sampleRate,y,2013)

    f_list=pickle.load(open('P_list_sharedSpace_mf_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-2]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)

    f_list=pickle.load(open('P_list_sharedSpace_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-2]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)

    f_list=pickle.load(open('P_list_sharedSpace_mfdw_'+label+'.dat', "rb"))
    lineEmbedding=f_list[-2]
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)

    lineEmbedding=pred.readLineAndDW('D:\\code\\LINE\\windows\\2012_output_1.txt',2012)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\LINE\\windows\\2012_output.txt',2012)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\deepwalk\\2012_dw_output.txt',2012)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    # pred.perPrediction(lineEmbedding,-1,gtNum,sampleRate)

    lineEmbedding=pred.readLineAndDW('D:\\code\\graphwave\\graph_wave_embeddings_2012.txt',2012)
    xtrain=lineEmbedding[xtrainIndex]
    xtest=lineEmbedding[xtestIndex]
    yTest=pred.gbrt(xtrain,ytrain,xtest)
    pred.results(yTest,yGround)
    
