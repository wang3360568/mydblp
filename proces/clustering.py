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
sys.path.append('.')
from myclass.myobj import Paper,Person,Author
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import spectral_clustering,KMeans
from networkTool import NetworkTool

class Clustering():
    # number of embedding dimensions
    m=32
    # number of sparse embedding dimensions
    p=128
    # number of clusters
    k=12

    d=7

    re=226

    theta=5 # paremeter for the second order
    alpha=30 # for ||S||
    lamda=10 # for ||W||
    rho=0.0001 # for tr(FLsF)
    gamma=0.2 # for ||D||
    epsilon=30 # for SI-1
    beta=10 # for ||DA-F||
    eta=100 # for ||I(S-f(X))||
    mu=2 # for ||Y-AM||
    theta=30 # for ||M||
    zeta=0.01 # for ||FF-I||

    lamda_pgd=100

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


    def clustering_preGraph(self):


        for key in self.edgeDict:
            clusterNum=8

            A=self.edgeDict[key]
            nt=NetworkTool()
            nt.initNetwork(A,nodeIndexDict[key])
            # X=self.initX(A)
            # labels = spectral_clustering(A, n_clusters=clusterNum, eigen_solver='arpack')
            # counter=self.counter(labels,clusterNum)

            Y=self.refexFeature[key]
            pca = PCA(n_components=50, svd_solver='full')
            Y_50=pca.fit_transform(Y)
            S=cosine_similarity(Y_50)
            S=(S+1.0)/2.0
            labels = spectral_clustering( S, n_clusters=clusterNum, eigen_solver='arpack')
            counter=self.counter(labels,clusterNum)

            self.draw(nodeIndexDict[key],nt,labels,str(key)+'_spectral_'+str(clusterNum)+'.png')
            self.output(nodeIndexDict[key],labels,str(key)+'_spectral')


            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(Y_50)
            labels_km=kmeans.labels_.tolist()
            counter=self.counter(labels_km,clusterNum)

            self.draw(nodeIndexDict[key],nt,labels_km,str(key)+'_kmean_'+str(clusterNum)+'.png')
            self.output(nodeIndexDict[key],labels_km,str(key)+'_kmean')


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

if __name__ == "__main__":

    label='0.005'
    edgeDict=pickle.load(open('edgeDict_'+label+'.dat', "rb"))
    nodeIndexDict=pickle.load(open('nodeIndexDict_'+label+'.dat', "rb"))
    attributeslist=pickle.load(open('attributeslist_'+label+'.dat', "rb"))
    refexFeature=pickle.load(open('refexFeature_'+label+'.dat', "rb"))
    authorDict_obj=pickle.load(open('./proces/authorDict_obj.dat', "rb"))

    clu=Clustering(edgeDict,nodeIndexDict,attributeslist,refexFeature,authorDict_obj)
    clu.clustering_preGraph()
