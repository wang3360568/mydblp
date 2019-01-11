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

class MTNE_learnSimilarity_nocompany():
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


    def __init__(self,edgeDict,nodeIndexDict,attributeslist,refexFeature,isBinary=True):
        if isBinary:
            self.edgeDict=edgeDict
            self.nodeIndexDict=nodeIndexDict
            self.attributeslist=attributeslist
            self.refexFeature=refexFeature
        else:
            self.edgeDict=pickle.load(open(edgeDict, "rb"))
            self.nodeIndexDict=pickle.load(open(nodeIndexDict, "rb"))
            self.attributeslist=pickle.load(open(attributeslist, "rb"))
            self.refexFeature=pickle.load(open(refexFeature, "rb"))

        # total number of nodes of all timestamps
        self.q=0
        for key in self.nodeIndexDict:
            self.q+=len(nodeIndexDict[key])


    def Clustering_preGraph(self):

        # Author
        Aprime_list=[]

        F_list=[]
        # similarity local
        X_list=[]
        # refex features
        Y_list=[]

        F_big=np.zeros((self.q,self.k))

        R_big=np.zeros((self.q,self.re))

        indexDict_local2global=collections.OrderedDict()
        indexDict_global2local=dict()
        globalIndex=0
        index=0

        for key in self.edgeDict:

            A=self.edgeDict[key]

            X=self.initX(A)
            # X=X/(X.max()-X.min())
            

            n=A.shape[0]

            Y_list.append(self.refexFeature[key])

            Aprime=np.random.rand(n, self.p)
            # Aprime = np.zeros((n, self.p))
            Aprime_list.append(Aprime)

            indexDict=dict()
            for i in range(n):
                indexDict[i]=globalIndex+i
                indexDict_global2local[globalIndex+i]=(key,i)
            indexDict_local2global[key]=indexDict

            F = np.random.rand(n, self.m)
            # F = np.zeros((n, self.m))
            F_list.append(F)

            globalIndex+=n
            index+=1
        
        F_big=self.concatenateMatrixInList(F_list,self.m,0)

        R_big=self.concatenateMatrixInList(Y_list,self.re,0)
        R_big=MinMaxScaler().fit_transform(R_big)
        pca = PCA(n_components='mle', svd_solver='full')
        R_new=pca.fit_transform(R_big)  
        S=cosine_similarity(R_new)
        S=(S+1)/2
        np.savetxt('s.csv', S, delimiter=',')   
        simMD,simML=self.getLaplacian(S)


        return [Aprime_list,F_list]


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
    mtne=MTNE_learnSimilarity_nocompany(edgeDict,nodeIndexDict,attributeslist,refexFeature)
    a_list,f_list=mtne.MTNE()
    pickle.dump(a_list, open('A_list_withSim_'+label+'.dat', "wb"), True)
    pickle.dump(f_list, open('F_list_withSim_'+label+'.dat', "wb"), True)