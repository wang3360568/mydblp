# -*- coding: utf-8 -*-
import os
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import util
import re
import csv
import time
import HTMLParser
import dblp
import cPickle as pickle
import collections
import numpy as np
sys.path.append('..')
from myclass.myobj import Paper,Person,Author
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

class MTNE_nocompany():
    # number of embedding dimensions
    m=32
    # number of sparse embedding dimensions
    p=128
    # number of clusters
    k=12

    # paremeter for the second order
    alpha=10
    lamda=0.1
    rho=1
    gamma=1

    epsilon=0.1
    t = 10000.


    def __init__(self,edgeDict,nodeIndexDict,isBinary=True):
        if isBinary:
            self.edgeDict=edgeDict
            self.nodeIndexDict=nodeIndexDict
        else:
            self.edgeDict=pickle.load(open(edgeDict, "rb"))
            self.nodeIndexDict=pickle.load(open(nodeIndexDict, "rb"))

        # total number of nodes of all timestamps
        self.q=0
        for key in self.nodeIndexDict:
            self.q+=len(nodeIndexDict[key])


    def MTNE(self):
        D=np.random.rand(self.p, self.m)

        Aprime_list=[]
        X_list=[]
        Xmask_list=[]
        U_list=[]
        B_list=[]

        # all sparse embeddings across all timestamps

        F=np.zeros((self.q,self.k))

        indexDict_local2global=collections.OrderedDict()
        indexDict_global2local=dict()
        globalIndex=0
        

        for key in self.edgeDict:

            A=self.edgeDict[key]

            X=self.initX(A,self.alpha)
            X=X/(X.max()-X.min())
            X_list.append(X)

            xmask=self.getMask(X)
            Xmask_list.append(xmask)

            # number of nodes in the current time
            n=A.shape[0]

            Aprime=np.random.rand(n, self.p)
            Aprime_list.append(Aprime)

            indexDict=dict()
            for i in range(n):
                indexDict[i]=globalIndex+i
                indexDict_global2local[globalIndex+i]=(key,i)
            indexDict_local2global[key]=indexDict

            globalIndex+=n

            U = np.random.rand(n, self.m)
            U_list.append(U)

            B = np.random.rand(n, self.m)
            B_list.append(B)
        
        loss_t1=1000000000.0

        print loss_t1
        loss_t = self.epsilon + loss_t1+1
        while abs(loss_t-loss_t1) >= self.epsilon:
        
            #% optimize each element in randomized sequence
            nita = 1. / np.sqrt(self.t)
            self.t = self.t + 1
            loss_t = loss_t1
            loss_t1=0.0

            counter=0

            for key in self.edgeDict:
                
                X=X_list[counter]
                U=U_list[counter]
                B=B_list[counter]
                Aprime=Aprime_list[counter]
                Xmask=Xmask_list[counter]

            # number of nodes in the current time

                LB=np.dot(Xmask*(np.dot(B,U.T)-X),U)+self.rho*B
                B=B-nita*LB
                B=self.chechnegtive(B,None,None)

                LU=np.dot(Xmask*(np.dot(B,U.T)-X),B)+self.lamda*(U-np.dot(Aprime,D))+self.rho*U
                U=U-nita*LU
                U=self.chechnegtive(U,None,None)

                p1=np.dot(Aprime,D)-U
                p2=np.dot(Aprime.T,p1)
                LD=self.lamda*p2+self.rho*D
                D=D-nita*LD

                LA=self.lamda*np.dot(np.dot(Aprime,D),D.T)+self.rho*Aprime
                Aprime=Aprime-nita*LA

                U_list[counter]=U
                B_list[counter]=B
                Aprime_list[counter]=Aprime

                loss_t1_part=self.lossfuction(X,B,U,Aprime,D)
                loss_t1+=loss_t1_part
                counter+=1

            E=self.concatenateMatrixInList(Aprime_list,0)

            simM=cosine_similarity(E)
            eignenVal,F=self.eigenVectorAndEigenValue(simM,self.k)

            up=np.dot(E[0],E[1].T)
            fm=np.linalg.norm(E[0])
            sm=np.linalg.norm(E[1])
            sval=up/(fm*sm)
            print sval
            print 'eignenVal: '+str(eignenVal)
            print F[0]
            print F[1]

            # LE=self.gamma*np.dot(F,F.T)
            # E=E-nita*LE

            # gIndex=0
            # for i in range(len(Aprime_list)):
            #     length=len(Aprime_list[i])
            #     Aprime_list[i]=E[gIndex:length]
            #     gIndex=gIndex+length

            loss_t1+=self.gamma*eignenVal


            if loss_t<loss_t1 and loss_t!=0:
                break
        # print loss_t
            print loss_t1
        return [U_list,Aprime_list,F]

    def lossfuction(self,X,B,U,A,D):

        # part1=0.5*(loss(AD,U.T,V,True)+loss(D,V.T,ND,False)+loss(A,U.T,NA,False))
        part1=0.5*(self.loss(X,B,U.T,True))
        part2=0.5*self.lamda*(self.loss(U,A,D,False))
        # part2=tau*(np.linalg.norm(U)+np.linalg.norm(V)+np.linalg.norm(NA)+np.linalg.norm(ND))
        part3=0.5*self.rho*(np.linalg.norm(U)+np.linalg.norm(B)+np.linalg.norm(A)+np.linalg.norm(D))

        print 'part1: '+ str(part1)
        print 'part2: '+ str(part2)
        print 'part3: '+ str(part3)

        return part1+part2+part3
    
    def loss(self,N,M,F,isnotdense):
        if not isnotdense:
            return np.linalg.norm(N-np.dot(M,F))
        else:
            newN=np.dot(M,F)
            return np.linalg.norm(N-self.getMask(N)*newN)

    def concatenateMatrixInList(self,matrixList,axis):
        if axis==0:
            E=np.empty([0,self.p])
        elif axis==1:
            E=np.empty([self.p,0])
        for i in range(len(matrixList)):
            E=np.concatenate((E, matrixList[i]), axis=axis)

        return E

    def initX(self,Ajen,alpha):
        secondOrder=cosine_similarity(Ajen)
        return Ajen+alpha*secondOrder

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

    def getMask(self,M):
        mask = np.zeros((M.shape[0], M.shape[1]))
        (index_i,index_j) = np.nonzero(M)
        for i in range(index_i.shape[0]):
            mask[index_i[i]][index_j[i]]=1
        # print mask
        return mask
    
    def chechnegtive(self,M,row,col):
        if row==None and col!=None:
            for i in range(M.shape[0]):
                if M[i,col]<0:
                    M[i,col]=0
        elif col==None and row!=None:
            for i in range(M.shape[1]):
                if M[row,i]<0:
                    M[row,i]=0
        elif col==None and row==None:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if M[i,j]<0:
                        M[i,j]=0
        return M
