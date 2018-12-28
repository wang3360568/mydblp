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
import random
import HTMLParser
import cPickle as pickle
import collections
import numpy as np
sys.path.append('..')
from myclass.myobj import Paper,Person,Author
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.decomposition import PCA

class MTNE_learnSimilarity_nocompany():
    # number of embedding dimensions
    m=16
    # number of sparse embedding dimensions
    p=64
    # number of clusters
    k=12

    theta=5 # paremeter for the second order
    alpha=1 # for ||S||
    lamda=10 # for ||W||
    rho=0.001 # for tr(FLsF)
    gamma=1 # for ||D||
    epsilon=1 # for SI-1
    beta=10 # for ||DA-F||
    eta=10 # for ||I(S-f(X))||


    lamda_pgd=100

    t = 100000000.


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
        # dictionary
        D=np.random.rand(self.p, self.m)

        lambda_ff_list=[]

        # Author
        Aprime_list=[]
        # weight
        W_list=[]
        # dense vector
        F_list=[]
        # similarity local
        X_list=[]
        X_mask_list=[]


        # all sparse embeddings across all timestamps

        # F_big=np.zeros((self.q,self.k))
        X_big=np.zeros((self.q,self.q))
        X_mask_big=np.zeros((self.q,self.q))
        
        S=np.random.rand(self.q,self.q)

        indexDict_local2global=collections.OrderedDict()
        indexDict_global2local=dict()
        globalIndex=0
        

        for key in self.edgeDict:

            A=self.edgeDict[key]

            X=self.initX(A,self.theta)
            X=X/(X.max()-X.min())
            X_list.append(X)

            X_mask=np.zeros((self.q,self.q))
            # number of nodes in the current time
            n=A.shape[0]

            Aprime=np.random.rand(n, self.p)
            Aprime_list.append(Aprime)

            indexDict=dict()
            for i in range(n):
                indexDict[i]=globalIndex+i
                indexDict_global2local[globalIndex+i]=(key,i)
            indexDict_local2global[key]=indexDict

            for i in range(n):
                i_big=indexDict[i]
                for j in range(n):
                    j_big=indexDict[j]
                    X_big[i_big,j_big]=X[i,j]
                    X_mask[i_big,j_big]=1.
                    X_mask_big[i_big,j_big]=1.
            X_mask_list.append(X_mask)

            globalIndex+=n

            W = np.random.rand(n, self.p)
            W_list.append(W)

            F = np.random.rand(n, self.m)
            F_list.append(F)

            lambda_ff_list.append(random.random())
        
        F_big=self.concatenateMatrixInList(F_list,self.m,0)

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
                W=W_list[counter]
                F=F_list[counter]
                Aprime=Aprime_list[counter]
                indexDict=indexDict_local2global[key]
                lambda_ff=lambda_ff_list[counter]
                X_mask=X_mask_list[counter]

                P=self.getP(X_mask,F_big,X_big)

                n=X.shape[0]

                for i in range(n):
                    # for A
                    z=Aprime[i]-nita*(np.dot(np.dot(Aprime[i],W.T)-X[i],W)+self.beta*np.dot(np.dot(Aprime[i],D)-F[i],D.T))
                    update=np.maximum(np.zeros(np.shape(z)),np.abs(z)-nita*self.lamda_pgd)
                    Aprime[i]=np.sign(z)*update

                    # for F
                    lf_part1=self.beta*F[i]-np.dot(Aprime[i],D)
                    lf_part2=np.zeros(self.m)
                    i_big_index=indexDict[i]
                    for j in range(self.q):
                        lf_part2+=self.rho*(F[i]-F_big[j])*S[i_big_index,j]
                    
                    val1=np.dot(F[i],F_big.T)
                    val2=val1-np.ones(self.q)
                    val3=np.dot(val2,F_big)
                    lf_part3=0.01*val3
                    F[i]=F[i]-nita*(lf_part1+lf_part2+lf_part3)
                    F_big[i_big_index]=F[i]

                    # vec=np.dot(F[i],F_big.T)-np.ones(self.q)
                    # # print vec.shape
                    # lambda_ff=lambda_ff-nita*np.linalg.norm(vec)

                    # for S
                    ls=(S[i_big_index]-P[i_big_index])-self.epsilon*np.ones(self.q)-self.alpha*S[i_big_index]
                    S[i_big_index]=S[i_big_index]-nita*ls

                Aprime=self.chechnegtive(Aprime,None,None)

                LW=np.dot((np.dot(W,Aprime.T)-X),Aprime)+self.lamda*W
                W=W-nita*LW
                W=self.chechnegtive(W,None,None)

                p1=np.dot(Aprime,D)-F
                p2=self.beta*np.dot(Aprime.T,p1)
                LD=p2+self.gamma*D
                D=D-nita*LD

                W_list[counter]=W
                F_list[counter]=F
                Aprime_list[counter]=Aprime
                lambda_ff_list[counter]=lambda_ff

                loss_t1_part=self.lossfuction(X,W,Aprime,F,D)
                loss_t1+=loss_t1_part
                counter+=1
            
            
            # loss_last=self.laplacianLoss(simM,F)
            simMD,simML=self.getLaplacian(S)
            trval=np.trace(np.dot(np.dot(F_big.T,simML),F_big))
            gapval=np.linalg.norm(X_mask_big*(S-X_big))

            loss_t1+=self.rho*trval+self.eta*gapval

            if loss_t<loss_t1 and loss_t!=0:
                break
        # print loss_t
            print loss_t1
        return [Aprime_list,F_list,S]

    def lossfuction(self,X,W,A,F,D):

        # part1=0.5*(loss(AD,U.T,V,True)+loss(D,V.T,ND,False)+loss(A,U.T,NA,False))
        part1=0.5*(self.loss(X,W,A.T,False))
        part2=0.5*self.beta*(self.loss(F,A,D,False))
        # part2=tau*(np.linalg.norm(U)+np.linalg.norm(V)+np.linalg.norm(NA)+np.linalg.norm(ND))
        part3=0.5*(self.lamda*np.linalg.norm(W)+self.gamma*np.linalg.norm(D))

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

    def laplacianLoss(self,W,M):
        val=0.0
    # print M.shape
        for i in range(M.shape[0]):
            for j in range(i+1,M.shape[0]):
                val+=W[i][j]*np.linalg.norm(M[i]-M[j])
        return val
    def concatenateMatrixInList(self,matrixList,dim,axis):
        if axis==0:
            E=np.empty([0,dim])
        elif axis==1:
            E=np.empty([dim,0])
        for i in range(len(matrixList)):
            E=np.concatenate((E, matrixList[i]), axis=axis)

        return E
    
    def getP(self,Xmask,F,X_big):
        size=F.shape[0]
        Q=np.zeros((size,size))
        P=np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                Q[i,j]=np.linalg.norm(F[i]-F[j])

        P=(2*Xmask*X_big-self.rho*Q)/(2*Xmask+ np.full((size,size), self.alpha))
        return P
    
    def updateMatrixInList(self,E,matrixList):
        globalLen=0
        for i in range(len(matrixList)):
            shape_i=matrixList[i].shape[0]
            matrixList[i]=E[globalLen:globalLen+shape_i]
            globalLen=globalLen+shape_i
        return matrixList

    def initX(self,Ajen,alpha):
        secondOrder=np.dot(Ajen,Ajen)
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

    def getLaplacian(self,W):
        D = np.zeros((W.shape[0], W.shape[1]))
        L = np.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[1]):
            D[i][i]=np.sum(W[:,i])
        L=D-W
        return [D,L]
