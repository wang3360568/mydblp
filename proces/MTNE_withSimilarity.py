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


    def MTNE(self):
        # dictionary
        D=np.random.rand(self.p, self.m)

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
            X_list.append(X)

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
                F=F_list[counter]
                Aprime=Aprime_list[counter]
                indexDict=indexDict_local2global[key]

                n=X.shape[0]

                for i in range(n):

                    # for F
                    lf_part1=np.dot(np.dot(F[i],F.T)-X[i],F)
                    lf_part2=self.beta*(F[i]-np.dot(Aprime[i],D))
                    i_big_index=indexDict[i]
                    lf_part3=0.0
                    for j in range(self.q):
                        lf_part3+=self.rho*(F[i]-F_big[j])*S[i_big_index,j]
                    
                    # lf_part4=self.lamda*F[i]
                    F[i]=F[i]-nita*(lf_part1+lf_part2+lf_part3)
                    F_big[i_big_index]=F[i]

                F=self.chechnegtive(F,None,None)

                Zpart1=self.beta*np.dot(np.dot(Aprime,D)-F,D.T)
                Z=Aprime-nita*Zpart1
                Update=np.maximum(np.zeros(np.shape(Z)),np.abs(Z)-nita*self.lamda_pgd)
                Aprime=np.sign(Z)*Update

                p1=np.dot(Aprime,D)-F
                p2=self.beta*np.dot(Aprime.T,p1)
                LD=p2+self.gamma*D
                D=D-nita*LD

                F_list[counter]=F
                Aprime_list[counter]=Aprime

                loss_t1_part=self.lossfuction(X,Aprime,F,D,F_big,simML)
                loss_t1+=loss_t1_part
                counter+=1

            if loss_t<loss_t1 and loss_t!=0:
                break
            print loss_t1
        np.savetxt('d.csv', D, delimiter=',')
        return [Aprime_list,F_list]

    def lossfuction(self,X,A,F,D,F_big,simML):

        part1=0.5*self.loss(X,F,F.T,False)
        part2=0.5*self.beta*(self.loss(F,A,D,False))
        part3=0.5*self.rho*np.trace(np.dot(np.dot(F_big.T,simML),F_big))
        part4=0.5*(self.gamma*np.linalg.norm(D))

        print 'part1: '+ str(part1)
        print 'part2: '+ str(part2)
        print 'part3: '+ str(part3)
        print 'part4: '+ str(part4)

        return part1+part2+part3+part4
    
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
    
    def getP(self,Xmask,F,Sim_big):
        size=F.shape[0]
        Q=np.zeros((size,size))
        P=np.zeros((size,size))
        for i in range(size):
            for j in range(i+1,size):
                Q[i,j]=np.linalg.norm(F[i]-F[j])
                Q[j,i]=Q[i,j]

        P=(2*self.eta*Xmask*Sim_big-self.rho*Q)/(2*Xmask+ np.full((size,size), self.alpha))
        return P
    
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