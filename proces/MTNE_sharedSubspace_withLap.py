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

    r=226#199

    lamda=0.02 # for ||WW-I||
    rho=0.5 # for||WY-MP||
    theta=0.1 # tr(PLP)
    epsilon=1 # 

    alpha=30 # for ||S||
    gamma=0.2 # for ||D||
    # epsilon=30 # for SI-1
    beta=10 # for ||DA-F||
    eta=100 # for ||I(S-f(X))||
    mu=2 # for ||Y-AM||
    zeta=0.01 # for ||FF-I||

    lamda_pgd=100

    t = 1000000000.


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
        M=np.random.rand(self.p, self.m)
        W=np.random.rand(self.r, self.p)
        I=np.eye(self.p)

        M_list=[]
        P_list=[]

        # refex features
        Y_list=[]
        L_list=[]

        indexDict_local2global=collections.OrderedDict()
        indexDict_global2local=dict()
        globalIndex=0
        index=0

        for key in self.edgeDict:

            A=self.edgeDict[key]
            X=self.initX(A)
            L=self.getLaplacian(X)
            L_list.append(L)

            n=self.refexFeature[key].shape[0]
            Y=MinMaxScaler().fit_transform(self.refexFeature[key])
            Y_list.append(Y)

            Mprime=np.random.rand(self.r, self.m)

            M_list.append(Mprime)

            indexDict=dict()
            for i in range(n):
                indexDict[i]=globalIndex+i
                indexDict_global2local[globalIndex+i]=(key,i)
            indexDict_local2global[key]=indexDict

            P = np.random.rand(n, self.m)
            P_list.append(P)

            globalIndex+=n
            index+=1

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
                
                Y=Y_list[counter]
                L=L_list[counter]
                P=P_list[counter]
                Mprime=M_list[counter]
                indexDict=indexDict_local2global[key]

                n=Y.shape[0]

                # for Mprime
                lmp_p1=np.dot(P,Mprime.T)-Y
                lmp_p2=np.dot(lmp_p1.T,P)
                Mprime=Mprime-nita*lmp_p2
                Mprime=self.chechnegtive(Mprime,None,None)

                # for M self.rho
                lm_p1=np.dot(P,M.T)-np.dot(Y,W)
                lm_p2=np.dot(lm_p1.T,P)
                M=M-nita*self.rho*lm_p2
                M=self.chechnegtive(M,None,None)

                # for W 
                lw_p1=np.dot(Y,W)-np.dot(P,M.T)
                lw_p2=np.dot(Y.T,lw_p1)
                lw_p3=np.dot(W.T,W)-I
                lw_p4=np.dot(W,lw_p3)
                W=W-nita*(self.rho*lw_p2+self.lamda*lw_p4)

                # for P
                lp_p1=np.dot(P,Mprime.T)-Y
                lp_p2=np.dot(lp_p1,Mprime)
                lp_p3=np.dot(P,M.T)-np.dot(Y,W)
                lp_p4=np.dot(lp_p3,M)
                lp_p5=np.dot(L,P)
                P=P-nita*(lp_p2+self.rho*lp_p4+self.theta*lp_p5)
                P=self.chechnegtive(P,None,None)

                P_list[counter]=P
                M_list[counter]=Mprime

                loss_t1_part=self.lossfuction(Y,Mprime,P,M,W,L)
                loss_t1+=loss_t1_part
                counter+=1

            if loss_t<loss_t1 and loss_t!=0:
                print 'done'
                break
            print loss_t1
        print 'done'
        return P_list

    def lossfuction(self,Y,Mprime,P,M,W,L):

        part1=0.5*self.loss(Y,P,Mprime.T,False)
        part2=0.5*self.rho*(self.loss(np.dot(Y,W),P,M.T,False))
        part3=self.theta*np.trace(np.dot(np.dot(P.T,L),P))
        # part4=0.5*(self.gamma*np.linalg.norm(D))

        print 'part1: '+ str(part1)
        print 'part2: '+ str(part2)
        print 'part3: '+ str(part3)
        # print 'part4: '+ str(part4)

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
        # firstOrder=MinMaxScaler().fit_transform(Ajen)
        # firstOrder=self.normalize(Ajen)
        # secondOrder=np.dot(Ajen,Ajen)
        # thirdOrder=np.dot(Ajen,secondOrder)
        # secondOrder=secondOrder/secondOrder.max()  
        # thirdOrder=thirdOrder/thirdOrder.max()
        # simM=firstOrder+secondOrder+thirdOrder
        # # simM=(firstOrder+secondOrder)/2.0
        # simM=simM/simM.max()
        firstD=np.power(np.sum(Ajen,axis=0),-1)
        firstD_norm=np.diag(firstD)
        firstOrder=np.dot(firstD_norm,Ajen)
        Aj2=np.dot(Ajen,Ajen)
        secondD=np.power(np.sum(Aj2,axis=0),-1)
        secondD_norm=np.diag(secondD)
        secondOrder=np.dot(secondD_norm,Aj2)
        simM=(firstOrder+secondOrder)
        return simM
    
    def normalize(self,A):
        maxval=np.percentile(A,99)
        if maxval==0:
            N=A/1.0
        else:
            N=A/np.percentile(A,99)
        N= np.where(N > 1, 1, N)
        return N

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
        return L

if __name__ == "__main__":

    label='0.005'
    edgeDict=pickle.load(open('edgeDict_'+label+'.dat', "rb"))
    nodeIndexDict=pickle.load(open('nodeIndexDict_'+label+'.dat', "rb"))
    attributeslist=pickle.load(open('attributeslist_'+label+'.dat', "rb"))
    refexFeature=pickle.load(open('refexFeature_'+label+'.dat', "rb"))
    mtne=MTNE_learnSimilarity_nocompany(edgeDict,nodeIndexDict,attributeslist,refexFeature)
    p_list=mtne.MTNE()
    pickle.dump(p_list, open('P_list_sharedSpace_withLap_'+label+'.dat', "wb"), True)