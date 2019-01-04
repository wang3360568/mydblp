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
import math
import mxnet as mx
sys.path.append('.')
from myclass.myobj import Paper,Person,Author
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.decomposition import PCA
from buildNetwork_mxnet import InitNetworks

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
        D=mx.nd.random.uniform(0,1,shape=(self.p, self.m))

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

        # F_big=mx.nd.zeros((self.q,self.k))
        X_big=mx.nd.zeros((self.q,self.q))
        X_mask_big=mx.nd.zeros((self.q,self.q))
        
        S=mx.nd.random.uniform(0,1,shape=(self.q,self.q))

        indexDict_local2global=collections.OrderedDict()
        indexDict_global2local=dict()
        globalIndex=0

        for key in self.edgeDict:

            A=self.edgeDict[key]

            X=self.initX(A,self.theta)
            
            # X_np=X.asnumpy()
            # print X_np.max()
            xMax=mx.ndarray.max(X).asscalar()
            xMin=mx.ndarray.min(X).asscalar()
            # print xMax
            # print np.linalg.norm(X_np)
            # print mx.ndarray.norm(X).asscalar()
            X=X/(xMax-xMin)
            X_list.append(X)
            

            X_mask=mx.nd.zeros((self.q,self.q))
            # number of nodes in the current time
            n=A.shape[0]

            Aprime=mx.nd.random.uniform(0,1,shape=(n, self.p))
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

            W = mx.nd.random.uniform(0,1,shape=(n, self.p))
            W_list.append(W)

            F = mx.nd.random.uniform(0,1,shape=(n, self.m))
            F_list.append(F)

            lambda_ff_list.append(random.random())
        
        F_big=self.concatenateMatrixInList(F_list,self.m,0)

        loss_t1=1000000000.0

        print loss_t1
        loss_t = self.epsilon + loss_t1+1
        while abs(loss_t-loss_t1) >= self.epsilon:
        
            #% optimize each element in randomized sequence
            nita = 1. / math.sqrt(self.t)
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
                    print i
                    z=Aprime[i]-nita*(mx.ndarray.dot(mx.ndarray.dot(Aprime[i],W.T)-X[i],W)+self.beta*mx.ndarray.dot(mx.ndarray.dot(Aprime[i],D)-F[i],D.T))
                    update=mx.ndarray.maximum(mx.nd.zeros(z.shape),mx.ndarray.abs(z)-nita*self.lamda_pgd)
                    Aprime[i]=mx.ndarray.sign(z)*update

                    # for F
                    lf_part1=self.beta*F[i]-mx.ndarray.dot(Aprime[i],D)
                    lf_part2=mx.nd.zeros(self.m)
                    i_big_index=indexDict[i]
                    for j in range(self.q):
                        lf_part2+=self.rho*(F[i]-F_big[j])*S[i_big_index,j]
                    
                    val1=mx.ndarray.dot(F[i],F_big.T)
                    val2=val1-mx.nd.ones(self.q)
                    val3=mx.ndarray.dot(val2,F_big)
                    lf_part3=0.01*val3
                    F[i]=F[i]-nita*(lf_part1+lf_part2+lf_part3)
                    F_big[i_big_index]=F[i]

                    # vec=mx.ndarray.dot(F[i],F_big.T)-np.ones(self.q)
                    # # print vec.shape
                    # lambda_ff=lambda_ff-nita*np.linalg.norm(vec)

                    # for SF
                    ls=(S[i_big_index]-P[i_big_index])-self.epsilon*mx.nd.ones(self.q)-self.alpha*S[i_big_index]
                    S[i_big_index]=S[i_big_index]-nita*ls

                print 'done'
                Aprime=self.chechnegtive(Aprime,None,None)

                LW=mx.ndarray.dot((mx.ndarray.dot(W,Aprime.T)-X),Aprime)+self.lamda*W
                W=W-nita*LW
                W=self.chechnegtive(W,None,None)

                p1=mx.ndarray.dot(Aprime,D)-F
                p2=self.beta*mx.ndarray.dot(Aprime.T,p1)
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
            trval=self.mytarce(mx.ndarray.dot(mx.ndarray.dot(F_big.T,simML),F_big))
            gapval=mx.ndarray.norm(X_mask_big*(S-X_big)).asscalar()

            loss_t1+=self.rho*trval+self.eta*gapval

            if loss_t<loss_t1 and loss_t!=0:
                break
        # print loss_t
            print loss_t1
        return [Aprime_list,F_list,S]

    def lossfuction(self,X,W,A,F,D):

        part1=0.5*(self.loss(X,W,A.T))
        part2=0.5*self.beta*(self.loss(F,A,D))
        part3=0.5*(self.lamda*mx.ndarray.norm(W).asscalar()+self.gamma*mx.ndarray.norm(D).asscalar())

        print 'part1: '+ str(part1)
        print 'part2: '+ str(part2)
        print 'part3: '+ str(part3)

        return part1+part2+part3
    
    def loss(self,N,M,F):
        return mx.ndarray.norm(N-mx.ndarray.dot(M,F)).asscalar()


    def laplacianLoss(self,W,M):
        val=0.0
    # print M.shape
        for i in range(M.shape[0]):
            for j in range(i+1,M.shape[0]):
                val+=W[i][j]*mx.ndarray.norm(M[i]-M[j]).asscalar()
        return val
    def concatenateMatrixInList(self,matrixList,dim,axis):
        E=None
        for i in range(len(matrixList)):
            m=matrixList[i]
            # print m.shape
            if E is None:
                E=m.copy()
            else:
                E=mx.ndarray.concat(E, m, dim=axis)

        return E
    
    def getP(self,Xmask,F,X_big):
        size=F.shape[0]
        Q=mx.nd.zeros((size,size))
        P=mx.nd.zeros((size,size))
        for i in range(size):
            for j in range(i+1,size):
                Q[i,j]=mx.ndarray.norm(F[i]-F[j]).asscalar()
                Q[j,i]=Q[i,j]

        P=(2*Xmask*X_big-self.rho*Q)/(2*Xmask+ mx.nd.full((size,size), self.alpha))
        return P
    
    def updateMatrixInList(self,E,matrixList):
        globalLen=0
        for i in range(len(matrixList)):
            shape_i=matrixList[i].shape[0]
            matrixList[i]=E[globalLen:globalLen+shape_i]
            globalLen=globalLen+shape_i
        return matrixList

    def initX(self,Ajen,alpha):
        firstOrder=mx.nd.array(Ajen)
        secondOrder=mx.ndarray.dot(firstOrder,firstOrder)
        return firstOrder+alpha*secondOrder
    
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
        D = mx.nd.zeros((W.shape[0], W.shape[1]))
        L = mx.nd.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[1]):
            D[i][i]=mx.ndarray.sum(W[:,i]).asscalar()
        L=D-W
        return [D,L]

    def mytarce(self,M):
        trval=0.0
        for i in range(M.shape[0]):
            trval+=M[i,i]
        return trval


if __name__ == "__main__":

    
    gpu_device=mx.gpu(int(sys.argv[1]))
    
    with mx.Context(gpu_device):
        init=InitNetworks('./proces/paperDict_obj.dat','./proces/authorDict_obj.dat','./proces/yearCount.dat')
        nodedict=init.selectAuthors()

        edgeDict,nodeIndexDict=init.generateEdge()

        mtne=MTNE_learnSimilarity_nocompany(edgeDict,nodeIndexDict)
        a_list,f_list,s=mtne.MTNE()
        pickle.dump(a_list, open('A_list.dat', "wb"), True)
        pickle.dump(f_list, open('F_list.dat', "wb"), True)
        pickle.dump(s, open('S.dat', "wb"), True)