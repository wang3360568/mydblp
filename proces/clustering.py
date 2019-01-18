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

    area=['DM','ML','IR','DB']


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

    def clustering_useFeature(self,f_list):
        i=0
        hardLabelDict,softLabelDict=self.getLabel()
        for key in self.edgeDict:
            groundTrues=hardLabelDict[key]
            clusterNum=12
            A=self.edgeDict[key]
            nt=NetworkTool()
            nt.initNetwork(A,nodeIndexDict[key])

            F=f_list[i]
            
            # X=self.initX(A)
            # labels = spectral_clustering(A, n_clusters=clusterNum, eigen_solver='arpack')
            # counter=self.counter(labels,clusterNum)
            S=cosine_similarity(F)
            S=(S+1.0)/2.0
            labels = spectral_clustering(S, n_clusters=clusterNum, eigen_solver='arpack')
            counter=self.counter(labels,clusterNum)
            nmi_sc=self.NMI(labels.tolist(),groundTrues,clusterNum)
            print nmi_sc

            # self.draw(nodeIndexDict[key],nt,labels,str(key)+'_spectral_'+str(clusterNum)+'.png')
            # self.output(nodeIndexDict[key],labels,str(key)+'_spectral')

            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(F)
            labels_km=kmeans.labels_.tolist()
            counter=self.counter(labels_km,clusterNum)
            nmi_km=self.NMI(labels_km,groundTrues,clusterNum)
            # nmi_km_sk=normalized_mutual_info_score(groundTrues,labels_km)
            print nmi_km

            pca = PCA(n_components=clusterNum, svd_solver='full')
            F_pca=pca.fit_transform(F)
            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(F_pca)
            labels_km=kmeans.labels_.tolist()
            counter=self.counter(labels_km,clusterNum)
            nmi_km=self.NMI(labels_km,groundTrues,clusterNum)
            # nmi_km_sk=normalized_mutual_info_score(groundTrues,labels_km)
            print nmi_km

            # self.draw(nodeIndexDict[key],nt,labels_km,str(key)+'_kmean_'+str(clusterNum)+'.png')
            # self.output(nodeIndexDict[key],labels_km,str(key)+'_kmean')
            i+=1

        # self.perClus('D:\\code\\deepwalk\\2013_dw_output.txt',2013,clusterNum,groundTrues,nt,'dw')
        # self.perClus('D:\\code\\LINE\\windows\\2013_output.txt',2013,clusterNum,groundTrues,nt,'line2')
        # self.perClus('D:\\code\\LINE\\windows\\2013_output_1.txt',2013,clusterNum,groundTrues,nt,'line1')
    
    def perClus(self,embeddingFile,year,clusterNum,groundTrues,nt,name):
        lineEmbedding=self.readLineAndDW(embeddingFile,year)
        kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(lineEmbedding)
        labels_km=kmeans.labels_.tolist()
        counter=self.counter(labels_km,clusterNum)
        nmi_km=self.NMI(labels_km,groundTrues,clusterNum)
        nmi_km_sk=normalized_mutual_info_score(groundTrues,labels_km)

        # self.draw(nodeIndexDict[year],nt,labels_km,str(year)+name+str(clusterNum)+'.png')
        # self.output(nodeIndexDict[year],labels_km,str(year)+name)

        print nmi_km
        print nmi_km_sk
    
    def clustering_preGraph(self):

        hardLabelDict,softLabelDict=self.getLabel()

        for key in self.edgeDict:
            groundTrues=hardLabelDict[key]
            clusterNum=12

            A=self.edgeDict[key]
            nt=NetworkTool()
            nt.initNetwork(A,nodeIndexDict[key])
            X=self.initX(A)
            labels_ajen = spectral_clustering(X, n_clusters=clusterNum, eigen_solver='arpack')
            nmi_sc=self.NMI(labels_ajen.tolist(),groundTrues,clusterNum)
            print nmi_sc
            # counter=self.counter(labels,clusterNum)

            Y=self.refexFeature[key]
            pca = PCA(n_components=50, svd_solver='full')
            Y_50=pca.fit_transform(Y)
            S=cosine_similarity(Y_50)
            S=(S+1.0)/2.0
            labels = spectral_clustering( S, n_clusters=clusterNum, eigen_solver='arpack')
            counter=self.counter(labels,clusterNum)
            nmi_sc=self.NMI(labels.tolist(),groundTrues,clusterNum)
            print nmi_sc

            # self.draw(nodeIndexDict[key],nt,labels,str(key)+'_spectral_'+str(clusterNum)+'.png')
            # self.output(nodeIndexDict[key],labels,str(key)+'_spectral')

            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(Y)
            labels_km=kmeans.labels_.tolist()
            counter=self.counter(labels_km,clusterNum)
            nmi_km=self.NMI(labels_km,groundTrues,clusterNum)
            print nmi_km

            # self.draw(nodeIndexDict[key],nt,labels_km,str(key)+'_kmean_'+str(clusterNum)+'.png')
            # self.output(nodeIndexDict[key],labels_km,str(key)+'_kmean')
        



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

    def getLabel(self):
        hardLabelDict=collections.OrderedDict()
        softLabelDict=collections.OrderedDict()
        for key in self.nodeIndexDict:
            hardLabelList=[]
            softLabelList=[]
            keylist=self.nodeIndexDict[key]
            for authorKey in keylist:
                theAuthor=self.authorObjDict[authorKey]
                statDict=theAuthor.areaDict[key]
                hardLabelList.append(self.getHardLabel(statDict))
                softLabelList.append(self.getSoftLabel(statDict))
            hardLabelDict[key]=hardLabelList
            softLabelDict[key]=softLabelList

        return hardLabelDict,softLabelDict
            

    def getHardLabel(self,statDict):
        maxIndex=-1
        theMax=-1
        counter=0
        for val in self.area:
            theVal=statDict[val]
            if theVal>theMax:
                theMax=theVal
                maxIndex=counter
            counter+=1
        return maxIndex
    
    def getSoftLabel(self,statDict):
        total=0.0
        valList=[]
        for val in self.area:
            theVal=statDict[val]
            total+=theVal
            valList.append(theVal)
        labelArray=np.array(valList)
        labelArray=labelArray/total
        return labelArray

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

    label='0.005'
    edgeDict=pickle.load(open('edgeDict_'+label+'.dat', "rb"))
    nodeIndexDict=pickle.load(open('nodeIndexDict_'+label+'.dat', "rb"))
    attributeslist=pickle.load(open('attributeslist_'+label+'.dat', "rb"))
    refexFeature=pickle.load(open('refexFeature_'+label+'.dat', "rb"))
    authorDict_obj=pickle.load(open('authorDict_obj.dat', "rb"))

    clu=Clustering(edgeDict,nodeIndexDict,attributeslist,refexFeature,authorDict_obj)
    clu.clustering_preGraph()
    # f_list=pickle.load(open('P_list_sharedSpace_dw_'+label+'.dat', "rb"))
    # clu.clustering_useFeature(f_list)
