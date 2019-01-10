# -*- coding: utf-8 -*-
import os
import json
import sys
import matplotlib.pyplot as plt
import util
import re
import csv
import time
import HTMLParser
import cPickle as pickle
import collections
import numpy as np
sys.path.append('.')
from myclass.myobj import Paper,Person,Author
from sklearn.cluster import KMeans
import pandas as pd

from sklearn.manifold import TSNE
class Clustering:

    def __init__(self,u_path,a_path,f_path,edgeDict_file,nodeIndexDict_file,paperObjFile,authorObjFile):
        self.u_list=pickle.load(open(u_path, "rb"))
        self.a_list=pickle.load(open(a_path, "rb"))
        self.F=pickle.load(open(f_path, "rb"))

        self.edgeDict=pickle.load(open(edgeDict_file, "rb"))
        self.nodeIndexDict=pickle.load(open(nodeIndexDict_file, "rb"))

        self.paperObjDict=pickle.load(open(paperObjFile, "rb"))
        self.authorObjDict=pickle.load(open(authorObjFile, "rb"))

    def clustering(self,k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit_predict(self.F)
        gobalIndex=0
        outputList=[]
        for key in self.nodeIndexDict:
            keyList=self.nodeIndexDict[key]
            length=len(keyList)
            for i in range(length):
                clu=kmeans[gobalIndex+i]
                theKey=keyList[i]
                outputList.append([theKey,self.authorObjDict[theKey].name,key,clu])
            gobalIndex+=length

        util.write_csv_inlist('./temporal/authorCluster.csv',outputList)

class Clustering_new:

    def __init__(self,f_path,a_path,s_path,edgeDict_file,nodeIndexDict_file,paperObjFile,authorObjFile):
        self.f_list=pickle.load(open(f_path, "rb"))
        self.a_list=pickle.load(open(a_path, "rb"))
        self.S=pickle.load(open(s_path, "rb"))

        self.edgeDict=pickle.load(open(edgeDict_file, "rb"))
        self.nodeIndexDict=pickle.load(open(nodeIndexDict_file, "rb"))

        self.paperObjDict=pickle.load(open(paperObjFile, "rb"))
        self.authorObjDict=pickle.load(open(authorObjFile, "rb"))

    def getTotalKeyList(self):
        totalKeyList=[]
        for key in self.nodeIndexDict:
            keyList=self.nodeIndexDict[key]
            length=len(keyList)
            for i in range(length):
                totalKeyList.append(keyList[i]+'_'+str(key))
        return totalKeyList

    def writeSimilarity(self,totalKeyList):
        I = pd.Index(totalKeyList, name="rows")
        C = pd.Index(totalKeyList, name="cols")
        df = pd.DataFrame(data=self.S, index=I, columns=C)
        df.to_csv('simularity.csv')

    def writeAandF(self,totalKeyList,inputList,outputName):
        I = pd.Index(totalKeyList, name="rows")
        input_big=None
        for sig in inputList:
            if input_big is None:
                input_big=sig
            else:
                input_big=np.concatenate((input_big, sig), axis=0)
        df = pd.DataFrame(data=input_big, index=I)
        df.to_csv(outputName)
        return input_big

    
    def clustering(self,k,repr):
        kmeans = KMeans(n_clusters=k, random_state=0).fit_predict(repr)
        gobalIndex=0
        outputList=[]
        for key in self.nodeIndexDict:
            keyList=self.nodeIndexDict[key]
            length=len(keyList)
            for i in range(length):
                clu=kmeans[gobalIndex+i]
                theKey=keyList[i]
                outputList.append([theKey,self.authorObjDict[theKey].name,key,clu])
            gobalIndex+=length

        util.write_csv_inlist('./authorCluster.csv',outputList)
        return kmeans

class MyTSNE:
    def __init__(self,input,dim):
        self.input=input
        self.dim=dim
    
    def TSNETransfer(self):
        tsne = TSNE(n_components=self.dim, init='pca', random_state=0)
# t0 = time()
        X_tsne = tsne.fit_transform(self.input)
        return X_tsne

    def plot_embedding_scatter(self, X,legendlist, label,clsnum,sizelist=None,labellist=None,title=None):
        # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)

        if sizelist==None:
            for i in range(X.shape[0]):
                plt.scatter(X[i, 0], X[i, 1], label=legendlist[i],
                     color=plt.cm.Set1(float(label[i])/clsnum ),s=2)
                plt.text(X[i, 0], X[i, 1], legendlist[i],
                     color=plt.cm.Set1(float(label[i])/clsnum ),
                     fontdict={'weight': 'bold', 'size': 4})
                # print label[i]/clsnum

        else:
            for i in range(X.shape[0]):
                plt.scatter(X[i, 0], X[i, 1], label=legendlist[i],
                     color=plt.cm.Set1(label[i]/clsnum ),s=sizelist[i]*15
                     )
                # #999999 #984EA3
                #  #F781BF
                # if sizelist[i]<25:
                #     continue
                # plt.text(X[i, 0], X[i, 1], labellist[ylist[i]]['name'],
                #      color=plt.cm.Set1(label[i]/clsnum ),
                #      fontdict={'weight': 'bold', 'size': sizelist[i]/2})

        # plt.legend(scatterpoints=1)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.show()
        plt.savefig('plot_embedding_scatter_'+title+'.png',dpi=300)

if __name__ == "__main__":
    cluNum=16
    label='0.005'
    clu=Clustering_new('./F_list_withSim_'+label+'.dat','./A_list_withSim_'+label+'.dat','./S_company_'+label+'.dat','./edgeDict_'+label+'.dat','./nodeIndexDict_'+label+'.dat','./proces/paperDict_obj.dat','./proces/authorDict_obj.dat')
    totalKeyList=clu.getTotalKeyList()
    # clu.writeSimilarity(totalKeyList)
    a_big=clu.writeAandF(totalKeyList,clu.a_list,'a.csv')
    f_big=clu.writeAandF(totalKeyList,clu.f_list,'f.csv')
    cluResults=clu.clustering(cluNum,f_big)
    mt=MyTSNE(f_big,2)
    a_total_new=mt.TSNETransfer()
    mt.plot_embedding_scatter(a_total_new,totalKeyList,cluResults,cluNum,title=label)


