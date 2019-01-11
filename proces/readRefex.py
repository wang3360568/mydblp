# -*- coding: utf-8 -*-
import os
import json
import sys
import collections
import util
import re
import csv
import time
import HTMLParser
import cPickle as pickle
sys.path.append('.')
from myclass.myobj import Paper,Person,Author,Person_node
import snap
import math
import collections
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.cluster import KMeans

class ReadRefex():
    
    def __init__(self,fileFolder,start,end,inOrderList):
        self.fileFolder=fileFolder
        self.start=start
        self.end=end
        self.inOrderList=inOrderList
    
    def readfile(self):
        refexFeatures=collections.OrderedDict()
        for year in range(self.start,self.end+1):
            contentDict=dict()
            fp=open(self.fileFolder+str(year)+'-featureValues.csv','r')
            while 1:
                line = fp.readline()
                if not line:
                    break
                content=line.strip().split(',')
                numericContent= map(float, content[1:])
                contentDict[content[0]]=numericContent
            keylist=self.inOrderList[year]
            outputlist=[]
            for key in keylist:
                outputlist.append(contentDict[key])
            farray=np.array(outputlist)
            refexFeatures[year]=farray
        return refexFeatures

    def testCluster(self,refexFeature):


        pca = PCA(n_components=150, svd_solver='full')
        X_150=pca.fit_transform(refexFeature)
        pca = PCA(n_components=100, svd_solver='full')
        X_100=pca.fit_transform(refexFeature)
        pca = PCA(n_components=50, svd_solver='full')
        X_50=pca.fit_transform(refexFeature)
        pca = PCA(n_components=20, svd_solver='full')
        X_20=pca.fit_transform(refexFeature)
        pca = PCA(n_components=10, svd_solver='full')
        X_10=pca.fit_transform(refexFeature)
        pca = PCA(n_components=2, svd_solver='full')
        X=pca.fit_transform(refexFeature)

        S1=cosine_similarity(refexFeature)
        S2=cosine_similarity(X_150)
        S3=cosine_similarity(X_100)
        S4=cosine_similarity(X_50)
        S5=cosine_similarity(X_20)
        S6=cosine_similarity(X_10)
        S7=cosine_similarity(X)


        # random_state = 170
        # y_pred = KMeans(n_clusters=6, random_state=random_state).fit_predict(r_new)

        # plt.subplot(221)
        # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        # plt.title("")
        # plt.show()

        # return r_new
    def filterFeatureName(self,folder):
        featureNameSet=set()
        for year in range(self.start,self.end+1):
            fp=open(folder+str(year)+'-featureNames.csv','r')
            while 1:
                line = fp.readline()
                if not line:
                    break
                content=line.strip().split(',')
                for val in content:
                    featureNameSet.add(val)

        output=''
        for key in featureNameSet:
            output+=key+','
        print len(featureNameSet)
        util.write_csv_inlist(folder+'out-featureNames.csv',[output])


if __name__ == "__main__":
    lowerVal=0.001
    nodeIndexDict=pickle.load(open('nodeIndexDict_'+str(lowerVal)+'.dat', "rb"))
    refex=ReadRefex('E:\\data\\dblp\\mydata\\refexfeatures_'+str(lowerVal)+'\\',2005,2006,nodeIndexDict)
    featureSet=refex.readfile()
    for fea in featureSet:
        refex.testCluster(featureSet[fea])

    # refex.filterFeatureName('D:\\code\\Refex_java\\featureName\\')
