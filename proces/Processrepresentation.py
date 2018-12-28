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
import cPickle as pickle
import collections
import numpy as np
sys.path.append('..')
from myclass.myobj import Paper,Person,Author
from sklearn.cluster import KMeans

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

if __name__ == "__main__":
    clu=Clustering('../u_list.dat','../a_list.dat','../F.dat','../edgeDict.dat','../nodeIndexDict.dat','./paperDict_obj.dat','./authorDict_obj.dat')
    clu.clustering(12)


