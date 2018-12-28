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
sys.path.append('../')
from myclass.myobj import Paper,Person,Author,Person_node
# sys.path.append('../snap_jure')
import snap
import math
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

class setupNetwork:
    def __init__(self,edgeDict_file,nodeIndexDict_file,objFile):
        self.edgeDict=pickle.load(open(edgeDict_file, "rb"))
        self.nodeIndexDict=pickle.load(open(nodeIndexDict_file, "rb"))

        self.objDict=pickle.load(open(objFile, "rb"))

    def buildNetwork(self):
        graphList=[]
        
        for key in self.nodeIndexDict:
            myGraph = snap.TNEANet.New()
            keyList=self.nodeIndexDict[key]
            length=len(keyList)
            for i in range(length):
                theKey=keyList[i]
                nid=myGraph.AddNode(i)
                myGraph.AddStrAttrDatN(nid, theKey, 'key')
            
            # there is only the first order!!!!!!!!!!!!!!!!!!
            A=self.edgeDict[key]
            # B=np.dot(A,A)
            B=np.dot(A,np.dot(A,A))
            # B_sim=cosine_similarity(A)
            C=A+B
            for i in range(length):
                for j in range(length):
                    if i!=j and C[i,j]>0:
                        eid=myGraph.AddEdge(i, j)
                        myGraph.AddFltAttrDatE(eid, A[i,j], 'weigth')

            print str(key)+'-original: '+str(myGraph.GetEdges())+' '+str(myGraph.GetNodes())
            MxWcc = snap.GetMxWcc(myGraph)
            print str(key)+'-mxWcc: '+str(MxWcc.GetEdges())+' '+str(MxWcc.GetNodes())


            graphList.append(myGraph)


if __name__ == "__main__":
    sN=setupNetwork('../edgeDict.dat','../nodeIndexDict.dat','./paperDict_obj.dat')
    sN.buildNetwork()