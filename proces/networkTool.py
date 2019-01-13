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
# sys.path.append('../snap_jure')
import snap
import math
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

class NetworkTool:

    def initNetwork(self,Ajen,keyList):

        self.Ajen=Ajen
        self.keyList=keyList
        self.myGraph = snap.TNEANet.New()
        self.nid2id=dict()
        self.id2nid=dict()

        length=len(keyList)
        for i in range(length):
            theKey=keyList[i]
            nid=self.myGraph.AddNode(i)
            self.myGraph.AddStrAttrDatN(nid, theKey, 'key')
            self.nid2id[nid]=theKey
            self.id2nid[theKey]=nid


        self.outputList=[]
        for i in range(length):
            for j in range(i+1,length):
                if Ajen[i,j]>0:
                    eid=self.myGraph.AddEdge(i, j)
                    self.myGraph.AddFltAttrDatE(eid, Ajen[i,j], 'weigth')
                    # eid=self.myGraph.AddEdge(j, i)
                    # self.myGraph.AddFltAttrDatE(eid, Ajen[j,i], 'weigth')
                    self.outputList.append([keyList[i],keyList[j], Ajen[i,j]])

        print '-original: '+str(self.myGraph.GetEdges())+' '+str(self.myGraph.GetNodes())
        self.MxWcc = snap.GetMxWcc(self.myGraph)
        print '-mxWcc: '+str(self.MxWcc.GetEdges())+' '+str(self.MxWcc.GetNodes())

    def toCSV(self,filename):
        util.write_csv_inlist(filename,self.outputList)


    def drawGraph(self,labelDict,theGraph,graphName):

        labels = snap.TIntStrH()
        for NI in theGraph.Nodes():
            thekey=self.nid2id[NI.GetId()]
            labels[NI.GetId()] = str(labelDict[thekey])
        snap.DrawGViz(theGraph, snap.gvlSfdp, graphName, " ", labels)
            
    def fromGraphtoList(self,theGraph):
        keyList_new=[]
            
        for node in theGraph.Nodes():
            keyList_new.append(self.keyList[int(node.GetId())])
            
            ajmatrix=np.zeros((theGraph.GetNodes(),theGraph.GetNodes()))

            counter_out=0
            for node_out in theGraph.Nodes():
                counter_in=0
                for node_in in theGraph.Nodes(): 
                    ajmatrix[counter_out,counter_in]=self.Ajen[int(node_out.GetId()),int(node_in.GetId())]
                    counter_in+=1
                counter_out+=1

        return ajmatrix, keyList_new

    def getNodeAttributes(self,UGraph):

        attriList=[]
        for index in range(UGraph.GetNodes()):
            nodelist=[]
            attriList.append(nodelist)
            
            #page rank
        PRankH = snap.TIntFltH()
        snap.GetPageRank(UGraph, PRankH)
        counter=0
        for item in PRankH:
            attriList[counter].append(PRankH[item])
            counter+=1
            #HIN
        counter=0
        NIdHubH = snap.TIntFltH()
        NIdAuthH = snap.TIntFltH()
        snap.GetHits(UGraph, NIdHubH, NIdAuthH)
        for item in NIdHubH:
            attriList[counter].append(NIdHubH[item])
            attriList[counter].append(NIdAuthH[item])
            counter+=1

            # Betweenness Centrality 
        counter=0
        Nodes = snap.TIntFltH()
        Edges = snap.TIntPrFltH()
        snap.GetBetweennessCentr(UGraph, Nodes, Edges, 1.0)
        for node in Nodes:
            attriList[counter].append(Nodes[node])
            counter+=1

            # closeness centrality 
        counter=0
        for NI in UGraph.Nodes():
            CloseCentr = snap.GetClosenessCentr(UGraph, NI.GetId())
            attriList[counter].append(CloseCentr)
            counter+=1

            # farness centrality 
        counter=0
        for NI in UGraph.Nodes():
            FarCentr = snap.GetFarnessCentr(UGraph, NI.GetId())
            attriList[counter].append(FarCentr)
            counter+=1

            # node eccentricity
        counter=0
        for NI in UGraph.Nodes():
            attriList[counter].append(snap.GetNodeEcc(UGraph, NI.GetId(), True))
            counter+=1

        atrriMarix=np.array(attriList)

        return atrriMarix

if __name__ == "__main__":
    # sN=SetupNetwork('./edgeDict_0.005.dat','./nodeIndexDict_0.005.dat','./proces/paperDict_obj.dat',isBin=False)
    # sN.buildNetwork()
    # attributeslist=sN.getNodeAttributes()
    print 'done'