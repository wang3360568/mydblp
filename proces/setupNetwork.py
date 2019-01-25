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

class SetupNetwork:
    def __init__(self,edgeDict,nodeIndexDict,obj,isBin=True):

        if isBin:
            self.edgeDict=edgeDict
            self.nodeIndexDict=nodeIndexDict
            self.objDict=obj
        else:
            self.edgeDict=pickle.load(open(edgeDict, "rb"))
            self.nodeIndexDict=pickle.load(open(nodeIndexDict, "rb"))
            self.objDict=pickle.load(open(obj, "rb"))

        self.graphList=[]

    def buildNetwork(self):
        edgeDict_new=collections.OrderedDict()
        edgeIndexDict_new=collections.OrderedDict()
        
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
            # B=np.dot(A,np.dot(A,A))
            # B_sim=cosine_similarity(A)
            C=A
            outputList=[]
            outputList_line=[]
            outputList_dw=[]
            for i in range(length):
                for j in range(i+1,length):
                    if C[i,j]>0:
                        eid=myGraph.AddEdge(i, j)
                        myGraph.AddFltAttrDatE(eid, A[i,j], 'weigth')
                        eid=myGraph.AddEdge(j, i)
                        myGraph.AddFltAttrDatE(eid, A[j,i], 'weigth')
                        outputList.append([keyList[i],keyList[j], A[i,j]])
                        outputList_line.append([keyList[j],keyList[i], A[j,i]])
                        outputList_line.append([keyList[i],keyList[j], A[i,j]])
                        outputList_dw.append([keyList[i],keyList[j]])
                        outputList_dw.append([keyList[j],keyList[i]])

            util.write_csv_inlist(str(key)+'.csv',outputList)
            util.write_csv_inlist(str(key)+'_line.txt',outputList_line)
            util.write_csv_inlist(str(key)+'_dw.txt',outputList_dw)

            print str(key)+'-original: '+str(myGraph.GetEdges())+' '+str(myGraph.GetNodes())
            MxWcc = snap.GetMxWcc(myGraph)
            print str(key)+'-mxWcc: '+str(MxWcc.GetEdges())+' '+str(MxWcc.GetNodes())

            # labels = snap.TIntStrH()
            # for NI in MxWcc.Nodes():
            #     labels[NI.GetId()] = str(NI.GetId())
            # snap.DrawGViz(MxWcc, snap.gvlSfdp, './graph/'+str(key)+".gif", " ", labels)
            
            keyList_new=[]
            
            for node in MxWcc.Nodes():
                keyList_new.append(keyList[int(node.GetId())])
            
            ajmatrix=np.zeros((MxWcc.GetNodes(),MxWcc.GetNodes()))

            counter_out=0
            for node_out in MxWcc.Nodes():
                counter_in=0
                for node_in in MxWcc.Nodes(): 
                    ajmatrix[counter_out,counter_in]=A[int(node_out.GetId()),int(node_in.GetId())]
                    counter_in+=1
                counter_out+=1

            edgeDict_new[key]=ajmatrix
            edgeIndexDict_new[key]=keyList_new
            self.graphList.append(MxWcc)

        return edgeDict_new, edgeIndexDict_new

    def getNodeAttributes(self):
        attributeslist=[]
        outputList=[]

        for UGraph in self.graphList:
            
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
            attributeslist.append(atrriMarix)
            outputList.append(attriList)
            # convert to undirected graph
            # GOut = snap.ConvertGraph(snap.PUNGraph, UGraph)

            # for NI in UGraph.Nodes():
            #     DegCentr = snap.GetDegreeCentr(UGraph, NI.GetId())
            #     print "node: %d centrality: %f" % (NI.GetId(), DegCentr)
            util.write_csv_inlist('attributeslist.csv',outputList)
        return attributeslist

if __name__ == "__main__":
    label='0.001'
    sN=SetupNetwork('./edgeDict_'+label+'.dat','./nodeIndexDict_'+label+'.dat','./paperDict_obj.dat',isBin=False)
    sN.buildNetwork()
    attributeslist=sN.getNodeAttributes()
    print 'done'