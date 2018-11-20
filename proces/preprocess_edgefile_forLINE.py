import sys
import os

path='F:\\data\\dblp\\graph\\'
inputpath='lcc-author-citation-graph-2013-2014.edgelist.txt'
outputpath='lcc-author-citation-graph-2013-2014.edgelist.LINE.txt'
fi = open(os.path.join(path,inputpath), 'r')
fo = open(os.path.join(path,outputpath), 'w')
for line in fi:
	items = line.strip().split()
	fo.write('{}\t{}\t1\n'.format(items[0], items[1]))
	fo.write('{}\t{}\t1\n'.format(items[1], items[0]))
fi.close()
fo.close()
