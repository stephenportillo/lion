import numpy as np
import scipy.spatial
import networkx as nx
import matplotlib.pyplot as plt

PCcat = np.load('chain_thinned.npz')

maxn = 2000 # source number
maxc = 10 # sample number
PCn = PCcat['n'][0:maxc].astype(np.int)
PCx = PCcat['x'][0:maxc,:]
PCy = PCcat['y'][0:maxc,:]
PCf = PCcat['f'][0:maxc,:]
PCi,junk = np.mgrid[0:maxc, 0:maxn]

mask = PCf > 0
PCc_all = np.zeros((np.sum(mask), 2))
PCc_all[:, 0] = PCx[mask].flatten()
PCc_all[:, 1] = PCy[mask].flatten()
PCi = PCi[mask].flatten()

#pos = {}
#weight = {}
#for i in xrange(np.sum(mask)):
# pos[i] = (PCc_all[i, 0], PCc_all[i,1])
# weight[i] = 0.5

#print pos[0]
#print PCc_all[0, :]
#print "graph..."
#G = nx.read_gpickle('graph')
#G = nx.geographical_threshold_graph(np.sum(mask), 1./0.75, alpha=1., dim=2., pos=pos, weight=weight)

kdtree = scipy.spatial.KDTree(PCc_all)
matches = kdtree.query_ball_tree(kdtree, 0.75)

G = nx.Graph()
G.add_nodes_from(xrange(0, PCc_all.shape[0]))

for i in xrange(PCc_all.shape[0]):
 for j in matches[i]:
  if PCi[i] != PCi[j]:
   G.add_edge(i, j)

current_catalogue = 0
for i in xrange(PCc_all.shape[0]):
 matches[i].sort()
 bincount = np.bincount(PCi[matches[i]]).astype(np.int)
 ending = np.cumsum(bincount).astype(np.int)
 starting = np.zeros(bincount.size).astype(np.int)
 starting[1:bincount.size] = ending[0:bincount.size-1]
 for j in xrange(bincount.size):
  if j == PCi[i]: # do not match to same catalogue
   continue
  if bincount[j] == 0: # no match to catalogue j
   continue
  if bincount[j] == 1: # exactly one match to catalogue j
   continue
  if bincount[j] > 1:
   dist2 = 0.75**2
   l = -1
   for k in xrange(starting[j], ending[j]):
    m = matches[i][k]
    newdist2 = np.sum((PCc_all[i,:] - PCc_all[m,:])**2)
    if newdist2 < dist2:
     l = m
     dist2 = newdist2
   if l == -1:
    print "didn't find edge even though mutiple matches from this catalogue?"
   for k in xrange(starting[j], ending[j]):
    m = matches[i][k]
    if m != l:
     if G.has_edge(i, m):
      G.remove_edge(i, m)
      print "killed", i, m

seeds = []

while nx.number_of_nodes(G) > 0:
 deg = nx.degree(G)
 i = max(deg, key=deg.get)
 neighbors = nx.all_neighbors(G, i)
 print 'found', i
 seeds.append([PCc_all[i, 0], PCc_all[i, 1], deg[i]])
 G.remove_node(i)
 G.remove_nodes_from(neighbors)

seeds = np.array(seeds)
print seeds
np.savetxt('seeds_lion.txt', seeds)
