import minorminer
import minorminer.busclique
import dwave_networkx as dnx

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)

#===============================================================================
#=============================    TRICLIQUE    =================================
#===============================================================================


# colors = ['steelblue']+['yellowgreen']+['red']
# chain_color = {k: color for k,color in enumerate(colors)}

C3_DIMS = (C3_c,C3_v,C3_h) = 3,0,0
C3_tri = minorminer.busclique.find_triclique_embedding(*C3_DIMS,C3)
plt.figure(); dnx.draw_chimera_embedding(C3,C3_tri)

P2_DIMS = (P2_c,P2_v,P2_h) = 2,0,0
P2_tri = minorminer.busclique.find_triclique_embedding(*P2_DIMS,P2)
plt.figure(); dnx.draw_pegasus_embedding(P2,P2_tri)
plt.figure(); dnx.draw_chimera_embedding(C10_2,{},node_size=60)
P2_busgraph.draw_fragment_embedding(P2_tri, node_size=60)

Z1_DIMS = (Z1_c,Z1_v,Z1_h) = 3,0,0
Z1_tri = minorminer.busclique.find_triclique_embedding(*Z1_DIMS,Z1)
plt.figure(); dnx.draw_zephyr_embedding(Z1,Z1_tri)
plt.figure(); dnx.draw_chimera_embedding(C3_8,{},node_size=100)
Z1_busgraph.draw_fragment_embedding(Z1_tri,node_size=100)
