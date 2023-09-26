import minorminer
import minorminer.busclique
import dwave_networkx as dnx

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)

#===============================================================================
#============================    BICLIQUE    ===================================
#===============================================================================
SHAPE = (12,12)
colors = ['steelblue','yellowgreen']
chain_color = {k:colors[k//12] for k in range(sum(SHAPE))}
bi_kwargs = {"chain_color":chain_color,"width":3, "show_labels":True}

C3 = dnx.chimera_graph(3)
C3_busgraph = minorminer.busclique.busgraph_cache(C3)
C3_biemb = C3_busgraph.find_biclique_embedding(*SHAPE)
plt.figure()
dnx.draw_chimera_embedding(C3,C3_biemb,**bi_kwargs)
plt.savefig('biclique_chimera.svg')

P3 = dnx.pegasus_graph(3)
P3_busgraph = minorminer.busclique.busgraph_cache(P3)
P3_biemb = P3_busgraph.find_biclique_embedding(*SHAPE)
plt.figure()
dnx.draw_pegasus_embedding(P3,P3_biemb,crosses=True,node_size=100,**bi_kwargs)
plt.savefig('biclique_pegasus.svg')

Z2 = dnx.zephyr_graph(2)
Z2_busgraph = minorminer.busclique.busgraph_cache(Z2)
Z2_biemb = Z2_busgraph.find_biclique_embedding(*SHAPE)
plt.figure()
dnx.draw_zephyr_embedding(Z2,Z2_biemb,node_size=100,**bi_kwargs)
plt.savefig('biclique_zephyr.svg')

#===============================================================================
#==============================    CLIQUE    ===================================
#===============================================================================
CLIQUE = 12

C3_cliemb = C3_busgraph.find_clique_embedding(CLIQUE)
plt.figure()
dnx.draw_chimera_embedding(C3,C3_cliemb,width=3)
plt.savefig('clique_chimera.svg')

P3_cliemb = P3_busgraph.find_clique_embedding(CLIQUE)
plt.figure()
dnx.draw_pegasus_embedding(P3,P3_cliemb,crosses=True,width=3,node_size=60)
plt.savefig('clique_pegasus.svg')

Z2_cliemb = Z2_busgraph.find_clique_embedding(CLIQUE)
plt.figure()
dnx.draw_zephyr_embedding(Z2,Z2_cliemb,width=3,node_size=150)
plt.savefig('clique_zephyr.svg')
