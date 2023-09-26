import minorminer
import minorminer.busclique
import dwave_networkx as dnx

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)

#===============================================================================
#========================    CHIMERA EQUIVALENCE    ============================
#===============================================================================

C3 = dnx.chimera_graph(3)
C3_busgraph = minorminer.busclique.busgraph_cache(C3)
C3_one2one = {v:(v,) for v in C3}
dnx.draw_chimera_embedding(C3,C3_one2one)

P2 = dnx.pegasus_graph(2)
P2_busgraph = minorminer.busclique.busgraph_cache(P2)
P2_one2one = {v:(v,) for v in P2}
plt.figure(); dnx.draw_pegasus_embedding(P2,P2_one2one,crosses=True)
C10_2 = dnx.chimera_graph(12,12,2)
plt.figure(); dnx.draw_chimera_embedding(C10_2,{},node_size=60)
P2_busgraph.draw_fragment_embedding(P2_one2one,node_size=60)

Z1 = dnx.zephyr_graph(1)
Z1_busgraph = minorminer.busclique.busgraph_cache(Z1)
Z1_one2one = {v:(v,) for v in Z1}
plt.figure(); dnx.draw_zephyr_embedding(Z1,Z1_one2one)
C3_8 = dnx.chimera_graph(3,3,8)
plt.figure(); dnx.draw_chimera_embedding(C3_8,{},node_size=100)
Z1_busgraph.draw_fragment_embedding(Z1_one2one,node_size=100)
