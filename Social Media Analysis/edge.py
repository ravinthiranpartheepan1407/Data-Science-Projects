import networkx as nx
import matplotlib.pyplot as plt
edge = nx.read_edgelist('fb-pages-tvshow.edges', create_using=nx.DiGraph(), nodetype=int)
nx.info(edge)


nx.draw_networkx(edge, with_labels=True)
plt.draw()
plt.show()
