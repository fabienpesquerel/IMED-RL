

import numpy as np
import time
import networkx as nx # Requires version 2.5
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
#import rendering.MDPLayout as mlayout

colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan']
layouts = {'spring':nx.spring_layout,'shell':nx.shell_layout,'spectral':nx.spectral_layout, 'fruchterman_reingold':nx.fruchterman_reingold_layout,'kamada_kawai':nx.kamada_kawai_layout,
           'spiral':nx.spiral_layout,'circular':nx.circular_layout}

class GraphRenderer:

    def __init__(self,layout="spring"):
        self.initializedRender = False
        self.layout = layout
        self.screenshotpath="screenshots/"

    def initRender(self, env):
        self.label= str(time.time())
        self.cpt=0
        nS = env.nS
        scalearrow = nS  # (int) (np.sqrt(self.nS))
        scalepos = 10 * nS  # *self.nS
        G = nx.MultiDiGraph(action=0, rw=0.)
        for s in env.states:
            for a in env.actions:
                for ssl in env.P[s][a]:  # ssl = (p(s),s, 'done')
                    G.add_edge(s, ssl[1], action=a, weight=ssl[0], rw=env.R[s][a].mean())
        # Other possible layouts:
        # pos = nx.shell_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.fruchterman_reingold_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)

        pos = layouts[self.layout](G)  # Dict of numpy array
        # if(self.layout=='MDP'):
        #     m = mlayout.MDPLayout(states,actions,R,P,nameActions,current)
        #     pos = m.positions
        # else:
        #     pos = layouts[self.layout](G)#Dict of numpy array




        for x in env.states:
            pos[x] = [pos[x][0] * scalepos, pos[x][1] * scalepos]

        self.G = G
        self.pos = pos
        self.numFigure = plt.gcf().number

        plt.figure(self.numFigure)
        plt.clf()
        ax = plt.gca()

        nx.draw_networkx_nodes(G, pos, node_size=400,
                               node_color=['tab:gray' for s in G.nodes()])
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        for n in G:
            c = Circle(pos[n], radius=0.2, alpha=0.)
            ax.add_patch(c)
            G.nodes[n]['patch'] = c
        counts = np.zeros((nS, nS))
        countsR = np.zeros((nS, nS))
        seen = {}
        for u, v, d in G.edges(data=True):
            n1 = G.nodes[u]['patch']
            n2 = G.nodes[v]['patch']
            rad = 0.1
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1
            alpha = d['weight']
            color = colors[d['action']]
            if alpha > 0:
                counts[u][v] = counts[u][v] + 1
                if (u != v):
                    e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                                        arrowstyle='-|>',
                                        connectionstyle='arc3,rad=%s' % rad,
                                        mutation_scale=15.0 + scalearrow,
                                        lw=2,
                                        alpha=alpha,
                                        color=color)
                    seen[(u, v)] = rad
                    ax.add_patch(e)
                    if (d['rw'] > 0):
                        countsR[u][v] = countsR[u][v] + 1
                        nx.draw_networkx_edge_labels([u, v, d], pos,
                                                     edge_labels=dict([((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                     label_pos=0.5 + 0.1 * countsR[u][v], font_color=color, alpha=alpha,
                                                     font_size=8)

                else:
                    n1c = [n1.center[0] + 0.1 * (2 * counts[u][v] + scalearrow),
                           n1.center[1] + 0.1 * (2 * counts[u][v] + scalearrow)]
                    e1 = FancyArrowPatch(n1.center, n1c,
                                         arrowstyle='-|>',
                                         connectionstyle='arc3,rad=1.',
                                         mutation_scale=15.0 + scalearrow,
                                         lw=2,
                                         alpha=alpha,
                                         color=color)
                    e2 = FancyArrowPatch(n1c, n1.center,
                                         arrowstyle='-|>',
                                         connectionstyle='arc3,rad=1.',
                                         mutation_scale=15.0 + scalearrow,
                                         lw=2,
                                         alpha=alpha,
                                         color=color
                                         )
                    ax.add_patch(e1)
                    ax.add_patch(e2)
                    if (d['rw'] > 0):
                        countsR[u][v] = countsR[u][v] + 1
                        pos[u] = [pos[u][0] + 0.1 * (2 * countsR[u][v] + scalearrow),
                                  pos[u][1] + 0.1 * (2 * countsR[u][v] + scalearrow)]
                        nx.draw_networkx_edge_labels([u, v, d], pos,
                                                     edge_labels=dict(
                                                         [((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                     label_pos=0.5, font_color=color,
                                                     alpha=alpha, font_size=8)
                        pos[u] = [pos[u][0] - 0.1 * (2 * countsR[u][v] + scalearrow),
                                  pos[u][1] - 0.1 * (2 * countsR[u][v] + scalearrow)]

        ax.autoscale()
        plt.axis('equal')
        plt.axis('off')
        #plt.savefig('demo/screenshots/MDP-discrete'+str(time.time())+'.png')
        plt.savefig(self.screenshotpath+'discreteMDP-nx-'+self.label+'.png')
        plt.show(block=False)
        plt.pause(0.5)


    def render(self,env,current,lastaction,lastreward):
        """
            # Print the MDP in an image MDP.png, MDP.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the probability with which we transit to that state.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)
            # Print also the MDP only shoinwg the rewards in MDPonlytherewards.pdg, MDPonlytherewards.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the value of the mean reward.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)
            """
        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        plt.figure(self.numFigure)
        nx.draw_networkx_nodes(self.G, self.pos, node_size=400,
                               node_color=['tab:gray' if s != current else 'tab:orange' for s in self.G.nodes()])
        plt.savefig(self.screenshotpath+'discreteMDP-nx-'+self.label+"-"+str(self.cpt)+'.png')
        plt.show(block=False)
        plt.pause(0.01)
        self.cpt+=1