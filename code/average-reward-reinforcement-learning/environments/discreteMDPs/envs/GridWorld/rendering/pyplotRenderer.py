

import numpy as np
import time
import networkx as nx # Requires version 2.5
import matplotlib.pyplot as plt

class GridworldWithWallRenderer:

    def __init__(self,layout="spring"):
        self.initializedRender = False
        self.layout = layout
        self.screenshotpath="screenshots/"

    def initRender(self, env):
        self.label= str(time.time())
        self.cpt=0

        self.numFigure = plt.gcf().number
        plt.figure(self.numFigure)
        plt.imshow(env.maze, cmap='hot', interpolation='nearest')
        plt.savefig(self.screenshotpath + 'Gridworldwithwall-' + self.label + '.png')
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
        row, col = env.from_s(current)
        v = env.maze[row][col]
        env.maze[row][col] = 1.5
        plt.imshow(env.maze, cmap='hot', interpolation='nearest')
        env.maze[row][col] = v

        # plt.annotate('A', fontsize=20, xy=(0.5, 0.5), xycoords='axes fraction', xytext=(1.5, 0.5),
        #             textcoords='offset points',
        #             arrowprops=dict(arrowstyle="->", linewidth=5., color='red'))

        plt.show(block=False)
        plt.pause(0.01)

        plt.savefig(self.screenshotpath+'Gridworldwithwall-'+self.label+"-"+str(self.cpt)+'.png')
        plt.show(block=False)
        plt.pause(0.01)
        self.cpt+=1



class GridworldRenderer:

    def __init__(self,layout="spring"):
        self.initializedRender = False
        self.layout = layout
        self.screenshotpath="screenshots/"

    def initRender(self, env):
        self.label= str(time.time())
        self.cpt=0

        self.numFigure = plt.gcf().number
        plt.figure(self.numFigure)
        plt.clf()
        plt.imshow(env.maze, cmap='hot', interpolation='nearest')
        plt.savefig(self.screenshotpath + 'Gridworld-' + self.label + '.png')
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
            plt.clf()
            row, col = env.from_s(env.mapping[current])
            v = env.maze[row][col]
            env.maze[row][col] = 1.5
            plt.imshow(env.maze, cmap='hot', interpolation='nearest')
            env.maze[row][col] = v

            xpos1 = (col + 0.25) / (env.sizeY + 0.)
            ypos1 = 1. - (row + 0.75) / (env.sizeX + 0.)
            xpos2 = (col + 0.25) / (env.sizeY + 0.)
            ypos2 = 1. - (row + 0.75) / (env.sizeX + 0.)

            if (lastaction != None):
                ann = plt.annotate(env.nameActions[lastaction], fontsize=20, xy=(0.5, -0.1),
                                   xycoords='axes fraction', xytext=(0.5, -0.1),
                                   textcoords='offset points')
                # ann=plt.annotate(self.nameActions[self.lastaction], fontsize=20, xy=(xpos1, ypos1), xycoords='axes fraction', xytext=(xpos2, ypos2),
                #         textcoords='offset points')#,                         arrowprops=dict(arrowstyle="->", linewidth=5., color='red'))
            else:
                ann = plt.annotate('', fontsize=20, xy=(xpos1, ypos1),
                                   xycoords='axes fraction', xytext=(xpos2, ypos2),
                                   textcoords='offset points')  # arrowprops=dict(arrowstyle="->", linewidth=5., color='red'))

            plt.savefig(self.screenshotpath+'Gridworld-'+self.label+"-"+str(self.cpt)+'.png')
            plt.show(block=False)
            plt.pause(0.01)
            self.cpt+=1