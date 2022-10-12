
import sys
from six import StringIO
from gym import utils
import string

class GridworldWithWallRenderer:

    def __init__(self):
        self.initializedRender = False


    def initRender(self, env):
        outfile = sys.stdout
        outfile.write("Actions: "+ str(env.nameActions) + "\n")

    def render(self,env,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(current)
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            outfile.write("  ({})\n".format(env.nameActions[lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")




        #if mode != 'text':
        #    return outfile



class GridworldRenderer:

    def __init__(self):
        self.initializedRender = False


    def initRender(self, env):
        outfile = sys.stdout
        outfile.write("Actions: "+ str(env.nameActions) + "\n")

    def render(self,env,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(env.mapping[current])
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)


        #desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            outfile.write("\t({})\tr={}\n\n".format(env.nameActions[lastaction],str(lastreward)))
        else:
            outfile.write(" \n")

        outfile.write("\n".join(''.join(line) for line in desc) + "\n")




        #if mode != 'text':
