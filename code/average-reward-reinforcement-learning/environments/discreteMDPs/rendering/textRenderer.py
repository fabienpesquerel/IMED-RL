
import sys
from six import StringIO
from gym import utils
import string

class textRenderer:

    def __init__(self):
        self.initializedRender = False


    def initRender(self, env):
        outfile = sys.stdout
        outfile.write("Actions: "+ str(env.nameActions) + "\n")

    def render(self,env,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(env)
            self.initializedRender = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = [str(s) for s in env.states]

        desc[current] = utils.colorize(desc[current], "red", highlight=True)
        for a in env.actions:
            for ssl in env.P[current][a]:
                if (ssl[0] > 0):
                    desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)

        #desc.append(" \t\tr=" + str(lastreward))

        if lastaction is not None:
            outfile.write("({})\tr={}\t".format(env.nameActions[lastaction % 26],str(lastreward)))
        else:
            outfile.write("\t\t\t")
        outfile.write("".join(''.join(line) for line in desc) + "\n")

        #if mode != 'text':
        #    return outfile