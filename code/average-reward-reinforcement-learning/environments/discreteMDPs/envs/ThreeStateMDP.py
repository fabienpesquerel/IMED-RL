
from environments.discreteMDPs.gymWrapper import DiscreteMDP
from environments.discreteMDPs.gymWrapper import Dirac
import scipy.stats as stat
import numpy as np

class ThreeState(DiscreteMDP):
    def __init__(self, delta=0.005, fixed_reward=True,name="ThreeState"):
        self.nS = 3
        self.nA = 2
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a 3-state MDP

        s = 0
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((delta, 1, False))
        self.transitions[s][0][1] = delta
        self.P[s][0].append((1. - delta, 2, False))
        self.transitions[s][0][2] = 1. - delta
        # Action 1 is just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        else:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)

        s = 1
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 0, False))
        self.transitions[s][0][0] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(1. / 3.)
            self.rewards[s][1] = Dirac(1. / 3.)
        else:
            self.rewards[s][0] = stat.bernoulli(1. / 3.)
            self.rewards[s][1] = stat.bernoulli(1. / 3.)

        s = 2
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 2, False))
        self.transitions[s][0][2] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.transitions[s][1] = {}
        self.P[s][1] = []  # 0=right", 1=left
        self.P[s][1].append((delta, 1, False))
        self.transitions[s][1][1] = delta
        self.P[s][1].append((1. - delta, 0, False))
        self.transitions[s][1][0] = 1. - delta
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(2. / 3.)
            self.rewards[s][1] = Dirac(2. / 3.)
        else:
            self.rewards[s][0] = stat.bernoulli(2. / 3.)
            self.rewards[s][1] = stat.bernoulli(2. / 3.)

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)
        super(ThreeState, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution,
                                         self.nameActions,name=name)


class Nasty(DiscreteMDP):
    def __init__(self, delta=0.005, epsilon=0.005, fixed_reward=True,name="Nasty"):
        self.nS = 7
        self.nA = 2
        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] = 1.
        # 231 0 456
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a 3-state MDP

        epsilon = epsilon

        s = 0
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((delta, 0, False))
        self.transitions[s][0][0] = delta
        self.P[s][0].append((1. - delta, 1, False))
        self.transitions[s][0][1] = 1. - delta
        # Action 1 is
        self.transitions[s][1] = {}
        self.P[s][1] = []  # 0=right", 1=left
        self.P[s][1].append((delta, 0, False))
        self.transitions[s][1][0] = delta
        self.P[s][1].append((1. - delta, 4, False))
        self.transitions[s][1][4] = 1. - delta
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)
        else:
            self.rewards[s][0] = Dirac(0.)
            self.rewards[s][1] = Dirac(0.)

        s = 1
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((0.5, 2, False))
        self.transitions[s][0][2] = 0.5
        self.P[s][0].append((0.5, 0, False))
        self.transitions[s][0][0] = 0.5
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99)
            self.rewards[s][1] = Dirac(0.99)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99)
            self.rewards[s][1] = stat.bernoulli(0.99)


        s = 2
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 3, False))
        self.transitions[s][0][3] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99)
            self.rewards[s][1] = Dirac(0.99)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99)
            self.rewards[s][1] = stat.bernoulli(0.99)

        s = 3
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 1, False))
        self.transitions[s][0][1] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99)
            self.rewards[s][1] = Dirac(0.99)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99)
            self.rewards[s][1] = stat.bernoulli(0.99)

        s = 4
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((0.5, 5, False))
        self.transitions[s][0][5] = 0.5
        self.P[s][0].append((0.5, 0, False))
        self.transitions[s][0][0] = 0.5
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99-epsilon)
            self.rewards[s][1] = Dirac(0.99-epsilon)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99-epsilon)
            self.rewards[s][1] = stat.bernoulli(0.99-epsilon)


        s = 5
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 6, False))
        self.transitions[s][0][6] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99-epsilon)
            self.rewards[s][1] = Dirac(0.99-epsilon)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99-epsilon)
            self.rewards[s][1] = stat.bernoulli(0.99-epsilon)



        s = 6
        self.P[s] = {}
        self.transitions[s] = {}
        # Action 0
        self.transitions[s][0] = {}
        self.P[s][0] = []  # 0=right", 1=left
        self.P[s][0].append((1., 4, False))
        self.transitions[s][0][4] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s] = {}
        if fixed_reward:
            self.rewards[s][0] = Dirac(0.99-epsilon)
            self.rewards[s][1] = Dirac(0.99-epsilon)
        else:
            self.rewards[s][0] = stat.bernoulli(0.99-epsilon)
            self.rewards[s][1] = stat.bernoulli(0.99-epsilon)

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)
        super(Nasty, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution,
                                         self.nameActions,name=name)

