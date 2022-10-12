from experiments.fullExperiment import *

import environments.RegisterEnvironments as bW
import learners.discreteMDPs.UCRL3 as le
import learners.discreteMDPs.PSRL as psrl
import learners.discreteMDPs.IRL as IRL
import learners.Generic.Qlearning as ql
import learners.Generic.Random as random

# To get the list of registered environments:
# print("List of registered environments: ")
# [print(k) for k in bW.registerWorlds.keys()]

#######################
# Pick an environment
#######################

# env = bW.makeWorld(bW.registerWorld('ergo-river-swim-6'))
env = bW.makeWorld(bW.registerWorld('river-swim-6'))
# env = bW.makeWorld(bW.registerWorld('grid-2-room'))
# env = bW.makeWorld(bW.registerWorld('grid-4-room'))
# env = bW.makeWorld(bW.registerWorld('river-swim-25'))
# env = bW.makeWorld(bW.registerWorld('ergo-river-swim-25'))
# env = bW.makeWorld(bW.registerWorld('grid-random-88'))
# env = bW.makeWorld(bW.registerWorld('grid-random-1212'))
# env = bW.makeWorld(bW.registerWorld('grid-random-1616'))
# env = bW.makeWorld(bW.registerWorld('random-rich'))
# env = bW.makeWorld(bW.registerWorld('ergodic-random-rich'))
# env = bW.makeWorld(bW.registerWorld('nasty'))


nS = env.observation_space.n
nA = env.action_space.n
delta = 0.05

#######################
# Select tested agents
#######################

agents = []
agents.append(([IRL.IRL, {"nbr_states":nS, "nbr_actions":nA}]))  # IMED-RL
agents.append( [psrl.PSRL, {"nS":nS, "nA":nA, "delta":delta}])  # PSRL
agents.append( [le.UCRL3_lazy, {"nS":nS, "nA":nA, "delta":delta}])  # UCRL3
agents.append([ql.Qlearning, {"nS": nS, "nA": nA}])  # Q-learning
# agents.append( [random.Random, {"env": env.env}])  # Random agent

#######################
# Run a full experiment
#######################
runLargeMulticoreExperiment(env, agents, timeHorizon=5000, nbReplicates=32)
