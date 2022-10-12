import gym
from gym.envs.registration import  register
import numpy as np



"""
This file contains methods to register several MDP environments into gym.
Name of environments should not contain symbol '_'. 

"""
def registerRandomMDP(nbStates=5, nbActions=4, max_steps=np.infty, reward_threshold=np.infty, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5, ergodic=0.,seed=None):
    name = 'RandomMDP-S'+str(nbStates)+'_A'+str(nbActions)+'_s'+str(seed)+'-v0'
    if(ergodic>0):
        name = 'ErgodicRandomMDP-S' + str(nbStates) + '_A' + str(nbActions) + '_s' + str(seed) + '-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.RandomMDP:RandomMDP',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'nbActions': nbActions, 'maxProportionSupportTransition': maxProportionSupportTransition, 'maxProportionSupportReward': maxProportionSupportReward,
                'maxProportionSupportStart': maxProportionSupportStart, 'minNonZeroProbability':minNonZeroProbability, 'minNonZeroReward':minNonZeroReward, 'rewardStd':rewardStd, 'ergodic':ergodic, 'seed':seed, 'name':name }
    )
    return name

def registerRiverSwim(nbStates=5, max_steps=np.infty, reward_threshold=np.infty, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    name = 'RiverSwim-S'+str(nbStates)+'-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.RiverSwim:RiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'name':name }
    )
    return name


def registerErgodicRiverSwim(nbStates=5, max_steps=np.infty, reward_threshold=np.infty, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., ergodic=0.001):
    name = 'ErgodicRiverSwim-S'+str(nbStates)+'-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.RiverSwim:ErgodicRiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'ergodic': ergodic, 'name':name }
    )
    return name


def registerGridworld(sizeX=10, sizeY=10, map_name="4-room", rewardStd=0., initialSingleStateDistribution=False, max_steps=np.infty, reward_threshold=np.infty, start=None, goal=None, seed=0):
    name ='Gridworld-'+map_name+'-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.GridWorld.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':map_name,'rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution,'start':start, 'goal':goal, 'seed':seed, 'name':name}
    )
    return name


def registerRandomGridworld(sizeX=10, sizeY=10,rewardStd=0., initialSingleStateDistribution=False, max_steps=np.infty, reward_threshold=np.infty, density=0.2, seed=0):
    name ='RandomGridworld-'+str(sizeX)+'x'+str(sizeY)+'_s'+str(seed)+'-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.GridWorld.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':'random','rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution,'start':None, 'goal':None, 'name':name, 'seed':seed, 'density':density}
    )
    return name


def registerThreeState(delta = 0.005, max_steps=np.infty, reward_threshold=np.infty, fixed_reward = True):
    name = 'ThreeState-v0'
    register(
        id=name,
        entry_point='environments.discreteMDP:ThreeState',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'fixed_reward': fixed_reward,'name':name }
    )
    return name


def registerNasty(delta = 0.005, epsilon=0.05, max_steps=np.infty, reward_threshold=np.infty, fixed_reward = True):
    name = 'Nasty-v0'
    register(
        id=name,
        entry_point='environments.discreteMDPs.envs.ThreeStateMDP:Nasty',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'epsilon': epsilon, 'fixed_reward': fixed_reward,'name':name }
    )
    return name



registerWorlds = {
    "random-rich": lambda x: registerRandomMDP(nbStates=10, nbActions=4, maxProportionSupportTransition=0.12,
                                            maxProportionSupportReward=0.8, maxProportionSupportStart=0.1,
                                            minNonZeroProbability=0.15, minNonZeroReward=0.4, rewardStd=0, seed=10),

    "ergodic-random-rich": lambda x: registerRandomMDP(nbStates=10, nbActions=4, maxProportionSupportTransition=0.12,
                                               maxProportionSupportReward=0.8, maxProportionSupportStart=0.1,
                                               minNonZeroProbability=0.15, minNonZeroReward=0.4, rewardStd=0, ergodic=0.01, seed=10),
    "random-12" : lambda x: registerRandomMDP(nbStates=12, nbActions=2, maxProportionSupportTransition=0.15, maxProportionSupportReward=0.25, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=6),
    "random-small" : lambda x: registerRandomMDP(nbStates=3, nbActions=4, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.4, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.25, rewardStd=0.1,seed=5),
    "random-small-sparse" : lambda x: registerRandomMDP(nbStates=4, nbActions=4, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.08, maxProportionSupportStart=0.1, minNonZeroProbability=0.1, minNonZeroReward=0.2, rewardStd=0.1,seed=2),
    "random-100" : lambda x: registerRandomMDP(nbStates=100, nbActions=3, maxProportionSupportTransition=0.1, maxProportionSupportReward=0.1, maxProportionSupportStart=0.1, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.1,seed=10),
    "three-state" : lambda x: registerThreeState(delta = 0.005),
    "nasty": lambda x: registerNasty(delta=0.005,epsilon=0.05),
    "river-swim-6" : lambda x: registerRiverSwim(nbStates=6, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.),
    "ergo-river-swim-6": lambda x: registerErgodicRiverSwim(nbStates=6, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005,
                                                rewardR=0.99, ergodic=0.001),
    "ergo-river-swim-25": lambda x: registerErgodicRiverSwim(nbStates=25, rightProbaright=0.4, rightProbaLeft=0.05,
                                                            rewardL=0.005,
                                                            rewardR=0.99, ergodic=0.001),
    "river-swim-25" : lambda x: registerRiverSwim(nbStates=25, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=0.99),
    "grid-random-1616" : lambda x: registerRandomGridworld(sizeX=16, sizeY=16, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=7),
    "grid-random-1212" : lambda x: registerRandomGridworld(sizeX=12, sizeY=12, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=5),
    "grid-random-88" : lambda x: registerRandomGridworld(sizeX=8, sizeY=8, rewardStd=0.01, initialSingleStateDistribution=False, density=0.4, seed=1),#16 16 7# 12 12 5 # 8 8 1
    "grid-2-room" : lambda x: registerGridworld(sizeX=9, sizeY=11, map_name="2-room", rewardStd=0.0, initialSingleStateDistribution=True,start=[1,1],goal=[7,9],seed=1),
    "grid-4-room" : lambda x: registerGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.0, initialSingleStateDistribution=True,start=[1,1],goal=[5,5], seed=1)
}


def registerWorld(envName):
    if (envName in registerWorlds.keys()):
        regName = (registerWorlds[envName])(0)
        print("[INFO] Environment " + envName + " registered as " + regName)
        return regName

def makeWorld(registername):
    """

    :param registername: name of the environment to be registered into gym
    :return:  full name of the registered environment
    """
    return gym.make(registername)