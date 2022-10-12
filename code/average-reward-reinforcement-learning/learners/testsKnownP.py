import numpy as np
import environments.RegisterEnvironments as bW
from pandas import *


def invariant_measure(Pa):
    nbS = len(Pa)
    # From https://towardsdatascience.com/markov-chain-analysis-and-simulation-using-python-4507cee0b06e
    # E.g. with
    # Pa = np.array([[0.2, 0.7, 0.1],
    #                [0.9, 0.0, 0.1],
    #                [0.2, 0.8, 0.0]])

    A = np.append(np.transpose(Pa) - np.identity(nbS), [np.ones(nbS)], axis=0)
    one = np.zeros(nbS+1)
    one[-1]=1.
    #print("rank:", np.linalg.matrix_rank(A))
    #print("rank:", np.linalg.matrix_rank(np.append(A,np.transpose([one]),axis=1)))
    #print(A)
    #print(b)
    pi = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(np.transpose(one))) # Rouché–Capelli
    #print("Invariant measure:", pi, "(Check:",pi.dot(Pa),")")

    #print(pi, pi.dot(Pa)) # Checking pi is indeed invariant measure: this sould be the same.
    #print(pi, Pa.dot(pi)) # Note that these quantities are usually different.
    return pi




def demo():

    testName = 'random10'
    envName = (bW.registerWorlds[testName])(0)
    env = bW.makeWorld(envName)
    nbS = env.observation_space.n
    nbA = env.action_space.n

    s = env.observation_space.sample()
    a = env.action_space.sample()

    print(s,a, env.P[s][a])

    Psa = env.getTransition(s,a)
    Rsa = env.getMeanReward(s,a)
    print(Psa)
    print(Rsa)


    Pa = env.getTransitionMatrix(a)
    #Pa = np.array([[0.2, 0.7, 0.1],
    #               [0.9, 0.0, 0.1],
    #               [0.2, 0.8, 0.0]])

    print(DataFrame(Pa))
    print(invariant_measure(Pa))
    #  Pa[s1][s2] is proba from sa to s2.


    # w,v = np.linalg.eig(Pa) # w is Right eig vector: v_i@P = w_i*v_i
    # print("Eigenvalues/Vectors:")
    # print(w)
    # wfilter = [0. if np.abs(x)<1. else 1. for x in w]
    # print(wfilter)
    # #print(v)
    # print([ v[i]/ np.linalg.norm(v[i]) for i in range(len(v)) ])
    #
    # vfilter = [[] if np.abs(w[i])<1. else v[i]/ np.linalg.norm(v[i]) for i in range(len(w))]
    # print("Eig.v associated to 1.:", vfilter)
    #
    # Pa2= np.transpose(Pa)
    #
    # w,v = np.linalg.eig(Pa2) # w is Right eig vector: v_i@P = w_i*P
    # print("Eigenvalues/Vectors:")
    # print(w)
    # wfilter = [0. if np.abs(x)<1. else 1. for x in w]
    # print(wfilter)
    # #print(v)
    #
    # vfilter = [[] if (np.abs(w[i])<1.) or (np.linalg.norm(v[i])==0.) else v[i]/ np.linalg.norm(v[i]) for i in range(len(w))]
    # print("Eig.v associated to 1.:", vfilter)





demo()


