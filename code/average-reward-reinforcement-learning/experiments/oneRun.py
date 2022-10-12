import pickle
import time
import os

from experiments.utils import get_project_root_dir
ROOT= get_project_root_dir()+"/experiments/"

def oneXpNoRender(env,learner,timeHorizon):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("[Info] New initialization of ", learner.name(), ' for environment ',env.name)
    #print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        learner.update(state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info
        except TypeError:
            cummean += reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation=env.reset()
            #break

    #print("Cumreward: " + str(cumreward))
    #print("Cummean: " + str(cummean))
    return cummeans #cumrewards,cummeans


def oneXpNoRenderWithDump(env,learner,timeHorizon):
    observation = env.reset()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("[Info] New initialization of ", learner.name(), ' for environment ',env.name)
    #print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, done, info = env.step(action)
        learner.update(state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info
        except TypeError:
            cummean +=reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation = env.reset() # converts an episodic MDP into an infinite time horizon MDP
            #break

    filename = ROOT+"results/cumMeans_" + env.name + "_" + learner.name() + "_" + str(timeHorizon) +"_" + str(time.time())
    file =  open(filename,'wb')
    file.truncate(0)
    pickle.dump(cummeans, file)
    file.close()
    return filename

def oneRunOptWithDump(env, opti_learner, timeHorizon):
 ## Cumlative reward of optimal policy:
    opttimeHorizon = min(max((1000000, timeHorizon)),10**8)
    cumReward_opti = oneXpNoRender(env, opti_learner, opttimeHorizon)
    gain =  cumReward_opti[-1] / len(cumReward_opti)
    #print("Average gain is ", gain)
    opti_cumgain = [[t * gain for t in range(timeHorizon)]]
    filename = ROOT+"results/cumMeans_" + env.name + "_" + opti_learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    file = open(filename, 'wb')
    file.truncate(0)
    pickle.dump(opti_cumgain, file)
    file.close()
    return filename

def clear_auxiliaryfiles(env):
        for file in os.listdir(ROOT+"results/"):
            if file.startswith("cumMeans_" + env.name):
                os.remove(ROOT+"results/"+file)