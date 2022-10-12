
import pickle
import time
import numpy as np


from experiments.utils import get_project_root_dir
ROOT= get_project_root_dir()+"/experiments/"

def computeCumulativeRegrets(names, dump_cumulativerewards_, timeHorizon, envName =""):
    """

    :param names: get list of algorithm names
    :param dump_cumulativerewards_: list of filenames, each getting cumulative rewards for multiple runs. Last file of the list is cum reward of Oracle.
    :param timeHorizon:
    :param envName:
    :return: vectors median, quantile0.25, quantile0.75, timesteps, where median[i] is median of expreimnts at time timesteps[i]
    """
    median = []
    mean = []
    quantile1 = []
    quantile2 = []
    nbAlgs = len(dump_cumulativerewards_) - 1

    #Downsample the times, especially in case timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0,timeHorizon,skip)]

    for j in range(nbAlgs):
        data_j = []
        for i in range(len(dump_cumulativerewards_[j])):
            file_oracle = open(dump_cumulativerewards_[-1], 'rb')
            cum_rewards_oracle = pickle.load(file_oracle)
            cum_rewards_oracle = cum_rewards_oracle[0]
            file = open(dump_cumulativerewards_[j][i], 'rb')
            cum_rewards_ij = pickle.load(file)
            data_j.append([cum_rewards_oracle[t] - cum_rewards_ij[t] for t in range(0,timeHorizon,skip)])
            file_oracle.close()
            file.close()

        filename = ROOT+"results/cumRegret_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
            j) + "_" + str(
            time.time())
        file = open(filename, 'wb')
        pickle.dump(data_j, file)
        file.close()

        mean.append(np.mean(data_j, axis=0))
        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.25, axis=0))
        quantile2.append(np.quantile(data_j, 0.75, axis=0))

    return mean,median,quantile1,quantile2,times