import environments.RegisterEnvironments as bW

import time
import copy
from joblib import Parallel, delayed

import multiprocessing


## Parallelization
def multicoreRuns(envRegisterName, learner, nbReplicates, timeHorizon, oneRunFunction):
    num_cores = multiprocessing.cpu_count()
    envs = []
    learners = []
    timeHorizons = []

    for i in range(nbReplicates):
        envs.append(bW.makeWorld(envRegisterName))
        learners.append(copy.deepcopy(learner))
        timeHorizons.append(copy.deepcopy(timeHorizon))

    t0 = time.time()

    cumRewards = Parallel(n_jobs=num_cores)(delayed(oneRunFunction)(*i) for i in zip(envs,learners,timeHorizons))

    elapsed = time.time()-t0
    return cumRewards, elapsed / nbReplicates


