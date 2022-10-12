
import pylab as pl
import pickle
import os
import sys
import numpy as np


from experiments.utils import get_project_root_dir
ROOT= get_project_root_dir()+"/experiments/"

def plotCumulativeRegrets(learnersName, envName, title, mean, median, quantile1, quantile2, times, timeHorizon, logfile='', timestamp=0):
    if (logfile==''):
        logfile=sys.stdout
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = ROOT+"results/Regret_"
    #colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
     '#f781bf', '#a65628', '#984ea3',
     '#999999', '#e41a1c', '#dede00']

    style = ['o','v','s','d','<']
    pl.title(title)
    for i in range(len(median)):
        pl.plot(times, median[i], style[i% len(style)], label=learnersName[i], color=colors[i % len(colors)], linewidth=2.0, linestyle='--', markevery=0.05)
        #pl.plot(times,median[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.8)
        pl.plot(times,quantile1[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.6)
        pl.plot(times,quantile2[i], color=colors[i % len(colors)],linestyle=':',linewidth=0.6)
        textfile += learnersName[i] + "_"
        logfile.write(learnersName[i] + ' has regret ' + str(median[i][-1]) + ' after ' + str(timeHorizon) + ' time steps with quantiles ' +
              str(quantile1[i][-1]) +' and '+ str(quantile2[i][-1])+"\n")

    textfile+="_"+str(timeHorizon)+"_"+envName+"_"+str(timestamp)
    pl.legend(loc=2)
    pl.xlabel("Time steps", fontsize=13, fontname = "Arial")
    pl.ylabel("Regret Tg*-sum_t r_t", fontsize=13, fontname = "Arial")
    #pl.xticks(times)
    pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    pl.ylim(0)
    pl.savefig(textfile+'.png')
    pl.savefig(textfile+ '.pdf')
    pl.xscale('log')
    pl.savefig(textfile + '_xlog.png')
    pl.savefig(textfile + '_xlog.pdf')
    pl.ylim(1)
    if(timeHorizon>10):
        pl.xlim(10,timeHorizon)
    pl.xscale('linear')
    pl.yscale('log')
    pl.savefig(textfile + '_ylog.png')
    pl.savefig(textfile + '_ylog.pdf')
    pl.xscale('log')
    pl.savefig(textfile + '_loglog.png')
    pl.savefig(textfile + '_loglog.pdf')
    logfile.write("\nPlots are depicted in files "+textfile + ".pdf/png, etc.")


def plot_results_from_dump(cumRegretfiles, tplot,folder=ROOT+"results/",logfile=''):
    """
    Requires result files to be named "cumRegret_" + envName + "_" + learner.name() + "_" + str(tmax)+"_" + whatever
    that is, the result of parse=filename.split('_') is e.g. ['cumRegret', 'RiverSwim-6-v0', 'UCRL3', '1000', ...]
    the envName should be the same for all files
    :param tplot: the results are plotted only until time horizon tplot (which should be less than tmax, the number of points in the data)
    :return:
    """
    if (logfile==''):
        logfile=sys.stdout
    median = []
    quantile1 = []
    quantile2 = []

    #print("File:", cumRegretfiles[0])
    tmax = int((cumRegretfiles[0].split('_'))[3])
    skip = max(1, (tmax // 1000))
    itimes = [t for t in range(0,tmax,skip)]
    times = [itimes[i] for i in range(len(itimes)) if i*skip<tplot]

    learnerNames=[]
    envName = (cumRegretfiles[0].split('_'))[1]

    for filename in cumRegretfiles:
        #filename = "results/cumRegret_" + envName + "_" + lname + "_" + str(tmax)
        parse=filename.split('_')
        #print(parse)
        if (parse[1] == envName):
            learnerNames.append(parse[2])
        file = open(folder+filename, 'rb')
        data_j = pickle.load(file)
        file.close()

        q = np.quantile(data_j, 0.5, axis=0)
        median.append([q[i] for i in range(len(q)) if i*skip<tplot])
        q=np.quantile(data_j, 0.25, axis=0)
        quantile1.append([q[i] for i in range(len(q)) if i*skip<tplot])
        q = np.quantile(data_j, 0.75, axis=0)
        quantile2.append([q[i] for i in range(len(q)) if i*skip<tplot])

    plotCumulativeRegrets(learnerNames, envName, envName, None , median, quantile1, quantile2, times, tplot, logfile=logfile)


def search_dump_cumRegretfiles(regName):
    # Search cumReret dumpfiles, choosing only one file (first in the list) for each algo:
    files = []
    algos = []
    for file in os.listdir(ROOT+"results/"):
        #print(file)
        if file.startswith("cumRegret_"+regName):
            parse = file.split('_')
            #print(parse)
            if (parse[2] not in algos):
                algos.append(parse[2])
                files.append(file)
    return files
