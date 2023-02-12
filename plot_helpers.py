import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def linearRegression(X, Y):
    linear_regressor = LinearRegression()
    linear_regressor.fit(X.values.reshape(-1,1), Y.values.reshape(-1,1))

    Ypred = linear_regressor.predict(X.values.reshape(-1,1)) 

    Ypred = Ypred.reshape(-1,)
    X = np.array(X).reshape(-1,)
    r2 = r2_score(Y.values.reshape(-1,1), Ypred)
    
    return X, Ypred, r2, linear_regressor.coef_


def plot_with_lin_reg(axs, partydf, ax, clor, partyname, indVar, depVar='govtSupport', axesTicks=None):

    axs[ax[0],ax[1]].scatter(partydf[indVar], partydf[depVar], color=clor)

    X, Ypred, r2, coeff = linearRegression(partydf[indVar], partydf[depVar])

    axs[ax[0],ax[1]].plot(X, Ypred, color='#000000')

    axs[ax[0],ax[1]].set_title(partyname+'\n(a1=%.2f; r2=%.2f)' % (coeff, r2), size=11)

    if not axesTicks:
        axs[ax[0],ax[1]].axes.xaxis.set_visible(False)
        axs[ax[0],ax[1]].axes.yaxis.set_visible(False)


def plot_over_parliament(parlList, X, Y, labels, colorDict, title=None, xlabel=None, ylabel=None):
    axes = [[0,0], [0,1], [1,0], [1,1], [2,0]]
    fig, axs = plt.subplots(3,2, figsize=(10,10))

    for i, parl in enumerate(parlList):
        if type(X)==list:
            indVar = X[i]+'_share'
        else:
            indVar = X
        colorMap = parl[labels].map(colorDict) 
        plot_with_lin_reg(axs, parl, axes[i], colorMap, str(38+i)+' Parliament', indVar, depVar=Y)


    fig.delaxes(axs[2,1])

    fig.suptitle(title, fontsize=13, y=0.95)
    fig.text(0.5, 0.1, xlabel, ha='center', fontsize = 12)
    fig.text(0.08, 0.5, ylabel, va='center', rotation ='vertical', fontsize=12)

    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(colorDict.values())[0:len(colorDict.values())-1]]
    plt.legend(markers, list(colorDict.keys())[0:len(colorDict.keys())-1], numpoints=1, bbox_to_anchor=(1.25, 1))


def plot_over_parl_party(byPartyParliament, X, title=None, xlabel=None, ylabel=None, depVar='govtSupport'):
    partyaxesList = [[[rowi, columni] for rowi in range(0,6,1)] for columni in range(0,6,1)]
    partycolors = ['#f37021', '#3d9b35', '#33b2cc', '#d71920', '#1a4782']
    fig, axs = plt.subplots(5,5, figsize=(10,10))

    for i in range(5):
        if type(X)==list:
            indVar = X[i]+'_share'
        else:
            indVar = X
        for j in range(5):
            try:
                plot_with_lin_reg(axs, byPartyParliament[i][j], partyaxesList[i][j], partycolors[j], '', indVar, axesTicks=False, depVar=depVar)
            except:
                axs[partyaxesList[i][j][0],partyaxesList[i][j][1]].axes.xaxis.set_visible(False)
                axs[partyaxesList[i][j][0],partyaxesList[i][j][1]].axes.yaxis.set_visible(False)


    fig.suptitle(title, fontsize=13, y=0.97)
    fig.text(0.5, 0.1, xlabel, ha='center', fontsize = 12)
    fig.text(0.07, 0.5, ylabel, va='center', rotation ='vertical', fontsize=12)

    parl_party_subheads(fig)


def parl_party_subheads(fig):
    fig.text(0.1, 0.823, 'NDP', va='center', rotation ='vertical', fontsize=12)
    fig.text(0.1, 0.66, 'Green', va='center', rotation ='vertical', fontsize=12)
    fig.text(0.1, 0.51, 'Bloc', va='center', rotation ='vertical', fontsize=12)
    fig.text(0.1, 0.35, 'Liberal', va='center', rotation ='vertical', fontsize=12)
    fig.text(0.1, 0.193, 'Conservative', va='center', rotation ='vertical', fontsize=12)

    fig.text(0.18, 0.91, '38', va='center', fontsize=12)
    fig.text(0.33, 0.91, '39', va='center', fontsize=12)
    fig.text(0.50, 0.91, '40', va='center', fontsize=12)
    fig.text(0.67, 0.91, '41', va='center', fontsize=12)
    fig.text(0.82, 0.91, '42', va='center', fontsize=12)