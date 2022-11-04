#! /usr/bin/env python3
import os, multiprocessing
import copy
import math
import yaml
import time
#import uproot
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep
from array import array
#from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
#from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
#from ROOT import TStyle, TCanvas, TPad
#from ROOT import TLegend, TLatex, TText, TLine, TProfile
#from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
#gROOT.SetBatch(True)
#gStyle.SetOptStat(0000)
plt.style.use(hep.style.CMS)

PRELIMINARY = False
out_fold = "fig/Preliminary/" if PRELIMINARY else "fig/"

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def plot_ROC_supplementary(ERA,extra_label=""):
    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH','SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH','SUSY_mh400_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH','SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH','SUSY_mh1800_ctau3000_HH',
    ]
    back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET","VV"]
    ROCDIR = "/nfs/dust/cms/user/lbenato/ROC_numpy_"+ERA+"/"
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    NTUPLEDIR_v5 = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_v5_ntuples/"

    
    #Load fpr, tpr and so on
    tpr = {}
    fpr = {}
    idx = {}
    thresholds = {}
    eff_idx = {}
    new_idx = {}
    cut_fpr = 0.00040904540701505433
    g = {}
    p = {}
    for i,s in enumerate(sign):
        print("Loading ", s, ROCDIR+"tpr_"+s+".npy")
        fpr[s] = (np.load(ROCDIR+"fpr_"+s+".npy"))#[0:1000000]
        tpr[s] = (np.load(ROCDIR+"tpr_"+s+".npy"))#[0:1000000]
        idx[s] = (np.load(ROCDIR+"idx_"+s+".npy"))
        thresholds[s] = (np.load(ROCDIR+"thresholds_"+s+".npy"))
        eff_idx[s], _ = find_nearest(thresholds[s],0.996)
        new_idx[s], _ = find_nearest(fpr[s],cut_fpr)
        print("size: ", len(fpr[s]))
        print("fpr at eff_idx: ", fpr[s][eff_idx[s]])
        print("tpr at eff_idx: ", tpr[s][eff_idx[s]])
        #print("fpr at idx: ", fpr[s][idx[s]])
        #print("fpr at new_idx: ", fpr[s][new_idx[s]])
        print("threshold at eff_idx: ", thresholds[s][eff_idx[s]])
        #print("threshold new_idx: ", thresholds[s][new_idx[s]])
        ###scan_t = fpr[s][eff_idx[s]]



    print("Done loading, save graphs")

    colors = ['blue','blue','crimson','crimson','green','green','darkorange','darkorange','gray','gray','magenta','magenta','chocolate','chocolate','yellow','yellow','black','black','olive','olive']
    linestyles = ['-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--',]
    pointstyles = ['o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s',]
    facecolors= ['blue','none','crimson','none','green','none','darkorange','none','gray','none','magenta','none','chocolate','none','yellow','none','black','none','olive','none']

    label_dict = {
        'ref' : r'$m_{\chi}$ = 400 GeV, c$\tau$ = 1 m (w. p. 0.996)',
        'SUSY_mh127_ctau500_HH' : r'$m_{\chi}$ = 127 GeV, c$\tau$ = 0.5 m',
        'SUSY_mh127_ctau3000_HH' : r'$m_{\chi}$ = 127 GeV, c$\tau$ = 3 m',
        'SUSY_mh400_ctau500_HH' : r'$m_{\chi}$ = 400 GeV, c$\tau$ = 0.5 m',
        'SUSY_mh400_ctau3000_HH' : r'$m_{\chi}$ = 400 GeV, c$\tau$ = 3 m',
        'SUSY_mh1000_ctau500_HH' : r'$m_{\chi}$ = 1 TeV, c$\tau$ = 0.5 m',
        'SUSY_mh1000_ctau3000_HH' : r'$m_{\chi}$ = 1 TeV, c$\tau$ = 3 m',
        'SUSY_mh1800_ctau500_HH' : r'$m_{\chi}$ = 1.8 TeV, c$\tau$ = 0.5 m',
        'SUSY_mh1800_ctau3000_HH' : r'$m_{\chi}$ = 1.8 TeV, c$\tau$ = 3 m',
    }
    
    #hep
    fig, ax = plt.subplots(figsize=(10, 10))
    #plt.rcParams.update({"font.size": 50})
    AUC = 0.42
    if ERA=="2018":
        #load also the original roc
        print("Loading also the original 2018 ROC")
        inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
        store = pd.HDFStore(inp_file)
        df_test = store.select("df")
        print("Remove negative weights at testing!!!!!!")
        df_test = df_test.loc[df_test['EventWeight']>=0]
        fpr_ref, tpr_ref, thresholds_ref = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])
        #original
        cut_fpr = 0.00040904540701505433
        #in the other samples
        ###cut_fpr = scan_t
        idx_ref, _ = find_nearest(fpr_ref,cut_fpr)
        eff_idx_ref, _ = find_nearest(thresholds_ref,0.996)
        print("reference threshold at eff_idx_ref: ", thresholds_ref[eff_idx_ref])
        print("reference fpr at eff_idx_ref: ", fpr_ref[eff_idx_ref])
        print("reference fpr at idx_ref: ", fpr_ref[idx_ref])

        plt.plot(fpr_ref, tpr_ref, color='black', lw=3, linestyle='-')
        plt.plot(fpr_ref[idx_ref], tpr_ref[idx_ref], marker='P', lw=3, color='black', markersize=10, label=label_dict['ref'])

    for i,s in enumerate(sign):
        #plt.plot(fpr[s], tpr[s], color=colors[i], lw=3, linestyle=linestyles[i], label=label_dict[s])
        #plt.plot(fpr[s][eff_idx[s]], tpr[s][eff_idx[s]], marker=pointstyles[i], color=colors[i], label="eff. {0:.3f}".format(tpr[s][eff_idx[s]]))
        #nope#plt.plot(fpr[s][eff_idx[s]], tpr[s][eff_idx[s]], lw=0, marker=pointstyles[i], edgecolor=colors[i], facecolor=facecolors[i], label=label_dict[s])
        #fpr and tpr at 0.996 w.p.
        plt.plot(fpr[s][eff_idx[s]], tpr[s][eff_idx[s]], lw=0, marker=pointstyles[i], color=colors[i], mfc=facecolors[i], label=label_dict[s], markersize=10)

        #fpr and tpr at reference fpr
        plt.plot(fpr[s][new_idx[s]], tpr[s][new_idx[s]], lw=0, marker="P", color=colors[i], mfc=facecolors[i], markersize=10)


    plt.ylim([0.57, 1.01])
    plt.xlim([0.0001, 1.05])
    plt.xlim([0.0002, 0.0006])
    plt.xscale("log")

    #Same as the paper ROC
    #plt.ylim([0.5, 1.01])
    #plt.xlim([0., 0.01])

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", title="")
    plt.grid(True)
    hep.cms.label("Supplementary",data=False, year=int(ERA))
    #hep.cms.text("Simulation Supplementary")
    fig.savefig(out_fold+"TDJ_ROC_supplementary_"+ERA+extra_label+".pdf")
    fig.savefig(out_fold+"TDJ_ROC_supplementary_"+ERA+extra_label+".png")
    print("Written ", out_fold+"TDJ_ROC_supplementary_"+ERA+extra_label+".pdf")


    #Matplotlib
    #for i,s in enumerate(sign):
    #    plt.plot(fpr[s], tpr[s], color=colors[i], lw=2, linestyle=linestyles[i], label=samples[s]['label'])#"AUC = {0:.4f}".format(AUC))
    #    #plt.plot(fpr[l][idx[l]], tpr[l][idx[l]],"ro",color=colors[i],label="w.p. {0:.4f}".format(thresholds[l][idx[l]]))
    ##plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ##plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based, all eta(2018)")
    #plt.title(str(ERA)+' MC')
    #plt.savefig(out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".pdf")
    #plt.savefig(out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".png")
    #print "Info: ROC curve file "+out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".pdf has been created"

    #plt.figure(figsize=(8,7))
    #plt.rcParams.update({"font.size": 15}) #Larger font size                                                                                                                        
    #AUC = 0.4
    #for i,l in enumerate(list_comparison):
    #    tpr[l] = np.load(PLOTDIR+"tpr"+l+".npy")
    #    fpr[l] = np.load(PLOTDIR+"fpr"+l+".npy")
    #    idx[l] = np.load(PLOTDIR+"idx"+l+".npy")
    #    thresholds[l] = np.load(PLOTDIR+"thresholds"+l+".npy")
    #    plt.plot(fpr[l], tpr[l], color=colors[i], lw=2, linestyle=linestyles[i], label="ROC"+l)#"AUC = {0:.4f}".format(AUC))                                                        
    #    plt.plot(fpr[l][idx[l]], tpr[l][idx[l]],"ro",color=colors[i],label="w.p. {0:.4f}".format(thresholds[l][idx[l]]))
    #plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    #plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based, all eta(2018)")
    #plt.title(str(ERA)+' MC')
    #plt.ylim([0.6, 1.05])
    #plt.xlim([0.0001, 1.05])
    #plt.xscale("log")
    #plt.ylabel("True Positive Rate")
    #plt.xlabel("False Positive Rate")
    #plt.legend(loc="lower right", title="FCN")
    #plt.grid(True)
    #plt.savefig(PLOTDIR+"ROC_comparison.pdf")
    #plt.savefig(PLOTDIR+"ROC_comparison.png")
    #print "Info: ROC curve file "+PLOTDIR+"ROC_comparison.pdf has been created"
    #fpr, tpr, thresholds = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])
    #cut_fpr = 0.00040904540701505433
    #idx, _ = find_nearest(fpr,cut_fpr)

                

#plot_ROC_supplementary(ERA="2016",extra_label="")
#plot_ROC_supplementary(ERA="2017",extra_label="")
plot_ROC_supplementary(ERA="2018",extra_label="")

