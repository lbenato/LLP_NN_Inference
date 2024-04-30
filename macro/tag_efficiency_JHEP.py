#! /usr/bin/env python

import os, multiprocessing
import subprocess
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import time
#import uproot
#import pandas as pd
import gc
import random
from array import array
##from awkward import *
#import awkward
#import root_numpy
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox, TRandom3
#from ROOT import RDataFrame
from ctypes import c_double
#from scipy.interpolate import Rbf, interp1d
#from scipy.interpolate import NearestNDInterpolator
#from scipy.interpolate import LinearNDInterpolator

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict, OrderedDict
#from itertools import chain
#import tensorflow as tf
#from tensorflow import keras

########## SETTINGS ##########

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-v", "--variable", action="store", type="string", dest="variable", default="met_pt_nomu")#"nPV")
parser.add_option("-c", "--cut", action="store", type="string", dest="cut", default="met_test")
parser.add_option("-C", "--compare", action="store", type="string", dest="compare", default="")
parser.add_option("-D", "--drawsignal", action="store_true", dest="drawsignal", default=False)
parser.add_option("-n", "--normalized", action="store_true", dest="normalized", default=False)
parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="mu")#"mu3nPV"
parser.add_option("-r", "--run", action="store", type="string", dest="run", default="G")
parser.add_option("-e", "--efficiency", action="store_true", dest="efficiency", default=False)
parser.add_option("-s", "--sample", action="store", type="string", dest="sample", default="All")
parser.add_option("-g", "--goodplots", action="store_true", default=False, dest="goodplots")#not needed in 2016
parser.add_option("-a", "--all", action="store_true", default=False, dest="all")
parser.add_option("-b", "--bash", action="store_true", default=True, dest="bash")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
parser.add_option("-f", "--final", action="store_true", default=False, dest="final")
parser.add_option("-R", "--rebin", action="store_true", default=False, dest="rebin")
parser.add_option("-p", "--public", action="store_true", default=False, dest="public")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)
#gStyle.SetPadTopMargin(-0.05)
#gStyle.SetPadBottomMargin(-0.05)
#gStyle.SetPadRightMargin(-0.2)
#gStyle.SetPadLeftMargin(-0.2)

#ERA
ERA                = "2017"
REGION             = "SR"#"SR"#"HBHE"#"ZtoEEBoost"#"WtoMN"#"WtoEN"
CUT                = "isSR"#"isSR"#"isSRHBHE"#"isZtoEE"#"isWtoMN"#"isWtoEN"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"
#KILL_QCD           = True#False
#DO_ETA             = True
#DO_PHI             = False#False#
#if DO_PHI:
#    DO_ETA = False
#CUT_ETA            = True#True#True#False#True#True#False
CUT_PHI            = True
BLIND              = False
TOYS               = True

print "\n"
print "region: ", REGION
#print "kill qcd: ", KILL_QCD
#print "do eta: ", DO_ETA
#print "do phi: ", DO_PHI
#print "eta cut: ", CUT_ETA
#print "phi cut: ", CUT_PHI
print "\n"



#REGION             = "JetMET"#"WtoEN"#"SR_HEM"#"debug_new_with_overlap"#"SR"#"ZtoMM_CR"

#CUT                = "isJetMET_dPhi_Lep"
#CUT                = "isJetMET_low_dPhi_Lep"
#CUT                = "isJetMET_dPhi_MET_200_Lep"
#CUT                = "isJetMET_low_dPhi_MET_200_Lep"

#CUT                = "isJetMET_dPhi"
#CUT                = "isJetMET_low_dPhi_500"#THIS!
#CUT                = "isJetMET_low_dPhi"#---> shows signal!
#CUT                = "isJetMET_dPhi_MET_200"
#CUT                = "isJetMET_low_dPhi_MET_200"

#CUT                = "isJetMET_dPhi_MET"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"
#NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_ZtoMM_CR/"
#PLOTDIR            = "plots/Efficiency/v5_calo_AOD_2018_ZtoMM_CR/"
NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
if REGION=="ZtoMMBoost":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_ZtoMM/"
if REGION=="ZtoEEBoost":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_ZtoEE/"
if REGION=="WtoEN":
    print "This for data:"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
    #print "This for MC"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN/"
if REGION=="WtoMN":
    print "This for data:"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"
    #print "This for MC"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN/"
if REGION=="SRtoEN":
    print "This for data:"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
    #print "This for MC"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN/"
if REGION=="SRtoMN":
    print "This for data:"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"
    #print "This for MC"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN/"
if REGION=="WtoEN_MET":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
if REGION=="WtoMN_MET":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"
if REGION=="HBHE":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_HBHE/"
if REGION=="SR":
    print "SR in v6"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"

PRE_PLOTDIR        = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"

#Approval:
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_unblinding_ARC/"
#CWR
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_CWR/"
#JHEP
PLOTDIR = "plots/Efficiency_JHEP/v6_calo_AOD_"+ERA+"_"+REGION+"/"

if REGION=="SR":
    #For approval:
    #YIELDDIR_BASE      = "plots/Yields_AN_fix_ARC_xcheck/v6_calo_AOD_"+ERA+"_"
    #YIELDDIR           = "plots/Yields_AN_fix_ARC_xcheck/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    #UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_fix/"
    #For CWR:
    YIELDDIR_BASE      = "plots/Yields_CWR/v6_calo_AOD_"+ERA+"_"
    YIELDDIR           = "plots/Yields_CWR/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_CWR/"
    #For JHEP:
    YIELDDIR_BASE      = "plots/Yields_JHEP/v6_calo_AOD_"+ERA+"_"
    YIELDDIR           = "plots/Yields_JHEP/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_JHEP/"

else:
    YIELDDIR_BASE      = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"
    YIELDDIR_BASE      = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"
    YIELDDIR           = PLOTDIR
    UNCDIR             = ""

CHAN               = "SUSY"
particle           = "#chi"
ctaupoint          = 500

#A bit less points:
ctaus_500          = np.array([10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 25000, 50000, 100000])
ctaus = np.unique(np.concatenate((ctaus_500,ctaus_3000)))

signalMultFactor   = 0.001#!!!
signalBRfactor     = 0.9
PRELIMINARY        = False
TAGVAR             = "nTagJets_0p996_JJ"

REBIN              = options.rebin
SAVE               = True

back = ["DYJetsToLL"]
back = ["All"]

data = ["SingleMuon"]
data = ["HighMET"]

sign = [
    'SUSY_mh127_ctau500',
    'SUSY_mh150_ctau500',
    'SUSY_mh175_ctau500',
    'SUSY_mh200_ctau500',
    'SUSY_mh250_ctau500',
    'SUSY_mh300_ctau500',
    'SUSY_mh400_ctau500',
    'SUSY_mh600_ctau500',
    'SUSY_mh800_ctau500',
    'SUSY_mh1000_ctau500',
    'SUSY_mh1250_ctau500',
    'SUSY_mh1500_ctau500',
    'SUSY_mh1800_ctau500',
    'SUSY_mh127_ctau3000',
    'SUSY_mh150_ctau3000',
    'SUSY_mh175_ctau3000',
    'SUSY_mh200_ctau3000',
    'SUSY_mh250_ctau3000',
    'SUSY_mh300_ctau3000',
    'SUSY_mh400_ctau3000',
    'SUSY_mh600_ctau3000',
    'SUSY_mh800_ctau3000',
    'SUSY_mh1000_ctau3000',
    'SUSY_mh1250_ctau3000',
    'SUSY_mh1500_ctau3000',
    'SUSY_mh1800_ctau3000',

]


if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    #print "Fake jet tag"
    jet_tag=""
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]


COMPARE = options.compare
DRAWSIGNAL = options.drawsignal

########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]

colors = [418,2,801,4]
lines = [2,1,3,4,1,2,2,2,2]
markers = [20,20,20,20,20,24,24,24,24]
markers = [20,21,24,25,20,24,24,24,24]
markers_MC = [20,20,20,20,20,24,24,24,24]
siz = 1.3
marker_sizes = [siz,siz,siz,siz,siz,siz,siz,siz,siz]
#markers = [20,21]#[24,25,21,20]
########## ######## ##########

#gROOT.SetBatch(True)

#cut_den = CUT
#cut_num = cut_den + " && "
chain = {}
filename = {}
hist_den = {}
hist_num = {}
hist_num_cutbased = {}
eff = {} 
eff_cutbased = {} 
dnn_bins = array('d', [0.,.00001,.0001,0.001,.01,.05,.1,.25,.5,.75,1.,])
#less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,10000])
#less_bins_plot = array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#New version from Jiajing
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])
less_bins_pt = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500])#,10000])
less_bins_pt = array('d', [1,10,20,30,40,50,60,75,100,200,500])#,10000])#this is good
less_bins_plot = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])

#print "Warning, overflow causing nans, remove last bin"
np_bins = np.array(less_bins)
np_bins = np_bins[np_bins>=30]#only interested in non-zero bins
np_bins = np_bins[np_bins<10000]#only interested in non-nan bins

#bins=array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#bins=np.array([1,10,20,30,40,50,60,70,80,90,100,1000])
#bins = bins.astype(np.float32)

prev_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,10000])
bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,250,300,400,500,600,700,800,900,10000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000,2500,5000,10000,100000])
np_more_bins = np.array(more_bins)
#np_more_bins = np_more_bins[np_more_bins>=30]#only interested in non-zero bins

#more_bins_eta = array('d', [-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
more_bins_eta = array('d',[-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1.,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5 ])
#less_bins_eta = array('d',[-1.5, -1.25, -1., -0.5, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.85, 1., 1.25, 1.5])
#homogeneous:
less_bins_eta = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
#more_bins=np.array([1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
#more_bins = more_bins.astype(np.float32)
###more_bins = bins

more_bins_phi = array('d',[-3.2 , -3.04, -2.88, -2.72, -2.56, -2.4 , -2.24, -2.08, -1.92,
                           -1.76, -1.6 , -1.44, -1.28, -1.12, -0.96, -0.8 , -0.64, -0.48,
                           -0.32, -0.16,  0.  ,  0.16,  0.32,  0.48,  0.64,  0.8 ,  0.96,
                           1.12,  1.28,  1.44,  1.6 ,  1.76,  1.92,  2.08,  2.24,  2.4 ,
                           2.56,  2.72,  2.88,  3.04,  3.2                       ])
less_bins_phi = array('d',[
    -3.2 , -2.56, -1.92, -1.28, -0.64,  0.  ,  0.64,  1.28,  1.92,
    2.56,  3.2
])

np_bins_eta = np.array(less_bins_eta)
np_bins_eta = np_bins_eta[0:-1]

maxeff = 0.0015#15#08#2#15

def round_to_1(x):
    return round(x, -int(math.floor(math.log10(abs(x)))))

def interpolate2D(x,y,z,hist,epsilon=0.2,smooth=0,norm = 'euclidean', inter = 'linear'):

    binWidthX = float(hist.GetXaxis().GetBinWidth(1))
    binWidthY = float(hist.GetYaxis().GetBinWidth(1))

    mgMin = hist.GetXaxis().GetBinCenter(1)
    mgMax = hist.GetXaxis().GetBinCenter(hist.GetNbinsX())#+hist.GetXaxis().GetBinWidth(hist.GetNbinsX())
    mchiMin = hist.GetYaxis().GetBinCenter(1)
    mchiMax = hist.GetYaxis().GetBinCenter(hist.GetNbinsY())#+hist.GetYaxis().GetBinWidth(hist.GetNbinsY())
    myX = np.linspace(mgMin, mgMax, hist.GetNbinsX())
    myY = np.linspace(mchiMin, mchiMax, hist.GetNbinsY())
    myXI, myYI = np.meshgrid(myX,myY)

    if inter == 'linear':rbf = LinearNDInterpolator(list(zip(x, y)), z)
    else: rbf = Rbf(x, y, z, function='multiquadric', epsilon=epsilon,smooth=smooth, norm = norm)

    myZI = rbf(myXI, myYI)
    for i in range(1, hist.GetNbinsX()+1):
        for j in range(1, hist.GetNbinsY()+1):
            hist.SetBinContent(i,j,10**(myZI[j-1][i-1]))
    return hist

def log_scale_conversion(h):
    
    ##########################
    # convert x and y axis to mass/ctau
    ###########################

    oldX = []
    for i_bin in range(1, h.GetNbinsX()+1):
        oldX.append(h.GetXaxis().GetBinLowEdge(i_bin))
    oldX.append(h.GetXaxis().GetBinUpEdge(h.GetNbinsX()))
    
    oldY = []
    for i_bin in range(1, h.GetNbinsY()+1):
        oldY.append(h.GetYaxis().GetBinLowEdge(i_bin))
    oldY.append(h.GetYaxis().GetBinUpEdge(h.GetNbinsY()))
    
    myX = 10**np.array(oldX)
    myY = 10**np.array(oldY)
    h_new = TH2D('', '', len(myX)-1, array('f',myX), len(myY)-1, array('f', myY))
    for i in range(1, h.GetNbinsX()+1):
        for j in range(1, h.GetNbinsY()+1):
            h_new.SetBinContent(i,j,h.GetBinContent(i,j))
    h_new.GetXaxis().SetTitle(h.GetXaxis().GetTitle())
    h_new.GetYaxis().SetTitle(h.GetYaxis().GetTitle())
    h_new.GetZaxis().SetTitle(h.GetZaxis().GetTitle())
    return h_new

def tau_weight_calc(llp_ct, new_ctau, old_ctau):
    '''
    llp_ct is a numpy array
    new_ctau and old_ctau are float
    must be converted in cm
    '''
    source = np.exp(-1.0*llp_ct/old_ctau)/old_ctau**2
    weight = 1.0/new_ctau**2 * np.exp(-1.0*llp_ct/new_ctau)/source
    #weight = (old_ctau**2/new_ctau**2)*np.exp(llp_ct*( 1./old_ctau -1./new_ctau))
    return weight

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = abs(p1 - p2)
    if res > math.pi:
        res -= 2*math.pi
    return res

def deltaR( e1, p1, e2, p2):
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return math.sqrt(de*de + dp*dp)


def get_tree_weights(sample_list,dataset_label="",main_pred_sample="HighMET"):

    #Fix LUMI
    if ERA=="2016":
        if dataset_label == "_G-H":
            LUMI  = lumi[ main_pred_sample ]["G"]+lumi[ main_pred_sample ]["H"]#["tot"]
        elif dataset_label == "_B-F":
            LUMI  = lumi[ main_pred_sample ]["B"]+lumi[ main_pred_sample ]["C"]+lumi[ main_pred_sample ]["D"]+lumi[ main_pred_sample ]["E"]+lumi[ main_pred_sample ]["F"]#["tot"]
    else:
        LUMI  = lumi[ main_pred_sample ]["tot"]

    tree_w_dict = defaultdict(dict)
    for i, s in enumerate(sample_list):
        for l, ss in enumerate(samples[s]['files']):
            #Tree weight
            if ('Run201') in ss:
                t_w = 1.
            else:
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                nevents = filename.Get("c_nEvents").GetBinContent(1)
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    print "SUSY central, consider sample dictionary for nevents!"
                    nevents = sample[ss]['nevents']
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                filename.Close()
                if('GluGluH2_H2ToSSTobbbb') in ss:
                    xs = 1.
                elif('XXTo4J') in ss:
                    xs = 1.*0.001
                elif('gluinoGMSB') in ss:
                    xs = 1.*0.001
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.*0.001
                elif ('n3n2-n1-hbb-hbb') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                    #print "Scaling SUSY to its true excluded x-sec"
                    #xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                elif ('TChiHH') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('splitSUSY') in ss:
                    print "Scaling splitSUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    #print "But consider BR!"
                    xs = 1.
                    #xs *= sample[name]['BR']
                else:
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                print "LUMI ", LUMI
                print "xs ", xs
                print "nevents ", nevents
                t_w = LUMI * xs / nevents
                if(b_skipTrain>0):
                    #print("Skip even events: ")
                    #print "n_pass: ", n_pass
                    #print "n_odd: ", n_odd
                    #if(n_pass>0):
                    #    print "ratio: ", float(n_odd/n_pass)
                    if(n_odd>0):
                        #print "even/odd weight: ", float(n_pass/n_odd)
                        t_w *= float(n_pass/n_odd)
            print("%s has tree weight %f")%(ss,t_w)
            tree_w_dict[s][ss] = t_w

    return tree_w_dict


def get_tree_weights_BR_scan(sample_list,br_h,dataset_label="",main_pred_sample="HighMET"):

    #Fix this_lumi
    if ERA=="2016":
        if dataset_label == "_G-H":
            this_lumi  = lumi[ main_pred_sample ]["G"]+lumi[ main_pred_sample ]["H"]#["tot"]
        elif dataset_label == "_B-F":
            this_lumi  = lumi[ main_pred_sample ]["B"]+lumi[ main_pred_sample ]["C"]+lumi[ main_pred_sample ]["D"]+lumi[ main_pred_sample ]["E"]+lumi[ main_pred_sample ]["F"]#["tot"]
    else:
        this_lumi  = lumi[ main_pred_sample ]["tot"]

    def eq(H,Z):
        return H**2 + 2*H*Z + Z**2

    def retZ(H):
        #x**2 + 2*H*x + (H**2 - 1.) = 0
        return -H + 1

    h = br_h/100. if br_h>1 else br_h
    z = retZ(float(h))
    w_hh = h*h
    w_hz = 2*h*z
    w_zz = z*z
    print "\n"
    print "------------------------"
    print "BR: H(", h*100,"%), Z(", z*100,"%)"
    print ("Decay composition: HH %.2f, HZ %.2f, ZZ %.2f" % (w_hh*100, w_hz*100, w_zz*100))
    print "------------------------"

    tree_w_dict = defaultdict(dict)
    for i, pr in enumerate(sample_list):
        new_list = [pr+"_HH",pr+"_HZ",pr+"_ZZ"]
        for s in new_list:
            #print s
            for l, ss in enumerate(samples[s]['files']):
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                #Tree weight
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    #print "SUSY central, consider sample dictionary for nevents!"
                    nevents = sample[ss]['nevents']
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                #tree = filename.Get("tree")
                #print "tree entries: ", tree.GetEntries()
                filename.Close()
                if('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    #print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    #print "But consider BR!"
                    xs = 1.
                    xs *= sample[ss]['BR']
                print "LUMI ", this_lumi
                print "xs ", xs
                print "nevents ", nevents
                t_w = this_lumi * xs / nevents
                if(b_skipTrain>0):
                    if(n_odd>0):
                        t_w *= float(n_pass/n_odd)
                if "_HH" in s:
                    t_w *= w_hh
                if "_HZ" in s:
                    t_w *= w_hz
                if "_ZZ" in s:
                    t_w *= w_zz
                print("%s has tree weight %f")%(ss,t_w)
                tree_w_dict[pr][ss] = t_w

    print "\n"
    return tree_w_dict
    
def shape_prediction(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,plot_distr="",phi_cut=False):#,eta=False,phi=False,eta_cut=False):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"

    mult_factor = 1.
    if sample_list==sign:
        print "\n"
        print "   ** Performing estimation on signal! Rescaling signal by signalMultFactor ", signalMultFactor, " **"
        mult_factor = signalMultFactor
        print "\n"

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    
    print "Apply acceptance cut |eta|<1."
    add_label+="_eta_1p0"
    if phi_cut:
        add_label+="_phi_cut"
    add_label+="_vs_eta"

    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
        
    print "Reading input root files: ", NTUPLEDIR

    #Region-dependent  objects
    EFFDIR = {}
    TEff = {}
    eff = {}
    effUp = {}
    effDown = {}
    table_yield = {}
    table_pred = {}
    table_pred_new = {}
    table_integral = {}
    tag_bins = np.array([0,1,2,3,4,5,6])
    uproot_tree = OrderedDict()#defaultdict(dict)
    chain = {}
    hist = {}
    dfR = {}
    h0 = {}
    h1 = {}
    h2 = {}
    results = defaultdict(dict)
    row = {}
    rowP = {}
    rowPn = {}
    infiles = {}

    if extr_regions==[]:
        extr_regions.append(REGION)

    for i,r in enumerate(extr_regions):
        reg_label = ""
        dset = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]
        results[r+reg_label+dset] = {}
        EFFDIR[r+reg_label+dset] = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+ r + "/"
        table_yield[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err', 'Bin 2 Err combine'])
        table_pred[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
        table_pred_new[r+reg_label+dset] =  PrettyTable(['Bin 1 Pred', 'Bin 1 Discr. %', 'Bin 2 Pred from 0', 'Bin 2 from 0 Discr. %', 'Bin 2 Pred from 1', 'Bin 2 from 1 Discr. %', 'Bin 2 Pred Stat Err', 'Bin 1 Pred Stat Err'])
        table_integral[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err'])

    for i,r in enumerate(extr_regions):
        if "ZtoMM" in r or "WtoMN" in r or "MN" in r:
            eff_name="SingleMuon"
        elif "ZtoEE" in r or "WtoEN" in r or "EN" in r:
            if ERA=="2018":
                eff_name="EGamma"
            else:
                eff_name="SingleElectron"
        elif "TtoEM" in r:
            eff_name=r#"MuonEG"
        elif "SR" in r:
            eff_name=r#"HighMET"
        elif "HBHE" in r:
            eff_name=r#"HighMET"
        elif "JetHT" in r:
            eff_name="JetHT"
        elif "JetMET" in r:
            eff_name="JetHT"
        elif "MR" in r:
            eff_name="HighMET"
        elif "MRPho" in r:
            eff_name="HighMET"
        elif "SR_HEM" in r:
            eff_name="HighMET"
        elif "ZtoLL" in r or "WtoLN" in r:
            eff_name=r

        #HERE?
        if len(datasets)>0:
            if datasets[i]!="":
                eff_name = datasets[i]
                dset = datasets[i]
            else:
                #Otherwise it stores the one from before, wrong keys
                dset = ""

        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]

        infiles[r+reg_label+dset] = TFile(EFFDIR[r+reg_label+dset]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root", "READ")
        print "Opening TEff["+r+reg_label+dset+"]  " + EFFDIR[r+reg_label+dset]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root"
        TEff[r+reg_label+dset] = infiles[r+reg_label+dset].Get("TEff_"+eff_name)
        for s in sample_list:
            #Define dictionaries
            results[r+reg_label+dset][s] = {}
            
    for i, s in enumerate(sample_list):
        ##Define TH1F as a cross-check for event yields
        h0[s] = TH1F(s+"_0", s+"_0", 4, -1, 2)
        h0[s].Sumw2()
        h1[s] = TH1F(s+"_1", s+"_1", 4, -1, 2)
        h1[s].Sumw2()
        h2[s] = TH1F(s+"_2", s+"_2", 4, -1, 2)
        h2[s].Sumw2()

        print s

        ##Prepare TChain and determine tree weights
        ##Note: if you open the same file twice, project doesn't work
        ##Tree weight  must be determined in a separate function
        tree_weights = {}
        tree_weights_array = np.array([])
        chain_entries = {}
        chain_entries_cumulative = {}
        chain[s] = TChain("tree")
        list_files_for_uproot = []


        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.phi","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","HLT*PFMETNoMu*","Jets.timeRecHitsEB","Jets.cHadEFrac","Jets.nTrackConstituents",CUT]#"nLeptons"
        if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR and CUT=="isSR":
            list_of_variables += ["dt_ecal_dist","min_dPhi*"]
        if REGION=="ZtoMMBoost" or REGION=="ZtoEEBoost":
            list_of_variables += ["Z_pt"]
        if ERA=="2018" and CUT=="isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

        #The only thing we can afford in RAM are final numbers
        #hence only a few arrays
        b0 = np.array([])
        b1 = np.array([])
        b2 = np.array([])
        pr_1 = {}
        pr_2 = {}
        pr_2_from_1 = {}

        #plot_distr: here define variables in bins
        distr_0 = np.array([])
        distr_1 = np.array([])
        distr_2 = np.array([])
        distr_1_t = np.array([])
        distr_1_pr = {}
        distr_2_t = np.array([])
        distr_2_pr_from_0 = np.array([])
        distr_2_pr_from_1 = np.array([])
        weight_0 = np.array([])
        weight_1 = np.array([])
        weight_2 = np.array([])
        weight_1_t = np.array([])
        weight_1_pr_wrong = {}
        weight_1_pr = {}
        weight_2_t = np.array([])
        weight_2_pr_from_0 = np.array([])
        weight_2_pr_from_1 = np.array([])

        weight_pr_1 = {}
        weight_pr_2_from_0 = {}
        weight_pr_2_from_1 = {}

        hist_1_pr = {}

        nice_labels = {}
        short_labels = {}

        #pr_1/2 depend on TEff, dictionaries
        for r in TEff.keys():
            if "WtoLN" in r:
                nice_labels[r] = "Nominal MR"
                short_labels[r] = "WtoLN"
            if "ZtoLL" in r:
                nice_labels[r] = "Z+jets MR"
                short_labels[r] = "ZtoLL"
            if "TtoEM" in r:
                nice_labels[r] = "t#bar{t} MR"
                short_labels[r] = "TtoEM"
            if "JetHT" in r:
                nice_labels[r] = "QCD MR"
                short_labels[r] = "JetHT"
            pr_1[r] = np.array([])#[]
            pr_2[r] = np.array([])#[]
            pr_2_from_1[r] = np.array([])#[]

            #plot_distr: here define variables in bins
            weight_pr_1[r] = np.array([])
            weight_pr_2_from_0[r] = np.array([])
            weight_pr_2_from_1[r] = np.array([])

            distr_1_pr[r] = np.array([])
            weight_1_pr[r] = np.array([])
            
            eff[r] = []
            effUp[r] = []
            effDown[r] = []
            #0. Fill efficiencies np arrays once and forever
            for b in np_bins_eta:
                binN = TEff[r].GetPassedHistogram().FindBin(b)
                eff[r].append(TEff[r].GetEfficiency(binN))
                print(r," eta: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))                    
            print r, eff[r]

        #b0 etc filled at every chunk
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            #print "Entries in ", ss, " : ", chain[s].GetEntries()
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]

            #tree_weights_array = np.concatenate( (tree_weights_array,tree_weights[l] * np.ones(chain_entries[l])) )#np.concatenate( tree_weights_array, tree_weights[l] * np.ones(chain_entries[l]))
            print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            #import concurrent.futures
            #executor = concurrent.futures.ThreadPoolExecutor(max_workers=5) 
            #gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size,executor = executor)
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)

            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                        #cosmic
                        #print "With cosmic veto!"
                        cosmic_veto = arrays["dt_ecal_dist"]<0.5
                        cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if CUT=="isSR" and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))

                #print "   QCD killer cut!"
                cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)


                #Default cut_jets, does nothing basically                                                                    
                cut_jets = arrays["Jets.pt"]>-999
                cut_jets = np.logical_and(cut_mask,cut_jets)
                cut_mask = (cut_jets.any()==True)

                cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)                    
                cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                #This is needed to guarantee the shapes are consistent
                cut_mask = (cut_mask_phi_eta.any()==True)
                cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)#new

                #Beam Halo veto
                if CUT == "isSR":
                    if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                        cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
                        cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
                        cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
                        cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)
                        cut_mask = np.logical_and(cut_mask,cut_mask_bh)
                        cut_jets = np.logical_and(cut_jets,cut_mask)

                ##Fill pt and sigprob arrays #new
                pt = arrays["Jets.eta"][cut_jets][cut_mask]
                sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                runnumber = arrays["RunNumber"][cut_mask]
                luminumber = arrays["LumiNumber"][cut_mask]
                eventnumber = arrays["EventNumber"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]

                pt_v = arrays["Jets.pt"][cut_jets][cut_mask]
                eta_v = arrays["Jets.eta"][cut_jets][cut_mask]
                score_v = arrays["Jets.sigprob"][cut_jets][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*mult_factor

                n_obj = -1

                del arrays
                
                #dnn_threshold = 0.996
                tag_mask = (sigprob > dnn_threshold)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                pt_untag = pt[untag_mask]

                bin0_m = (sigprob[tag_mask].counts ==0)
                bin1_m = (sigprob[tag_mask].counts ==1)
                bin2_m = (sigprob[tag_mask].counts >1)

                bin0 = np.multiply(bin0_m,weight)
                bin1 = np.multiply(bin1_m,weight)
                bin2 = np.multiply(bin2_m,weight)


                #Per-chunk bin1 

                tmp_eta_0 = eta_v[bin0_m]
                tmp_w_0   = np.multiply(bin0_m,weight)[bin0_m]

                tmp_eta_1 = eta_v[bin1_m]
                tmp_score_1 = score_v[bin1_m]
                tmp_w_1   = np.multiply(bin1_m,weight)[bin1_m]

                tmp_eta_1_t = eta_v[tag_mask][bin1_m]
                tmp_score_1_t = score_v[tag_mask][bin1_m]
                tmp_w_1_t   = np.multiply(bin1_m,weight)[bin1_m]

                '''
                print "tmp_eta_0"
                print tmp_eta_0.shape
                print tmp_eta_0
                print tmp_w_0.shape
                print tmp_w_0

                print "tmp_eta_1"
                print tmp_eta_1.shape
                print tmp_eta_1
                print tmp_score_1.shape
                print tmp_score_1
                print tmp_w_1.shape
                print tmp_w_1
                '''

                distr_0    = np.concatenate( (distr_0,   np.hstack(tmp_eta_0)     if len(tmp_eta_0)>0    else np.array([])) )
                distr_1_t  = np.concatenate( (distr_1_t, np.hstack(tmp_eta_1_t)   if len(tmp_eta_1_t)>0  else np.array([])) )

                weight_0   = np.concatenate( (weight_0,   np.hstack(tmp_eta_0.astype(bool)*tmp_w_0) if len(tmp_eta_0)>0    else np.array([]) ) )
                weight_1_t = np.concatenate( (weight_1_t, np.hstack(tmp_eta_1_t.astype(bool)*tmp_w_1_t) if len(tmp_eta_1_t)>0    else np.array([]) ) )


                #Predictions
                bin1_pred = defaultdict(dict)
                bin2_pred = defaultdict(dict)
                bin2_pred_from_1 = defaultdict(dict)
                prob_vec = {}
                tmp_w_1_pr = {}

                tmp_eta_1_pr = eta_v[bin0_m]#this is pt_untag in the right bin

                for r in TEff.keys():
                    bin1_pred[r] = np.array([])
                    bin2_pred[r] = np.array([])
                    bin2_pred_from_1[r] = np.array([])
                    prob_vec[r] = []
                    ####tmp_w_1_pr[r] = np.array([])
                    #tmp_w_1_pr_wrong   = np.multiply(bin0_m,bin1_pred[r])[bin0_m]
                    tmp_w_1_pr[r] = tmp_eta_1_pr.astype(bool)*0

                #print "will sort this vector:"
                #print tmp_eta_1_pr.shape
                #print tmp_eta_1_pr

                for r in TEff.keys():
                    #print r
                    #print "at the moment it looks like:"
                    #print tmp_w_1_pr[r].shape
                    #print tmp_w_1_pr[r]

                    for i in range(len(np_bins_eta)):
                        if i<len(np_bins_eta)-1:
                            prob_vec[r].append(np.logical_and(pt_untag>=np_bins_eta[i],pt_untag<np_bins_eta[i+1])*eff[r][i])#*weight)
                            tmp_per_bin = np.logical_and(tmp_eta_1_pr>=np_bins_eta[i],tmp_eta_1_pr<np_bins_eta[i+1])*eff[r][i]
                        else:
                            prob_vec[r].append((pt_untag>=np_bins_eta[i])*eff[r][i])#*weight)
                            tmp_per_bin = (tmp_eta_1_pr>=np_bins_eta[i])*eff[r][i]
                            #HERE just multiply and check size

                        #print "test per bin i: ", i
                        #print tmp_per_bin.shape
                        #print tmp_per_bin
                        #print "now sum!"
                        tmp_w_1_pr[r] = tmp_w_1_pr[r] + tmp_per_bin

                    #print "summed up: "
                    #print tmp_w_1_pr[r].shape
                    #print tmp_w_1_pr[r]

                    prob_tot = sum(prob_vec[r])
                    somma = (prob_tot*weight).sum()
                    cho = prob_tot.choose(2)
                    combi = cho.unzip()[0] * cho.unzip()[1] * weight

                    bin1_pred[r] = np.concatenate( (bin1_pred[r],somma) )
                    bin2_pred[r] = np.concatenate( (bin2_pred[r],combi.sum()) )
                    bin2_pred_from_1[r] = np.concatenate( (bin2_pred_from_1[r], somma[bin1_m]  )  )#.append(0.)

                    #print "checking the shapes"
                    #print tmp_eta_1_pr.shape
                    #print np.hstack(tmp_eta_1_pr).shape
                    #print tmp_w_1_pr[r].shape
                    #print np.hstack(tmp_w_1_pr[r]).shape

                    distr_1_pr[r] = np.concatenate( (distr_1_pr[r], np.hstack(tmp_eta_1_pr)  if len(tmp_eta_1_pr)>0 else np.array([])) )
                    weight_1_pr[r] = np.concatenate( (weight_1_pr[r], np.hstack(tmp_w_1_pr[r]) if len(tmp_eta_1_pr)>0  else np.array([]) ) )
                    #*weight[bin0_m]? needed?
                
                #Here: still in chunk loop
                #Here concatenate to full pred
                for r in TEff.keys():
                    pr_1[r] = np.concatenate((pr_1[r],bin1_pred[r]))
                    pr_2[r] = np.concatenate((pr_2[r],bin2_pred[r]))
                    pr_2_from_1[r] = np.concatenate((pr_2_from_1[r],bin2_pred_from_1[r]))


                b0 = np.concatenate((b0,bin0))
                b1 = np.concatenate((b1,bin1))
                b2 = np.concatenate((b2,bin2))


                #del tmp
                en_it = time.time()

                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"
                    c+=1


            del gen

        end_uproot = time.time()
        print "\n"
        print "   --- > Tot size of arrays: ", array_size_tot
        print "Size of tree_weights_array: ", len(tree_weights_array)
        print "Time elapsed to fill uproot array: ", end_uproot-start_uproot
        print "************************************"
        print "\n"

        #Plotting variables
        if plot_distr!="":

            print distr_1_pr[r].shape
            print weight_1_pr[r].shape

            bins = np.array([-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
            hist_0    = TH1D("bin0", "bin0", len(less_bins_eta)-1,less_bins_eta)
            hist_1  = TH1D("SR", "SR", len(less_bins_eta)-1,less_bins_eta)
            hist_0.Sumw2()
            hist_1.Sumw2()
            _ = root_numpy.fill_hist( hist_0, distr_0, weights=weight_0 )
            _ = root_numpy.fill_hist( hist_1, distr_1_t, weights=weight_1_t )
            for r in TEff.keys():
                hist_1_pr[r] = TH1D(short_labels[r], short_labels[r], len(less_bins_eta)-1,less_bins_eta)
                hist_1_pr[r].Sumw2()
                _ = root_numpy.fill_hist( hist_1_pr[r], distr_1_pr[r], weights=weight_1_pr[r] )


            #Plot bin0
            can0 = TCanvas("bin0","bin0", 1000, 900)
            can0.cd()
            if variable[plot_distr]["log"]==True:
                can0.SetLogy() 
            leg0 = TLegend(0.5, 0.7, 0.9, 0.9)
            leg0.SetTextSize(0.035)
            leg0.SetBorderSize(0)
            leg0.SetFillStyle(0)
            leg0.SetFillColor(0)
            hist_0.SetTitle("")
            hist_0.GetXaxis().SetTitle(variable[plot_distr]["title"])
            hist_0.GetYaxis().SetTitle("Events")
            hist_0.SetMarkerSize(1.2)
            hist_0.SetLineWidth(2)
            hist_0.SetMarkerStyle(21)
            hist_0.SetLineColor(colors[0])
            hist_0.SetMarkerColor(colors[0])
            hist_0.Draw("PL")
            leg0.AddEntry(hist_0,"bin 0; "+r,"PL")
            leg0.Draw()
            can0.Print(PLOTDIR+'Bin0_'+plot_distr.replace('.', '_')+label_2+'.png')
            can0.Print(PLOTDIR+'Bin0_'+plot_distr.replace('.', '_')+label_2+'.pdf')


            #Plot bin1
            can1 = TCanvas("bin1","bin1", 1000, 900)
            can1.cd()
            if variable[plot_distr]["log"]==True:
                can1.SetLogy() 
            leg1 = TLegend(0.5, 0.7, 0.9, 0.9)
            leg1.SetTextSize(0.035)
            leg1.SetBorderSize(0)
            leg1.SetFillStyle(0)
            leg1.SetFillColor(0)
            hist_1.SetTitle("")
            hist_1.GetXaxis().SetTitle(variable[plot_distr]["title"])
            hist_1.GetYaxis().SetTitle("Events")
            hist_1.SetMarkerSize(1.2)
            hist_1.SetLineWidth(2)
            hist_1.SetMarkerStyle(20)
            hist_1.SetLineColor(1)
            hist_1.SetMarkerColor(1)
            leg1.AddEntry(hist_1,"bin 1","PL")
            
            count = 0
            for r in TEff.keys():
                hist_1_pr[r].SetMarkerSize(1.2)
                hist_1_pr[r].SetLineWidth(2)
                hist_1_pr[r].SetMarkerStyle(24)
                hist_1_pr[r].SetLineColor(colors[count])
                hist_1_pr[r].SetLineStyle(2)
                hist_1_pr[r].SetMarkerColor(colors[count])
                hist_1_pr[r].SetTitle("")
                hist_1_pr[r].GetXaxis().SetTitle(variable[plot_distr]["title"])
                hist_1_pr[r].GetYaxis().SetTitle("Events")
                hist_1_pr[r].SetMinimum(0.9)
                if count==0:
                    hist_1_pr[r].Draw("PL")
                else:
                    hist_1_pr[r].Draw("PL,SAMES")
                leg1.AddEntry(hist_1_pr[r],"bin 1 pred.; "+nice_labels[r],"PL")
                count+=1

            can1.cd()
            hist_1.Draw("PL,sames")
            leg1.Draw()
            can1.Print(PLOTDIR+'Bin1_'+plot_distr.replace('.', '_')+label_2+'.png')
            can1.Print(PLOTDIR+'Bin1_'+plot_distr.replace('.', '_')+label_2+'.pdf')

            outfile = TFile(PLOTDIR+"Bin1_"+plot_distr.replace('.', '_')+label_2+".root","RECREATE")
            outfile.cd()
            for r in TEff.keys():
                hist_1_pr[r].Write(short_labels[r])
            hist_1.Write("SR")
            print "Info in <TFile::Write>: root file "+PLOTDIR+"Bin1_"+plot_distr.replace('.', '_')+label_2+".root has been created"
            outfile.Close()

            exit()


        y_0 = np.sum(np.array(b0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(b1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(b2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in b0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in b1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in b2) )#*tree_weight --> already in w


        #Predictions are now dictionaries
        pred_1 = {}
        e_pred_1 = {}
        pred_up_1 = {}
        e_pred_up_1 = {}
        pred_low_1 = {}
        e_pred_low_1 = {}
        pred_2 = {}
        e_pred_2 = {}
        pred_2_from_1 = {}
        e_pred_2_from_1 = {}
        sorted_pr_2 = {}
        sorted_pr_2_from_1 = {}

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
                if len(datasets)>0:
                    dset = datasets[i]

            sorted_pr_2[r+reg_label+dset] = -np.sort(-pr_2[r+reg_label+dset])
            sorted_pr_2_from_1[r+reg_label+dset] = -np.sort(-pr_2_from_1[r+reg_label+dset])
            pred_1[r+reg_label+dset] = np.sum(np.array(pr_1[r+reg_label+dset]))
            e_pred_1[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_1[r+reg_label+dset]) ).sum()
            pred_2[r+reg_label+dset] = np.sum(np.array(pr_2[r+reg_label+dset]))
            e_pred_2[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_2[r+reg_label+dset]) ).sum()
            pred_2_from_1[r+reg_label+dset] = np.sum(np.array(pr_2_from_1[r+reg_label+dset]))
            e_pred_2_from_1[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_2_from_1[r+reg_label+dset]) )

            results[r+reg_label+dset][s]["y_0"] = y_0
            results[r+reg_label+dset][s]["e_0"] = e_0
            results[r+reg_label+dset][s]["y_1"] = y_1
            results[r+reg_label+dset][s]["e_1"] = e_1
            results[r+reg_label+dset][s]["y_2"] = y_2
            results[r+reg_label+dset][s]["e_2"] = e_2
            results[r+reg_label+dset][s]["pred_1"] = pred_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_1"] = e_pred_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["pred_2"] = pred_2[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_2"] = e_pred_2[r+reg_label+dset]
            results[r+reg_label+dset][s]["pred_2_from_1"] = pred_2_from_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_2_from_1"] = e_pred_2_from_1[r+reg_label+dset]

            print "round to 5"
            round_val = 5#2
            round_pred = 5#4
            row[r+reg_label+dset] = [s, round(y_0,round_val), round(e_0,round_val), round(y_1,round_val), round(e_1,round_val), round(y_2,5), round(e_2,round_val),round( (1. + e_2/y_2),5)]
            table_yield[r+reg_label+dset].add_row(row[r+reg_label+dset])

            rowP[r+reg_label+dset] = [s, round(pred_1[r+reg_label+dset],round_val), round(e_pred_1[r+reg_label+dset],round_val), round(pred_2[r+reg_label+dset],round_pred), round(e_pred_2[r+reg_label+dset],round_pred), round(pred_2_from_1[r+reg_label+dset],round_pred), round(e_pred_2_from_1[r+reg_label+dset],round_pred)]
            table_pred[r+reg_label+dset].add_row(rowP[r+reg_label+dset])

            rowPn[r+reg_label+dset] = [round(pred_1[r+reg_label+dset],round_val), round( 100*(pred_1[r+reg_label+dset]-y_1)/y_1,round_val), round(pred_2[r+reg_label+dset],round_pred), round(100*(pred_2[r+reg_label+dset]-y_2)/y_2,round_pred), round(pred_2_from_1[r+reg_label+dset],round_pred), round(100*(pred_2_from_1[r+reg_label+dset]-y_2)/y_2,round_pred), round(e_pred_2_from_1[r+reg_label+dset],round_pred), round(e_pred_1[r+reg_label+dset],round_pred)]
            table_pred_new[r+reg_label+dset].add_row(rowPn[r+reg_label+dset])


        #Here I would like to keep the name of the samples
        #Try to align here:

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
                if len(datasets)>0:
                    dset = datasets[i]

            ##if i==0:
            print '\n\n======================= Yields and predictions extrapolated from '+r+reg_label+dset+' ==============================' 
            print(table_yield[r+reg_label+dset])
            print(table_pred_new[r+reg_label+dset])
            print(table_pred[r+reg_label+dset])

        wr  = False#True#False#False

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]
            if wr:
                with open(PLOTDIR+'BkgPred_extr_region_'+r+reg_label+dset+add_label+label_2+'.txt', 'w') as w:
                    w.write('\n\n======================= Yields and predictions extrapolated from '+r+' ==============================\n')
                    w.write(str(table_yield[r+reg_label+dset])+"\n")
                    w.write(str(table_pred[r+reg_label+dset])+"\n")
                    w.write(str(table_pred_new[r+reg_label+dset])+"\n")
                    w.write('\n\n======================= From histogram  ==============================\n')
                    w.close()
                    print "Info: tables written in file "+PLOTDIR+"BkgPred_extr_region_"+r+reg_label+dset+add_label+label_2+".txt"
            else:
                print "NO tables written in file !!!!!!"    


        if wr:
            with open(YIELDDIR+"BkgPredResults_"+ERA+"_"+REGION+ "_"+s+ add_label+label_2+".yaml","w") as f:
                yaml.dump(results, f)
                f.close()
                print "Info: dictionary written in file "+YIELDDIR+"BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".yaml"


samples_to_run = data
jet_tag = ""#+
clos = False
dataset_tag = ""
#kill QCD
#ERA
if ERA=="2016":
    #jet_tag+="_B-F"
    jet_tag+="_G-H"#"_B-F"#"_B-F"#"_G-H"#"_G-H"#"_B-F"#"_B-F"#
    dataset_tag +=jet_tag

jet_tag += "_MinDPhi_0p5"

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2018"
        MINPHI = 0.9 #min allowed
        MAXPHI = 0.4 #max allowed
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2017"
        MINPHI = 3.5 #min allowed
        MAXPHI = 2.7 #max allowed
    #print "Fake lumi of 100 /fb"
    #LUMI = 1000.*1000.*100

elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if CUT_PHI==True:
        print "Here phi cuts 2016 --> to be re-set as false"
        CUT_PHI=False
        MINPHI = -9 #min allowed
        MAXPHI = 9 #max allowed
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

print "Ntupledir: ", NTUPLEDIR

sample_dict = {}
reg_comb = ""
isMC = False
if samples_to_run==back:
    isMC = True
    if "Wto" in REGION:
        sample_dict["WtoMN"] = "WJetsToLNu"
        sample_dict["WtoEN"] = "WJetsToLNu"
        reg_comb = "WtoLN"
    elif "Zto" in REGION:
        sample_dict["ZtoMM"] = "DYJetsToLL"
        sample_dict["ZtoEE"] = "DYJetsToLL"
        reg_comb = "ZtoLL"
    else:
        sample_dict[REGION] = samples_to_run[0]
        print "This won't work for lists, hence MC... keep in mind"
        reg_comb = REGION
else:
    if "Wto" in REGION:
        #sample_dict["WtoMN"] = "SingleMuon"
        sample_dict["WtoMN_MET"] = "SingleMuon"
        if ERA=="2018":
            #sample_dict["WtoEN"] = "EGamma"
            sample_dict["WtoEN_MET"] = "EGamma"
        else:
            if "_B-F" not in jet_tag:
                #sample_dict["WtoEN"] = "SingleElectron"
                sample_dict["WtoEN_MET"] = "SingleElectron"
        reg_comb = "WtoLN"
        if "MET" in REGION:
            reg_comb = "WtoLN_MET"
    elif "Zto" in REGION:
        sample_dict["ZtoMM"] = "SingleMuon"
        if ERA=="2018":
            sample_dict["ZtoEE"] = "EGamma"
        else:
            if "_B-F" not in jet_tag:
                sample_dict["ZtoEE"] = "SingleElectron"
        reg_comb = "ZtoLL"
        if "Boost" in REGION:
            sample_dict = {}
            sample_dict["ZtoMMBoost"] = "SingleMuon"
            if ERA=="2018":
                sample_dict["ZtoEEBoost"] = "EGamma"
            else:
                if "_B-F" not in jet_tag:
                    sample_dict["ZtoEEBoost"] = "SingleElectron"
            reg_comb = "ZtoLLBoost"
    else:
        sample_dict[REGION] = samples_to_run[0]
        print "This won't work for lists, hence MC... keep in mind"
        reg_comb = REGION

sign = ["SUSY_mh400_ctau500"]
print get_tree_weights_BR_scan(sign,100)

sign = ["SUSY_mh400_ctau500_HH"]
print get_tree_weights(sign)

sign = ["SUSY_mh400_ctau500_HZ"]
print get_tree_weights(sign)

sign = ["SUSY_mh400_ctau500_ZZ"]
print get_tree_weights(sign)

exit()
#Bkg pred

shape_prediction(
    get_tree_weights(samples_to_run,LUMI),
    samples_to_run,
    extr_regions = ["WtoLN","ZtoLL","TtoEM","JetHT",],
    regions_labels = [jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    add_label="",
    label_2=dataset_tag,
    check_closure=clos,
    plot_distr = "Jets.eta",#"Jets[0].pt",#"nCHSJets",#"Jets[2].phi",#"Jets.sigprob"#"nTagJets_0p996_JJ"#
    #eta = DO_ETA,
    #phi = DO_PHI,
    #eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
)
