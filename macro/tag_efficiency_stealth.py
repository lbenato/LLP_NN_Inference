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
import uproot
import pandas as pd
import gc
import random
import csv
from array import array
#from awkward import *
import awkward
import root_numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox, TRandom3
from ROOT import RDataFrame
from ctypes import c_double
from scipy.interpolate import Rbf, interp1d
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict, OrderedDict
from itertools import chain
import tensorflow as tf
from tensorflow import keras

########## SETTINGS ##########

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-r", "--run_era", action="store", type="string", dest="run_era", default="2018")
parser.add_option("-e", "--era_str", action="store", type="string", dest="era_str", default="")
parser.add_option("-c", "--decay_ch", action="store", type="string", dest="decay_ch", default="SHH")
parser.add_option("-t", "--mass_stop", action="store", type="int", dest="mass_stop", default=0)
parser.add_option("-s", "--mass_s", action="store", type="int", dest="mass_s", default=100)
parser.add_option("-b", "--bash", action="store_true", default=True, dest="bash")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)

#ERA
ERA                = options.run_era#"2018"
REGION             = "SR"#"SR"#"HBHE"#"ZtoEEBoost"#"WtoMN"#"WtoEN"
CUT                = "isSR"#"isSR"#"isSRHBHE"#"isZtoEE"#"isWtoMN"#"isWtoEN"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"
KILL_QCD           = True#False
DO_ETA             = True
DO_PHI             = False#False#
if DO_PHI:
    DO_ETA = False
CUT_ETA            = True#True#True#False#True#True#False
CUT_PHI            = True
BLIND              = False
TOYS               = True

print "\n"
print "era: ", ERA
print "region: ", REGION
print "kill qcd: ", KILL_QCD
print "do eta: ", DO_ETA
print "do phi: ", DO_PHI
print "eta cut: ", CUT_ETA
print "phi cut: ", CUT_PHI
print "\n"

NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
if REGION=="SR":
    print "SR in v6, DS review paper"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared/"

PRE_PLOTDIR        = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"
PRE_PLOTDIR        = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"
PRE_PLOTDIR        = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"
#PRE_PLOTDIR        = "plots/Efficiency_AN_additional_material/v5_calo_AOD_"+ERA+"_"

PLOTDIR            = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
PLOTDIR            = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
PLOTDIR            = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#

PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_TEST_remove_outliers/"
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_preapproval/"
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_unblinding/"
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_unblinding_one_sided_Si/"
#Approval:
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_unblinding_ARC/"
#CWR
PLOTDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_CWR/"

#PLOTDIR            = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_v3_tagger_pt_flat/"#"_2017_signal/"#
#PRE_PLOTDIR        = PLOTDIR

#Time smearing file
OUT_TtoEM = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_TtoEM_v5_ntuples_validate_timeRecHits/"

#Zgamma SF file
OUT_pho = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_ZtoLLPho_v5_ntuples_updated/smearing/"

#Ele SF file
OUT_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_E_v5_ntuples_reweighted/weighted/smearing/"

#PLOTDIR            = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_tagger_v3_no_time/"#"_2017_signal/"#
#PRE_PLOTDIR        = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_tagger_v3_no_time/"#"_2017_signal/"#

#PLOTDIR            = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_tagger_v3_pt_weighted/"#"_2017_signal/"#
#PRE_PLOTDIR        = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"_tagger_v3_pt_weighted/"#"_2017_signal/"#


if REGION=="SR":
    #For approval:
    #YIELDDIR_BASE      = "plots/Yields_AN_fix_ARC_xcheck/v6_calo_AOD_"+ERA+"_"
    #YIELDDIR           = "plots/Yields_AN_fix_ARC_xcheck/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    #UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_fix/"
    #For CWR:
    YIELDDIR_BASE      = "plots/Yields_CWR/v6_calo_AOD_"+ERA+"_"
    YIELDDIR           = "plots/Yields_CWR/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_CWR/"
else:
    YIELDDIR_BASE      = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"
    YIELDDIR_BASE      = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"
    YIELDDIR           = PLOTDIR
    UNCDIR             = ""

#PLOTDIR            = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"+REGION+"__debug/"#"_2017_signal/"#
#PLOTDIR            = "plots/Efficiency_AN_additional_material/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#


#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"/"
#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"_AN/"#"/"
OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"/"
OUTCOMBI           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"/"


OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_TEST_remove_outliers/"#"/"
OUTCOMBI           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_TEST_remove_outliers/"

OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_preapproval/"#"/"
OUTCOMBI           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_preapproval/"

OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_unblinding/"#"/"
OUTCOMBI           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_unblinding/"

OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_unblinding_one_sided_Si/"#"/"
OUTCOMBI           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_unblinding_one_sided_Si/"

#Approval:
OUTPUTDIR_unblinding_ARC          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_unblinding_ARC/"#"/"
OUTCOMBI_unblinding_ARC           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_unblinding_ARC/"
PLOTLIMITSDIR_unblinding_ARC = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_unblinding_ARC/"

OUTPUTDIR = OUTPUTDIR_unblinding_ARC
OUTCOMBI  = OUTCOMBI_unblinding_ARC
PLOTLIMITSDIR = PLOTLIMITSDIR_unblinding_ARC

#CWR:
OUTPUTDIR_CWR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_CWR/"#"/"
OUTCOMBI_CWR           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_CWR/"
PLOTLIMITSDIR_CWR      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_CWR/"

OUTPUTDIR = OUTPUTDIR_CWR
OUTCOMBI  = OUTCOMBI_CWR
PLOTLIMITSDIR = PLOTLIMITSDIR_CWR

#DS review paper:
OUTPUTDIR_DS          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_DS/"#"/"
OUTCOMBI_DS           = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+REGION+"_DS/"
PLOTLIMITSDIR_DS      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_DS/"

OUTPUTDIR = OUTPUTDIR_DS
OUTCOMBI  = OUTCOMBI_DS
PLOTLIMITSDIR = PLOTLIMITSDIR_DS

### History of limits
###        
###        #PLOTLIMITSDIR      = "plots/Limits_AN/v6_calo_AOD_"+ERA+"_"+REGION+"_TEST_remove_outliers/"+br_scan_fold+"/"
###        PLOTLIMITSDIR      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_preapproval/"+br_scan_fold+"/"
###        PLOTLIMITSDIR      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_unblinding/"+br_scan_fold+"/"
###        #Approval:
###        PLOTLIMITSDIR      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_unblinding_ARC/"+br_scan_fold+"/"
###        #CWR:
###        PLOTLIMITSDIR      = "plots/Limits_AN/v6_calo_AOD_"+REGION+"_CWR/"+br_scan_fold+"/"
###        

#print "OUTPUTDIR for combination attempt"
#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"_AN_combi/"#"/"


CHAN               = "SUSY"
particle           = "#chi"
ctaupoint          = 500
#ctaus              = np.array([1, 3, 5, 7, 10, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 1100, 1200, 1300, 1400, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 50000, 100000])
ctaus              = np.array([1, 3, 5, 7, 10, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2500, 3000, 4000, 5000, 10000, 20000, 30000, 50000, 100000])
ctaus              = np.array([10, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2500, 3000, 4000, 5000, 10000, 20000, 30000, 50000, 100000])
#ctaus = np.array([100,500,1000,5000])
ctaus_500          = np.array([10, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 20000, 30000, 50000, 100000])

#A bit less points:
ctaus_500          = np.array([10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 25000, 50000, 100000])


ctaus = np.unique(np.concatenate((ctaus_500,ctaus_3000)))


signalMultFactor   = 0.001#!!!
signalBRfactor     = 0.9
PRELIMINARY        = False
TAGVAR             = "nTagJets_0p996_JJ"

SAVE               = True

back = ["All"]
data = ["HighMET"]
sign = []

ctaus_stealth = [0.01,0.1,1,10,100,1000]
mZ=options.mass_stop
mS=options.mass_s
ch=options.decay_ch
sign_stealth = []
#    'StealthSHH_2t6j_mstop1500_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau1000_SHH',
#    'StealthSHH_2t6j_mstop1500_ms1275_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau1_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau10_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau100_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau1000_SHH'

for ct in ctaus_stealth:
    sign_stealth.append("Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(mS)+"_ctau"+str(ct).replace(".","p")+"_"+ch)

print sign_stealth
#exit()

'''
#SYY
mZ=1500
ch="SYY"
sign_stealth = [
#    'StealthSYY_2t6j_mstop1500_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau1000_SYY',
    'StealthSYY_2t6j_mstop1500_ms1275_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau1_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau10_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau100_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau1000_SYY'
]
'''
#sign_stealth = ['StealthSHH_2t6j_mstop1500_ms1275_ctau1000_SHH']
#ctaus_stealth = [1000]


#era_tag here!!
era_tag = options.era_str


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



########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]
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
                elif ('HeavyHiggsToLLP') in ss:
                    print "Scaling HeavyHiggs to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('Stealth') in ss:
                    print "Scaling stealth to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('zPrime') in ss:
                    print "Scaling zPrime to 1. for absolute x-sec sensitivity"
                    print "IMPORTANT! Matthew corrections!"
                    corr_dict = {}
                    print s
                    print "~~~"
                    with open("signalEffs.csv",mode='r') as csvfile:
                        reader = csv.reader(csvfile)
                        #print reader
                        #dictreader = csv.DictReader(csvfile)
                        for row in reader:
                            if len(row)>0:
                                #print row[0],row[1]
                                pieces = row[0].split("_")
                                new_name = pieces[0]+"To"+pieces[7]+"_mZ"+pieces[2]+"_mX"+pieces[4]+"_ct"+pieces[6]
                                new_era = pieces[8]
                                #print new_name, new_era
                                if new_era==ERA and new_name==s:
                                    print "Gotcha!!!"
                                    print new_name, row[1]
                                    xs = 1./float(row[1])
                    print "~~~"
                    print xs
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

def write_datacards(tree_weight_dict,sign,main_pred_reg,main_pred_sample,extr_region,unc_list,dataset_label="",comb_fold_label="",add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False,blind=True,BR_SCAN_H=100,do_time_smearing = False,do_ct_average=True):
    print "\n"
    #Fix 2016 lumi
    if ERA=="2016":
        if dataset_label == "_G-H":
            this_lumi  = lumi[ main_pred_sample ]["G"]+lumi[ main_pred_sample ]["H"]#["tot"]
        elif dataset_label == "_B-F":
            this_lumi  = lumi[ main_pred_sample ]["B"]+lumi[ main_pred_sample ]["C"]+lumi[ main_pred_sample ]["D"]+lumi[ main_pred_sample ]["E"]+lumi[ main_pred_sample ]["F"]#["tot"]
    else:
        this_lumi  = lumi[ main_pred_sample ]["tot"]

    if check_closure:
        dnn_threshold = 0.9#
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996
        print  "DNN threshold: ", dnn_threshold

    print "\n"
    print " ********************************************************** "
    print "\n"

    print "Inferring limits on absolute x-sec in fb"
    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    DATACARDDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)

    br_scan_fold = ""
    if BR_SCAN_H==100:
        br_scan_fold = "BR_h100_z0"
    if BR_SCAN_H==75:
        br_scan_fold = "BR_h75_z25"
    if BR_SCAN_H==50:
        br_scan_fold = "BR_h50_z50"
    if BR_SCAN_H==25:
        br_scan_fold = "BR_h25_z75"
    if BR_SCAN_H==0:
        br_scan_fold = "BR_h0_z100"

    if "splitSUSY" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "splitSUSY"
    if "zPrime" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "zPrime"
    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"
    if "HeavyHiggs" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "HeavyHiggs"

    DATACARDDIR += br_scan_fold+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"datacards/"
    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    RESULTS = DATACARDDIR+"combine_results/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    #elif signalMultFactor == 0.0001:
    #    print '  x-sec calculated in fb --> for impacts '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
        print '-'*11*2

    #Bkg yield
    PREDDIR = YIELDDIR_BASE+main_pred_reg+"/"
    pred_file_name = PREDDIR+"BkgPredResults_"+ERA+"_"+main_pred_reg+"_"+main_pred_sample

    if eta_cut:
        pred_file_name+="_eta_1p0"
    if phi_cut==True:
        pred_file_name+="_phi_cut"

    if eta:
        pred_file_name+= "_vs_eta"
    if phi:
        pred_file_name+= "_vs_phi"

    if check_closure:
        pred_file_name+="_closure"+str(dnn_threshold).replace(".","p")

    #pred_file_name+=dataset_label
    pred_file_name+=comb_fold_label

    print"\n"
    print "  ERA and LUMI:"
    print "  ", ERA, dataset_label, this_lumi

    print"\n"
    with open(pred_file_name+".yaml","r") as f:
        print "  --  --  --  --  --"
        print "  Info: opening dictionary in file "+pred_file_name+".yaml"
        print "  Extrapolation region: ", extr_region+comb_fold_label
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()
        print "  --  --  --  --  --"

    print"\n"
    #Cosmic bkg yield
    cosmic_file_name = PREDDIR+"BkgPredResults_"+ERA+"_SR_cosmic"+dataset_label+".yaml"
    with open(cosmic_file_name,"r") as f:
        print "  --  --  --  --  --"
        print "  Info: opening dictionary in file "+cosmic_file_name
        cosmic = yaml.load(f, Loader=yaml.Loader)
        f.close()
        print "  ", cosmic
        print "  --  --  --  --  --"

    print"\n"
    #Beam halo bkg yield
    bh_file_name = PREDDIR+"BkgPredResults_"+ERA+"_SR_beam_halo"+dataset_label+".yaml"
    with open(bh_file_name,"r") as f:
        print "  --  --  --  --  --"
        print "  Info: opening dictionary in file "+bh_file_name
        bh = yaml.load(f, Loader=yaml.Loader)
        f.close()
        print "  ", bh
        print "  --  --  --  --  --"

    #extr_region+dataset_label+comb_fold_label --> extr_region+comb_fold_label
    y_bkg  = results[extr_region+comb_fold_label][main_pred_sample]['pred_2_from_1']
    y_bkg_from_0 = results[extr_region+comb_fold_label][main_pred_sample]['pred_2']
    #print results[extr_region+comb_fold_label][main_pred_sample].keys()
    #print y_bkg
    #print y_bkg_from_0

    y_data = 0#y_bkg+cosmic["bkg_cosmic"]+bh["bkg_bh"]
    if blind:
        y_data = results[extr_region+comb_fold_label][main_pred_sample]['y_2']+cosmic["bkg_cosmic"]+bh["bkg_bh"]

    #Syst uncertainties
    bkg_unc_dict = defaultdict(dict)
    sgn_unc_dict = defaultdict(dict)
    for l in unc_list:
        with open(UNCDIR+"signal_"+l+"_datacard_unc"+dataset_label+".yaml","r") as f:
            uncertainties = yaml.load(f, Loader=yaml.Loader)
            f.close()
        #print "   --> Unc: ", l
        if l=="bkg":
            bkg_unc_dict = uncertainties
        else:
            for s in sign:
                if "splitSUSY" in s:
                    print "!! Uncertainties on splitSUSY taken from GMSB SUSY !!"
                    if samples[s]['ctau']<600:
                        s_ref = "SUSY_mh1800_ctau500"
                    if samples[s]['ctau']>=600:
                        s_ref = "SUSY_mh1800_ctau3000"
                    ##sgn_unc_dict[s] = dict(list(sgn_unc_dict[s].items()) + list(uncertainties[s].items()))#
                    sgn_unc_dict[s].update(uncertainties[s_ref])
                elif "zPrime" in s:
                    print "!! Uncertainties on zPrime taken from GMSB SUSY !!"
                    #4b
                    if(ch=="To4b" and mZ==3000 and samples[s]['mass']==300):
                        if(samples[s]['ctau']<=1000):
                            s_ref = "SUSY_mh1800_ctau500"
                        else:
                            s_ref = "SUSY_mh1800_ctau3000"
                    if(ch=="To4b" and mZ==3000 and samples[s]['mass']==1450):
                        if(samples[s]['ctau']<=1000):
                            s_ref = "SUSY_mh1000_ctau500"
                        else:
                            s_ref = "SUSY_mh1000_ctau3000"                        
                    #zPrimeTo4b_mZ3000_mX300_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo4b_mZ3000_mX300_ct1000 - SUSY_mh1800_ctau500
                    #zPrimeTo4b_mZ3000_mX300_ct10000 - SUSY_mh1800_ctau3000
                    #zPrimeTo4b_mZ3000_mX1450_ct100 - SUSY_mh1500_ctau3000
                    #zPrimeTo4b_mZ3000_mX1450_ct1000 - SUSY_mh1000_ctau500
                    #zPrimeTo4b_mZ3000_mX1450_ct10000 - SUSY_mh1000_ctau3000

                    if(ch=="To4b" and mZ==4500 and samples[s]['mass']==450):
                        if(samples[s]['ctau']<=1000):
                            s_ref = "SUSY_mh1800_ctau500"
                        else:
                            s_ref = "SUSY_mh1800_ctau3000"
                    if(ch=="To4b" and mZ==4500 and samples[s]['mass']==2200):
                        if(samples[s]['ctau']<=100):
                            s_ref = "SUSY_mh1800_ctau500"
                        else:
                            s_ref = "SUSY_mh1250_ctau3000"
                    #zPrimeTo4b_mZ4500_mX450_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo4b_mZ4500_mX450_ct1000 - SUSY_mh1800_ctau500
                    #zPrimeTo4b_mZ4500_mX450_ct10000 - SUSY_mh1500_ctau500
                    #zPrimeTo4b_mZ4500_mX2200_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo4b_mZ4500_mX2200_ct1000 - SUSY_mh1500_ctau500
                    #zPrimeTo4b_mZ4500_mX2200_ct10000 - SUSY_mh1250_ctau3000

                    #2b2nu
                    if(ch=="To2b2nu" and mZ==3000 and samples[s]['mass']==300):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh1800_ctau500"
                        else:
                            s_ref = "SUSY_mh1800_ctau3000"
                    if(ch=="To2b2nu" and mZ==3000 and samples[s]['mass']==1450):
                        if(samples[s]['ctau']<=1000):
                            s_ref = "SUSY_mh1000_ctau3000"
                        else:
                            s_ref = "SUSY_mh1500_ctau3000"                        
                    #zPrimeTo2b2nu_mZ3000_mX300_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo2b2nu_mZ3000_mX300_ct1000 - SUSY_mh1800_ctau3000
                    #zPrimeTo2b2nu_mZ3000_mX300_ct10000 - SUSY_mh1800_ctau3000
                    #zPrimeTo2b2nu_mZ3000_mX1450_ct100 - SUSY_mh1800_ctau3000
                    #zPrimeTo2b2nu_mZ3000_mX1450_ct1000 - SUSY_mh1500_ctau3000
                    #zPrimeTo2b2nu_mZ3000_mX1450_ct10000 - SUSY_mh1250_ctau500
                    
                    if(ch=="To2b2nu" and mZ==4500 and samples[s]['mass']==450):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh1800_ctau500"
                        else:
                            s_ref = "SUSY_mh1500_ctau3000"
                    if(ch=="To2b2nu" and mZ==4500 and samples[s]['mass']==2200):
                        s_ref = "SUSY_mh1800_ctau500"
                    #zPrimeTo2b2nu_mZ4500_mX450_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo2b2nu_mZ4500_mX450_ct1000 - SUSY_mh1500_ctau3000
                    #zPrimeTo2b2nu_mZ4500_mX450_ct10000 - SUSY_mh1250_ctau3000
                    #zPrimeTo2b2nu_mZ4500_mX2200_ct100 - SUSY_mh1800_ctau500
                    #zPrimeTo2b2nu_mZ4500_mX2200_ct1000 - SUSY_mh1800_ctau500
                    #zPrimeTo2b2nu_mZ4500_mX2200_ct10000 - SUSY_mh1800_ctau500
                    sgn_unc_dict[s].update(uncertainties[s_ref])

                elif "HeavyHiggs" in s:
                    print "!! Uncertainties on HeavyHiggs taken from GMSB SUSY !!"
                    #4b
                    if(ch=="To4b" and mZ==800 and samples[s]['mass']==80):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh800_ctau500"
                        else:
                            s_ref = "SUSY_mh800_ctau3000"
                    if(ch=="To4b" and mZ==800 and samples[s]['mass']==350):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh400_ctau500"
                        else:
                            s_ref = "SUSY_mh400_ctau3000"                        
                    if(ch=="To4b" and mZ==400 and samples[s]['mass']==40):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh400_ctau500"
                        else:
                            s_ref = "SUSY_mh400_ctau3000"
                    if(ch=="To4b" and mZ==400 and samples[s]['mass']==150):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh200_ctau500"
                        else:
                            s_ref = "SUSY_mh200_ctau3000"

                    #2b2nu
                    if(ch=="To2b2nu" and mZ==800 and samples[s]['mass']==80):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh800_ctau3000"
                        else:
                            s_ref = "SUSY_mh800_ctau3000"
                    if(ch=="To2b2nu" and mZ==800 and samples[s]['mass']==350):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh400_ctau500"
                        else:
                            s_ref = "SUSY_mh400_ctau3000"                        
                    
                    if(ch=="To2b2nu" and mZ==400 and samples[s]['mass']==40):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh400_ctau500"
                        else:
                            s_ref = "SUSY_mh400_ctau3000"
                    if(ch=="To2b2nu" and mZ==400 and samples[s]['mass']==150):
                        s_ref = "SUSY_mh150_ctau3000"

                    sgn_unc_dict[s].update(uncertainties[s_ref])

                elif "Stealth" in s:
                    print "!! Uncertainties on Stealth taken from GMSB SUSY !!"
                    if(mZ==300):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh300_ctau500"
                        else:
                            s_ref = "SUSY_mh300_ctau3000"
                    if(mZ==500):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh400_ctau500"
                        else:
                            s_ref = "SUSY_mh400_ctau3000"
                    if(mZ==700):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh800_ctau500"
                        else:
                            s_ref = "SUSY_mh800_ctau3000"
                    if(mZ==900):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh800_ctau500"
                        else:
                            s_ref = "SUSY_mh800_ctau3000"
                    if(mZ==1100):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh1000_ctau500"
                        else:
                            s_ref = "SUSY_mh1000_ctau3000"
                    if(mZ==1300):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh1250_ctau500"
                        else:
                            s_ref = "SUSY_mh1250_ctau3000"
                    if(mZ==1500):
                        if(samples[s]['ctau']<1000):
                            s_ref = "SUSY_mh1500_ctau500"
                        else:
                            s_ref = "SUSY_mh1500_ctau3000"
                    sgn_unc_dict[s].update(uncertainties[s_ref])

                else:
                    s = s.replace("_HH","")
                    ##sgn_unc_dict[s] = dict(list(sgn_unc_dict[s].items()) + list(uncertainties[s].items()))#
                    sgn_unc_dict[s].update(uncertainties[s])

    print"\n"
    print "  Bkg uncertainties:"
    for bu in bkg_unc_dict.keys(): 
        print "  ", bu, bkg_unc_dict[bu]
    print"\n"
    print "  Sgn uncertainties:"
    for su in sgn_unc_dict.keys(): 
        print "  ", su, sgn_unc_dict[su]

    chainSignal = {}
    list_of_variables = [TAGVAR,"isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","JetsNegative.*","JetsNegative.pt","JetsNegative.phi","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRMSRecHitsEB","JetsNegative.nRecHitsEB","JetsNegative.timeRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","TriggerWeight","PUWeight","PUReWeight","GenLLPs.travelRadius","GenLLPs.travelX","GenLLPs.travelY","GenLLPs.travelZ","GenLLPs.travelTime","GenLLPs.beta","GenLLPs.*",CUT,"MeanNumInteractions","GenHiggs.*"]#"nLeptons"

    if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR and CUT=="isSR":
        list_of_variables += ["dt_ecal_dist","min_dPhi_jets*"]

    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    print"\n"
    print "  Time smearing file: ",OUT_TtoEM+"data_smear_file_CSV_0p8_all_jets"+label+".root"

    time_smear_file = TFile(OUT_TtoEM+"data_smear_file_CSV_0p8_all_jets"+label+".root","READ")
    time_smear_file.cd()
    data_time_CB_ttbar = time_smear_file.Get("data_CB")
    back_time_CB_ttbar = time_smear_file.Get("back_CB")
    time_smear_file.Close()

    print "\n"
    print "  Loading keras model: ", "nn_inference/tagger_AK4_v3/model.h5"
    model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')

    print "\n"
    print "  SF files pho: ",OUT_pho+"data_MC_SF"+label+"_1bin.root"
    print "  SF files ele: ",OUT_ele+"data_MC_SF"+label+"_1bin.root"

    sf_pho_file = TFile(OUT_pho+"data_MC_SF"+label+"_1bin.root","READ")
    sf_pho_file.cd()
    sf_pho_1ns = sf_pho_file.Get("ratio_1ns")
    sf_pho_2ns = sf_pho_file.Get("ratio_2ns")
    sf_pho_1ns.SetDirectory(0)
    sf_pho_2ns.SetDirectory(0)
    sf_pho_file.Close()

    sf_ele_file = TFile(OUT_ele+"data_MC_SF"+label+"_1bin.root","READ")
    sf_ele_file.cd()
    sf_ele_1ns = sf_ele_file.Get("ratio_1ns")
    sf_ele_2ns = sf_ele_file.Get("ratio_2ns")
    sf_ele_1ns.SetDirectory(0)
    sf_ele_2ns.SetDirectory(0)
    sf_ele_file.Close()

    sf_pho = sf_pho_1ns.GetBinContent(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinContent(1)
    sf_unc_pho = sf_pho_1ns.GetBinError(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinError(1)
    sf_pho_up = sf_pho+sf_unc_pho
    sf_pho_down = sf_pho-sf_unc_pho
    #print "sf pho 1 ns", sf_pho_1ns.GetBinContent(1),"+-",sf_pho_1ns.GetBinError(1)
    #print "sf pho 2 ns", sf_pho_2ns.GetBinContent(1),"+-",sf_pho_2ns.GetBinError(1)
    #print "most distant from 1", sf_pho, "+-", sf_unc_pho

    sf_ele = sf_ele_1ns.GetBinContent(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinContent(1)
    sf_unc_ele = sf_ele_1ns.GetBinError(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinError(1)
    #print "sf ele 1 ns", sf_ele_1ns.GetBinContent(1),"+-",sf_ele_1ns.GetBinError(1)
    #print "sf ele 2 ns", sf_ele_2ns.GetBinContent(1),"+-",sf_ele_2ns.GetBinError(1)
    #print "most distant from 1", sf_ele, "+-", sf_unc_ele
    sf_ele_up = sf_ele+sf_unc_ele
    sf_ele_down = sf_ele-sf_unc_ele

    bin2 = defaultdict(dict)
    bin2_entries = defaultdict(dict)
    bin2_up_ele = defaultdict(dict)
    bin2_up_pho = defaultdict(dict)
    bin2_down_ele = defaultdict(dict)
    bin2_down_pho = defaultdict(dict)
    time_distr = defaultdict(dict)
    ctau_weights = defaultdict(dict)
    sgn_unc_dict_ctau = defaultdict(dict)

    y_2_ctau = defaultdict(dict)
    e_2_ctau = defaultdict(dict)
    
    chain = defaultdict(dict)
    h_time = defaultdict(dict)
    h_time_smear = defaultdict(dict)
    smear_cb = defaultdict(dict)
    f_time = defaultdict(dict)
    fit_time = defaultdict(dict)
    #n_mass = {}

    #I want to store all the relevant input variables to the dnn as arrays
    #So that I don't need to re-loop
    #But this works only if I get real arrays, not jagged!
    

    masses = []

    #Initialize
    loop_ct = ctaus
    if "splitSUSY" in sign[0]:
        loop_ct = ctaus_split
    if "zPrime" in sign[0]:
        loop_ct = ctaus_zPrime
    if "Stealth" in sign[0]:
        loop_ct = ctaus_stealth
    if "HeavyHiggs" in sign[0]:
        loop_ct = ctaus_HeavyHiggs

    for pr in sign:
        masses.append(samples[pr]['mass'])
        loop_ct = np.append(loop_ct,np.array([samples[pr]['ctau']]))
        h_time[pr] = TH1F("timeRecHitsEB_"+pr,";"+variable["JetsNegative.timeRecHitsEB"]['title'],variable["JetsNegative.timeRecHitsEB"]['nbins'], -20, 20)
        h_time[pr].Sumw2()
        h_time_smear[pr] = TH1F("smear_timeRecHitsEB_"+pr,";"+variable["JetsNegative.timeRecHitsEB"]['title'],variable["JetsNegative.timeRecHitsEB"]['nbins'], -20, 20)
        h_time_smear[pr].Sumw2()
        smear_cb[pr] = data_time_CB_ttbar.Clone("smear_cb_"+pr)
        f_time[pr] = TF1()
        fit_time[pr] = TF1()

    #for m in masses:
    #    n_mass[m] = masses.count(m)
    masses = np.unique(np.array(masses))
    loop_ct = np.unique(loop_ct)

    
    for i,pr in enumerate(sign):
        for ct in loop_ct:
            ctau_weights[pr][ct] = np.array([])
            y_2_ctau[ samples[pr]['mass'] ][ct] = np.array([])
            e_2_ctau[ samples[pr]['mass'] ][ct] = np.array([])
            sgn_unc_dict_ctau[ samples[pr]['mass'] ][ct] = {}
            for k in sgn_unc_dict[sign[0]].keys():
                sgn_unc_dict_ctau[ samples[pr]['mass'] ][ct][k] = 0


    
    for i,pr in enumerate(sign):
        if "splitSUSY" in sign[0]:
            new_list = [pr]
        elif "zPrime" in sign[0]:
            new_list = [pr]
        elif "Stealth" in sign[0]:
            new_list = [pr]
        elif "HeavyHiggs" in sign[0]:
            new_list = [pr]
        else:
            new_list = [pr+"_HH",pr+"_HZ",pr+"_ZZ"]
        bin2[pr] = np.array([])
        bin2_entries[pr] = np.array([])
        bin2_up_ele[pr] = np.array([])
        bin2_up_pho[pr] = np.array([])
        bin2_down_ele[pr] = np.array([])
        bin2_down_pho[pr] = np.array([])
        time_distr[pr] = np.array([])
        y_2 = {}
        e_2 = {}

        files_list = []
        print "\n"

        #approach A)
        #save all the arrays, after the loop perform the time fit
        #then perform the smearing
        #then all the cuts we need

        #approach B)
        #loop once for the time histo and fit
        #loop once for all the cuts, smearing and inference
        #and store only the final output
        
        #First loop: TChain for jet time
        chain[pr] = TChain("tree")
        for s in new_list:
            for l, ss in enumerate(samples[s]['files']):
                filename = NTUPLEDIR + ss + '.root'
                #Set the right weight considering the BR
                if tree_weight_dict[pr][ss]>0:
                    #print "Adding ", ss, " to the time TChain"
                    chain[pr].SetWeight(tree_weight_dict[pr][ss])
                    chain[pr].Add(filename)
        chain[pr].Project("timeRecHitsEB_"+pr, "JetsNegative.timeRecHitsEB", "EventWeight*PUReWeight*TriggerWeight*(MinJetMetDPhi>0.5 && fabs(JetsNegative.eta)<1)")

        h_time[pr].Scale(1./h_time[pr].Integral() if h_time[pr].Integral()>0 else 1.)
        #f_time[pr] = TF1("f_time_"+pr,"gaus", -20, 20)
        ##f_time[pr] = TF1("f_time_"+pr,"crystalball", -20, 20)
        #f_time[pr].SetParameter(0,0.01)
        #f_time[pr].SetParameter(1,h_time[pr].GetMean())
        #f_time[pr].SetParameter(2,h_time[pr].GetRMS())
        #f_time[pr].SetParameter(3,1)
        #f_time[pr].SetParameter(4,1)
        #h_time[pr].Fit(f_time[pr],"E")
        #fit_time[pr] = h_time[pr].GetFunction("f_time_"+pr)
        #h_time[pr].GetListOfFunctions().Remove(h_time[pr].GetFunction("f_time_"+pr))
        #fit_time[pr].SetLineColor(4)
        #fit_time[pr].SetLineStyle(1)
        h_time[pr].SetLineWidth(2)
        h_time[pr].SetLineColor(856)
        h_time[pr].SetFillColorAlpha(856,0.1)

        smear_cb[pr].SetParameter(0,data_time_CB_ttbar.GetParameter(0)) 
        smear_cb[pr].SetParameter(1,data_time_CB_ttbar.GetParameter(1) - back_time_CB_ttbar.GetParameter(1))
        smear_cb[pr].SetParameter(2, math.sqrt( abs(data_time_CB_ttbar.GetParameter(2)**2 - back_time_CB_ttbar.GetParameter(2)**2)) )
        smear_cb[pr].SetParameter(3,data_time_CB_ttbar.GetParameter(3))
        smear_cb[pr].SetParameter(4,data_time_CB_ttbar.GetParameter(4))
        smear_cb[pr].SetLineColor(2)



        #Here throw time smearing and randomize manually
        name_for_n_gen = samples[s]['files'][0]
        #zPrime
        if "zPrime" not in sign[0] and "HeavyHiggs" not in sign[0] and "Stealth" not in sign[0]:
            print name_for_n_gen, sample[name_for_n_gen]['nevents']
            n_random = sample[name_for_n_gen]['nevents']*3*10
        else:
            n_random = 1000000
            print "TEST!!!!!!!!!!!!!"
        time_random_vec = [smear_cb[pr].GetRandom() for nr in range(n_random)]
        #print n_random
        #print len(time_random_vec)
        ##shuffle
        np.random.shuffle(time_random_vec)

        jet_counter_for_time_random_vec = 0

        print "\n"
        print "   NTUPLEDIR: ", NTUPLEDIR

        for s in new_list:
            if samples[s]['mass']<210 and "splitSUSY" not in sign[0] and "zPrime" not in sign[0] and "Stealth" not in sign[0] and "HeavyHiggs" not in sign[0]:
                print "Adjust signal multiplication (enhance signal to avoid combine instabilities)"
                signalMultFactor_adjusted = signalMultFactor*10
            else:
                signalMultFactor_adjusted = signalMultFactor
            for l, ss in enumerate(samples[s]['files']):
                print "Uproot iterating over ", ss, " . . . "
                filename = NTUPLEDIR + ss + '.root'
                gen = uproot.iterate(filename,"tree",list_of_variables)
                for arrays in gen:
                    st_it = time.time()
                    key_list = arrays.keys()
                    tree_w_array = tree_weight_dict[pr][ss]*np.ones( len(arrays[ key_list[0] ])  )

                    if CUT == "isSR":
                        cut_mask = arrays[CUT]>0
                        if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                            #cosmic
                            cosmic_veto = arrays["dt_ecal_dist"]<0.5
                            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
                    else:
                        cut_mask = (arrays[CUT]>0)

                    if KILL_QCD:
                        cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                    
                    #1. Selections on JetsNegative
                    cut_jets = arrays["JetsNegative.pt"]>-999
                    cut_jets = np.logical_and(cut_mask,cut_jets)
                    cut_mask = (cut_jets.any()==True)
                    
                    if phi_cut==True and eta_cut==False:
                        cut_mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                        cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                        cut_mask = (cut_mask_phi.any()==True)
                        cut_jets = np.logical_and(cut_jets,cut_mask_phi)#new

                    elif eta_cut==True and phi_cut==False:
                        cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                        cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                        #This is needed to guarantee the shapes are consistent
                        cut_mask = (cut_mask_eta.any()==True)
                        cut_jets = np.logical_and(cut_jets,cut_mask_eta)#new

                    elif phi_cut and eta_cut:
                        cut_mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                        cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                        cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                        cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                        #This is needed to guarantee the shapes are consistent
                        cut_mask = (cut_mask_phi_eta.any()==True)
                        cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)#new

                    #Beam Halo veto
                    #This is technically inconsistent, as it is calculated before the time smearing. It has a small effect anyways. Let's keep it as it is.
                    if CUT == "isSR":
                        if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                            cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
                            cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
                            cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
                            cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)
                            cut_mask = np.logical_and(cut_mask,cut_mask_bh)
                            cut_jets = np.logical_and(cut_jets,cut_mask)

                    

                    #1b. Store all the needed inputs as arrays. We'll then perform the time fit, the inference and the final cuts without re-looping.
                    arr_pt = arrays["JetsNegative.pt"][cut_jets][cut_mask]
                    arr_isGenMatched = arrays["JetsNegative.isGenMatched"][cut_jets][cut_mask]
                    arr_timeRecHitsEB = arrays["JetsNegative.timeRecHitsEB"][cut_jets][cut_mask]
                    arr_nTrackConstituents = arrays["JetsNegative.nTrackConstituents"][cut_jets][cut_mask]
                    arr_nSelectedTracks= arrays["JetsNegative.nSelectedTracks"][cut_jets][cut_mask]
                    arr_eFracRecHitsEB= arrays["JetsNegative.eFracRecHitsEB"][cut_jets][cut_mask]
                    arr_nRecHitsEB= arrays["JetsNegative.nRecHitsEB"][cut_jets][cut_mask]
                    arr_sig1EB= arrays["JetsNegative.sig1EB"][cut_jets][cut_mask]
                    arr_sig2EB= arrays["JetsNegative.sig2EB"][cut_jets][cut_mask]
                    arr_ptDEB= arrays["JetsNegative.ptDEB"][cut_jets][cut_mask]
                    arr_cHadEFrac= arrays["JetsNegative.cHadEFrac"][cut_jets][cut_mask]
                    arr_nHadEFrac= arrays["JetsNegative.nHadEFrac"][cut_jets][cut_mask]
                    arr_eleEFrac= arrays["JetsNegative.eleEFrac"][cut_jets][cut_mask]
                    arr_photonEFrac= arrays["JetsNegative.photonEFrac"][cut_jets][cut_mask]
                    arr_ptAllTracks= arrays["JetsNegative.ptAllTracks"][cut_jets][cut_mask]
                    arr_ptAllPVTracks= arrays["JetsNegative.ptAllPVTracks"][cut_jets][cut_mask]
                    arr_alphaMax= arrays["JetsNegative.alphaMax"][cut_jets][cut_mask]
                    arr_betaMax= arrays["JetsNegative.betaMax"][cut_jets][cut_mask]
                    arr_gammaMax= arrays["JetsNegative.gammaMax"][cut_jets][cut_mask]
                    arr_gammaMaxEM= arrays["JetsNegative.gammaMaxEM"][cut_jets][cut_mask]
                    arr_gammaMaxHadronic= arrays["JetsNegative.gammaMaxHadronic"][cut_jets][cut_mask]
                    arr_gammaMaxET= arrays["JetsNegative.gammaMaxET"][cut_jets][cut_mask]
                    arr_minDeltaRAllTracks= arrays["JetsNegative.minDeltaRAllTracks"][cut_jets][cut_mask]
                    arr_minDeltaRPVTracks= arrays["JetsNegative.minDeltaRPVTracks"][cut_jets][cut_mask]
                    n_j_per_event = arr_timeRecHitsEB.counts
                    n_tot_j = len(arr_timeRecHitsEB.flatten())

                    #gRandom is not really random, as the seed repeats itself
                    #print "gRandom: ", gRandom.Rndm()
                    #time_smearer = np.array([ smear_cb[pr].GetRandom() for n in range(n_tot_j)  ])

                    time_smearer = np.array(time_random_vec[jet_counter_for_time_random_vec:jet_counter_for_time_random_vec+n_tot_j])
                    jet_counter_for_time_random_vec += n_tot_j

                    arr_timeRecHitsEB_smeared = time_smearer+arr_timeRecHitsEB.flatten() if do_time_smearing else arr_timeRecHitsEB.flatten()
                    arr_sigprob = arrays["JetsNegative.sigprob"][cut_jets][cut_mask]

                    #print "original time: "
                    #print arr_timeRecHitsEB.flatten()
                    #print "smeared time:  "
                    #print arr_timeRecHitsEB_smeared

                    time_distr[pr] = np.concatenate( (time_distr[pr],arr_timeRecHitsEB_smeared) )

                    arr_dat_list = [
                        arr_nTrackConstituents.flatten(),
                        arr_nSelectedTracks.flatten(),
                        arr_timeRecHitsEB_smeared,
                        arr_eFracRecHitsEB.flatten(),
                        arr_nRecHitsEB.flatten(),
                        arr_sig1EB.flatten(),
                        arr_sig2EB.flatten(),
                        arr_ptDEB.flatten(),
                        arr_cHadEFrac.flatten(),
                        arr_nHadEFrac.flatten(),
                        arr_eleEFrac.flatten(),
                        arr_photonEFrac.flatten(),
                        arr_ptAllTracks.flatten(),
                        arr_ptAllPVTracks.flatten(),
                        arr_alphaMax.flatten(),
                        arr_betaMax.flatten(),
                        arr_gammaMax.flatten(),
                        arr_gammaMaxEM.flatten(),
                        arr_gammaMaxHadronic.flatten(),
                        arr_gammaMaxET.flatten(),
                        arr_minDeltaRAllTracks.flatten(),
                        arr_minDeltaRPVTracks.flatten(),
                    ]

                    arr_X = np.transpose(np.stack((arr_dat_list)))
                    #print len(arr_X)
                    #print arr_X
                    if len(arr_X)>0:
                        arr_probs = model.predict(arr_X)[:,1] 
                    else:
                        arr_probs = []
                    prob_len = len(arr_probs)

                    list_sigprob = []
                    list_timeRecHitsEB = []
                    start = 0
                    for j in n_j_per_event:
                        list_sigprob.append(arr_probs[start:start+j])
                        list_timeRecHitsEB.append(arr_timeRecHitsEB_smeared[start:start+j])
                        start+=j

                    sigprob = awkward.fromiter(list_sigprob)
                    timeRecHitsEB = awkward.fromiter(list_timeRecHitsEB)

                    #SR cut
                    #Reject jets with negative time
                    cut_mask_time = timeRecHitsEB > -1.
                    cut_mask_time_any = (cut_mask_time.any()==True)

                    #events with at least one jet with valid time
                    #print "timeRecHitsEB[cut_mask_time]"
                    #only jets with right time, filtering away events without valid jet time
                    #print "timeRecHitsEB[cut_mask_time_any]"

                    sigprob = sigprob[cut_mask_time][cut_mask_time_any]
                    timeRecHitsEB = timeRecHitsEB[cut_mask_time][cut_mask_time_any]

                    tag_mask = (sigprob > dnn_threshold)
                    if (sigprob[tag_mask]).shape[0]==0: continue
                    bin2_m = (sigprob[tag_mask].counts >1)
                    
                    #We need to store only the pt of tagged jets
                    sigprob_bin2 = sigprob[bin2_m]
                    pt = arr_pt[cut_mask_time][cut_mask_time_any][bin2_m]
                    #Old method (wrong in v10): with pt
                    #Pre-approval: with pt_tag
                    pt_tag = pt[sigprob_bin2>dnn_threshold]

                    #This: corrects all the jets, also the not tagged
                    #pt_ele_mask = (pt > 70)
                    #pt_pho_mask = (pt <= 70)
                    #This: corrects only the tagged jets
                    pt_ele_mask = (pt_tag > 70)
                    pt_pho_mask = (pt_tag <= 70)

                    #print pt_tag
                    #print pt_ele_mask
                    #print pt_pho_mask

                    ##pt_ele_mask_filter = pt_ele_mask[pt_ele_mask]
                    ##pt_pho_mask_filter = pt_pho_mask[pt_pho_mask]

                    #perform the product among each of the jets
                    #shape: n of events passinf
                    #Given the exclusive categorization, I am sure I have no zeros in this vector
                    dnnweight = (sf_ele*pt_ele_mask+sf_pho*pt_pho_mask).prod()
                    dnnweight_up_ele = (sf_ele_up*pt_ele_mask+sf_pho*pt_pho_mask).prod()
                    dnnweight_up_pho = (sf_ele*pt_ele_mask+sf_pho_up*pt_pho_mask).prod()
                    dnnweight_down_ele = (sf_ele_down*pt_ele_mask+sf_pho*pt_pho_mask).prod()
                    dnnweight_down_pho = (sf_ele*pt_ele_mask+sf_pho_down*pt_pho_mask).prod()

                    eventweight = arrays["EventWeight"][cut_mask][cut_mask_time_any][bin2_m] 
                    pureweight = arrays["PUReWeight"][cut_mask][cut_mask_time_any][bin2_m]
                    trgweight = arrays["TriggerWeight"][cut_mask][cut_mask_time_any][bin2_m]
                    ##print "weight pre-dnn"
                    ##weight = np.multiply(eventweight,np.multiply(pureweight,trgweight))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted
                    #print "weight post-dnn"
                    weight = np.multiply(dnnweight,np.multiply(eventweight,np.multiply(pureweight,trgweight)))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    #no dnnweight:
                    #weight = np.multiply(eventweight,np.multiply(pureweight,trgweight))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    weight_up_ele = np.multiply(dnnweight_up_ele,np.multiply(eventweight,np.multiply(pureweight,trgweight)))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    weight_up_pho = np.multiply(dnnweight_up_pho,np.multiply(eventweight,np.multiply(pureweight,trgweight)))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    weight_down_ele = np.multiply(dnnweight_down_ele,np.multiply(eventweight,np.multiply(pureweight,trgweight)))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    weight_down_pho = np.multiply(dnnweight_down_pho,np.multiply(eventweight,np.multiply(pureweight,trgweight)))*tree_weight_dict[pr][ss]*signalMultFactor_adjusted*signalBRfactor
                    #Here: gen level stuff
                    if "Stealth" in sign[0]:
                        genRadius = arrays["GenHiggs.travelRadius"][cut_mask][cut_mask_time_any][bin2_m]
                        genX = arrays["GenHiggs.travelX"][cut_mask][cut_mask_time_any][bin2_m]
                        genY = arrays["GenHiggs.travelY"][cut_mask][cut_mask_time_any][bin2_m]
                        genZ = arrays["GenHiggs.travelZ"][cut_mask][cut_mask_time_any][bin2_m]
                        genTime = arrays["GenHiggs.travelTime"][cut_mask][cut_mask_time_any][bin2_m]
                        genBeta = arrays["GenHiggs.beta"][cut_mask][cut_mask_time_any][bin2_m]
                    else:
                        genRadius = arrays["GenLLPs.travelRadius"][cut_mask][cut_mask_time_any][bin2_m]
                        genX = arrays["GenLLPs.travelX"][cut_mask][cut_mask_time_any][bin2_m]
                        genY = arrays["GenLLPs.travelY"][cut_mask][cut_mask_time_any][bin2_m]
                        genZ = arrays["GenLLPs.travelZ"][cut_mask][cut_mask_time_any][bin2_m]
                        genTime = arrays["GenLLPs.travelTime"][cut_mask][cut_mask_time_any][bin2_m]
                        genBeta = arrays["GenLLPs.beta"][cut_mask][cut_mask_time_any][bin2_m]
                    genGamma = np.divide(1.,np.sqrt(1-np.multiply(genBeta,genBeta)))
                    genTravelDist = np.sqrt( np.multiply(genX,genX) + np.multiply(genY,genY) + np.multiply(genZ,genZ) )
                    genPosteriorTime = np.divide(genTravelDist,np.multiply(genBeta , genGamma))
                    del arrays

                    if scale_mu!=1.:
                        print "Scaling mu up by factor", scale_mu
                        weight *= scale_mu

                    bin2[pr] = np.concatenate( (bin2[pr],weight) )
                    bin2_entries[pr] = np.concatenate( (bin2_entries[pr], weight.astype(bool)) )
                    bin2_up_ele[pr] = np.concatenate( (bin2_up_ele[pr],weight_up_ele) )
                    bin2_up_pho[pr] = np.concatenate( (bin2_up_pho[pr],weight_up_pho) )
                    bin2_down_ele[pr] = np.concatenate( (bin2_down_ele[pr],weight_down_ele) )
                    bin2_down_pho[pr] = np.concatenate( (bin2_down_pho[pr],weight_down_pho) )

                    #it was:
                    #for ct in ctaus:
                    for ct in loop_ct:
                        if samples[pr]['ctau']==ct:
                            ctau_weights[pr][ct] = np.concatenate( (ctau_weights[pr][ct], weight))
                        else:
                            ctau_weights[pr][ct] = np.concatenate(( ctau_weights[pr][ct], np.multiply(tau_weight_calc(genPosteriorTime.sum(), float(ct)/10., float( samples[pr]['ctau'] )/10.), weight) ))

                del gen

    '''
    #Draw jet time
    for pr in sign:
        root_numpy.fill_hist(h_time_smear[pr], time_distr[pr])
        h_time_smear[pr].Scale(1./h_time_smear[pr].Integral())
        h_time_smear[pr].SetLineColor(1)
        h_time_smear[pr].SetLineWidth(3)
        can = TCanvas("can","can",900,800)
        can.SetRightMargin(0.05)
        can.cd()
        leg = TLegend(0.65, 0.6, 1., 0.9)
        h_time[pr].Draw("HISTO,sames")
        #fit_time[pr].Draw("L,sames")
        smear_cb[pr].Draw("L,sames")
        #data_time_CB_ttbar.SetLineColor(8)
        #data_time_CB_ttbar.Draw("L,sames")
        h_time_smear[pr].Draw("HISTO,sames")
        leg.SetHeader(pr)
        #leg.AddEntry(data_time_CB_ttbar,"data ttbar m "+str(round(data_time_CB_ttbar.GetParameter(1),2))+",#sigma "+str(round(data_time_CB_ttbar.GetParameter(2),2)),"L")
        #leg.AddEntry(fit_time[pr],"signal fit m "+str(round(fit_time[pr].GetParameter(1),2))+",#sigma "+str(round(fit_time[pr].GetParameter(2),2)),"L")
        leg.AddEntry(smear_cb[pr],"smear CB","L")
        leg.AddEntry(h_time_smear[pr],"smeared signal","PL")
        OUTSTRING = "jet_time_"+pr
        can.SetLogy()
        can.Update()
        leg.Draw()
        can.Print(OUTSTRING+'_log.png')
        can.Print(OUTSTRING+'_log.pdf')
    '''

    #DNN uncertainties after shift dnn up and down
    for pr in sign:
        ele_shift_up = abs(bin2_up_ele[pr].sum()-bin2[pr].sum())/bin2[pr].sum() if bin2[pr].sum()!=0 else 0
        ele_shift_down = abs(bin2_down_ele[pr].sum()-bin2[pr].sum())/bin2[pr].sum() if bin2[pr].sum()!=0 else 0
        pho_shift_up = abs(bin2_up_pho[pr].sum()-bin2[pr].sum())/bin2[pr].sum() if bin2[pr].sum()!=0 else 0
        pho_shift_down = abs(bin2_down_pho[pr].sum()-bin2[pr].sum())/bin2[pr].sum() if bin2[pr].sum()!=0 else 0
        shift_up = math.sqrt( ele_shift_up**2 + pho_shift_up**2  )
        shift_down = math.sqrt( ele_shift_down**2 + pho_shift_down**2  )
        #Useful printouts
        #print pr
        #print "ele shift up: %.3f" % ele_shift_up
        #print "ele shift down: %.3f" % ele_shift_down
        #print "pho shift up: %.3f" % pho_shift_up
        #print "pho shift down: %.3f" % pho_shift_down
        #print "shift up %.3f" % shift_up
        #print "shift down %.3f" % shift_down
        #print "unc: %.3f" % (100*max(shift_up,shift_down))
        ##print "sf ele unc: %.3f" % (100*sf_unc_ele/sf_ele)
        ##print "sf pho unc: %.3f" % (100*sf_unc_pho/sf_pho)
        ##print "quadrature: %.3f" % (100*math.sqrt(  (sf_unc_ele/sf_ele)**2 + (sf_unc_pho/sf_pho)**2  ))
        #print "----"
        dnn_unc_labels = [k for k in sgn_unc_dict[pr].keys() if "DNN" in k]
        if len(dnn_unc_labels)>1:
            print "Too many DNN uncertainty keys in uncertainty dictionary, aborting...."
            exit()
        #print dnn_unc_labels
        #print sgn_unc_dict[pr][ dnn_unc_labels[0] ]
        sgn_unc_dict[pr][ dnn_unc_labels[0] ] = (100*max(shift_up,shift_down))
        #print sgn_unc_dict[pr][ dnn_unc_labels[0] ]



    #print "global yields and stat. uncertainties(%)"
    y_a = defaultdict(dict)
    stat_unc_a = defaultdict(dict)
    y_b = defaultdict(dict)
    stat_unc_b = defaultdict(dict)
    y_comb = defaultdict(dict)
    stat_unc_comb = defaultdict(dict)

    print "\n"
    for k in bin2.keys():
        print "  --> Debugging purposes:"
        print "  -->", k, " ; yield=", bin2[k].sum(), " ; entries=", bin2_entries[k].sum(), " x check: ", bin2[k].astype(bool).sum()


    for m in masses:
        for ct in loop_ct:#ctaus:
            y_a[m][ct] = 0
            y_b[m][ct] = 0
            y_comb[m][ct] = 0
            stat_unc_a[m][ct] = 0
            stat_unc_b[m][ct] = 0
            stat_unc_comb[m][ct] = 0


    #This loops removes outliers
    for k in ctau_weights.keys():
        m = samples[k]['mass']
        print "----------------------------------------"
        print "using sample ", k
        if "splitSUSY" in k:
            ct = samples[k]['ctau']
            y_comb[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
            stat_unc_comb[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
            for u in sgn_unc_dict[k].keys():
                sgn_unc_dict_ctau[samples[k]['mass']][ct][u] = sgn_unc_dict[k][u]
        elif "Stealth" in k:
            ct = samples[k]['ctau']
            y_comb[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
            stat_unc_comb[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
            for u in sgn_unc_dict[k].keys():
                sgn_unc_dict_ctau[samples[k]['mass']][ct][u] = sgn_unc_dict[k][u]
        elif "zPrime" in k:
            ct = samples[k]['ctau']
            y_comb[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
            stat_unc_comb[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
            for u in sgn_unc_dict[k].keys():
                sgn_unc_dict_ctau[samples[k]['mass']][ct][u] = sgn_unc_dict[k][u]
            for ctl in loop_ct:
                s_name = "zPrime"+ch+"_mZ"+str(mZ)+"_mX"+str(m)+"_ct"
                #This is ok for 4b and mZ 3000
                if ch=="To4b":
                    if mZ==3000 or (mZ==4500 and m==2200):
                        if ctl==ct:
                            s_name_sel = k
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                        if ctl>1 and ctl<=10:
                            s_name_sel = s_name + str(int(10))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>10 and ctl<=100:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>100 and ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                    if mZ==4500 and samples[k]['mass']==450:
                        if ctl==ct:
                            s_name_sel = k
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                        if ctl<=100:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>100 and ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]



                if ch=="To2b2nu":
                    if mZ==3000 and m==300:# or (mZ==4500 and m==2200):
                        if ctl==ct:
                            s_name_sel = k
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                        if ctl>1 and ctl<=10:
                            s_name_sel = s_name + str(int(10))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>10 and ctl<=500:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>500 and ctl<10000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=10000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                    '''
                    if mZ==4500 and samples[k]['mass']==450:
                        if ctl==ct:
                            s_name_sel = k
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                        if ctl<=100:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>100 and ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                    '''

        elif "HeavyHiggs" in k:
            ct = samples[k]['ctau']
            y_comb[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
            stat_unc_comb[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
            for u in sgn_unc_dict[k].keys():
                sgn_unc_dict_ctau[samples[k]['mass']][ct][u] = sgn_unc_dict[k][u]
            for ctl in loop_ct:
                #print "printing weights"
                #print ct, ctl
                s_name = "HeavyHiggsToLLP"+ch+"_mH"+str(mZ)+"_mX"+str(m)+"_ct"

                if ctl==ct:
                    #if ct<1:
                    #    s_name+="0p1"
                    #else:
                    #    s_name+=str(int(ct))
                    #print "Correct sample"
                    #print k, ctl, ctau_weights[k][ctl].sum()
                    s_name_sel = k
                    print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                    print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                if ch=="To2b2nu":
                    if mZ==400 and samples[k]['mass']==40:
                        #Take all from 100 mm (2016)
                        #very poor acceptance
                        if ctl<5000000:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                    else:
                        if ctl>1 and ctl<=10:
                            s_name_sel = s_name + str(int(10))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>10 and ctl<=100:
                            s_name_sel = s_name + str(int(100))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>100 and ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()
                            
                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                if ch=="To4b":
                    if mZ==800 and samples[k]['mass']==80:
                        #Do not look at 100, too few stat
                        if ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                    if mZ==800 and samples[k]['mass']==350:
                        #Take all from 10 meters
                        if ctl<5000000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                    if mZ==400 and samples[k]['mass']==150:
                        if ctl<5000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
                        if ctl>=5000:
                            s_name_sel = s_name + str(int(10000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]

                    if mZ==400 and samples[k]['mass']==40:
                        #Take all from 1 meters
                        if ctl<5000000:
                            s_name_sel = s_name + str(int(1000))
                            print "Will use this sample for extrapolation to ", ctl ,": ", s_name_sel
                            print k, ctl, ctau_weights[s_name_sel][ctl].sum()

                            y_comb[samples[k]['mass']][ctl] = ctau_weights[s_name_sel][ctl].sum()
                            stat_unc_comb[samples[k]['mass']][ctl] = np.sqrt( sum(x*x for x in ctau_weights[s_name_sel][ctl]) ).sum()
                            for u in sgn_unc_dict[k].keys():
                                sgn_unc_dict_ctau[samples[k]['mass']][ctl][u] = sgn_unc_dict[k][u]
        else:
            for ct in loop_ct:#ctaus:
                ##Here we need a switch
                #print "before: ", k,"m:%d, ct:%d, entries:%d, yield:%.6f +- %.2f (percentage)" % (samples[k]['mass'], ct, len(ctau_weights[k][ct]), ctau_weights[k][ct].sum(), 100*np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()/ctau_weights[k][ct].sum())
                #print "entries? ", (ctau_weights[k][ct].astype(bool)).sum()
                flat = ctau_weights[k][ct].tolist
                ##print ctau_weights[k][ct].shape, " weights: ", flat

                if samples[k]['ctau']==500:
                    ##print "a-type: predicted from 500"
                    ###do it twice
                    ###remove outliers
                    for x in range(4):
                        max_weight = np.max(ctau_weights[k][ct])
                        max_weight_position = np.argmax(ctau_weights[k][ct])
                        ###min_weight = np.min(ctau_weights[k][ct] [ np.nonzero(ctau_weights[k][ct]) ])
                        ###min_weight_position = np.argmin(ctau_weights[k][ct] [ np.nonzero(ctau_weights[k][ct]) ])
                        mean_weight = np.mean(ctau_weights[k][ct])
                        ##print "max weight outlier: ", max_weight/mean_weight
                        if max_weight/mean_weight>10 and mean_weight>0:
                            ctau_weights[k][ct] = np.delete(ctau_weights[k][ct],max_weight_position)
                        if k=="SUSY_mh127_ctau500" and max_weight/mean_weight<=10:
                            for x in range(2):
                                max_weight = np.max(ctau_weights[k][ct])
                                max_weight_position = np.argmax(ctau_weights[k][ct])
                                mean_weight = np.mean(ctau_weights[k][ct])
                                ctau_weights[k][ct] = np.delete(ctau_weights[k][ct],max_weight_position)
                        
                        ##print "still looking at: ", k, ct
                        ##print "weights size? ", ctau_weights[k][ct].shape
                        ##print "min weight outlier: ", min_weight
                        ##print "min weight: ", ctau_weights[k][ct][ min_weight_position ]
                    y_a[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
                    stat_unc_a[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
                elif samples[k]['ctau']==3000:
                    ##print "b-type: predicted from 3000"
                    ##do it twice
                    ##remove outliers
                    for x in range(4):
                        max_weight = np.max(ctau_weights[k][ct])
                        max_weight_position = np.argmax(ctau_weights[k][ct])
                        mean_weight = np.mean(ctau_weights[k][ct])
                        ##print "max weight outlier: ", max_weight/mean_weight
                        if max_weight/mean_weight>10 and mean_weight>0:
                            ctau_weights[k][ct] = np.delete(ctau_weights[k][ct],max_weight_position)
                        if k=="SUSY_mh175_ctau3000" and max_weight/mean_weight<=10 and BR_SCAN_H==100:
                            ##print "for ctau 3, problems"
                            ##print np.max(ctau_weights[k][ct])
                            for x in range(4):
                                max_weight = np.max(ctau_weights[k][ct])
                                max_weight_position = np.argmax(ctau_weights[k][ct])
                                mean_weight = np.mean(ctau_weights[k][ct])
                                ctau_weights[k][ct] = np.delete(ctau_weights[k][ct],max_weight_position)
                    y_b[samples[k]['mass']][ct] = ctau_weights[k][ct].sum()
                    stat_unc_b[samples[k]['mass']][ct] = np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()
                else:
                    print "Exceeded size of samples to combine, aborting..."
                    exit()

                #print "after: ", k,"m:%d, ct:%d, entries:%d, yield:%.6f +- %.2f (percentage)" % (samples[k]['mass'], ct, len(ctau_weights[k][ct]), ctau_weights[k][ct].sum(), 100*np.sqrt( sum(x*x for x in ctau_weights[k][ct]) ).sum()/ctau_weights[k][ct].sum())
                i +=1

    print y_comb
    print stat_unc_comb
    print "STOOOP"
    #exit()

    #print "let's not average uselessly"

    if "splitSUSY" in sign[0]:
        print "No average, moving on..."
    elif "zPrime" in sign[0]:
        print "No average, moving on..."
    elif "Stealth" in sign[0]:
        print "No average, moving on..."
    elif "HeavyHiggs" in sign[0]:
        print "No average, moving on..."
        #For ctau reweighting, the needed things are:
        #y_comb[m][ct]
        #stat_unc_comb[m][ct]
    else:
        for m in masses:
            #print "    **  ", " ---- mass: ", m, " ----"
            for ct in loop_ct:#ctaus:
                ##Super_Conservative approach: save whatever gives non-zero!
                #print "    **  ", "combine: "
                #print "    **  ", y_a[m][ct], "+-",stat_unc_a[m][ct]
                #print "    **  ", y_b[m][ct], "+-",stat_unc_b[m][ct]
                y_comb[m][ct] = error_weighted_average(y_a[m][ct],y_b[m][ct],stat_unc_a[m][ct]/y_a[m][ct],stat_unc_b[m][ct]/y_b[m][ct])# if y_a[m][ct]>0 and y_b[m][ct]>0 else 0
                stat_unc_comb[m][ct] = y_comb[m][ct]*error_weighted_relative_uncertainty(y_a[m][ct],y_b[m][ct],stat_unc_a[m][ct],stat_unc_b[m][ct],stat_unc_a[m][ct]/y_a[m][ct],stat_unc_b[m][ct]/y_b[m][ct])# if y_a[m][ct]>0 and y_b[m][ct]>0 else 0
                #print "    **  ", "ct: %d, combined yield: %.6f +- %.6f (%.2f perc.)" % (ct, y_comb[m][ct], stat_unc_comb[m][ct], 100* stat_unc_comb[m][ct]/y_comb[m][ct] if y_comb[m][ct]>0 else 0)
                #print "    **  ", "individual yields:  %.6f,  %.6f" %(y_a[m][ct],y_b[m][ct])

                for k in sgn_unc_dict[sign[0]].keys():
                    #FIXME!
                    unc_up = 100*abs(error_weighted_average(y_a[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau500"][k]/100.),y_b[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau3000"][k]/100.),stat_unc_a[m][ct]/y_a[m][ct],stat_unc_b[m][ct]/y_b[m][ct])/y_comb[m][ct] - 1) if y_comb[m][ct]>0 else 0
                    unc_down = 100*abs(error_weighted_average(y_a[m][ct]*(1-sgn_unc_dict["SUSY_mh"+str(m)+"_ctau500"][k]/100.),y_b[m][ct]*(1-sgn_unc_dict["SUSY_mh"+str(m)+"_ctau3000"][k]/100.),stat_unc_a[m][ct]/y_a[m][ct],stat_unc_b[m][ct]/y_b[m][ct])/y_comb[m][ct] - 1) if y_comb[m][ct]>0 else 0
                    sgn_unc_dict_ctau[m][ct][k] = max(unc_up,unc_down)
                    #print "   syst unc considered",k
                    #print "   yield 500:",y_a[m][ct]
                    #print "   yield 3000:",y_b[m][ct]
                    ##print "   stat unc 500:",stat_unc_a[m][ct]
                    ##print "   stat unc 3000:",stat_unc_b[m][ct]
                    ##print "   stat rel unc 500:",stat_unc_a[m][ct]/y_a[m][ct]
                    ##print "   stat rel unc 3000:",stat_unc_b[m][ct]/y_b[m][ct]
                    #print "   yield combined:",y_comb[m][ct]
                    #print "   syst unc 500: %.3f" % (sgn_unc_dict["SUSY_mh"+str(m)+"_ctau500"][k]/100.)
                    #print "   syst unc 3000: %.3f" % (sgn_unc_dict["SUSY_mh"+str(m)+"_ctau3000"][k]/100.)
                    ##print "   yield up 500:",y_a[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau500"][k]/100.)
                    ##print "   yield up 3000:",y_b[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau3000"][k]/100.)
                    ##print "   yield up combined:",error_weighted_average(y_a[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau500"][k]/100.),y_b[m][ct]*(1+sgn_unc_dict["SUSY_mh"+str(m)+"_ctau3000"][k]/100.),stat_unc_a[m][ct]/y_a[m][ct],stat_unc_b[m][ct]/y_b[m][ct])
                    #print "   unc rel up combined: %.3f" % (max(unc_up,unc_down)/100.)
                    #print "   unc rel down combined: %.3f" % (unc_down)
                    #print "\n"
                    #print "check dict: "
                    #print "sgn_unc_dict_ctau[",m," ][",ct,"]",k,"]", sgn_unc_dict_ctau[m][ct][k]

                #print "We can write the datacard at this point!"
                #print "Cosmic and BH must go as separate contribution"
                #print "Check if all the uncertainties have the right order of magnitude/relative/percentage"


    print "\n"

    #print bin2_entries.keys()
    #print bin2_entries[  bin2_entries.keys()[0]   ].sum()

    #Here: datacard
    for m in masses:
        print " ---- mass: ", m, " ----"
        n_pass_dc = 0
        for ct in loop_ct:#ctaus:
            #Rename for split susy
            if "splitSUSY" in sign[0]:
                s_rename = "splitSUSY_M2400_"+str(m)+"_ctau"+str(int(ct))+"p0"
                if ct!=samples[s_rename]['ctau']:
                    continue
                
                    y_2 = y_comb[m][ct]
                    e_2 = stat_unc_comb[m][ct]
            elif "Stealth" in sign[0]:
                if ct<1:
                    s_rename = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(ct).replace(".","p")+"_"+ch
                else:
                    s_rename = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(int(ct))+"_"+ch
                y_2 = y_comb[m][ct]
                e_2 = stat_unc_comb[m][ct]
            elif "zPrime" in sign[0]:
                print "FIXMEEEEE!!!!"
                print "FIXMEEEEE!!!!"
                print "FIXMEEEEE!!!!"
                print "FIXMEEEEE!!!!"
                s_rename = "zPrime"+ch+"_mZ"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    s_rename += str(int(ct))
                else:
                    s_rename += "0p1"
                y_2 = y_comb[m][ct]
                e_2 = stat_unc_comb[m][ct]
                #print masses
                #print loop_ct
                #if ct!=samples[s_rename]['ctau']:
                #    continue
                #else:
                #    print ct
                #    print samples[s_rename]['ctau']
                #    y_2 = y_comb[m][ct]
                #    e_2 = stat_unc_comb[m][ct]

            elif "HeavyHiggs" in sign[0]:
                print "FIXMEEEEE!!!!"
                #print masses
                #print loop_ct
                s_rename = "HeavyHiggsToLLP"+ch+"_mH"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    s_rename += str(int(ct))
                else:
                    s_rename += "0p1"
                y_2 = y_comb[m][ct]
                e_2 = stat_unc_comb[m][ct]
                #if ct!=samples[s_rename]['ctau']:
                #    y_2 = y_comb[m][ct]
                #    e_2 = stat_unc_comb[m][ct]
                #    #continue
                #else:
                #    y_2 = y_comb[m][ct]
                #    e_2 = stat_unc_comb[m][ct]

            else:
                s_rename = "SUSY_mh"+str(m)+"_ctau"+str(ct)
                y_2 = y_comb[m][ct]
                e_2 = stat_unc_comb[m][ct]
                if ct<=500:
                    s_rename_tmp = "SUSY_mh"+str(m)+"_ctau500"
                    n_pass_dc = bin2_entries[s_rename_tmp].sum()
                else:
                    s_rename_tmp = "SUSY_mh"+str(m)+"_ctau3000"
                    n_pass_dc = bin2_entries[s_rename_tmp].sum()

            #if y_2==0:
            #    print "Zero yield, next.... "
            #    continue
            ##Super_conservative: keep everything, will discard later
            #if y_2!=0 and e_2/y_2>0.75:
            #    print "Stat uncertainty exceeds 75 perc, next.... "
            #    continue

            #*******************************************************#
            #                                                       #
            #                      Datacard                         #
            #                                                       #
            #*******************************************************#
        
            card  = 'imax 1\n'#n of bins
            card += 'jmax *\n'#n of backgrounds
            card += 'kmax *\n'#n of nuisance parmeters
            card += 'shapes * * FAKE \n'#Fake shapes!
            card += '-----------------------------------------------------------------------------------\n'
            card += 'bin                            %s\n' % CHAN
            card += 'observation                    %f\n' % y_data
            card += '-----------------------------------------------------------------------------------\n'
            card += 'bin                            %-55s%-30s%-30s%-30s\n' % (CHAN, CHAN, CHAN, CHAN)
            card += 'process                        %-55s%-30s%-30s%-30s\n' % (s_rename, 'Bkg', 'Cosmic', 'BeamHalo')
            card += 'process                        %-55s%-30s%-30s%-30s\n' % ('0', '1', '2', '3')
            card += 'rate                           %-55f%-30f%-30f%-30f\n' % (y_2, y_bkg,cosmic["bkg_cosmic"],bh["bkg_bh"])
            ##One sided approach: Si
            ##card += 'rate                           %-30f%-30f%-30f%-30f\n' % (y_2, (y_bkg+y_bkg_from_0)/2.,cosmic["bkg_cosmic"],bh["bkg_bh"])
            card += '-----------------------------------------------------------------------------------\n'
            #Syst uncertainties
            #ORIGINAL

            #bkg
            for u in bkg_unc_dict.keys():
                #One sided approach
                #if u=="bkg_method":
                #    #Karim: one sided uncertainty
                #    #card += ('%s '+(22-len(u))*' '+'lnN     %-30s%-30s%-30s%-30s\n') % (u,'-', str(1.+bkg_unc_dict[u]/100.)+'/0.00001','-','-')
                #    #Si: half the uncertainty
                #    card += ('%s '+(22-len(u))*' '+'lnN     %-30s%-30f%-30s%-30s\n') % (u,'-', 1.+bkg_unc_dict[u]/200.,'-','-')
                #else:
                #    card += ('%s '+(22-len(u))*' '+'lnN     %-30s%-30f%-30s%-30s\n') % (u,'-', 1.+bkg_unc_dict[u]/100.,'-','-')
                card += ('%s '+(22-len(u))*' '+'lnN     %-55s%-30f%-30s%-30s\n') % (u,'-', 1.+bkg_unc_dict[u]/100.,'-','-')
            #cosmic
            card += ('%s '+(22-len("unc_cosmic"))*' '+'lnN     %-55s%-30s%-30f%-30s\n') % ("unc_cosmic",'-', '-', 1.+ cosmic["unc_cosmic"]/100., '-')
            #bh
            card += ('%s '+(22-len("unc_bh"))*' '+'lnN     %-55s%-30s%-30s%-30f\n') % ("unc_bh",'-', '-', '-', 1.+bh["unc_bh"]/100.)
            #sgn
            card += '%-18s     lnN     %-55f%-30s%-30s%-30s\n' % ('sig_stat_'+ERA,1.+e_2/y_2 if y_2>0 else 1.,'-','-','-')
            for u in sgn_unc_dict_ctau[m][ct].keys():
                card += ('%s '+(22-len(u))*' '+'lnN     %-55f%-30s%-30s%-30s\n') % (u, 1.+sgn_unc_dict_ctau[m][ct][u]/100.,'-','-','-')
                #print "sgn_unc_dict_ctau[ ",m," ][",ct,"]",u,"]", 1.+sgn_unc_dict_ctau[m][ct][u]/100.

            '''

            #KARIM

            #bkg
            for u in bkg_unc_dict.keys():
                #card += ('%s '+(22-len(u))*' '+'lnN     %-30s%-30f%-30s%-30s\n') % (u,'-', min(1.+bkg_unc_dict[u]/100.,1.999),'-','-')
                card += ('%s '+(22-len(u))*' '+'lnN     %-30s%f/%-20f %-30s%-30s\n') % (u,'-', max(0.001,1.-bkg_unc_dict[u]/100.),min(1.+bkg_unc_dict[u]/100.,1.999),'-','-')
            #cosmic
            card += ('%s '+(22-len("unc_cosmic"))*' '+'lnN     %-30s%-30s %f/%-20f  %-30s\n') % ("unc_cosmic",'-', '-', max(0.001, 1.- cosmic["unc_cosmic"]/100.),min(1.+ cosmic["unc_cosmic"]/100.,1.999), '-')
            #bh
            card += ('%s '+(22-len("unc_bh"))*' '+'lnN     %-30s%-30s%-30s %f/%-20f\n') % ("unc_bh",'-', '-', '-', max(0.001,1.-bh["unc_bh"]/100.),min(1.+bh["unc_bh"]/100.,1.999))
            #sgn
            card += '%-18s     lnN     %f/%-20f%-30s%-30s%-30s\n' % ('sig_stat_'+ERA,max(0.001,1.-e_2/y_2) if y_2>0 else 1.,min(1.+e_2/y_2,1.999) if y_2>0 else 1.,'-','-','-')
            for u in sgn_unc_dict_ctau[m][ct].keys():
                #card += ('%s '+(22-len(u))*' '+'lnN     %-30f%-30s%-30s%-30s\n') % (u, min(1.+sgn_unc_dict_ctau[m][ct][u]/100.,1.99),'-','-','-')
                card += ('%s '+(22-len(u))*' '+'lnN     %f/%-20f%-30s%-30s%-30s\n') % (u, max(0.001,1.-sgn_unc_dict_ctau[m][ct][u]/100.),min(1.+sgn_unc_dict_ctau[m][ct][u]/100.,1.99),'-','-','-')

            print card
            exit()
            '''
            print card
            outname = DATACARDS+ s_rename + dataset_label+'.txt'
            cardfile = open(outname, 'w')
            cardfile.write(card)
            cardfile.close()
            print "Info: " , outname, " written"

            with open(DATACARDS+ s_rename + dataset_label+".yaml","w") as f:
                y_sign = {}
                y_sign["y"] = y_2
                y_sign["e"] = e_2
                y_sign["r"] = e_2/y_2 if y_2>0 else 0.
                y_sign["y_500"] = y_a[m][ct]
                y_sign["e_500"] = stat_unc_a[m][ct]
                y_sign["y_3000"] = y_b[m][ct]
                y_sign["e_3000"] = stat_unc_b[m][ct]
                y_sign["stat"] = e_2/y_2*100. if y_2>0 else 0.
                y_sign["syst"] = 0
                #here dump also syst unc, gen level yield and efficiency?
                for u in sgn_unc_dict_ctau[m][ct].keys():
                    y_sign[u] = sgn_unc_dict_ctau[m][ct][u]
                    y_sign["syst"] += (sgn_unc_dict_ctau[m][ct][u]/100.)**2
                y_sign["syst"] = math.sqrt(y_sign["syst"])*100.
                y_sign["y_entries"] = n_pass_dc
                #y_sign["eff"] = 100.*n_pass/sample[samples[s_rename]['files'][0]]['nevents']
                y_sign["eff"] = 0.
                if (ct==500 or ct==3000) and ("SUSY" in s_rename):
                    if BR_SCAN_H==100:
                        y_sign["eff"] = 100.*n_pass_dc/sample[samples[s_rename+"_HH"]['files'][0]]['nevents']
                    if BR_SCAN_H==0:
                        y_sign["eff"] = 100.*n_pass_dc/sample[samples[s_rename+"_ZZ"]['files'][0]]['nevents']
                yaml.dump(y_sign, f)
                f.close()
            print "Info: dictionary written in file "+DATACARDS+ s_rename + dataset_label+".yaml"
        
            #combine commands
            #Limits
            if run_combine:
                original_location = os.popen("pwd").read()
                os.chdir("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit")
                #os.system("pwd")
                print "\n"

                #os.system("source /cvmfs/cms.cern.ch/cmsset_default.sh")
                #os.system("eval `scramv1 runtime -sh`")
                #os.system("pwd")
                #os.system("echo $CMSSW_BASE \n")
                tmp  = "#!/bin/sh\n"
                tmp += "source /etc/profile.d/modules.sh\n"
                tmp += "module use -a /afs/desy.de/group/cms/modulefiles/\n"
                tmp += "module load cmssw\n"
                #tmp += "pwd\n"
                tmp += "cmsenv\n"
                workspace = s_rename+dataset_label+".root"
                #writes directly without displaying errors
                #tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[s]['mass']) + " | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ".txt\n"
                #tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
                #tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ".txt\n"

                #print screen
                #Asymptotic
                tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[pr]['mass']) + " -n " + s_rename+dataset_label +" \n"
                #Toys
                #Observed
                #tmp += "combine -M HybridNew --LHCmode LHC-limits --datacard " + outname + "  -m " + str(samples[pr]['mass']) + " -n " + s_rename+dataset_label +" \n"
                #Result stored in the limit branch
                #Expected median: --expectedFromGrid=0.025,0.16,0.5,0.84,0.975
                #For blind limits: add -t -1 (Asimov instead of data)
                #Result stored in the quantileExpected branch
                tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
                tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 -n Sign"+s_rename+dataset_label+" \n"
                if ct==500:
                    print "Run fit diagnostics..."
                    print "Pulls"
                    print "combine -M FitDiagnostics " + outname + " --name " + s_rename+dataset_label + " --plots --forceRecreateNLL -m "+ str(samples[pr]['mass'])+" -n "+s_rename + dataset_label
                    #tmp += "combine -M FitDiagnostics " + outname + " --name " + s+contam_lab + " --plots --forceRecreateNLL -m "+ str(samples[s]['mass'])+"\n"
                    print "python /afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics"+s_rename+dataset_label +".root --all --abs --pullDef relDiffAsymErrs -g pulls"+s_rename+dataset_label+".root"
                    #tmp += "python /afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics"+s+contam_lab +".root --all --abs --pullDef relDiffAsymErrs -g pulls"+s+contam_lab +".root \n"
                    print "Impacts"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --doInitialFit --robustFit 1 --rMin -5 --rMax 5 \n"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --robustFit 1 --doFits --rMin -5 --rMax 5 \n"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" -o impacts"+ s+contam_lab +".json \n"
                    #tmp += "plotImpacts.py -i impacts"+ s+contam_lab +".json -o impacts"+ s+contam_lab + " \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[pr]['mass']) +" --doInitialFit --robustFit 1 --rMin -5 --rMax 5 \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[pr]['mass']) +" --robustFit 1 --doFits --rMin -5 --rMax 5 \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[pr]['mass']) +" -o impacts"+ s_rename +".json \n"
                    print "plotImpacts.py -i impacts"+ s_rename +dataset_label+".json -o impacts"+ s_rename+dataset_label + " \n"
                    #exit()
                job = open("job.sh", 'w')
                job.write(tmp)
                job.close()
                os.system("sh job.sh > log.txt \n")
                os.system("\n")
                os.system("cat log.txt \n")
                os.system("cat log.txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s_rename +dataset_label+".txt  \n")
                os.system("cat log.txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s_rename +dataset_label+".txt\n")
                #os.system("cat "+ RESULTS + "/" + s + ".txt  \n")
                print "\n"
                print "Limits written in ", RESULTS + "/" + s_rename + dataset_label+".txt"
                print "Significance written in ", RESULTS + "/Significance_" + s_rename + dataset_label + ".txt"
                print "*********************************************"

                os.chdir(original_location[:-1])
                os.system("eval `scramv1 runtime -sh`")
                os.system("pwd")



def error_weighted_average(y_a,y_b,rel_u_a,rel_u_b):
    #print "y_a,y_b,rel_u_a,rel_u_b"
    #print y_a,y_b,rel_u_a,rel_u_b
    y_comb = 0.
    if y_a!=0. and y_b!=0.:
        y_comb = (y_a * (1/(rel_u_a)) + y_b * (1/(rel_u_b)) )/( 1/(rel_u_a) + 1/(rel_u_b)  )
    #New: treat cases of y_a or y_b being zero
    if y_a!=0. and y_b==0.:
        y_comb = (y_a * (1/(rel_u_a)) )/( 1/(rel_u_a) )
    if y_a==0. and y_b!=0.:
        y_comb = (y_b * (1/(rel_u_b)) )/( 1/(rel_u_b) )
    #print "y_comb"
    #print y_comb
    return y_comb

def error_weighted_relative_uncertainty(y_a,y_b,u_a,u_b,rel_u_a,rel_u_b):
    unc_comb = 0
    if y_a!=0 and y_b!=0:
        unc_comb = math.sqrt(  (u_a/rel_u_a)**2 + (u_b/rel_u_b)**2 ) / ( y_a*(1/rel_u_a) + y_b*(1/rel_u_b) )
    if y_a!=0 and y_b==0:
        unc_comb = rel_u_a
    if y_a==0 and y_b!=0:
        unc_comb = rel_u_b
    return unc_comb



def submit_combine_condor(sign,comb_fold_label="",eras=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=True,phi_cut=False,BR_SCAN_H=100,blind=True,toys=False):

    NCPUS   = 1
    MEMORY  = 512
    RUNTIME = 3600
    
    print "Inferring limits on absolute x-sec in fb"

    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_preapproval/"#"_TEST_remove_outliers/"
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding/"#"_TEST_remove_outliers/"
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding_one_sided_Si/"#"_TEST_remove_outliers/"
    #Approval:
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding_ARC/"#"_TEST_remove_outliers/"
    #CWR
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_CWR/"
    #DS
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_DS/"

    if not os.path.isdir(OUTCOMBI): os.mkdir(OUTCOMBI)
    DATACARDDIR = OUTCOMBI
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDDIR+=CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    CARDERA += CHAN+"/"

    br_scan_fold = ""
    if BR_SCAN_H==100:
        br_scan_fold = "BR_h100_z0"
    if BR_SCAN_H==75:
        br_scan_fold = "BR_h75_z25"
    if BR_SCAN_H==50:
        br_scan_fold = "BR_h50_z50"
    if BR_SCAN_H==25:
        br_scan_fold = "BR_h25_z75"
    if BR_SCAN_H==0:
        br_scan_fold = "BR_h0_z100"
    if "splitSUSY" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "splitSUSY"
    if "zPrime" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "zPrime"
    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"
    if "HeavyHiggs" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "HeavyHiggs"

    DATACARDDIR += br_scan_fold+"/"
    CARDERA += br_scan_fold+"/"
    if comb_fold_label!="":
        if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
        print comb_fold_label
        DATACARDDIR += comb_fold_label+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"datacards/"
    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    CARDERA += "datacards/"

    #This is needed to read the asymptotic results for setting the toys grid
    RESULTS_ASYMPTOTIC = DATACARDDIR+"combine_results/"

    RESULTS = DATACARDDIR+"combine_results"
    if toys:
        RESULTS+="_toys"
    RESULTS+="/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

    print "Writing results in: " , RESULTS

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    #Dictionary with various predictions
    mass = []
    mean_val = {}
    sigma_1_up = {}
    sigma_1_down = {}
    sigma_2_up = {}
    sigma_2_down = {}

    card_dict = {}
    for era in eras:
        if "2016" in era:
            card_dict[era] = CARDERA % ("2016")
        else:
            card_dict[era] = CARDERA % (era)



    masses = []

    if "splitSUSY" in sign[0]:
        loop_ct = ctaus_split
    elif "zPrime" in sign[0]:
        loop_ct = ctaus_zPrime
    elif "Stealth" in sign[0]:
        loop_ct = ctaus_stealth
    elif "HeavyHiggs" in sign[0]:
        loop_ct = ctaus_HeavyHiggs
    else:
        loop_ct = ctaus

    for pr in sign:
        loop_ct = np.append(loop_ct,np.array([samples[pr]['ctau']]))
        masses.append(samples[pr]['mass'])
    masses = np.unique(np.array(masses))
    loop_ct = np.unique(loop_ct)

    #print "Evaluating:"
    #print masses, loop_ct
    #exit()

    COND_DIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/condor_"+br_scan_fold
    COMB_DIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"
    if toys:
        COND_DIR="/nfs/dust/cms/user/lbenato/condor_"+br_scan_fold+"_toys"
    if not(os.path.exists(COND_DIR)):
        os.mkdir(COND_DIR)

    print "\n"
    print "Condor stuff being stored in ", COND_DIR
    print "\n"

    original_location = os.popen("pwd").read()

    for m in masses:
        for ct in loop_ct:
            print "extrapolation to ctau",ct
            inp_card_string = ""
            if "splitSUSY" in sign[0]:
                card_name = "splitSUSY_M2400_"+str(m)+"_ctau"+str(int(ct))+"p0"
            elif "zPrime" in sign[0]:
                #QUA
                card_name = "zPrime"+ch+"_mZ"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    card_name += str(int(ct))
                else:
                    card_name += "0p1"
                print card_name

            elif "Stealth" in sign[0]:
                #QUA
                if ct<1:
                    card_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(ct).replace(".","p")+"_"+ch
                else:
                    card_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(int(ct))+"_"+ch
                print card_name

            elif "HeavyHiggs" in sign[0]:
                #QUA
                card_name = "HeavyHiggsToLLP"+ch+"_mH"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    card_name += str(int(ct))
                else:
                    card_name += "0p1"
                print card_name

            else:
                card_name = "SUSY_mh"+str(m)+"_ctau"+str(ct)
            n_valid = 0
            yield_dict_era = {}
            error_dict_era = {}
            rel_error_dict_era = {}
            for c in card_dict.keys():
                tmp_yaml = card_dict[c]+card_name
                tmp_card = card_dict[c]+card_name
                if "2016" in c:
                    tmp_card += c.replace("2016","")
                    tmp_yaml += c.replace("2016","")
                print "checking ", tmp_card
                tmp_card += ".txt"
                tmp_yaml += ".yaml"
                if os.path.isfile(tmp_card):
                    n_valid+=1

                with open(tmp_yaml,"r") as f:
                    y_sign = yaml.load(f, Loader=yaml.Loader)
                    yield_dict_era[c] = y_sign['y'] if y_sign!=None else 0
                    error_dict_era[c] = y_sign['e'] if y_sign!=None else 0
                    rel_error_dict_era[c] = y_sign['r'] if y_sign!=None else 0
                    #print "era", c, ", yield", y_sign['y'] if y_sign!=None else 0,"+-", y_sign['e'] if y_sign!=None else 0," (", 100*y_sign['e']/y_sign['y'] if y_sign!=None and y_sign['y']>0 else 0," perc.)"
                f.close()

            stat_unc_tot = 0
            ev_yield_tot = 0
            for c in yield_dict_era.keys():
                #print "Scrutinize era",c
                ev_yield_tot += yield_dict_era[c]

                #naive:
                #stat_unc_tot += (error_dict_era[c])**2                
                    
                #Reason: we use 2016 MC twice. Let's sum the stat uncertainty only once
                if "2016" not in c:
                    stat_unc_tot += (error_dict_era[c])**2
                else:
                    stat_unc_tot +=( rel_error_dict_era["2016_G-H"]*(yield_dict_era["2016_G-H"] + yield_dict_era["2016_B-F"])  )**2

            print "yield: ",ev_yield_tot,"+-", math.sqrt(stat_unc_tot)," (", 100*math.sqrt(stat_unc_tot)/ev_yield_tot if ev_yield_tot>0 else 0," perc.)"

            #if n_valid==4:
            if ev_yield_tot>0:
                if "SUSY" in sign[0]:
                    if math.sqrt(stat_unc_tot)/ev_yield_tot<0.75:
                        for c in card_dict.keys():
                            inp_card_string += card_dict[c]+card_name
                            if "2016" in c:
                                inp_card_string += c.replace("2016","")
                            inp_card_string +=".txt" + " "
                    else:
                        continue
                else:
                    for c in card_dict.keys():
                        inp_card_string += card_dict[c]+card_name
                        if "2016" in c:
                            inp_card_string += c.replace("2016","")
                        inp_card_string +=".txt" + " "
            else:
                continue

            #print inp_card_string
            original_location = os.popen("pwd").read()


            #condor stuff
            if run_combine and n_valid==4:
                #os.system("pwd")
                print "\n"
                #print "Combination card " + DATACARDS+card_name+".txt "


                #Delete old stuff

                #One per quantile/obs
                if toys:

                    #############################
                    os.chdir(COMB_DIR)
                
                    print "~~~~~~ Limits with toys and grid ~~~~~~"
                    blind = False
                    #print "1. Read the asymptotic result to decide the grid"
                    card_asymptotic_name =  RESULTS_ASYMPTOTIC+card_name+".txt"
                    card_asymptotic = open( card_asymptotic_name, 'r')
                    string_val = card_asymptotic.read().splitlines()
                    val = [float(x) for x in string_val]
                    #if m<210:
                    #    print "Adjust signal multiplication (enhance signal to avoid combine instabilities)"
                    #    val = [10.*x for x in val]

                    #grid = np.linspace()
                    print "Asymptotic limits for ", card_name
                    print val

                    if len(val) == 0:
                        continue
                    if len(val) != 6 and not blind:
                        continue
                
                    min_val = min(val)
                    min_val_digit = round_to_1(min_val)
                    max_val = max(val)
                    max_val_digit = round_to_1(max_val)
                    n_grid = 50#2#0
                    N_TOYS = 1000#500 was already okay but I need more...
                    N_ITERATIONS = 20#1
                    if BR_SCAN_H==50 and (m==1000 or m==400 or ct==500 or ct==3000):
                        N_TOYS = 1000
                        N_ITERATIONS = 50
                        MEMORY = 512*5
                        RUNTIME = 3600*5

                    step = round_to_1( (max_val_digit-min_val_digit)/n_grid)
                    grid = np.arange(min_val_digit-step,max_val_digit+2*step,step)
                    start_grid = min_val_digit-step if min_val_digit-step>0 else min_val_digit
                    stop_grid = max_val_digit+2*step

                    print "min value for grid: ", min_val, min_val_digit
                    print "max value for grid: ", max_val, max_val_digit
                    print "step: ", step
                    print "grid: ", grid
                    print "start_grid: ", start_grid
                    print "stop_grid: ", stop_grid

                    #print "2. Create a new folder for the grid points and move there"
                    os.system("pwd")
                    grid_dir = COND_DIR+"/"+card_name+"_grid"
                    if not os.path.isdir(grid_dir): os.mkdir(grid_dir)
                    os.chdir(grid_dir)

                    print "Deleting old condor outputs ... "
                    print 'job_'+card_name+'.sh'#'m'+str(m)+'_ctau'+str(ct)+'.sh'
                    if os.path.isfile('job_'+card_name+'.sh'):#'job_m'+str(m)+'_ctau'+str(ct)+'.sh'):
                        os.system("rm " + 'job_'+card_name+'.sh*')#'job_m'+str(m)+'_ctau'+str(ct)+'.sh*')
                    if os.path.isfile('submit_'+card_name+'.submit'):
                        os.system("rm " + 'submit_'+card_name+'.submit*')
                    if os.path.isfile('log_'+card_name+'.txt'):
                        os.system("rm " + 'log_'+card_name+'.txt*')
                    if os.path.isfile('out_'+card_name+'.txt'):
                        os.system("rm " + 'out_'+card_name+'.txt*')
                        os.system("rm " + 'higgsCombine*.root')
                        os.system("rm " + 'merged*.root')
                    if os.path.isfile('error_'+card_name+'.txt'):
                        os.system("rm " + 'error_'+card_name+'.txt*')

                    #print "3. Generate the workspace"
                    #print "4. Generate the grid"
                    #print "5. Do the merging and remove the unnecessary stuff"
                    #print "6. Debug the test statistics"
                    #print "7. Calculate the quantiles --> need to understand how easy this is to read from the output, might be the order is messed up ---> if it's a problem, read it from root at retrieving step"

                    with open('job_'+card_name+'.sh', 'w') as fout:
                        fout.write("#!/bin/sh\n")
                        fout.write("source /etc/profile.d/modules.sh\n")
                        fout.write("module use -a /afs/desy.de/group/cms/modulefiles/\n")
                        fout.write("module load cmssw\n")
                        fout.write('cd '+COMB_DIR+' \n')
                        fout.write("cmsenv\n")
                        fout.write('cd '+grid_dir+' \n')
                        workspace = card_name+".root"
                        #fout.write("combineCards.py " + inp_card_string + " > " + DATACARDS+card_name+".txt " + " \n")
                        fout.write("text2workspace.py " + DATACARDS+card_name+".txt  -o " + DATACARDS+workspace + " \n")
                        fout.write("echo Running on: " + DATACARDS+workspace +" from directory: \n")
                        fout.write("pwd \n")
                        fout.write("combineTool.py -M HybridNew --LHCmode LHC-limits --datacard "+DATACARDS+workspace+" -m "+str(m)+" -n " + card_name+"_br"+str(BR_SCAN_H) + " -T "+str(N_TOYS)+" --iterations "+str(N_ITERATIONS)+" --singlePoint "+str(start_grid)+":"+str(stop_grid)+":"+str(step)+" --saveToys --saveHybridResult --clsAcc 0 --verbose 1 --rMax "+str( max(20,stop_grid)  )+" \n")
                        fout.write("hadd -f merged"+card_name+"_br"+str(BR_SCAN_H)+".root higgsCombine"+card_name+"_br"+str(BR_SCAN_H)+".POINT*.root \n")
                        fout.write("rm higgsCombine"+card_name+"_br"+str(BR_SCAN_H)+".POINT*.root \n")
                        fout.write("python /afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit//test/plotTestStatCLs.py --input merged"+card_name+"_br"+str(BR_SCAN_H)+".root --poi r --val all --mass "+str(m)+" \n")

                        for gr in ["0.025","0.16","0.5","0.84","0.975","obs"]:
                            if gr=="obs":
                                fout.write("combine -M HybridNew --LHCmode LHC-limits --datacard " +DATACARDS+workspace+ " -m "+str(m)+" -n "+card_name+"_br"+str(BR_SCAN_H)+"_"+gr+" --readHybridResults --grid=merged"+card_name+"_br"+str(BR_SCAN_H)+".root  --plot=limit_scan"+card_name+"_br"+str(BR_SCAN_H)+"_"+gr+".png \n")
                            else:
                                fout.write("combine -M HybridNew --LHCmode LHC-limits --datacard " +DATACARDS+workspace+ " -m "+str(m)+" -n "+card_name+"_br"+str(BR_SCAN_H)+"_"+gr+" --readHybridResults  --expectedFromGrid="+str(gr)+" --grid=merged"+card_name+"_br"+str(BR_SCAN_H)+".root  --plot=limit_scan"+card_name+"_br"+str(BR_SCAN_H)+"_exp_q_"+str(gr).replace(".","p")+".png \n")

                    os.system('chmod 755 job_'+card_name+'.sh')

                    with open('submit_'+card_name+'.submit', 'w') as fout:
                        fout.write('executable   = ' + grid_dir + '/job_'+card_name+'.sh \n')
                        fout.write('output       = ' + grid_dir + '/out_'+card_name+'.txt \n')
                        fout.write('error        = ' + grid_dir + '/error_'+card_name+'.txt \n')
                        fout.write('log          = ' + grid_dir + '/log_'+card_name+'.txt \n')
                        fout.write(' \n')
                        fout.write('Requirements = OpSysAndVer == "CentOS7" \n')
                        fout.write(' \n')
                        fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                        #when issues: 8000
                        TOYS_MEMORY = MEMORY*2#*10
                        fout.write('Request_Memory = ' + str(TOYS_MEMORY) + ' \n')
                        #when issues: 43200
                        TOYS_RUNTIME = RUNTIME*10
                        fout.write('+RequestRuntime = ' + str(TOYS_RUNTIME) + ' \n')
                        fout.write('batch_name = m'+str(m)+'ct'+str(ct)+' \n')
                        fout.write('queue 1 \n')

                    #submit condor
                    os.system('condor_submit ' + 'submit_'+card_name+'.submit' + ' \n')
                    print "\n"
                    print "\n"
                    #os.system('cat job_m'+str(m)+'_ctau'+str(ct)+'.sh \n')
                    #print grid_dir +'/job_m'+str(m)+'_ctau'+str(ct)+'.sh'

                    ############################

                else:
                    os.chdir(COND_DIR)
                    print "----->"
                    print "Doing Asymptotic!!!!!!!!!"
                    if "zPrime" in sign[0] or "HeavyHiggs" in sign[0]:
                        if ct>1:
                            ctn = str(int(ct))
                        else:
                            ctn = "0p1"
                        ctn+=ch
                    else:
                        if "Stealth" in sign[0]:
                            if ct<1:
                                ctn=str(ct).replace(".","p")
                            else:
                                ctn=str(int(ct))

                    with open('job_'+card_name+'.sh', 'w') as fout:
                        fout.write("#!/bin/sh\n")
                        fout.write("source /etc/profile.d/modules.sh\n")
                        fout.write("module use -a /afs/desy.de/group/cms/modulefiles/\n")
                        fout.write("module load cmssw\n")
                        fout.write("cmsenv\n")
                        fout.write('cd '+COND_DIR+' \n')
                        workspace = card_name+".root"
                        fout.write("combineCards.py " + inp_card_string + " > " + DATACARDS+card_name+".txt " + " \n")
                        fout.write("echo Running on: " + DATACARDS+card_name+".txt from directory: \n")
                        fout.write("pwd \n")
                        if blind:
                            fout.write("combine -M AsymptoticLimits --datacard " + DATACARDS+card_name+".txt " + "  --run blind -m " + str(m) + " -n " +card_name+"_br"+str(BR_SCAN_H)+ " \n")
                        else:
                            fout.write("combine -M AsymptoticLimits --datacard " + DATACARDS+card_name+".txt " + " -m " + str(m) + " -n " +card_name+"_br"+str(BR_SCAN_H)+ " \n")

                    os.system('chmod 755 job_'+card_name+'.sh')

                    with open('submit_'+card_name+'.submit', 'w') as fout:
                        fout.write('executable   = ' + COND_DIR + '/job_'+card_name+ '.sh \n')
                        fout.write('output       = ' + COND_DIR + '/out_'+ card_name+ '.txt \n')
                        fout.write('error        = ' + COND_DIR + '/error_'+ card_name+ '.txt \n')
                        fout.write('log          = ' + COND_DIR + '/log_'+ card_name+ '.txt \n')
                        fout.write(' \n')
                        fout.write('Requirements = OpSysAndVer == "CentOS7" \n')
                        fout.write(' \n')
                        fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                        fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                        #fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                        fout.write('batch_name = m'+str(m)+'ct'+str(ctn)+' \n')
                        fout.write('queue 1 \n')

                    #submit condor
                    #os.system('condor_submit ' + 'submit_'+card_name+'.submit' + ' \n')
                    os.system('sh job_'+card_name+'.sh >  out_'+ card_name+ '.txt \n')

                os.chdir(original_location[:-1])
                os.system("eval `scramv1 runtime -sh`")
                #os.system("pwd")



def retrieve_combine_condor(sign,comb_fold_label="",eras=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=True,phi_cut=False,BR_SCAN_H=100,blind=True,toys=False):

    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_preapproval/"#"_TEST_remove_outliers/"
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding/"#"_TEST_remove_outliers/"
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding_one_sided_Si/"#"_TEST_remove_outliers/"
    #Approval:
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_unblinding_ARC/"#"_TEST_remove_outliers/"
    #CWR:
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_CWR/"
    #DS:
    CARDERA            = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_"+REGION+"_DS/"

    if not os.path.isdir(OUTCOMBI): os.mkdir(OUTCOMBI)
    DATACARDDIR = OUTCOMBI+CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    CARDERA += CHAN+"/"

    br_scan_fold = ""
    if BR_SCAN_H==100:
        br_scan_fold = "BR_h100_z0"
    if BR_SCAN_H==75:
        br_scan_fold = "BR_h75_z25"
    if BR_SCAN_H==50:
        br_scan_fold = "BR_h50_z50"
    if BR_SCAN_H==25:
        br_scan_fold = "BR_h25_z75"
    if BR_SCAN_H==0:
        br_scan_fold = "BR_h0_z100"
    if "splitSUSY" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "splitSUSY"
    if "zPrime" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "zPrime"
    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"
    if "HeavyHiggs" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "HeavyHiggs"

    DATACARDDIR += br_scan_fold
    CARDERA += br_scan_fold

    if toys:
        DATACARDDIR+="_toys"
    DATACARDDIR+="/"

    if comb_fold_label!="":
        if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
        print comb_fold_label
        DATACARDDIR += comb_fold_label+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"datacards/"
    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    CARDERA += "datacards/"

    RESULTS = DATACARDDIR+"combine_results/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

    print "Writing results in: " , RESULTS


    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    #Dictionary with various predictions
    mass = []
    mean_val = {}
    sigma_1_up = {}
    sigma_1_down = {}
    sigma_2_up = {}
    sigma_2_down = {}

    card_dict = {}
    for era in eras:
        if "2016" in era:
            card_dict[era] = CARDERA % ("2016")
        else:
            card_dict[era] = CARDERA % (era)

    masses = []

    if "splitSUSY" in sign[0]:
        loop_ct = ctaus_split
    elif "zPrime" in sign[0]:
        loop_ct = ctaus_zPrime
    elif "Stealth" in sign[0]:
        loop_ct = ctaus_stealth
    elif "HeavyHiggs" in sign[0]:
        loop_ct = ctaus_HeavyHiggs
    else:
        loop_ct = ctaus
    for pr in sign:
        loop_ct = np.append(loop_ct,np.array([samples[pr]['ctau']]))
        masses.append(samples[pr]['mass'])
    masses = np.unique(np.array(masses))
    loop_ct = np.unique(loop_ct)

    COND_DIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/condor_"+br_scan_fold
    if toys:
        COND_DIR="/nfs/dust/cms/user/lbenato/condor_"+br_scan_fold+"_toys"
        ###COND_DIR+="_approval_presentation_frozen_version"
        #COND_DIR+="_toys"
    COND_DIR+="/"
    #if not(os.path.exists(COND_DIR)):
    #    os.mkdir(COND_DIR)
    print COND_DIR
    print masses
    print loop_ct
    

    for m in masses:
        for ct in loop_ct:

            if toys:

                '''
                #METHOD without grid
                print "toys"
                out_values = []
                card_name = "SUSY_mh"+str(m)+"_ctau"+str(ct)

                for gr in ["obs","0.025","0.16","0.5","0.84","0.975"]:
                    out_name = COND_DIR + '/out_m'+ str(m) +'_ctau'+str(ct)+'_'+gr.replace('.','p')+".txt"
                    if not os.path.isfile(out_name):
                        continue
                    print "I read the condor output from here: ", out_name
                    #os.system("cat "+out_name+" \n")
                    file_read = open(out_name, "r")
                    lines = file_read.readlines()
                    for line in lines:
                        if "Limit: r <" in line:
                            print gr, line
                            line = line.replace("Limit: r < ","")
                            out_values.append(line.split('+')[0])
                #now store these out_values in one file
                with open(RESULTS+"/"+card_name+'.txt', 'w') as fout:
                    for v in out_values:
                        print v
                        fout.write(v + "\n")
                    fout.close()

                #os.system("cat "+out_name+" | grep -e Limit: r | awk '{print $NF}' > " + RESULTS + "/" + card_name +".txt  \n")
                #os.system("cat "+out_name+" | grep -e 'Limit: r <' | awk '{print $NF}' > " + RESULTS + "/" + card_name +".txt  \n")
                print "Limits written in ", RESULTS + "/" + card_name + ".txt"
                print "*********************************************"
                '''

                #"METHOD with grid"
                print "toys"
                out_values = []
                card_name = "SUSY_mh"+str(m)+"_ctau"+str(ct)
                print "Doing: ", card_name
                grid_dir = COND_DIR+"/"+card_name+"_grid/"
                if not os.path.isdir(grid_dir): 
                    continue
                #os.system("ls "+grid_dir+" \n")
                #Combine default labels have more digits
                dict_combine_names = {"obs":"obs","0.025":"0.025","0.16":"0.160","0.5":"0.500","0.84":"0.840","0.975":"0.975"}
                for gr in ["obs","0.025","0.16","0.5","0.84","0.975"]:
                    if gr=="obs":
                        root_file_name = "higgsCombine"+card_name+"_br"+str(BR_SCAN_H)+"_"+str(gr)+".HybridNew.mH"+str(m)+".root"
                    else:
                        root_file_name = "higgsCombine"+card_name+"_br"+str(BR_SCAN_H)+"_"+str(gr)+".HybridNew.mH"+str(m)+".quant"+str(dict_combine_names[gr])+".root"

                    if not os.path.isfile(grid_dir+root_file_name):
                        print " --->"
                        print " ---> Seems like ", grid_dir+root_file_name, " does not exist!!! "
                        print " --->"
                        continue
                    
                    print root_file_name
                    combine_file = TFile(grid_dir+root_file_name,"READ")
                    combine_file.cd()
                    #combine_file.ls()
                    limit = combine_file.Get("limit")
                    #limit.Scan("limit:quantileExpected")
                    limit.GetEntry(0)
                    limit.SetDirectory(0)
                    combine_file.Close()
                    out_values.append(str(limit.limit))

                #os.system("cat "+grid_dir+"out_*txt \n")
                print out_values

                #now store these out_values in one file
                with open(RESULTS+"/"+card_name+'.txt', 'w') as fout:
                    for v in out_values:
                        print v
                        fout.write(v + "\n")
                    fout.close()

                print "Limits written in ", RESULTS + "/" + card_name + ".txt"
                print "*********************************************"

                '''
                for gr in ["obs","0.025","0.16","0.5","0.84","0.975"]:
                    out_name = COND_DIR + '/out_m'+ str(m) +'_ctau'+str(ct)+'_'+gr.replace('.','p')+".txt"
                    if not os.path.isfile(out_name):
                        continue
                    print "I read the condor output from here: ", out_name
                    #os.system("cat "+out_name+" \n")
                    file_read = open(out_name, "r")
                    lines = file_read.readlines()
                    for line in lines:
                        if "Limit: r <" in line:
                            print gr, line
                            line = line.replace("Limit: r < ","")
                            out_values.append(line.split('+')[0])
                #now store these out_values in one file
                with open(RESULTS+"/"+card_name+'.txt', 'w') as fout:
                    for v in out_values:
                        print v
                        fout.write(v + "\n")
                    fout.close()

                #os.system("cat "+out_name+" | grep -e Limit: r | awk '{print $NF}' > " + RESULTS + "/" + card_name +".txt  \n")
                #os.system("cat "+out_name+" | grep -e 'Limit: r <' | awk '{print $NF}' > " + RESULTS + "/" + card_name +".txt  \n")
                print "Limits written in ", RESULTS + "/" + card_name + ".txt"
                print "*********************************************"

                '''


            else:
                if "splitSUSY" in sign[0]:
                    card_name = "splitSUSY_M2400_"+str(m)+"_ctau"+str(int(ct))+"p0"
                    ctn = ct
                elif "zPrime" in sign[0]:
                    #QUA
                    card_name = "zPrime"+ch+"_mZ"+str(mZ)+"_mX"+str(m)+"_ct"
                    if ct>1:
                        card_name += str(int(ct))
                        ctn = str(int(ct))
                    else:
                        card_name += "0p1"
                        ctn = "0p1"
                    ctn+=ch
                    print card_name
                elif "Stealth" in sign[0]:
                    if ct<1:
                        card_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(ct).replace(".","p")+"_"+ch
                    else:
                        card_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(int(ct))+"_"+ch
                    print card_name

                elif "HeavyHiggs" in sign[0]:
                    #QUA
                    card_name = "HeavyHiggs"+ch+"_mH"+str(mZ)+"_mX"+str(m)+"_ct"
                    if ct>1:
                        card_name += str(int(ct))
                        ctn = str(int(ct))
                    else:
                        card_name += "0p1"
                        ctn = "0p1"
                    ctn+=ch
                    print card_name
                else:
                    card_name = "SUSY_mh"+str(m)+"_ctau"+str(ct)
                    ctn = ct

                if not os.path.isfile(COND_DIR+"out_"+card_name+".txt"):
                    continue

                print "I read the condor output from here: ", COND_DIR
                os.system("cat "+COND_DIR+"out_"+card_name+".txt \n")
                os.system("cat "+COND_DIR+"out_"+card_name+".txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + card_name +".txt  \n")
                #os.system("cat "+COND_DIR+"out_m"+str(m)+"_ctau"+str(ct)+".txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + card_name +".txt\n")
                #os.system("cat "+ RESULTS + "/" + card_name + ".txt  \n")
                print "\n"
                print "Limits written in ", RESULTS + "/" + card_name + ".txt"
                ##print "Significance written in ", RESULTS + "/Significance_" + card_name + ".txt"
                print "*********************************************"


def plot_limits_vs_ctau_split(out_dir,plot_dir,sign,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False,BR_SCAN_H=100,blind=True,toys=False):

    #Fix this_lumi
    if combination==False:
        if ERA=="2016":
            if dataset_label == "_G-H":
                this_lumi  = lumi[ "HighMET" ]["G"]+lumi[ "HighMET" ]["H"]
            elif dataset_label == "_B-F":
                this_lumi  = lumi[ "HighMET" ]["B"]+lumi[ "HighMET" ]["C"]+lumi[ "HighMET" ]["D"]+lumi[ "HighMET" ]["E"]+lumi[ "HighMET" ]["F"]
        else:
            this_lumi  = lumi[ "HighMET" ]["tot"]
    else:
        if comb_fold_label=="2016_BFGH_2017_2018":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_50":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_75_average":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_GH_2017_2018":
            this_lumi = 111941.400399
        else:
            print "Invalid combination folder, can't calculate lumi, aborting"
            exit()

    print "LUMI: ", this_lumi

    br_scan_fold = ""
    if BR_SCAN_H==100:
        br_scan_fold = "BR_h100_z0"
    if BR_SCAN_H==75:
        br_scan_fold = "BR_h75_z25"
    if BR_SCAN_H==50:
        br_scan_fold = "BR_h50_z50"
    if BR_SCAN_H==25:
        br_scan_fold = "BR_h25_z75"
    if BR_SCAN_H==0:
        br_scan_fold = "BR_h0_z100"


    #zPrime corrections


    if "splitSUSY" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "splitSUSY"
    if "zPrime" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "zPrime"
    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"
    if "HeavyHiggs" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "HeavyHiggs"

    DATACARDDIR = out_dir+CHAN+"/"
    DATACARDDIR += br_scan_fold+"/"

    if combination:
        DATACARDDIR += comb_fold_label+"/"
        plot_dir += br_scan_fold+"/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plot_dir += comb_fold_label+"/"
    else:
        plot_dir += br_scan_fold+"/"
    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    print "Plotting these limits: ", DATACARDDIR

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    #Dictionary with various predictions
    '''
    mass = []
    theory = defaultdict(dict)
    mean_val = defaultdict(dict)
    sigma_1_up = defaultdict(dict)
    sigma_1_down = defaultdict(dict)
    sigma_2_up = defaultdict(dict)
    sigma_2_down = defaultdict(dict)
    '''

    masses = []

    for i,s in enumerate(sign):
        #print "theory x-sec (pb): ", sample[ samples[s]['files'][0] ]['xsec']
        masses.append(samples[s]['mass'])

    masses = np.array(masses)
    masses = np.unique(np.array(masses))

    loop_ct = ctaus
    if "splitSUSY" in sign[0]:
        loop_ct = ctaus_split
    if "zPrime" in sign[0]:
        loop_ct = ctaus_zPrime 
    if "Stealth" in sign[0]:
        loop_ct = ctaus_stealth 
    if "HeavyHiggs" in sign[0]:
        loop_ct = ctaus_HeavyHiggs 
    loop_ct = np.unique(loop_ct)

    for m in masses:
        #Initialize here dictionaries
        #Dictionary with various predictions
        theory = defaultdict(dict)
        obs = defaultdict(dict)
        mean_val = defaultdict(dict)
        sigma_1_up = defaultdict(dict)
        sigma_1_down = defaultdict(dict)
        sigma_2_up = defaultdict(dict)
        sigma_2_down = defaultdict(dict)


        #for ct in np.sort(np.append(ctaus,np.array([samples[s]['ctau']]))):
        for ct in np.sort(loop_ct):
            print ct
            #QUA
            if "splitSUSY" in sign[0]:
                sign_name = "splitSUSY_M2400_"+str(m)+"_ctau"+str(int(ct))+"p0"
            elif "zPrime" in sign[0]:
                #QUA
                sign_name = "zPrime"+ch+"_mZ"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    sign_name += str(int(ct))
                else:
                    sign_name += "0p1"
            elif "Stealth" in sign[0]:
                if ct<1:
                    sign_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(ct).replace(".","p")+"_"+ch
                else:
                    sign_name = "Stealth"+ch+"_2t6j_mstop"+str(mZ)+"_ms"+str(m)+"_ctau"+str(int(ct))+"_"+ch
            elif "HeavyHiggs" in sign[0]:
                #QUA
                sign_name = "HeavyHiggs"+ch+"_mH"+str(mZ)+"_mX"+str(m)+"_ct"
                if ct>1:
                    sign_name += str(int(ct))
                else:
                    sign_name += "0p1"
            #sign_name = "splitSUSY_M2400_"+str(m)+"_ctau"+str(int(ct))+"p0"
            card_name = RESULTS + "/" + sign_name+".txt"#s.replace('_ctau'+str(samples[s]['ctau']),'_ctau'+str(ct)) + ".txt"
            print "read ",card_name
            if not os.path.isfile(card_name):
                continue
            card = open( card_name, 'r')
            val = card.read().splitlines()
            if len(val) == 0:
                continue
            if len(val) != 6 and not blind:
                continue
                                
            if blind:
                min_dist_sigma_mean = min( (float(val[3])-float(val[2]))/float(val[2]), (float(val[2])-float(val[1]))/float(val[2]))
            else:
                min_dist_sigma_mean = min( (float(val[3+1])-float(val[2+1]))/float(val[2+1]), (float(val[2+1])-float(val[1+1]))/float(val[2+1]))
            if min_dist_sigma_mean<0.1  and not toys:
                continue

            if m<210 and ("splitSUSY" not in sign[0] and "zPrime" not in sign[0] and "Stealth" not in sign[0] and "HeavyHiggs" not in sign[0]):
                    if blind:
                        print "Adjust signal multiplication (enhance signal to avoid combine instabilities)"
                        sigma_2_down[m][ct] = float(val[0])*10.
                        sigma_1_down[m][ct] = float(val[1])*10.
                        mean_val[m][ct]     = float(val[2])*10.
                        sigma_1_up[m][ct]   = float(val[3])*10.
                        sigma_2_up[m][ct]   = float(val[4])*10.
                    else:
                        obs[m][ct]          = float(val[0])*10.
                        sigma_2_down[m][ct] = float(val[1])*10.
                        sigma_1_down[m][ct] = float(val[2])*10.
                        mean_val[m][ct]     = float(val[3])*10.
                        sigma_1_up[m][ct]   = float(val[4])*10.
                        sigma_2_up[m][ct]   = float(val[5])*10.
            else:
                if blind:
                    sigma_2_down[m][ct] = float(val[0])
                    sigma_1_down[m][ct] = float(val[1])
                    mean_val[m][ct]     = float(val[2])
                    sigma_1_up[m][ct]   = float(val[3])
                    sigma_2_up[m][ct]   = float(val[4])
                else:
                    obs[m][ct]          = float(val[0])
                    sigma_2_down[m][ct] = float(val[1])
                    sigma_1_down[m][ct] = float(val[2])
                    mean_val[m][ct]     = float(val[3])
                    sigma_1_up[m][ct]   = float(val[4])
                    sigma_2_up[m][ct]   = float(val[5])
            if "splitSUSY" not in sign[0] and "zPrime" not in sign[0] and "Stealth" not in sign[0] and "HeavyHiggs" not in sign[0]:
                theory[m][ct]       = 1000.*sample[ samples[s]['files'][0] ]['xsec']

        Obs0s = TGraph()
        Exp0s = TGraph()
        Exp1s = TGraphAsymmErrors()
        Exp2s = TGraphAsymmErrors()
        Theory = TGraph()

        n=0
        #for ct in np.sort(np.append(ctaus,np.array([ctaupoint]))):
        for ct in np.sort(loop_ct):
            if ct in mean_val[m].keys():
                print m, ct/1., mean_val[m][ct]
                if not blind:
                    Obs0s.SetPoint(n, ct/1., obs[m][ct])
                Exp0s.SetPoint(n, ct/1., mean_val[m][ct])
                Exp1s.SetPoint(n, ct/1., mean_val[m][ct])
                Exp1s.SetPointError(n, 0., 0., mean_val[m][ct]-sigma_1_down[m][ct], sigma_1_up[m][ct]-mean_val[m][ct])
                Exp2s.SetPoint(n, ct/1., mean_val[m][ct])
                Exp2s.SetPointError(n, 0., 0., mean_val[m][ct]-sigma_2_down[m][ct], sigma_2_up[m][ct]-mean_val[m][ct])
                if "splitSUSY" not in sign[0] and "zPrime" not in sign[0] and "Stealth" not in sign[0] and "HeavyHiggs" not in sign[0]:
                    Theory.SetPoint(n, ct/1., theory[m][ct])
                n+=1

        particle = "split"
        if "zPrime" in sign[0] or "HeavyHiggs" in sign[0]:
            particle = "X"
        if "Stealth" in sign[0]:
            particle = "S"
        Exp2s.SetLineWidth(2)
        Exp2s.SetLineStyle(1)
        Exp0s.SetLineStyle(2)
        Exp0s.SetLineWidth(1)#(3)
        Obs0s.SetLineStyle(1)
        Obs0s.SetLineWidth(1)#(3)
        Exp1s.SetFillColor(417)
        Exp1s.SetLineColor(417)
        Exp2s.SetFillColor(800)
        Exp2s.SetLineColor(800)
        Exp2s.GetXaxis().SetTitle("c #tau_{"+particle+"} (mm)")
        Exp2s.GetXaxis().SetNoExponent(True)
        Exp2s.GetXaxis().SetMoreLogLabels(True)
        Exp2s.GetXaxis().SetTitleSize(0.048)
        Exp2s.GetYaxis().SetTitleSize(0.048)
        Exp2s.GetYaxis().SetTitleOffset(0.8)
        Theory.SetLineWidth(2)
        Theory.SetLineColor(2)
        Theory.SetLineStyle(2)
        if contamination:
            Exp0s.GetYaxis().SetTitleOffset(1.5)
            print Exp0s.GetYaxis().GetLabelOffset()
        Exp2s.GetXaxis().SetTitleOffset(0.9)    
        top = 0.9
        nitems = 4
        leg = TLegend(0.45, top-nitems*0.3/5., 0.8, top)
        leg.SetBorderSize(0)
        leg.SetHeader("95% CL limits, M_{Z'}="+str(mZ)+" GeV, m_{X}="+str(m)+" GeV")
        leg.SetTextSize(0.04)

        c1 = TCanvas("c1", "Exclusion Limits", 800, 600)
        c1.cd()
        #c1.SetGridx()
        #c1.SetGridy()
        c1.GetPad(0).SetTopMargin(0.06)
        c1.GetPad(0).SetRightMargin(0.05)
        c1.GetPad(0).SetTicks(1, 1)
        c1.GetPad(0).SetLogy()
        c1.GetPad(0).SetLogx()
        if contamination:
            c1.GetPad(0).SetLeftMargin(0.13)
        if contamination:
            leg.AddEntry(Exp0s,  "#mu ratio", "l")
        else:
            ##leg.AddEntry(Theory, "Theory", "l")
            #leg.AddEntry(Exp0s,  "Expected", "l")
            #leg.AddEntry(Exp1s, "#pm 1 std. deviation", "f")
            #leg.AddEntry(Exp2s, "#pm 2 std. deviations", "f")
            leg.AddEntry(Exp0s,  "Median expected", "l")
            leg.AddEntry(Exp1s, "68% expected", "f")
            leg.AddEntry(Exp2s, "95% expected", "f")
            if not blind:
                leg.AddEntry(Obs0s,  "Observed", "l")

        Exp2s.GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")

        if signalMultFactor == 0.001:
            #Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} Z #tilde{G} #rightarrow b #bar{b} #tilde{G} q #bar{q} #tilde{G}) (fb)")
            #after review:
            Exp2s.GetYaxis().SetTitle("#sigma(Z') (fb)")
        else:
            Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
        Exp2s.GetXaxis().SetTitle("c #tau_{"+particle+"} (mm)")
        Exp1s.GetXaxis().SetTitle("c #tau_{"+particle+"} (mm)")
        Exp0s.GetXaxis().SetTitle("c #tau_{"+particle+"} (mm)")
        axis = Exp2s.GetXaxis()
        axis.SetLimits(1.,100001.)
        Exp2s.SetMinimum(0.009)
        Exp2s.SetMaximum(100001)
        if contamination:
            Exp0s.SetLineColor(2)
            Exp0s.SetLineWidth(2)
            Exp0s.SetMarkerColor(2)
            Exp0s.SetMarkerStyle(24)
            Exp0s.SetMinimum(0.98)
            Exp0s.SetMaximum(1.02)
            Exp0s.Draw("APL")
            Exp0s.GetYaxis().SetTitle("#frac{Excluded #mu with signal contamination}{Excluded #mu w/o signal contamination}")
            c1.SetLogy(0)
        else:
            Exp2s.Draw("A3")
            Exp1s.Draw("SAME, 3")
            Exp0s.Draw("SAME, L")
            #Theory.Draw("SAME, L")
            if not blind:
                Obs0s.Draw("SAME, L")
            leg.Draw()
            #Draw above again
            Exp2s.Draw("SAME,3")
            Exp1s.Draw("SAME, 3")
            Exp0s.Draw("SAME, L")
            Theory.Draw("SAME, L")
            if not blind:
                Obs0s.Draw("SAME, L")

            
        if PRELIMINARY:
            drawCMS(samples, this_lumi, "Preliminary",left_marg_CMS=0.3,onTop=True)
        else:
            drawCMS(samples, this_lumi, "",left_marg_CMS=0.32)

        drawRegion(CHAN,top=0.7)
        #drawBR(BR_SCAN_H)
        #drawAnalysis("LL"+CHAN)
        #drawTagVar(TAGVAR)
        OUTSTRING = ""
        if "zPrime" in sign[0]:
            OUTSTRING = plot_dir+"/Exclusion_zPrime"+ch+"_vs_ctau_mZ_"+str(mZ)+"_mX_"+str(m)#+TAGVAR+comb_fold_label+"_"+CHAN
        elif "HeavyHiggs" in sign[0]:
            OUTSTRING = plot_dir+"/Exclusion_HeavyHiggs"+ch+"_vs_ctau_mH_"+str(mZ)+"_mX_"+str(m)#+TAGVAR+comb_fold_label+"_"+CHAN
        elif "Stealth" in sign[0]:
            OUTSTRING = plot_dir+"/Exclusion_Stealth"+ch+"_vs_ctau_mstop_"+str(mZ)+"_ms_"+str(m)#+TAGVAR+comb_fold_label+"_"+CHAN
        newFile = TFile(OUTSTRING+".root", "RECREATE")#+"_"+TAGVAR+comb_fold_label+ ".root", "RECREATE")
        newFile.cd()
        if not blind:
            Obs0s.Write("m"+str(m)+"_obs")
        Exp0s.Write("m"+str(m)+"_exp")
        Exp1s.Write("m"+str(m)+"_1sigma")
        Exp2s.Write("m"+str(m)+"_2sigma")
        c1.Write()
        newFile.Close()
        print "Info: written file: ", OUTSTRING+".root"#"_"+TAGVAR+comb_fold_label+".root"

        c1.Print(OUTSTRING+".png")
        c1.Print(OUTSTRING+".pdf")
        c1.Close()



def plot_limits_vs_mass_split(out_dir,plot_dir,sign,ms_inp,ctau_inp,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False,BR_SCAN_H=100,blind=True,toys=False):

    #Fix this_lumi
    if combination==False:
        if ERA=="2016":
            if dataset_label == "_G-H":
                this_lumi  = lumi[ "HighMET" ]["G"]+lumi[ "HighMET" ]["H"]#["tot"]
            elif dataset_label == "_B-F":
                this_lumi  = lumi[ "HighMET" ]["B"]+lumi[ "HighMET" ]["C"]+lumi[ "HighMET" ]["D"]+lumi[ "HighMET" ]["E"]+lumi[ "HighMET" ]["F"]#["tot"]
        else:
            this_lumi  = lumi[ "HighMET" ]["tot"]
    else:
        if comb_fold_label=="2016_BFGH_2017_2018":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_50":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_75_average":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_GH_2017_2018":
            this_lumi = 111941.400399
        else:
            print "Invalid combination folder, can't calculate lumi, aborting"
            exit()

    print "LUMI: ", this_lumi

    br_scan_fold = ""
    if BR_SCAN_H==100:
        br_scan_fold = "BR_h100_z0"
    if BR_SCAN_H==75:
        br_scan_fold = "BR_h75_z25"
    if BR_SCAN_H==50:
        br_scan_fold = "BR_h50_z50"
    if BR_SCAN_H==25:
        br_scan_fold = "BR_h25_z75"
    if BR_SCAN_H==0:
        br_scan_fold = "BR_h0_z100"

    if toys:
        br_scan_fold += "_toys"

    if "splitSUSY" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "splitSUSY"
    if "zPrime" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "zPrime"
    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"
    if "HeavyHiggs" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "HeavyHiggs"


    DATACARDDIR = out_dir+CHAN+"/"
    DATACARDDIR += br_scan_fold+"/"

    if combination:
        DATACARDDIR += comb_fold_label+"/"
        plot_dir += br_scan_fold+"/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plot_dir += comb_fold_label+"/"
    else:
        plot_dir += comb_fold_label+"/"

    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    print "Plotting these limits: ", DATACARDDIR
    print "And saving them here: ", plot_dir


    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2


    #Dictionary with various predictions
    stop_masses = []
    s_masses = []
    ctau = []
    theory = defaultdict(dict)
    obs = defaultdict(dict)
    mean_val = defaultdict(dict)
    sigma_1_up = defaultdict(dict)
    sigma_1_down = defaultdict(dict)
    sigma_2_up = defaultdict(dict)
    sigma_2_down = defaultdict(dict)


    #First: get all the stop masses
    for i,s in enumerate(sign):
        m = samples[s]['stop_mass']
        if m not in stop_masses:
            stop_masses.append(m)

    stop_masses = np.unique(np.array(stop_masses))
    print stop_masses
    #ms_inp
    #ctau_inp

    #Second: get all the values
    ms_string = ""
    for m in stop_masses:
        if ms_inp==100:
            ms=100
            ms_string = "100"
        else:
            ms = m-225
            ms_string = "min_225"
        base_string = "Stealth%s_2t6j_mstop%s_ms%s_ctau%s_%s"
        if ctau_inp<1:
            name = base_string % (ch, str(m), str(ms),str(ctau_inp).replace('.','p'),ch)
        else:
            name = base_string % (ch, str(m), str(ms),str(int(ctau_inp)),ch)
        card_name = RESULTS + "/" + name + ".txt"
        print card_name
        if not os.path.isfile(card_name):
            continue
        card = open( card_name, 'r')
        val = card.read().splitlines()

        if len(val) == 0:
            continue
        if len(val) != 6 and not blind:
            continue

        if blind:
            min_dist_sigma_mean = min( (float(val[3])-float(val[2]))/float(val[2]), (float(val[2])-float(val[1]))/float(val[2]))
        else:
            if float(val[3])==0:
                print "Warning, median zero for ", card_name, "!!!!!"
                print "Skipping this point"
                continue
            min_dist_sigma_mean = min( (float(val[4])-float(val[3]))/float(val[3]), (float(val[3])-float(val[2]))/float(val[3]))
            if min_dist_sigma_mean<0.1 and not toys:
                continue

        if blind:
            sigma_2_down[m] = float(val[0])
            sigma_1_down[m] = float(val[1])
            mean_val[m]     = float(val[2])
            sigma_1_up[m]   = float(val[3])
            sigma_2_up[m]   = float(val[4])
        else:
            obs[m]          = float(val[0])*signalMultFactor
            sigma_2_down[m] = float(val[1])*signalMultFactor
            sigma_1_down[m] = float(val[2])*signalMultFactor
            mean_val[m]     = float(val[3])*signalMultFactor
            sigma_1_up[m]   = float(val[4])*signalMultFactor
            sigma_2_up[m]   = float(val[5])*signalMultFactor

    print "filled:"
    for ok in obs.keys():
        print ok, obs[ok]
    #CHIPS
    #exit()

    Obs0s = TGraph()
    Exp0s = TGraph()
    Exp1s = TGraphAsymmErrors()
    Exp2s = TGraphAsymmErrors()
    Theory = TGraph()

    mass = np.sort(np.array(obs.keys()))
    print mass

    n=0
    for m in mass:
        if not blind:
            Obs0s.SetPoint(n, m, obs[m])
        Exp0s.SetPoint(n, m, mean_val[m])
        Exp1s.SetPoint(n, m, mean_val[m])
        Exp1s.SetPointError(n, 0., 0., mean_val[m]-sigma_1_down[m], sigma_1_up[m]-mean_val[m])
        Exp2s.SetPoint(n, m, mean_val[m])
        Exp2s.SetPointError(n, 0., 0., mean_val[m]-sigma_2_down[m], sigma_2_up[m]-mean_val[m])

        print ct/1000., m, mean_val[m]
        n+=1

    Exp2s.SetLineWidth(2)
    Exp2s.SetLineStyle(1)
    Exp0s.SetLineStyle(2)
    Exp0s.SetLineWidth(1)#(3)
    Obs0s.SetLineStyle(1)
    Obs0s.SetLineWidth(1)#(3)
    Exp1s.SetFillColor(417)
    Exp1s.SetLineColor(417)
    Exp2s.SetFillColor(800)
    Exp2s.SetLineColor(800)
    Exp2s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp2s.GetXaxis().SetNoExponent(True)
    Exp2s.GetXaxis().SetMoreLogLabels(True)
    Exp2s.GetXaxis().SetTitleSize(0.048)
    Exp2s.GetYaxis().SetTitleSize(0.048)
    Exp2s.GetXaxis().SetTitleOffset(0.9)    
    Exp2s.GetYaxis().SetTitleOffset(0.85)
    #Editing style
    nitems = 5
    #CWR: move legend lower
    #top = 0.825
    #leg = TLegend(0.55, 0.525, 0.9-0.05, top)
    top = 0.825
    leg = TLegend(0.55, 0.525-0.025, 0.9-0.05, top-0.025)

    leg.SetBorderSize(0)
    #Original
    #leg.SetHeader("95% CL limits, c#tau_{"+particle+"}="+str(int(ct/1000) if ct>500 else 0.5)+" m")
    #Editing style
    leg.SetHeader("95% CL limits, c#tau_{"+particle+"} = "+str(int(ct/1000) if ct>500 else 0.5)+" m")
    leg.SetTextSize(0.04)

    #Original
    #c1 = TCanvas("c1", "Exclusion Limits", 800, 600)
    #Editing style
    c1 = TCanvas("c1", "Exclusion Limits", 900, 675)
    c1.cd()
    #FR comment
    c1.GetPad(0).SetBottomMargin(0.12)
    c1.GetPad(0).SetTopMargin(0.08)
    c1.GetPad(0).SetRightMargin(0.05)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetLogy()
    #c1.GetPad(0).SetLogx()
    #FR
    leg.AddEntry(Exp0s,  "Median expected", "l")
    leg.AddEntry(Exp1s, "68% expected", "f")
    leg.AddEntry(Exp2s, "95% expected", "f")
    if not blind:
        leg.AddEntry(Obs0s, "Observed", "l")
        
    #Exp2s.GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
    if signalMultFactor == 0.001:
        #after review:
        Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") (pb)")

    else:
        Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
    Exp2s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp1s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp0s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    #CWR comment: make y-range smaller
    Exp2s.SetMinimum(0.09*signalMultFactor)
    Exp2s.SetMaximum(10001*signalMultFactor)
    #FR comment: larger labels --> need also different borders
    Exp2s.GetXaxis().SetTitleSize(0.055)    
    Exp2s.GetYaxis().SetTitleSize(0.055)    
    #FR comment: larger label size
    Exp2s.GetXaxis().SetLabelSize(0.045)#original: 0.035    
    Exp2s.GetYaxis().SetLabelSize(0.045) 

    Exp2s.Draw("A3")
    Exp1s.Draw("SAME, 3")
    Exp0s.Draw("SAME, L")
    if not blind:
        Obs0s.Draw("SAME, L")
    leg.Draw()

    if PRELIMINARY:
        #Original
        #drawCMS(samples, this_lumi, "Preliminary",left_marg_CMS=0.3,onTop=True)
        #Editing style
        drawCMS_simple(this_lumi, "Preliminary", ERA="", onTop=True, left_marg_CMS=0.2,top_marg_lumi = 0.98)
    else:
        #Original
        #drawCMS(samples, this_lumi, "",left_marg_CMS=0.32)
        #Editing style
        #drawCMS_simple(this_lumi, "", ERA="", onTop=True, left_marg_CMS=0.2,top_marg_lumi = 0.98)
        #FR comment
        drawCMS_simple(this_lumi, "", ERA="", onTop=True, left_marg_CMS=0.27,top_marg_cms = 0.87,top_marg_lumi = 0.98)

    OUTSTRING = plot_dir+"/Exclusion_Stealth"+ch+"_vs_stop_mass_ms_"+ms_string+"_ctau_"+str(ctn)#+TAGVAR+comb_fold_label+"_"+CHAN
    newFile = TFile(OUTSTRING+".root", "RECREATE")#+"_"+TAGVAR+comb_fold_label+ ".root", "RECREATE")
    newFile.cd()
    if not blind:
        Obs0s.Write("ms_"+str(ms_string)+"_obs")
    Exp0s.Write("ms_"+str(ms_string)+"_exp")
    Exp1s.Write("ms_"+str(ms_string)+"_1sigma")
    Exp2s.Write("ms_"+str(ms_string)+"_2sigma")
    c1.Write()
    newFile.Close()
    print "Info: written file: ", OUTSTRING+".root"#"_"+TAGVAR+comb_fold_label+".root"
    
    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")

    c1.Close()


def plot_limits_2D_split_pre(out_dir,plot_dir,sign,ms_inp,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False,BR_SCAN_H=100,blind=True,toys=False):

    gStyle.SetPalette(100)

    #Fix this_lumi
    if combination==False:
        if ERA=="2016":
            if dataset_label == "_G-H":
                this_lumi  = lumi[ "HighMET" ]["G"]+lumi[ "HighMET" ]["H"]
            elif dataset_label == "_B-F":
                this_lumi  = lumi[ "HighMET" ]["B"]+lumi[ "HighMET" ]["C"]+lumi[ "HighMET" ]["D"]+lumi[ "HighMET" ]["E"]+lumi[ "HighMET" ]["F"]
        else:
            this_lumi  = lumi[ "HighMET" ]["tot"]
    else:
        if comb_fold_label=="2016_BFGH_2017_2018":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_50":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_75_average":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_GH_2017_2018":
            this_lumi = 111941.400399
        else:
            print "Invalid combination folder, can't calculate lumi, aborting"
            exit()

    print "LUMI: ", this_lumi

    br_scan_fold = ""

    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"

    DATACARDDIR = out_dir+CHAN+"/"
    DATACARDDIR += br_scan_fold+"/"

    if combination:
        DATACARDDIR += comb_fold_label+"/"
        plot_dir += br_scan_fold+"/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plot_dir += comb_fold_label+"/"
    else:
        plot_dir += br_scan_fold+"/"
    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    print "Plotting these limits: ", DATACARDDIR

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    masses = []

    for i,s in enumerate(sign):
        #print "theory x-sec (pb): ", sample[ samples[s]['files'][0] ]['xsec']
        masses.append(samples[s]['stop_mass'])

    masses = np.array(masses)
    masses = np.unique(np.array(masses))

    loop_ct = ctaus
    if "Stealth" in sign[0]:
        loop_ct = ctaus_stealth 
    loop_ct = np.unique(loop_ct)

    #Initialize here dictionaries
    #Dictionary with various predictions
    theory = defaultdict(dict)
    obs = defaultdict(dict)
    mean_val = defaultdict(dict)
    sigma_1_up = defaultdict(dict)
    sigma_1_down = defaultdict(dict)
    sigma_2_up = defaultdict(dict)
    sigma_2_down = defaultdict(dict)

    for sn in sign:
        print sn
        card_name = RESULTS + "/" + sn +".txt"#s.replace('_ctau'+str(samples[s]['ctau']),'_ctau'+str(ct)) + ".txt"
        print "read ",card_name
        if not os.path.isfile(card_name):
            continue
        card = open( card_name, 'r')
        val = card.read().splitlines()
        if len(val) == 0:
            continue
        if len(val) != 6 and not blind:
            continue
                                
        if blind:
            min_dist_sigma_mean = min( (float(val[3])-float(val[2]))/float(val[2]), (float(val[2])-float(val[1]))/float(val[2]))
        else:
            min_dist_sigma_mean = min( (float(val[3+1])-float(val[2+1]))/float(val[2+1]), (float(val[2+1])-float(val[1+1]))/float(val[2+1]))
        if min_dist_sigma_mean<0.1  and not toys:
            continue

        m = samples[sn]['stop_mass']
        ct = samples[sn]['ctau']

        obs[m][ct]          = float(val[0])#/1000.
        sigma_2_down[m][ct] = float(val[1])#/1000.
        sigma_1_down[m][ct] = float(val[2])#/1000.
        mean_val[m][ct]     = float(val[3])#/1000.
        sigma_1_up[m][ct]   = float(val[4])#/1000.
        sigma_2_up[m][ct]   = float(val[5])#/1000.

    loop_ctau_minimal = []
    for m in masses:
        for ct in obs[m].keys():
            print m, ct, obs[m][ct]
            loop_ctau_minimal.append(ct)

    loop_ctau_minimal = np.unique( np.sort( np.array(loop_ctau_minimal) )  )
    
    x = array('d',[])
    y = array('d',[])
    z = array('d',[])
    xt = array('d',[])
    yt = array('d',[])
    zt = array('d',[])
       
         
    n_bins_x = 100
    min_bin_x = np.log10(250)
    max_bin_x = np.log10(1550)

    n_bins_y = 60
    #in meters
    min_bin_y = np.log10(loop_ctau_minimal[0]/1000.)-0.5
    max_bin_y = np.log10(loop_ctau_minimal[-1]/1000.)+0.5
    #Let's try go higher in ctau
    max_bin_y = np.log10(100000./1000.)+0.5

    #Add theory border points
    #need to fill the entire plane up to the borders with theory values to avoid extrapolations
    min_bin_y_lin = 1000*(10**(min_bin_y) )
    max_bin_y_lin = 1000*(10**(max_bin_y) ) - 10 #avoid rejection because of overflow

    loop_ct_th = [min_bin_y_lin]
    for ct in loop_ct:
        loop_ct_th.append(ct)
    loop_ct_th.append(max_bin_y_lin)
    loop_ct_th = np.array(loop_ct_th)


    h = TH2D("h","h", n_bins_x,min_bin_x,max_bin_x,n_bins_y,min_bin_y,max_bin_y)
    ht = TH2D("ht","ht", n_bins_x,min_bin_x,max_bin_x,n_bins_y,min_bin_y,max_bin_y)

    #xsec in pb
    xsec = { 300:10.00*1000., 500:0.609*1000., 700:0.0783*1000., 900:0.0145*1000., 1100:0.00335*1000., 1300:0.000887*1000., 1500:0.000257*1000.}

    max_lim = 0
    th_limits_array = []
    for m in masses:
        valid_ctau = []
        all_ctau = []
        list_limits_each_mass = []
        list_limits_each_mass_th = []
        th_limits_array.append(xsec[m])
        for ct in loop_ctau_minimal:
            if ct in obs[m].keys():
                if obs[m][ct]==0:
                    print "no limits for: ", m, ct
                if obs[m][ct]!=0:
                    #in meters
                    h.Fill(np.log10(m),np.log10(ct/1000.),obs[m][ct])
                    valid_ctau.append(np.log10(ct/1000.))
                    #h.Fill(np.log10(m),np.log10(ct),np.log10(obs[m][ct]))
                    #valid_ctau.append(np.log10(ct))
                    list_limits_each_mass.append(obs[m][ct])
                    max_lim = max(max_lim,obs[m][ct])

        for ct_th in loop_ct_th:
            all_ctau.append(np.log10(ct_th/1000.))
            ht.Fill(np.log10(m),np.log10(ct_th/1000.),xsec[m])
            list_limits_each_mass_th.append(xsec[m])


        x += array('d', [math.log10(float(m))]*len(valid_ctau))
        y += array('d', valid_ctau)
        z += array('d', np.log10(list_limits_each_mass))

        xt += array('d', [math.log10(float(m))]*len(all_ctau))
        yt += array('d', all_ctau)
        zt += array('d', np.log10(list_limits_each_mass_th))


    c1 = TCanvas("c1", "Exclusion Limits", 900, 675)
    c1.cd()
    c1.GetPad(0).SetBottomMargin(0.12)
    c1.GetPad(0).SetTopMargin(0.08)
    c1.GetPad(0).SetRightMargin(0.15)#12)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetLogz()
    c1.GetPad(0).SetLogy()
    #c1.GetPad(0).SetLogx()

    h = interpolate2D(x,y,z, h, 0.02,0.1, inter = 'rbf', norm = 'euclidean')
    h = log_scale_conversion(h)

    ht = interpolate2D(xt,yt,zt, ht, 0.02,0.1, inter = 'rbf', norm = 'euclidean')
    ht = log_scale_conversion(ht)


    h.SetMinimum(0.1)
    #h.SetMaximum(3e6)
    h.SetMaximum(1000)
    #h.SetMaximum(max_lim)
    h.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    h.GetYaxis().SetTitle("c #tau_{"+particle+"} (m)")
    h.GetYaxis().SetTitleOffset(1.6)
    h.GetZaxis().SetTitle("95% CL limits on #sigma("+particle+particle+") (fb)")
    h.Draw("COLZ")

    hr = ht.Clone("hr")
    hr.Divide(h)
    contours = array('d', [1.0])
    hr.SetContour(1,contours)
    #contours = array('d', [1.0,10.,100.,1000.])
    #hr.SetContour(4,contours)
    hr.SetLineColor(1)#new
    hr.SetLineWidth(4)#new
    hr.SetLineStyle(2)#new
    hr.SetFillColorAlpha(1,0.15)
    hr.SetFillStyle(3444)
    hr.Draw("CONT LIST same")#CONT creates a filled area
    #hr.Draw("CONT 3 same")#CONT creates a filled area
    
    c1.Update()
    conts = gROOT.GetListOfSpecials().FindObject("contours")
    print "these many conts: ", conts.GetSize()
    conts_gr = conts.At(0)
    print "these many conts_gr: ", conts_gr.GetSize()
    #print conts_gr.Print()
    curv1 = conts_gr.First()
    print "first graph in conts_gr: "
    print curv1.Print()
    curv2 = conts_gr.After(curv1)
    print "second graph in conts_gr: "
    print curv2.Print()

    new_x_val1 = []
    new_y_val1 = []
    new_x_val2 = []
    new_y_val2 = []

    for ngr in range(curv1.GetN()):
        print "curv1"
        print ngr, curv1.GetPointX(ngr), 10**(curv1.GetPointY(ngr))
        new_x_val1.append(10**curv1.GetPointY(ngr))
        new_y_val1.append(curv1.GetPointX(ngr))
    for ngr in range(curv2.GetN()):
        print "curv2"
        print ngr, curv2.GetPointX(ngr), 10**(curv2.GetPointY(ngr))
        new_x_val2.append(10**curv2.GetPointY(ngr))
        new_y_val2.append(curv2.GetPointX(ngr))
    #print hr.Print()

    print "This works, now try to rotate the thing!"

    ##hr.Draw("CONT 3 same")#CONT creates a filled area
    ##c1.Update()

    h.GetXaxis().SetNoExponent(True)
    h.GetXaxis().SetMoreLogLabels(True)
    h.GetXaxis().SetTitleSize(0.048)
    h.GetYaxis().SetTitleSize(0.048)
    h.GetZaxis().SetTitleSize(0.048)
    h.GetYaxis().SetTitleOffset(1.)
    h.GetXaxis().SetTitleOffset(0.8)

    #FR comment: larger labels --> need also different borders
    h.GetXaxis().SetTitleSize(0.055)
    h.GetYaxis().SetTitleSize(0.055)
    h.GetZaxis().SetTitleSize(0.055)
    drawCMS_simple(this_lumi, "", ERA="", onTop=True, left_marg_CMS=0.175,top_marg_lumi = 0.98)

    if ms_inp==100:
        ms_string = "100"
    else:
        ms_string = "min_225"
            

    OUTSTRING = plot_dir+"/Exclusion_2D_ms_"+str(ms_string)+ch
    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")
    c1.Close()


def plot_limits_2D_split(out_dir,plot_dir,sign,ms_inp,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False,BR_SCAN_H=100,blind=True,toys=False):

    gStyle.SetPalette(100)

    #Fix this_lumi
    if combination==False:
        if ERA=="2016":
            if dataset_label == "_G-H":
                this_lumi  = lumi[ "HighMET" ]["G"]+lumi[ "HighMET" ]["H"]
            elif dataset_label == "_B-F":
                this_lumi  = lumi[ "HighMET" ]["B"]+lumi[ "HighMET" ]["C"]+lumi[ "HighMET" ]["D"]+lumi[ "HighMET" ]["E"]+lumi[ "HighMET" ]["F"]
        else:
            this_lumi  = lumi[ "HighMET" ]["tot"]
    else:
        if comb_fold_label=="2016_BFGH_2017_2018":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_50":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_BFGH_2017_2018_stat_unc_75_average":
            this_lumi = 137478.722953
        elif comb_fold_label=="2016_GH_2017_2018":
            this_lumi = 111941.400399
        else:
            print "Invalid combination folder, can't calculate lumi, aborting"
            exit()

    print "LUMI: ", this_lumi

    br_scan_fold = ""

    if "Stealth" in sign[0]:
        print "ignore br scan"
        br_scan_fold = "Stealth"

    DATACARDDIR = out_dir+CHAN+"/"
    DATACARDDIR += br_scan_fold+"/"

    if combination:
        DATACARDDIR += comb_fold_label+"/"
        plot_dir += br_scan_fold+"/"
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plot_dir += comb_fold_label+"/"
    else:
        plot_dir += br_scan_fold+"/"
    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    print "Plotting these limits: ", DATACARDDIR

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    masses = []

    for i,s in enumerate(sign):
        #print "theory x-sec (pb): ", sample[ samples[s]['files'][0] ]['xsec']
        masses.append(samples[s]['stop_mass'])

    masses = np.array(masses)
    masses = np.unique(np.array(masses))

    loop_ct = ctaus
    if "Stealth" in sign[0]:
        loop_ct = ctaus_stealth 
    loop_ct = np.unique(loop_ct)

    #Initialize here dictionaries
    #Dictionary with various predictions
    theory = defaultdict(dict)
    obs = defaultdict(dict)
    mean_val = defaultdict(dict)
    sigma_1_up = defaultdict(dict)
    sigma_1_down = defaultdict(dict)
    sigma_2_up = defaultdict(dict)
    sigma_2_down = defaultdict(dict)

    for sn in sign:
        print sn
        card_name = RESULTS + "/" + sn +".txt"#s.replace('_ctau'+str(samples[s]['ctau']),'_ctau'+str(ct)) + ".txt"
        print "read ",card_name
        if not os.path.isfile(card_name):
            continue
        card = open( card_name, 'r')
        val = card.read().splitlines()
        if len(val) == 0:
            continue
        if len(val) != 6 and not blind:
            continue
                                
        if blind:
            min_dist_sigma_mean = min( (float(val[3])-float(val[2]))/float(val[2]), (float(val[2])-float(val[1]))/float(val[2]))
        else:
            min_dist_sigma_mean = min( (float(val[3+1])-float(val[2+1]))/float(val[2+1]), (float(val[2+1])-float(val[1+1]))/float(val[2+1]))
        if min_dist_sigma_mean<0.1  and not toys:
            continue

        m = samples[sn]['stop_mass']
        ct = samples[sn]['ctau']

        obs[m][ct]          = float(val[0])#/1000.
        sigma_2_down[m][ct] = float(val[1])#/1000.
        sigma_1_down[m][ct] = float(val[2])#/1000.
        mean_val[m][ct]     = float(val[3])#/1000.
        sigma_1_up[m][ct]   = float(val[4])#/1000.
        sigma_2_up[m][ct]   = float(val[5])#/1000.

    loop_ctau_minimal = []
    for m in masses:
        for ct in obs[m].keys():
            print m, ct, obs[m][ct]
            loop_ctau_minimal.append(ct)

    loop_ctau_minimal = np.unique( np.sort( np.array(loop_ctau_minimal) )  )
    
    x = array('d',[])
    y = array('d',[])
    z = array('d',[])
    xt = array('d',[])
    yt = array('d',[])
    zt = array('d',[])
       
         
    n_bins_y = 100
    #orig:
    #min_bin_y = np.log10(250)
    #max_bin_y = np.log10(1550)
    min_bin_y = np.log10(200)
    max_bin_y = np.log10(1600)

    #original
    n_bins_x = 60
    n_bins_x = 100
    #in meters
    min_bin_x = np.log10(loop_ctau_minimal[0]/1000.)-0.5
    max_bin_x = np.log10(loop_ctau_minimal[-1]/1000.)+0.5
    #Let's try go higher/lower in ctau
    min_bin_x = np.log10(0.001)-0.5
    max_bin_x = np.log10(100000./1000.)+0.5

    #Try
    n_bins_x = 100
    n_bins_y = 100

    #Add theory border points
    #need to fill the entire plane up to the borders with theory values to avoid extrapolations
    min_bin_x_lin = 1000*(10**(min_bin_x) )
    max_bin_x_lin = 1000*(10**(max_bin_x) ) - 10 #avoid rejection because of overflow

    loop_ct_th = [min_bin_x_lin]
    for ct in loop_ct:
        loop_ct_th.append(ct)
    loop_ct_th.append(max_bin_x_lin)
    loop_ct_th = np.array(loop_ct_th)


    h = TH2D("h","h", n_bins_x,min_bin_x,max_bin_x,n_bins_y,min_bin_y,max_bin_y)
    ht = TH2D("ht","ht", n_bins_x,min_bin_x,max_bin_x,n_bins_y,min_bin_y,max_bin_y)

    #xsec in pb
    xsec = { 300:10.00*1000., 500:0.609*1000., 700:0.0783*1000., 900:0.0145*1000., 1100:0.00335*1000., 1300:0.000887*1000., 1500:0.000257*1000.}

    max_lim = 0
    th_limits_array = []
    for m in masses:
        valid_ctau = []
        all_ctau = []
        list_limits_each_mass = []
        list_limits_each_mass_th = []
        th_limits_array.append(xsec[m])
        for ct in loop_ctau_minimal:
            if ct in obs[m].keys():
                if obs[m][ct]==0:
                    print "no limits for: ", m, ct
                if obs[m][ct]!=0:
                    #in meters
                    h.Fill(np.log10(ct/1000.),np.log10(m),obs[m][ct])
                    valid_ctau.append(np.log10(ct/1000.))
                    #h.Fill(np.log10(m),np.log10(ct),np.log10(obs[m][ct]))
                    #valid_ctau.append(np.log10(ct))
                    list_limits_each_mass.append(obs[m][ct])
                    max_lim = max(max_lim,obs[m][ct])

        for ct_th in loop_ct_th:
            all_ctau.append(np.log10(ct_th/1000.))
            ht.Fill(np.log10(ct_th/1000.),np.log10(m),xsec[m])
            list_limits_each_mass_th.append(xsec[m])


        y += array('d', [math.log10(float(m))]*len(valid_ctau))
        x += array('d', valid_ctau)
        z += array('d', np.log10(list_limits_each_mass))

        yt += array('d', [math.log10(float(m))]*len(all_ctau))
        xt += array('d', all_ctau)
        zt += array('d', np.log10(list_limits_each_mass_th))


    c1 = TCanvas("c1", "Exclusion Limits", 900, 675)
    c1.cd()
    c1.GetPad(0).SetBottomMargin(0.12)
    c1.GetPad(0).SetTopMargin(0.08)
    c1.GetPad(0).SetRightMargin(0.17)#12)
    c1.GetPad(0).SetLeftMargin(0.12)#12)
    c1.GetPad(0).SetTicks(1, 1)
    c1.GetPad(0).SetLogz()
    #c1.GetPad(0).SetLogy()
    c1.GetPad(0).SetLogx()

    h = interpolate2D(x,y,z, h, 0.02,0.1, inter = 'rbf', norm = 'euclidean')
    h = log_scale_conversion(h)

    ht = interpolate2D(xt,yt,zt, ht, 0.02,0.1, inter = 'rbf', norm = 'euclidean')
    ht = log_scale_conversion(ht)


    h.SetMinimum(0.1)
    #h.SetMaximum(3e6)
    h.SetMaximum(1000)
    #h.SetMaximum(max_lim)
    h.GetYaxis().SetTitle("m_{#tilde{t}} (GeV)")
    h.GetXaxis().SetTitle("c #tau_{S} (m)")
    h.GetYaxis().SetTitleOffset(1.6)
    h.GetZaxis().SetTitle("95% CL limits on #sigma (fb)")
    h.Draw("COLZ")

    hr = ht.Clone("hr")
    hr.Divide(h)
    contours = array('d', [1.0])
    hr.SetContour(1,contours)
    #contours = array('d', [1.0,10.,100.,1000.])
    #hr.SetContour(4,contours)
    hr.SetLineColor(1)#new
    hr.SetLineWidth(4)#new
    hr.SetLineStyle(2)#new
    hr.SetFillColorAlpha(1,0.15)
    hr.SetFillStyle(3444)
    hr.Draw("CONT LIST same")#CONT creates a filled area
    #hr.Draw("CONT 3 same")#CONT creates a filled area
    
    c1.Update()
    conts = gROOT.GetListOfSpecials().FindObject("contours")
    print "these many conts: ", conts.GetSize()
    conts_gr = conts.At(0)
    print "these many conts_gr: ", conts_gr.GetSize()
    #print conts_gr.Print()

    curv1 = conts_gr.First()
    print "first graph in conts_gr: "
    print curv1.Print()
    if conts_gr.GetSize()>1:
        curv2 = conts_gr.After(curv1)
        print "second graph in conts_gr: "
        print curv2.Print()

    new_x_val1 = []
    new_y_val1 = []
    new_x_val2 = []
    new_y_val2 = []

    new_x_val = []
    new_y_val = []

    for ngr in range(curv1.GetN()):
        print "curv1"
        print ngr, 10**(curv1.GetPointX(ngr)), curv1.GetPointY(ngr)
        new_x_val1.append(10**curv1.GetPointX(ngr))
        new_y_val1.append(curv1.GetPointY(ngr))

        new_x_val.append(10**curv1.GetPointX(ngr))
        new_y_val.append(curv1.GetPointY(ngr))

    if conts_gr.GetSize()>1:
        for ngr in range(curv2.GetN()):
            print "curv2"
            print ngr, 10**(curv2.GetPointX(ngr)), curv2.GetPointY(ngr)
            new_x_val2.append(10**curv2.GetPointX(ngr))
            new_y_val2.append(curv2.GetPointY(ngr))

            new_x_val.append(10**curv2.GetPointX(ngr))
            new_y_val.append(curv2.GetPointY(ngr))
    #print hr.Print()

    print "This works, now try to rotate the thing!"
    new_gr = TGraph(len(new_x_val),np.array(new_x_val),np.array(new_y_val))
    new_gr.SetLineColor(1)
    new_gr.SetLineStyle(2)
    new_gr.SetLineWidth(2)
    h.Draw("COLZ")
    new_gr.Draw("L sames")

    ##hr.Draw("CONT 3 same")#CONT creates a filled area
    ##c1.Update()

    h.GetYaxis().SetNoExponent(True)
    h.GetYaxis().SetMoreLogLabels(True)
    h.GetYaxis().SetTitleSize(0.048)
    h.GetXaxis().SetTitleSize(0.048)
    h.GetZaxis().SetTitleSize(0.048)
    h.GetYaxis().SetTitleOffset(1.)
    h.GetXaxis().SetTitleOffset(0.8)

    #FR comment: larger labels --> need also different borders
    h.GetXaxis().SetTitleSize(0.055)
    h.GetYaxis().SetTitleSize(0.055)
    h.GetZaxis().SetTitleSize(0.055)

    drawCMS_simple(this_lumi, "", ERA="", onTop=True, left_marg_CMS=0.175,top_marg_lumi = 0.98)

    if ms_inp==100:
        ms_string = "100"
    else:
        ms_string = "mstop_min_225"

    OUTSTRING = plot_dir+"/Exclusion_2D_ms_"+str(ms_string)+"_"+ch
    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")
    c1.Close()

    newFile = TFile(OUTSTRING+".root", "RECREATE")#+"_"+TAGVAR+comb_fold_label+ ".root", "RECREATE")
    newFile.cd()
    new_gr_mm = TGraph(len(new_x_val),np.array(new_x_val)*1000.,np.array(new_y_val))
    new_gr_mm.SetLineColor(1)
    new_gr_mm.SetLineStyle(2)
    new_gr_mm.SetLineWidth(2)
    new_gr_mm.Write("contour")
    c1.Write()
    newFile.Close()
    print "Info: written file: ", OUTSTRING+".root"#"_"+TAGVAR+comb_fold_label+".root"


    

def evaluate_median_expected_difference(file_names,labels,plot_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,LUMI=LUMI):

    if combination:
        print "COMMENTED because crashes, fixmeee"
        #exit()
        PLOTLIMITSDIR      = "plots/Limits/v5_calo_AOD_combination/"#"/"#"_2017_signal/"#
        #exit()
        LUMI = 137.4*1000
    else:
        PLOTLIMITSDIR      = "plots/Limits/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
    #    LUMI = LUMI

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()

    colors = [801,856,825,881,2,602,880,798,856,5]
    lines = [1,2,1,2,1,2,1,2,1,3]
    gStyle.SetLegendFillColor(0)
    c2 = TCanvas("c2", "c2", 800, 600)
    #c2 = TCanvas("c2", "c2", 1200, 600)
    c2.cd()
    c2.SetGrid()
    c2.GetPad(0).SetTopMargin(0.06)
    c2.GetPad(0).SetRightMargin(0.05)
    c2.GetPad(0).SetTicks(1, 1)
    c2.GetPad(0).SetGridx()
    c2.GetPad(0).SetGridy()
    #c2.GetPad(0).SetLogy()
    c2.GetPad(0).SetLogx()
    top = 0.9
    nitems = len(file_names)
    leg = TLegend(0.5,0.2+0.5,0.9,0.4+0.5)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetHeader("95% CL limits, c#tau_{"+particle+"}="+str(ctaupoint)+" mm")
    leg.SetTextSize(0.04)    
    graph_exp = {}
    graph_1sigma = {}

    x = {}
    y = {}

    for i,l in enumerate(file_names):
        filename = TFile(l, "READ")
        #graph_1sigma[i] = TGraph()
        #graph_exp[i] = TGraph()
        graph_1sigma[i] = filename.Get("pl"+str(ctaupoint)+"_1sigma")
        graph_exp[i] = filename.Get("pl"+str(ctaupoint)+"_exp")
        x[i] = []
        y[i] = []

        for r in range(graph_exp[i].GetN()):
            a = c_double()
            b = c_double()
            graph_exp[i].GetPoint(r,a,b)
            x[i].append(a.value)
            y[i].append(b.value)
        #print x[i]
        #print y[i]
        #print graph_exp[i].Print()
    #print y
    pair_diff = {}
    pair_graph = {}
    for l in range(0,len(x.keys()),2):
        pair_diff[l] = 100.*(np.array(y[l+1])-np.array(y[l]))/( (np.array(y[l])+np.array(y[l+1]))/2.  )
        pair_graph[l] = TGraph()
        #print l, y[l], y[l+1], pair_diff[l]
        for point in range(len(y[l])):
            pair_graph[l].SetPoint(point, x[l][point], pair_diff[l][point])
        pair_graph[l].SetLineStyle(lines[l])
        pair_graph[l].SetLineWidth(3)
        pair_graph[l].SetLineColor(colors[l])
        pair_graph[l].SetMarkerColor(colors[l])
        pair_graph[l].SetFillColorAlpha(colors[l],0.5)
        pair_graph[l].SetMinimum(-.1)
        pair_graph[l].SetMaximum(3.5)
        pair_graph[l].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        pair_graph[l].GetYaxis().SetTitle("Impact on median expected limit (%)")
        pair_graph[l].GetXaxis().SetNoExponent(True)
        pair_graph[l].GetXaxis().SetMoreLogLabels(True)
        pair_graph[l].GetXaxis().SetTitleSize(0.048)
        pair_graph[l].GetYaxis().SetTitleSize(0.048)
        pair_graph[l].GetYaxis().SetTitleOffset(0.8)
        pair_graph[l].GetXaxis().SetTitleOffset(0.9)
        if l == 0:
            pair_graph[l].SetMarkerStyle(20)
            pair_graph[l].Draw("APL3")
        else:
            pair_graph[l].SetMarkerStyle(20)
            pair_graph[l].Draw("SAME,PL3")
        leg.AddEntry(pair_graph[l],labels[l+1],"L")
        filename.Close()
    leg.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = PLOTLIMITSDIR+"/Expected_median_difference_ctau"+str(ctaupoint)+"_combination_"+CHAN
    c2.Print(OUTSTRING+plot_label+".png")
    c2.Print(OUTSTRING+plot_label+".pdf")
    c2.Close()

        

samples_to_run = data#sign#data#back#data#back#data#sign#data#back#data#back#data#back#data#sign#back#data#back#data#back+data#data#data+back#+data
jet_tag = ""#+
clos = False

#ERA
#if ERA=="2016":
#    #jet_tag+="_B-F"
#    era_tag+="_G-H"#"_B-F"#"_B-F"#"_G-H"#"_G-H"#"_B-F"#"_B-F"#
jet_tag += era_tag

kill_tag = ""
if KILL_QCD:
    kill_tag += "_MinDPhi_0p5"
jet_tag+=kill_tag

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
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]


print "Ntupledir: ", NTUPLEDIR
#print "Luminosity: ", data[0], LUMI

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

mu_scale = 1.#50.#0.01#1.#2.#5.#10.#0.1#
contam = False
br_h=100#75#50#25#0#

#ERA
label = ""
if ERA=="2016" and "B-F" in jet_tag:
    label = "_B-F"
if ERA=="2016" and "G-H" in jet_tag:
    label = "_G-H"

combi_folder = "2016_BFGH_2017_2018"#"2016_GH_2017_2018",#
era_list = ["2016_B-F","2016_G-H","2017","2018"]#["2016_G-H","2017","2018"],#
limit_datacard_dir = OUTCOMBI#OUTCOMBI,#OUTPUTDIR,
limit_plot_dir = PLOTLIMITSDIR


#DS paper
#sign_to_do = sign_zPrime
#sign_to_do = sign_HeavyHiggs
sign_to_do = sign_stealth

print "Test one thing at a time"
#get_tree_weights(sign_to_do,dataset_label=label,main_pred_sample="HighMET")
#exit()

'''
write_datacards(
    get_tree_weights(sign_to_do,dataset_label=label,main_pred_sample="HighMET"),#LUMI),#137.4*1000 full Run2
    sign_to_do,
    main_pred_reg = REGION,
    main_pred_sample="HighMET",
    #extrapolation
    extr_region= "WtoLN",
    unc_list = ["lumi","bkg","PU","QCD_scales","PDF","jet","tau","DNN"],
    dataset_label = label,
    comb_fold_label = jet_tag,
    check_closure=False,
    pred_unc = 1.,#here to boost it if needed
    run_combine = False,#True,#False,#True,#False,#True,#True,
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
    do_time_smearing = False,
    do_ct_average=False,
    blind = BLIND

)
#exit()

#submit_combine_condor(
retrieve_combine_condor(
    sign_to_do,
    comb_fold_label = combi_folder,
    eras = era_list,
    add_label="",
    label_2=jet_tag,
    check_closure=clos,
    pred_unc = 1.,
    run_combine = True,
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    BR_SCAN_H=br_h,
    blind = BLIND,
    toys = False
)
#exit()



plot_limits_vs_ctau_split(
    limit_datacard_dir,
    limit_plot_dir,
    sign_to_do,
    comb_fold_label = combi_folder,
    combination = True if limit_datacard_dir==OUTCOMBI else False,
    eta = DO_ETA,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
    BR_SCAN_H=br_h,
    blind = BLIND
)

exit()
'''
#Arrange samples per ctau, ms, and list different mstop
ctaus_stealth = [0.01,0.1,1,10,100,1000]
l_mstop = [300,500,700,900,1100,1300,1500,]
l_ms = ["100","_mstop_m_225"]
l_ms = ["100"]

'''
for ms_str in l_ms:
    ms = 0
    if ms_str=="100":
        print "Now: ", ms_str
        l_mstop = [300,500,700,900,1100,1300,1500,]
        for ct in ctaus_stealth:
            list_stealth = []
            if ct<1:
                ctn=str(ct).replace(".","p")
            else:
                ctn = str( int(ct) )
            for mstop in l_mstop:
                print "Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms100_ctau"+str(ctn)+"_"+ch
                list_stealth.append("Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms100_ctau"+str(ctn)+"_"+ch)
            print "Collected: ", ctn, list_stealth
            plot_limits_vs_mass_split(
                limit_datacard_dir,
                limit_plot_dir,
                list_stealth,
                ms_inp=100,
                ctau_inp=ct,
                comb_fold_label=combi_folder,
                combination = True if limit_datacard_dir==OUTCOMBI else False,
                eta = DO_ETA,
                eta_cut = CUT_ETA,
                phi_cut = CUT_PHI,
                scale_mu=mu_scale,
                contamination = contam,
                BR_SCAN_H=br_h,
                blind = BLIND,
                toys=False)

    else:
        print "Now: ", ms_str
        l_mstop = [500,700,900,1100,1300,1500,]
        for ct in ctaus_stealth:
            list_stealth = []
            if ct<1:
                ctn=str(ct).replace(".","p")
            else:
                ctn = str( int(ct) )
            for mstop in l_mstop:
                print "Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms"+str(int(mstop-225))+"_ctau"+str(ctn)+"_"+ch
                list_stealth.append("Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms"+str(int(mstop-225))+"_ctau"+str(ctn)+"_"+ch)
            print "Collected: ", ctn, list_stealth
            plot_limits_vs_mass_split(
                limit_datacard_dir,
                limit_plot_dir,
                list_stealth,
                ms_inp=0,
                ctau_inp=ct,
                comb_fold_label=combi_folder,
                combination = True if limit_datacard_dir==OUTCOMBI else False,
                eta = DO_ETA,
                eta_cut = CUT_ETA,
                phi_cut = CUT_PHI,
                scale_mu=mu_scale,
                contamination = contam,
                BR_SCAN_H=br_h,
                blind = BLIND,
                toys=False)
'''

ctaus_stealth = [0.01,0.1,1,10,100,1000]
l_ms = ["100","_mstop_m_225"]
#l_ms = ["100"]
#l_ms = ["_mstop_m_225"]


for ms_str in l_ms:
    ms = 0
    list_stealth = []
    if ms_str=="100":
        ms = 100
        print "Now: ", ms_str
        l_mstop = [300,500,700,900,1100,1300,1500,]
        for ct in ctaus_stealth:
            if ct<1:
                ctn=str(ct).replace(".","p")
            else:
                ctn = str( int(ct) )
            for mstop in l_mstop:
                print "Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms100_ctau"+str(ctn)+"_"+ch
                list_stealth.append("Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms100_ctau"+str(ctn)+"_"+ch)
    else:
        print "Now: ", ms_str
        l_mstop = [500,700,900,1100,1300,1500,]
        for ct in ctaus_stealth:
            if ct<1:
                ctn=str(ct).replace(".","p")
            else:
                ctn = str( int(ct) )
            for mstop in l_mstop:
                print "Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms"+str(int(mstop-225))+"_ctau"+str(ctn)+"_"+ch
                list_stealth.append("Stealth"+ch+"_2t6j_mstop"+str(mstop)+"_ms"+str(int(mstop-225))+"_ctau"+str(ctn)+"_"+ch)

    print "Go with ",ms
    print "And list: ", list_stealth
    plot_limits_2D_split(
    limit_datacard_dir,
    limit_plot_dir,
    list_stealth,
    ms_inp=ms,
    comb_fold_label=combi_folder,
    combination = True if limit_datacard_dir==OUTCOMBI else False,
    eta = DO_ETA,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
    BR_SCAN_H=br_h,
    blind = BLIND,
    toys=False)
