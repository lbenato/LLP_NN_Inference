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
from array import array
from awkward import *
import root_numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict, OrderedDict
from itertools import chain

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

ERA                = "2018"
REGION             = "SR"#"WtoMN"#"WtoEN"#"WtoMN"#"ZtoEE"#"JetHT"
CUT                = "isSR"#"isWtoMN"#"isWtoEN"#"isWtoMN"#"isZtoEE"#"isJetHT"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"
KILL_QCD           = True#True#False#True
DO_ETA             = True
DO_PHI             = False#False#
if DO_PHI:
    DO_ETA = False
CUT_ETA            = True#True#True#False#True#True#False
CUT_PHI            = True

print "\n"
print "region: ", REGION
print "kill qcd: ", KILL_QCD
print "do eta: ", DO_ETA
print "do phi: ", DO_PHI
print "eta cut: ", CUT_ETA
print "phi cut: ", CUT_PHI
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
if REGION=="WtoEN":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
if REGION=="WtoMN":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"
if REGION=="WtoEN_MET":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoEN_noMT/"
if REGION=="WtoMN_MET":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_WtoMN_noMT/"
if REGION=="SR":
    #print "Old dir good for MC bkg"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#

    #print "SR: updated to CSC analysis triggers"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#

    #print " IMPORTANT NOT GOOD FOR SIGNAL!!! "

    #print "BUT for MC we use the JER updated!"
    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_"+ERA+"_"+REGION+"_JER_for_AN/"#"_2017_signal/"#

    #NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+REGION+"_bin_1_2/"#"_2017_signal/"#
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+REGION+"/"#_bin_1_2/"#"_2017_signal/"#

    ####print "DEBUUUUG"
    ####NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_2018_SR_xcheck_tf_and_skim_condor_v5_updated/"


#print "Tranche 2!!!"
#NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"_tranche2/"
#print " OLD ! ! "
#NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_faulty_eta/v5_calo_AOD_"+ERA+"_"+REGION+"/"

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_SR_xcheck_slimmedJets/"

PRE_PLOTDIR        = "plots/Efficiency_xcheck/v6_calo_AOD_"+ERA+"_"
PLOTDIR            = "plots/Efficiency_xcheck/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"/"
OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+ERA+"_"+REGION+"_AN/"#"/"

#print "OUTPUTDIR for combination attempt"
#OUTPUTDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_"+ERA+"_"+REGION+"_AN_combi/"#"/"


CHAN               = "SUSY"
particle           = "#chi"
ctaupoint          = 1000
ctaus              = np.array([1, 3, 5, 7, 10, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 980, 1100, 1200, 1300, 1400, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 50000, 100000])
signalMultFactor   = 0.001#!!!
PRELIMINARY        = True
TAGVAR             = "nTagJets_0p996_JJ"

REBIN              = options.rebin
SAVE               = True
#LUMI               = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#8507.558453#59.74*1000#Run2018

back = ["DYJetsToLL"]
#back = ["TTbarGenMET"]
#back = ["WJetsToLNu"]
#back = ["VV"]
#back = ["QCD"]
#back = ["ZJetsToNuNu"]
#back = ["TTbar"]
#back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET","VV"]

#back = ["All"]
#back = ["VV"]
#back = ["QCD","WJetsToLNu","TTbarGenMET"]
#data = ["SingleMuon"]
#data = ["SingleElectron"]
#data = ["EGamma"]
#data = ["MuonEG"]
#data = ["MET"]
data = ["HighMET"]
#data = ["JetHT"]
sign = [
    #"SUSY_all",
    #"SUSY_mh400_pl1000",
    #"SUSY_mh200_pl1000",
    #"SUSY_mh400_pl1000_XL",
    "SUSY_mh400_pl1000","SUSY_mh300_pl1000","SUSY_mh250_pl1000","SUSY_mh200_pl1000","SUSY_mh175_pl1000","SUSY_mh150_pl1000","SUSY_mh127_pl1000",

    #'splitSUSY_M-2400_CTau-300mm','splitSUSY_M-2400_CTau-1000mm','splitSUSY_M-2400_CTau-10000mm','splitSUSY_M-2400_CTau-30000mm',
    #'gluinoGMSB_M2400_ctau1','gluinoGMSB_M2400_ctau3','gluinoGMSB_M2400_ctau10','gluinoGMSB_M2400_ctau30','gluinoGMSB_M2400_ctau100','gluinoGMSB_M2400_ctau300','gluinoGMSB_M2400_ctau1000','gluinoGMSB_M2400_ctau3000','gluinoGMSB_M2400_ctau10000','gluinoGMSB_M2400_ctau30000','gluinoGMSB_M2400_ctau50000',
    #'XXTo4J_M100_CTau100mm','XXTo4J_M100_CTau300mm','XXTo4J_M100_CTau1000mm','XXTo4J_M100_CTau3000mm','XXTo4J_M100_CTau50000mm',
    #'XXTo4J_M300_CTau100mm','XXTo4J_M300_CTau300mm','XXTo4J_M300_CTau1000mm','XXTo4J_M300_CTau3000mm','XXTo4J_M300_CTau50000mm',
    #'XXTo4J_M1000_CTau100mm','XXTo4J_M1000_CTau300mm','XXTo4J_M1000_CTau1000mm','XXTo4J_M1000_CTau3000mm','XXTo4J_M1000_CTau50000mm',
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
    print "Fake jet tag"
    jet_tag=""
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

print LUMI

COMPARE = options.compare
DRAWSIGNAL = options.drawsignal

########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors_jj = [1,2,4,418,801856]
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

print "Warning, overflow causing nans, remove last bin"
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

maxeff = 0.0015#2#15

def tau_weight_calc(llp_ct, new_ctau, old_ctau):
    '''
    llp_ct is a numpy array
    new_ctau and old_ctau are float
    '''
    source = np.exp(-1.0*llp_ct/old_ctau)/old_ctau**2
    weight = 1.0/new_ctau**2 * np.exp(-1.0*llp_ct/new_ctau)/source
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

'''
def jet_correlations(sample_list,add_label="",check_closure=False):
    for i, s in enumerate(sample_list):
        wp_0p9 = 0.045757491
        wp_0p996 = 0.0017406616
        hist = TH2F(s,s, len(dnn_bins)-1, dnn_bins , len(dnn_bins)-1, dnn_bins)
        lineX = TLine(dnn_bins[0],wp_0p996,wp_0p996,wp_0p996)
        lineY = TLine(wp_0p996,dnn_bins[0],wp_0p996,wp_0p996)

        lineX_ = TLine(dnn_bins[0],wp_0p9,wp_0p9,wp_0p9)
        lineY_ = TLine(wp_0p9,dnn_bins[0],wp_0p9,wp_0p9)
        min_bin = 0.#-20.#0.0001
        max_bin = 1.#0#1.0001
        n_bins = 20#50
        #hist = TH2F(s,s, n_bins,min_bin,max_bin,n_bins,min_bin,max_bin)
        hist.Sumw2()
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            print "Adding ", ss
            chain[s].Add(NTUPLEDIR + ss + ".root")
        print(chain[s].GetEntries())

        ev_weight  = "EventWeight*PUReWeight"
        if CUT == "isJetMET_low_dPhi_500":
            cutstring = ev_weight+"*(MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && ( HLT_PFJet500_v))"
        elif CUT == "isDiJetMET":
            #change!
            cutstring = ev_weight+"*(MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
        elif CUT == "isMN":
            cutstring = ev_weight+"*(nLeptons==1)"
        elif CUT == "isEN":
            cutstring = ev_weight+"*(nLeptons==1)"
        else:
            cutstring = ev_weight

        if check_closure:
            cutstring+="*(Jets[0].sigprob<0.996 && Jets[1].sigprob<0.996)"

        print cutstring
        chain[s].Project(s, "-TMath::Log10(Jets[1].sigprob) : -TMath::Log10(Jets[0].sigprob)",cutstring)

        profX = TProfile(hist.ProfileX("prof"))
        profX.SetLineColor(881)
        profX.SetFillColor(1)
        profX.SetLineWidth(2)
        profX.SetMarkerStyle(25)
        profX.SetMarkerColor(881)

        can = TCanvas("can","can", 1000, 900)
        can.cd()
        can.SetLogy()
        can.SetLogx()
        can.SetLogz()
        leg = TLegend(0.7, 0.1, 0.9, 0.3)
        leg.SetTextSize(0.035)

        hist.SetTitle("")
        hist.GetXaxis().SetTitle("-log10( jet[0] DNN score )")
        hist.GetYaxis().SetTitle("-log10( jet[1] DNN score )")
        hist.SetMarkerSize(1.5)
        hist.Draw("COLZ")
        profX.Draw("PL,sames")
        leg.AddEntry(profX,"TProfileX","PL")
        leg.AddEntry(lineX,"DNN 0.996","L")
        leg.AddEntry(lineX_,"DNN 0.9","L")
        lineY.SetLineColor(2)
        lineY.SetLineWidth(2)
        lineY.SetLineStyle(1)
        lineY.Draw("sames")
        lineX.SetLineColor(2)
        lineX.SetLineWidth(2)
        lineX.SetLineStyle(1)
        lineX.Draw("sames")

        lineY_.SetLineColor(4)
        lineY_.SetLineWidth(2)
        lineY_.SetLineStyle(2)
        lineY_.Draw("sames")
        lineX_.SetLineColor(4)
        lineX_.SetLineWidth(2)
        lineX_.SetLineStyle(2)
        lineX_.Draw("sames")
        leg.Draw()

        drawRegion(REGION,setNDC=False,color=0)
        drawCMS(samples, LUMI, "Preliminary",onTop=True,data_obs=data)
        can.Print(PLOTDIR+"JetCorrelation_"+s+add_label+".png")
        can.Print(PLOTDIR+"JetCorrelation_"+s+add_label+".pdf")
'''        

def plot_roc_curve(tree_weight_dict_s,tree_weight_dict_b,sign,back,add_label="",check_closure=False,eta=False,j_idx=-1,eta_cut=0.,eta_invert=False):

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut!=0.:
        if eta_invert==False:
            print "Apply acceptance cut |eta|<"+str(eta_cut)
            add_label+="_eta_"+str(abs(eta_cut)).replace(".","p")
        else:
            print "Apply acceptance cut |eta|>"+str(eta_cut)
            add_label+="_eta_larger_"+str(abs(eta_cut)).replace(".","p")

    tree_weights = {}
    chain_entries = {}
    chain_entries_cumulative = {}
    chain = {}


    truth_s = np.array([])
    prob_s = np.array([])
    w_s = np.array([])
    for i, s in enumerate(sign):
        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept",CUT]#"nLeptons"
        if CUT=="isDiJetMET":
            list_of_variables += ["MinSubLeadingJetMetDPhi"]
        if ERA=="2018" and CUT == "isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]["files"]):
            tree_weights[l] = tree_weight_dict_s[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
                print "Entries per sample ", ss, " : ", chain_entries[l]
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict_s[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    #cut_mask = np.logical_and(arrays["HLT_PFJet500_v"]==True , arrays["pt"]<25)
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #change!
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #if KILL_QCD:
                    #    cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #if KILL_QCD:
                    #    cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #enhance MET
                    #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                    #no MT
                    #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if CUT == "isSR" and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))

                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)

                print "Skip events with negative weights . . ."
                cut_mask = np.logical_and( cut_mask, arrays["EventWeight"]>0)
                #cut_mask = np.logical_and( cut_mask, arrays["HT"]>100)

                #Signal: consider only gen-matched jets
                cut_mask_gen = np.logical_and(cut_mask,arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1)
                cut_mask = (cut_mask_gen.any()==True)
                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]

                if eta_cut!=0:
                    if eta_invert==False:
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-abs(eta_cut) , arrays["Jets.eta"]<abs(eta_cut))
                    else:
                        cut_mask_eta = np.logical_or(arrays["Jets.eta"]>abs(eta_cut) , arrays["Jets.eta"]<-abs(eta_cut))

                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    cut_mask_eta = np.logical_and(cut_mask_eta,cut_mask_gen)
                    cut_mask = (cut_mask_eta.any()==True)
                    #if eta:
                    #    pt = arrays["Jets.eta"][cut_mask_eta][cut_mask]
                    #else:
                    #    pt = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_eta][cut_mask]

                

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]
                del arrays

                prob_s = np.concatenate( (prob_s, np.hstack(sigprob)) )
                w_s = np.concatenate( (w_s, np.hstack( sigprob.astype(bool)*weight ) ) )

                #We'll have to flatten and remove unnecessary zeros...
                #print np_den[130.0].shape
                en_it = time.time()
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"

        #Here vectors should be ready
        #print s, np_num, np_w_num
        #print s, np_den, np_w_den

    print "tot signal: ", prob_s, w_s
    print "shapes: ",
    print prob_s.shape
    print w_s.shape
    truth_s_bin = np.dstack( (np.ones(prob_s.shape[0]),np.zeros(prob_s.shape[0])) ).reshape(prob_s.shape[0],2) 
    truth_s = np.ones(prob_s.shape[0]) 



    truth_b = np.array([])
    prob_b = np.array([])
    w_b = np.array([])
    for i, s in enumerate(back):

        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept","HT",CUT]#"nLeptons"
        if CUT=="isDiJetMET":
            list_of_variables += ["MinSubLeadingJetMetDPhi"]
        if ERA=="2018" and CUT == "isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]["files"]):
            tree_weights[l] = tree_weight_dict_b[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
                print "Entries per sample ", ss, " : ", chain_entries[l]
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict_b[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    #cut_mask = np.logical_and(arrays["HLT_PFJet500_v"]==True , arrays["pt"]<25)
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #change!
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #if KILL_QCD:
                    #    cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                    #enhance MET
                    #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                    #no MT
                    #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if CUT == "isSR" and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))
                ###cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))

                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)

                print "Skip events with negative weights . . ."
                cut_mask = np.logical_and( cut_mask, arrays["EventWeight"]>0)
                #cut_mask = np.logical_and( cut_mask, arrays["HT"]>100)

                sigprob = arrays["Jets.sigprob"][cut_mask]

                if eta_cut!=0:
                    if eta_invert==False:
                        cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-abs(eta_cut) , arrays["Jets.eta"]<abs(eta_cut))
                    else:
                        cut_mask_eta = np.logical_or(arrays["Jets.eta"]>abs(eta_cut) , arrays["Jets.eta"]<-abs(eta_cut))
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    cut_mask = (cut_mask_eta.any()==True)
                    #if eta:
                    #    pt = arrays["Jets.eta"][cut_mask_eta][cut_mask]
                    #else:
                    #    pt = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_eta][cut_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_b[s][ss]
                del arrays

                prob_b = np.concatenate( (prob_b, np.hstack(sigprob) if len(sigprob)>0 else np.array([]) ) )
                w_b = np.concatenate( (w_b, np.hstack(sigprob.astype(bool)*weight) if len(sigprob)>0 else np.array([]) ) )

                #We'll have to flatten and remove unnecessary zeros...
                #print np_den[130.0].shape
                en_it = time.time()
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


    print "tot background: ", prob_b, w_b
    print "shapes: ",
    print prob_b.shape
    print w_b.shape
    truth_b_bin = np.dstack( (np.zeros(prob_b.shape[0]),np.ones(prob_b.shape[0])) ).reshape(prob_b.shape[0],2)
    truth_b = np.zeros(prob_b.shape[0]) 
    print truth_b


    ##Normalize weights
    norm_s = w_s.sum(axis=0)
    w_s_norm = np.true_divide(w_s,norm_s) 

    norm_b = w_b.sum(axis=0)
    w_b_norm = np.true_divide(w_b,norm_b) 

    #Upsampling signal x5, not very useful
    #truth_s = np.repeat(truth_s,5)
    #w_s_norm = np.repeat(w_s_norm,5)
    #prob_s = np.repeat(prob_s,5)

    y_test    = np.concatenate((truth_b,truth_s))
    w_test    = np.concatenate((w_b_norm,w_s_norm))#.reshape(-1, 1)
    prob_test = np.concatenate((prob_b,prob_s))#.reshape(-1, 1)

    tpr_L = 0.636040507008249
    fpr_L = 0.00040904540701505433
    cut_fpr = fpr_L
    
    AUC = roc_auc_score(y_test, prob_test, sample_weight=w_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob_test, sample_weight=w_test)
    idx, _ = find_nearest(fpr,cut_fpr)

    plt.figure(figsize=(8,7))
    plt.rcParams.update({"font.size": 15}) #Larger font size                                                
    plt.plot(fpr, tpr, color="crimson", lw=2, label="AUC = {0:.4f}".format(AUC))
    plt.plot(fpr[idx], tpr[idx],"ro",color="crimson",label="w.p. {0:.4f}".format(thresholds[idx]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based")
    plt.ylim([0.6, 1.05])
    plt.xlim([0.0001, 1.05])
    plt.xscale("log")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", title="FCN")
    plt.grid(True)
    plt.savefig(PLOTDIR+"ROC"+add_label+".pdf")
    plt.savefig(PLOTDIR+"ROC"+add_label+".png")
    print "Info: ROC curve file "+PLOTDIR+"ROC"+add_label+".pdf has been created"

    with open(PLOTDIR+"fpr"+add_label+".npy", "wb") as f:
        np.save(f, fpr)
    with open(PLOTDIR+"tpr"+add_label+".npy", "wb") as f:
        np.save(f, tpr)
    with open(PLOTDIR+"thresholds"+add_label+".npy", "wb") as f:
        np.save(f, thresholds)
    with open(PLOTDIR+"idx"+add_label+".npy", "wb") as f:
        np.save(f, idx)

    print "Info: fpr/tpr/thresholds/idx saved in " +PLOTDIR+"*"+add_label+".npy"
    #print fpr
    #print tpr
    #print thresholds
    #print idx

    #print y_test.shape
    #print prob_test.shape
    #print w_test.shape
    #print AUC


def compare_roc_curves(list_comparison,add_label=""):    
    tpr = {}
    fpr = {}
    idx = {}
    thresholds = {}
    tpr_L = 0.636040507008249
    fpr_L = 0.00040904540701505433
    cut_fpr = fpr_L

    colors = ['crimson','green','skyblue','orange','gray','magenta','chocolate','yellow','black','olive']
    linestyles = ['-', '--', '-.', ':','-','--','-.',':']
    linestyles = ['-', '--', '-.', '-','--','-.',':','-', '--', '-.',]

    plt.figure(figsize=(8,7))
    plt.rcParams.update({"font.size": 15}) #Larger font size
    AUC = 0.4
    for i,l in enumerate(list_comparison):
        tpr[l] = np.load(PLOTDIR+"tpr"+l+".npy")
        fpr[l] = np.load(PLOTDIR+"fpr"+l+".npy")
        idx[l] = np.load(PLOTDIR+"idx"+l+".npy")
        thresholds[l] = np.load(PLOTDIR+"thresholds"+l+".npy")
        plt.plot(fpr[l], tpr[l], color=colors[i], lw=2, linestyle=linestyles[i], label="ROC"+l)#"AUC = {0:.4f}".format(AUC))
        plt.plot(fpr[l][idx[l]], tpr[l][idx[l]],"ro",color=colors[i],label="w.p. {0:.4f}".format(thresholds[l][idx[l]]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based, all eta(2018)")
    plt.title(str(ERA)+' MC')
    plt.ylim([0.6, 1.05])
    plt.xlim([0.0001, 1.05])
    plt.xscale("log")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", title="FCN")
    plt.grid(True)
    plt.savefig(PLOTDIR+"ROC_comparison.pdf")
    plt.savefig(PLOTDIR+"ROC_comparison.png")
    print "Info: ROC curve file "+PLOTDIR+"ROC_comparison.pdf has been created"



def var_vs_eta(sample_list,var,add_label="",check_closure=False):
    for i, s in enumerate(sample_list):
        hist = TH2D(s,s, len(more_bins_eta)-1, more_bins_eta, variable[var]['nbins'],variable[var]['min'],variable[var]['max'])
        #lineX = TLine(dnn_bins[0],wp_0p996,wp_0p996,wp_0p996)
        #lineY = TLine(wp_0p996,dnn_bins[0],wp_0p996,wp_0p996)

        #lineX_ = TLine(dnn_bins[0],wp_0p9,wp_0p9,wp_0p9)
        #lineY_ = TLine(wp_0p9,dnn_bins[0],wp_0p9,wp_0p9)
        #min_bin = 0.#-20.#0.0001
        #max_bin = 1.#0#1.0001
        #n_bins = 20#50
        #hist = TH2F(s,s, n_bins,min_bin,max_bin,n_bins,min_bin,max_bin)
        hist.Sumw2()
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            print "Adding ", ss
            chain[s].Add(NTUPLEDIR + ss + ".root")
        print(chain[s].GetEntries())

        ev_weight  = "EventWeight*PUReWeight"
        if CUT == "isJetMET_low_dPhi_500":
            cutstring = ev_weight+"*(MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && ( HLT_PFJet500_v))"
        elif CUT == "isDiJetMET":
            #change!
            cutstring = ev_weight+"*(MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
        elif CUT == "isMN":
            cutstring = ev_weight+"*(nLeptons==1)"
        elif CUT == "isEN":
            cutstring = ev_weight+"*(nLeptons==1)"
        else:
            cutstring = ev_weight

        if check_closure:
            cutstring+="*(Jets[0].sigprob<0.986 && Jets[1].sigprob<0.986)"

        print cutstring
        chain[s].Project(s, var+":Jets[0].eta",cutstring)

        profX = TProfile(hist.ProfileX("prof"))
        profX.SetLineColor(881)
        profX.SetFillColor(1)
        profX.SetLineWidth(2)
        profX.SetMarkerStyle(25)
        profX.SetMarkerColor(881)

        can = TCanvas("can","can", 1000, 900)
        can.cd()
        #if variable[var]['log']==True:
        #    can.SetLogy()
        #can.SetLogx()
        can.SetLogz()
        leg = TLegend(0.7, 0.1, 0.9, 0.3)
        leg.SetTextSize(0.035)

        hist.SetTitle("")
        hist.GetXaxis().SetTitle("jet[0] eta")
        hist.GetYaxis().SetTitle(variable[var]['title'])
        hist.SetMarkerSize(1.5)
        hist.Draw("COLZ")
        profX.Draw("PL,sames")
        leg.AddEntry(profX,"TProfileX","PL")
        #leg.AddEntry(lineX,"DNN 0.996","L")
        #leg.AddEntry(lineX_,"DNN 0.9","L")
        #lineY.SetLineColor(2)
        #lineY.SetLineWidth(2)
        #lineY.SetLineStyle(1)
        #lineY.Draw("sames")
        #lineX.SetLineColor(2)
        #lineX.SetLineWidth(2)
        #lineX.SetLineStyle(1)
        #lineX.Draw("sames")

        #lineY_.SetLineColor(4)
        #lineY_.SetLineWidth(2)
        #lineY_.SetLineStyle(2)
        #lineY_.Draw("sames")
        #lineX_.SetLineColor(4)
        #lineX_.SetLineWidth(2)
        #lineX_.SetLineStyle(2)
        #lineX_.Draw("sames")
        #leg.Draw()

        drawRegion(REGION,setNDC=False,color=0)
        drawCMS(samples, LUMI, "Preliminary",onTop=True,data_obs=data,text_size=0.04)
        can.Print(PLOTDIR+var.replace(".","_")+"_vs_eta_"+s+add_label+".png")
        can.Print(PLOTDIR+var.replace(".","_")+"_vs_eta_"+s+add_label+".pdf")

'''
def old_calculate_tag_eff(sample_list,add_label="",check_closure=False,eta=False,j_idx=-1):

    if check_closure:
        dnn_threshold = 0.95
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996
        print  "DNN threshold: ", dnn_threshold

    for i, s in enumerate(sample_list):
        if eta==True:
            hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins_eta)-1, more_bins_eta)
            hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins_eta)-1, more_bins_eta)
            hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins_eta)-1, more_bins_eta)
        else:
            hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins)-1, more_bins)
            hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins)-1, more_bins)
            hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins)-1, more_bins)
        hist_den[s].Sumw2()
        hist_num[s].Sumw2()
        hist_num_cutbased[s].Sumw2()
        print s
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            print "Adding ", ss
            chain[s].Add(NTUPLEDIR + ss + ".root")
        print(chain[s].GetEntries())
        n=0
        n_ev=0
        for event in chain[s]:
            #print "Event: ", n_ev
            #print REGION
            #print CUT, event.isSR
            #print s
            n_ev+=1
            #print "nJets: ", event.nCHSJetsAcceptanceCalo
            ##CR cuts
            if (CUT == "isZtoMM" and not(event.isZtoMM)): continue
            if (CUT == "isZtoEE" and not(event.isZtoEE)): continue
            if (CUT == "isWtoEN" and not(event.isWtoEN and event.MT<100)): continue
            if (CUT == "isWtoEN_MET" and not(event.isWtoEN and event.MT<100 and event.MEt.pt>200)): continue
            if (CUT == "isWtoMN" and not(event.isWtoMN and event.MT<100)): continue
            if (CUT == "isMN" and not(event.isMN and event.nLeptons==1)): continue
            if (CUT == "isEN" and not(event.isEN and event.nLeptons==1)): continue
            if (CUT == "isWtoMN_MET" and not(event.isWtoMN and event.MT<100 and event.MEt.pt>200)): continue
            if (CUT == "isTtoEM" and not(event.isTtoEM)): continue
            if (CUT == "isSR" and not(event.isSR)): continue
            if (CUT == "isJetHT" and not(event.isJetHT and event.HLT_PFJet500_v and event.MEt.pt<25)): continue
            if (CUT == "isJetMET" and not(event.isJetMET)): continue
            if (CUT == "isDiJetMET" and not(event.isDiJetMET and event.nCHSJetsAcceptanceCalo==2 and event.MinLeadingJetMetDPhi<0.4 and event.MEt.pt<100 and event.HLT_PFJet500_v)): continue
            if (CUT == "isJetMET_unprescaled" and not(event.isJetMET and event.HLT_PFJet500_v)): continue

            #combination single jet
            if (CUT == "isJetMET_dPhi" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_dPhi_MET_200" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue
            if (CUT == "isJetMET_low_dPhi" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_CR" and not(event.isJetMET and event.MinLeadingJetMetDPhi>=0.5 and event.MinLeadingJetMetDPhi<=2 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_500" and not(event.isJetMET and event.MinLeadingJetMetDPhi>=0.5 and event.MinLeadingJetMetDPhi<=2 and (event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_MET_200" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue

            #Lep, to be revised with  single jet trigger
            if (CUT == "isJetMET_dPhi_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and event.nLeptons>0 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_dPhi_MET_200_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and event.nLeptons>0  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue
            if (CUT == "isJetMET_low_dPhi_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and event.nLeptons>0 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_MET_200_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and event.nLeptons>0  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue



            if (REGION=="SR_HEM" and s=="HighMET"):
                #print "SR_HEM, clean from HEM"
                #print "TEST"
                if(event.RunNumber>=319077 and event.nCHSJets_in_HEM>0): continue

            #apply HEM cleaning!
            if(event.RunNumber>=319077 and event.nCHSJets_in_HEM>0): continue

            for j in range(event.nCHSJetsAcceptanceCalo):
                #print event.Jets[j].pt
                if j_idx>-1 and not j==j_idx: continue
                if (event.Jets[j].muEFrac<0.6 and event.Jets[j].eleEFrac<0.6 and event.Jets[j].photonEFrac<0.8 and event.Jets[j].timeRecHitsEB>-1):
                    if eta==True:
                        hist_den[s].Fill(event.Jets[j].eta,event.EventWeight*event.PUReWeight)
                    else:
                        hist_den[s].Fill(event.Jets[j].pt,event.EventWeight*event.PUReWeight)#eta
                    if(check_closure):
                        if(event.Jets[j].sigprob>dnn_threshold and event.Jets[j].sigprob<0.996):
                            if eta==True:
                                hist_num[s].Fill(event.Jets[j].eta,event.EventWeight*event.PUReWeight)#pt                        
                            else:
                                hist_num[s].Fill(event.Jets[j].pt,event.EventWeight*event.PUReWeight)#eta)#pt                        
                    else:
                        if(event.Jets[j].sigprob>0.996):
                            if eta==True:
                                hist_num[s].Fill(event.Jets[j].eta,event.EventWeight*event.PUReWeight)#pt
                            else:
                                hist_num[s].Fill(event.Jets[j].pt,event.EventWeight*event.PUReWeight)#eta)#pt
                    if(event.Jets[j].timeRecHitsEB>0.09 and event.Jets[j].gammaMaxET<0.16 and event.Jets[j].minDeltaRPVTracks>0.06 and event.Jets[j].cHadEFrac<0.06):
                        if eta==True:
                            hist_num_cutbased[s].Fill(event.Jets[j].eta,event.EventWeight*event.PUReWeight)#pt
                        else:
                            hist_num_cutbased[s].Fill(event.Jets[j].pt,event.EventWeight*event.PUReWeight)#eta)#pt
                    n+=1
            if(n_ev%10000==0):
                print ("event n. %d/%d (%.2f perc.)")%(n_ev,chain[s].GetEntries(),100.*float(n_ev)/float(chain[s].GetEntries()))

            #if(n>=1000): break

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(hist_num[s],hist_den[s])
        eff[s].SetMarkerSize(1.)
        eff[s].SetMarkerStyle(markers[i])#(sign_sampl[s]['marker'])
        eff[s].SetMarkerColor(colors[i])#(2)
        eff[s].SetFillColor(colors[i])#(2) 
        eff[s].SetLineColor(colors[i])#(2)
        eff[s].SetLineWidth(2)
        eff[s].GetYaxis().SetRangeUser(0.,maxeff)
        eff[s].GetYaxis().SetTitle("Efficiency")#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleOffset(1.2)#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta==True:
            eff[s].GetXaxis().SetRangeUser(more_bins_eta[0],more_bins_eta[-1])
            eff[s].GetXaxis().SetTitle("Jet #eta")
        else:
            eff[s].GetXaxis().SetRangeUser(more_bins[0],more_bins[-1])
            eff[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")#pt
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.1)

        eff_cutbased[s] = TGraphAsymmErrors()
        eff_cutbased[s].BayesDivide(hist_num_cutbased[s],hist_den[s])

        if eta==True:
            add_label+="_vs_eta"

        can = TCanvas("can","can", 1000, 800)
        can.cd()
        if i==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")
        can.Print(PLOTDIR+"TagEff_"+s+add_label+".png")
        can.Print(PLOTDIR+"TagEff_"+s+add_label+".pdf")

        outfile = TFile(PLOTDIR+"TagEff_"+s+add_label+".root","RECREATE")
        outfile.cd()
        eff[s].Write("eff_"+s)
        eff_cutbased[s].Write("eff_cutbased_"+s)
        hist_den[s].Write("den_"+s)
        hist_num[s].Write("num_"+s)
        hist_num_cutbased[s].Write("num_cutbased_"+s)
        can.Write()
        print "Info in <TFile::Write>: root file "+PLOTDIR+"TagEff_"+s+add_label+".root has been created"
        outfile.Close()
'''

def calculate_tag_eff(tree_weight_dict,sample_list,add_label="",check_closure=False,eta=False,phi=False,j_idx=-1,eta_cut=False,phi_cut=False):

    tree_weights = {}
    chain_entries = {}
    chain_entries_cumulative = {}
    chain = {}
    np_num = {}
    np_den = {}
    np_w_num = {}
    np_w_den = {}
    np_weight = {}
    #print np_more_bins
    #Initialize dictionary of arrays
    '''
    for b in np_more_bins:
        np_den[b] = []#np.array([])
        np_num[b] = []#np.array([])
        np_w_den[b] = []#np.array([])
        np_w_num[b] = []#np.array([])
        np_weight[b] = []
    '''

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut:
        print "Apply acceptance cut |eta|<1."

    if phi_cut:
        print "Apply acceptance cut phi"

    for i, s in enumerate(sample_list):
        if eta==True:
            hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins_eta)-1, more_bins_eta)
            hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins_eta)-1, more_bins_eta)
            hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins_eta)-1, more_bins_eta)
        else:
            if phi==True:
                hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins_phi)-1, more_bins_phi)
                hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins_phi)-1, more_bins_phi)
                hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins_phi)-1, more_bins_phi)
            else:
                hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins)-1, more_bins)
                hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins)-1, more_bins)
                hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins)-1, more_bins)
        hist_den[s].Sumw2()
        hist_num[s].Sumw2()
        hist_num_cutbased[s].Sumw2()
        print s

        passed_num = np.array([])
        passed_den = np.array([])
        w_num = np.array([])
        w_den = np.array([])

        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight",CUT]#"nLeptons"
        if CUT=="isDiJetMET":
            list_of_variables += ["MinSubLeadingJetMetDPhi"]
        if ERA=="2018" and CUT == "isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
                print "Entries per sample ", ss, " : ", chain_entries[l]
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    #cut_mask = np.logical_and(arrays["HLT_PFJet500_v"]==True , arrays["pt"]<25)
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #change!
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #if KILL_QCD:
                    #    cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                    #enhance MET
                    #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                    #no MT
                    #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if CUT == "isSR" and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))
                ###cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))

                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)

                if eta:
                    pt = arrays["Jets.eta"][cut_mask]
                else:
                    if phi:
                        pt = arrays["Jets.phi"][cut_mask]
                    else:
                        pt = arrays["Jets.pt"][cut_mask]
                sigprob = arrays["Jets.sigprob"][cut_mask]

                if phi_cut==True and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi][cut_mask]

                if eta_cut==True and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_eta][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_eta][cut_mask]

                if phi_cut and eta_cut:
                    #print "Pre cuts"
                    #print arrays["Jets.phi"]
                    #print arrays["Jets.eta"]
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi_eta][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi_eta][cut_mask]
                    #print "Post cuts"
                    #print arrays["Jets.phi"][cut_mask_phi_eta][cut_mask]
                    #print arrays["Jets.eta"][cut_mask_phi_eta][cut_mask]
                    #print "Attempt at implementing phi and eta simultaneous cut!!!"
                    #exit()



                tag_mask = (sigprob > dnn_threshold)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                pt_untag = pt[untag_mask]
                pt_tag = pt[tag_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]
                del arrays

                #Global fill
                #Is the pt_tag>0 necessary? It messes up with eta, comment
                #passed_num = np.concatenate( (passed_num, np.hstack(pt_tag[pt_tag>0])) )
                #passed_den = np.concatenate( (passed_den, np.hstack(pt_untag[pt_untag>0])) )
                #w_num = np.concatenate( (w_num, np.hstack( pt_tag[pt_tag>0].astype(bool)*weight ) ) )
                #w_den = np.concatenate( (w_den, np.hstack( pt_untag[pt_untag>0].astype(bool)*weight ) ) )

                passed_num = np.concatenate( (passed_num, np.hstack(pt_tag)) )
                passed_den = np.concatenate( (passed_den, np.hstack(pt_untag)) )
                w_num = np.concatenate( (w_num, np.hstack( pt_tag.astype(bool)*weight ) ) )
                w_den = np.concatenate( (w_den, np.hstack( pt_untag.astype(bool)*weight ) ) )

                #print passed_num.shape
                #print passed_num
                #print w_num.shape
                #print w_num

                '''
                #per-bin fill
                #If pt in certain range, assign specific bin
                for i in range(len(np_more_bins)):
                    if i<len(np_more_bins)-1:
                        passed_den = np.logical_and(pt_untag>=np_more_bins[i],pt_untag<np_more_bins[i+1])*pt_untag
                        w_den = np.logical_and(pt_untag>=np_more_bins[i],pt_untag<np_more_bins[i+1])*weight
                        passed_num = np.logical_and(pt_tag>=np_more_bins[i],pt_tag<np_more_bins[i+1])*pt_tag
                        w_num = np.logical_and(pt_tag>=np_more_bins[i],pt_tag<np_more_bins[i+1])*weight
                    else:
                        passed_den = (pt_untag>=np_more_bins[i])*pt_untag
                        passed_num = (pt_tag>=np_more_bins[i])*pt_tag
                        w_den = (pt_untag>=np_more_bins[i])*weight
                        w_num = (pt_tag>=np_more_bins[i])*weight
                    passed_den = np.hstack(passed_den[passed_den>0])
                    passed_num = np.hstack(passed_num[passed_num>0])
                    w_den = np.hstack(w_den[w_den>0])
                    w_num = np.hstack(w_num[w_num>0])
                    print "bin: ", np_more_bins[i]
                    print "size np_den: ", passed_den.shape
                    print "size np_w_den: ", w_den.shape
                    np_num[ np_more_bins[i] ].append(passed_num)
                    np_den[ np_more_bins[i] ].append(passed_den)
                    np_w_num[ np_more_bins[i] ].append(w_num)
                    np_w_den[ np_more_bins[i] ].append(w_den)

                '''
                #We'll have to flatten and remove unnecessary zeros...
                #print np_den[130.0].shape
                en_it = time.time()
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"

        #Here vectors should be ready
        #print s, np_num, np_w_num
        #print s, np_den, np_w_den
        
        #Fill histo from array
        #print type(np_den[45.0])
        #print np_den[45.0].shape
        #print np_den[45.0]
        #print np.array(np_den[45.0]).shape
        #print np.array(np_den[45.0]).flatten()
        _ = root_numpy.fill_hist( hist_den[s], passed_den, weights=w_den )
        print hist_den[s].Print()
        _ = root_numpy.fill_hist( hist_num[s], passed_num, weights=w_num )
        print hist_num[s].Print()
        
        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(hist_num[s],hist_den[s])
        eff[s].SetMarkerSize(1.)
        eff[s].SetMarkerStyle(markers[i])#(sign_sampl[s]['marker'])
        eff[s].SetMarkerColor(colors[i])#(2)
        eff[s].SetFillColor(colors[i])#(2) 
        eff[s].SetLineColor(colors[i])#(2)
        eff[s].SetLineWidth(2)
        eff[s].GetYaxis().SetRangeUser(0.,maxeff)
        eff[s].GetYaxis().SetTitle("Efficiency")#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleOffset(1.2)#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta==True:
            eff[s].GetXaxis().SetRangeUser(more_bins_eta[0],more_bins_eta[-1])
            eff[s].GetXaxis().SetTitle("Jet #eta")
        else:
            if phi:
                eff[s].GetXaxis().SetRangeUser(more_bins_phi[0],more_bins_phi[-1])
                eff[s].GetXaxis().SetTitle("Jet #varphi")
            else:
                eff[s].GetXaxis().SetRangeUser(more_bins[0],more_bins[-1])
                eff[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")#pt
                eff[s].GetXaxis().SetTitleSize(0.04)
                eff[s].GetXaxis().SetTitleOffset(1.1)

        eff_cutbased[s] = TGraphAsymmErrors()
        eff_cutbased[s].BayesDivide(hist_num_cutbased[s],hist_den[s])


        if eta_cut==True:
            add_label+="_eta_1p0"

        if phi_cut==True:
            add_label+="_phi_cut"

        if eta==True:
            add_label+="_vs_eta"

        if phi==True:
            add_label+="_vs_phi"

        if check_closure:
            add_label+="_closure"+str(dnn_threshold).replace(".","p")
        #else:
        #    add_label+="_wp"+str(dnn_threshold).replace(".","p")

        can = TCanvas("can","can", 1000, 800)
        can.cd()
        if i==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")
            can.Print(PLOTDIR+"TagEff_"+s+add_label+".png")
            can.Print(PLOTDIR+"TagEff_"+s+add_label+".pdf")

        outfile = TFile(PLOTDIR+"TagEff_"+s+add_label+".root","RECREATE")
        outfile.cd()
        eff[s].Write("eff_"+s)
        eff_cutbased[s].Write("eff_cutbased_"+s)
        hist_den[s].Write("den_"+s)
        hist_num[s].Write("num_"+s)
        hist_num_cutbased[s].Write("num_cutbased_"+s)
        can.Write()
        print "Info in <TFile::Write>: root file "+PLOTDIR+"TagEff_"+s+add_label+".root has been created"
        outfile.Close()

'''
def old_draw_tag_eff(sample_list,add_label="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False):
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
    if phi:
        add_label+="_vs_phi"
    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")

    for i, s in enumerate(sample_list):
        infiles = TFile(PLOTDIR+"TagEff_"+s+add_label+".root", "READ")
        hist_den[s] = TH1F()
        hist_num[s] = TH1F()
        graph[s] = TGraphAsymmErrors()
        hist_den[s] = infiles.Get("den_"+s)
        hist_num[s] = infiles.Get("num_"+s)
        #rebin
        if eta:
            less_bins = less_bins_eta
        else:
            if phi:
                less_bins = less_bins_phi
            else:
                less_bins = less_bins_pt
        den = hist_den[s].Rebin(len(less_bins)-1,s+"_den2",less_bins)
        num = hist_num[s].Rebin(len(less_bins)-1,s+"_num2",less_bins)
        graph[s].BayesDivide(num,den)
        eff[s] = TEfficiency(num,den)
        eff[s].SetStatisticOption(TEfficiency.kBBayesian)
        eff[s].SetConfidenceLevel(0.68)
        #maxeff = 1000.#?
        #print(bins)#?
        #hist_den[s].Rebin(len(bins)-1,s+"_den3",bins)#?
        #hist_num[s].Rebin(len(bins)-1,s+"_num3",bins)#?
        #graph[s] = TH1F(hist_den[s])#?
        #graph[s].Rebin(len(more_bins)-1,"was",more_bins)#? 
        #graph[s].Rebin(2)#? 
        graph[s].SetMarkerSize(1.3)
        graph[s].SetMarkerStyle(20)#(sign_sampl[s]['marker'])
        graph[s].SetMarkerColor(samples[s]['linecolor'])#(2)
        graph[s].SetFillColor(samples[s]['linecolor'])#(2) 
        graph[s].SetLineColor(samples[s]['linecolor'])#(2)
        graph[s].SetLineStyle(2)#(2)
        graph[s].SetLineWidth(2)
        graph[s].GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else maxeff)
        #graph[s].GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else 0.7)#Lisa for signal
        graph[s].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
        graph[s].GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        graph[s].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta:
            #graph[s].GetXaxis().SetRangeUser(bins[0],bins[-1])
            graph[s].GetXaxis().SetTitle("Jet #eta")
        else:
            if phi:
                graph[s].GetXaxis().SetTitle("Jet #varphi")
            else:
                graph[s].GetXaxis().SetRangeUser(bins[2],bins[-1])
                graph[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
                can.SetLogx()#?
        graph[s].GetXaxis().SetTitleSize(0.04)
        graph[s].GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(graph[s], samples[s]['label'], "PL")
        can.SetGrid()
        if i==0:
            graph[s].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[s].Draw("P,sames")

        outfile = TFile(PLOTDIR+"TagTEfficiency_"+s+add_label+".root","RECREATE")
        print "Info in <TFile::Write>: TEfficiency root file "+PLOTDIR+"TagTEfficiency_"+s+add_label+".root has been created"
        outfile.cd()
        graph[s].Write("eff_"+s)
        eff[s].Write("TEff_"+s)
        outfile.Close()

    leg.Draw()
    drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    drawCMS_simple(LUMI, "Preliminary", onTop=True)
    can.Print(PLOTDIR+"TagEff_"+s+add_label+".png")
    can.Print(PLOTDIR+"TagEff_"+s+add_label+".pdf")
'''

def draw_tag_eff(sample_dict,reg_label,add_label="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False):
    isMC = False
    if sample_dict == back:
        isMC = True
    if sample_dict == sign:
        isMC = True
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
    if phi:
        add_label+="_vs_phi"
    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
    #else:
    #    add_label+="_wp"+str(dnn_threshold).replace(".","p")


    print sample_dict
    print reg_label
    NEWDIR = PRE_PLOTDIR+reg_label+"/"
    if not os.path.isdir(NEWDIR): os.mkdir(NEWDIR)

    if eta==True:
        hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_eta)-1, more_bins_eta)
        hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_eta)-1, more_bins_eta)
        less_bins = less_bins_eta
    else:
        if phi==True:
            hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_phi)-1, more_bins_phi)
            hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_phi)-1, more_bins_phi)
            less_bins = less_bins_phi
        else:
            hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins)-1, more_bins)
            hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins)-1, more_bins)
            less_bins = less_bins_pt


    for n,k in enumerate(sample_dict.keys()):
        print n,k
        print PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root"
        infiles = TFile(PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root", "READ")
        hist_den[k] = TH1F()
        hist_num[k] = TH1F()
        hist_den[k] = infiles.Get("den_"+sample_dict[k])
        hist_num[k] = infiles.Get("num_"+sample_dict[k])
        print hist_num[k].Print()
        print "now add if needed: ", n
        ##if n==0:
        ##    hnum = hist_num[k]
        ##    hden = hist_den[k]
        ##else:
        ##    print "do i crash here?"
        ##    exit()
        ##    print "hnum pre add ", hnum.Print()
        ##    hnum.Add(hist_num[k],1.)
        ##    print "hnum post add ", hnum.Print()
        ##    hden.Add(hist_den[k],1.)
        hnum.Add(hist_num[k],1.)
        hden.Add(hist_den[k],1.)
        

    graph = TGraphAsymmErrors()
    den = hden.Rebin(len(less_bins)-1,reg_label+"_den2",less_bins)
    num = hnum.Rebin(len(less_bins)-1,reg_label+"_num2",less_bins)
    graph.BayesDivide(num,den)
    eff = TEfficiency(num,den)
    #graph.BayesDivide(hnum,hden)
    #eff = TEfficiency(hnum,hden)
    eff.SetStatisticOption(TEfficiency.kBBayesian)
    eff.SetConfidenceLevel(0.68)
    graph.SetMarkerSize(1.3)
    graph.SetMarkerStyle(20)#(sign_sampl[s]['marker'])
    graph.SetMarkerColor(1)#(samples[s]['linecolor'])#(2)
    graph.SetFillColor(1)#(samples[s]['linecolor'])#(2) 
    graph.SetLineColor(1)#(samples[s]['linecolor'])#(2)
    graph.SetLineStyle(1)#(2)
    graph.SetLineWidth(2)
    graph.GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else maxeff)
    #graph.GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else 0.7)#Lisa for signal
    graph.GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
    graph.GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
    graph.GetYaxis().SetTitleSize(0.05)#DCMS
    if eta:
        #graph.GetXaxis().SetRangeUser(bins[0],bins[-1])
        graph.GetXaxis().SetTitle("Jet #eta")
    else:
        if phi:
            graph.GetXaxis().SetTitle("Jet #varphi")
        else:
            graph.GetXaxis().SetRangeUser(bins[2],bins[-1])
            graph.GetXaxis().SetTitle("Jet p_{T} (GeV)")
            can.SetLogx()#?
    graph.GetXaxis().SetTitleSize(0.04)
    graph.GetXaxis().SetTitleOffset(1.1)
    leg.AddEntry(graph, reg_label, "PL")
    can.SetGrid()
    graph.Draw("AP")


    outfile = TFile(NEWDIR+"TagTEfficiency_"+reg_label+add_label+".root","RECREATE")
    print "Info in <TFile::Write>: TEfficiency root file "+NEWDIR+"TagTEfficiency_"+reg_label+add_label+".root has been created"
    outfile.cd()
    graph.Write("eff_"+reg_label)
    eff.Write("TEff_"+reg_label)
    outfile.Close()

    leg.Draw()
    drawRegion(reg_label,left=True, left_marg_CMS=0.2, top=0.8)
    drawCMS_simple(LUMI, "Preliminary", onTop=True)
    can.Print(NEWDIR+"TagEff_"+reg_label+add_label+".png")
    can.Print(NEWDIR+"TagEff_"+reg_label+add_label+".pdf")

    outfile_name_check = NEWDIR+"TagEff_"+reg_label+add_label+".root"
    if not os.path.isfile(outfile_name_check):
        outfile_2 = TFile(outfile_name_check,"RECREATE")
        outfile_2.cd()
        eff.Write("eff_"+reg_label)
        hden.Write("den_"+reg_label)
        hnum.Write("num_"+reg_label)
        can.Write()
        print "Info in <TFile::Write>: root file "+outfile_name_check+" has been created"
        outfile_2.Close()


def draw_tag_eff_updated(sample_dict,reg_label,add_label="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False):
    isMC = False
    if sample_dict == back:
        isMC = True
    if sample_dict == sign:
        isMC = True
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
    if phi:
        add_label+="_vs_phi"
    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
    #else:
    #    add_label+="_wp"+str(dnn_threshold).replace(".","p")


    print sample_dict
    print reg_label
    NEWDIR = PRE_PLOTDIR+reg_label+"/"
    if not os.path.isdir(NEWDIR): os.mkdir(NEWDIR)

    for n,k in enumerate(sample_dict.keys()):
        if eta==True:
            hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_eta)-1, more_bins_eta)
            hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_eta)-1, more_bins_eta)
            less_bins = less_bins_eta
        else:
            if phi==True:
                hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_phi)-1, more_bins_phi)
                hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_phi)-1, more_bins_phi)
                less_bins = less_bins_phi
            else:
                hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins)-1, more_bins)
                hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins)-1, more_bins)
                less_bins = less_bins_pt

        print PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root"
        infiles = TFile(PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root", "READ")
        hist_den[k] = TH1F()
        hist_num[k] = TH1F()
        hist_den[k] = infiles.Get("den_"+sample_dict[k])
        hist_num[k] = infiles.Get("num_"+sample_dict[k])
        print hist_num[k].Print()
        print "now add if needed: ", n
        ##if n==0:
        ##    hnum = hist_num[k]
        ##    hden = hist_den[k]
        ##else:
        ##    print "do i crash here?"
        ##    exit()
        ##    print "hnum pre add ", hnum.Print()
        ##    hnum.Add(hist_num[k],1.)
        ##    print "hnum post add ", hnum.Print()
        ##    hden.Add(hist_den[k],1.)
        hnum.Add(hist_num[k],1.)
        hden.Add(hist_den[k],1.)
        
        graph = TGraphAsymmErrors()
        den = hden.Rebin(len(less_bins)-1,reg_label+"_den2",less_bins)
        num = hnum.Rebin(len(less_bins)-1,reg_label+"_num2",less_bins)
        graph.BayesDivide(num,den)
        eff = TEfficiency(num,den)
        #graph.BayesDivide(hnum,hden)
        #eff = TEfficiency(hnum,hden)
        eff.SetStatisticOption(TEfficiency.kBBayesian)
        eff.SetConfidenceLevel(0.68)
        graph.SetMarkerSize(1.3)
        graph.SetMarkerStyle(20)#(sign_sampl[s]['marker'])
        graph.SetMarkerColor(1)#(samples[s]['linecolor'])#(2)
        graph.SetFillColor(1)#(samples[s]['linecolor'])#(2) 
        graph.SetLineColor(1)#(samples[s]['linecolor'])#(2)
        graph.SetLineStyle(1)#(2)
        graph.SetLineWidth(2)
        graph.GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else maxeff)
        #graph.GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else 0.7)#Lisa for signal
        graph.GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
        graph.GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        graph.GetYaxis().SetTitleSize(0.05)#DCMS
        if eta:
            #graph.GetXaxis().SetRangeUser(bins[0],bins[-1])
            graph.GetXaxis().SetTitle("Jet #eta")
        else:
            if phi:
                graph.GetXaxis().SetTitle("Jet #varphi")
            else:
                graph.GetXaxis().SetRangeUser(bins[2],bins[-1])
                graph.GetXaxis().SetTitle("Jet p_{T} (GeV)")
                can.SetLogx()#?
        graph.GetXaxis().SetTitleSize(0.04)
        graph.GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(graph, reg_label, "PL")
        can.SetGrid()
        graph.Draw("AP")


        outfile = TFile(NEWDIR+"TagTEfficiency_"+sample_dict[k]+"_"+reg_label+add_label+".root","RECREATE")
        print "Info in <TFile::Write>: TEfficiency root file "+NEWDIR+"TagTEfficiency_"+sample_dict[k]+"_"+reg_label+add_label+".root has been created"
        outfile.cd()
        graph.Write("eff_"+reg_label)
        eff.Write("TEff_"+reg_label)
        outfile.Close()

        leg.Draw()
        drawRegion(reg_label,left=True, left_marg_CMS=0.2, top=0.8)
        drawCMS_simple(LUMI, "Preliminary", onTop=True)
        can.Print(NEWDIR+"TagEff_"+sample_dict[k]+"_"+reg_label+add_label+".png")
        can.Print(NEWDIR+"TagEff_"+sample_dict[k]+"_"+reg_label+add_label+".pdf")

        outfile_name_check = NEWDIR+"TagEff_"+sample_dict[k]+"_"+reg_label+add_label+".root"
        if not os.path.isfile(outfile_name_check):
            outfile_2 = TFile(outfile_name_check,"RECREATE")
            outfile_2.cd()
            eff.Write("eff_"+reg_label)
            hden.Write("den_"+reg_label)
            hnum.Write("num_"+reg_label)
            can.Write()
            print "Info in <TFile::Write>: root file "+outfile_name_check+" has been created"
            outfile_2.Close()


def draw_tag_eff_cutbased(sample_list,add_label=""):
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.035)
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)
    can.SetLogx()#?
    
    for i, s in enumerate(sample_list):
        infiles = TFile(PLOTDIR+"TagEff_"+s+add_label+".root", "READ")
        hist_den[s] = TH1F()
        hist_num_cutbased[s] = TH1F()
        graph[s] = TGraphAsymmErrors()
        hist_den[s] = infiles.Get("den_"+s)
        hist_num_cutbased[s] = infiles.Get("num_cutbased_"+s)
        #rebin
        den = hist_den[s].Rebin(len(less_bins)-1,s+"_den2",less_bins)
        num = hist_num_cutbased[s].Rebin(len(less_bins)-1,s+"_num2",less_bins)
        graph[s].BayesDivide(num,den)
        eff[s] = TEfficiency(num,den)
        eff[s].SetStatisticOption(TEfficiency.kBBayesian)
        eff[s].SetConfidenceLevel(0.68)
        #maxeff = 1000.#?
        #print(bins)#?
        #hist_den[s].Rebin(len(bins)-1,s+"_den3",bins)#?
        #hist_num[s].Rebin(len(bins)-1,s+"_num3",bins)#?
        #graph[s] = TH1F(hist_den[s])#?
        #graph[s].Rebin(len(more_bins)-1,"was",more_bins)#? 
        #graph[s].Rebin(2)#? 
        graph[s].SetMarkerSize(1.3)
        graph[s].SetMarkerStyle(21)#(sign_sampl[s]['marker'])
        graph[s].SetMarkerColor(samples[s]['fillcolor'])#(2)
        graph[s].SetFillColor(samples[s]['fillcolor'])#(2) 
        graph[s].SetLineColor(samples[s]['linecolor'])#(2)
        graph[s].SetLineStyle(2)#(2)
        graph[s].SetLineWidth(2)
        graph[s].GetYaxis().SetRangeUser(-0.0001,maxeff)
        graph[s].GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
        graph[s].GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        graph[s].GetYaxis().SetTitleSize(0.05)#DCMS
        graph[s].GetXaxis().SetRangeUser(bins[2],bins[-1])
        graph[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[s].GetXaxis().SetTitleSize(0.04)
        graph[s].GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(graph[s], samples[s]['label'], "PL")
        can.SetGrid()
        if i==0:
            graph[s].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[s].Draw("P,sames")

        outfile = TFile(PLOTDIR+"TagTEfficiency_cutbased_"+s+add_label+".root","RECREATE")
        print "Info in <TFile::Write>: TEfficiency root file "+PLOTDIR+"TagTEfficiency_cutbased_"+s+add_label+".root has been created"
        outfile.cd()
        graph[s].Write("eff_cutbased_"+s)
        eff[s].Write("TEff_cutbased_"+s)
        outfile.Close()

    leg.Draw()
    drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    drawCMS_simple(LUMI, "Preliminary", True)
    can.Print(PLOTDIR+"TagEff_cutbased"+add_label+".png")
    can.Print(PLOTDIR+"TagEff_cutbased"+add_label+".pdf")


def draw_data_combination(era,regions,regions_labels=[],datasets=[],add_label="",lab_2="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False,do_ratio=False):

    label_dict = {}
    label_dict["ZtoLL"] = "Z #rightarrow ll"
    label_dict["WtoLN"] = "W #rightarrow l#nu"
    label_dict["WtoLN_MET"] = "W #rightarrow l#nu + MET"
    label_dict["JetHT"] = "QCD"
    label_dict["TtoEM"] = "ttbar e + #mu"

    BASEDIR = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_combination/"
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1800, 800)#very horizontal
    can = TCanvas("can","can", 800, 800)#squared
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    #leg = TLegend(0.2, 0.6, 0.9, 0.9)#y: 0.6, 0.9
    if eta:
        leg = TLegend(0.7, 0.7-0.07, 1.0-0.05, 1.0-0.07)
    else:
        leg = TLegend(0.7, 0.7-0.07, 1.0-0.05, 1.0-0.07)#(0.6, 0.7, 1.0, 1.0)
    leg.SetTextSize(0.035)#very horizontal
    leg.SetTextSize(0.025)#squared
    leg.SetTextSize(0.03)#squared
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
        less_bins_plot = less_bins_eta
    else:
        can.SetLogx()#?
        less_bins_plot = less_bins_pt
    
    for i, r in enumerate(regions):
        if r=="ZtoMM" or r=="WtoMN" or r=="WtoMN_MET" or r=="MN" or r=="WtoMN_noMT":
            s = "SingleMuon"
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET" or r=="EN" or r=="WtoEN_noMT":
            if era=="2018":
                s = "EGamma"
            else:
                s = "SingleElectron"
        elif r=="TtoEM":
            s = "MuonEG"
        elif r=="JetHT":
            s = "JetHT"
        elif r=="JetMET":
            s = "JetHT"
        elif r=="DiJetMET":
            s = "JetHT"
        elif r=="JetMET_unprescaled":
            s = "JetHT"
        elif r=="JetHT_unprescaled":
            s = "JetHT"
        elif r=="SR":
            s = "MET"
            #s = "HighMET"
        elif r=="MR":
            s = "HighMET"
        elif r=="MRPho":
            s = "HighMET"
        elif r=="SR_HEM":
            s = "HighMET"
        elif r=="JetMET_all_triggers":
            s="JetHT"
        elif r=="JetMET_unprescaled_trigger":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_all_triggers":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_all_triggers":
            s="JetHT"
        elif r=="JetMET_MET_200_all_triggers":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_all_triggers":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet40":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet60":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet80":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet140":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet200":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet500":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_HLT_PFJet500":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_Lep":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_MET_200_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_MET_200":
            s="JetHT"
        elif "ZtoLL" in r or "WtoLN" in r:
            s=r
        else:
            print "Invalid region, aborting..."
            exit()
        if len(datasets)>0:
            if datasets[i]!="":
                s = datasets[i]
        INPDIR  = BASEDIR + r + "/"#"_CR/"
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        
        colors_2016 = colors#[1,4,801,856] + [881, 798, 602, 921]

        print "Opening this file: ", INPDIR+"TagEff_"+s+reg_label+add_label+".root"
        infile = TFile(INPDIR+"TagEff_"+s+reg_label+add_label+".root", "READ")
        hist_den[r+reg_label+s] = TH1F()
        hist_num[r+reg_label+s] = TH1F()
        graph[r+reg_label+s] = TGraphAsymmErrors()
        hist_den[r+reg_label+s] = infile.Get("den_"+s)
        hist_num[r+reg_label+s] = infile.Get("num_"+s)
        #rebin
        #less_bins = bins#!
        den = hist_den[r+reg_label+s].Rebin(len(less_bins_plot)-1,r+reg_label+s+"_den2",less_bins_plot)
        num = hist_num[r+reg_label+s].Rebin(len(less_bins_plot)-1,r+reg_label+s+"_num2",less_bins_plot)
        graph[r+reg_label+s].BayesDivide(num,den)
        eff[r+reg_label+s] = TEfficiency(num,den)
        eff[r+reg_label+s].SetStatisticOption(TEfficiency.kBBayesian)
        eff[r+reg_label+s].SetConfidenceLevel(0.68)
        graph[r+reg_label+s].SetMarkerSize(marker_sizes[i])#(1.3)
        graph[r+reg_label+s].SetMarkerStyle(markers[i])#(21)#(sign_sampl[s]['marker'])
        if ERA=="2016" and "B-F" in reg_label:
            graph[r+reg_label+s].SetMarkerColor(colors_2016[i])#(samples[s]['fillcolor'])#(2)
            graph[r+reg_label+s].SetFillColor(colors_2016[i])#(samples[s]['fillcolor'])#(2) 
            graph[r+reg_label+s].SetLineColor(colors_2016[i])#(samples[s]['linecolor'])#(2)
        else:
            graph[r+reg_label+s].SetMarkerColor(colors[i])#(samples[s]['fillcolor'])#(2)
            graph[r+reg_label+s].SetFillColor(colors[i])#(samples[s]['fillcolor'])#(2) 
            graph[r+reg_label+s].SetLineColor(colors[i])#(samples[s]['linecolor'])#(2)
        graph[r+reg_label+s].SetLineStyle(lines[i])#(2)#(2)
        graph[r+reg_label+s].SetLineWidth(2)
        graph[r+reg_label+s].GetYaxis().SetRangeUser(-0.0001,0.002 if check_closure else maxeff)
        graph[r+reg_label+s].GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
        graph[r+reg_label+s].GetYaxis().SetTitleOffset(1.5)#("Efficiency (L1+HLT)")
        graph[r+reg_label+s].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta:
            graph[r+reg_label+s].GetXaxis().SetTitle("Jet #eta")
        else:
            graph[r+reg_label+s].GetXaxis().SetRangeUser(bins[4],bins[-1])
            graph[r+reg_label+s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[r+reg_label+s].GetXaxis().SetTitleSize(0.04)
        graph[r+reg_label+s].GetXaxis().SetTitleOffset(1.1)
        #leg.AddEntry(graph[r+reg_label+s], samples[s]['label']+"; "+r+reg_label, "PL")
        #leg.AddEntry(graph[r+reg_label+s], samples[s]['label']+"; "+r, "PL")
        leg.AddEntry(graph[r+reg_label+s], label_dict[r], "PL")
        can.SetGrid()
        if i==0:
            graph[r+reg_label+s].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[r+reg_label+s].Draw("P,sames")
        infile.Close()

    leg.Draw()
    #drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    era_lab = ERA
    if "B-F" in reg_label:
        era_lab+= " B-F"
    elif "G-H" in reg_label:
        era_lab+= " G-H"
    drawCMS_simple(LUMI, "Preliminary", ERA=era_lab, onTop=True)
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".png")
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".pdf")
    #can.Print(OUTDIR+"buh.pdf")

    #for r in regions:
    #    print r
    #    #print eff[r].Print()
   
    if not do_ratio:
        exit()
    keys = graph.keys()
    ratio = {}
    basis = ""
    for k in keys:
        if "WtoLN" in k:
            basis = k
            np = graph[basis].GetN()
            print np

    print basis
    colorsr = [1,4,418]
    markersr = [20,25,24,20,24,24,24,24]
    new_den = {}
    new_num = {}
    n=0
    for k in keys:
        if "WtoLN" not in k:
            print k
            ratio[k] = TGraphAsymmErrors()
            #print graph[k].Print()
            for i in range(0,np):
                #print i
                #print graph[k].GetPointX(i)
                #print graph[k].GetPointY(i)
                ratio[k].SetPoint(i,graph[k].GetPointX(i),graph[k].GetPointY(i)/graph[basis].GetPointY(i))
                ex = abs(abs(graph[k].GetPointX(i)) - abs(less_bins_plot[i+2]))
                #print less_bins_plot[i+2], graph[k].GetPointX(i)
                #print ex
                ratio[k].SetPointError(i,ex,ex,0,0)
            #print ratio[k].Print()
            if "ZtoLL" in k:
                ratio[k].SetLineColor(1)
                ratio[k].SetMarkerColor(1)
                ratio[k].SetMarkerStyle(20)
            elif "JetHT" in k:
                ratio[k].SetLineColor(418)
                ratio[k].SetLineStyle(2)
                ratio[k].SetMarkerColor(418)
                ratio[k].SetMarkerStyle(25)
            elif "TtoEM" in k:
                ratio[k].SetLineColor(4)
                ratio[k].SetLineStyle(2)
                ratio[k].SetMarkerColor(4)
                ratio[k].SetMarkerStyle(24)
            ratio[k].SetMarkerSize(marker_sizes[n])
            ratio[k].SetLineWidth(2)
            n+=1

    canr = TCanvas("canr","canr", 800, 800)#squared
    #canr.SetRightMargin(0.1)
    canr.SetLeftMargin(0.15)
    canr.cd()
    canr.SetGrid()
    if eta:
        legr = TLegend(0.7, 0.7, 1.0-0.05, 1.0-0.07)
    else:
        legr = TLegend(0.6, 0.7, 1.0, 1.0)
    legr.SetTextSize(0.03)#squared

    n=0
    for k in keys:
        if "WtoLN" not in k:
            print ratio[k]
            legr.AddEntry(ratio[k],label_dict[ k.split("_")[0] ],"PL")
            if n==0:
                ratio[k].Draw("AP")
                ratio[k].SetMinimum(0.)
                ratio[k].SetMaximum(2.)
            else:
                ratio[k].Draw("P,sames")
            n+=1

    legr.Draw()
    lineX = TLine(-1,1,1,1)
    lineX.SetLineStyle(2)
    lineX.SetLineWidth(3)
    lineX.SetLineColor(1)
    lineX.Draw("same")
    drawCMS_simple(LUMI, "Preliminary", ERA=era_lab, onTop=True)
    canr.Print(OUTDIR+"RatioTagEffCombiData_"+era+add_label+lab_2+".png")
    canr.Print(OUTDIR+"RatioTagEffCombiData_"+era+add_label+lab_2+".pdf")

def draw_MC_combination(era,sample_list,r,region_label,datasets=[],add_label="",lab_2="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False,do_ratio=False):

    label_dict = {}
    label_dict["SR"] = "SR"
    label_dict["ZtoLL"] = "Z #rightarrow ll"
    label_dict["WtoLN"] = "W #rightarrow l#nu"
    label_dict["WtoLN_MET"] = "W #rightarrow l#nu + MET"
    label_dict["JetHT"] = "QCD"
    label_dict["TtoEM"] = "ttbar e + #mu"

    BASEDIR = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_combination/"
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 800, 800)#squared
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    can.SetGrid()
    leg = TLegend(0.6, 0.7-0.07, 1.0-0.05, 1.0-0.07)
    leg.SetTextSize(0.035)#very horizontal
    leg.SetTextSize(0.025)#squared
    leg.SetTextSize(0.03)#squared
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
        less_bins_plot = less_bins_eta
    else:
        can.SetLogx()#?
        less_bins_plot = less_bins_pt
    
    INPDIR  = BASEDIR + r + "/"#"_CR/"

    print region_label
    for i, s in enumerate(sample_list):
        print i, s
        print "Opening this file: ", INPDIR+"TagEff_"+s+"_"+r+region_label+add_label+".root"
        infile = TFile(INPDIR+"TagEff_"+s+"_"+r+region_label+add_label+".root", "READ")
        hist_den[s+r] = TH1F()
        hist_num[s+r] = TH1F()
        graph[s+r] = TGraphAsymmErrors()
        hist_den[s+r] = infile.Get("den_"+r)
        hist_num[s+r] = infile.Get("num_"+r)
        den = hist_den[s+r].Rebin(len(less_bins_plot)-1,s+r+"_den2",less_bins_plot)
        num = hist_num[s+r].Rebin(len(less_bins_plot)-1,s+r+"_num2",less_bins_plot)
        graph[s+r].BayesDivide(num,den)
        eff[s+r] = TEfficiency(num,den)
        eff[s+r].SetStatisticOption(TEfficiency.kBBayesian)
        eff[s+r].SetConfidenceLevel(0.68)
        graph[s+r].SetMarkerSize(marker_sizes[i])#(1.3)
        graph[s+r].SetMarkerStyle(markers_MC[i])#(21)#(sign_sampl[s]['marker'])
        graph[s+r].SetMarkerColor(samples[s]['fillcolor'])#(colors[i])#
        graph[s+r].SetFillColor(samples[s]['fillcolor'])# (colors[i])#
        graph[s+r].SetLineColor(samples[s]['linecolor'])#(colors[i])#
        graph[s+r].SetLineStyle(lines[i])#(2)#(2)
        graph[s+r].SetLineWidth(2)
        graph[s+r].GetYaxis().SetRangeUser(-0.0001,0.002 if check_closure else maxeff)
        graph[s+r].GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
        graph[s+r].GetYaxis().SetTitleOffset(1.5)#("Efficiency (L1+HLT)")
        graph[s+r].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta:
            graph[s+r].GetXaxis().SetTitle("Jet #eta")
        else:
            graph[s+r].GetXaxis().SetRangeUser(bins[4],bins[-1])
            graph[s+r].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[s+r].GetXaxis().SetTitleSize(0.04)
        graph[s+r].GetXaxis().SetTitleOffset(1.1)
        #leg.AddEntry(graph[s+r], samples[s]['label']+"; "+r, "PL")
        #leg.AddEntry(graph[s+r], samples[s]['label']+"; "+r, "PL")
        leg.AddEntry(graph[s+r], label_dict[r]+";"+s, "PL")
        if i==0:
            graph[s+r].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[s+r].Draw("P,sames")
        infile.Close()
    leg.Draw()
    #drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    era_lab = ERA
    if "B-F" in region_label:
        era_lab+= " B-F"
    elif "G-H" in region_label:
        era_lab+= " G-H"
    drawCMS_simple(LUMI, "Preliminary", ERA=era_lab, onTop=True)
    can.Print(OUTDIR+"TagEffCombiMC_"+era+add_label+lab_2+".png")
    can.Print(OUTDIR+"TagEffCombiMC_"+era+add_label+lab_2+".pdf")
    #can.Print(OUTDIR+"buh.pdf")

def draw_comparison(era,regions,extra="",col=0,maxeff=maxeff):
    BASEDIR = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency_AN/v5_calo_AOD_"+era+"_combination/"
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.5, 0.6, 0.9, 0.9)
    leg.SetTextSize(0.035)
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)
    can.SetLogx()#?
    
    for i, r in enumerate(regions):
        if "JetHT" in r  or "JetMET" in r:
            print r
            print "WEEEE"
            s="JetHT"
        if r=="ZtoMM" or r=="WtoMN" or r=="WtoMN_MET" or r=="MN":
            s = "SingleMuon"
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET" or r=="EN" or r=="WtoEN_noMT":
            if era=="2018":
                s = "EGamma"
            else:
                s = "SingleElectron"
        elif r=="TtoEM":
            s = "MuonEG"
        elif r=="JetHT":
            s = "JetHT"
        elif r=="JetMET":
            s = "JetHT"
        elif r=="JetMET_unprescaled":
            s = "JetHT"
        elif r=="JetHT_unprescaled":
            s = "JetHT"
        elif r=="SR":
            s = "MET"
        elif r=="SR_HEM":
            s = "HighMET"
        elif r=="JetMET_all_triggers":
            s="JetHT"
        elif r=="JetMET_unprescaled_trigger":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_all_triggers":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_all_triggers":
            s="JetHT"
        elif r=="JetMET_MET_200_all_triggers":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_all_triggers":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet40":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet60":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet80":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet140":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet200":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_HLT_PFJet500":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_HLT_PFJet500":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_200_Lep":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_MET_Lep":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5":
            s="JetHT"
        elif r=="JetMET_dPhi_1p5_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_MET_200_Lep":
            s="JetHT"
        elif r=="JetMET_low_dPhi_1p5_MET_200":
            s="JetHT"

        else:
            print "Invalid region, aborting..."
            exit()
        INPDIR  = BASEDIR + r + "/"#"_CR/"
        print "Opening this file: ", INPDIR+"TagEff_"+s+".root"
        infile = TFile(INPDIR+"TagEff_"+s+".root", "READ")
        hist_den[r] = TH1F()
        hist_num[r] = TH1F()
        #graph[r] = TGraphAsymmErrors()
        hist_den[r] = infile.Get("den_"+s)
        hist_num[r] = infile.Get("num_"+s)
        #rebin
        den = hist_den[r].Rebin(len(less_bins)-1,r+"_den2",less_bins)
        num = hist_num[r].Rebin(len(less_bins)-1,r+"_num2",less_bins)

        infile_jj = TFile(OUTDIR+"jj.root","READ")
        eff_jj = infile_jj.Get(r)

        eff[r] = TEfficiency(num,den)
        eff[r].SetStatisticOption(TEfficiency.kBBayesian)
        eff[r].SetConfidenceLevel(0.68)
        eff[r].SetMarkerSize(marker_sizes[i])#(1.3)
        eff[r].SetMarkerStyle(markers[i])#(21)#(sign_sampl[s]['marker'])
        eff[r].SetMarkerColor(colors[i] if col==0 else col)#(samples[s]['fillcolor'])#(2)
        eff[r].SetFillColor(colors[i] if col==0 else col)#(samples[s]['fillcolor'])#(2) 
        eff[r].SetLineColor(colors[i] if col==0 else col)#(samples[s]['linecolor'])#(2)
        eff[r].SetLineStyle(lines[i])#(2)#(2)
        eff[r].SetLineWidth(2)

        eff_jj.SetMarkerColor(colors[i])
        eff_jj.SetMarkerSize(marker_sizes[i])#(1.3)
        eff_jj.SetMarkerStyle(25)#(21)#(sign_sampl[s]['marker'])
        eff_jj.SetMarkerColor(colors[i] if col==0 else col)#(samples[s]['fillcolor'])#(2)
        eff_jj.SetFillColor(colors[i] if col==0 else col)#(samples[s]['fillcolor'])#(2) 
        eff_jj.SetLineColor(colors[i] if col==0 else col)#(samples[s]['linecolor'])#(2)
        eff_jj.SetLineStyle(lines[i])#(2)#(2)
        eff_jj.SetLineWidth(2)

        #eff[r].GetYaxis().SetRangeUser(-0.0001,maxeff)
        #eff[r].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
        #eff[r].GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        #eff[r].GetYaxis().SetTitleSize(0.05)#DCMS
        #eff[r].GetXaxis().SetRangeUser(bins[4],bins[-1])
        #eff[r].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        #eff[r].GetXaxis().SetTitleSize(0.04)
        #eff[r].GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(eff[r], samples[s]['label']+"; "+r, "PL")
        leg.AddEntry(eff_jj, samples[s]['label']+"; Jiajing", "PL")
        can.SetGrid()
        if i==0:
            eff[r].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            eff[r].Draw("P,sames")
        eff_jj.Draw("P,sames")
        gPad.Update()
        graph = eff[r].GetPaintedGraph()
        graph.SetMinimum(-0.0001)
        graph.SetMaximum(maxeff) 
        gPad.Update()
    leg.Draw()
    #drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    #drawCMS_simple(LUMI, "Preliminary", True)
    can.Print(OUTDIR+"TagEffCombiData_compare_"+era+extra+".png")
    can.Print(OUTDIR+"TagEffCombiData_compare_"+era+extra+".pdf")


def get_tree_weights(sample_list,LUMI):

    tree_w_dict = defaultdict(dict)
    for i, s in enumerate(sample_list):
        for l, ss in enumerate(samples[s]['files']):
            #Tree weight
            if ('Run201') in ss:
                t_w = 1.
            else:
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                nevents = filename.Get("c_nEvents").GetBinContent(1)
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    print "SUSY central, consider sample dictionary!"
                    nevents = sample[ss]['nevents']
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
                elif ('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('TChiHH') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
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
    
'''
def very_old_background_prediction_very_old(tree_weight_dict,sample_list):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"
    print "Reading input root files: ", NTUPLEDIR
    tag_bins = np.array([0,1,2,3,4,5,6])
    chain = {}
    hist = {}
    df = {}
    h0 = {}
    h1 = {}
    TEff = {}

    table_yield =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err'])
    table_pred =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
    table_integral =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err'])

    for i, s in enumerate(sample_list):
        ##Prepare efficiency graph
        infiles = TFile(PLOTDIR+"TagTEfficiency_"+s+".root", "READ")
        TEff[s] = infiles.Get("TEff_"+s)

        ##Define TH1F as a cross-check for event yields
        h0[s] = TH1F(s+"_0", s+"_0", 2, 0, 1)
        h0[s].Sumw2()
        h1[s] = TH1F(s+"_1", s+"_1", 2, 0, 1)
        h1[s].Sumw2()

        print s

        ##Prepare TChain and determine tree weights
        ##Note: if you open the same file twice, project doesn't work
        ##Tree weight  must be determined in a separate function
        tree_weights = {}
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")

        print("Entries in chain: %d")%(chain[s].GetEntries())

        max_n=chain[s].GetEntries()+10#100000
        #ev_weight = "(1/"+str(tree_weights[...])+")"
        #ev_weight = "1"
        ev_weight  = "EventWeight*PUReWeight"
        chain[s].Project(s+"_0", "isMC",ev_weight+"*(nTagJets_0p996_JJ==0 && nCHSJetsAcceptanceCalo>0)","",max_n)
        chain[s].Project(s+"_1", "isMC",ev_weight+"*(nTagJets_0p996_JJ==1)","",max_n)
        print "h0 entries: ", h0[s].GetEntries()
        print "h1 entries: ", h1[s].GetEntries()

        ##Here: RDataFrame working example. Requires to load objects from ROOT
        ##https://nbviewer.jupyter.org/url/root.cern/doc/master/notebooks/df026_AsNumpyArrays.py.nbconvert.ipynb
        ##df[s] = RDataFrame(chain[s])
        ##df = df[s].Range(max_n)
        ##npy = df.Filter(CUT+" && nCHSJetsAcceptanceCalo>0").AsNumpy(["MEt","Jets"])#["nCHSJetsAcceptanceCalo", "isMC"])
        ##print(npy["MEt"][0].pt)
        ##print(npy["Jets"][0][0].pt)
        ##dfp = pd.DataFrame(npy)
        ##print(dfp)
        ##exit()

        n=0
        bin0 = []
        bin1 = []
        bin2 = []
        bin1_pred = []
        bin1_pred_up = []
        bin1_pred_low = []
        bin2_pred = []
        bin2_pred_up = []
        bin2_pred_low = []
        bin2_pred_from_1 = []
        bin2_pred_from_1_up = []
        bin2_pred_from_1_low = []
        for event in chain[s]:
            #print "----"
            #print "Event n. ",n
            ##Get the corresponding tree weight from the tree number
            ##https://root.cern.ch/root/roottalk/roottalk07/0595.html
            tree_weight = tree_weights[chain[s].GetTreeNumber()]

            ##CR cuts
            if (CUT == "isZtoMM" and not(event.isZtoMM)): continue
            if (CUT == "isZtoEE" and not(event.isZtoEE)): continue
            if (CUT == "isWtoEN" and not(event.isWtoEN and event.MT<100)): continue
            if (CUT == "isWtoMN" and not(event.isWtoMN and event.MT<100)): continue
            if (CUT == "isTtoEM" and not(event.isTtoEM)): continue
            if (CUT == "isSR" and not(event.isSR)): continue

            n+=1

            if(event.nCHSJetsAcceptanceCalo>0):
                #print ("nTagJets_0p996_JJ\t%d")%(event.nTagJets_0p996_JJ)
                #print "eventn. ", n
                #print "nJets: ", event.nCHSJetsAcceptanceCalo
                n_untagged, n_tagged, EffW, EffWUp, EffWLow = GetEffWeightBin1(event, TEff[s])
                EffW2, EffW2Up, EffW2Low = GetEffWeightBin2(event, TEff[s], n)
                #print "n_untagged debug: ", n_untagged
                #print "n_tagged debug: ", n_tagged
                if (ev_weight=="(1/"+str(tree_weight)+")"):
                    w = 1.
                elif (ev_weight=="1"):
                    w = 1.*tree_weight
                else:
                    w = event.EventWeight*event.PUReWeight*tree_weight

                if(n_tagged==0):
                    bin0.append(w)
                elif(n_tagged==1):
                    bin1.append(w)
                    bin2_pred_from_1.append(EffW*w)
                    bin2_pred_from_1_up.append(EffWUp*w)
                    bin2_pred_from_1_low.append(EffWLow*w)
                else:
                    bin2.append(w)
                bin1_pred.append(n_untagged*EffW*w)
                bin1_pred_up.append(n_untagged*EffWUp*w)
                bin1_pred_low.append(n_untagged*EffWLow*w)
                bin2_pred.append(n_untagged*EffW2*w)
                bin2_pred_up.append(n_untagged*EffW2Up*w)
                bin2_pred_low.append(n_untagged*EffW2Low*w)

            #if(event.nTagJets_0p996_JJ>0):
            #    print ("nTagJets_0p996_JJ\t%d")%(event.nTagJets_0p996_JJ)
            #    print ("nCHSJetsAcceptanceCalo\t%d")%(event.nCHSJetsAcceptanceCalo)
            #    #print "at event number: ", n
            #if(n_bin1>0 or n_bin0>0):
            #    print "n_bin0\tn_bin1\tpred\tEffW\tEffWUp\tEffWLow"
            #    print ("%d\t%d\t %.3f\t %.4f\t %.4f\t %.4f")%(n_bin0, n_bin1, n_bin0*EffW, EffW, EffWUp, EffWLow
            if n==max_n:
                print "done!"
                break

        #if ev_weight=="(1/"+str(tree_weight)+")":#"1":
        #    y_0 = len(np.array(bin0))
        #    y_1 = len(np.array(bin1))
        #elif ev_weight=="1":
        #    y_0 = len(np.array(bin0))*tree_weight
        #    y_1 = len(np.array(bin1))*tree_weight
        #else:
        #    y_0 = np.sum(np.array(bin0))*tree_weight
        #    y_1 = np.sum(np.array(bin1))*tree_weight
        #    e_0 = math.sqrt( sum(x*x for x in bin0) )*tree_weight
        #    e_1 = math.sqrt( sum(x*x for x in bin1) )*tree_weight
        y_0 = np.sum(np.array(bin0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(bin1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(bin2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in bin0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in bin1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in bin2) )#*tree_weight --> already in w
        pred_1 = np.sum(np.array(bin1_pred))
        e_pred_1 = math.sqrt( sum(x*x for x in bin1_pred) )
        pred_up_1 = np.sum(np.array(bin1_pred_up))
        e_pred_up_1 = math.sqrt( sum(x*x for x in bin1_pred_up) )
        pred_low_1 = np.sum(np.array(bin1_pred_low))
        e_pred_low_1 = math.sqrt( sum(x*x for x in bin1_pred_low) )
        pred_2 = np.sum(np.array(bin2_pred))
        e_pred_2 = math.sqrt( sum(x*x for x in bin2_pred) )
        pred_2_from_1 = np.sum(np.array(bin2_pred_from_1))
        e_pred_2_from_1 = math.sqrt( sum(x*x for x in bin2_pred_from_1) )

        row = [s, round(y_0,2), round(e_0,2), round(y_1,2), round(e_1,2), round(y_2,2), round(e_2,2)]
        table_yield.add_row(row)

        rowP = [s, round(pred_1,2), round(e_pred_1,2), round(pred_2,4), round(e_pred_2,4), round(pred_2_from_1,4), round(e_pred_2_from_1,4)]
        table_pred.add_row(rowP)

        error_0 = c_double()#Double()
        error_1 = c_double()#Double()
        y_0 = h0[s].IntegralAndError(h0[s].GetXaxis().FindBin(0),h0[s].GetXaxis().FindBin(2),error_0,"")
        y_1 = h1[s].IntegralAndError(h1[s].GetXaxis().FindBin(0),h1[s].GetXaxis().FindBin(2),error_1,"")
        rowI = [s, round(y_0,2), round(error_0.value,2), round(y_1,2), round(error_1.value,2)]
        table_integral.add_row(rowI)



    print('\n\n======================= Yields and predictions ==============================')
    print(table_yield)
    print(table_pred)
    print('\n\n======================= From histogram  ==============================')
    print(table_integral)

    wr  = True#False
    if wr:
        with open(PLOTDIR+'BkgPred.txt', 'w') as w:
            w.write('\n\n======================= Yields and predictions ==============================\n')
            w.write(str(table_yield)+"\n")
            w.write(str(table_pred)+"\n")
            w.write('\n\n======================= From histogram  ==============================\n')
            w.write(str(table_integral)+"\n")
        print "Info: tables written in file "+PLOTDIR+"BkgPred.txt"
    else:
        print "NO tables written in file !!!!!!"    
'''
'''
def one_eff_background_prediction_new_one_eff(tree_weight_dict,sample_list,extr_region="",add_label=""):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"

    print "Reading input root files: ", NTUPLEDIR
    EFFDIR = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+ (REGION if extr_region=="" else extr_region )+"/"
    tag_bins = np.array([0,1,2,3,4,5,6])
    chain = {}
    hist = {}
    df = {}
    h0 = {}
    h1 = {}
    #TEff = {}
    results = defaultdict(dict)


    table_yield =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err'])
    table_pred =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
    table_integral =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err'])

    for i, s in enumerate(sample_list):
        ##Prepare efficiency graph
        if extr_region=="":
            eff_name = s
        else:
            if "ZtoMM" in extr_region or "WtoMN" in extr_region:
                eff_name="SingleMuon"
            elif "ZtoEE" in extr_region or "WtoEN" in extr_region:
                if ERA=="2018":
                    eff_name="EGamma"
                else:
                    eff_name="SingleElectron"
            elif "TtoEM" in extr_region:
                eff_name="MuonEG"
            elif "SR" in extr_region:
                eff_name="MET"
            elif "SR_HEM" in extr_region:
                eff_name="HighMET"
        infiles = TFile(EFFDIR+"TagTEfficiency_"+eff_name+".root", "READ")
        TEff = infiles.Get("TEff_"+eff_name)

        ##Define TH1F as a cross-check for event yields
        h0[s] = TH1F(s+"_0", s+"_0", 2, 0, 1)
        h0[s].Sumw2()
        h1[s] = TH1F(s+"_1", s+"_1", 2, 0, 1)
        h1[s].Sumw2()

        print s

        ##Prepare TChain and determine tree weights
        ##Note: if you open the same file twice, project doesn't work
        ##Tree weight  must be determined in a separate function
        tree_weights = {}
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")

        print("Entries in chain: %d")%(chain[s].GetEntries())

        max_n=100#chain[s].GetEntries()+10#100000
        #ev_weight = "(1/"+str(tree_weights[...])+")"
        #ev_weight = "1"
        ev_weight  = "EventWeight*PUReWeight"
        chain[s].Project(s+"_0", "isMC",ev_weight+"*(nTagJets_0p996_JJ==0 && nCHSJetsAcceptanceCalo>0)","",max_n)
        chain[s].Project(s+"_1", "isMC",ev_weight+"*(nTagJets_0p996_JJ==1)","",max_n)
        print "h0 entries: ", h0[s].GetEntries()
        print "h1 entries: ", h1[s].GetEntries()

        ##Here: RDataFrame working example. Requires to load objects from ROOT
        ##https://nbviewer.jupyter.org/url/root.cern/doc/master/notebooks/df026_AsNumpyArrays.py.nbconvert.ipynb
        ##df[s] = RDataFrame(chain[s])
        ##df = df[s].Range(max_n)
        ##npy = df.Filter(CUT+" && nCHSJetsAcceptanceCalo>0").AsNumpy(["MEt","Jets"])#["nCHSJetsAcceptanceCalo", "isMC"])
        ##print(npy["MEt"][0].pt)
        ##print(npy["Jets"][0][0].pt)
        ##dfp = pd.DataFrame(npy)
        ##print(dfp)
        ##exit()

        n=0
        bin0 = []
        bin1 = []
        bin2 = []
        bin1_pred = []
        bin1_pred_up = []
        bin1_pred_low = []
        bin2_pred = []
        bin2_pred_up = []
        bin2_pred_low = []
        bin2_pred_from_1 = []
        bin2_pred_from_1_up = []
        bin2_pred_from_1_low = []
        for event in chain[s]:
            #print "----"
            #print "Event n. ",n
            ##Get the corresponding tree weight from the tree number
            ##https://root.cern.ch/root/roottalk/roottalk07/0595.html
            tree_weight = tree_weights[chain[s].GetTreeNumber()]

            ##CR cuts
            if (CUT == "isZtoMM" and not(event.isZtoMM)): continue
            if (CUT == "isZtoEE" and not(event.isZtoEE)): continue
            if (CUT == "isWtoEN" and not(event.isWtoEN and event.MT<100)): continue
            if (CUT == "isWtoMN" and not(event.isWtoMN and event.MT<100)): continue
            if (CUT == "isTtoEM" and not(event.isTtoEM)): continue
            if (CUT == "isSR" and not(event.isSR)): continue
            if (REGION=="SR_HEM" and s=="HighMET"):
                print "SR_HEM, clean from HEM"
                print "TEST"
                if(event.RunNumber>=319077 and event.nCHSJets_in_HEM>0): continue

            n+=1

            n_j, n_untagged, n_tagged, EffW, EffWUp, EffWLow = GetEffWeightBin1New_one_eff(event, TEff)
            EffW2, EffW2Up, EffW2Low = GetEffWeightBin2New_one_eff(event, TEff, n)
            #print "n_untagged debug: ", n_untagged
            #print "n_tagged debug: ", n_tagged
            if (ev_weight=="(1/"+str(tree_weight)+")"):
                w = 1.
            elif (ev_weight=="1"):
                w = 1.*tree_weight
            else:
                w = event.EventWeight*event.PUReWeight*tree_weight

            if(n_tagged==0 and n_j>0):
                bin0.append(w)
                #bin1_pred.append(w*EffW)#(n_untagged*w)#*EffW*w)
            elif(n_tagged==1):
                bin1.append(w)
                bin2_pred_from_1.append(EffW*w)
                bin2_pred_from_1_up.append(EffWUp*w)
                bin2_pred_from_1_low.append(EffWLow*w)
            elif(n_tagged>1):
                bin2.append(w)

            bin1_pred.append(w*EffW)#(n_untagged*w)#*EffW*w)
            bin1_pred_up.append(EffWUp*w)
            bin1_pred_low.append(EffWLow*w)
            bin2_pred.append(EffW2*w)
            bin2_pred_up.append(EffW2Up*w)
            bin2_pred_low.append(EffW2Low*w)

            #if(event.nTagJets_0p996_JJ>0):
            #    print ("nTagJets_0p996_JJ\t%d")%(event.nTagJets_0p996_JJ)
            #    print ("nCHSJetsAcceptanceCalo\t%d")%(event.nCHSJetsAcceptanceCalo)
            #    #print "at event number: ", n
            #if(n_bin1>0 or n_bin0>0):
            #    print "n_bin0\tn_bin1\tpred\tEffW\tEffWUp\tEffWLow"
            #    print ("%d\t%d\t %.3f\t %.4f\t %.4f\t %.4f")%(n_bin0, n_bin1, n_bin0*EffW, EffW, EffWUp, EffWLow
            if n==max_n:
                print "done!"
                break

        print "Size of bin0: ", len(bin0)
        print "Size of bin1: ", len(bin1)
        print "Size of bin1_pred: ", len(bin1_pred)
        
        #if ev_weight=="(1/"+str(tree_weight)+")":#"1":
        #    y_0 = len(np.array(bin0))
        #    y_1 = len(np.array(bin1))
        #elif ev_weight=="1":
        #    y_0 = len(np.array(bin0))*tree_weight
        #    y_1 = len(np.array(bin1))*tree_weight
        #else:
        #    y_0 = np.sum(np.array(bin0))*tree_weight
        #    y_1 = np.sum(np.array(bin1))*tree_weight
        #    e_0 = math.sqrt( sum(x*x for x in bin0) )*tree_weight
        #    e_1 = math.sqrt( sum(x*x for x in bin1) )*tree_weight
        
        y_0 = np.sum(np.array(bin0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(bin1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(bin2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in bin0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in bin1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in bin2) )#*tree_weight --> already in w
        pred_1 = np.sum(np.array(bin1_pred))
        e_pred_1 = math.sqrt( sum(x*x for x in bin1_pred) )
        pred_up_1 = np.sum(np.array(bin1_pred_up))
        e_pred_up_1 = math.sqrt( sum(x*x for x in bin1_pred_up) )
        pred_low_1 = np.sum(np.array(bin1_pred_low))
        e_pred_low_1 = math.sqrt( sum(x*x for x in bin1_pred_low) )
        pred_2 = np.sum(np.array(bin2_pred))
        e_pred_2 = math.sqrt( sum(x*x for x in bin2_pred) )
        pred_2_from_1 = np.sum(np.array(bin2_pred_from_1))
        e_pred_2_from_1 = math.sqrt( sum(x*x for x in bin2_pred_from_1) )

        results[s]["y_0"] = y_0
        results[s]["e_0"] = e_0
        results[s]["y_1"] = y_1
        results[s]["e_1"] = e_1
        results[s]["y_2"] = y_2
        results[s]["e_2"] = e_2
        results[s]["pred_1"] = pred_1
        results[s]["e_pred_1"] = e_pred_1
        results[s]["pred_2"] = pred_2
        results[s]["e_pred_2"] = e_pred_2
        results[s]["pred_2_from_1"] = pred_2_from_1
        results[s]["e_pred_2_from_1"] = e_pred_2_from_1

        row = [s, round(y_0,2), round(e_0,2), round(y_1,2), round(e_1,2), round(y_2,2), round(e_2,2)]
        table_yield.add_row(row)

        rowP = [s, round(pred_1,2), round(e_pred_1,2), round(pred_2,4), round(e_pred_2,4), round(pred_2_from_1,4), round(e_pred_2_from_1,4)]
        table_pred.add_row(rowP)

        error_0 = c_double()#Double()
        error_1 = c_double()#Double()
        y_0 = h0[s].IntegralAndError(h0[s].GetXaxis().FindBin(0),h0[s].GetXaxis().FindBin(2),error_0,"")
        y_1 = h1[s].IntegralAndError(h1[s].GetXaxis().FindBin(0),h1[s].GetXaxis().FindBin(2),error_1,"")
        rowI = [s, round(y_0,2), round(error_0.value,2), round(y_1,2), round(error_1.value,2)]
        table_integral.add_row(rowI)



    print('\n\n======================= NEW Yields and predictions ==============================')
    print(table_yield)
    print(table_pred)
    #print('\n\n======================= From histogram  ==============================')
    #print(table_integral)

    wr  = True#False
    if wr:
        with open(PLOTDIR+'BkgPred'+add_label+'.txt', 'w') as w:
            w.write('\n\n======================= NEW Yields and predictions ==============================\n')
            w.write(str(table_yield)+"\n")
            w.write(str(table_pred)+"\n")
            w.write('\n\n======================= From histogram  ==============================\n')
            w.write(str(table_integral)+"\n")
            w.close()
        print "Info: tables written in file "+PLOTDIR+"BkgPred"+add_label+".txt"
    else:
        print "NO tables written in file !!!!!!"    

    with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+add_label+".json", "w") as w:
        w.write("results = " + json.dumps(results))
        w.close()
        print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+add_label+".json"

    with open("python/BkgPredResults_"+ERA+"_"+REGION+add_label+".py", "w") as w:
        w.write("#! /usr/bin/env python")
        w.write("results = " + json.dumps(results))
        w.close()
        print "Info: dictionary written in file "+"python/BkgPredResults_"+ERA+"_"+REGION+add_label+".py"

    ##To open a json:
    #a_file = open(PLOTDIR+'BkgPredResults.json', "r")
    #output = a_file.read()
    #print(output)


def new_background_prediction_new(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"

    print "Reading input root files: ", NTUPLEDIR

    #Region-dependent  objects
    EFFDIR = {}
    TEff = {}
    table_yield = {}
    table_pred = {}
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
    infiles = {}

    if extr_regions==[]:
        extr_regions.append(REGION)

    for i,r in enumerate(extr_regions):
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        results[r+reg_label] = {}
        EFFDIR[r+reg_label] = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+ r + "/"
        table_yield[r+reg_label] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err'])
        table_pred[r+reg_label] =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
        table_integral[r+reg_label] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err'])

    for i,r in enumerate(extr_regions):
        if "ZtoMM" in r or "WtoMN" in r or "MN" in r:
            eff_name="SingleMuon"
        elif "ZtoEE" in r or "WtoEN" in r or "EN" in r:
            if ERA=="2018":
                eff_name="EGamma"
            else:
                eff_name="SingleElectron"
        elif "TtoEM" in r:
            eff_name="MuonEG"
        elif "SR" in r:
            eff_name="HighMET"
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

        #HERE?
        if len(datasets)>0:
            if datasets[i]!="":
                eff_name = datasets[i]

        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]

        infiles[r+reg_label] = TFile(EFFDIR[r+reg_label]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root", "READ")
        print "Opening TEff["+r+reg_label+"]  " + EFFDIR[r+reg_label]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root"
        TEff[r+reg_label] = infiles[r+reg_label].Get("TEff_"+eff_name)
        for s in sample_list:
            #Define dictionaries
            results[r+reg_label][s] = {}

    for i, s in enumerate(sample_list):
        ##Define TH1F as a cross-check for event yields
        h0[s] = TH1F(s+"_0", s+"_0", 2, 0, 1)
        h0[s].Sumw2()
        h1[s] = TH1F(s+"_1", s+"_1", 2, 0, 1)
        h1[s].Sumw2()
        h2[s] = TH1F(s+"_2", s+"_2", 2, 0, 1)
        h2[s].Sumw2()

        print s

        ##Prepare TChain and determine tree weights
        ##Note: if you open the same file twice, project doesn't work
        ##Tree weight  must be determined in a separate function
        tree_weights = {}
        tree_weights_array = np.array([])
        chain_entries = {}
        chain_entries_cumulative = {}
        start_chain = time.time()
        chain[s] = TChain("tree")
        list_files_for_uproot = []
        list_files_for_uproot_tree = []
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            #print "Entries in ", ss, " : ", chain[s].GetEntries()
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
            tree_weights_array = np.concatenate( (tree_weights_array,tree_weights[l] * np.ones(chain_entries[l])) )#np.concatenate( tree_weights_array, tree_weights[l] * np.ones(chain_entries[l]))
            #print "Entries cumulative ", chain_entries_cumulative[l]
            #print "Entries per sample ", chain_entries[l]
            #print "Size of tree_weights_array ",  len(tree_weights_array)
            list_files_for_uproot.append(NTUPLEDIR + ss + ".root")#:tree")

        print list_files_for_uproot

        max_n=chain[s].GetEntries()+10#100000

        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight",CUT]#"nLeptons"
        for arrays in uproot.iterate(list_files_for_uproot,"tree",list_of_variables,entrysteps=max_n):#["Jets*"]):#,entrysteps=max_n):
            print "uproot iteration n. ", c
            key_list = arrays.keys()
            array_size_tot+=len(arrays[ key_list[0] ])
            #Add tree weight
            arrays["tree_weight"] = tree_weights[c]*np.ones( len(arrays[ key_list[0] ])  )
            #Pandas df from dict
            tmp = pd.DataFrame.from_dict(arrays)
            print tmp.shape
            if c==0:
                df = tmp
            else:
                df = pd.concat([df,tmp])
            c+=1

        print "Final df: ", df.shape
        print df.columns
        end_uproot = time.time()
        print "Tot size of arrays: ", array_size_tot
        print "Time elapsed to fill uproot array: ", end_uproot-start_uproot
        print "************************************"

        print("Entries in chain: %d")%(chain[s].GetEntries())
        exit()

        #ev_weight = "(1/"+str(tree_weights[...])+")"
        #ev_weight = "1"
        ev_weight  = "EventWeight*PUReWeight"
        tagvar = "nTagJets_0p996_JJ"
        #tagvar = "nTagJets_cutbased"
        print tagvar
        cutstring_0 = ev_weight+"*("+tagvar+"==0)"
        cutstring_1 = ev_weight+"*("+tagvar+"==1)"
        cutstring_2 = ev_weight+"*("+tagvar+">1)"
        if "Wto" in REGION:
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MT<100)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MT<100)"


        #combination single jet
        if CUT == "isJetHT":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MEt.pt<25 && (HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MEt.pt<25 && (HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MEt.pt<25 && (HLT_PFJet500_v))"

        if CUT == "isJetMET_dPhi":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"

        if CUT == "isJetMET_low_dPhi":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"

        if CUT == "isJetMET_low_dPhi_CR":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v))"

        if CUT == "isJetMET_low_dPhi_500":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && ( HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && ( HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=2 && MinLeadingJetMetDPhi>=0.5 && ( HLT_PFJet500_v))"

        if CUT == "isJetMET_dPhi_MET_200":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"

        if CUT == "isJetMET_low_dPhi_MET_200":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200)"

        #Lep:
        if CUT == "isJetMET_dPhi_Lep":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && nLeptons>0)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && nLeptons>0)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v)  && nLeptons>0)"

        if CUT == "isJetMET_low_dPhi_Lep":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && nLeptons>0)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && nLeptons>0)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && nLeptons>0)"

        if CUT == "isJetMET_dPhi_MET_200_Lep":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi>1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"

        if CUT == "isJetMET_low_dPhi_MET_200_Lep":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=1.5 && (HLT_PFJet260_v || HLT_PFJet320_v || HLT_PFJet400_v || HLT_PFJet450_v || HLT_PFJet500_v) && MEt.pt>200 && nLeptons>0)"

        if CUT == "isDiJetMET":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"


        print "CUT: ", CUT
        print "cutstring bin 0: ", cutstring_0
        chain[s].Project(s+"_0", "isMC",cutstring_0,"",max_n)
        chain[s].Project(s+"_1", "isMC",cutstring_1,"",max_n)
        chain[s].Project(s+"_2", "isMC",cutstring_2,"",max_n)
        end_chain = time.time()
        print "Time elapsed to project TChain: ", end_chain-start_chain
        print "************************************"
        print "bin0 entries: ", h0[s].GetEntries()
        print "bin1 entries: ", h1[s].GetEntries()
        print "bin2 entries: ", h2[s].GetEntries()
        #exit()

        ##Here: RDataFrame working example. Requires to load objects from ROOT
        ##https://nbviewer.jupyter.org/url/root.cern/doc/master/notebooks/df026_AsNumpyArrays.py.nbconvert.ipynb
        start_frame = time.time()
        dfR[s] = RDataFrame(chain[s])
        dfR = dfR[s].Range(max_n)
        #npy = df.Filter(CUT+" && nCHSJetsAcceptanceCalo>0").AsNumpy(["Jets"])#["nCHSJetsAcceptanceCalo", "isMC", "MEt"])
        npy = dfR.Filter(CUT).AsNumpy(["nCHSJetsAcceptanceCalo","Jets.pt"])#["nCHSJetsAcceptanceCalo", "isMC", "MEt"])
        print npy["Jets.pt"][1:2][0]
        #exit()
        #print(npy["MEt"][0].pt)
        #print(npy["MEt"][0].phi)
        #print(npy["Jets"][0][0].pt)
        dfp = pd.DataFrame(npy)
        print(dfp[1:2][ ["Jets.pt","nCHSJetsAcceptanceCalo"] ])
        end_frame = time.time()
        print "Time elapsed to fill RDataFrame: ", end_frame-start_frame
        print "************************************"
        exit()

        #Here uproot methods
        ##tree = OrderedDict()
        ##root_dir = uproot.open(home_dir+fnames[tag])
        ##tree[s] = root_dir['tree']
        ##met = tree[s]['MEt.pt'].array()
        exit()


        n=0
        bin0 = []
        bin1 = []
        bin2 = []
        #bin predictions must now be dictionaries, they depend on the extrapolation region
        bin1_pred = defaultdict(dict)
        bin1_pred_up = defaultdict(dict)
        bin1_pred_low = defaultdict(dict)
        bin2_pred = defaultdict(dict)
        bin2_pred_up = defaultdict(dict)
        bin2_pred_low = defaultdict(dict)
        bin2_pred_from_1 = defaultdict(dict)
        bin2_pred_from_1_up = defaultdict(dict)
        bin2_pred_from_1_low = defaultdict(dict)

        #Initialize as empty list each bin prediction
        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
            bin1_pred[r+reg_label] = []
            bin1_pred_up[r+reg_label] = []
            bin1_pred_low[r+reg_label] = []
            bin2_pred[r+reg_label] = []
            bin2_pred_up[r+reg_label] = []
            bin2_pred_low[r+reg_label] = []
            bin2_pred_from_1[r+reg_label] = []
            bin2_pred_from_1_up[r+reg_label] = []
            bin2_pred_from_1_low[r+reg_label] = []

        n_ev_passing = 0
        for event in chain[s]:
            #print "----"
            #print "Event n. ",n
            #print "MT: ", event.MT
            #print "CUT: ", CUT
            n+=1
            ##Get the corresponding tree weight from the tree number
            ##https://root.cern.ch/root/roottalk/roottalk07/0595.html
            tree_weight = tree_weights[chain[s].GetTreeNumber()]

            ##CR cuts
            if (CUT == "isZtoMM" and not(event.isZtoMM)): continue
            if (CUT == "isZtoEE" and not(event.isZtoEE)): continue
            if (CUT == "isWtoEN" and not(event.isWtoEN and event.MT<100)): continue
            if (CUT == "isWtoEN_MET" and not(event.isWtoEN and event.MT<100 and event.MEt.pt>200)): continue
            if (CUT == "isWtoMN" and not(event.isWtoMN and event.MT<100)): continue
            if (CUT == "isWtoMN_MET" and not(event.isWtoMN and event.MT<100 and event.MEt.pt>200)): continue
            if (CUT == "isMN" and not(event.isMN and event.nLeptons==1)): continue
            if (CUT == "isEN" and not(event.isEN and event.nLeptons==1)): continue
            if (CUT == "isTtoEM" and not(event.isTtoEM)): continue
            if (CUT == "isSR" and not(event.isSR)): continue
            if (CUT == "isJetHT" and not(event.isJetHT and event.HLT_PFJet500_v and event.MEt.pt<25)): continue
            if (CUT == "isJetMET" and not(event.isJetMET)): continue
            if (CUT == "isDiJetMET" and not(event.isDiJetMET and event.nCHSJetsAcceptanceCalo==2 and event.MinLeadingJetMetDPhi<0.4 and event.MEt.pt<100 and event.HLT_PFJet500_v)): continue



            #combination single jet
            if (CUT == "isJetMET_dPhi" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_dPhi_MET_200" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue
            if (CUT == "isJetMET_low_dPhi" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_CR" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=2 and event.MinLeadingJetMetDPhi>=0.5  and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_500" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=2 and event.MinLeadingJetMetDPhi>=0.5  and (event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_MET_200" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue

            #Lep, to be revised with  single jet trigger
            if (CUT == "isJetMET_dPhi_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and event.nLeptons>0 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_dPhi_MET_200_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi>1.5 and event.nLeptons>0  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue
            if (CUT == "isJetMET_low_dPhi_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and event.nLeptons>0 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v) )): continue
            if (CUT == "isJetMET_low_dPhi_MET_200_Lep" and not(event.isJetMET and event.MinLeadingJetMetDPhi<=1.5 and event.nLeptons>0  and event.MEt.pt>200 and (event.HLT_PFJet260_v or event.HLT_PFJet320_v or event.HLT_PFJet400_v or event.HLT_PFJet450_v or event.HLT_PFJet500_v))): continue

            if (REGION=="SR_HEM" and s=="HighMET"):
                print "SR_HEM, clean from HEM"
                print "TEST"
                if(event.RunNumber>=319077 and event.nCHSJets_in_HEM>0): continue

            #apply HEM cleaning!
            if(event.RunNumber>=319077 and event.nCHSJets_in_HEM>0): continue

            n_ev_passing+=1

            ##print "Debug: events passing"
            ##print ("Run: %d; Lumi: %d; Event: %d" % (event.RunNumber,event.LumiNumber,event.EventNumber))
            ##print ("nTagJets_0p996_JJ: %d; MinLeadingJetMetDPhi: %f; MEt.pt: %f; HLT_PFJet500_v: %d" % (event.nTagJets_0p996_JJ,event.MinLeadingJetMetDPhi,event.MEt.pt,event.HLT_PFJet500_v))
            #TEff here is a dictionary
            #EffW, EffWUp, EffWLow are also dictionaries
            n_j, n_untagged, n_tagged, EffW, EffWUp, EffWLow = GetEffWeightBin1New(event, TEff, check_closure)
            EffW2, EffW2Up, EffW2Low = GetEffWeightBin2New(event, TEff, n, check_closure)
            #print "n_untagged debug: ", n_untagged
            #print "n_tagged debug: ", n_tagged
            if (ev_weight=="(1/"+str(tree_weight)+")"):
                w = 1.
            elif (ev_weight=="1"):
                w = 1.*tree_weight
            else:
                w = event.EventWeight*event.PUReWeight*tree_weight

            #if(n_tagged==0 and n_j>0):
            if(n_tagged==0):#include events without jets
                bin0.append(w)
                #bin1_pred.append(w*EffW)#(n_untagged*w)#*EffW*w)
            elif(n_tagged==1):
                bin1.append(w)
                for i,r in enumerate(extr_regions):
                    reg_label = ""
                    if len(regions_labels)>0:
                        reg_label = regions_labels[i]
                    bin2_pred_from_1[r+reg_label].append(EffW[r+reg_label]*w)
                    bin2_pred_from_1_up[r+reg_label].append(EffWUp[r+reg_label]*w)
                    bin2_pred_from_1_low[r+reg_label].append(EffWLow[r+reg_label]*w)
            elif(n_tagged>1):
                bin2.append(w)

            for i,r in enumerate(extr_regions):
                reg_label = ""
                if len(regions_labels)>0:
                    reg_label = regions_labels[i]
                bin1_pred[r+reg_label].append(w*EffW[r+reg_label])#(n_untagged*w)#*EffW*w)
                bin1_pred_up[r+reg_label].append(EffWUp[r+reg_label]*w)
                bin1_pred_low[r+reg_label].append(EffWLow[r+reg_label]*w)
                bin2_pred[r+reg_label].append(EffW2[r+reg_label]*w)
                bin2_pred_up[r+reg_label].append(EffW2Up[r+reg_label]*w)
                bin2_pred_low[r+reg_label].append(EffW2Low[r+reg_label]*w)

            #print "DEBUG event ", n , " : efficiencies we got"
            #print EffW, EffW2
            #if(event.nTagJets_0p996_JJ>0):
            #    print ("nTagJets_0p996_JJ\t%d")%(event.nTagJets_0p996_JJ)
            #    print ("nCHSJetsAcceptanceCalo\t%d")%(event.nCHSJetsAcceptanceCalo)
            #    #print "at event number: ", n
            #if(n_bin1>0 or n_bin0>0):
            #    print "n_bin0\tn_bin1\tpred\tEffW\tEffWUp\tEffWLow"
            #    print ("%d\t%d\t %.3f\t %.4f\t %.4f\t %.4f")%(n_bin0, n_bin1, n_bin0*EffW, EffW, EffWUp, EffWLow
            if(n%10000==0):
                print ("event n. %d/%d (%.2f perc.)")%(n,chain[s].GetEntries(),100.*float(n)/float(chain[s].GetEntries()))

            if n>=max_n:
                print "done!"
                break


        print "N. events passing selections: ",  n_ev_passing
        #print "Size of bin0: ", len(bin0)
        #print "Size of bin1: ", len(bin1)
        ##print "Size of bin1_pred: ", len(bin1_pred)
        y_0 = np.sum(np.array(bin0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(bin1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(bin2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in bin0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in bin1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in bin2) )#*tree_weight --> already in w

        error_0_i = c_double()#Double()
        error_1_i = c_double()#Double()
        y_0_i = h0[s].IntegralAndError(h0[s].GetXaxis().FindBin(0),h0[s].GetXaxis().FindBin(2),error_0_i,"")
        y_1_i = h1[s].IntegralAndError(h1[s].GetXaxis().FindBin(0),h1[s].GetXaxis().FindBin(2),error_1_i,"")
        rowI = [s, round(y_0_i,2), round(error_0_i.value,2), round(y_1_i,2), round(error_1_i.value,2)]
        #rowI = [s, round(y_0_i,2), round(error_0_i,2), round(y_1_i,2), round(error_1_i,2)]

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

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]

            pred_1[r+reg_label] = np.sum(np.array(bin1_pred[r+reg_label]))
            e_pred_1[r+reg_label] = math.sqrt( sum(x*x for x in bin1_pred[r+reg_label]) )
            pred_up_1[r+reg_label] = np.sum(np.array(bin1_pred_up[r+reg_label]))
            e_pred_up_1[r+reg_label] = math.sqrt( sum(x*x for x in bin1_pred_up[r+reg_label]) )
            pred_low_1[r+reg_label] = np.sum(np.array(bin1_pred_low[r+reg_label]))
            e_pred_low_1[r+reg_label] = math.sqrt( sum(x*x for x in bin1_pred_low[r+reg_label]) )
            pred_2[r+reg_label] = np.sum(np.array(bin2_pred[r+reg_label]))
            e_pred_2[r+reg_label] = math.sqrt( sum(x*x for x in bin2_pred[r+reg_label]) )
            pred_2_from_1[r+reg_label] = np.sum(np.array(bin2_pred_from_1[r+reg_label]))
            e_pred_2_from_1[r+reg_label] = math.sqrt( sum(x*x for x in bin2_pred_from_1[r+reg_label]) )

            results[r+reg_label][s]["y_0"] = y_0
            results[r+reg_label][s]["e_0"] = e_0
            results[r+reg_label][s]["y_1"] = y_1
            results[r+reg_label][s]["e_1"] = e_1
            results[r+reg_label][s]["y_2"] = y_2
            results[r+reg_label][s]["e_2"] = e_2
            results[r+reg_label][s]["pred_1"] = pred_1[r+reg_label]
            results[r+reg_label][s]["e_pred_1"] = e_pred_1[r+reg_label]
            results[r+reg_label][s]["pred_2"] = pred_2[r+reg_label]
            results[r+reg_label][s]["e_pred_2"] = e_pred_2[r+reg_label]
            results[r+reg_label][s]["pred_2_from_1"] = pred_2_from_1[r+reg_label]
            results[r+reg_label][s]["e_pred_2_from_1"] = e_pred_2_from_1[r+reg_label]

            row[r+reg_label] = [s, round(y_0,2), round(e_0,2), round(y_1,2), round(e_1,2), round(y_2,2), round(e_2,2)]
            table_yield[r+reg_label].add_row(row[r+reg_label])

            rowP[r+reg_label] = [s, round(pred_1[r+reg_label],2), round(e_pred_1[r+reg_label],2), round(pred_2[r+reg_label],4), round(e_pred_2[r+reg_label],4), round(pred_2_from_1[r+reg_label],4), round(e_pred_2_from_1[r+reg_label],4)]
            table_pred[r+reg_label].add_row(rowP[r+reg_label])

            table_integral[r+reg_label].add_row(rowI)


    for i,r in enumerate(extr_regions):
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]

        if i==0:
            print('\n\n======================= From histogram  ==============================')
            print(table_integral[r+reg_label])
        print '\n\n======================= Yields and predictions extrapolated from '+r+reg_label+' ==============================' 
        print(table_yield[r+reg_label])
        print(table_pred[r+reg_label])

    wr  = True#False

    for i,r in enumerate(extr_regions):
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        if wr:
            with open(PLOTDIR+'BkgPred_extr_region_'+r+reg_label+add_label+label_2+'.txt', 'w') as w:
                w.write('\n\n======================= Yields and predictions extrapolated from '+r+' ==============================\n')
                w.write(str(table_yield[r+reg_label])+"\n")
                w.write(str(table_pred[r+reg_label])+"\n")
                w.write('\n\n======================= From histogram  ==============================\n')
                w.write(str(table_integral[r+reg_label])+"\n")
                w.close()
            print "Info: tables written in file "+PLOTDIR+"BkgPred_extr_region_"+r+reg_label+add_label+label_2+".txt"
        else:
            print "NO tables written in file !!!!!!"    

        #Output separated per CR
        #with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ '_extr_region_' +r+  add_label+".json", "w") as w:
        #    w.write("results = " + json.dumps(results[r+reg_label]))
        #    w.close()
        #    print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ '_extr_region_' +r+  add_label+".json"

        #with open("python/BkgPredResults_"+ERA+"_"+REGION+'_extr_region_' +r+add_label+".py", "w") as w:
        #    w.write("#! /usr/bin/env python \n")
        #    w.write("results = " + json.dumps(results[r+reg_label]))
        #    w.close()
        #    print "Info: dictionary written in file "+"python/BkgPredResults_"+ERA+"_"+REGION+'_extr_region_' +r+add_label+".py"

    with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ add_label+label_2+".json", "w") as w:
        w.write("results = " + json.dumps(results))
        w.close()
        print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+add_label+label_2+".json"

    with open("python/BkgPredResults_"+ERA+"_"+REGION+add_label+label_2+".py", "w") as w:
        w.write("#! /usr/bin/env python \n")
        w.write("results = " + json.dumps(results))
        w.close()
        print "Info: dictionary written in file "+"python/BkgPredResults_"+ERA+"_"+REGION+add_label+label_2+".py"

'''
def background_prediction(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,plot_distr="",eta=False,phi=False,eta_cut=False,phi_cut=False):

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

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        #print "Something fishy, to be debugged" --> probably binning problem, 1 out of the range
        #exit()

    '''
    if eta_cut==True:
        if eta==False:
            add_label+="_eta_1p0"
        else:
            label_2+="_eta_1p0"
            #add_label+="_eta_1p0"

    if phi_cut==True:
        if eta_cut==True:
            add_label+="_eta_1p0"
        if phi==False:
            add_label+="_phi_cut"
        else:
            label_2+="_phi_cut"
            #add_label+="_eta_1p0"
    '''

    if eta_cut==True:
        add_label+="_eta_1p0"

    if phi_cut==True:
        add_label+="_phi_cut"

    if eta:
        add_label+="_vs_eta"
        if phi:
            add_label+="_vs_phi"

    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
    #else:
    #    add_label+="_wp"+str(dnn_threshold).replace(".","p")
        
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
        EFFDIR[r+reg_label+dset] = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"+ r + "/"
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
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.phi","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","HLT*PFMETNoMu*",CUT]#"nLeptons"
        if ERA=="2018" and CUT=="isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]
            if plot_distr!="":
                print "Plotting: ", plot_distr
                if plot_distr not in list_of_variables:
                    if "[" in plot_distr:
                        pre = plot_distr.partition("[")[0]
                        post = plot_distr.partition("]")[-1]
                        tmp_var = pre+post
                        list_of_variables.append(tmp_var)
                    elif "MEt.phi" in plot_distr:
                        list_of_variables.append("phi")
                    elif "MEt.pt" in plot_distr:
                        print "MEt.pt already there" 
                    else:
                        list_of_variables.append(plot_distr)


        if CUT=="isDiJetMET":
            list_of_variables += ["MinSubLeadingJetMetDPhi"]

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
        distr_2_t = np.array([])
        weight_0 = np.array([])
        weight_1 = np.array([])
        weight_2 = np.array([])
        weight_1_t = np.array([])
        weight_2_t = np.array([])
        weight_pr_1 = {}
        weight_pr_2_from_0 = {}
        weight_pr_2_from_1 = {}

        #pr_1/2 depend on TEff, dictionaries
        for r in TEff.keys():
            pr_1[r] = np.array([])#[]
            pr_2[r] = np.array([])#[]
            pr_2_from_1[r] = np.array([])#[]

            #plot_distr: here define variables in bins
            weight_pr_1[r] = np.array([])
            weight_pr_2_from_0[r] = np.array([])
            weight_pr_2_from_1[r] = np.array([])

            eff[r] = []
            effUp[r] = []
            effDown[r] = []
            #0. Fill efficiencies np arrays once and forever
            if eta:
                for b in np_bins_eta:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    print(r," eta: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))

            elif eta==False and phi==True:
                for b in np_bins_phi:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    print(r," phi: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))

            else:
                for b in np_bins:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    #print "Making fake pt bins!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    #eff[r].append(0.0005)
                    print(r," pt: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))
                    
            print r, eff[r]
            #exit()

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
                #for k in arrays.keys():
                #    max_arr_size = 10
                #    print "Looking only at max_arr_size of ", max_arr_size
                #    arrays[k] = arrays[k][0:max_arr_size]
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                #Pandas df from dict
                #tmp = pd.DataFrame.from_dict(arrays)
                #tmp["tree_weight"] = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    #cut_mask = np.logical_and(arrays["HLT_PFJet500_v"]==True , arrays["pt"]<25)
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #cut_mask = np.logical_and(arrays["HLT_PFJet500_v"]==True , arrays["pt"]<100, arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinLeadingJetMetDPhi"]<=0.4)
                    #change!
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                        #cut_mask = arrays[CUT]>0
                        #print "!!!! try to kill QCD!!!"
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                        #cut_mask = arrays[CUT]>0
                        #print "!!!! try to kill QCD!!!"
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                        #enhance MET
                        #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        #no MT
                        #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if CUT=="isSR" and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))
                #    ##cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_eta_2p5"]==0)))


                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)



                if phi_cut==True and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi][cut_mask]

                elif eta_cut==True and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_eta][cut_mask]
                        #print "Do we have eta out of range?"
                        #print pt
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_eta][cut_mask]

                elif phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi_eta][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi_eta][cut_mask]

                #print "\n"
                #print "   WARNING! Here just for usefulness, save only non-null jets"
                #print "\n"
                #cut_mask = np.logical_and(cut_mask, arrays["Jets.pt"][cut_mask].counts>0)

                else:
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                runnumber = arrays["RunNumber"][cut_mask]
                mindphi = arrays["MinJetMetDPhi"][cut_mask]
                luminumber = arrays["LumiNumber"][cut_mask]
                eventnumber = arrays["EventNumber"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                pt_v = arrays["Jets.pt"][cut_mask]
                eta_v = arrays["Jets.eta"][cut_mask]
                phi_v = arrays["Jets.phi"][cut_mask]
                score_v = arrays["Jets.sigprob"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*mult_factor
                #Default value, we'll need more masks afterwards
                n_obj = -1


                if plot_distr!="":

                    #print "Still have to implement the eta cut, aborting. . . "
                    #exit()
                    if eta_cut:
                        #cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                        #cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                        #cut_mask = (cut_mask_eta.any()==True)
                        print "cut_mask and cut_mask_eta already known"

                    #plot_distr: here define variables in bins
                    ##First: acceptance cut
                    ##base_var = arrays["Jets.pt"][cut_mask]
                    ##base_var = base_var[ base_var[base_var>0].counts>1 ]
                    ##print base_var[:,0]
                    if "[" in plot_distr:
                        #This includes only the desired jet indices
                        #Rename variable: split name without [N]
                        pre = plot_distr.partition("[")[0]
                        post = plot_distr.partition("]")[-1]
                        tmp_var = pre+post
                        #Retain [N] for selectiong obj multiplicity
                        n_obj = int(plot_distr.replace(pre,"").replace(post,"").replace("[","").replace("]",""))
                        base_var = arrays[tmp_var][cut_mask]
                        base_pt_var = arrays["Jets.pt"][cut_mask]
                        if eta_cut:
                            print "To be checked if it works!"
                            base_var = arrays[tmp_var][cut_mask_eta][cut_mask]
                            base_pt_var = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                            #This mask will depend on jet sorting, to be defined later
                            #n_obj_mask = (base_var.counts > n_obj)
                        else:
                            #These do not depend on cut_eta
                            if plot_distr=="MEt.pt":
                                base_var = arrays["pt"][cut_mask]      
                            elif plot_distr=="MEt.phi":
                                base_var = arrays["phi"][cut_mask]                            
                            else:
                                base_var = arrays[plot_distr][cut_mask]
                                base_pt_var = arrays["Jets.pt"][cut_mask]

                del arrays
                
                #dnn_threshold = 0.996
                tag_mask = (sigprob > dnn_threshold)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                pt_untag = pt[untag_mask]
                #test  = (tag_mask.any() == True)#at least one tag
                
                #Translate to dataframe only for debugging purposes
                #df_all = pd.DataFrame.from_dict(arrays)
                #df_all["tag_mask"] = tag_mask
                #df_all["n_tag"] = sigprob[tag_mask].counts

                bin0_m = (sigprob[tag_mask].counts ==0)
                bin1_m = (sigprob[tag_mask].counts ==1)
                bin2_m = (sigprob[tag_mask].counts >1)
                runnumber = runnumber[bin2_m]
                mindphi = mindphi[bin2_m]
                luminumber = luminumber[bin2_m]
                eventnumber = eventnumber[bin2_m]
                pt_v = pt_v[bin2_m]
                eta_v = eta_v[bin2_m]
                phi_v = phi_v[bin2_m]
                score_v = score_v[bin2_m]
                bin0 = np.multiply(bin0_m,weight)
                bin1 = np.multiply(bin1_m,weight)
                bin2 = np.multiply(bin2_m,weight)

                #Second: distribute per bin
                if plot_distr!="":
                    if n_obj>-1:
                        #base_var includes all the objects, tagged and not, in all bins
                        #First: let's sort in tagged and untagged
                        tag_base_var = base_var[tag_mask]
                        untag_base_var = base_var[untag_mask]
                        tag_base_pt_var = base_pt_var[tag_mask]
                        untag_base_pt_var = base_pt_var[untag_mask]

                        #Include object multiplicity mask
                        n_tag_obj_mask = (tag_base_var.counts > n_obj)
                        n_untag_obj_mask = (untag_base_var.counts > n_obj)

                        d0 = untag_base_var[np.multiply(n_untag_obj_mask,bin0_m)]
                        d1 = untag_base_var[np.multiply(n_untag_obj_mask,bin1_m)]
                        d1_t = tag_base_var[np.multiply(n_tag_obj_mask,bin1_m)]
                        d2_t = tag_base_var[np.multiply(n_tag_obj_mask,bin2_m)]

                        print "I need base_var and pt of the same size as bin1_m, in order to choose later"
                        pt1_all   = untag_base_pt_var[bin1_m]
                        pt1_t_all = tag_base_pt_var[bin1_m]
                        d1_all   = untag_base_var[bin1_m]
                        d1_t_all = tag_base_var[bin1_m]
                        #print "Let's print them"
                        #print "d1, d1_t"
                        print "primo termine: ", pt1_all
                        print pt1_all.max()
                        #print pt1_all.argmax()
                        print "secondo termine: ", pt1_t_all
                        print pt1_t_all.max()
                        #print pt1_t_all.argmax()
                        #print pt1_t_all.max().shape
                        #print pt1_t_all.argmax().shape
                        untag_larger_pt_mask = (pt1_all.max()>pt1_t_all.max())
                        tag_larger_pt_mask = (pt1_t_all.max()>pt1_all.max())
                        print untag_larger_pt_mask
                        print tag_larger_pt_mask
                        #print d1_t_all
                        #print d1_all.shape
                        #print d1_t_all.shape
                        #print "pt1, pt1_t"
                        #print pt1_all
                        #print pt1_t_all
                        #print "Problem is, they have different size, hence the order is messed up..."
                        #print "I should keep the same vector size, probably apply only bin1_m and do later the n_tag_obj_mask"
                        exit()

                        print "I also need weight with the correct size"
                        w1_all = weight[bin1_m]
                        w1_t_all = weight[bin1_m]
                        print pt1_t_all.shape
                        print w1_all.shape
                        exit()
                        
                        w0 = weight[np.multiply(n_untag_obj_mask,bin0_m)]
                        w1 = weight[np.multiply(n_untag_obj_mask,bin1_m)]
                        w1_t = weight[np.multiply(n_tag_obj_mask,bin1_m)]
                        w2_t = weight[np.multiply(n_tag_obj_mask,bin2_m)]
                        #print base_var[bin1_m]
                        #print sigprob[bin1_m]
                        #print "untag"
                        #print d1
                        #print "tag"
                        #print d1_t
                        #exit()
                        #print base_var[np.multiply(n_obj_mask,bin1_m)]
                        #print sigprob[np.multiply(n_obj_mask,bin1_m)]

                        #For plotting purposes: select objects that are tagged in bin1 and bin2
                        #Weights stay the same, they are per-event
                        #t1_m = sigprob[np.multiply(n_obj_mask,bin1_m)]>dnn_threshold
                        #t2_m = sigprob[np.multiply(n_obj_mask,bin2_m)]>dnn_threshold
                        #d1_t = d1[t1_m]
                        #d2_t = d2[t2_m]

                    else:
                        #take base_var and apply masks, then multiply by weight
                        d0 = base_var[bin0_m]
                        d1 = base_var[bin1_m]
                        d2 = base_var[bin2_m]
                        w0 = weight[bin0_m]
                        w1 = weight[bin1_m]
                        w2 = weight[bin2_m]

                #print "bin1 shape as multiplication ", bin1.shape
                #print bin1
                #df_all["bin0"] = bin0
                #df_all["bin1"] = bin1
                #df_all["bin2"] = bin2
                #print df_all[ ["Jets.sigprob","n_tag","bin0","bin1"]]

                print "events in bin 2:"
                print "RunNumber: ", runnumber
                print "LumiNumber: ", luminumber
                print "EventNumber: ", eventnumber
                print "Mindphi: ", mindphi
                print "score: ", score_v
                print "pt: ", pt_v
                print "eta: ", eta_v
                print "phi: ", phi_v
                #Predictions
                bin1_pred = defaultdict(dict)
                bin1_pred_up = defaultdict(dict)
                bin1_pred_low = defaultdict(dict)
                bin2_pred = defaultdict(dict)
                bin2_pred_up = defaultdict(dict)
                bin2_pred_low = defaultdict(dict)
                bin2_pred_from_1 = defaultdict(dict)
                bin2_pred_from_1_up = defaultdict(dict)
                bin2_pred_from_1_low = defaultdict(dict)
                p1 = {}
                p1Up = {}
                p1Down = {}
                p2 = {}
                p2Up = {}
                p2Down = {}
                eff1 = {}
                errUp1 = {}
                errDown1 = {}
                eff2 = {}
                errUp2 = {}
                errDown2 = {}

                BEST = True

                for r in TEff.keys():
                    #per-file, reset before any loop
                    bin1_pred[r] = np.array([]) if BEST else []
                    bin2_pred[r] = np.array([]) if BEST else []
                    bin2_pred_from_1[r] = np.array([]) if BEST else []

                
                ####
                ####  Best method
                ####


                
                #Avoid looping on events!
                #1. Define a per-chunk and per-bin probability vector
                prob_vec = {}

                #print "I am at iteration ", c
                #print pt
                #print sigprob
                #print pt_untag
                for r in TEff.keys():
                    prob_vec[r] = []
                    if eta:
                        for i in range(len(np_bins_eta)):
                            #print "Here there is something wrong, bins are not being skipped...."
                            if i<len(np_bins_eta)-1:
                                prob_vec[r].append(np.logical_and(pt_untag>=np_bins_eta[i],pt_untag<np_bins_eta[i+1])*eff[r][i])#*weight)
                            else:
                                prob_vec[r].append((pt_untag>=np_bins_eta[i])*eff[r][i])#*weight)

                    else:
                        if phi:
                            for i in range(len(np_bins_phi)):
                                if i<len(np_bins_phi)-1:
                                    prob_vec[r].append(np.logical_and(pt_untag>=np_bins_phi[i],pt_untag<np_bins_phi[i+1])*eff[r][i])#*weight)
                                else:
                                    prob_vec[r].append((pt_untag>=np_bins_phi[i])*eff[r][i])#*weight)
                        else:
                            for i in range(len(np_bins)):
                                if i<len(np_bins)-1:
                                    prob_vec[r].append(np.logical_and(pt_untag>=np_bins[i],pt_untag<np_bins[i+1])*eff[r][i])#*weight)
                                else:
                                    prob_vec[r].append((pt_untag>=np_bins[i])*eff[r][i])#*weight)

                    #print "prob_vec after all bins: ", prob_vec[r]
                    prob_tot = sum(prob_vec[r])
                    #print "prob_tot: ", prob_tot
                    somma = (prob_tot*weight).sum()
                    #print "per-event bin 1sum: ", somma
                    #print "bin 1sum so far: ", somma.sum()
                    cho = prob_tot.choose(2)
                    ##u = cho.unzip()
                    combi = cho.unzip()[0] * cho.unzip()[1] * weight
                    #print "somma shape ", somma.shape
                    #print "bin1_m shape ", bin1_m.shape
                    #print "somma[bin1_m] shape ", (somma[bin1_m]).sum()#.shape
                    #print "bin 1 pred: ", somma.sum()
                    #print "bin 2 pred: ", combi.sum().sum()

                    bin1_pred[r] = np.concatenate( (bin1_pred[r],somma) )
                    bin2_pred[r] = np.concatenate( (bin2_pred[r],combi.sum()) )
                    bin2_pred_from_1[r] = np.concatenate( (bin2_pred_from_1[r], somma[bin1_m]  )  )#.append(0.)
                    ###bin2_pred_from_1[r].append(somma if bin1[i]!=0 else 0.)

                    #Third: distributions per bin
                    if plot_distr!="":
                        if n_obj>-1:
                            #No need of stack, filter only relevant objects
                            ##Example:
                            ##lll = d0.astype(bool)*somma[np.multiply(n_obj_mask,bin0_m)]
                            ##lll = lll[lll.counts > n_obj,n_obj]
                            #weight_pr_1 should have the size of d0 (non-tagged objects)
                            prep_pr_1 = d0.astype(bool)*somma[np.multiply(n_untag_obj_mask,bin0_m)]
                            prep_pr_1 = prep_pr_1[prep_pr_1.counts > n_obj,n_obj]
                            weight_pr_1[r] = np.concatenate( (weight_pr_1[r], prep_pr_1) )

                            #weight_pr_2_from_0 should have the size of d0 (non-tagged objects)
                            prep_pr_2_from_0 = d0.astype(bool)*combi.sum()[np.multiply(n_untag_obj_mask,bin0_m)]
                            prep_pr_2_from_0 = prep_pr_2_from_0[prep_pr_2_from_0.counts > n_obj,n_obj]
                            weight_pr_2_from_0[r] = np.concatenate( (weight_pr_2_from_0[r], prep_pr_2_from_0) )

                            #weight_pr_2_from_1 should have the size of d1 (non-tagged objects)
                            prep_pr_2_from_1 = d1.astype(bool)*somma[np.multiply(n_untag_obj_mask,bin1_m)]
                            prep_pr_2_from_1 = prep_pr_2_from_1[prep_pr_2_from_1.counts > n_obj,n_obj]
                            weight_pr_2_from_1[r] = np.concatenate( (weight_pr_2_from_1[r],prep_pr_2_from_1 if len(prep_pr_2_from_1)>0 else np.array([])  ) )

                        else:
                            ##Stack is not that relevant for per-event variables
                            ##but it's needed for Jets.pt z.B.
                            weight_pr_1[r] = np.concatenate( (weight_pr_1[r], np.hstack(d0.astype(bool)*somma[bin0_m])) )
                            weight_pr_2_from_0[r] = np.concatenate( (weight_pr_2_from_0[r], np.hstack(d0.astype(bool)*combi.sum()[bin0_m]) ) )
                            weight_pr_2_from_1[r] = np.concatenate( (weight_pr_2_from_1[r],np.hstack(d1.astype(bool)*somma[bin1_m]) if len(d1)>0 else np.array([])  ) )


                '''
                ####
                ####  Worse method
                ####

                #Loop over events
                for i in range(len(pt_untag)):
                    #Reset every event!
                    for r in TEff.keys():
                        p1[r] = 0.
                        p1Up[r] = 0.
                        p1Down[r] = 0.
                        p2[r] = 0.
                        p2Up[r] = 0.
                        p2Down[r] = 0.
                        eff1[r] = 0.
                        errUp1[r] = 0.
                        errDown1[r] = 0.
                        eff2[r] = 0.
                        errUp2[r] = 0.
                        errDown2[r] = 0.

                    #print "Event n. ", i
                    w = weight[i]#scalar weight
                    #Loop over jets sub-indices
                    n_j1 = 0
                    for j1 in range( len(pt_untag[i])):
                        for r in TEff.keys():
                            binN1 = TEff[r].GetPassedHistogram().FindBin( pt_untag[i][j1])
                            eff1[r]  = TEff[r].GetEfficiency(binN1)
                            errUp1[r] = TEff[r].GetEfficiencyErrorUp(binN1)
                            errDown1[r] = TEff[r].GetEfficiencyErrorLow(binN1)
                            p1[r]+=eff1[r]*w
                            p1Up[r]+=(eff1[r]+errUp1[r])*w
                            p1Down[r]+=(eff1[r]-errDown1[r])*w
                        #Loop on jet 2
                        for j2 in range( len(pt_untag[i]) ):
                            if j2>j1:
                                for r in TEff.keys():
                                    binN2 = TEff[r].GetPassedHistogram().FindBin( pt_untag[i][j2] )
                                    eff2[r]  = TEff[r].GetEfficiency(binN2)
                                    errUp2[r] = TEff[r].GetEfficiencyErrorUp(binN2)
                                    errDown2[r] = TEff[r].GetEfficiencyErrorLow(binN2)
                                    p2[r] += eff1[r]*eff2[r]*w
                                    p2Up[r] += (eff1[r]+errUp1[r])*(eff2[r]+errUp2[r])*w
                                    p2Down[r] += (eff1[r]-errDown1[r])*(eff2[r]-errDown2[r])*w

                        n_j1+=1
                    #Here we have a per-event prediction
                    for r in TEff.keys():
                        #print r, " pred event: ", p1[r]
                        bin1_pred[r].append(p1[r])# = np.concatenate((bin1_pred[r],np.array(p1[r])))#
                        bin2_pred[r].append(p2[r])# = np.concatenate((bin2_pred[r],p2[r]))#
                        bin2_pred_from_1[r].append(p1[r] if bin1[i]!=0 else 0.)# = np.concatenate((bin2_pred_from_1[r],p1[r] if bin1[i]!=0 else np.array([0.])))#

                bin1_pred[r] = np.array(bin1_pred[r])
                bin2_pred[r] = np.array(bin2_pred[r])
                bin2_pred_from_1[r] = np.array(bin2_pred_from_1[r])
                #end of worse method
                '''
                
                #Here: still in chunk loop
                #Here concatenate to full pred
                for r in TEff.keys():
                    #print pr_1[r].shape, bin1_pred[r].shape
                    #print "What I concatenate to pr_1[",r,"]: ", bin1_pred[r]
                    pr_1[r] = np.concatenate((pr_1[r],bin1_pred[r]))
                    pr_2[r] = np.concatenate((pr_2[r],bin2_pred[r]))
                    pr_2_from_1[r] = np.concatenate((pr_2_from_1[r],bin2_pred_from_1[r]))
                    #pr_1[r] += bin1_pred[r]
                    #pr_2[r] += bin2_pred[r]
                    #pr_2_from_1[r] += bin2_pred_from_1[r]

                    #per-chunk pred
                    #print "per-chunk pred [",r,"] bin 1: ", np.sum(bin1_pred[r])
                    #print "per-chunk pred [",r,"] bin 2: ", np.sum(bin2_pred[r])


                b0 = np.concatenate((b0,bin0))
                b1 = np.concatenate((b1,bin1))
                b2 = np.concatenate((b2,bin2))

                if plot_distr!="":

                    if n_obj>-1:

                        print "\n"
                        print "     Here I need to choose between tagged jet or promoted jet based on their pt"
                        print "\n"
                        print base_var[base_var>0]
                        print sigprob[base_var>0]
                        print tag_base_pt_var
                        print untag_base_pt_var
                        #sort_pt_mask = tag_base_pt_var>untag_base_pt_var
                        #print sort_pt_mask
                        exit()


                        #At this point we only have to select relevant jets
                        distr_0 = np.concatenate( (distr_0,d0[d0.counts > n_obj,n_obj]) )
                        distr_1 = np.concatenate( (distr_1,d1[d1.counts > n_obj,n_obj]) )
                        #distr_2 = np.concatenate( (distr_2,d2[d2.counts > n_obj,n_obj]) )
                        distr_1_t = np.concatenate( (distr_1_t,d1_t[d1_t.counts > n_obj,n_obj]) )
                        distr_2_t = np.concatenate( (distr_2_t,d2_t[d2_t.counts > n_obj,n_obj]) )
                        weight_1_t = np.concatenate( (weight_1_t,w1_t) )
                        weight_2_t = np.concatenate( (weight_2_t,w2_t) )
                        
                        '''                        
                        ##Adjust t1_m and t2_m such as they become positional
                        ##We want to mask only n_obj-th object
                        #t1_m = t1_m[t1_m.counts > n_obj,n_obj]
                        #t2_m = t2_m[t2_m.counts > n_obj,n_obj]
                        #d1_t = d1[d1.counts > n_obj,n_obj]
                        #d1_t = d1_t[t1_m]
                        #d2_t = d2[d2.counts > n_obj,n_obj]
                        #d2_t = d2_t[t2_m]
                        ##print d1
                        ##print d1[d1.counts > n_obj,n_obj]
                        ##print d1_t
                        distr_1_t = np.concatenate( (distr_1_t,d1_t) )
                        distr_2_t = np.concatenate( (distr_2_t,d2_t) )
                        #Also weights positionally adjusted
                        weight_1_t = np.concatenate( (weight_1_t,w1[t1_m]) )
                        weight_2_t = np.concatenate( (weight_2_t,w2[t2_m]) )
                        '''
                        weight_0 = np.concatenate( (weight_0,w0) )
                        weight_1 = np.concatenate( (weight_1,w1) )
                        #weight_2 = np.concatenate( (weight_2,w2) )

                    else:

                        #one very last thing to do
                        #for distr_1 and distr_2 we want to save also tagged jets
                        distr_0 = np.concatenate( (distr_0,np.hstack(d0)) )
                        distr_1 = np.concatenate( (distr_1,np.hstack(d1) if len(d1)>0 else np.array([])) )
                        distr_2 = np.concatenate( (distr_2,np.hstack(d2) if len(d2)>0 else np.array([]) ) )
                        weight_0 = np.concatenate( (weight_0, np.hstack(d0.astype(bool)*w0) ) )
                        weight_1 = np.concatenate( (weight_1, np.hstack(d1.astype(bool)*w1) if len(d1)>0 else np.array([]) ) )
                        weight_2 = np.concatenate( (weight_2, np.hstack(d2.astype(bool)*w2) if len(d2)>0 else np.array([]) ) )

                    #print "hstack puts everything together, we should not"
                    #d0 = d0[ d0[d0>0].counts>1 ]
                    #print d0[:,1]
                    #print d0.shape

                    #print "shape of distr_0", distr_0.shape
                    #print distr_0
                    #print "shape of w0",w0.shape
                    #print w0
                    #print "shape of weight_0",weight_0.shape
                    #print weight_0

                    #print "shape of w1",w1.shape
                    #print w1
                    #print "shape of weight_1",weight_1.shape
                    #print weight_1

                    #print "shape of w2",w2.shape
                    #print w2
                    #print "shape of weight_2",weight_2.shape
                    #print weight_2

                #del tmp
                en_it = time.time()
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"
                    #print "uproot iteration n. ", c,"/",n_iter, " took ", en_it-st_it
                    c+=1
                    #new#gc.collect()

            #for r in TEff.keys():
            #    print r, " per-sample pred bin 1: ", np.sum(bin1_pred[r])
            #    print r, " per-sample pred bin 2: ", np.sum(bin2_pred[r])
            #    print r, " per-sample pred bin 2 from 1: ", np.sum(bin2_pred_from_1[r])
            #print "per-sample bin0 yield: ",  np.sum(bin0)
            #print "per-sample bin1 yield: ",  np.sum(bin1)
            #print "per-sample bin2 yield: ",  np.sum(bin2)

            del gen

            #Here in files loop ss

            #list_files_for_uproot.append(NTUPLEDIR + ss + ".root")#:tree")
            #Write to h5 and delete list
            #print "Perform concatenation of df...."
            #df = pd.concat(df_list,ignore_index=True)
            #df.rename(columns={"pt" : "MEt.pt"},inplace=True)
            #df["tree_weight"] = tree_weights_array
            #df.convert_objects()
            #df.to_hdf(NTUPLEDIR+ss+'.h5', 'df', format='table')
            #print "Df saved in ", NTUPLEDIR+ss+".h5"

            #print df["Jets.pt"]
            #print "Write csv/pkl"
            #df.to_csv(NTUPLEDIR+ss+'.csv')
            #print "Df saved in ", NTUPLEDIR+ss+".csv"
            #df.to_pickle(NTUPLEDIR+ss+'.pkl')
            #print "Df saved in ", NTUPLEDIR+ss+".pkl"

            #del df_list
            #del df

        #df = pd.concat(df_list,ignore_index=True)#concat is time/RAM consuming, do it once at the end
        #df.rename(columns={"pt" : "MEt.pt"},inplace=True)
        ##Add tree weight, check if order is correct!
        #df["tree_weight"] = tree_weights_array#tree_weights[c]*np.ones( len(arrays[ key_list[0] ])  )

        #print "************************************"
        #for r in TEff.keys():
        #    #print np.array(pr_1[r]).shape
        #    #print (np.sum(np.array(pr_1[r]))).shape
        #    print r, " total final pred bin 1: ", np.sum(pr_1[r])
        #    print r, " total final pred bin 2: ", np.sum(np.array(pr_2[r]))
        #    print r, " total final pred bin 2 from 1: ", sum(np.array(pr_2_from_1[r]))
        #print "final bin0 yield: ",  np.sum(np.array(b0))
        #print "final bin1 yield: ",  np.sum(np.array(b1))
        #print "final bin2 yield: ",  np.sum(np.array(b2))

        #del chain[s]
        end_uproot = time.time()
        print "\n"
        print "   --- > Tot size of arrays: ", array_size_tot
        print "Size of tree_weights_array: ", len(tree_weights_array)
        print "Time elapsed to fill uproot array: ", end_uproot-start_uproot
        print "************************************"
        print "\n"

        '''
        print "all correct so far, these should be the final numbers"
        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]
            
            print r+reg_label+dset
            print pr_1[r+reg_label+dset]
            print np.sum(np.array(pr_1[r+reg_label+dset]))
            print pr_1[r+reg_label+dset].sum()
        print "************************************"
        '''

        #exit()
        start_chain = time.time()
        ##chain[s] = TChain("tree")
        ##For fair time comparison, reopen chain
        #for l, ss in enumerate(samples[s]['files']):
        #    tree_weights[l] = tree_weight_dict[s][ss]
        #    chain[s].Add(NTUPLEDIR + ss + ".root")
        #    #print "Entries in ", ss, " : ", chain[s].GetEntries()
        
        print("Entries in chain: %d")%(chain[s].GetEntries())
        max_n=chain[s].GetEntries()+10#100000

        #ev_weight = "(1/"+str(tree_weights[...])+")"
        #ev_weight = "1"
        ev_weight  = "EventWeight*PUReWeight"
        tagvar = "nTagJets_0p996_JJ"
        #tagvar = "nTagJets_cutbased"
        print tagvar
        cutstring_0 = ev_weight+"*("+tagvar+"==0)"
        cutstring_1 = ev_weight+"*("+tagvar+"==1)"
        cutstring_2 = ev_weight+"*("+tagvar+">1)"

        #combination single jet
        if CUT == "isJetHT":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MEt.pt<25 && (HLT_PFJet140_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MEt.pt<25 && (HLT_PFJet140_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MEt.pt<25 && (HLT_PFJet140_v))"

        if CUT == "isDiJetMET":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"


        print "CUT: ", CUT
        print "cutstring bin 0: ", cutstring_0
        chain[s].Project(s+"_0", "isMC",cutstring_0,"")#,max_n)
        chain[s].Project(s+"_1", "isMC",cutstring_1,"",max_n)
        chain[s].Project(s+"_2", "isMC",cutstring_2,"",max_n)
        end_chain = time.time()
        print "Time elapsed to project TChain: ", end_chain-start_chain
        print "************************************"
        print "h0 entries: ", h0[s].GetEntries()
        print "h0 integral dummy: ", h0[s].Integral()
        print "h1 entries: ", h1[s].GetEntries()
        print "h1 integral dummy: ", h1[s].Integral()
        print "h2 entries: ", h2[s].GetEntries()
        print "h2 integral dummy: ", h2[s].Integral()
        #exit()


        #Plotting variables
        if plot_distr!="":
            hist_0 = TH1D("bin0", "bin0", variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
            hist_1 = TH1D("bin1", "bin1", variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
            hist_2 = TH1D("bin2", "bin2", variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
            hist_0.Sumw2()
            hist_1.Sumw2()
            hist_2.Sumw2()
            _ = root_numpy.fill_hist( hist_0, distr_0, weights=weight_0 )
            if n_obj>-1:
                #If we look at single objects, we want them tagged
                print " - - - - Bin 1 and bin 2 will contain only tagged jets - - - -"
                _ = root_numpy.fill_hist( hist_1, distr_1_t, weights=weight_1_t )
                _ = root_numpy.fill_hist( hist_2, distr_2_t, weights=weight_2_t )
                print distr_1_t.shape
            else:
                _ = root_numpy.fill_hist( hist_1, distr_1, weights=weight_1 )
                _ = root_numpy.fill_hist( hist_2, distr_2, weights=weight_2 )
                print distr_1.shape
                print hist_1.Print()
                hist_1_pr = {}
                hist_2_from_0_pr = {}
                hist_2_from_1_pr = {}

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
                can0.Print(PLOTDIR+'Bin0_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.png')
                can0.Print(PLOTDIR+'Bin0_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.pdf')

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
                hist_1.SetMarkerStyle(21)
                hist_1.SetLineColor(colors[1])
                hist_1.SetMarkerColor(colors[1])
                leg1.AddEntry(hist_1,"bin 1; "+r,"PL")

            #Plot bin2
            can2 = TCanvas("bin2","bin2", 1000, 900)
            can2.cd()
            if variable[plot_distr]["log"]==True:
                can2.SetLogy() 
                leg2 = TLegend(0.5, 0.7, 0.9, 0.9)
                leg2.SetTextSize(0.035)
                leg2.SetBorderSize(0)
                leg2.SetFillStyle(0)
                leg2.SetFillColor(0)
                hist_2.SetTitle("")
                hist_2.GetXaxis().SetTitle(variable[plot_distr]["title"])
                hist_2.GetYaxis().SetTitle("Events")
                hist_2.SetMarkerSize(1.2)
                hist_2.SetLineWidth(2)
                hist_2.SetMarkerStyle(21)
                hist_2.SetLineColor(colors[2])
                hist_2.SetMarkerColor(colors[2])
                bin2_max = hist_2.GetMaximum()
                leg2.AddEntry(hist_2,"bin 2; "+r,"PL")

            for r in TEff.keys():
                hist_1_pr[r] = TH1D("bin1_pr"+r, "bin1_pr"+r, variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
                hist_2_from_0_pr[r] = TH1D("bin2_0_pr"+r, "bin2_0_pr"+r, variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
                hist_2_from_1_pr[r] = TH1D("bin2_1_pr"+r, "bin2_1_pr"+r, variable[plot_distr]['nbins'], variable[plot_distr]['min'], variable[plot_distr]['max'])
                hist_1_pr[r].Sumw2()
                hist_2_from_0_pr[r].Sumw2()
                hist_2_from_1_pr[r].Sumw2()
                _ = root_numpy.fill_hist( hist_1_pr[r], distr_0, weights=weight_pr_1[r] )#distr_1,weights=weight_1)#
                _ = root_numpy.fill_hist( hist_2_from_0_pr[r], distr_0, weights=weight_pr_2_from_0[r] )
                _ = root_numpy.fill_hist( hist_2_from_1_pr[r], distr_1, weights=weight_pr_2_from_1[r] )
                print distr_0.shape
                print hist_1_pr[r].Print()
                #Plot bin1
                can1.cd()
                hist_1_pr[r].SetMarkerSize(1.2)
                hist_1_pr[r].SetLineWidth(2)
                hist_1_pr[r].SetMarkerStyle(24)
                hist_1_pr[r].SetLineColor(colors[1])
                hist_1_pr[r].SetLineStyle(2)
                hist_1_pr[r].SetMarkerColor(colors[1])
                hist_1_pr[r].SetTitle("")
                hist_1_pr[r].GetXaxis().SetTitle(variable[plot_distr]["title"])
                hist_1_pr[r].GetYaxis().SetTitle("Events")
                hist_1_pr[r].Draw("PL")
                leg1.AddEntry(hist_1_pr[r],"bin 1 pred.; "+r,"PL")

                #Plot bin2
                can2.cd()
                hist_2_from_1_pr[r].SetMarkerSize(1.2)
                hist_2_from_1_pr[r].SetLineWidth(2)
                hist_2_from_1_pr[r].SetMarkerStyle(25)
                hist_2_from_1_pr[r].SetLineColor(colors[3])
                hist_2_from_1_pr[r].SetLineStyle(2)
                hist_2_from_1_pr[r].SetMarkerColor(colors[3])
                hist_2_from_1_pr[r].SetTitle("")
                hist_2_from_1_pr[r].GetXaxis().SetTitle(variable[plot_distr]["title"])
                hist_2_from_1_pr[r].GetYaxis().SetTitle("Events")
                bin2_max = max(bin2_max,hist_2_from_1_pr[r].GetMaximum())

                hist_2_from_0_pr[r].SetMarkerSize(1.2)
                hist_2_from_0_pr[r].SetLineWidth(2)
                hist_2_from_0_pr[r].SetMarkerStyle(24)
                hist_2_from_0_pr[r].SetLineColor(colors[4])
                hist_2_from_0_pr[r].SetLineStyle(2)
                hist_2_from_0_pr[r].SetMarkerColor(colors[4])
                hist_2_from_0_pr[r].SetTitle("")
                hist_2_from_0_pr[r].GetXaxis().SetTitle(variable[plot_distr]["title"])
                hist_2_from_0_pr[r].GetYaxis().SetTitle("Events")
                bin2_max = max(bin2_max,hist_2_from_0_pr[r].GetMaximum())

                hist_2_from_1_pr[r].Draw("PL")
                hist_2_from_1_pr[r].SetMaximum(bin2_max*(1+0.1))
                hist_2_from_0_pr[r].Draw("PL,sames")
                leg2.AddEntry(hist_2_from_0_pr[r],"bin 2 pred. from 0; "+r,"PL")
                leg2.AddEntry(hist_2_from_1_pr[r],"bin 2 pred. from 1; "+r,"PL")


            can1.cd()
            hist_1.Draw("PL,sames")
            leg1.Draw()
            can1.Print(PLOTDIR+'Bin1_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.png')
            can1.Print(PLOTDIR+'Bin1_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.pdf')

            can2.cd()
            hist_2.Draw("PL,sames")
            leg2.Draw()
            can2.Print(PLOTDIR+'Bin2_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.png')
            can2.Print(PLOTDIR+'Bin2_'+plot_distr.replace('.', '_')+'_'+r+dset+add_label+label_2+'.pdf')

            exit()


        y_0 = np.sum(np.array(b0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(b1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(b2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in b0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in b1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in b2) )#*tree_weight --> already in w

        error_0_i = c_double()#Double()
        error_1_i = c_double()#Double()
        y_0_i = h0[s].IntegralAndError(h0[s].GetXaxis().FindBin(0),h0[s].GetXaxis().FindBin(2),error_0_i,"")
        y_1_i = h1[s].IntegralAndError(h1[s].GetXaxis().FindBin(0),h1[s].GetXaxis().FindBin(2),error_1_i,"")
        rowI = [s, round(y_0_i,2), round(error_0_i.value,2), round(y_1_i,2), round(error_1_i.value,2)]
        #rowI = [s, round(y_0_i,2), round(error_0_i,2), round(y_1_i,2), round(error_1_i,2)]

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
            '''
            print "-----------------------------"
            print "Extrapolation from ", r+reg_label+dset
            print "Bin 2 from 0"
            print "Sorted pred weights ", sorted_pr_2[r+reg_label+dset]
            print "Mean weight per event ", np.mean(sorted_pr_2[r+reg_label+dset])
            print "Prediction: ", np.sum(sorted_pr_2[r+reg_label+dset])
            print "Prediction skipping first 10 events: ", np.sum(sorted_pr_2[r+reg_label+dset][10:])
            print "Prediction skipping first 100 events: ", np.sum(sorted_pr_2[r+reg_label+dset][100:])
            print "Bin 2 from 1"
            print "Sorted pred weights ", sorted_pr_2_from_1[r+reg_label+dset]
            print "Mean weight per event ", np.mean(sorted_pr_2_from_1[r+reg_label+dset])
            print "Prediction: ", np.sum(sorted_pr_2_from_1[r+reg_label+dset])
            print "Prediction skipping first 10 events: ", np.sum(sorted_pr_2_from_1[r+reg_label+dset][10:])
            print "Prediction skipping first 100 events: ", np.sum(sorted_pr_2_from_1[r+reg_label+dset][100:])
            '''
            pred_1[r+reg_label+dset] = np.sum(np.array(pr_1[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
            e_pred_1[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_1[r+reg_label+dset]) ).sum()
            pred_2[r+reg_label+dset] = np.sum(np.array(pr_2[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
            e_pred_2[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_2[r+reg_label+dset]) ).sum()
            pred_2_from_1[r+reg_label+dset] = np.sum(np.array(pr_2_from_1[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
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

            rowI = [s, round(y_0,round_val), round(e_0,round_val), round(y_1,round_val), round(e_1,round_val)]

            table_integral[r+reg_label+dset].add_row(rowI)


        #Here I would like to keep the name of the samples
        #Try to align here:

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
                if len(datasets)>0:
                    dset = datasets[i]

            ##if i==0:
            print('\n\n======================= From histogram  ==============================')
            print(table_integral[r+reg_label+dset])
            print '\n\n======================= Yields and predictions extrapolated from '+r+reg_label+dset+' ==============================' 
            print(table_yield[r+reg_label+dset])
            print(table_pred_new[r+reg_label+dset])
            print(table_pred[r+reg_label+dset])

        wr  = True#False#False

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
                    w.write(str(table_integral[r+reg_label+dset])+"\n")
                    w.close()
                    print "Info: tables written in file "+PLOTDIR+"BkgPred_extr_region_"+r+reg_label+dset+add_label+label_2+".txt"
            else:
                print "NO tables written in file !!!!!!"    

            #Output separated per CR
            #with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ '_extr_region_' +r+  add_label+".json", "w") as w:
            #    w.write("results = " + json.dumps(results[r+reg_label+dset]))
            #    w.close()
            #    print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ '_extr_region_' +r+  add_label+".json"

            #with open("python/BkgPredResults_"+ERA+"_"+REGION+'_extr_region_' +r+add_label+".py", "w") as w:
            #    w.write("#! /usr/bin/env python \n")
            #    w.write("results = " + json.dumps(results[r+reg_label+dset]))
            #    w.close()
            #    print "Info: dictionary written in file "+"python/BkgPredResults_"+ERA+"_"+REGION+'_extr_region_' +r+add_label+".py"

        with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+ "_"+s+ add_label+label_2+".yaml","w") as f:
            yaml.dump(results, f)
            f.close()
            print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".yaml"

        with open(PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+"_"+s+ add_label+label_2+".json", "w") as w:
            w.write("results = " + json.dumps(results))
            w.close()
            print "Info: dictionary written in file "+PLOTDIR+"BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".json"

        with open("python/BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".py", "w") as w:
            w.write("#! /usr/bin/env python \n")
            w.write("results = " + json.dumps(results))
            w.close()
            print "Info: dictionary written in file "+"python/BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".py"

    ##To open a json:
    #a_file = open(PLOTDIR+'BkgPredResults.json', "r")
    #output = a_file.read()
    #print(output)
    ##Here: RDataFrame working example. Requires to load objects from ROOT
    ##The conversion to pandas seems slow
    ##https://nbviewer.jupyter.org/url/root.cern/doc/master/notebooks/df026_AsNumpyArrays.py.nbconvert.ipynb
    #start_frame = time.time()
    #dfR[s] = RDataFrame(chain[s])
    #dfR = dfR[s].Range(max_n)
    ##npy = df.Filter(CUT+" && nCHSJetsAcceptanceCalo>0").AsNumpy(["Jets"])#["nCHSJetsAcceptanceCalo", "isMC", "MEt"])
    #npy = dfR.Filter(CUT).AsNumpy(["nCHSJetsAcceptanceCalo","Jets.pt"])#["nCHSJetsAcceptanceCalo", "isMC", "MEt"])
    #print npy["Jets.pt"][1:2][0]
    ##exit()
    ##print(npy["MEt"][0].pt)
    ##print(npy["MEt"][0].phi)
    ##print(npy["Jets"][0][0].pt)
    #dfp = pd.DataFrame(npy)
    #print(dfp[1:2][ ["Jets.pt","nCHSJetsAcceptanceCalo"] ])
    #end_frame = time.time()
    #print "Time elapsed to fill RDataFrame: ", end_frame-start_frame
    #print "************************************"
    #exit()


def write_datacards(tree_weight_dict,sign,main_pred_reg,main_pred_sample,extr_region,unc_dict,comb_fold_label="",regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False):


    PREDDIR = "plots/Efficiency_AN/v5_calo_AOD_"+ERA+"_"+main_pred_reg+"/"
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
    #else:
    #    pred_file_name+="_wp"+str(dnn_threshold).replace(".","p")

    pred_file_name+=comb_fold_label


    with open(pred_file_name+".yaml","r") as f:
        print "\n"
        print "Info: opening dictionary in file "+pred_file_name+".yaml"
        print "Extrapolation region: ", extr_region+comb_fold_label
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()

    print "Inferring limits on absolute x-sec in fb"
    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    DATACARDDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)

    ###DATACARDDIR += TAGVAR+comb_fold_label
    DATACARDDIR += TAGVAR
    if eta_cut:
        DATACARDDIR += "_eta_1p0"
    if phi_cut:
        DATACARDDIR += "_phi_cut"

    if eta:
        DATACARDDIR += "_vs_eta"+comb_fold_label
    else:
        if phi:
            DATACARDDIR += "_vs_phi"+comb_fold_label
        else:
            DATACARDDIR += comb_fold_label

    DATACARDDIR += "/"
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

    #mass = []
    print "tree_weight_dict"
    print tree_weight_dict
    chainSignal = {}
    list_of_variables = [TAGVAR,"isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","TriggerWeight","PUWeight","PUReWeight","GenLLPs.travelRadius","GenLLPs.travelTime","GenLLPs.beta","GenLLPs.*",CUT]#"nLeptons"
    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        chainSignal[s] = TChain("tree")
        tree_weights = {}
        chain_entries_cumulative = {}
        chain_entries = {}
        array_size_tot = 0
        bin2 = np.array([])
        ctau_weights = defaultdict(dict)
        y_2_ctau = defaultdict(dict)#.....check?
        e_2_ctau = defaultdict(dict)#....check?
        #initialize
        for ct in ctaus:
            ctau_weights[ct] = np.array([])

        mu_no_cont_limit = 0.
        bin2_contamination_yield = 0.
        bin2_contamination_yield_error = 0.
        bin2_contamination_yield_from_1 = 0.
        bin2_contamination_yield_error_from_1 = 0.

        if(contamination):
            print "Adding to bin 2 signal contamination coming from bin 1"
            pred_file_name_signal = PREDDIR+"BkgPredResults_"+ERA+"_"+main_pred_reg+"_"+s
            if eta_cut:
                pred_file_name_signal+="_eta_1p0"
            if phi_cut==True:
                pred_file_name_signal+="_phi_cut"
            if eta:
                pred_file_name_signal+= "_vs_eta"
            if phi:
                pred_file_name_signal+= "_vs_phi"
            pred_file_name_signal+=comb_fold_label
            print "Opening yaml file: ",  pred_file_name_signal+".yaml"
            with open(pred_file_name_signal+".yaml","r") as f_s:
                print "\n"
                print "Info: opening dictionary in file "+pred_file_name_signal+".yaml"
                results_contam = yaml.load(f_s, Loader=yaml.Loader)
                f_s.close()
                bin2_contamination_yield = results_contam[extr_region][s]['pred_2']
                bin2_contamination_yield_error = results_contam[extr_region][s]['e_pred_2']
                bin2_contamination_yield_from_1 = results_contam[extr_region][s]['pred_2_from_1']
                bin2_contamination_yield_error_from_1 = results_contam[extr_region][s]['e_pred_2_from_1']
                print "Contamination accounted: ", bin2_contamination_yield_from_1*scale_mu

            print "Opening also median expected without contamination (to set mu)"
            tmp_contam_lab=""
            if scale_mu!=1.:
                tmp_contam_lab+='_scale_mu_'+str(scale_mu).replace(".","p")
            no_contam_limit = RESULTS + "/" + s + tmp_contam_lab+label_2+".txt"
            with open(no_contam_limit) as l:
                lines = l.readlines()
            mu_no_cont_limit = float(lines[2])
            print mu_no_cont_limit

        for l, ss in enumerate(samples[s]['files']):
            print "ss", ss
            chainSignal[s].Add(NTUPLEDIR + ss + '.root')
            tree_weights[l] = tree_weight_dict[s][ss]
            chainSignal[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chainSignal[s].GetEntries()
            if l==0:
                chain_entries[l]=chainSignal[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                        #cut_mask = arrays[CUT]>0
                        #print "!!!! try to kill QCD!!!"
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )

                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        if "MET" in REGION:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        else:
                            cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                        #cut_mask = arrays[CUT]>0
                        #print "!!!! try to kill QCD!!!"
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                        #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                        #enhance MET
                        #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                        #no MT
                        #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = (arrays[CUT]>0)
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)
                        
                #HEM
                if CUT == "isSR" and ERA==2018:
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM__pt_30_all_eta"]==0)))
                    ###cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))

                if KILL_QCD:
                    print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)

                if eta_cut and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    cut_mask = (cut_mask_eta.any()==True)

                if phi_cut and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)

                if phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)


                #SR cut
                cut_mask = np.logical_and( cut_mask, arrays[TAGVAR]>1 )
                tag = arrays[TAGVAR][cut_mask] !=0
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                trgweight = arrays["TriggerWeight"][cut_mask]
                #Here: gen level stuff
                genRadius = arrays["GenLLPs.travelRadius"][cut_mask]
                genTime = arrays["GenLLPs.travelTime"][cut_mask]
                genBeta = arrays["GenLLPs.beta"][cut_mask]
                genGamma = np.divide(1.,np.sqrt(1-np.multiply(genBeta,genBeta)))
                genPosteriorTime = np.divide(genRadius,np.multiply(genBeta , genGamma))
                genAccept = arrays["GenLLPs.isLLPInCaloAcceptance"][cut_mask]
                #print genBeta
                #print genGamma
                #print "pre time ", genTime
                print "post time ", genPosteriorTime
                print "radius ", genRadius
                print genAccept
                print genPosteriorTime.sum()
                print tau_weight_calc(genPosteriorTime.sum(), 10./10., float(ctaupoint)/10.)#cm
                del arrays

                for ct in ctaus:
                    ctau_weights[ct] = np.concatenate( (ctau_weights[ct],tau_weight_calc(genPosteriorTime.sum(), float(ct)/10., float(ctaupoint)/10.)) )
                
                if scale_mu!=1.:
                    print "Scaling mu up by factor", scale_mu
                    weight = np.multiply(eventweight,np.multiply(pureweight,trgweight))*tree_weight_dict[s][ss]*signalMultFactor*scale_mu
                else:
                    weight = np.multiply(eventweight,np.multiply(pureweight,trgweight))*tree_weight_dict[s][ss]*signalMultFactor
                bin2 = np.concatenate( (bin2,np.multiply(tag,weight)) )

            del gen
            print ctau_weights

        y_2 = np.sum(bin2)#*tree_weight --> already in w
        e_2 = np.sqrt( sum(x*x for x in bin2) ).sum()#*tree_weight --> already in w
        for ct in ctaus:
            y_2_ctau[ct] = np.sum( np.multiply(bin2,ctau_weights[ct]) )
            e_2_ctau[ct] = np.sqrt( sum(x*x for x in np.multiply(bin2,ctau_weights[ct])) )


        #for simplicity to avoid overwriting
        #if eta_cut:
        #    res_reg = extr_region+add_label#+label_2
        #else:
        #    res_reg = extr_region+add_label#+label_2
        res_reg = extr_region+comb_fold_label
        
        #print s, y_2, " +- ", e_2
        #print results
        key_dict = results[res_reg].keys()[0]
        if pred_unc==0:
            pred_unc = abs(results[res_reg][key_dict]['pred_2_from_1'] - results[res_reg][key_dict]['pred_2'])/(results[res_reg][key_dict]['pred_2_from_1']+results[res_reg][key_dict]['pred_2'])/2.


        print y_2
        print y_2_ctau

        y_2_ctau[ctaupoint] = y_2
        e_2_ctau[ctaupoint] = e_2

        print "Now I must only organize datacards accordingly (good naming and good output folder)"
        print "need for sure to have a datacard dictionary to avoid re-writing the same stuff"
        for ct in np.append(np.array([ctaupoint]),ctaus):
        #for ct in np.array([ctaupoint]):

            #*******************************************************#
            #                                                       #
            #                      Datacard                         #
            #                                                       #
            #*******************************************************#
        
            card  = 'imax 1\n'#n of bins
            card += 'jmax *\n'#n of backgrounds
            card += 'kmax *\n'#n of nuisance parmeters
            card += '-----------------------------------------------------------------------------------\n'
            card += 'bin                      %s\n' % CHAN
            card += 'observation              %f\n' % unc_dict["obs"]
            card += '-----------------------------------------------------------------------------------\n'
            card += 'bin                      %-33s%-33s\n' % (CHAN, CHAN)
            card += 'process                  %-33s%-33s\n' % (s, 'Bkg')
            card += 'process                  %-33s%-33s\n' % ('0', '1')
            if contamination:
                print "Due to contamination: scaling with mu taken from no contamination case: ", mu_no_cont_limit
                print "Original values: y_2_ctau[ct] ", y_2_ctau[ct], " ; contam: ", bin2_contamination_yield_from_1*scale_mu
                print "Scaled values: y_2_ctau[ct] ", y_2_ctau[ct]*mu_no_cont_limit, " ; contam: ", bin2_contamination_yield_from_1*scale_mu*mu_no_cont_limit
                y_2_contam = (results[res_reg][key_dict]['pred_2_from_1'] + bin2_contamination_yield_from_1*scale_mu*mu_no_cont_limit)
                card += 'rate                     %-33f%-33f\n' % (y_2_ctau[ct]*mu_no_cont_limit, y_2_contam)
            else:
                card += 'rate                     %-33f%-33f\n' % (y_2_ctau[ct], unc_dict["bkg_yield"])
                #card += 'rate                     %-33f%-33f\n' % (y_2_ctau[ct], results[res_reg][key_dict]['pred_2_from_1'])

            #kill QCD
            #card += 'rate                     %-33f%-33f\n' % (y_2_ctau[ct], max(results[res_reg][key_dict]['pred_2_from_1'],results[res_reg][key_dict]['pred_2']))
            card += '-----------------------------------------------------------------------------------\n'
            #Syst uncertainties
            card += '%-12s     lnN     %-33f%-33s\n' % ('sig_norm',1.+e_2_ctau[ct]/y_2_ctau[ct],'-')
            if contamination:
                print "Adding signal contamination in bin 2 bkg, but keeping same level of uncertainty"
                #card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm','-', 1.+pred_unc)
                card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm_stat','-', unc_dict["bkg_yield_stat"])
                card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm_syst','-', unc_dict["bkg_yield_syst"])
            else:
                #card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm','-', 1.+pred_unc)
                card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm_stat','-', unc_dict["bkg_yield_stat"])
                card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm_syst','-', unc_dict["bkg_yield_syst"])
            card += '%-12s     lnN     %-33f%-33s\n' % ('lumi',unc_dict["lumi"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('PU',unc_dict["PU"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('JES',unc_dict["JES"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('JER',unc_dict["JER"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('uncl_en',unc_dict["uncl_en"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('PDF',unc_dict["PDF"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('alpha_s',unc_dict["alpha_s"],'-')
            card += '%-12s     lnN     %-33f%-33s\n' % ('tau',unc_dict["tau"],'-')
            
            print card

            contam_lab = "_ctau"+str(ct)
            if scale_mu!=1.:
                contam_lab+='_scale_mu_'+str(scale_mu).replace(".","p")
            if contamination:
                contam_lab+='_contamination'
            #exit()
            outname = DATACARDS+ s + contam_lab + label_2+'.txt'
            cardfile = open(outname, 'w')
            cardfile.write(card)
            cardfile.close()
            print "Info: " , outname, " written"
        
            original_location = os.popen("pwd").read()
            os.chdir("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit")
            #os.system("pwd")
            print "\n"

            #combine commands
            #Limits
            if run_combine:
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
                workspace = s+contam_lab+".root"
                #writes directly without displaying errors
                #tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[s]['mass']) + " | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ".txt\n"
                #tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
                #tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ".txt\n"

                #print screen
                tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[s]['mass']) + " \n"
                tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
                tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 \n"
                if ct==1000:
                    print "Run fit diagnostics..."
                    print "Pulls"
                    print "combine -M FitDiagnostics " + outname + " --name " + s+contam_lab + " --plots --forceRecreateNLL -m "+ str(samples[s]['mass'])
                    #tmp += "combine -M FitDiagnostics " + outname + " --name " + s+contam_lab + " --plots --forceRecreateNLL -m "+ str(samples[s]['mass'])+"\n"
                    print "python /afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics"+s+contam_lab +".root --all --abs --pullDef relDiffAsymErrs -g pulls"+s+contam_lab +".root"
                    #tmp += "python /afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics"+s+contam_lab +".root --all --abs --pullDef relDiffAsymErrs -g pulls"+s+contam_lab +".root \n"
                    print "Impacts"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --doInitialFit --robustFit 1 --rMin -5 --rMax 5 \n"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --robustFit 1 --doFits --rMin -5 --rMax 5 \n"
                    #tmp += "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" -o impacts"+ s+contam_lab +".json \n"
                    #tmp += "plotImpacts.py -i impacts"+ s+contam_lab +".json -o impacts"+ s+contam_lab + " \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --doInitialFit --robustFit 1 --rMin -5 --rMax 5 \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" --robustFit 1 --doFits --rMin -5 --rMax 5 \n"
                    print "combineTool.py -M Impacts -d "+DATACARDS + "/" + workspace+" -m "+ str(samples[s]['mass']) +" -o impacts"+ s+contam_lab +".json \n"
                    print "plotImpacts.py -i impacts"+ s+contam_lab +".json -o impacts"+ s+contam_lab + " \n"
                    #exit()
                job = open("job.sh", 'w')
                job.write(tmp)
                job.close()
                os.system("sh job.sh > log.txt \n")
                os.system("\n")
                os.system("cat log.txt \n")
                os.system("cat log.txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + contam_lab+label_2+".txt  \n")
                os.system("cat log.txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + contam_lab+label_2+".txt\n")
                #os.system("cat "+ RESULTS + "/" + s + ".txt  \n")
                print "\n"
                print "Limits written in ", RESULTS + "/" + s + contam_lab + label_2+".txt"
                print "Significance written in ", RESULTS + "/Significance_" + s + contam_lab + label_2 + ".txt"
                print "*********************************************"

            os.chdir(original_location[:-1])
            os.system("eval `scramv1 runtime -sh`")
            os.system("pwd")
        

    #print "Aborting, simply calculating integrals . . . ."

'''
def evaluate_signal_contamination(tree_weight_dict,sign,main_pred_reg,extr_region,comb_fold_label="",regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=False,phi_cut=False,mu=1.):

    #Zero, we need to run the bkg pred on MC but with data SF (possible?)
    #First, we need background prediction from MC bin 1
    #Second, we add signal*mu and evaluate bin 1
    #Third, we compute the additional contribution to bin 2 based on WtoMN mistag * bin 1 signal
    #Fourth, new datacards with signal contamination (depending on mu)
    #Remember, we also need mu=0 (but signal in bin 2)
    #Also, bin2 signal scaled by the same mu, if mu=0 then keep the nominal mu=1 and call that datacard "_no_contamination" or so
    exit()

    PREDDIR = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+main_pred_reg+"/"
    pred_file_name = PREDDIR+"BkgPredResults_"+ERA+"_"+main_pred_reg

    pred_file_name+=comb_fold_label
    if eta_cut:
        pred_file_name+="_eta_1p0"
    if phi_cut==True:
        pred_file_name+="_phi_cut"

    if eta:
        pred_file_name+= "_vs_eta"
    if phi:
        pred_file_name+= "_vs_phi"


    with open(pred_file_name+".yaml","r") as f:
        print "\n"
        print "Info: opening dictionary in file "+pred_file_name+".yaml"
        print "Extrapolation region: ", extr_region+comb_fold_label
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()

    print "Inferring limits on absolute x-sec in fb"
    if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)
    DATACARDDIR = OUTPUTDIR+CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)

    if eta:
        DATACARDDIR += TAGVAR+"_vs_eta"+comb_fold_label
    else:
        if phi:
            DATACARDDIR += TAGVAR+"_vs_phi"+comb_fold_label
        else:
            DATACARDDIR += TAGVAR+comb_fold_label

    ###DATACARDDIR += TAGVAR+comb_fold_label
    if eta_cut:
        DATACARDDIR += "_eta_1p0"
    if phi_cut:
        DATACARDDIR += "_phi_cut"
    DATACARDDIR += "/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"datacards/"
    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    RESULTS = DATACARDDIR+"combine_results/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    #mass = []
    print "tree_weight_dict"
    print tree_weight_dict
    chainSignal = {}
    list_of_variables = [TAGVAR,"isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight",CUT]#"nLeptons"
    if ERA=="2018":
        list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        chainSignal[s] = TChain("tree")
        tree_weights = {}
        chain_entries_cumulative = {}
        chain_entries = {}
        array_size_tot = 0
        bin2 = np.array([])
        for l, ss in enumerate(samples[s]['files']):
            print "ss", ss
            chainSignal[s].Add(NTUPLEDIR + ss + '.root')
            tree_weights[l] = tree_weight_dict[s][ss]
            chainSignal[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chainSignal[s].GetEntries()
            if l==0:
                chain_entries[l]=chainSignal[s].GetEntries()
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
            print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isJetHT":
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , arrays["pt"]<25)
                elif CUT == "isDiJetMET":
                    #DiJetMET140
                    cut_mask = np.logical_and(arrays["HLT_PFJet140_v"]==True , np.logical_and(arrays["pt"]<100, np.logical_and(arrays["nCHSJetsAcceptanceCalo"]==2, arrays["MinSubLeadingJetMetDPhi"]<=0.4) ) )
                elif CUT == "isWtoMN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )

                elif CUT == "isWtoEN":
                    if "noMT" in REGION:
                        cut_mask = arrays[CUT]>0
                    else:    
                        cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                    #cut_mask = arrays[CUT]>0
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhi"]>0.5,arrays["MT"]<100) )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , np.logical_and(arrays["MinJetMetDPhiBarrel"]>0.5,arrays["MT"]<100) )
                    #enhance MET
                    #cut_mask = np.logical_and(arrays[CUT]>0, arrays["pt"]>100 )
                    #no MT
                    #cut_mask = arrays[CUT]>0
                elif CUT == "isSR":
                    cut_mask = (arrays[CUT]>0)
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhi"]>0.5 )
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)
                    
                #HEM
                if CUT == "isSR" and ERA==2018:
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM__pt_30_all_eta"]==0)))
                ###cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))

                if KILL_QCD:
                    print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                    #cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhiBarrel"]>0.5)

                if eta_cut and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    cut_mask = (cut_mask_eta.any()==True)

                if phi_cut and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)

                if phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)


                #SR cut
                cut_mask = np.logical_and( cut_mask, arrays[TAGVAR]>1 )
                tag = arrays[TAGVAR][cut_mask] !=0
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                del arrays
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*signalMultFactor
                bin2 = np.concatenate( (bin2,np.multiply(tag,weight)) )

            del gen
        y_2 = np.sum(bin2)#*tree_weight --> already in w
        e_2 = np.sqrt( sum(x*x for x in bin2) ).sum()#*tree_weight --> already in w

        #for simplicity to avoid overwriting
        #if eta_cut:
        #    res_reg = extr_region+add_label#+label_2
        #else:
        #    res_reg = extr_region+add_label#+label_2
        res_reg = extr_region+comb_fold_label
        
        #print s, y_2, " +- ", e_2
        #print results
        key_dict = results[res_reg].keys()[0]
        if pred_unc==0:
            pred_unc = abs(results[res_reg][key_dict]['pred_2_from_1'] - results[res_reg][key_dict]['pred_2'])/(results[res_reg][key_dict]['pred_2_from_1']+results[res_reg][key_dict]['pred_2'])/2.

        #*******************************************************#
        #                                                       #
        #                      Datacard                         #
        #                                                       #
        #*******************************************************#
        
        card  = 'imax 1\n'#n of bins
        card += 'jmax *\n'#n of backgrounds
        card += 'kmax *\n'#n of nuisance parmeters
        card += '-----------------------------------------------------------------------------------\n'
        card += 'bin                      %s\n' % CHAN
        card += 'observation              %s\n' % '-1.0'
        card += '-----------------------------------------------------------------------------------\n'
        card += 'bin                      %-33s%-33s\n' % (CHAN, CHAN)
        card += 'process                  %-33s%-33s\n' % (s, 'Bkg')
        card += 'process                  %-33s%-33s\n' % ('0', '1')
        card += 'rate                     %-33f%-33f\n' % (y_2, results[res_reg][key_dict]['pred_2_from_1'])
        #kill QCD
        #card += 'rate                     %-33f%-33f\n' % (y_2, max(results[res_reg][key_dict]['pred_2_from_1'],results[res_reg][key_dict]['pred_2']))
        card += '-----------------------------------------------------------------------------------\n'
        #Syst uncertainties
        card += '%-12s     lnN     %-33f%-33s\n' % ('sig_norm',1.+e_2/y_2,'-')
        card += '%-12s     lnN     %-33s%-33f\n' % ('bkg_norm','-', 1.+pred_unc)
        card += '%-12s     lnN     %-33f%-33s\n' % ('lumi',1.025,'-')
        

        print card

        outname = DATACARDS+ s + '.txt'
        cardfile = open(outname, 'w')
        cardfile.write(card)
        cardfile.close()
        print "Info: " , outname, " written"
        
        original_location = os.popen("pwd").read()
        os.chdir("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit")
        #os.system("pwd")
        print "\n"

        #combine commands
        #Limits
        if run_combine:
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
            workspace = s+".root"
            #writes directly without displaying errors
            #tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[s]['mass']) + " | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ".txt\n"
            #tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
            #tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ".txt\n"

            #print screen
            tmp += "combine -M AsymptoticLimits --datacard " + outname + "  --run blind -m " + str(samples[s]['mass']) + " \n"
            tmp += "text2workspace.py " + outname + " " + " -o " + DATACARDS + "/" + workspace+"\n"
            tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 \n"
            job = open("job.sh", 'w')
            job.write(tmp)
            job.close()
            os.system("sh job.sh > log.txt \n")
            os.system("\n")
            os.system("cat log.txt \n")
            os.system("cat log.txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ".txt  \n")
            os.system("cat log.txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ".txt\n")
            #os.system("cat "+ RESULTS + "/" + s + ".txt  \n")
            print "\n"
            print "Limits written in ", RESULTS + "/" + s + ".txt"
            print "Significance written in ", RESULTS + "/Significance_" + s + ".txt"
            print "*********************************************"

        os.chdir(original_location[:-1])
        os.system("eval `scramv1 runtime -sh`")
        os.system("pwd")
        

    #print "Aborting, simply calculating integrals . . . ."
'''

def combine_datacards(sign,comb_fold_label="",regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=True,phi_cut=False):

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        add_label+="_eta_1p0"

    if phi_cut:
        print "Apply acceptance cut phi"
        add_label+="_phi_cut"

    OUT_2016_BF = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_2016_SR_2017_signal/"+CHAN+"/"+TAGVAR+comb_fold_label+"_B-F"+"/"
    OUT_2016_GH = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_2016_SR_2017_signal/"+CHAN+"/"+TAGVAR+comb_fold_label+"_G-H"+"/"
    OUT_2017 = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_2017_SR/"+CHAN+"/"+TAGVAR+comb_fold_label+"/"
    OUT_2018 = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_2018_SR_2017_signal/"+CHAN+"/"+TAGVAR+comb_fold_label+"/"

    print "Inferring limits on absolute x-sec in fb"
    DATACARDDIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_combination/"+CHAN+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDDIR += TAGVAR+comb_fold_label+"/"
    if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"datacards/"
    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    RESULTS = DATACARDDIR+"combine_results/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

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


    for i,s in enumerate(sign):
        card_2016_BF = OUT_2016_BF + 'datacards/'  + s + '.txt'
        card_2016_GH = OUT_2016_GH + 'datacards/'  + s + '.txt'
        card_2017 = OUT_2017 + 'datacards/'  + s + '.txt'
        card_2018 = OUT_2018 + 'datacards/'  + s + '.txt'
        card_comb = DATACARDS + s + '.txt'
        print card_2016_BF
        print card_2016_GH
        print card_2017
        print card_2018
        print card_comb
        
        original_location = os.popen("pwd").read()
        os.chdir("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit")
        #os.system("pwd")
        print "\n"

        #combine commands
        #Limits
        if run_combine:
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
            workspace = s+".root"
            tmp += "combineCards.py " + card_2016_BF + " " + card_2016_GH + " " + card_2017 + " " + card_2018 + " > " + card_comb + " \n"
            tmp += "combine -M AsymptoticLimits --datacard " + card_comb + "  --run blind -m " + str(samples[s]['mass']) + " \n"
            tmp += "text2workspace.py " + card_comb + " " + " -o " + DATACARDS + "/" + workspace+"\n"
            tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 \n"
            job = open("job.sh", 'w')
            job.write(tmp)
            job.close()
            os.system("sh job.sh > log.txt \n")
            os.system("\n")
            os.system("cat log.txt \n")
            os.system("cat log.txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ".txt  \n")
            os.system("cat log.txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ".txt\n")
            #os.system("cat "+ RESULTS + "/" + s + ".txt  \n")
            print "\n"
            print "Limits written in ", RESULTS + "/" + s + ".txt"
            print "Significance written in ", RESULTS + "/Significance_" + s + ".txt"
            print "*********************************************"

        os.chdir(original_location[:-1])
        os.system("eval `scramv1 runtime -sh`")
        os.system("pwd")

def combine_datacards_AN(sign,comb_fold_label="",regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,pred_unc=0,run_combine=False,eta=False,phi=False,eta_cut=True,phi_cut=False):

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        add_label+="_eta_1p0"

    if phi_cut:
        print "Apply acceptance cut phi"
        add_label+="_phi_cut"

    print "Inferring limits on absolute x-sec in fb"
    DATACARDDIR = OUTPUTDIR+CHAN+'/'
    DATACARDDIR += TAGVAR

    if eta_cut:
        DATACARDDIR += "_eta_1p0"
    if phi_cut:
        DATACARDDIR += "_phi_cut"

    if eta:
        DATACARDDIR += "_vs_eta"+comb_fold_label
    else:
        if phi:
            DATACARDDIR += "_vs_phi"+comb_fold_label
        else:
            DATACARDDIR += comb_fold_label

    #if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
    DATACARDS = DATACARDDIR+"/datacards/"
    #if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    RESULTS = DATACARDDIR+"/combine_results/"
    if not os.path.isdir(RESULTS): os.mkdir(RESULTS)





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


    for i,s in enumerate(sign):
        for ct in np.append(np.array([ctaupoint]),ctaus):
            ct_lab = "_ctau"+str(ct)
            card_BF = DATACARDS + "/" + s + ct_lab+"_BF.txt"
            card_REST = DATACARDS + "/" + s + ct_lab+"_REST.txt"
            card_comb = DATACARDS + "/" + s +ct_lab+".txt"
            original_location = os.popen("pwd").read()
            os.chdir("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit")
            #os.system("pwd")
            print "\n"

            #combine commands
            #Limits
            if run_combine:
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
                workspace = s+ct_lab+".root"
                tmp += "combineCards.py " + card_BF + " " + card_REST + " > " + card_comb + " \n"
                tmp += "combine -M AsymptoticLimits --datacard " + card_comb + "  --run blind -m " + str(samples[s]['mass']) + " \n"
                tmp += "text2workspace.py " + card_comb + " " + " -o " + DATACARDS + "/" + workspace+"\n"
                tmp += "combine -M Significance " + DATACARDS + "/" + workspace + "  -t -1 --expectSignal=1 \n"
                job = open("job.sh", 'w')
                job.write(tmp)
                job.close()
                print tmp
                os.system("sh job.sh > log.txt \n")
                os.system("\n")
                os.system("cat log.txt \n")
                os.system("cat log.txt | grep -e Observed -e Expected | awk '{print $NF}' > " + RESULTS + "/" + s + ct_lab+".txt  \n")
                os.system("cat log.txt | grep -e Significance: | awk '{print $NF}' > " + RESULTS + "/Significance_" + s + ct_lab+".txt\n")
                os.system("cat "+ RESULTS + "/" + s + ".txt  \n")
                print "\n"
                print "Limits written in ", RESULTS + "/" + s + ".txt"
                print "Significance written in ", RESULTS + "/Significance_" + s + ".txt"
                print "*********************************************"
                
            os.chdir(original_location[:-1])
            os.system("eval `scramv1 runtime -sh`")
            os.system("pwd")



def plot_limits_vs_mass(sign,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False):

    LUMI = 137.4*1000
    print "Using full run 2 lumi!! ", LUMI

    #signalMultFactor = 1.
    if combination:
        DATACARDDIR = OUTPUTDIR+CHAN+"/"+TAGVAR
        PLOTLIMITSDIR      = "plots/Limits/v5_calo_AOD_"+ERA+"_"+REGION+"_combination/"+TAGVAR#"/"#"_2017_signal/"#
        print PLOTLIMITSDIR
        exit()
    else:
        DATACARDDIR = OUTPUTDIR+CHAN+"/"+TAGVAR
        #PLOTLIMITSDIR      = "plots/Limits_AN_combi/v5_calo_AOD_"+ERA+"_"+REGION+"/"+TAGVAR#_signal/"#"/"#
        PLOTLIMITSDIR      = "plots/Limits_AN/v5_calo_AOD_"+ERA+"_"+REGION+"/"+TAGVAR#_signal/"#"/"#

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        DATACARDDIR+= "_eta_1p0"
        PLOTLIMITSDIR+= "_eta_1p0"
    if phi_cut:
        print "Apply acceptance cut phi"
        DATACARDDIR+= "_phi_cut"
        PLOTLIMITSDIR+="_phi_cut"
    if eta:
        DATACARDDIR+="_vs_eta"
        PLOTLIMITSDIR+="_vs_eta"
    if phi:
        DATACARDDIR+="_vs_phi"
        PLOTLIMITSDIR+="_vs_phi"


    print "Here I miss combine results or datacards"

    contam_lab = "_ctau"+str(ctaupoint)
    if scale_mu!=1.:
        contam_lab+='_scale_mu_'+str(scale_mu).replace(".","p")
    if contamination:
        contam_lab+='_contamination'

    DATACARDDIR += comb_fold_label
    PLOTLIMITSDIR+=comb_fold_label+"/"
    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(PLOTLIMITSDIR):
        os.mkdir(PLOTLIMITSDIR)
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
    mass = []
    theory = []
    mean_val = {}
    sigma_1_up = {}
    sigma_1_down = {}
    sigma_2_up = {}
    sigma_2_down = {}

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print samples[s]['ctau']
        print "theory x-sec (pb): ", sample[ samples[s]['files'][0] ]['xsec']
        m = samples[s]['mass']
        mass.append(m)
        theory.append( 1000.*sample[ samples[s]['files'][0] ]['xsec']  )
        card_name = RESULTS + "/" + s + contam_lab+".txt"
        card = open( card_name, 'r')
        val = card.read().splitlines()
        if len(val) == 0:
            continue
        sigma_2_down[m] = float(val[0])
        sigma_1_down[m] = float(val[1])
        mean_val[m]     = float(val[2])
        sigma_1_up[m]   = float(val[3])
        sigma_2_up[m]   = float(val[4])

    mean_arr = []
    sigma_1_up_arr = []
    sigma_1_down_arr = []
    sigma_2_up_arr = []
    sigma_2_down_arr = []
    Obs0s = TGraph()
    Exp0s = TGraph()
    Exp1s = TGraphAsymmErrors()
    Exp2s = TGraphAsymmErrors()
    Theory = TGraph()

    mass = np.sort(np.array(mass))
    theory = -np.sort(-np.array(theory))
    print mass
    print theory
    n=0
    for m in mass:
        mean_arr.append(mean_val[m])
        sigma_1_up_arr.append(sigma_1_up[m])
        sigma_1_down_arr.append(sigma_1_down[m])
        sigma_2_up_arr.append(sigma_2_up[m])
        sigma_2_down_arr.append(sigma_2_down[m])

        Exp0s.SetPoint(n, m, mean_val[m])
        Exp1s.SetPoint(n, m, mean_val[m])
        Exp1s.SetPointError(n, 0., 0., mean_val[m]-sigma_1_down[m], sigma_1_up[m]-mean_val[m])
        Exp2s.SetPoint(n, m, mean_val[m])
        Exp2s.SetPointError(n, 0., 0., mean_val[m]-sigma_2_down[m], sigma_2_up[m]-mean_val[m])

        Theory.SetPoint(n, m, theory[n])#just a list with no index
        n+=1
    print mass
    print mean_arr

    Exp2s.SetLineWidth(2)
    Exp2s.SetLineStyle(1)
    Exp0s.SetLineStyle(2)
    Exp0s.SetLineWidth(3)
    Exp1s.SetFillColor(417)
    Exp1s.SetLineColor(417)
    Exp2s.SetFillColor(800)
    Exp2s.SetLineColor(800)
    Exp2s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
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
    leg = TLegend(0.45+0.15+0.05, top-nitems*0.3/5., 0.75+0.15+0.03, top)
    leg.SetBorderSize(0)
    leg.SetHeader("95% CL limits, c#tau_{"+particle+"}="+str(int(ctaupoint/1000))+" m")
    leg.SetTextSize(0.04)

    c1 = TCanvas("c1", "Exclusion Limits", 800, 600)
    c1.cd()
    c1.SetGridx()
    c1.SetGridy()
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
        leg.AddEntry(Theory, "Theory", "l")
        leg.AddEntry(Exp0s,  "Expected", "l")
        leg.AddEntry(Exp1s, "#pm 1 std. deviation", "f")
        leg.AddEntry(Exp2s, "#pm 2 std. deviations", "f")

    Exp2s.GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
    if CHAN=="SUSY":
        #Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
        if signalMultFactor == 0.001:
            Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b}) (fb)")
        else:
            Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
    Exp2s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp1s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp0s.GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
    Exp2s.SetMinimum(0.09)
    Exp2s.SetMaximum(10001)
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
        Theory.Draw("SAME, L")
    leg.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",left_marg_CMS=0.3,onTop=True)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    #drawTagVar(TAGVAR)

    OUTSTRING = PLOTLIMITSDIR+"/Exclusion"+contam_lab#+TAGVAR+comb_fold_label+"_"+CHAN
    newFile = TFile(PLOTLIMITSDIR+"/Exclusion" + contam_lab+".root", "RECREATE")#+"_"+TAGVAR+comb_fold_label+ ".root", "RECREATE")
    newFile.cd()
    Exp0s.Write("pl"+str(ctaupoint)+"_exp")
    Exp1s.Write("pl"+str(ctaupoint)+"_1sigma")
    Exp2s.Write("pl"+str(ctaupoint)+"_2sigma")
    c1.Write()
    newFile.Close()
    print "Info: written file: ", PLOTLIMITSDIR+"/Exclusion" + contam_lab+".root"#"_"+TAGVAR+comb_fold_label+".root"

    c1.Print(OUTSTRING+".png")
    c1.Print(OUTSTRING+".pdf")
    c1.Close()

def plot_limits_vs_ctau(sign,comb_fold_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,scale_mu=1.,contamination=False):

    LUMI = 137.4*1000
    print "Using full run 2 lumi!! ", LUMI

    #signalMultFactor = 1.
    if combination:
        DATACARDDIR = ""#"/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v5_calo_AOD_combination/"+CHAN+"/"+TAGVAR
        PLOTLIMITSDIR      = ""#"plots/Limits/v5_calo_AOD_combination/"+TAGVAR#"/"#"_2017_signal/"#
    else:
        DATACARDDIR = OUTPUTDIR+CHAN+"/"+TAGVAR
        #PLOTLIMITSDIR      = "plots/Limits_AN_combi/v5_calo_AOD_"+ERA+"_"+REGION+"/"+TAGVAR#_signal/"#"/"#
        PLOTLIMITSDIR      = "plots/Limits_AN/v5_calo_AOD_"+ERA+"_"+REGION+"/"+TAGVAR#_signal/"#"/"#

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        DATACARDDIR+= "_eta_1p0"
        PLOTLIMITSDIR+= "_eta_1p0"
    if phi_cut:
        print "Apply acceptance cut phi"
        DATACARDDIR+= "_phi_cut"
        PLOTLIMITSDIR+="_phi_cut"
    if eta:
        DATACARDDIR+="_vs_eta"
        PLOTLIMITSDIR+="_vs_eta"
    if phi:
        DATACARDDIR+="_vs_phi"
        PLOTLIMITSDIR+="_vs_phi"


    print "Here I miss combine results or datacards"


    DATACARDDIR += comb_fold_label
    PLOTLIMITSDIR+=comb_fold_label+"/"
    RESULTS = DATACARDDIR + "/combine_results/"

    if not os.path.isdir(PLOTLIMITSDIR):
        os.mkdir(PLOTLIMITSDIR)
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
    mass = []
    theory = []
    mean_val = {}
    sigma_1_up = {}
    sigma_1_down = {}
    sigma_2_up = {}
    sigma_2_down = {}

    for i,s in enumerate(sign):
        print "m chi: ",samples[s]['mass']
        print s
        print "theory x-sec (pb): ", sample[ samples[s]['files'][0] ]['xsec']
        mass = samples[s]['mass']
        #Initialize here dictionaries
        #Dictionary with various predictions
        theory = []
        mean_val = {}
        sigma_1_up = {}
        sigma_1_down = {}
        sigma_2_up = {}
        sigma_2_down = {}

        #Here loop on card names and fill the values
        #Add also original ctau
        #ctaus = np.array([2000,30000])
        #ctaus_old = ctaus
        #ctaus = np.append(ctaus_old,np.array([ctaupoint]))
        #ctaus = np.sort(ctaus)
        for ct in np.sort(np.append(ctaus,np.array([ctaupoint]))):
            print ct
            contam_lab = ""
            if scale_mu!=1.:
                contam_lab+='_scale_mu_'+str(scale_mu).replace(".","p")
            if contamination:
                contam_lab+='_contamination'
            contam_lab += "_ctau"+str(ct)
            card_name = RESULTS + "/" + s + contam_lab+".txt"
            card = open( card_name, 'r')
            val = card.read().splitlines()
            if len(val) == 0:
                continue
            sigma_2_down[ct] = float(val[0])
            sigma_1_down[ct] = float(val[1])
            mean_val[ct]     = float(val[2])
            sigma_1_up[ct]   = float(val[3])
            sigma_2_up[ct]   = float(val[4])
            theory.append( 1000.*sample[ samples[s]['files'][0] ]['xsec']  )

        mean_arr = []
        sigma_1_up_arr = []
        sigma_1_down_arr = []
        sigma_2_up_arr = []
        sigma_2_down_arr = []
        Obs0s = TGraph()
        Exp0s = TGraph()
        Exp1s = TGraphAsymmErrors()
        Exp2s = TGraphAsymmErrors()
        Theory = TGraph()

        n=0
        for ct in np.sort(np.append(ctaus,np.array([ctaupoint]))):
            if ct in mean_val.keys():
                mean_arr.append(mean_val[ct])
                sigma_1_up_arr.append(sigma_1_up[ct])
                sigma_1_down_arr.append(sigma_1_down[ct])
                sigma_2_up_arr.append(sigma_2_up[ct])
                sigma_2_down_arr.append(sigma_2_down[ct])
            
                Exp0s.SetPoint(n, ct/1000., mean_val[ct])
                Exp1s.SetPoint(n, ct/1000., mean_val[ct])
                Exp1s.SetPointError(n, 0., 0., mean_val[ct]-sigma_1_down[ct], sigma_1_up[ct]-mean_val[ct])
                Exp2s.SetPoint(n, ct/1000., mean_val[ct])
                Exp2s.SetPointError(n, 0., 0., mean_val[ct]-sigma_2_down[ct], sigma_2_up[ct]-mean_val[ct])

                Theory.SetPoint(n, ct/1000., theory[n])#just a list with no index
                n+=1

        print ctaus
        print mean_arr

        Exp2s.SetLineWidth(2)
        Exp2s.SetLineStyle(1)
        Exp0s.SetLineStyle(2)
        Exp0s.SetLineWidth(3)
        Exp1s.SetFillColor(417)
        Exp1s.SetLineColor(417)
        Exp2s.SetFillColor(800)
        Exp2s.SetLineColor(800)
        Exp2s.GetXaxis().SetTitle("c #tau_{"+particle+"} (m)")
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
        leg.SetHeader("95% CL limits, m_{"+particle+"}="+str(mass)+" GeV")
        leg.SetTextSize(0.04)

        c1 = TCanvas("c1", "Exclusion Limits", 800, 600)
        c1.cd()
        c1.SetGridx()
        c1.SetGridy()
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
            leg.AddEntry(Theory, "Theory", "l")
            leg.AddEntry(Exp0s,  "Expected", "l")
            leg.AddEntry(Exp1s, "#pm 1 std. deviation", "f")
            leg.AddEntry(Exp2s, "#pm 2 std. deviations", "f")

        Exp2s.GetYaxis().SetTitle("#sigma(H) B("+particle+particle+" #rightarrow b #bar{b} b #bar{b})/#sigma_{SM}")
        if CHAN=="SUSY":
            #Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
            if signalMultFactor == 0.001:
                Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b}) (fb)")
            else:
                Exp2s.GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
        Exp2s.GetXaxis().SetTitle("c #tau_{"+particle+"} (m)")
        Exp1s.GetXaxis().SetTitle("c #tau_{"+particle+"} (m)")
        Exp0s.GetXaxis().SetTitle("c #tau_{"+particle+"} (m)")
        Exp2s.SetMinimum(0.09)
        Exp2s.SetMaximum(10001)
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
            Theory.Draw("SAME, L")
            leg.Draw()
            
        if PRELIMINARY:
            drawCMS(samples, LUMI, "Preliminary",left_marg_CMS=0.3,onTop=True)
        else:
            drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

        drawRegion(CHAN,top=0.7)
        drawAnalysis("LL"+CHAN)
        #drawTagVar(TAGVAR)

        OUTSTRING = PLOTLIMITSDIR+"/Exclusion_vs_ctau_m_"+str(mass)#+TAGVAR+comb_fold_label+"_"+CHAN
        newFile = TFile(PLOTLIMITSDIR+"/Exclusion_vs_ctau_m_" + str(mass)+".root", "RECREATE")#+"_"+TAGVAR+comb_fold_label+ ".root", "RECREATE")
        newFile.cd()
        Exp0s.Write("m"+str(mass)+"_exp")
        Exp1s.Write("m"+str(mass)+"_1sigma")
        Exp2s.Write("m"+str(mass)+"_2sigma")
        c1.Write()
        newFile.Close()
        print "Info: written file: ", PLOTLIMITSDIR+"/Exclusion_vs_ctau_m_" +str(mass)+".root"#"_"+TAGVAR+comb_fold_label+".root"

        c1.Print(OUTSTRING+".png")
        c1.Print(OUTSTRING+".pdf")
        c1.Close()

def combine_limits(file_names,labels,plot_label="",combination=False,eta=False,phi=False,eta_cut=False,phi_cut=False,LUMI=LUMI):

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

    colors = [801,856,825,881,2,602,880,798,5,6]
    lines = [1,2,1,2,1,2,1,2,1,2,1,2,]
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
    c2.GetPad(0).SetLogy()
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

    for i,l in enumerate(file_names):
        filename = TFile(l, "READ")
        #graph_1sigma[i] = TGraph()
        #graph_exp[i] = TGraph()
        graph_1sigma[i] = filename.Get("pl"+str(ctaupoint)+"_1sigma")
        graph_exp[i] = filename.Get("pl"+str(ctaupoint)+"_exp")
        graph_1sigma[i].SetLineColor(colors[i])
        graph_1sigma[i].SetLineStyle(lines[i])
        graph_exp[i].SetLineStyle(lines[i])
        graph_1sigma[i].SetLineWidth(3)
        graph_1sigma[i].SetFillStyle(3003)#3002
        graph_1sigma[i].SetFillColorAlpha(colors[i],0.5)
        if signalMultFactor == 0.001:
            graph_1sigma[i].GetYaxis().SetTitle("#sigma("+particle+particle+") (fb)")# B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b}) (fb)")
        else:
            graph_1sigma[i].GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
        graph_1sigma[i].GetXaxis().SetNoExponent(True)
        graph_1sigma[i].GetXaxis().SetMoreLogLabels(True)
        graph_1sigma[i].GetXaxis().SetTitleSize(0.048)
        graph_1sigma[i].GetYaxis().SetTitleSize(0.048)
        graph_1sigma[i].GetYaxis().SetTitleOffset(0.8)
        graph_1sigma[i].GetXaxis().SetTitleOffset(0.9)
        graph_1sigma[i].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        graph_exp[i].SetLineColor(colors[i])
        graph_exp[i].SetMarkerColor(colors[i])
        graph_exp[i].SetFillColorAlpha(colors[i],0.5)
        if signalMultFactor == 0.001:
            graph_exp[i].GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b}) (fb)")
            graph_1sigma[i].SetMinimum(0.09 if combination else 0.9)
            graph_1sigma[i].SetMaximum(101 if combination else 101)
            graph_exp[i].SetMinimum(0.09 if combination else 0.009)
            graph_exp[i].SetMaximum(101 if combination else 1001)
        else:
            graph_exp[i].GetYaxis().SetTitle("#sigma("+particle+particle+") B( h #tilde{G} h #tilde{G} #rightarrow b #bar{b} b #bar{b})/#sigma_{SUSY}")
            graph_1sigma[i].SetMinimum(0.005)
            graph_1sigma[i].SetMaximum(1)
        graph_exp[i].GetXaxis().SetTitle("m_{"+particle+"} (GeV)")
        graph_exp[i].GetXaxis().SetNoExponent(True)
        graph_exp[i].GetXaxis().SetMoreLogLabels(True)
        graph_exp[i].GetXaxis().SetTitleSize(0.048)
        graph_exp[i].GetYaxis().SetTitleSize(0.048)
        graph_exp[i].GetYaxis().SetTitleOffset(0.8)
        graph_exp[i].GetXaxis().SetTitleOffset(0.9)
        if i == 0:
            #graph_1sigma[i].Draw("AL3")
            #graph_exp[i].Draw("SAME,L3")
            graph_exp[i].SetMarkerStyle(20)
            graph_exp[i].Draw("APL3")
        else:
            #graph_1sigma[i].Draw("SAME,L3")
            if i%2==0:
                graph_exp[i].SetMarkerStyle(20)
                graph_exp[i].Draw("SAME,PL3")
            else:
                graph_exp[i].SetMarkerStyle(24)
                graph_exp[i].Draw("SAME,PL3")
        print "Drawn: expected ", l
        leg.AddEntry(graph_exp[i],labels[i],"L")
        filename.Close()
    leg.Draw()

    if PRELIMINARY:
        drawCMS(samples, LUMI, "Preliminary",onTop=True,left_marg_CMS=0.1)
    else:
        drawCMS(samples, LUMI, "",left_marg_CMS=0.32)

    drawRegion(CHAN,top=0.7)
    drawAnalysis("LL"+CHAN)
    drawTagVar(TAGVAR)

    OUTSTRING = PLOTLIMITSDIR+"/Exclusion_ctau"+str(ctaupoint)+"_combination_"+CHAN
    c2.Print(OUTSTRING+plot_label+".png")
    c2.Print(OUTSTRING+plot_label+".pdf")
    c2.Close()


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

        
'''
def GetEffWeightBin1(event, TEff):
    EffW=1.
    EffWUp=1.
    EffWLow=1.
    cnt=0
    p1=0
    p1Up=0
    p1Low=0

    n_untagged = 0
    n_tagged = 0
    for j in range(event.nCHSJetsAcceptanceCalo):
        if (event.Jets[j].muEFrac<0.6 and event.Jets[j].eleEFrac<0.6 and event.Jets[j].photonEFrac<0.8 and event.Jets[j].timeRecHitsEB>-1):
            if(event.Jets[j].sigprob<=0.996):#bin0 selection
                n_untagged+=1
                binN = TEff.GetPassedHistogram().FindBin(event.Jets[j].pt)
                eff  = TEff.GetEfficiency(binN)
                errUp = TEff.GetEfficiencyErrorUp(binN)
                errLow = TEff.GetEfficiencyErrorLow(binN)
                effUp = eff+errUp
                effLow = eff-errLow
                p1 = p1 + eff
                p1Up = p1Up + effUp
                p1Low = p1Low + effLow
            else:
                n_tagged+=1

    EffW = p1
    EffWup = p1Up
    EffWLow = p1Low
    return n_untagged, n_tagged, EffW, EffWUp, EffWLow

def GetEffWeightBin2(event, TEff, n):
    EffW=1.
    EffWUp=1.
    EffWLow=1.
    cnt=0
    p2=0
    p2Up=0
    p2Low=0

    #n_untagged = 0
    #n_tagged = 0
    for j1 in range(event.nCHSJetsAcceptanceCalo):
        if (event.Jets[j1].muEFrac<0.6 and event.Jets[j1].eleEFrac<0.6 and event.Jets[j1].photonEFrac<0.8 and event.Jets[j1].timeRecHitsEB>-1 and event.Jets[j1].sigprob<=0.996):#bin0 selection n_untagged+=1
                binN1 = TEff.GetPassedHistogram().FindBin(event.Jets[j1].pt)
                eff1  = TEff.GetEfficiency(binN1)
                errUp1 = TEff.GetEfficiencyErrorUp(binN1)
                errLow1 = TEff.GetEfficiencyErrorLow(binN1)
                effUp1 = eff1 + errUp1
                effLow1 = eff1 - errLow1

                #Second loop: find all the jet pairs
                for j2 in range(event.nCHSJetsAcceptanceCalo):
                    if (event.Jets[j2].muEFrac<0.6 and event.Jets[j2].eleEFrac<0.6 and event.Jets[j2].photonEFrac<0.8 and event.Jets[j2].timeRecHitsEB>-1 and event.Jets[j2].sigprob<=0.996):#bin0 selection
                        binN2 = TEff.GetPassedHistogram().FindBin(event.Jets[j2].pt)
                        eff2  = TEff.GetEfficiency(binN2)
                        errUp2 = TEff.GetEfficiencyErrorUp(binN2)
                        errLow2 = TEff.GetEfficiencyErrorLow(binN2)
                        effUp2 = eff2 + errUp2
                        effLow2 = eff2 - errLow2
                        if(j2>j1):
                            #print("=======")
                            #print("Event n. %d")%(n)
                            #print("Pair jets %d %d")%(j1,j2)
                            p2 = p2 + eff1*eff2
                            p2Up = p2Up + effUp1*effUp2
                            p2Low = p2Low + effLow1*effLow2

    EffW = p2
    EffWup = p2Up
    EffWLow = p2Low
    #return n_untagged, n_tagged, EffW, EffWUp, EffWLow
    return EffW, EffWUp, EffWLow

def GetEffWeightBin1New_one_eff(event, TEff):
    EffW=0.
    EffWUp=0.
    EffWLow=0.
    cnt=0
    p1=0
    p1Up=0
    p1Low=0

    n_j = 0
    n_untagged = 0
    n_tagged = 0
    for j in range(event.nCHSJetsAcceptanceCalo):
        n_j += 1
        if(event.Jets[j].sigprob<=0.996):#bin0 selection
            n_untagged+=1
            binN = TEff.GetPassedHistogram().FindBin(event.Jets[j].pt)
            eff  = TEff.GetEfficiency(binN)
            errUp = TEff.GetEfficiencyErrorUp(binN)
            errLow = TEff.GetEfficiencyErrorLow(binN)
            effUp = eff+errUp
            effLow = eff-errLow
            p1 += eff
            p1Up += effUp
            p1Low += effLow
        else:
            n_tagged+=1

    EffW = p1
    EffWup = p1Up
    EffWLow = p1Low
    return n_j, n_untagged, n_tagged, EffW, EffWUp, EffWLow

def GetEffWeightBin2New_one_eff(event, TEff, n):
    EffW=0.
    EffWUp=0.
    EffWLow=0.
    cnt=0
    p2=0
    p2Up=0
    p2Low=0

    #n_untagged = 0
    #n_tagged = 0
    for j1 in range(event.nCHSJetsAcceptanceCalo):
        if (event.Jets[j1].sigprob<=0.996):#bin0 selection n_untagged+=1
            binN1 = TEff.GetPassedHistogram().FindBin(event.Jets[j1].pt)
            eff1  = TEff.GetEfficiency(binN1)
            errUp1 = TEff.GetEfficiencyErrorUp(binN1)
            errLow1 = TEff.GetEfficiencyErrorLow(binN1)
            effUp1 = eff1 + errUp1
            effLow1 = eff1 - errLow1

            #Second loop: find all the jet pairs
            for j2 in range(event.nCHSJetsAcceptanceCalo):
                if (event.Jets[j2].sigprob<=0.996):#bin0 selection
                    binN2 = TEff.GetPassedHistogram().FindBin(event.Jets[j2].pt)
                    eff2  = TEff.GetEfficiency(binN2)
                    errUp2 = TEff.GetEfficiencyErrorUp(binN2)
                    errLow2 = TEff.GetEfficiencyErrorLow(binN2)
                    effUp2 = eff2 + errUp2
                    effLow2 = eff2 - errLow2
                    if(j2>j1):
                        #print("=======")
                        #print("Event n. %d")%(n)
                        #print("Pair jets %d %d")%(j1,j2)
                        p2 = p2 + eff1*eff2
                        p2Up = p2Up + effUp1*effUp2
                        p2Low = p2Low + effLow1*effLow2

    EffW = p2
    EffWup = p2Up
    EffWLow = p2Low
    #return n_untagged, n_tagged, EffW, EffWUp, EffWLow
    return EffW, EffWUp, EffWLow

def GetEffWeightBin1New(event, TEff, check_closure):
    EffW={}
    EffWUp={}
    EffWLow={}
    #cnt={}
    p1={}
    p1Up={}
    p1Low={}

    if check_closure:
        dnn_threshold = 0.95
    else:
        dnn_threshold = 0.996

    for r in TEff.keys():
        EffW[r]=0.
        EffWUp[r]=0.
        EffWLow[r]=0.
        #cnt[r]=0
        p1[r]=0
        p1Up[r]=0
        p1Low[r]=0

    n_j = 0
    n_untagged = 0
    n_tagged = 0
    for j in range(event.nCHSJetsAcceptanceCalo):
        n_j += 1
        if(event.Jets[j].sigprob<=dnn_threshold):
            n_untagged+=1
            for r in TEff.keys():
                binN = TEff[r].GetPassedHistogram().FindBin(event.Jets[j].pt)
                eff  = TEff[r].GetEfficiency(binN)
                errUp = TEff[r].GetEfficiencyErrorUp(binN)
                errLow = TEff[r].GetEfficiencyErrorLow(binN)
                effUp = eff+errUp
                effLow = eff-errLow
                p1[r] += eff
                p1Up[r] += effUp
                p1Low[r] += effLow
        else:
            if(check_closure):
                if(event.Jets[j].sigprob<0.996):
                    n_tagged+=1
                    #blind region with DNN>0.996
            else:
                n_tagged+=1

    for r in TEff.keys():
        EffW[r] = p1[r]
        EffWUp[r] = p1Up[r]
        EffWLow[r] = p1Low[r]
    return n_j, n_untagged, n_tagged, EffW, EffWUp, EffWLow

def GetEffWeightBin2New(event, TEff, n, check_closure):
    EffW={}
    EffWUp={}
    EffWLow={}
    cnt={}
    p2={}
    p2Up={}
    p2Low={}
    eff1={}
    effUp1={}
    effLow1={}
    eff2={}
    effUp2={}
    effLow2={}
    eff3={}
    effUp3={}
    effLow3={}
    eff4={}
    effUp4={}
    effLow4={}

    for r in TEff.keys():
        EffW[r]=0.
        EffWUp[r]=0.
        EffWLow[r]=0.
        cnt[r]=0
        p2[r]=0
        p2Up[r]=0
        p2Low[r]=0
        eff1[r]=0
        effUp1[r]=0
        effLow1[r]=0
        eff2[r]=0
        effUp2[r]=0
        effLow2[r]=0
        eff3[r]=0
        effUp3[r]=0
        effLow3[r]=0
        eff4[r]=0
        effUp4[r]=0
        effLow4[r]=0

    if check_closure:
        dnn_threshold = 0.95
    else:
        dnn_threshold = 0.996

    #n_untagged = 0
    #n_tagged = 0

    for j1 in range(event.nCHSJetsAcceptanceCalo):
        if (event.Jets[j1].sigprob<=dnn_threshold):#bin0 selection n_untagged+=1
            for r in TEff.keys():
                binN1 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j1].pt)
                eff1[r]  = TEff[r].GetEfficiency(binN1)
                errUp1 = TEff[r].GetEfficiencyErrorUp(binN1)
                errLow1 = TEff[r].GetEfficiencyErrorLow(binN1)
                effUp1[r] = eff1[r] + errUp1
                effLow1[r] = eff1[r] - errLow1

        #Second loop: find all the jet pairs
        for j2 in range(event.nCHSJetsAcceptanceCalo):
            if(j2>j1):
                if (event.Jets[j2].sigprob<=dnn_threshold):#bin0 selection
                    for r in TEff.keys():
                        binN2 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j2].pt)
                        eff2[r]  = TEff[r].GetEfficiency(binN2)
                        errUp2 = TEff[r].GetEfficiencyErrorUp(binN2)
                        errLow2 = TEff[r].GetEfficiencyErrorLow(binN2)
                        effUp2[r] = eff2[r] + errUp2
                        effLow2[r] = eff2[r] - errLow2
                        #print("=======")
                        #print("Event n. %d")%(n)
                        #print("n. jets %d")%(event.nCHSJetsAcceptanceCalo)
                        #print("Region: %s")%(r)
                        #print("Pair jets %d %d")%(j1,j2)
                        p2[r] = p2[r] + eff1[r]*eff2[r]
                        p2Up[r] = p2Up[r] + effUp1[r]*effUp2[r]
                        p2Low[r] = p2Low[r] + effLow1[r]*effLow2[r]

                
                #Third loop: find all the jet triplets
                for j3 in range(event.nCHSJetsAcceptanceCalo):
                    if(j3>j2):
                        if (event.Jets[j3].sigprob<=dnn_threshold):#bin0 selection
                            for r in TEff.keys():
                                binN3 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j3].pt)
                                eff3[r]  = TEff[r].GetEfficiency(binN3)
                                errUp3 = TEff[r].GetEfficiencyErrorUp(binN3)
                                errLow3 = TEff[r].GetEfficiencyErrorLow(binN3)
                                effUp3[r] = eff3[r] + errUp3
                                effLow3[r] = eff3[r] - errLow3
                                #print("=======")
                                #print("Event n. %d")%(n)
                                #print("Region: %s")%(r)
                                #print("Triplet jets %d %d %d")%(j1,j2,j3)
                                p2[r] = p2[r] + eff1[r]*eff2[r]*eff3[r]
                                p2Up[r] = p2Up[r] + effUp1[r]*effUp2[r]*effUp3[r]
                                p2Low[r] = p2Low[r] + effLow1[r]*effLow2[r]*effLow3[r]

                        #Fourth loop: find all the jet triplets
                        for j4 in range(event.nCHSJetsAcceptanceCalo):
                            if(j4>j3):
                                if (event.Jets[j4].sigprob<=dnn_threshold):#bin0 selection
                                    for r in TEff.keys():
                                        binN4 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j4].pt)
                                        eff4[r]  = TEff[r].GetEfficiency(binN4)
                                        errUp4 = TEff[r].GetEfficiencyErrorUp(binN4)
                                        errLow4 = TEff[r].GetEfficiencyErrorLow(binN4)
                                        effUp4[r] = eff4[r] + errUp4
                                        effLow4[r] = eff4[r] - errLow4
                                        #print("=======")
                                        #print("Event n. %d")%(n)
                                        #print("Region: %s")%(r)
                                        #print("Quadruplet jets %d %d %d %d")%(j1,j2,j3,j4)
                                        p2[r] = p2[r] + eff1[r]*eff2[r]*eff3[r]*eff4[r]
                                        p2Up[r] = p2Up[r] + effUp1[r]*effUp2[r]*effUp3[r]*effUp4[r]
                                        p2Low[r] = p2Low[r] + effLow1[r]*effLow2[r]*effLow3[r]*effLow4[r]            
                

    for r in TEff.keys():
        EffW[r] = p2[r]
        EffWUp[r] = p2Up[r]
        EffWLow[r] = p2Low[r]
    #return n_untagged, n_tagged, EffW, EffWUp, EffWLow
    return EffW, EffWUp, EffWLow
'''


# Just for comparison with CSC
'''
write_datacards(
    get_tree_weights(sign),
    sign,
    main_pred_reg = REGION,
    #extrapolation
    extr_region= "WtoMN",#["ZtoMM","WtoMN","TtoEM","JetHT","DiJetMET"],
    comb_fold_label = "_new_bins",
    #comb_fold_label = "_MinDPhi_0p5_eta_1p0_new_bins",#"_MinDPhi_0p5_new_bins",#"",
    regions_labels = ["","_MinDPhi_0p5","","","","","","","","","","","",""],
    datasets= ["","","","","","","","","","","","","","",""],
    add_label="",
    label_2="",#"_MinDPhi_0p5",#"",#"",#"_MinDPhi_0p5_new_bins",#"_extrapolation_0p996",
    check_closure=False,
    pred_unc = 1.,#here to boost it if needed
    run_combine = False,
    eta_cut = False
)
exit()
'''



samples_to_run = data#back#data#sign#data#back#data#back#data#back#data#sign#back#data#back#data#back+data#data#data+back#+data
#jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
#jet_tag = "_jet_1"
jet_tag = ""#+
#jet_tag="_QCD"
#jet_tag+="_closure_0p95"
clos = True#False#True#False#True
clos = False
#jet_tag += "_noMT"
#jet_tag += "_closure_0p95"
#jet_tag = "_A-B"
#kill QCD

#jet_tag += "_B-F"#"_G-H"#"_B-F"#"_G-H"#"_B-F"#"_G-H"#"_B-F"#"_G-H"#"_B-F"#"_G-H"#"_G-H"#"_G-H"#"_B-F"#

if KILL_QCD:
    jet_tag += "_MinDPhi_0p5"
    #jet_tag += "_MinDPhiBarrel_0p5"

#var_vs_eta(samples_to_run,var="Jets[0].nTrackConstituents",add_label="",check_closure=False)
#exit()


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

#print "Warning!!!!!!!"
#print "Warning!!!!!!!"
#print "Artificially setting lumi to 137*1000 /pb"
#LUMI = 137*1000
#print "Warning!!!!!!!"
#print "Warning!!!!!!!"


print "Ntupledir: ", NTUPLEDIR
print "Luminosity: ", data[0], LUMI

sample_dict = {}
reg_comb = ""
if "Wto" in REGION:
    sample_dict["WtoMN"] = "SingleMuon"
    #sample_dict["WtoMN_MET"] = "SingleMuon"
    if ERA=="2018":
        sample_dict["WtoEN"] = "EGamma"
        #sample_dict["WtoEN_MET"] = "EGamma"
    else:
        if "_B-F" not in jet_tag:
            sample_dict["WtoEN"] = "SingleElectron"
            #sample_dict["WtoEN_MET"] = "SingleElectron"
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
else:
    sample_dict[REGION] = samples_to_run[0]
    print "This won't work for lists, hence MC... keep in mind"
    reg_comb = REGION

#LUMI = 137.4*1000
#print "Using full run 2 lumi!! ", LUMI

#calculate_tag_eff(get_tree_weights(samples_to_run,LUMI),samples_to_run,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,j_idx=-1,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
##draw_tag_eff(samples_to_run,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#draw_tag_eff_updated(sample_dict,reg_label=reg_comb,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#exit()

#ROC:
#plot_roc_curve(get_tree_weights(sign,LUMI),get_tree_weights(back,LUMI),sign,back,add_label="",check_closure=False,eta=False,j_idx=-1,eta_cut=0,eta_invert=False)
#plot_roc_curve(get_tree_weights(sign),get_tree_weights(back),sign,back,add_label="",check_closure=False,eta=False,j_idx=-1,eta_cut=1.0,eta_invert=False)
#plot_roc_curve(get_tree_weights(sign),get_tree_weights(back),sign,back,add_label="",check_closure=False,eta=False,j_idx=-1,eta_cut=1.0,eta_invert=True)
#compare_roc_curves(list_comparison=["","_eta_1p0","_eta_larger_1p0"],add_label="")
#exit()


##calculate_tag_eff(samples_to_run,add_label=jet_tag+"_closure_0p99",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi
##draw_tag_eff(samples_to_run,add_label=jet_tag+"_closure_0p99",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi

#jet_correlations(samples_to_run,add_label=jet_tag+"_zoom",check_closure=False)#_low_dPhi_0p5_2_HLT_PFJet_combi
#exit()


#jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"

'''
draw_MC_combination(
    ERA,
    back,
    "SR",
    region_label = jet_tag,
    add_label="",
    lab_2=jet_tag,
    check_closure=clos,
    eta_cut = CUT_ETA,
    eta=DO_ETA,
    phi_cut = CUT_PHI,
    phi=DO_PHI
)

exit()
'''

'''
draw_data_combination(
    ERA,
    #["WtoMN","WtoEN","ZtoMM","ZtoEE","JetHT","DiJetMET","TtoEM"],
    #["JetHT","JetHT"],
    #["ZtoMM","TtoEM",],#["ZtoMM","WtoMN","TtoEM","JetHT"],#,"JetHT",SR
    ##regions_labels=["","","",jet_tag,"","","","","","","","",""],
    #datasets=["","","","","QCD","","","","","","","",],
    #datasets=["","QCD","","","","","","","",],
    #
    # universal
    #
    #["ZtoMM","ZtoEE","WtoMN","WtoEN","TtoEM","JetHT",],#"DiJetMET"],#SR
    ["ZtoLL","WtoLN","TtoEM","JetHT",],#"DiJetMET"],#SR
    #["ZtoLL","JetHT",],#"WtoLN","TtoEM","JetHT",],#"DiJetMET"],#SR
    #2016 B-F
    #["ZtoMM","WtoMN","TtoEM","JetHT",],#"DiJetMET"],#SR
    regions_labels = [jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    #["WtoMN","WtoMN","WtoEN","WtoEN","WtoMN","WtoEN"],
    #regions_labels=["","_MinDPhiBarrel_0p5","","_MinDPhiBarrel_0p5","_MinDPhi_0p5","_MinDPhi_0p5"],
    #compare different MET cuts
    #["WtoEN","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT"],
    #regions_labels = ["","_noMT","_MET100","_MET150","_MET200"],
    #regions_labels=["_jet_0","_jet_1","_jet_0","_jet_1","","","","","","","","",""],
    #MC in SR
    #["TtoEM","TtoEM"],
    #datasets = ["","TTbar"],
    #["SR","SR","SR","SR","SR","SR"],
    #datasets=["ZJetsToNuNu","WJetsToLNu","QCD","TTbarGenMET","VV","All"],
    #2016
    #["ZtoMM","WtoMN","TtoEM","JetHT"],
    #regions_labels=["_B-F","_B-F","_B-F","_B-F",],
    #dphi
    #["WtoMN","WtoMN","WtoMN",],
    #regions_labels=["","_MinDPhiBarrel_0p5","_MinDPhi_0p5"],
    #regions_labels=["_B-F","_B-F_MinDPhiBarrel_0p5","_B-F_MinDPhi_0p5"],
    #["JetHT","JetHT"],
    #regions_labels = ["","_InvertBeamHalo"],
    add_label="",#"_vs_QCD_MC",#"_closure"
    lab_2=jet_tag,#"",#"_JetHT_BeamHalo",#"_MinDPhi_and_Barrel_0p5",#"_B-F",#"_MinDPhi_and_Barrel_0p5",#"_MC",#"_vs_MC",#"_0p95",#""
    check_closure=clos,#True#False#True
    eta_cut = CUT_ETA,
    eta=DO_ETA,
    phi_cut = CUT_PHI,
    phi=DO_PHI
)

exit()
'''


#vs eta
'''
draw_data_combination(
    ERA,
    ["ZtoMM","ZtoEE","WtoMN","WtoEN"],
    #["ZtoMM","WtoMN",],
    #regions_labels=["_B-F","_B-F"],
    #regions_labels=["_B-F","_B-F_MinDPhiBarrel_0p5","_B-F_MinDPhi_0p5"],
    add_label="",
    lab_2="",
    check_closure=False,
    eta = True
)

exit()
'''

#SR

#closure
#clos = True#False#True
#jet_tag = "_closure_0p95"

'''
draw_data_combination(
    ERA,
    ["ZtoMM","ZtoEE","WtoMN","WtoEN","TtoEM","JetHT","DiJetMET"],
    add_label="_closure_0p95",#"_vs_QCD_MC",#"_closure"
    lab_2="",#"_MC",#"_vs_MC",#"_0p95",#""
    check_closure=clos
)
exit()
'''

##'''
### Older version
##background_prediction(
##    get_tree_weights(samples_to_run,LUMI),
##    samples_to_run,
##    #extr_regions= ["ZtoMM","ZtoEE","WtoMN","WtoEN","TtoEM","JetHT","DiJetMET"],
##    add_label="_closure_0p95",#"",#
##    label_2="_extrapolation_0p95",check_closure=clos
##)
##exit()
##'''


##'''
### Older version
##jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
##
##background_prediction_new(get_tree_weights(samples_to_run,LUMI),
##                          samples_to_run,
##                          extr_regions=["JetHT","DiJetMET","JetMET","WtoMN","MN","MR","SR"],
##                          regions_labels = ["","",jet_tag,"","","","","","","","","","","","","","","","",""],
##                          add_label="",
##                          label_2="_extrapolation_triplet_quadruplet",check_closure=False)
##
##exit()
##'''


#jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
#clos = False


background_prediction(
    get_tree_weights(samples_to_run,LUMI),
    samples_to_run,
    #extr_regions= ["JetHT","DiJetMET","WtoEN","WtoMN","SR"],#["QCD","WtoMN","WtoEN"],#["WtoEN"],#["JetHT","DiJetMET","JetMET","ZtoMM","WtoMN","MN","MR","SR"],
    #regions_labels = ["","","","","","","","","","","","",""],#["","",""],#[""],#["","",jet_tag,"","","","","","","","","","","","","","","","",""],
    #datasets= ["","","","","","","","","","","","","","",""],#["QCD","",""],#[""],#["","","","","","","","","","","","","",],
    # 2017:
    extr_regions = ["WtoLN"],#["WtoEN","WtoMN"],
    #extr_regions = ["ZtoLL","JetHT"],#"SR",
    #extr_regions = ["WtoLN_MET","WtoLN","ZtoLL","TtoEM","JetHT"],#["ZtoMM"],
    #
    # universal
    #
    #extr_regions = ["ZtoMM","ZtoEE","WtoEN","WtoMN","TtoEM","JetHT"],#"ZtoMM","ZtoEE","SR"],
    #2016 B-F
    #extr_regions = ["ZtoMM","WtoMN","TtoEM","JetHT"],###"ZtoMM","ZtoEE","SR"],
    regions_labels = [jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    #extr_regions= ["ZtoMM","WtoMN","TtoEM","JetHT","DiJetMET","SR"],
    #datasets= ["VV","","","","","","","","","","","","","","","","","","","","","","","","",],
    # 2018: no DiJetMET available
    #extr_regions= ["ZtoMM","WtoMN","TtoEM","JetHT","SR"],
    # 2016: no electron
    #extr_regions = ["ZtoMM",],#["WtoMN"],#"WtoMN","TtoEM","JetHT"],
    #regions_labels = ["_B-F","_B-F","_B-F","_B-F",],
    #regions_labels = ["_B-F_MinDPhi_0p5","_B-F_MinDPhi_0p5","_B-F","_B-F",],
    #JJ compare
    #["WtoMN","WtoEN"],
    #regions_labels = ["_A-B_MinDPhi_0p5","_A-B_MinDPhi_0p5"],
    #regions_labels=["_MinDPhiBarrel_0p5","_MinDPhiBarrel_0p5",],
    #regions_labels=["_MinDPhi_0p5","_MinDPhi_0p5",],
    #"_MinDPhi_0p5","_MinDPhi_0p5"
    ##regions_labels = ["","","","","","","","","","","","",""],
    #datasets = ["","","","","","","","","","","","","","","",""],
    #MC extrapolation
    #extr_regions = ["SR"],
    #datasets= ["ZJetsToNuNu"],
    #
    #MET enriched WtoEN
    #
    #extr_regions= ["WtoEN","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT","ZtoEE","JetHT","DiJetMET"],
    #regions_labels = ["","_noMT","_MET100","_MET150","_MET200","","","","","","","",""],
    add_label="",#"_closure_0p95",#"",#
    label_2=jet_tag,#+"_D",#+"_check",#"_MinDPhi_0p5",#"",#"_MinDPhiBarrel_0p5",#"_MinDPhi_0p5",#"",#"_A-B_MinDPhi_0p5",#"_B-F",#_MinDPhi_0p5",#"_MinDPhi_0p5",#"",#"_B-F",#"",#"",#"_MinDPhi_0p5",#"_extrapolation_0p996_MET100",
    check_closure=clos,#False
    plot_distr = "",#"Jets[0].pt",#"nCHSJets",#"Jets[2].phi",#"Jets.sigprob"#"nTagJets_0p996_JJ"#
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
)
exit()


mu_scale = 1.#50.#0.01#1.#2.#5.#10.#0.1#
contam = False
uncertainties = {
    "obs" : -1,#0.2,#20.,#-1.,
    "bkg_yield" : 0.148,
    "bkg_yield_stat" : 1.07,
    "bkg_yield_syst" : 1. + math.sqrt( (0.031)**2 + (0.068)**2 )/0.148,
    "lumi" : 1.025,
    "JES" : 1.084,
    "JER" : 1.083,
    "uncl_en" : 1.026,
    "PDF" : 0.,
    "alpha_s" : 1.189,
    "PU" : 1.027,
    "tau" : 1.03,
}

uncertainties_BF = {
    "obs" : -1,#0.2,#20.,#-1.,
    "bkg_yield" : 0.065,#0.148,
    "bkg_yield_stat" : 1.+0.11,#1.07,
    "bkg_yield_syst" : 1.+math.sqrt( (0.015)**2 + (0.034)**2 )/0.065,#1. + math.sqrt( (0.031)**2 + (0.040)**2 )/0.148,
    "lumi" : 1.025,
    "JES" : 1.084,
    "JER" : 1.083,
    "uncl_en" : 1.026,
    "PDF" : 0.,
    "alpha_s" : 1.189,
    "PU" : 1.027,
    "tau" : 1.03,
}

uncertainties_REST = {
    "obs" : -1,#0.2,#20.,#-1.,
    "bkg_yield" : 0.083,#0.148,
    "bkg_yield_stat" : 1.077,#1.07,
    "bkg_yield_syst" : 1. + math.sqrt( (0.016)**2 + (0.035)**2 )/0.083,#1. + math.sqrt( (0.031)**2 + (0.040)**2 )/0.148,
    "lumi" : 1.025,
    "JES" : 1.084,
    "JER" : 1.083,
    "uncl_en" : 1.026,
    "PDF" : 0.,
    "alpha_s" : 1.189,
    "PU" : 1.027,
    "tau" : 1.03,
}

LUMI_BF = 19933
LUMI_REST = LUMI - LUMI_BF


'''
write_datacards(
    get_tree_weights(sign,LUMI_REST),
    sign,
    main_pred_reg = REGION,
    main_pred_sample="HighMET",#"All",#"HighMET",#
    #extrapolation
    extr_region= "WtoLN",#"WtoMN",#["ZtoMM","WtoMN","TtoEM","JetHT","DiJetMET"],
    unc_dict = uncertainties_REST,
    comb_fold_label = jet_tag,#jet_tag,#"",#"_MinDPhi_0p5_eta_1p0_new_bins",
    #comb_fold_label = "_MinDPhi_0p5_eta_1p0_new_bins",#"_MinDPhi_0p5_new_bins",#"",
    regions_labels = [jet_tag],#["_MinDPhi_0p5","","","","","","","","","","","",""],
    datasets= ["","","","","","","","","","","","","","",""],
    add_label="",
    label_2="_REST",#"_MinDPhi_0p5",#"",#"",#"_MinDPhi_0p5_new_bins",#"_extrapolation_0p996",
    check_closure=False,
    pred_unc = 1.,#here to boost it if needed
    run_combine = False,#True,#True,
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
)
exit()
'''

'''
combine_datacards_AN(
    sign,
    comb_fold_label = jet_tag,#"",#"_MinDPhi_0p5_eta_1p0_new_bins",
    regions_labels = [jet_tag],#["_MinDPhi_0p5","","","","","","","","","","","",""],
    datasets= ["","","","","","","","","","","","","","",""],
    add_label="",
    label_2=jet_tag,#"_MinDPhi_0p5",#"",#"",#"_MinDPhi_0p5_new_bins",#"_extrapolation_0p996",
    check_closure=clos,
    pred_unc = 1.,#here to boost it if needed
    run_combine = True,
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
)

exit()
'''

'''
write_datacards(
    get_tree_weights(sign,LUMI),
    sign,
    main_pred_reg = REGION,
    main_pred_sample="HighMET",#"All",#"HighMET",#
    #extrapolation
    extr_region= "WtoLN",#"WtoMN",#["ZtoMM","WtoMN","TtoEM","JetHT","DiJetMET"],
    unc_dict = uncertainties,
    comb_fold_label = jet_tag,#"",#"_MinDPhi_0p5_eta_1p0_new_bins",
    #comb_fold_label = "_MinDPhi_0p5_eta_1p0_new_bins",#"_MinDPhi_0p5_new_bins",#"",
    regions_labels = [jet_tag],#["_MinDPhi_0p5","","","","","","","","","","","",""],
    datasets= ["","","","","","","","","","","","","","",""],
    add_label="",
    label_2=jet_tag,#"_MinDPhi_0p5",#"",#"",#"_MinDPhi_0p5_new_bins",#"_extrapolation_0p996",
    check_closure=False,
    pred_unc = 1.,#here to boost it if needed
    run_combine = True,#True,#True,
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
)
#exit()
'''

plot_limits_vs_ctau(
    sign,
    #main_pred_reg = REGION,
    #extr_region= "WtoMN",
    comb_fold_label = jet_tag,#"_eta_1p0_new_bins",#"_MinDPhi_0p5_eta_1p0_new_bins",#"_MinDPhiBarrel_0p5",#"",#"_MinDPhiBarrel_0p5",#"",
    eta = DO_ETA,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
)
#exit()

plot_limits_vs_mass(
    sign,
    #main_pred_reg = REGION,
    #extr_region= "WtoMN",
    comb_fold_label = jet_tag,#"_eta_1p0_new_bins",#"_MinDPhi_0p5_eta_1p0_new_bins",#"_MinDPhiBarrel_0p5",#"",#"_MinDPhiBarrel_0p5",#"",
    eta = DO_ETA,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
    scale_mu=mu_scale,
    contamination = contam,
)
exit()


'''
combine_datacards_AN(
    sign,
    comb_fold_label = jet_tag,#"",#"_MinDPhi_0p5_eta_1p0_new_bins",
    regions_labels = [jet_tag],#["_MinDPhi_0p5","","","","","","","","","","","",""],
    datasets= ["","","","","","","","","","","","","","",""],
    add_label="",
    label_2=jet_tag,#"_MinDPhi_0p5",#"",#"",#"_MinDPhi_0p5_new_bins",#"_extrapolation_0p996",
    check_closure=False,
    pred_unc = 1.,#here to boost it if needed
    run_combine = True,
    eta_cut = False
)

exit()
'''

plot_limits(
    sign,
    comb_fold_label = jet_tag,
    combination=True
)
#exit()


evaluate_median_expected_difference(
    #signal contamination
    file_names = [
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0_contamination.root",
    ],
    labels = ["no S contam. (#mu = 1.)", "with S contam. (#mu = 1.)","no S contam. (#mu = 0.1)", "with S contam. (#mu = 0.1)","no S contam. (#mu = 10.)", "with S contam. (#mu = 10.)","no S contam. (#mu = 50.)", "with S contam. (#mu = 50.)","no S contam. (#mu = 100.)", "with S contam. (#mu = 100.)",],
    plot_label = "_signal_contamination",
)

#exit()

combine_limits(
    #very preliminary comparisons
    #file_names = ["plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_dPhi_0p5.root",],
    #file_names = ["plots/Limits/v5_calo_AOD_2017_SR/Exclusion_ctau1000_nTagJets_0p996_JJ_new_bins.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5_new_bins.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhiBarrel_0p5_new_bins.root"],
    #labels = ["SR cuts","+ min #Delta #phi (jets-MET)>0.5","+ min #Delta #phi (barrel jets-MET)>0.5"],
    #file_names = ["plots/Limits/v5_calo_AOD_2018_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ.root","plots/Limits/v5_calo_AOD_2018_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5.root"],
    #labels = ["SR cuts","+ min #Delta #phi (jets-MET)>0.5"],
    #2016
    #file_names = ["plots/Limits/v5_calo_AOD_2016_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ_B-F.root","plots/Limits/v5_calo_AOD_2016_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5_B-F.root"],
    #file_names = ["plots/Limits/v5_calo_AOD_2016_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ_G-H.root","plots/Limits/v5_calo_AOD_2016_SR_2017_signal/Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5_G-H.root"],
    #labels = ["SR cuts G-H","+ min #Delta #phi (jets-MET)>0.5 G-H","SR cuts B-F","+ min #Delta #phi (jets-MET)>0.5 B-F",],
    #plot_label = "_G-H_absolute_xsec",#"_LUMI_137_absolute_xsec"
    #
    # compare eta/pt
    #file_names = ["plots/Limits/v5_calo_AOD_2017_SR/Exclusion_ctau1000_nTagJets_0p996_JJ_vs_eta.root","plots/Limits/v5_calo_AOD_2017_SR/Exclusion_ctau1000_nTagJets_0p996_JJ_vs_eta_eta_1p0.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_vs_eta_MinDPhi_0p5.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_vs_eta_MinDPhi_0p5_eta_1p0.root"],
    #labels = ["SR cuts","+ min #Delta #phi (jets-MET)>0.5","SR cuts vs #eta","+ min #Delta #phi (jets-MET)>0.5 vs #eta"],
    #plot_label = "_compare_param_vs_eta_absolute_xsec", 
    #
    # full combination!
    #file_names = ["plots/Limits/v5_calo_AOD_combination/Exclusion_ctau1000_nTagJets_0p996_JJ.root","plots/Limits/v5_calo_AOD_combination/Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5.root"],
    #labels = ["SR cuts","+ min #Delta #phi (jets-MET)>0.5"],
    #file_names = ["plots/Limits/v5_calo_AOD_combination/Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5.root"],
    #labels = ["SR cuts + min #Delta #phi (jets-MET)>0.5"],
    #plot_label = "_full_combi_no_correlations_absolute_xsec",
    #combination=True,
    #
    #
    #eta cuts
    #file_names = ["plots/Limits/v5_calo_AOD_2017_SR/Exclusion_ctau1000_nTagJets_0p996_JJ_eta_1p0_new_bins.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5_eta_1p0_new_bins.root","plots/Limits/v5_calo_AOD_2017_SR/Exclusion_ctau1000_nTagJets_0p996_JJ_new_bins.root","plots/Limits/v5_calo_AOD_2017_SR//Exclusion_ctau1000_nTagJets_0p996_JJ_MinDPhi_0p5_new_bins.root",],
    #labels = ["SR cuts, |#eta|<1.", "+ min #Delta #phi (jets-MET)>0.5, |#eta|<1.","SR cuts","+ min #Delta #phi (jets-MET)>0.5"],
    #plot_label = "_absolute_xsec_compare_eta_1p0_new_bins",
    #
    #eta/phi cuts
    #file_names = [
    #    "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta//Exclusion_ctau1000.root",
    #    "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_eta_1p0_vs_eta//Exclusion_ctau1000.root",
    #    "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_phi_cut_vs_eta//Exclusion_ctau1000.root",
    #    "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_eta_1p0_phi_cut_vs_eta//Exclusion_ctau1000.root",
    #],
    #labels = ["SR cuts", "|#eta|<1.", "#varphi veto", "|#eta|<1. & #varphi veto"],
    #plot_label = "_absolute_xsec_compare_eta_1p0_phi_cut",

    #signal contamination
    file_names = [
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_0p1_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_10p0_contamination.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0.root",
        "plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_50p0_contamination.root",
        #"plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0.root",
        #"plots/Limits/v5_calo_AOD_2017_SR/nTagJets_0p996_JJ_vs_eta/Exclusion_ctau1000_scale_mu_100p0_contamination.root",
    ],
    labels = ["no S contam. (#mu = 1.)", "with S contam. (#mu = 1.)","no S contam. (#mu = 0.1)", "with S contam. (#mu = 0.1)","no S contam. (#mu = 10.)", "with S contam. (#mu = 10.)","no S contam. (#mu = 50.)", "with S contam. (#mu = 50.)","with S contam. (#mu = 10.)","no S contam. (#mu = 100.)", "with S contam. (#mu = 100.)",],
    plot_label = "_signal_contamination",
)

exit()


#2016:
#calculate_tag_eff(samples_to_run,add_label="_RunB-F")#(back)#
#calculate_tag_eff(samples_to_run,add_label="_RunG-H")#(back)#
#draw_tag_eff(samples_to_run,add_label="_RunG-H")#(back)#

'''
background_prediction_new(get_tree_weights(samples_to_run,LUMI),
                          samples_to_run,
                          extr_regions=["WtoMN","SR","MN",],
                          regions_labels = ["","",""],
                          add_label="",
                          label_2="_extrapolation_to_MN_quadruplet")
'''


#draw_data_combination(ERA,["WtoMN","SR","JetMET_low_dPhi_1p5_Lep","JetMET_dPhi_1p5_Lep","JetMET_low_dPhi_1p5_MET_200_Lep","JetMET_dPhi_1p5_MET_200_Lep"],"","_JetMET_Lep_combinations")

#QUAA
'''
background_prediction_new(get_tree_weights(samples_to_run),
                          samples_to_run,
                          extr_regions=["WtoMN","SR","MN","MR","JetMET","JetMET",],
                          regions_labels = ["","","","","_low_dPhi_0p5_2_HLT_PFJet_combi","_low_dPhi_0p5_2_HLT_PFJet_500"],
                          add_label="",
                          label_2="_extrapolation_to_dPhi_0p5_2_HLT_PFJet_500")
'''
#draw_data_combination(ERA,["WtoMN","SR","JetMET_low_dPhi_1p5_Lep","JetMET_dPhi_1p5_Lep","JetMET_low_dPhi_1p5_MET_200_Lep","JetMET_dPhi_1p5_MET_200_Lep"],"","_JetMET_Lep_combinations")


#background_prediction_new(get_tree_weights(samples_to_run),samples_to_run,extr_regions=["WtoMN","MR","SR"],add_label="_RunB-F",label_2="_extrapolation")
#background_prediction_new(get_tree_weights(samples_to_run),samples_to_run,extr_regions=["WtoMN","MR","SR"],add_label="_RunG-H",label_2="_extrapolation")

#?#background_prediction_new_one_eff(get_tree_weights(samples_to_run),samples_to_run,extr_region="",add_label="_test_more_CR")
#
#
#background_prediction_new(get_tree_weights(samples_to_run),samples_to_run,extr_regions=["WtoMN","WtoEN","MR","SR"],add_label="_RunG-H",label_2="")

#draw_data_combination(ERA,["WtoMN","SR","JetMET_all_triggers","JetMET_unprescaled_trigger","JetMET_dPhi_1p5_all_triggers","JetMET_MET_200_all_triggers","JetMET_dPhi_1p5_MET_200_all_triggers"],"","_MET_check")

#HERE
'''
draw_data_combination(
    ERA,
    ["WtoMN","JetMET","JetMET","MR","MN",],
    regions_labels=["","_low_dPhi_0p5_2_HLT_PFJet_500","_low_dPhi_0p5_2_HLT_PFJet_combi","","",""],
    add_label="",
    lab_2="_0_lepton_regions"

    #["JetMET","JetMET","JetMET","JetMET","JetMET","JetMET","JetMET"],
    #["JetMET","JetMET","JetMET","JetMET","JetMET","JetMET","JetMET"],
    #regions_labels=["_dPhi_1p5_MET_200_HLT_PFJet_combi","_dPhi_1p5_MET_200","_dPhi_1p5_MET_200_HLT_PFJet260","_dPhi_1p5_MET_200_HLT_PFJet320","_dPhi_1p5_MET_200_HLT_PFJet400","_dPhi_1p5_MET_200_HLT_PFJet450","_dPhi_1p5_MET_200_HLT_PFJet500",],
    #add_label="",
    #lab_2="_up_macro_test_MET_200"

    #["WtoMN","JetMET","JetMET","JetMET","JetMET","SR"],
    #regions_labels=["","_dPhi_1p5_Lep_HLT_PFJet_combi","_dPhi_1p5_MET_200_Lep_HLT_PFJet_combi","_low_dPhi_1p5_Lep_HLT_PFJet_combi","_low_dPhi_1p5_MET_200_Lep_HLT_PFJet_combi",""],
    #add_label="",
    #lab_2="_Lep_HLT_PFJet_combination"

    #["WtoMN","JetMET","JetMET","SR"],
    #regions_labels=["","_dPhi_1p5_Lep_HLT_PFJet_combi","_dPhi_1p5_HLT_PFJet_combi",""],
    #add_label="",
    #lab_2="_compare_dPhi_1p5_Lep_HLT_PFJet_combination_more_bins"

    #["WtoMN","JetMET","JetMET","SR"],
    #regions_labels=["","_dPhi_1p5_MET_200_Lep_HLT_PFJet_combi","_dPhi_1p5_MET_200_HLT_PFJet_combi",""],
    #add_label="",
    #lab_2="_compare_dPhi_1p5_MET_200_Lep_HLT_PFJet_combination_more_bins"

    #["WtoMN","JetMET","JetMET","JetMET","SR"],
    #regions_labels=["","_low_dPhi_1p5_Lep_HLT_PFJet_combi","_low_dPhi_1p5_HLT_PFJet_combi","_low_dPhi_0p5_2_HLT_PFJet_combi",""],
    #add_label="",
    #lab_2="_compare_low_dPhi_1p5_Lep_HLT_PFJet_combination_more_bins"

    #["WtoMN","JetMET","JetMET","SR"],
    #regions_labels=["","_low_dPhi_1p5_MET_200_Lep_HLT_PFJet_combi","_low_dPhi_1p5_MET_200_HLT_PFJet_combi",""],
    #add_label="",
    #lab_2="_compare_low_dPhi_1p5_MET_200_Lep_HLT_PFJet_combination_more_bins"

    #["WtoMN","JetMET","JetMET","MN","MR","SR"],
    #regions_labels=["","_low_dPhi_1p5_Lep_HLT_PFJet_combi","_low_dPhi_1p5_HLT_PFJet_combi","","",""],
    #add_label="",
    #lab_2="_test_MN"
)
'''

#draw_data_combination(ERA,["JetMET","JetMET","JetMET","JetMET","JetMET","JetMET","JetMET",],regions_labels=["_dPhi_1p5","_dPhi_1p5_HLT_PFJet140","_dPhi_1p5_HLT_PFJet200","_dPhi_1p5_HLT_PFJet260","_dPhi_1p5_HLT_PFJet320","_dPhi_1p5_HLT_PFJet400","_dPhi_1p5_HLT_PFJet450","_dPhi_1p5_HLT_PFJet500",],add_label="",lab_2="_up_macro_test")
#draw_data_combination(ERA,["WtoMN","SR","JetMET_low_dPhi_1p5","JetMET_low_dPhi_1p5_Lep","JetMET_dPhi_1p5_all_triggers","JetMET_dPhi_1p5_Lep"],regions_labels=[],add_label="",lab_2="_MET_dPhi_cut")

#draw_data_combination(ERA,["WtoMN","SR","MR","JetMET_low_dPhi_1p5_MET_200","JetMET_low_dPhi_1p5_MET_200_Lep","JetMET_dPhi_1p5_MET_200_all_triggers","JetMET_dPhi_1p5_MET_200_Lep"],"","_MET_dPhi_cut_and_MET_200")
#draw_data_combination(ERA,["WtoMN","SR","JetMET_low_dPhi_1p5_Lep","JetMET_dPhi_1p5_Lep","JetMET_low_dPhi_1p5_MET_200_Lep","JetMET_dPhi_1p5_MET_200_Lep"],"","_JetMET_Lep_combinations")


#draw_data_combination(ERA,["WtoMN","SR","JetMET_all_triggers","JetMET_dPhi_1p5_all_triggers","JetMET_dPhi_1p5_MET_200_all_triggers","JetMET_dPhi_1p5_MET_200_Lep","JetMET_low_dPhi_1p5","JetMET_low_dPhi_1p5_Lep","JetMET_low_dPhi_1p5","JetMET_low_dPhi_1p5_Lep"],"","_MET_check_lower_met")
#draw_data_combination(ERA,["WtoMN","SR","JetMET_dPhi_1p5_MET_200_all_triggers","JetMET_dPhi_1p5_MET_200_Lep"],"","_MET_check")

#draw_data_combination(ERA,["WtoMN","WtoEN","MR","JetHT","JetHT_unprescaled"],"","_all")
#draw_data_combination(ERA,["WtoMN","WtoEN","MR","JetHT"],"","_JetHT_check")
#draw_data_combination(ERA,["ZtoMM","WtoMN","ZtoEE","WtoEN","SR"],"_RunG-H")
#draw_data_combination(ERA,["ZtoMM","WtoMN","ZtoEE","WtoEN","SR"],"_RunB-F")
#draw_data_combination(ERA,["WtoMN","ZtoMM","WtoEN","ZtoEE","SR_HEM"],"_no_HEM_cut")
#draw_data_combination(ERA,["WtoMN","WtoEN","MR","SR","MRPho"],"","_more_CRs")
#draw_data_combination(ERA,["ZtoMM","TtoEM","ZtoEE","WtoEN","SR"],"")
#draw_data_combination(ERA,["SR","SR_HEM"],"","_HEM_effect")
#draw_data_combination(ERA,["ZtoMM","WtoMN","ZtoEE","WtoEN","SR_HEM"],"_no_HEM_cut")
#draw_data_combination(ERA,["ZtoMM","ZtoEE","TtoEM","WtoMN"],"_no_HEM_cut")
#draw_data_combination(ERA,["SR","WtoMN","ZtoMM","WtoEN","ZtoEE"])#,"TtoEM"])

##Compare with Jiajing
#draw_comparison("2018",["WtoMN","ZtoMM","WtoEN"],"_all",maxeff=0.001)
#draw_comparison("2018",["WtoMN"],"_SingleMuon",col=1)
#draw_comparison("2018",["ZtoMM"],"_ZtoMM",col=2)
#draw_comparison("2018",["WtoEN"],"_SingleElectron",col=4)
#draw_comparison("2018",["ZtoEE"],"_ZtoEE",col=418)
