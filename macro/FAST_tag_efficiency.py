#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
import json
import time
#import pandas as pd
from array import array
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict

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
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
parser.add_option("-f", "--final", action="store_true", default=False, dest="final")
parser.add_option("-R", "--rebin", action="store_true", default=False, dest="rebin")
parser.add_option("-p", "--public", action="store_true", default=False, dest="public")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)

ERA                = "2017"
#REGION             = "EN"
#CUT                = "isEN"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"

REGION             = "JetMET"#"WtoEN"#"SR_HEM"#"debug_new_with_overlap"#"SR"#"ZtoMM_CR"

#CUT                = "isJetMET_dPhi_Lep"
#CUT                = "isJetMET_low_dPhi_Lep"
#CUT                = "isJetMET_dPhi_MET_200_Lep"
#CUT                = "isJetMET_low_dPhi_MET_200_Lep"

#CUT                = "isJetMET_dPhi"
CUT                = "isJetMET_low_dPhi_500"#THIS!
#CUT                = "isJetMET_low_dPhi"#---> shows signal!
#CUT                = "isJetMET_dPhi_MET_200"
#CUT                = "isJetMET_low_dPhi_MET_200"

#CUT                = "isJetMET_dPhi_MET"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"
#NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_ZtoMM_CR/"
#PLOTDIR            = "plots/Efficiency/v5_calo_AOD_2018_ZtoMM_CR/"
NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"
PLOTDIR            = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+REGION+"/"
REBIN              = options.rebin
SAVE               = True
#LUMI               = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#8507.558453#59.74*1000#Run2018

#back = ["DYJetsToLL"]
#back = ["TTbarGenMET"]
#back = ["WJetsToLNu"]
#back = ["QCD"]
#back = ["ZJetsToNuNu"]
#back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET"]
#back = ["QCD","WJetsToLNu","TTbarGenMET"]
#data = ["SingleMuon"]
#data = ["SingleElectron"]
#data = ["EGamma"]
#data = ["MuonEG"]
#data = ["MET"]
#data = ["HighMET"]
data = ["JetHT"]

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
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]

print "Ntupledir: ", NTUPLEDIR
print "Luminosity: ", data[0], LUMI

COMPARE = options.compare
DRAWSIGNAL = options.drawsignal

########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]
lines = [1,1,1,1,1,2,2,2,2]
markers = [20,20,20,20,20,24,24,24,24]
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
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#bins=array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#bins=np.array([1,10,20,30,40,50,60,70,80,90,100,1000])
#bins = bins.astype(np.float32)

bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,1000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
#more_bins=np.array([1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
#more_bins = more_bins.astype(np.float32)
###more_bins = bins
maxeff = 0.002

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


def get_tree_weights(sample_list):
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
                filename.Close()
                if('GluGluH2_H2ToSSTobbbb') in ss:
                    xs = 1.
                elif('XXTo4J') in ss:
                    xs = 1.
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.
                else:
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
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
    
def background_prediction_new(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],add_label="",label_2="",check_closure=False):

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
    chain = {}
    hist = {}
    df = {}
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

        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]

        infiles[r+reg_label] = TFile(EFFDIR[r+reg_label]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root", "READ")
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
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")

        print("Entries in chain: %d")%(chain[s].GetEntries())

        max_n=200#200#chain[s].GetEntries()+10#100000
        #ev_weight = "(1/"+str(tree_weights[...])+")"
        #ev_weight = "1"
        ev_weight  = "EventWeight*PUReWeight"
        tagvar = "nTagJets_0p996_JJ"
        #tagvar = "nTagJets_cutbased"
        print tagvar
        cutstring_0 = ev_weight+"*("+tagvar+"==0 && nCHSJetsAcceptanceCalo>0)"
        cutstring_1 = ev_weight+"*("+tagvar+"==1)"
        cutstring_2 = ev_weight+"*("+tagvar+">1)"
        if "Wto" in REGION:
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && nCHSJetsAcceptanceCalo>0 && MT<100)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MT<100)"


        #combination single jet
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


        print "CUT: ", CUT
        print "cutstring bin 0: ", cutstring_0
        chain[s].Project(s+"_0", "isMC",cutstring_0,"",max_n)
        chain[s].Project(s+"_1", "isMC",cutstring_1,"",max_n)
        chain[s].Project(s+"_2", "isMC",cutstring_2,"",max_n)
        print "h0 entries: ", h0[s].GetEntries()
        print "h1 entries: ", h1[s].GetEntries()
        print "h2 entries: ", h2[s].GetEntries()
        #exit()

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
            if (CUT == "isJetHT" and not(event.isJetHT)): continue
            if (CUT == "isJetMET" and not(event.isJetMET)): continue



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


            print "----"
            print "Event n. ",n

            n_ev_passing+=1

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
        exit()
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
        
    ##To open a json:
    #a_file = open(PLOTDIR+'BkgPredResults.json', "r")
    #output = a_file.read()
    #print(output)

def GetEffWeightBin1New(event, TEff, check_closure):
    EffW={}
    EffWUp={}
    EffWLow={}
    #cnt={}
    p1={}
    p1Up={}
    p1Low={}

    if check_closure:
        dnn_threshold = 0.9
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

    if check_closure:
        dnn_threshold = 0.9
    else:
        dnn_threshold = 0.996

    #n_untagged = 0
    #n_tagged = 0
    start = time.time()

    '''
    #Slow
    for r in TEff.keys():
        for j1 in range(event.nCHSJetsAcceptanceCalo):
            if (event.Jets[j1].sigprob<=dnn_threshold):#bin0 selection n_untagged+=1
                binN1 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j1].pt)
                eff1[r]  = TEff[r].GetEfficiency(binN1)
                errUp1 = TEff[r].GetEfficiencyErrorUp(binN1)
                errLow1 = TEff[r].GetEfficiencyErrorLow(binN1)
                effUp1[r] = eff1[r] + errUp1
                effLow1[r] = eff1[r] - errLow1

            #Second loop: find all the jet pairs
            for j2 in range(event.nCHSJetsAcceptanceCalo):
                if (event.Jets[j2].sigprob<=dnn_threshold):#bin0 selection
                    #for r in TEff.keys():
                    binN2 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j2].pt)
                    eff2[r]  = TEff[r].GetEfficiency(binN2)
                    errUp2 = TEff[r].GetEfficiencyErrorUp(binN2)
                    errLow2 = TEff[r].GetEfficiencyErrorLow(binN2)
                    effUp2[r] = eff2[r] + errUp2
                    effLow2[r] = eff2[r] - errLow2
                    if(j2>j1):
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
                            if (event.Jets[j3].sigprob<=dnn_threshold):#bin0 selection
                                binN3 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j3].pt)
                                eff3[r]  = TEff[r].GetEfficiency(binN3)
                                errUp3 = TEff[r].GetEfficiencyErrorUp(binN3)
                                errLow3 = TEff[r].GetEfficiencyErrorLow(binN3)
                                effUp3[r] = eff3[r] + errUp3
                                effLow3[r] = eff3[r] - errLow3
                                if(j3>j2):
                                    #print("=======")
                                    #print("Event n. %d")%(n)
                                    #print("Region: %s")%(r)
                                    #print("Triplet jets %d %d %d")%(j1,j2,j3)
                                    p2[r] = p2[r] + eff1[r]*eff2[r]*eff3[r]
                                    p2Up[r] = p2Up[r] + effUp1[r]*effUp2[r]*effUp3[r]
                                    p2Low[r] = p2Low[r] + effLow1[r]*effLow2[r]*effLow3[r]

                                    #Fourth loop: find all the jet triplets
                                    for j4 in range(event.nCHSJetsAcceptanceCalo):
                                        if (event.Jets[j4].sigprob<=dnn_threshold):#bin0 selection
                                            binN4 = TEff[r].GetPassedHistogram().FindBin(event.Jets[j4].pt)
                                            eff4[r]  = TEff[r].GetEfficiency(binN4)
                                            errUp4 = TEff[r].GetEfficiencyErrorUp(binN4)
                                            errLow4 = TEff[r].GetEfficiencyErrorLow(binN4)
                                            effUp4[r] = eff4[r] + errUp4
                                            effLow4[r] = eff4[r] - errLow4
                                            if(j4>j3):
                                                #print("=======")
                                                #print("Event n. %d")%(n)
                                                #print("Region: %s")%(r)
                                                #print("Quadruplet jets %d %d %d %d")%(j1,j2,j3,j4)
                                                p2[r] = p2[r] + eff1[r]*eff2[r]*eff3[r]*eff4[r]
                                                p2Up[r] = p2Up[r] + effUp1[r]*effUp2[r]*effUp3[r]*effUp4[r]
                                                p2Low[r] = p2Low[r] + effLow1[r]*effLow2[r]*effLow3[r]*effLow4[r]

    '''

    #Fast
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

    #end Fast
    
    for r in TEff.keys():
        EffW[r] = p2[r]
        EffWUp[r] = p2Up[r]
        EffWLow[r] = p2Low[r]
    #return n_untagged, n_tagged, EffW, EffWUp, EffWLow

    end = time.time()
    print "nJets: ", event.nCHSJetsAcceptanceCalo
    print "Eff: ", EffW
    print "Time elapsed: ", end-start
    print "************************************"
    return EffW, EffWUp, EffWLow


samples_to_run = data#back+data#data#data+back#+data
jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
jet_tag = ""
#calculate_tag_eff(samples_to_run,add_label=jet_tag+"_closure",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi
#draw_tag_eff(samples_to_run,add_label=jet_tag+"_closure",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi

jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
background_prediction_new(get_tree_weights(samples_to_run),
                          samples_to_run,
                          extr_regions=["JetMET","WtoMN","ZtoMM","ZtoEE","MN","EN","MR","SR"],
                          regions_labels = [jet_tag,"","","","","","","","","","","","","","","","",""],
                          add_label="_closure",
                          label_2="_check_speed",check_closure=True)

