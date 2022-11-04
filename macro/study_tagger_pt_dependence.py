#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
import yaml
from collections import defaultdict, OrderedDict
import tensorflow as tf
from tensorflow import keras

#import awkward1 as ak
#import matplotlib.pyplot as plt

ERA = "2018"

CHAN = "SUSY"
LUMI = -1

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi[ lumi.keys()[0]]["tot"]    

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

SEL = "SR"

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
    'SUSY_mh1800_ctau500'
]


sign += [
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
    'SUSY_mh1800_ctau3000'
]

sign = ['SUSY_mh400_ctau3000_HH','SUSY_mh1000_ctau3000_HH',]
sign = ['SUSY_mh400_ctau500_HH','SUSY_mh1000_ctau500_HH',]

MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+SEL+"/"

OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_study_tagger_pt_dependence/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

def plot_time_in_pt_bins(var,cut,label="",scale=True, do_smear=False):

    weight = "EventWeight*PUReWeight*TriggerWeight"

    #pt bins:
    #30-50
    #50-100
    #100-200
    #>200


    for s in sign:
        print "Doing ", s, " .... "
        h30 = TH1F(s+"30",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h30.Sumw2()
        h50 = TH1F(s+"50",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h50.Sumw2()
        h100 = TH1F(s+"100",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h100.Sumw2()
        h200 = TH1F(s+"200",";"+variable[var]['title'],variable[var]['nbins'], variable[var]['min'],variable[var]['max'])
        h200.Sumw2()
        chain = TChain("tree")

        for j, ss in enumerate(samples[s]['files']):
            chain.Add(MAIN + ss + ".root")

        chain.Project(s+"30", var, weight+"*("+cut+" && Jets.pt<50)")
        chain.Project(s+"50", var, weight+"*("+cut+" && Jets.pt<100 && Jets.pt>=50)")
        chain.Project(s+"100", var, weight+"*("+cut+" && Jets.pt<200 && Jets.pt>=100)")
        chain.Project(s+"200", var, weight+"*("+cut+" && Jets.pt>=200)")

        h30.SetMarkerStyle(21)
        h30.SetMarkerColor(1)
        h30.SetLineColor(1)
        h30.SetLineWidth(2)

        h50.SetMarkerStyle(21)
        h50.SetMarkerColor(2)
        h50.SetLineColor(2)
        h50.SetLineWidth(2)

        h100.SetMarkerStyle(21)
        h100.SetMarkerColor(8)
        h100.SetLineColor(8)
        h100.SetLineWidth(2)

        h200.SetMarkerStyle(21)
        h200.SetMarkerColor(4)
        h200.SetLineColor(4)
        h200.SetLineWidth(2)

        if scale:
            h30.Scale(1./h30.Integral())
            h50.Scale(1./h50.Integral())
            h100.Scale(1./h100.Integral())
            h200.Scale(1./h200.Integral())

        can = TCanvas("can","can",900,800)
        can.cd()
        #can.SetGrid()
        can.SetRightMargin(0.05)
        leg = TLegend(0.65, 0.7, 1., 1.)
        leg.SetHeader(samples[s]['label'])
        leg.AddEntry(h30,"30<p_{T}<50 GeV","PL")
        leg.AddEntry(h50,"50<p_{T}<100 GeV","PL")
        leg.AddEntry(h100,"100<p_{T}<200 GeV","PL")
        leg.AddEntry(h200,"200<p_{T} GeV","PL")
        h30.GetXaxis().SetTitle(variable[var]['title'])
        h30.Draw("PE,sames")
        h50.Draw("PE,sames")
        h100.Draw("PE,sames")
        h200.Draw("PE,sames")
        leg.Draw()
        OUTSTRING = OUT
        OUTSTRING += s+"_"+var.replace(".","_") 

        can.SetLogy()
        can.Print(OUTSTRING+label+'.pdf')
        can.Print(OUTSTRING+label+'.png')
        can.Close()


variables = ['Jets.nTrackConstituents', 'Jets.nSelectedTracks', 'Jets.timeRecHitsEB', 'Jets.eFracRecHitsEB', 'Jets.nRecHitsEB', 'Jets.sig1EB', 'Jets.sig2EB', 'Jets.ptDEB', 'Jets.cHadEFrac', 'Jets.nHadEFrac', 'Jets.eleEFrac', 'Jets.photonEFrac', 'Jets.ptAllTracks', 'Jets.ptAllPVTracks', 'Jets.alphaMax', 'Jets.betaMax', 'Jets.gammaMax', 'Jets.gammaMaxEM', 'Jets.gammaMaxHadronic', 'Jets.gammaMaxET', 'Jets.minDeltaRAllTracks', 'Jets.minDeltaRPVTracks']
for v in variables:
    print "Doing: ", v
    plot_time_in_pt_bins(v,cut="isSR && MinJetMetDPhi>0.5 && abs(Jets.eta)<1 && Jets.sigprob>0.95",label="",scale=True, do_smear=False)
