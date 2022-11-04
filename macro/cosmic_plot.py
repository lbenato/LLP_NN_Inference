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
#from sklearn.cluster import DBSCAN
#from sklearn.linear_model import LinearRegression
#from scipy.stats import chisquare
import yaml
from collections import defaultdict, OrderedDict
#import tensorflow as tf
#from tensorflow import keras

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = ["HighMET"]
sign = ['SUSY_mh1800_ctau500_HH','SUSY_mh127_ctau500_HH','SUSY_mh300_ctau500_HH','SUSY_mh600_ctau500_HH','SUSY_mh1000_ctau500_HH',]
SEL = "SR"

def cosmic_plot(var,cut_d,cut_s):

    if SEL=="SR":
        prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    else:
        prj_weight = "EventWeight*PUReWeight"

    d_chain = TChain("tree")
    d_hist = TH1F("d_hist","",variable[var]['nbins'],variable[var]['min'],variable[var]['max'])
    d_hist.Sumw2()
    s_chain = {}
    s_hist = {}
    for s in sign:
        s_chain[s] = TChain("tree")
        s_hist[s] = TH1F("s_hist_"+s,"",variable[var]['nbins'],variable[var]['min'],variable[var]['max'])
        s_hist[s].Sumw2()


    #with cosmic one leg requirement
    MAIN_d = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_"+SEL+"_bin_1_2/"
    MAIN_s = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_"+SEL+"/"
    #MAIN_d = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_no_cosmicOneLeg/"
    #MAIN_s = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_no_cosmicOneLeg/"
    eras = ["2016","2017","2018"]

    for s in data:
        LUMI = 0
        for era in eras:
            if era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
                LUMI += lumi["HighMET"]["tot"]
            if era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
                LUMI += lumi["HighMET"]["tot"]
            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
                LUMI += lumi["HighMET"]["tot"]

            for j, ss in enumerate(samples[s]['files']):
                print "TChain "+(MAIN_d%era)+ss + ".root"
                d_chain.Add((MAIN_d%era) + ss + ".root")
    
    #count how many events passed the selection
    print "Data events being vetoed"
    print d_chain.GetEntries("MinJetMetDPhi>0.5 && dt_ecal_dist<0.5")
                
    d_chain.Project("d_hist", var, prj_weight+"*("+cut_d+")")
    d_hist.SetMarkerStyle(21)
    d_hist.SetLineColor(1)
    d_hist.SetLineWidth(2)
    d_hist.SetMarkerColor(1)

    for s in sign:
        for era in eras:
            if era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
            if era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples

            for j, ss in enumerate(samples[s]['files']):
                print "TChain "+(MAIN_s%era)+ss + ".root"
                s_chain[s].Add((MAIN_s%era) + ss + ".root")

    print "Signal events being vetoed"
    for s in sign:
        print s
        print s_chain[s].GetEntries("MinJetMetDPhi>0.5 && dt_ecal_dist<0.5")
        s_chain[s].Project("s_hist_"+s, var, prj_weight+"*("+cut_d+")")
        s_hist[s].SetLineColor(samples[s]['linecolor'])
        s_hist[s].SetLineWidth(2)


    can = TCanvas("can","can",900,800)
    can.SetRightMargin(0.04)
    can.SetLeftMargin(0.12)
    can.cd()
    can.SetLogy()
    can.SetGrid()
    leg = TLegend(0.6, 0.65, 0.96, 0.9)
    d_hist.GetYaxis().SetTitle("Events")
    d_hist.GetXaxis().SetTitle(variable[var]['title'])
    #d_hist.Scale(1./d_hist.Integral())
    d_hist.Draw("PE,sames")
    leg.AddEntry(d_hist,"HighMET data, bin 1","P")
    for s in sign:
        s_hist[s].Scale(d_hist.Integral()/s_hist[s].Integral())
        s_hist[s].Draw("HISTO,sames")
        leg.AddEntry(s_hist[s],samples[s]['label'],"L")
    leg.Draw()
    OUTSTRING = "plots/"+var
    OUTSTRING += "_all_years"
    #OUTSTRING += "_all_years_noCosmicOneLeg"
    drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    drawRegion(SEL)
    leg.Draw()
    can.Update()
    can.Print(OUTSTRING+'.png')
    can.Print(OUTSTRING+'.pdf')


cosmic_plot(var = "dt_ecal_dist",cut_d="isDT_fit && nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5",cut_s="isDT_fit && MinJetMetDPhi>0.5")
