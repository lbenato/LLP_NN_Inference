#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1, TRatioPlot, TFitResult
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
import yaml
from collections import defaultdict, OrderedDict

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)


ERA = "2016"
ERA_TAG = ""
if ERA=="2016":
    ERA_TAG = "GH"
LUMI = -1
SEL = "E"

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if SEL=="E":
        LUMI = lumi["SingleElectron"]["H"]
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    if SEL=="E":
        LUMI = lumi["SingleElectron"]["tot"]
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    if SEL=="E":
        LUMI = lumi["EGamma"]["tot"]

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

CHAN = "SUSY"

if SEL=="E":
    if ERA=="2018":
        data = ["EGamma"]
    else:
        data = ["SingleElectron"+ERA_TAG]
    back = ["Wtop"]

if SEL=="TtoEM":
    data = ["MuonEG"]
    back = ["TTbarGenMET"]

OUT_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_E_v5_ntuples_reweighted/weighted/smearing/"
OUT_pho = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_ZtoLLPho_v5_ntuples_updated/smearing/"

filename = "data_MC_SF_"
if ERA=="2016":
    if ERA_TAG == "GH":
        filename+="G-H_"
    elif ERA_TAG=="BF":
        filename+="B-F_"
filename+="1bin.root"

file_ele = TFile(OUT_ele + filename,"READ")
r1ns_ele = file_ele.Get("ratio_1ns")
r2ns_ele = file_ele.Get("ratio_2ns")
r1ns_ele.SetDirectory(0)
r2ns_ele.SetDirectory(0)

file_pho = TFile(OUT_pho + filename, "READ")
r1ns_pho = file_pho.Get("ratio_1ns")
r2ns_pho = file_pho.Get("ratio_2ns")
r1ns_pho.SetDirectory(0)
r2ns_pho.SetDirectory(0)

can = TCanvas("can","can",900,800)
can.SetRightMargin(0.05)
can.cd()
leg = TLegend(0.65, 0.7, 0.95, 0.9)
leg.AddEntry(r1ns_ele,"+1 ns time shift, e + E_{T}^{miss}","PL")
leg.AddEntry(r1ns_pho,"+1 ns time shift, Z #rightarrow ll #gamma","PF")
leg.AddEntry(r2ns_ele,"+2 ns time shift, e + E_{T}^{miss}","PL")
leg.AddEntry(r2ns_pho,"+2 ns time shift, Z #rightarrow ll #gamma","PF")
r1ns_pho.SetMarkerStyle(21)
r2ns_pho.SetMarkerStyle(21)
r1ns_pho.SetMarkerColor(2)
r2ns_pho.SetMarkerColor(4)
r1ns_pho.SetLineColor(2)
r2ns_pho.SetLineColor(4)
r1ns_pho.SetFillColorAlpha(2,0.2)
r2ns_pho.SetFillColorAlpha(4,0.2)

r1ns_ele.SetMarkerStyle(24)
r2ns_ele.SetMarkerStyle(24)
r1ns_ele.SetMarkerColor(2)
r2ns_ele.SetMarkerColor(4)
r1ns_ele.SetLineColor(2)
r2ns_ele.SetLineColor(4)
r1ns_ele.SetLineWidth(2)
r2ns_ele.SetLineWidth(2)


r1ns_pho.Draw("PE2,sames")
r2ns_pho.Draw("PE2,sames")
r1ns_ele.Draw("PE,sames")
r2ns_ele.Draw("PE,sames")
r1ns_ele.GetXaxis().SetTitle("jet #eta")
r1ns_ele.GetYaxis().SetTitle("data/MC scale factor")
r1ns_ele.SetMinimum(0.5)
r1ns_ele.SetMaximum(1.5)
leg.Draw()
OUTSTRING = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/"
OUTSTRING += "data_back_SF_compare_ele_pho_"+ERA
drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
#drawRegion(SEL)
can.Update()
can.Print(OUTSTRING+'.png')
can.Print(OUTSTRING+'.pdf')
can.Close()
