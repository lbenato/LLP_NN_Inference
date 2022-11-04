#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
import json
from array import array
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from collections import defaultdict

gROOT.SetBatch(True)
gStyle.SetOptStat(0)

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]
lines = [1,1,1,1,1,2,2,2,2]
markers = [20,20,20,20,20,24,24,24,24]
siz = 1.3
marker_sizes = [siz,siz,siz,siz,siz,siz,siz,siz,siz]
bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,1000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])

ERA = "2016"
if ERA=="2016":
    SUB_ERA = "B-F"
    #SUB_ERA = "G-H"
    L_SUB_ERA = "_Run"+SUB_ERA
elif ERA=="2018":
    SUB_ERA = "AB"
    #SUB_ERA = "CD"
    L_SUB_ERA = "_Run"+SUB_ERA
else:
    SUB_ERA = ""
    L_SUB_ERA = ""
REGIONS = ["CR","MR"]#,"SR"]#"WtoMN"#"SR","MR"
J = {}
L = {}

can = TCanvas("can","can", 1000, 800)
can.SetLeftMargin(0.15)
can.cd()
leg = TLegend(0.65, 0.77, 1., .99)
leg.SetTextSize(0.034)
can.SetLogx()

for i,r in enumerate(REGIONS):

    if r=="CR":
        DATA  = "SingleMuon" #MET
        L_REG = "WtoMN"
        L_DATA = "SingleMuon"
    else:
        DATA  = "MET"
        L_REG = r
        L_DATA = "HighMET"

    infile_jj = TFile("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/ForLisa/no_minDR_cut/"+r+"_"+DATA+"_Eff.root","READ")
    print "Data "+ERA+SUB_ERA+";1"
    J[r] = TEfficiency(infile_jj.FindObjectAny("Data "+ERA+SUB_ERA+";1"))


    infile_l = TFile("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v5_calo_AOD_"+ERA+"_"+L_REG+"/TagTEfficiency_"+L_DATA+L_SUB_ERA+".root","READ")
    L[r] = infile_l.Get("TEff_"+L_DATA)
    
    J[r].SetLineColor(colors[i])
    J[r].SetMarkerColor(colors[i])
    J[r].SetMarkerSize(siz)
    J[r].SetMarkerStyle(20)
    J[r].SetLineWidth(2)
    J[r].SetTitle("")

    L[r].SetLineColor(colors[i])
    L[r].SetMarkerColor(colors[i])
    L[r].SetMarkerSize(siz)
    L[r].SetMarkerStyle(24)
    L[r].SetLineWidth(2)
    L[r].SetLineStyle(2)
    L[r].SetTitle("")
    if i==0:
        J[r].Draw("AP")
    else:
        J[r].Draw("P,sames")
    leg.AddEntry(J[r],r+" "+DATA+ERA+SUB_ERA,"PL")
    leg.AddEntry(L[r],r+" Lisa","PL")

for i,r in enumerate(REGIONS):
    L[r].Draw("P,sames")
    gPad.Update()
    graph = J[r].GetPaintedGraph()
    graph.SetMinimum(-0.0000001)
    graph.SetMaximum(0.0012)
    #graph.GetXaxis().SetTitle("jet p_{T} (GeV)")
    gPad.Update()

leg.Draw()
can.Print("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v5_calo_AOD_"+ERA+"_combination/TagTEfficiency_compare_"+ERA+L_SUB_ERA+".png")
can.Print("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v5_calo_AOD_"+ERA+"_combination/TagTEfficiency_compare_"+ERA+L_SUB_ERA+".pdf")


#TEfficiency *T = (TEfficiency*)(_file0->FindObjectAny("Data 2016B-F;1"));

