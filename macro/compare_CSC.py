#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
import numpy as np

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)
gStyle.SetOptStat(0)



can = TCanvas("can","can",1000,800)
can.cd()

x = np.array([127,150,200,300,400])
csc = np.array([1297.4198,459.63263,197.40341,46.85089,4.9493933])
calo_prev = np.array([678.701599,283.337708,259.862213,248.979462,228.738632])#from prev production

calo = np.array([730.544404,317.621143,273.940743,249.328664,215.253658])
calo_dphi = np.array([643.408549,238.033163,147.460812,113.025792,133.195419])
calo_dphi_ntuple = np.array([532.475519,199.414747,124.115046,89.704445,111.461306])
calo_dphi_barrel = np.array([653.214985,268.732309,160.382303,124.030620,145.629585])

csc_g = TGraph()#len(x)-1,x,csc)
calo_g = TGraph()#len(x)-1,x,calo)
calo_g_dphi = TGraph()
calo_g_dphi_ntuple = TGraph()
calo_g_dphi_barrel = TGraph()

for a in range(len(x)):
    csc_g.SetPoint(a,x[a],csc[a])
    calo_g.SetPoint(a,x[a],calo[a])
    calo_g_dphi.SetPoint(a,x[a],calo_dphi[a])
    calo_g_dphi_ntuple.SetPoint(a,x[a],calo_dphi_ntuple[a])
    calo_g_dphi_barrel.SetPoint(a,x[a],calo_dphi_barrel[a])

csc_g.SetTitle("")
calo_g.SetTitle("")
calo_g_dphi.SetTitle("")
calo_g_dphi_ntuple.SetTitle("")
calo_g_dphi_barrel.SetTitle("")

csc_g.SetLineColor(8)
csc_g.SetLineWidth(2)
csc_g.SetLineStyle(1)
csc_g.SetMarkerColor(8)
csc_g.SetMarkerStyle(20)

calo_g.SetLineColor(2)
calo_g.SetLineWidth(2)
calo_g.SetLineStyle(2)
calo_g.SetMarkerColor(2)
calo_g.SetMarkerStyle(21)

calo_g_dphi.SetLineColor(1)
calo_g_dphi.SetLineWidth(2)
calo_g_dphi.SetLineStyle(2)
calo_g_dphi.SetMarkerColor(1)
calo_g_dphi.SetMarkerStyle(21)

calo_g_dphi_ntuple.SetLineColor(856)
calo_g_dphi_ntuple.SetLineWidth(2)
calo_g_dphi_ntuple.SetLineStyle(2)
calo_g_dphi_ntuple.SetMarkerColor(856)
calo_g_dphi_ntuple.SetMarkerStyle(21)

calo_g_dphi_barrel.SetLineColor(801)
calo_g_dphi_barrel.SetLineWidth(2)
calo_g_dphi_barrel.SetLineStyle(2)
calo_g_dphi_barrel.SetMarkerColor(801)
calo_g_dphi_barrel.SetMarkerStyle(21)
#drawCMS(samples, 137*1000, "Simulation",left_marg_CMS=0.2)

leg = TLegend(0.65, 0.85-0.1, .9, 1.-0.1)
#with 4 graphs
leg = TLegend(0.55, 0.85-0.2, .9, 1.-0.1)
#leg.SetHeader(sampl)
leg.AddEntry(csc_g,"CSC analysis","PL")
leg.AddEntry(calo_g,"calo analysis","PLL")
leg.AddEntry(calo_g_dphi,"calo analysis, min dphi","PLL")
#leg.AddEntry(calo_g_dphi_ntuple,"calo analysis, min dphi ntuple","PLL")
#leg.AddEntry(calo_g_dphi_barrel,"calo analysis, min dphi barrel","PLL")

csc_g.GetXaxis().SetTitle("m_{#chi} (GeV)")
csc_g.GetYaxis().SetTitle("Event yield (weighted with current limits on SUSY x-sec)")
csc_g.SetMaximum(1.5e03)
csc_g.SetMinimum(-0.1)
#drawCMS(samples, 137*1000, "Simulation",left_marg_CMS=0.32)

csc_g.Draw("APL")
calo_g.Draw("PL,same")
calo_g_dphi.Draw("PL,same")
#calo_g_dphi_ntuple.Draw("PL,same")
#calo_g_dphi_barrel.Draw("PL,same")
leg.Draw()
drawCMS(samples, 137*1000, "Simulation",onTop=True,left_marg_CMS=0.1)
#can.Print('CSC_vs_calo_SUSY_event_yield.png')
#can.Print('CSC_vs_calo_SUSY_event_yield.pdf')
can.Print('CSC_vs_calo_SUSY_event_yield_min_dphi.png')
can.Print('CSC_vs_calo_SUSY_event_yield_min_dphi.pdf')

can.SetLogy()
csc_g.SetMaximum(1.e04)
csc_g.SetMinimum(1)
can.Update()
can.Print('CSC_vs_calo_SUSY_event_yield_min_dphi_log.png')
can.Print('CSC_vs_calo_SUSY_event_yield_min_dphi_log.pdf')
