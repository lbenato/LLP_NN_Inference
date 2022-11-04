#! /usr/bin/env python
import os, multiprocessing
import copy
import math
#import uproot
import yaml
import root_numpy
import numpy as np
import pandas as pd
##import mplhep as hep
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)
#plt.style.use(hep.style.CMS)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def plot_nTag():
    '''SF'''
    FOLD = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_Gen/"
    f1= "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HH.root"
    f2 = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HH.root"
    f3= "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HH.root"
    f4 = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HH.root"
    LUMI = -1

    h1 = TH1F("h1","",6,0,6)
    ch1 = TChain("tree")
    ch1.Add(FOLD+f1)
    ch1.Project("h1","nTagJets_0p996","")

    h2 = TH1F("h2","",5,0,5)
    ch2 = TChain("tree")
    ch2.Add(FOLD+f2)
    ch2.Project("h2","nTagJets_0p996","")

    h3 = TH1F("h3","",6,0,6)
    ch3 = TChain("tree")
    ch3.Add(FOLD+f3)
    ch3.Project("h3","nTagJets_0p996","")

    h4 = TH1F("h4","",6,0,6)
    ch4 = TChain("tree")
    ch4.Add(FOLD+f4)
    ch4.Project("h4","nTagJets_0p996","")

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.5, 0.83, 0.95, 0.98)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    h1.SetLineColor(861)

    h1.SetMarkerColor(861)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(1.5)
    h1.SetLineWidth(2)

    h2.SetLineColor(413)
    h2.SetMarkerColor(413)
    h2.SetMarkerStyle(21)
    h2.SetMarkerSize(1.5)
    h2.SetLineWidth(2)

    h3.SetLineColor(2)
    h3.SetMarkerColor(2)
    h3.SetMarkerStyle(21)
    h3.SetMarkerSize(1.5)
    h3.SetLineWidth(2)

    h4.SetLineColor(1)
    h4.SetMarkerColor(1)
    h4.SetMarkerStyle(21)
    h4.SetMarkerSize(1.5)
    h4.SetLineWidth(2)

    #h_3m.Scale(1./h_3m.Integral())
    #h_0p5m.Scale(1./h_0p5m.Integral())
    h1.Draw("P")
    h2.Draw("P,sames")
    h3.Draw("P,sames")
    h4.Draw("P,sames")
    h1.SetMinimum(0)
    h1.GetXaxis().SetTitle("n tagged jets")
    h1.GetYaxis().SetTitle("Events")
    h1.GetXaxis().SetTitleSize(0.04)
    h1.GetYaxis().SetTitleSize(0.04)
    h2.GetYaxis().SetTitle("Events")
    h2.GetXaxis().SetTitleSize(0.04)
    h2.GetYaxis().SetTitleSize(0.04)
    drawCMS_simple(LUMI, "Simulation", ERA="", onTop=True)
    leg.AddEntry(h1,samples["SUSY_mh400_ctau500"]["label"],"PL")
    leg.AddEntry(h2,samples["SUSY_mh400_ctau3000"]["label"],"PL")
    leg.AddEntry(h3,samples["SUSY_mh1000_ctau500"]["label"],"PL")
    leg.AddEntry(h4,samples["SUSY_mh1000_ctau3000"]["label"],"PL")
    leg.Draw()

    can.Print("fig_arc/n_tag_sig.pdf")
    can.Print("fig_arc/n_tag_sig.png")
    can.Close()

def plot_ctau():
    '''SF'''
    FOLD = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_Gen/"
    filename_3m = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HH.root"
    filename_0p5m = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HH.root"
    LUMI = -1

    #bins = 
    h_3m = TH1F("h_3m","",100,0,200)
    chain_3m = TChain("tree")
    chain_3m.Add(FOLD+filename_3m)
    #posterior time
    chain_3m.Project("h_3m","Math::sqrt(GenLLPs[0].travelX/10.*GenLLPs[0].travelX/10. + GenLLPs[0].travelY/10.*GenLLPs[0].travelY/10. + GenLLPs[0].travelZ/10.*GenLLPs[0].travelZ/10.)/(GenLLPs[0].beta * (1./( Math::sqrt(1. - GenLLPs[0].beta*GenLLPs[0].beta ) )))","")

    h_0p5m = TH1F("h_0p5m","",100,0,200)
    chain_0p5m = TChain("tree")
    chain_0p5m.Add(FOLD+filename_0p5m)
    #posterior time
    chain_0p5m.Project("h_0p5m","Math::sqrt(GenLLPs[0].travelX/10.*GenLLPs[0].travelX/10. + GenLLPs[0].travelY/10.*GenLLPs[0].travelY/10. + GenLLPs[0].travelZ/10.*GenLLPs[0].travelZ/10.)/(GenLLPs[0].beta * (1./( Math::sqrt(1. - GenLLPs[0].beta*GenLLPs[0].beta ) )))","")

    f05 = TF1("f05","[0]*exp(-x/[1])",0,200)
    f3 = TF1("f3","[0]*exp(-x/[1])",0,200)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.35, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    h_3m.SetLineColor(861)
    h_3m.SetMarkerColor(861)
    h_3m.SetMarkerStyle(20)
    h_3m.SetMarkerSize(1.5)
    h_3m.SetLineWidth(2)

    h_0p5m.SetLineColor(413)
    h_0p5m.SetMarkerColor(413)
    h_0p5m.SetMarkerStyle(21)
    h_0p5m.SetMarkerSize(1.5)
    h_0p5m.SetLineWidth(2)

    h_3m.Scale(1./h_3m.Integral())
    h_0p5m.Scale(1./h_0p5m.Integral())
    h_0p5m.Draw("P")
    h_3m.Draw("P,sames")
    h_3m.GetXaxis().SetTitle("LLP physical decay length (m)")
    h_3m.GetYaxis().SetTitle("Events")
    h_3m.GetXaxis().SetTitleSize(0.04)
    h_3m.GetYaxis().SetTitleSize(0.04)
    h_0p5m.GetXaxis().SetTitle("LLP physical decay length (m)")
    h_0p5m.GetYaxis().SetTitle("Events")
    h_0p5m.GetXaxis().SetTitleSize(0.04)
    h_0p5m.GetYaxis().SetTitleSize(0.04)
    f05.SetParameter(0,8.28411e-02)
    f05.SetParameter(1,0.5)
    f05.SetParameter(2,0.5*0.5)
    f3.SetParameter(0,8.28411e-02)
    f3.SetParameter(1,0.5)
    f3.SetParameter(2,0.5*0.5)
    h_3m.Fit(f3,"ERBS")
    h_0p5m.Fit(f05,"ERBS")
    #h_3m.SetMaximum(0.2)
    #h_0p5m.SetMaximum(0.2)
    drawCMS_simple(LUMI, "Simulation", ERA="", onTop=True)
    leg.AddEntry(h_3m,samples["SUSY_mh400_ctau3000"]["label"],"PL")
    leg.AddEntry(h_0p5m,samples["SUSY_mh400_ctau500"]["label"],"PL")
    leg.Draw()
    #f05.SetLineColor(2)
    #f05.Draw("L,sames")
    can.Print("fig_arc/ctau.pdf")
    can.Print("fig_arc/ctau.png")
    can.Close()


def plot_trigger():
    '''SF'''
    FOLD = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_Gen/"
    filename_3m = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HH.root"
    filename_0p5m = "SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HH.root"
    LUMI = -1

    #bins = 
    num3 = TH1F("num3","",100,0,1000)
    num05 = TH1F("num05","",100,0,1000)
    den3 = TH1F("den3","",100,0,1000)
    den05 = TH1F("den05","",100,0,1000)
    num3.Sumw2()
    den3.Sumw2()
    num05.Sumw2()
    den05.Sumw2()

    cut = "nMuonsPassing==0 && nTausPassing==0 && nPhotonsPassing==0 && nElectronsPassing==0 && Jets[0].sigprob>0.996 && Jets[1].sigprob>0.996"# && Jets[1].radiusLLP>30 && abs(Jets[0].eta)<1. && abs(Jets[1].eta)<1."
    cut_num = cut + " && (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v)"
    cut_num = cut + " && (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v)"

    chain3 = TChain("tree")
    chain3.Add(FOLD+filename_3m)
    chain3.Project("den3","MEt.pt",cut)
    chain3.Project("num3","MEt.pt",cut_num)

    chain05 = TChain("tree")
    chain05.Add(FOLD+filename_0p5m)
    chain05.Project("den05","MEt.pt",cut)
    chain05.Project("num05","MEt.pt",cut_num)

    eff3 = TGraphAsymmErrors()
    eff05 = TGraphAsymmErrors()
    eff3.BayesDivide(num3,den3)
    eff05.BayesDivide(num05,den05)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.55, 0.85, 0.9, 0.99)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    eff3.SetLineColor(861)
    eff3.SetMarkerColor(861)
    eff3.SetMarkerStyle(20)
    eff3.SetMarkerSize(1.5)
    eff3.SetLineWidth(2)

    eff05.SetLineColor(413)
    eff05.SetMarkerColor(413)
    eff05.SetMarkerStyle(21)
    eff05.SetMarkerSize(1.5)
    eff05.SetLineWidth(2)

    eff05.Draw("AP")
    eff3.Draw("P,sames")
    eff3.GetXaxis().SetTitle("LLP physical decay length (m)")
    eff3.GetYaxis().SetTitle("Events")
    eff3.GetXaxis().SetTitleSize(0.04)
    eff3.GetYaxis().SetTitleSize(0.04)
    eff05.GetXaxis().SetTitle("LLP physical decay length (m)")
    eff05.GetYaxis().SetTitle("Events")
    eff05.GetXaxis().SetTitleSize(0.04)
    eff05.GetYaxis().SetTitleSize(0.04)
    drawCMS_simple(LUMI, "Simulation", ERA="", onTop=True)
    leg.AddEntry(eff3,samples["SUSY_mh400_ctau3000"]["label"],"PL")
    leg.AddEntry(eff05,samples["SUSY_mh400_ctau500"]["label"],"PL")
    leg.Draw()
    #f05.SetLineColor(2)
    #f05.Draw("L,sames")
    can.Print("fig_arc/met_turnon.pdf")
    can.Print("fig_arc/met_turnon.png")
    can.Close()

    exit()
    for a in lol:
        tmp_den_b = infile.Get("den_b")
        tmp_num_b = infile.Get("num_b")
        tmp_den_d = infile.Get("den_d")
        tmp_num_d = infile.Get("num_d")
        tmp_den_b_2ns = infile.Get("den_b_2ns")
        tmp_num_b_2ns = infile.Get("num_b_2ns")
        tmp_den_d_2ns = infile.Get("den_d_2ns")
        tmp_num_d_2ns = infile.Get("num_d_2ns")
        tmp_den_b.SetDirectory(0)
        tmp_num_b.SetDirectory(0)
        tmp_den_d.SetDirectory(0)
        tmp_num_d.SetDirectory(0)
        tmp_den_b_2ns.SetDirectory(0)
        tmp_num_b_2ns.SetDirectory(0)
        tmp_den_d_2ns.SetDirectory(0)
        tmp_num_d_2ns.SetDirectory(0)
        if i==0:
            num_b = tmp_num_b
            den_b = tmp_den_b
            num_d = tmp_num_d
            den_d = tmp_den_d
            num_b_2ns = tmp_num_b_2ns
            den_b_2ns = tmp_den_b_2ns
            num_d_2ns = tmp_num_d_2ns
            den_d_2ns = tmp_den_d_2ns
        else:
            num_b.Add(tmp_num_b)
            den_b.Add(tmp_den_b)
            num_d.Add(tmp_num_d)
            den_d.Add(tmp_den_d)
            num_b_2ns.Add(tmp_num_b_2ns)
            den_b_2ns.Add(tmp_den_b_2ns)
            num_d_2ns.Add(tmp_num_d_2ns)
            den_d_2ns.Add(tmp_den_d_2ns)
        den_b.SetDirectory(0)
        num_b.SetDirectory(0)
        den_d.SetDirectory(0)
        num_d.SetDirectory(0)
        den_b_2ns.SetDirectory(0)
        num_b_2ns.SetDirectory(0)
        den_d_2ns.SetDirectory(0)
        num_d_2ns.SetDirectory(0)

    print den_b.Print()
    ratio = num_d.Clone("r")
    ratio.Divide(den_d)
    ratio.Divide(num_b)
    ratio.Multiply(den_b)

    ratio_2ns = num_d_2ns.Clone("r_2ns")
    ratio_2ns.Divide(den_d_2ns)
    ratio_2ns.Divide(num_b_2ns)
    ratio_2ns.Multiply(den_b_2ns)



    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    eff_b.SetLineColor(861)
    eff_b.SetFillColorAlpha(861,0.3)
    #eff_b.SetFillStyle(3001)
    eff_b.SetLineWidth(0)
    eff_b.SetMarkerColor(861)
    eff_b.SetMarkerStyle(21)
    eff_b.SetMarkerSize(1.3)

    eff_d.SetLineColor(1)
    eff_d.SetLineWidth(2)
    eff_d.SetMarkerColor(1)
    eff_d.SetMarkerStyle(20)
    eff_d.SetMarkerSize(1.3)

    eff_b.GetYaxis().SetRangeUser(-0.01,1.01)
    eff_b.Draw("APE2")
    eff_d.Draw("P,sames")
    eff_b.GetXaxis().SetTitle("Jet #eta")
    eff_b.GetYaxis().SetTitle("Tag efficiency")
    eff_b.GetXaxis().SetTitleSize(0.04)
    eff_b.GetYaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    drawCMS_simple(LUMI, "", ERA="", onTop=True)
    leg.AddEntry(eff_b,"MC background","PF")
    leg.AddEntry(eff_d,"data","PL")
    leg.Draw()
    can.Print("fig_arc/TDJSignalEfficiency_pho.pdf")
    can.Print("fig_arc/TDJSignalEfficiency_pho.png")
    can.Close()


    #Add discrepancy wrt 2 ns in quadrature
    g = TGraphAsymmErrors(ratio.GetNbinsX()-1)
    for b in range(1,ratio.GetNbinsX()+1):
        i = b-1
        #print "point: ", b
        diff = abs(ratio.GetBinContent(b)-ratio_2ns.GetBinContent(b))
        diff_rel = diff/ratio.GetBinContent(b)
        bin_error_rel = ratio.GetBinError(b)/ratio.GetBinContent(b)
        error_rel = math.sqrt( diff_rel**2 + bin_error_rel**2  )
        error = error_rel*ratio.GetBinContent(b)
        g.SetPointX(i,ratio.GetBinCenter(b))
        g.SetPointY(i,ratio.GetBinContent(b))
        g.SetPointEXhigh(i, 0.5)
        g.SetPointEXlow(i, 0.5)
        #Should assign 0.5 of the error to high, 0.5 to low
        print "BUGFIX!"
        g.SetPointEYhigh(i, 0.5*error)
        g.SetPointEYlow(i, 0.5*error)

    print g.Print()
    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    g.SetTitle("")
    g.SetLineColor(418)
    g.SetFillColorAlpha(418,0.5)#(920,1.)
    #g.SetFillStyle(3001)
    g.SetLineWidth(2)
    g.SetMarkerColor(418)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.3)
    g.GetYaxis().SetRangeUser(-0.01,2.01)
    ##g.GetYaxis().SetRangeUser(0.9,1.1)
    g.Draw("APE2")
    g.GetXaxis().SetTitle("Jet #eta")
    g.GetYaxis().SetTitle("Data/MC scale factor")
    g.GetXaxis().SetTitleSize(0.04)
    g.GetYaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    drawCMS_simple(LUMI, "", ERA="", onTop=True)
    #leg.AddEntry(ratio,"data","PL")
    #leg.Draw()
    can.Print("fig_arc/TDJSignalEfficiencyScaleFactor_pho.pdf")
    can.Print("fig_arc/TDJSignalEfficiencyScaleFactor_pho.png")
    can.Close()



def plot_met_sf():
    met_file = "/nfs/dust/cms/group/cms-llp/MET_trigger_SF_Caltech/METTriggers_SF.root"
    eras = ["2016","2017","2018"]
    labels = ["trigger_efficiency_Summer16","trigger_efficiency_Fall17","trigger_efficiency_Fall18"]
    lumis = [1,1,1]
    LUMI = 137478.722953

    infile = TFile(met_file,"READ")
    print infile.ls()
    sf = {}

    for i,era in enumerate(eras):
        sf[era] = infile.Get(labels[i])

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.4+0.2,0.3, 0.65+0.2, 0.55)
    leg.SetTextSize(0.035)

    colors = [1,418,2]
    markers = [24,25,26]
    for i,era in enumerate(eras):
        sf[era].SetTitle("")
        sf[era].SetMarkerColor(colors[i])
        sf[era].SetLineColor(colors[i])
        sf[era].SetLineWidth(2)
        sf[era].SetMarkerStyle(markers[i])
        sf[era].SetMarkerSize(1.25)
        sf[era].GetXaxis().SetTitle("E_{T}^{miss} (GeV)")
        sf[era].GetXaxis().SetTitleSize(0.04)
        sf[era].GetXaxis().SetLabelSize(0.04)
        sf[era].GetYaxis().SetTitle("Data/MC scale factor")
        sf[era].GetYaxis().SetTitleSize(0.04)
        sf[era].GetYaxis().SetLabelSize(0.04)
        sf[era].Draw("PL,sames")
        leg.AddEntry(sf[era],era,"PL")


    drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print("fig_arc/met_sf.pdf")
    can.Print("fig_arc/met_sf.png")



#plot_ctau()
#plot_trigger()
#plot_met_sf()
plot_nTag()
