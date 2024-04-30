#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TMultiGraph, TGraphAsymmErrors, TSpline, TSpline3
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox

#from NNIferenceCMSSW.LLP_NN_Inference.samples_v3 import sample, samples
#from NNInferenceCMSSW.LLP_NN_Inference.samples import sample, samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import selection
from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from decimal import *

gROOT.SetBatch(True)
gStyle.SetOptStat(0)
ERA = "2017"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared_compare_SUSY_HH/"

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples

#OUTPUTDIR = "plots/v5_calo_AOD_"+ERA+"_Gen/"
OUTPUTDIR = "plots/v6_calo_AOD_"+ERA+"_SR_compare_kinematics_SUSY_HeavyHiggs/"


colors = [856, 2, 881, 798, 602, 921, 870, 906, 801, 410, 4, 634, 838, 1, 398, 6,7,8,9,10,11]
colors+=colors

def correct_round(ratio):
    digit = Decimal( str(ratio) ).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP)
    return digit

def plot(sign,var,nbins=50,minimum=0,maximum=2000,bins=np.array([]),filename="",string="",part_var="GenBquarks",particle="#pi",norm=False):
    chain = {}
    hist = {}

    leg = TLegend(0.5, 0.75, 0.9, 0.9)

    #pal= 98
    #gStyle.SetPalette(pal)
    #gStyle.SetPaintTextFormat(".f")

    if part_var=="GenBquarks":
        cutstring = "isMC"
        cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 && Jets[0].sigprob>0.996 )"
        #cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 )"
        #cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 )"
        #cutstring = "(EventWeight * PUReWeight) * (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && MEt.pt>200 && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0)"
        #cutstring = "(EventWeight * PUReWeight)"
    #else:
    #    cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v && Flag2_globalSuperTightHalo2016Filter && Flag2_goodVertices && Flag2_EcalDeadCellTriggerPrimitiveFilter && Flag2_HBHENoiseFilter && Flag2_HBHEIsoNoiseFilter && Flag2_ecalBadCalibFilter && Flag2_eeBadScFilter && Flag2_BadPFMuonFilter) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 )"

    c1 = TCanvas("c1", "c1", 2000, 1000)
    c1.cd()

    for i, s in enumerate(sign):
        chain[s] = TChain("tree")
        if filename=="":
            for p, ss in enumerate(samples[s]['files']):
                chain[s].Add(NTUPLEDIR + ss + ".root")
        else:
            chain[s].Add(NTUPLEDIR + filename+".root")
        print "Entries: ", chain[s].GetEntries()
        #filename[s] = TFile("VBFH_HToSSTobbbb_MH-125_MS-30_ctauS-1000.root", "READ")
        if len(bins) ==0:
            hist[s] = TH1F(s, "", nbins, minimum, maximum)
        else:
            hist[s] = TH1F(s, "", len(bins)-1, bins)
        hist[s].Sumw2()
        chain[s].Project(s, var, cutstring)
        hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
        if norm:
            hist[s].Scale(1./hist[s].Integral())
            gStyle.SetPaintTextFormat('5.1f')
        leg.AddEntry(hist[s],samples[s]["label"],"PE")
        #c1.SetGrid()
        #c1.SetLogz()
        #c1.SetLogx()
        c1.SetLogy()
        hist[s].GetYaxis().SetTitle("Events/bin")
        hist[s].GetYaxis().SetTitleOffset(1.4)
        hist[s].GetXaxis().SetTitle(variable[var]["title"])
        hist[s].SetTitle(samples[s]['label'] if filename=="" else filename)
        hist[s].SetMarkerColor(colors[i])#(samples[s]['linecolor'])
        hist[s].SetLineColor(colors[i])#(samples[s]['linecolor'])
        hist[s].SetMarkerStyle(20)
        hist[s].SetLineStyle(samples[s]['linestyle'])
        if "HeavyHiggsToLLP" in s:
            hist[s].SetLineStyle(2)
            hist[s].SetMarkerStyle(25)
        #hist[s].Draw("colz")#()#
        hist[s].SetMarkerSize(1.2)#(2)#
        hist[s].Draw("PE,sames")#()#

    leg.Draw("sames")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".png")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".pdf")
    c1.Print("test1.png")
    c1.Print("test1.pdf")

    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")
    c1.Close()



def find_closest(probe,sign,var,nbins=50,minimum=0,maximum=2000,bins=np.array([]),filename="",string="",norm=False):
    chain = {}
    hist = {}

    print "\n"
    print "  --------------------- "
    print "  Probing ", probe

    leg1 = TLegend(0.5, 0.75, 0.9, 0.9)
    leg2 = TLegend(0.5, 0.75, 0.9, 0.9)
    cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 )"
    #cutstring = "(EventWeight * PUReWeight) * ( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) && nMuonsPassing==0 && nElectronsPassing==0 && nPhotonsPassing==0 && nTausPassing==0 && MEt.pt>200 && Jets[0].sigprob>0.996 )"

    c1 = TCanvas("c1", "c1", 2000, 1000)
    c1.cd()
    for i, s in enumerate(sign+[probe]):
        chain[s] = TChain("tree")
        if filename=="":
            for p, ss in enumerate(samples[s]['files']):
                chain[s].Add(NTUPLEDIR + ss + ".root")
        else:
            chain[s].Add(NTUPLEDIR + filename+".root")
        print "Entries: ", chain[s].GetEntries()
        #filename[s] = TFile("VBFH_HToSSTobbbb_MH-125_MS-30_ctauS-1000.root", "READ")
        if len(bins) ==0:
            hist[s] = TH1F(s, "", nbins, minimum, maximum)
        else:
            hist[s] = TH1F(s, "", len(bins)-1, bins)
        hist[s].Sumw2()
        chain[s].Project(s, var, cutstring)
        hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
        if norm:
            hist[s].Scale(1./hist[s].Integral())
            gStyle.SetPaintTextFormat('5.1f')
        c1.SetLogy()
        hist[s].GetYaxis().SetTitle("Events/bin")
        hist[s].GetYaxis().SetTitleOffset(1.4)
        hist[s].GetXaxis().SetTitle(variable[var]["title"])
        hist[s].SetTitle(samples[s]['label'] if filename=="" else filename)
        hist[s].SetMarkerColor(colors[i])#(samples[s]['linecolor'])
        hist[s].SetLineColor(colors[i])#(samples[s]['linecolor'])
        hist[s].SetMarkerStyle(20)
        hist[s].SetLineStyle(samples[s]['linestyle'])
        if "HeavyHiggsToLLP" in s:
            hist[s].SetLineStyle(2)
            hist[s].SetMarkerStyle(25)
        #hist[s].Draw("colz")#()#
        hist[s].SetMarkerSize(1.2)#(2)#
        hist[s].Draw("PE,sames")#()#
        leg1.AddEntry(hist[s],samples[s]["label"]+"; mean "+str(correct_round(hist[s].GetMean())),"PE")
    leg1.Draw("sames")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".png")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".pdf")
    c1.Print("test1.png")
    c1.Print("test1.pdf")

    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")
    c1.Close()

    
    mean_diff = {}
    md = []
    min_choice = 99999
    min_name = ""
    for i, s in enumerate(sign):
        mean_diff[s] = abs(hist[probe].GetMean()-hist[s].GetMean())
        print s, hist[probe].GetMean(), hist[s].GetMean(), mean_diff[s]
        md.append(mean_diff[s])
        if(mean_diff[s]<min_choice):
            min_choice = mean_diff[s]
            min_name = s

    c2 = TCanvas("c1", "c1", 2000, 1000)
    c2.cd()
    hist_diff = {}
    for i, s in enumerate(sign):
        hist_diff[s] = TGraph(1)
        hist_diff[s].SetPoint(0,mean_diff[s],1)
        hist_diff[s].GetYaxis().SetTitle("Events/bin")
        hist_diff[s].GetYaxis().SetTitleOffset(1.4)
        hist_diff[s].GetXaxis().SetTitle(variable[var]["title"])
        hist_diff[s].SetTitle(samples[s]['label'] if filename=="" else filename)
        hist_diff[s].SetMarkerColor(colors[i])#(samples[s]['linecolor'])
        hist_diff[s].SetLineColor(colors[i])#(samples[s]['linecolor'])
        hist_diff[s].SetMarkerStyle(21)
        hist_diff[s].SetLineStyle(samples[s]['linestyle'])
        hist_diff[s].SetMarkerSize(2.5)#(2)#
        print s, hist_diff[s].Print()
        hist_diff[s].SetTitle(samples[probe]['label'])
        if i==0:
            axis = hist_diff[s].GetXaxis()
            axis.SetLimits(min(md),max(md))
            axis.SetLimits(0,max(md))
            hist_diff[s].Draw("AP")
        else:
            hist_diff[s].Draw("P,sames")
        leg2.AddEntry(hist_diff[s],samples[s]["label"]+" : "+str( abs(correct_round(mean_diff[s])) ),"P")

    leg2.Draw("sames")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".png")
    #c1.Print(OUTPUTDIR+"2D_gen_b_quark_"+var+"_"+(s if filename=="" else filename)+string+".pdf")
    c2.Print("test_diff1.png")
    c2.Print("test_diff1.pdf")

    print min_name, min_choice

    if not gROOT.IsBatch(): raw_input("Press Enter to continue...")
    c2.Close()



taglio = "_preselections"
signal = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000']#["mchi200_pl1000","mchi300_pl1000","mchi400_pl1000"]

signal = ['SUSY_mh127_ctau500','SUSY_mh300_ctau500','SUSY_mh600_ctau500','SUSY_mh1000_ctau500','SUSY_mh1800_ctau500']
signal += ['SUSY_mh127_ctau3000','SUSY_mh300_ctau3000','SUSY_mh600_ctau3000','SUSY_mh1000_ctau3000','SUSY_mh1800_ctau3000']

signal = ['SUSY_mh127_ctau500_HH','SUSY_mh250_ctau500_HH','HeavyHiggsToLLPTo4b_mH600_mX60_ct1000','HeavyHiggsToLLPTo4b_mH600_mX250_ct1000','HeavyHiggsToLLPTo4b_mH400_mX150_ct1000','HeavyHiggsToLLPTo4b_mH400_mX40_ct1000']

signal = ['SUSY_mh1000_ctau500_HH','HeavyHiggsToLLPTo4b_mH1000_mX450_ct1000','HeavyHiggsToLLPTo4b_mH1000_mX100_ct1000']
signal = ['HeavyHiggsToLLPTo4b_mH1000_mX450_ct1000','HeavyHiggsToLLPTo4b_mH1000_mX100_ct1000','SUSY_mh1000_ctau500_HH','HeavyHiggsToLLPTo4b_mH600_mX60_ct1000','HeavyHiggsToLLPTo4b_mH600_mX250_ct1000','SUSY_mh600_ctau500_HH','SUSY_mh400_ctau500_HH',]

#signal = ['HeavyHiggsToLLPTo4b_mH600_mX250_ct1000','HeavyHiggsToLLPTo4b_mH600_mX60_ct1000','SUSY_mh600_ctau500_HH','SUSY_mh400_ctau500_HH',]
#signal = ['HeavyHiggsToLLPTo4b_mH1000_mX450_ct1000','HeavyHiggsToLLPTo4b_mH1000_mX100_ct1000','SUSY_mh600_ctau500_HH','SUSY_mh1000_ctau500_HH']
#plot(signal,"GenBquarks[0].pt",nbins=25,minimum=0.,maximum=1000,bins=np.array([]),filename="",string=taglio,norm=True)
#plot(signal,"Jets[0].pt",nbins=30,minimum=0.,maximum=1000,bins=np.array([]),filename="",string=taglio,norm=True)

signal = ['HeavyHiggsToLLPTo4b_mH400_mX150_ct1000','SUSY_mh200_ctau500_HH','SUSY_mh127_ctau500_HH',]
signal = ['HeavyHiggsToLLPTo4b_mH400_mX40_ct1000','SUSY_mh200_ctau3000_HH','SUSY_mh127_ctau3000_HH','SUSY_mh200_ctau500_HH','SUSY_mh127_ctau500_HH',]

sign = ['SUSY_mh127_ctau3000','SUSY_mh127_ctau500','SUSY_mh200_ctau500','SUSY_mh200_ctau3000',]

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
    #'SUSY_mh1250_ctau500',
    #'SUSY_mh1500_ctau500',
    #'SUSY_mh1800_ctau500',
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
    #'SUSY_mh1250_ctau3000',
    #'SUSY_mh1500_ctau3000',
    #'SUSY_mh1800_ctau3000',
]

plot(['HeavyHiggsToLLPTo4b_mH800_mX80_ct100','HeavyHiggsToLLPTo4b_mH800_mX80_ct1000','HeavyHiggsToLLPTo4b_mH800_mX350_ct100','HeavyHiggsToLLPTo4b_mH800_mX350_ct1000'],"Jets[0].sigprob",nbins=40,minimum=0.,maximum=1,bins=np.array([]),filename="",string=taglio,norm=True)
exit()

find_closest('HeavyHiggsToLLPTo4b_mH800_mX80_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH800_mX80_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH800_mX80_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo4b_mH800_mX350_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH800_mX350_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH800_mX350_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo4b_mH400_mX40_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH400_mX40_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH400_mX40_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo4b_mH400_mX150_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH400_mX150_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo4b_mH400_mX150_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)


#2b2nu
find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

exit()

find_closest('zPrimeTo4b_mZ4500_mX450_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo4b_mZ4500_mX450_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo4b_mZ4500_mX450_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('zPrimeTo4b_mZ4500_mX2200_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo4b_mZ4500_mX2200_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo4b_mZ4500_mX2200_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('zPrimeTo2b2nu_mZ4500_mX450_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo2b2nu_mZ4500_mX450_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo2b2nu_mZ4500_mX450_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

find_closest('zPrimeTo2b2nu_mZ4500_mX2200_ct100',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo2b2nu_mZ4500_mX2200_ct1000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)
find_closest('zPrimeTo2b2nu_mZ4500_mX2200_ct10000',sign,"TaggedJets.pt",nbins=500,minimum=0.,maximum=3000,bins=np.array([]),filename="",string=taglio,norm=True)

#plot(signal,"GenBquarks[0].pt",nbins=30,minimum=0.,maximum=500,bins=np.array([]),filename="",string=taglio,norm=True)

#plot(signal,"GenBquarks[0].pt",nbins=30,minimum=0.,maximum=500,bins=np.array([]),filename="",string=taglio,norm=True)
#plot(signal,"GenHiggs[0].dRdaughters",nbins=30,minimum=0.,maximum=5,bins=np.array([]),filename="",string=taglio,norm=True)
#plot(signal,"MEt.pt",nbins=40,minimum=0.,maximum=1000,bins=np.array([]),filename="",string=taglio,norm=True)

##plot(signal,"Jets.eta",nbins=50,minimum=-2.,maximum=2,bins=np.array([]),filename="",string=taglio,norm=True)


#plot(signal,"pt",nbins=50,minimum=0.,maximum=50000,bins=np.array([9.9,25,50,100,250,500,1000,5000,10000,50000]),filename="",string=taglio,part_var="GenBquarks")
