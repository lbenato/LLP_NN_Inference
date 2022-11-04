#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
import json
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
REGION             = "MR"
CUT                = "isMR"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"

#REGION             = "JetMET"#"WtoEN"#"SR_HEM"#"debug_new_with_overlap"#"SR"#"ZtoMM_CR"

#CUT                = "isJetMET_dPhi_Lep"
#CUT                = "isJetMET_low_dPhi_Lep"
#CUT                = "isJetMET_dPhi_MET_200_Lep"
#CUT                = "isJetMET_low_dPhi_MET_200_Lep"

#CUT                = "isJetMET_dPhi"
#CUT                = "isJetMET_low_dPhi_500"#
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
data = ["HighMET"]
#data = ["JetHT"]

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

def calculate_tag_eff(sample_list,add_label="",check_closure=False):
    for i, s in enumerate(sample_list):
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
            if (CUT == "isJetHT" and not(event.isJetHT)): continue
            if (CUT == "isJetMET" and not(event.isJetMET)): continue
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

            for j in range(event.nCHSJetsAcceptanceCalo):
                #print event.Jets[j].pt
                if (event.Jets[j].muEFrac<0.6 and event.Jets[j].eleEFrac<0.6 and event.Jets[j].photonEFrac<0.8 and event.Jets[j].timeRecHitsEB>-1):
                    hist_den[s].Fill(event.Jets[j].pt)
                    if(check_closure):
                        if(event.Jets[j].sigprob>0.9 and event.Jets[j].sigprob<0.996):
                            hist_num[s].Fill(event.Jets[j].pt)                        
                    else:
                        if(event.Jets[j].sigprob>0.996):
                            hist_num[s].Fill(event.Jets[j].pt)
                    if(event.Jets[j].timeRecHitsEB>0.09 and event.Jets[j].gammaMaxET<0.16 and event.Jets[j].minDeltaRPVTracks>0.06 and event.Jets[j].cHadEFrac<0.06):
                        hist_num_cutbased[s].Fill(event.Jets[j].pt)
                    n+=1
            if(n_ev%10000==0):
                print ("event n. %d/%d (%.2f perc.)")%(n_ev,chain[s].GetEntries(),100.*float(n_ev)/float(chain[s].GetEntries()))

            #if(n>=10000): break

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
        eff[s].GetXaxis().SetRangeUser(more_bins[0],more_bins[-1])
        eff[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.1)

        eff_cutbased[s] = TGraphAsymmErrors()
        eff_cutbased[s].BayesDivide(hist_num_cutbased[s],hist_den[s])

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

def draw_tag_eff(sample_list,add_label=""):
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
        hist_num[s] = TH1F()
        graph[s] = TGraphAsymmErrors()
        hist_den[s] = infiles.Get("den_"+s)
        hist_num[s] = infiles.Get("num_"+s)
        #rebin
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
        graph[s].SetMarkerStyle(21)#(sign_sampl[s]['marker'])
        graph[s].SetMarkerColor(samples[s]['fillcolor'])#(2)
        graph[s].SetFillColor(samples[s]['fillcolor'])#(2) 
        graph[s].SetLineColor(samples[s]['linecolor'])#(2)
        graph[s].SetLineStyle(2)#(2)
        graph[s].SetLineWidth(2)
        graph[s].GetYaxis().SetRangeUser(-0.0001,maxeff)
        graph[s].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
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

        outfile = TFile(PLOTDIR+"TagTEfficiency_"+s+add_label+".root","RECREATE")
        print "Info in <TFile::Write>: TEfficiency root file "+PLOTDIR+"TagTEfficiency_"+s+add_label+".root has been created"
        outfile.cd()
        graph[s].Write("eff_"+s)
        eff[s].Write("TEff_"+s)
        outfile.Close()

    leg.Draw()
    drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    drawCMS_simple(LUMI, "Preliminary", True)
    can.Print(PLOTDIR+"TagEff"+add_label+".png")
    can.Print(PLOTDIR+"TagEff"+add_label+".pdf")

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
        graph[s].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
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


def draw_data_combination(era,regions,regions_labels=[],add_label="",lab_2=""):
    BASEDIR = "plots/Efficiency/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency/v5_calo_AOD_"+era+"_combination/"
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1800, 800)#krass
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    #leg = TLegend(0.2, 0.6, 0.9, 0.9)#y: 0.6, 0.9
    leg = TLegend(0.3, 0.8, 1.0, 1.0)
    leg.SetTextSize(0.035)
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)
    can.SetLogx()#?
    
    for i, r in enumerate(regions):
        if r=="ZtoMM" or r=="WtoMN" or r=="WtoMN_MET" or r=="MN":
            s = "SingleMuon"
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET":
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
            #s = "MET"
            s = "HighMET"
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
        else:
            print "Invalid region, aborting..."
            exit()
        INPDIR  = BASEDIR + r + "/"#"_CR/"
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        print "Opening this file: ", INPDIR+"TagEff_"+s+reg_label+add_label+".root"
        infile = TFile(INPDIR+"TagEff_"+s+reg_label+add_label+".root", "READ")
        hist_den[r+reg_label] = TH1F()
        hist_num[r+reg_label] = TH1F()
        graph[r+reg_label] = TGraphAsymmErrors()
        hist_den[r+reg_label] = infile.Get("den_"+s)
        hist_num[r+reg_label] = infile.Get("num_"+s)
        #rebin
        #less_bins = bins#!
        den = hist_den[r+reg_label].Rebin(len(less_bins)-1,r+reg_label+"_den2",less_bins)
        num = hist_num[r+reg_label].Rebin(len(less_bins)-1,r+reg_label+"_num2",less_bins)
        graph[r+reg_label].BayesDivide(num,den)
        eff[r+reg_label] = TEfficiency(num,den)
        eff[r+reg_label].SetStatisticOption(TEfficiency.kBBayesian)
        eff[r+reg_label].SetConfidenceLevel(0.68)
        graph[r+reg_label].SetMarkerSize(marker_sizes[i])#(1.3)
        graph[r+reg_label].SetMarkerStyle(markers[i])#(21)#(sign_sampl[s]['marker'])
        graph[r+reg_label].SetMarkerColor(colors[i])#(samples[s]['fillcolor'])#(2)
        graph[r+reg_label].SetFillColor(colors[i])#(samples[s]['fillcolor'])#(2) 
        graph[r+reg_label].SetLineColor(colors[i])#(samples[s]['linecolor'])#(2)
        graph[r+reg_label].SetLineStyle(lines[i])#(2)#(2)
        graph[r+reg_label].SetLineWidth(2)
        graph[r+reg_label].GetYaxis().SetRangeUser(-0.0001,maxeff)
        graph[r+reg_label].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
        graph[r+reg_label].GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        graph[r+reg_label].GetYaxis().SetTitleSize(0.05)#DCMS
        graph[r+reg_label].GetXaxis().SetRangeUser(bins[4],bins[-1])
        graph[r+reg_label].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[r+reg_label].GetXaxis().SetTitleSize(0.04)
        graph[r+reg_label].GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(graph[r+reg_label], samples[s]['label']+"; "+r+reg_label, "PL")
        can.SetGrid()
        if i==0:
            graph[r+reg_label].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[r+reg_label].Draw("P,sames")
        infile.Close()

    leg.Draw()
    #drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    #drawCMS_simple(LUMI, "Preliminary", True)
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".png")
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".pdf")
    #can.Print(OUTDIR+"buh.pdf")

def draw_comparison(era,regions,extra="",col=0,maxeff=maxeff):
    BASEDIR = "plots/Efficiency/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency/v5_calo_AOD_"+era+"_combination/"
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
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET":
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
    
'''
def background_prediction(tree_weight_dict,sample_list):

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
def background_prediction_new_one_eff(tree_weight_dict,sample_list,extr_region="",add_label=""):

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
'''
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
        elif "ZtoEE" in r or "WtoEN" in r:
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

        max_n=chain[s].GetEntries()+10#100000
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

def GetEffWeightBin2New(event, TEff, n):
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

                        


    for r in TEff.keys():
        EffW[r] = p2[r]
        EffWUp[r] = p2Up[r]
        EffWLow[r] = p2Low[r]
    #return n_untagged, n_tagged, EffW, EffWUp, EffWLow
    return EffW, EffWUp, EffWLow


samples_to_run = data#back+data#data#data+back#+data
calculate_tag_eff(samples_to_run,add_label="_closure",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi
draw_tag_eff(samples_to_run,add_label="_closure",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi

#2016:
#calculate_tag_eff(samples_to_run,add_label="_RunB-F")#(back)#
#calculate_tag_eff(samples_to_run,add_label="_RunG-H")#(back)#
#draw_tag_eff(samples_to_run,add_label="_RunG-H")#(back)#

'''
background_prediction_new(get_tree_weights(samples_to_run),
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
