#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
import json
import time
import uproot
import pandas as pd
import gc
from array import array
from awkward import *
import root_numpy
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
REGION             = "SR"#"ZtoEE"#"JetHT"
CUT                = "isSR"#"isZtoEE"#"isJetHT"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"#"isWtoEN"# && MT<100"#"isZtoMM"

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
NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"
PLOTDIR            = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+REGION+"/"
REBIN              = options.rebin
SAVE               = True
#LUMI               = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#8507.558453#59.74*1000#Run2018

#back = ["DYJetsToLL"]
#back = ["TTbarGenMET"]
#back = ["WJetsToLNu"]
back = ["QCD"]
#back = ["ZJetsToNuNu"]
back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET","VV"]
back = ["All"]
#back = ["QCD","WJetsToLNu","TTbarGenMET"]
data = ["SingleMuon"]
#data = ["SingleElectron"]
data = ["EGamma"]
data = ["MuonEG"]
#data = ["MET"]
data = ["HighMET"]
#data = ["JetHT"]
sign = [
    'SUSY_mh200_pl1000',
    'splitSUSY_M-2400_CTau-300mm','splitSUSY_M-2400_CTau-1000mm','splitSUSY_M-2400_CTau-10000mm','splitSUSY_M-2400_CTau-30000mm',
    #'gluinoGMSB_M2400_ctau1','gluinoGMSB_M2400_ctau3','gluinoGMSB_M2400_ctau10','gluinoGMSB_M2400_ctau30','gluinoGMSB_M2400_ctau100','gluinoGMSB_M2400_ctau300','gluinoGMSB_M2400_ctau1000','gluinoGMSB_M2400_ctau3000','gluinoGMSB_M2400_ctau10000','gluinoGMSB_M2400_ctau30000','gluinoGMSB_M2400_ctau50000',
    'XXTo4J_M100_CTau100mm','XXTo4J_M100_CTau300mm','XXTo4J_M100_CTau1000mm','XXTo4J_M100_CTau3000mm','XXTo4J_M100_CTau50000mm',
    'XXTo4J_M300_CTau100mm','XXTo4J_M300_CTau300mm','XXTo4J_M300_CTau1000mm','XXTo4J_M300_CTau3000mm','XXTo4J_M300_CTau50000mm',
    'XXTo4J_M1000_CTau100mm','XXTo4J_M1000_CTau300mm','XXTo4J_M1000_CTau1000mm','XXTo4J_M1000_CTau3000mm','XXTo4J_M1000_CTau50000mm',
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
dnn_bins = array('d', [0.,.00001,.0001,0.001,.01,.05,.1,.25,.5,.75,1.,])
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
np_bins = np.array(less_bins)
np_bins = np_bins[np_bins>=30]#only interested in non-zero bins

#bins=array('d', [1,10,20,30,40,50,60,70,80,90,100,1000])
#bins=np.array([1,10,20,30,40,50,60,70,80,90,100,1000])
#bins = bins.astype(np.float32)

bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,1000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
np_more_bins = np.array(more_bins)
#np_more_bins = np_more_bins[np_more_bins>=30]#only interested in non-zero bins

more_bins_eta = array('d', [-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
#more_bins=np.array([1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000])
#more_bins = more_bins.astype(np.float32)
###more_bins = bins
maxeff = 0.0015

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

def calculate_tag_eff(tree_weight_dict,sample_list,add_label="",check_closure=False,eta=False,j_idx=-1):

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
    for b in np_more_bins:
        np_den[b] = []#np.array([])
        np_num[b] = []#np.array([])
        np_w_den[b] = []#np.array([])
        np_w_num[b] = []#np.array([])
        np_weight[b] = []

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


        passed_num = np.array([])
        passed_den = np.array([])
        w_num = np.array([])
        w_den = np.array([])

        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight",CUT]#"nLeptons"
        #list_of_variables += ["MinSubLeadingJetMetDPhi"]
        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
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
                    cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                elif CUT == "isWtoEN":
                    cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                #elif CUT == "isSR":
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)
                #HEM
                cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))
                pt = arrays["Jets.pt"][cut_mask]
                sigprob = arrays["Jets.sigprob"][cut_mask]
                tag_mask = (sigprob > dnn_threshold)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                pt_untag = pt[untag_mask]
                pt_tag = pt[tag_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]
                #print "untag: ", pt_untag[0:5]
                #print "tag:", pt_tag[0:5]
                del arrays
                #here fill  the histo
                # . . .

                #Global fill
                passed_num = np.concatenate( (passed_num, np.hstack(pt_tag[pt_tag>0])) )
                passed_den = np.concatenate( (passed_den, np.hstack(pt_untag[pt_untag>0])) )
                w_num = np.concatenate( (w_num, np.hstack( pt_tag[pt_tag>0].astype(bool)*weight ) ) )
                w_den = np.concatenate( (w_den, np.hstack( pt_untag[pt_untag>0].astype(bool)*weight ) ) )
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


def draw_tag_eff(sample_list,add_label="",check_closure=False):
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
        graph[s].GetYaxis().SetRangeUser(-0.0001,0.002 if check_closure else maxeff)
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


def draw_data_combination(era,regions,regions_labels=[],datasets=[],add_label="",lab_2="",check_closure=False):
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
    leg = TLegend(0.6, 0.7, 1.0, 1.0)
    leg.SetTextSize(0.035)
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)
    can.SetLogx()#?
    
    for i, r in enumerate(regions):
        if r=="ZtoMM" or r=="WtoMN" or r=="WtoMN_MET" or r=="MN":
            s = "SingleMuon"
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET" or r=="EN":
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
        if len(datasets)>0:
            if datasets[i]!="":
                s = datasets[i]
        INPDIR  = BASEDIR + r + "/"#"_CR/"
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        print "Opening this file: ", INPDIR+"TagEff_"+s+reg_label+add_label+".root"
        infile = TFile(INPDIR+"TagEff_"+s+reg_label+add_label+".root", "READ")
        hist_den[r+reg_label+s] = TH1F()
        hist_num[r+reg_label+s] = TH1F()
        graph[r+reg_label+s] = TGraphAsymmErrors()
        hist_den[r+reg_label+s] = infile.Get("den_"+s)
        hist_num[r+reg_label+s] = infile.Get("num_"+s)
        #rebin
        #less_bins = bins#!
        den = hist_den[r+reg_label+s].Rebin(len(less_bins)-1,r+reg_label+s+"_den2",less_bins)
        num = hist_num[r+reg_label+s].Rebin(len(less_bins)-1,r+reg_label+s+"_num2",less_bins)
        graph[r+reg_label+s].BayesDivide(num,den)
        eff[r+reg_label+s] = TEfficiency(num,den)
        eff[r+reg_label+s].SetStatisticOption(TEfficiency.kBBayesian)
        eff[r+reg_label+s].SetConfidenceLevel(0.68)
        graph[r+reg_label+s].SetMarkerSize(marker_sizes[i])#(1.3)
        graph[r+reg_label+s].SetMarkerStyle(markers[i])#(21)#(sign_sampl[s]['marker'])
        graph[r+reg_label+s].SetMarkerColor(colors[i])#(samples[s]['fillcolor'])#(2)
        graph[r+reg_label+s].SetFillColor(colors[i])#(samples[s]['fillcolor'])#(2) 
        graph[r+reg_label+s].SetLineColor(colors[i])#(samples[s]['linecolor'])#(2)
        graph[r+reg_label+s].SetLineStyle(lines[i])#(2)#(2)
        graph[r+reg_label+s].SetLineWidth(2)
        graph[r+reg_label+s].GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else maxeff)
        graph[r+reg_label+s].GetYaxis().SetTitle("Tag efficiency")#("Efficiency (L1+HLT)")
        graph[r+reg_label+s].GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
        graph[r+reg_label+s].GetYaxis().SetTitleSize(0.05)#DCMS
        graph[r+reg_label+s].GetXaxis().SetRangeUser(bins[4],bins[-1])
        graph[r+reg_label+s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[r+reg_label+s].GetXaxis().SetTitleSize(0.04)
        graph[r+reg_label+s].GetXaxis().SetTitleOffset(1.1)
        leg.AddEntry(graph[r+reg_label+s], samples[s]['label']+"; "+r+reg_label, "PL")
        can.SetGrid()
        if i==0:
            graph[r+reg_label+s].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[r+reg_label+s].Draw("P,sames")
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
        elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET" or r=="EN":
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
                    xs = 1.*0.001
                elif('gluinoGMSB') in ss:
                    xs = 1.*0.001
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.*0.001
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
def background_prediction(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"

    if check_closure:
        dnn_threshold = 0.95
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996
        print  "DNN threshold: ", dnn_threshold

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
        EFFDIR[r+reg_label+dset] = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+ r + "/"
        table_yield[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err', 'Bin 2 Err combine'])
        table_pred[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
        table_pred_new[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Discrepancy', 'Bin 2 Pred from 0', 'Bin 2 Pred from 1', 'Bin 2 Pred Err', 'Bin 2 Discrepancy'])
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

        infiles[r+reg_label+dset] = TFile(EFFDIR[r+reg_label+dset]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root", "READ")
        print "r? ", r
        print "reg_label? ", reg_label
        print "dset? ", dset
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
        list_of_variables = ["isMC","Jets.pt","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight",CUT]#"nLeptons"
        #list_of_variables += ["MinSubLeadingJetMetDPhi"]

        #The only thing we can afford in RAM are final numbers
        #hence only a few arrays
        b0 = np.array([])
        b1 = np.array([])
        b2 = np.array([])
        pr_1 = {}
        pr_2 = {}
        pr_2_from_1 = {}

        #pr_1/2 depend on TEff, dictionaries
        for r in TEff.keys():
            print "come si chiamano le TEff keys? ", r
            pr_1[r] = np.array([])#[]
            pr_2[r] = np.array([])#[]
            pr_2_from_1[r] = np.array([])#[]
            eff[r] = []
            effUp[r] = []
            effDown[r] = []
            #0. Fill efficiencies np arrays once and forever
            for b in np_bins:
                binN = TEff[r].GetPassedHistogram().FindBin(b)
                eff[r].append(TEff[r].GetEfficiency(binN))
                #print("pt: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))
            #print r, eff[r]

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
                    cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                elif CUT == "isWtoEN":
                    cut_mask = np.logical_and(arrays[CUT]>0, arrays["MT"]<100 )
                #elif CUT == "isSR":
                    #print "!!!! try to kill QCD!!!"
                    #cut_mask = np.logical_and(arrays[CUT]>0 , arrays["MinJetMetDPhiBarrel"]>0.5 )
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM"]==0)))

                pt = arrays["Jets.pt"][cut_mask]
                sigprob = arrays["Jets.sigprob"][cut_mask]
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]
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
                #print "bin1 shape as mask ", bin1.shape
                #print bin1
                bin2_m = (sigprob[tag_mask].counts >1)
                bin0 = np.multiply(bin0_m,weight)
                bin1 = np.multiply(bin1_m,weight)
                bin2 = np.multiply(bin2_m,weight)
                #print "bin1 shape as multiplication ", bin1.shape
                #print bin1
                #df_all["bin0"] = bin0
                #df_all["bin1"] = bin1
                #df_all["bin2"] = bin2
                #print df_all[ ["Jets.sigprob","n_tag","bin0","bin1"]]

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

                for r in TEff.keys():
                    prob_vec[r] = []

                    for i in range(len(np_bins)):
                        if i<len(np_bins)-1:
                            prob_vec[r].append(np.logical_and(pt_untag>=np_bins[i],pt_untag<np_bins[i+1])*eff[r][i])#*weight)
                        else:
                            prob_vec[r].append((pt_untag>=np_bins[i])*eff[r][i])#*weight)

                    prob_tot = sum(prob_vec[r])
                    somma = (prob_tot*weight).sum()
                    cho = prob_tot.choose(2)
                    #u = cho.unzip()
                    combi = cho.unzip()[0] * cho.unzip()[1] * weight
                    #print "somma shape ", somma.shape
                    #print "bin1_m shape ", bin1_m.shape
                    #print "somma[bin1_m] shape ", (somma[bin1_m]).sum()#.shape
                    #print "bin 1 pred: ", somma.sum()
                    #print "bin 2 pred: ", combi.sum().sum()

                    bin1_pred[r] = np.concatenate( (bin1_pred[r],somma) )
                    bin2_pred[r] = np.concatenate( (bin2_pred[r],combi.sum()) )
                    bin2_pred_from_1[r] = np.concatenate( (bin2_pred_from_1[r], somma[bin1_m]  )  )#.append(0.)
                    #bin2_pred_from_1[r].append(somma if bin1[i]!=0 else 0.)

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

                #Here full pred
                for r in TEff.keys():
                    #print pr_1[r].shape, bin1_pred[r].shape
                    pr_1[r] = np.concatenate((pr_1[r],bin1_pred[r]))
                    pr_2[r] = np.concatenate((pr_2[r],bin2_pred[r]))
                    pr_2_from_1[r] = np.concatenate((pr_2_from_1[r],bin2_pred_from_1[r]))
                    #pr_1[r] += bin1_pred[r]
                    #pr_2[r] += bin2_pred[r]
                    #pr_2_from_1[r] += bin2_pred_from_1[r]

                b0 = np.concatenate((b0,bin0))
                b1 = np.concatenate((b1,bin1))
                b2 = np.concatenate((b2,bin2))

                #per-chunk pred
                #print "per-chunk pred bin 1: ", np.sum(bin1_pred[r])
                #print "per-chunk pred bin 2: ", np.sum(bin2_pred[r])
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
        print "Tot size of arrays: ", array_size_tot
        print "Size of tree_weights_array: ", len(tree_weights_array)
        print "Time elapsed to fill uproot array: ", end_uproot-start_uproot
        print "************************************"



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
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MEt.pt<25 && (HLT_PFJet500_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MEt.pt<25 && (HLT_PFJet500_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MEt.pt<25 && (HLT_PFJet500_v))"

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

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]

            print "controlla keys pr_1: ", pr_1.keys()
            print "controlla keys pred_1: ", pred_1.keys()
            print "e io sto cercando: ", r+reg_label+dset
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

            row[r+reg_label+dset] = [s, round(y_0,2), round(e_0,2), round(y_1,2), round(e_1,2), round(y_2,5), round(e_2,2),round(1.+e_2/y_2,5)]
            table_yield[r+reg_label+dset].add_row(row[r+reg_label+dset])

            rowP[r+reg_label+dset] = [s, round(pred_1[r+reg_label+dset],2), round(e_pred_1[r+reg_label+dset],2), round(pred_2[r+reg_label+dset],4), round(e_pred_2[r+reg_label+dset],4), round(pred_2_from_1[r+reg_label+dset],4), round(e_pred_2_from_1[r+reg_label+dset],4)]
            table_pred[r+reg_label+dset].add_row(rowP[r+reg_label+dset])

            rowPn[r+reg_label+dset] = [s, round(pred_1[r+reg_label+dset],2), round( abs(pred_1[r+reg_label+dset]-y_1),2), round(pred_2[r+reg_label+dset],4), round(pred_2_from_1[r+reg_label+dset],4), round( abs(pred_2_from_1[r+reg_label+dset] - pred_2[r+reg_label+dset]),4), round( abs( pred_2_from_1[r+reg_label+dset]-y_2  ),4  )]
            table_pred_new[r+reg_label+dset].add_row(rowPn[r+reg_label+dset])

            table_integral[r+reg_label+dset].add_row(rowI)


    for i,r in enumerate(extr_regions):
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        if len(datasets)>0:
            dset = datasets[i]

        if i==0:
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

samples_to_run = data#sign#back#data#back#data#back+data#data#data+back#+data
#jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
#jet_tag = "_jet_1"
jet_tag = ""#+
#jet_tag="_QCD"
#jet_tag+="_closure_0p95"
calculate_tag_eff(get_tree_weights(samples_to_run),samples_to_run,add_label=jet_tag,check_closure=False,eta=False,j_idx=-1)
draw_tag_eff(samples_to_run,add_label=jet_tag,check_closure=False)
#exit()

##calculate_tag_eff(samples_to_run,add_label=jet_tag+"_closure_0p99",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi
##draw_tag_eff(samples_to_run,add_label=jet_tag+"_closure_0p99",check_closure=True)#_low_dPhi_0p5_2_HLT_PFJet_combi

#jet_correlations(samples_to_run,add_label=jet_tag+"_zoom",check_closure=False)#_low_dPhi_0p5_2_HLT_PFJet_combi
#exit()


jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"

draw_data_combination(
    ERA,
    ["WtoMN","WtoEN","ZtoMM","ZtoEE","JetHT","TtoEM","SR"],
    #["JetHT","JetHT"],
    ##["ZtoMM","WtoMN","MR","JetMET","DiJetMET","DiJetMET"],#,"JetHT",SR
    ##regions_labels=["","","",jet_tag,"","","","","","","","",""],
    #datasets=["","","","","QCD","","","","","","","",],
    #datasets=["","QCD","","","","","","","",],
    #["ZtoMM","ZtoMM","DiJetMET","DiJetMET","DiJetMET"],#SR
    #regions_labels=["_jet_0","_jet_1","_jet_0","_jet_1","","","","","","","","",""],
    #MC in SR
    #["SR","SR","SR","SR","SR","SR"],
    #datasets=["ZJetsToNuNu","WJetsToLNu","QCD","TTbarGenMET","VV","HighMET"],
    add_label="",#"_vs_QCD_MC",#"_closure"
    lab_2="",#"_vs_MC",#"_0p95",#""
    check_closure=False#True#False#True

)

exit()


#SR

jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
'''
background_prediction_new(get_tree_weights(samples_to_run),
                          samples_to_run,
                          extr_regions=["JetHT","DiJetMET","JetMET","WtoMN","MN","MR","SR"],
                          regions_labels = ["","",jet_tag,"","","","","","","","","","","","","","","","",""],
                          add_label="",
                          label_2="_extrapolation_triplet_quadruplet",check_closure=False)

exit()
'''

'''
jet_tag = "_low_dPhi_0p5_2_HLT_PFJet_500"
background_prediction(get_tree_weights(samples_to_run),
                      samples_to_run,
                      #extr_regions= ["JetHT","DiJetMET","WtoEN","WtoMN","SR"],#["QCD","WtoMN","WtoEN"],#["WtoEN"],#["JetHT","DiJetMET","JetMET","ZtoMM","WtoMN","MN","MR","SR"],
                      #regions_labels = ["","","","","","","","","","","","",""],#["","",""],#[""],#["","",jet_tag,"","","","","","","","","","","","","","","","",""],
                      #datasets= ["","","","","","","","","","","","","","",""],#["QCD","",""],#[""],#["","","","","","","","","","","","","",],
                      #MC extrapolation
                      extr_regions= ["SR"],
                      regions_labels = ["","","","","","","","","","","","",""],
                      datasets= ["","","","","","","","","","","","","","",""],
                      add_label="",#"_closure_0p95",
                      label_2="_extrapolation_0p9999",check_closure=False)
exit()
'''

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
