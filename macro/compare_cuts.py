#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
gROOT.SetBatch(True)
gStyle.SetOptStat(0)

ERA  = "2018"

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.variables import *


NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2017_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_v3/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_SR/"#_InvertBeamHalo/"
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_WtoMN_noMT/"#JetHT/"#WtoEN_noMT/"#_InvertBeamHalo/"
LUMI = 59.74*1000

MAIN_OUT = "plots/v5_calo_AOD_"+ERA+"_SR/"#_InvertBeamHalo/"
#MAIN_OUT = "plots/v5_calo_AOD_2018_WtoMN/"#JetHT/"#WtoEN/"#_InvertBeamHalo/"

#back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
#sgn = ["ggH_MH1000_MS150_ctau1000_XL"]
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000','SUSY_mh150_pl1000']
sgn = ["VV","WJetsToLNu","ZJetsToNuNu","TTbarGenMET","QCD"]
sgn = ["All"]
##sgn = ["VV"]
#sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000','SUSY_mh150_pl1000']
#sgn = ["SingleMuon"]
#sgn = ["SingleElectron"]
#sgn = ["EGamma"]
#sgn = ["JetHT"]
#sgn = ["HighMET"]
#sgn = ["HighMETBH"]

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)

h1 = {}
h2 = {}
h1_t = {}
h2_t = {}
num = {}
den = {}
graph = {}
tree = {}
c2 = {}
profX = {}
bins=np.array([0.9,2.5,5,7.5,10,12.5,15,20,30,40,50,60,70,80,90,100,120,140,160,200,250,500,1000,1500])
WEIGHT = "EventWeight"

def pl(tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut1="Jets.isGenMatchedCaloCorr",cut2="HT>0",do_radius=False,add_label=""):
    print "cut1: ", cut1
    print "cut2: ", cut2
    r_ecal = 129
    r_hcal = 179
    r_magnet = 295
    r_mb1 = 402
    r_mb4 = 738
    r_TOB3 = 110
    r_TOB2 = 95
    r_TOB1 = 70
    r_TIB3 = 55
    r_TIB2 = 40
    r_TIB1 = 25
    minimum = 0
    maximum = 1
    v_TOB3 = TLine(r_TOB3,minimum,r_TOB3,maximum)
    v_TOB2 = TLine(r_TOB2,minimum,r_TOB2,maximum)
    v_TOB1 = TLine(r_TOB1,minimum,r_TOB1,maximum)
    v_TIB3 = TLine(r_TIB3,minimum,r_TIB3,maximum)
    v_TIB2 = TLine(r_TIB2,minimum,r_TIB2,maximum)
    v_TIB1 = TLine(r_TIB1,minimum,r_TIB1,maximum)
    v_ecal = TLine(r_ecal,minimum,r_ecal,maximum)
    v_hcal = TLine(r_hcal,minimum,r_hcal,maximum)
    v_magnet = TLine(r_magnet,minimum,r_magnet,maximum)
    v_mb1 = TLine(r_mb1,minimum,r_mb1,maximum)
    v_mb4 = TLine(r_mb4,minimum,r_mb4,maximum)
    v_ecal.SetLineColor(2)
    v_hcal.SetLineColor(881)
    v_magnet.SetLineColor(1)
    v_mb1.SetLineColor(801)
    v_mb4.SetLineColor(4)
    v_ecal.SetLineWidth(4)
    v_hcal.SetLineWidth(4)
    v_magnet.SetLineWidth(4)
    v_mb1.SetLineWidth(4)
    v_mb4.SetLineWidth(3)
    v_TIB1.SetLineWidth(3)
    v_TIB2.SetLineWidth(3)
    v_TIB3.SetLineWidth(3)
    v_TOB1.SetLineWidth(3)
    v_TOB2.SetLineWidth(3)
    v_TOB3.SetLineWidth(3)

    v_TIB1.SetLineStyle(4)
    v_TIB2.SetLineStyle(4)
    v_TIB3.SetLineStyle(4)
    v_TOB1.SetLineStyle(4)
    v_TOB2.SetLineStyle(4)
    v_TOB3.SetLineStyle(4)
    v_ecal.SetLineStyle(9)
    v_hcal.SetLineStyle(2)
    v_magnet.SetLineStyle(3)
    v_mb1.SetLineStyle(8)
    v_mb4.SetLineStyle(8)

    for b in sgn:#back+sgn:
        tree[b] = TChain(tree_name)
        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(NTUPLEDIR + ss + '.root'):
                print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            tree[b].Add(NTUPLEDIR + ss + '.root')
        if do_radius:
            h1[b] = TH1F(str(b)+"_1",str(b)+"_1",len(bins)-1,bins)
            h2[b] = TH1F(str(b)+"_2",str(b)+"_2",len(bins)-1,bins)
            h1_t[b] = TH1F(str(b)+"_1_t",str(b)+"_1_t",len(bins)-1,bins)
            h2_t[b] = TH1F(str(b)+"_2_t",str(b)+"_2_t",len(bins)-1,bins)
        else:
            h1[b] = TH1F(str(b)+"_1",str(b)+"_1",variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"])
            h2[b] = TH1F(str(b)+"_2",str(b)+"_2",variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"])
            h1_t[b] = TH1F(str(b)+"_1_t",str(b)+"_1_t",variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"])
            h2_t[b] = TH1F(str(b)+"_2_t",str(b)+"_2_t",variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"])
        h1[b].Sumw2()
        h2[b].Sumw2()
        h1_t[b].Sumw2()
        h2_t[b].Sumw2()
        tree[b].Project(str(b)+"_1",v_x,cut1+"*"+WEIGHT+"*(Jets[0].sigprob<=0.9)")
        tree[b].Project(str(b)+"_2",v_x,cut2+"*"+WEIGHT+"*(Jets[0].sigprob<=0.9)")
        tree[b].Project(str(b)+"_1_t",v_x,cut1+"*"+WEIGHT+"*(Jets[0].sigprob>0.9)")
        tree[b].Project(str(b)+"_2_t",v_x,cut2+"*"+WEIGHT+"*(Jets[0].sigprob>0.9)")
        #print "hist content: ", h[b].GetEntries()
        #profX[b] = TProfile(h[b].ProfileX("prof"+str(b)))
        h1[b].GetXaxis().SetTitle(variable[v_x]["title"])
        h2[b].GetXaxis().SetTitle(variable[v_x]["title"])
        #h1[b].GetYaxis().SetTitle(variable[v_y]["title"])

        if do_radius:
            leg = TLegend(1-0.3, 0.65-0.2, 0.95, 0.9-0.1)
        else:
            leg = TLegend(1-0.3, 0.9-0.2, 0.95, 0.9-0.1)
            leg = TLegend(1-0.5, 0.9-0.2, 0.95, 0.9)
        leg.SetHeader(b+" ; era: "+ERA)
        leg.AddEntry(h1[b],"nominal","L")
        leg.AddEntry(h2[b],"problematic #varphi","L")
        leg.AddEntry(h1_t[b],"nominal DNN>0.9","L")
        leg.AddEntry(h2_t[b],"problematic #varphi DNN>0.9","L")
        if do_radius:
            leg.AddEntry(v_TOB3,"Strips", "L")
            leg.AddEntry(v_ecal,"ECAL","L")
            leg.AddEntry(v_hcal,"HCAL","L")
            leg.AddEntry(v_magnet,"solenoid","L")
            leg.AddEntry(v_mb1,"MB1","L")
            leg.AddEntry(v_mb4,"MB4","L")
            
        c2[b] = TCanvas("can","can",1000,800)
        c2[b].cd()
        c2[b].SetGrid()
        if do_radius:
            c2[b].SetLogx()
        #c2[b].SetLogy()
        #c2[b].SetLogz()
        h1[b].Scale(1./h1[b].Integral())
        h2[b].Scale(1./h2[b].Integral())
        h1_t[b].Scale(1./h1_t[b].Integral())
        h2_t[b].Scale(1./h2_t[b].Integral())

        massim = max(0,h1[b].GetMaximum())
        massim = max(massim,h2[b].GetMaximum())
        massim = max(massim,h1_t[b].GetMaximum())
        massim = max(massim,h2_t[b].GetMaximum())

        h1[b].SetLineColor(861)
        h1[b].SetLineWidth(2)
        h1[b].SetLineStyle(2)
        h1[b].Draw("HIST")#("COLZ")#()#
        h1[b].SetMaximum(massim*1.01)

        h2[b].SetLineColor(2)
        h2[b].SetLineWidth(2)
        h2[b].SetLineStyle(2)
        h2[b].Draw("HIST,SAMES")#("COLZ")#()#

        h1_t[b].SetLineColor(861)
        h1_t[b].SetLineWidth(2)
        h1_t[b].Draw("HIST,SAMES")#("COLZ")#()#

        h2_t[b].SetLineColor(2)
        h2_t[b].SetLineWidth(2)
        h2_t[b].Draw("HIST,SAMES")#("COLZ")#()#

        #profX[b].SetLineColor(632+2)
        #profX[b].SetMarkerColor(632+2)
        #profX[b].SetMarkerStyle(20)
        #profX[b].SetMarkerSize(1.2)
        #profX[b].SetLineWidth(3)
        #profX[b].Draw("PL,sames")

        if do_radius:
            v_ecal.Draw("sames")
            v_hcal.Draw("sames")
            v_magnet.Draw("sames")
            v_mb1.Draw("sames")
            v_mb4.Draw("sames")
            v_TOB3.Draw("sames")
            v_TOB2.Draw("sames")
            v_TOB1.Draw("sames")
            v_TIB3.Draw("sames")
            v_TIB2.Draw("sames")
            v_TIB1.Draw("sames")
        leg.Draw()
        c2[b].Print(MAIN_OUT+v_x.replace('.', '_')+'_'+b+add_label+'.png')
        c2[b].Print(MAIN_OUT+v_x.replace('.', '_')+'_'+b+add_label+'.pdf')


#pl(tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",do_radius=True)
#pl(tree_name="tree",v_x="Jets.eta",v_y="Jets.sigprob",cut="(HT>0)",do_radius=False)
CUT = "(HT>0)"
#CUT = "(( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1))"
#CUT = "( ( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && MinJetMetDPhi>0.5)"
#CUT = "(Jets[0].sigprob>0.996 && MinJetMetDPhi>0.5)"
#CUT = "(Jets[0].sigprob>0.996)"
label = ""
#label = "__MinDPhi_0p5_tagged_jets"
#label = "_tagged_jets_B-F"
#label = "_tagged_jets"

x_list = ["nTrackConstituents","nRecHitsEB","eFracRecHitsEB","eta","timeRecHitsEB"]
x_list = ['eta','nTrackConstituents','nSelectedTracks', 'timeRecHitsEB', 'eFracRecHitsEB', 'nRecHitsEB', 'sig1EB', 'sig2EB', 'ptDEB', 'cHadEFrac', 'nHadEFrac', 'eleEFrac', 'photonEFrac', 'ptAllTracks', 'ptAllPVTracks', 'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks', 'minDeltaRPVTracks']
#x_list = ["timeRecHitsEB"]

label = "_test"
for x in x_list:
    pl(tree_name="tree",v_x="Jets[0]."+x,cut1="(Jets[0].phi<=0 || Jets[0].phi>=1)",cut2="(Jets[0].phi>0 && Jets[0].phi<1)",do_radius=False,add_label=label)

exit()

y_list_tagged = ["eta"]
y_list_tagged = ['eta','nTrackConstituents','nSelectedTracks', 'timeRecHitsEB', 'eFracRecHitsEB', 'nRecHitsEB', 'sig1EB', 'sig2EB', 'ptDEB', 'cHadEFrac', 'nHadEFrac', 'eleEFrac', 'photonEFrac', 'ptAllTracks', 'ptAllPVTracks', 'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks', 'minDeltaRPVTracks']
#y_list_tagged = ['nRecHitsEB']

#CUT = "( ( isMC ? 1 : (RunNumber>=319077 ? nCHSJets_in_HEM_pt_30_all_eta==0 : 1)) && Jets[0].sigprob>0.996)"
CUT = "(Jets[0].sigprob>0.996)"
#CUT = "(HT>0)"
label = ""
label = "_tagged_jets"
#CUT = "(Jets[0].sigprob>0.0)"
#label = "_tagged_jets_0p0"


for y in y_list_tagged:
    pl(tree_name="tree",v_x="Jets[0].eta",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    pl(tree_name="tree",v_x="Jets[0].phi",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)

exit()



y_list_log = ["sigprob"]
CUT = "(( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1))"
CUT = "( isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM_pt_30_all_eta==0 : 1) )"
CUT = "(HT>0)"
for y in y_list_log:
    pl(tree_name="tree",v_x="Jets[0].eta",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    pl(tree_name="tree",v_x="Jets[0].phi",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    #for x in y_list_tagged:
    #    pl(tree_name="tree",v_x="Jets[0]."+x,v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)

exit()
