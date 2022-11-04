#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
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
if options.bash: gROOT.SetBatch(True)
gStyle.SetOptStat(0)


sampl = "QCD"
#sampl = "SUSY"
#sampl = "HH"

can = TCanvas("can","can",1000,800)
can.cd()
chain = TChain("tree")
##chain.Add("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/prova_"+sampl+".root")
chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1.root")
#chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/TChiHH_mass400_pl1000.root")
#chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2.root")

jet = TH1F("jet","", 50, 0, 3.2)
jet_b = TH1F("jet_b","", 50, 0, 3.2)
f_jet = TH1F("f_jet","", 50, 0, 3.2)
f_jet_b = TH1F("f_jet_b","", 50, 0, 3.2)
jet.Sumw2()
jet_b.Sumw2()
f_jet.Sumw2()
f_jet_b.Sumw2()

chain.Project("jet", "MinJetMetDPhi", "nCHSJets>0")
chain.Project("jet_b", "MinJetMetDPhiBarrel", "nCHSJetsAcceptanceCalo>0")
chain.Project("f_jet", "MinFatJetMetDPhi", "nCHSFatJets>0")
chain.Project("f_jet_b", "MinFatJetMetDPhiBarrel", "nCHSFatJetsAcceptanceCalo>0")

jet.SetLineColor(1)
jet_b.SetLineColor(8)
f_jet.SetLineColor(4)
f_jet_b.SetLineColor(2)

jet.SetLineWidth(3)
jet_b.SetLineWidth(3)
f_jet.SetLineWidth(3)
f_jet_b.SetLineWidth(3)

jet.SetLineStyle(2)
f_jet.SetLineStyle(2)
jet.SetTitle("")
jet_b.SetTitle("")
f_jet.SetTitle("")
f_jet_b.SetTitle("")

jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")

leg = TLegend(0.3, 0.8, .6, 1.)
leg.SetHeader(sampl)
leg.AddEntry(jet,"AK4","L")
leg.AddEntry(jet_b,"AK4 barrel","L")
leg.AddEntry(f_jet,"AK8","L")
leg.AddEntry(f_jet_b,"AK8 barrel","L")

if sampl=="QCD":
    jet.Draw("HISTO")
f_jet.Draw("HISTO,same")
f_jet_b.Draw("HISTO,same")
f_jet.Draw("HISTO,same")
jet_b.Draw("HISTO,same")
jet.Draw("HISTO,same")
leg.Draw()
can.Print('MinDPhi_'+sampl+'.png')
can.Print('MinDPhi_'+sampl+'.pdf')
