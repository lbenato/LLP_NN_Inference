#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
import numpy as np

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
chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tagger_eff_vs_dR/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/0000/output_1.root")



#j_num = TH1F("j_num","", 50, 0., 5.)
#j_den = TH1F("j_den","", 50, 0., 5.)
#fj_num = TH1F("fj_num","", 50, 0., 5.)
#fj_den = TH1F("fj_den","", 50, 0., 5.)


fj_wp = "0.999"
fj_wp2 = "0.99999"
fj_wp2_2m = "0.99999"
bins = np.array([0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,3.0,5.0])
bins = np.array([0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.25,1.5,2.0,3.0,5.0])
#bins = np.array([0.,1.,2.,3.,4.,5.])

j_num = TH1F("j_num","", len(bins)-1,bins)
j_den = TH1F("j_den","", len(bins)-1,bins)
j_num_2t = TH1F("j_num_2t","", len(bins)-1,bins)
j_den_2t = TH1F("j_den_2t","", len(bins)-1,bins)
fj_num = TH1F("fj_num","", len(bins)-1,bins)
fj_den = TH1F("fj_den","", len(bins)-1,bins)
fj_num2 = TH1F("fj_num2","", len(bins)-1,bins)
fj_den2 = TH1F("fj_den2","", len(bins)-1,bins)
fj_num2_2m = TH1F("fj_num2_2m","", len(bins)-1,bins)
fj_den2_2m = TH1F("fj_den2_2m","", len(bins)-1,bins)

j_num.Sumw2()
j_den.Sumw2()
j_num_2t.Sumw2()
j_den_2t.Sumw2()
fj_num.Sumw2()
fj_den.Sumw2()
fj_num2.Sumw2()
fj_den2.Sumw2()
fj_num2_2m.Sumw2()
fj_den2_2m.Sumw2()

chain.Project("j_den", "dR_bb_jet_0", "")
chain.Project("j_num", "dR_bb_jet_0", "Jets[0].sigprob>0.996")

match = "nTagJets_0p996>1"
match = "((Jets[1].isGenMatchedCaloCorrLLPAccept && Jets[1].radiusLLP == Jets[0].radiusLLP) || (Jets[2].isGenMatchedCaloCorrLLPAccept && Jets[2].radiusLLP == Jets[0].radiusLLP) || (Jets[3].isGenMatchedCaloCorrLLPAccept && Jets[3].radiusLLP == Jets[0].radiusLLP) || (Jets[4].isGenMatchedCaloCorrLLPAccept && Jets[4].radiusLLP == Jets[0].radiusLLP) )"
match = "((Jets[1].isGenMatchedCaloCorrLLPAccept && Jets[1].radiusLLP == Jets[0].radiusLLP))"

chain.Project("j_den_2t", "dR_bb_jet_0", match)
chain.Project("j_num_2t", "dR_bb_jet_0", "Jets[0].sigprob>0.996 && "+match)
chain.Project("fj_den", "dR_bb_fatjet_0", "")
chain.Project("fj_num", "dR_bb_fatjet_0", "FatJets[0].sigprob>"+fj_wp)
chain.Project("fj_den2", "dR_bb_fatjet_0", "")
chain.Project("fj_num2", "dR_bb_fatjet_0", "FatJets[0].sigprob>"+fj_wp2)
chain.Project("fj_den2_2m", "dR_bb_fatjet_0", "")
chain.Project("fj_num2_2m", "dR_bb_fatjet_0", "FatJets[0].sigprob>"+fj_wp2_2m + " &&  FatJets[0].nMatchedGenBquarksCaloCorr==2")

#j_den.Rebin(len(bins)-1,"j_den_2",bins)
#j_num.Rebin(len(bins)-1,"j_num_2",bins)
#fj_den.Rebin(len(bins)-1,"fj_den",bins)
#fj_num.Rebin(len(bins)-1,"fj_num",bins)

jet = TGraphAsymmErrors()
jet.BayesDivide(j_num,j_den)

jet_2t = TGraphAsymmErrors()
jet_2t.BayesDivide(j_num_2t,j_den_2t)

f_jet = TGraphAsymmErrors()
f_jet.BayesDivide(fj_num,fj_den)

f_jet2 = TGraphAsymmErrors()
f_jet2.BayesDivide(fj_num2,fj_den2)

f_jet2_2m = TGraphAsymmErrors()
f_jet2_2m.BayesDivide(fj_num2_2m,fj_den2_2m)

jet.SetLineColor(1)
f_jet.SetLineColor(2)
f_jet2.SetLineColor(8)
f_jet2_2m.SetLineColor(870-10)
jet_2t.SetLineColor(801)

jet.SetMarkerColor(1)
f_jet.SetMarkerColor(2)
f_jet2.SetMarkerColor(8)
f_jet2_2m.SetMarkerColor(870-10)
jet_2t.SetMarkerColor(801)

jet.SetMarkerStyle(21)
f_jet.SetMarkerStyle(21)
f_jet2.SetMarkerStyle(21)
f_jet2_2m.SetMarkerStyle(21)
jet_2t.SetMarkerStyle(21)

jet.SetLineWidth(3)
f_jet.SetLineWidth(3)
f_jet2.SetLineWidth(3)
f_jet2_2m.SetLineWidth(3)
jet_2t.SetLineWidth(3)

jet.SetLineStyle(2)
f_jet.SetLineStyle(2)
f_jet2.SetLineStyle(2)
f_jet2_2m.SetLineStyle(2)
jet_2t.SetLineStyle(2)

jet.SetTitle("")
f_jet.SetTitle("")

jet.GetXaxis().SetTitle("#Delta R (b,b)  - LLP matched to leading jet")
#f_jet.GetXaxis().SetTitle("#Delta R (b,b)")
jet.GetYaxis().SetTitle("Efficiency")
f_jet.GetYaxis().SetTitle("Efficiency")

leg = TLegend(0.5, 0.75, 1., 1.)
leg.SetHeader("Heavy Higgs: m_{H2}=2 TeV; m_{S}=250 GeV; c#tau_{S}=1 m")
leg.AddEntry(jet,"AK4, w.p. 0.996","PL")
leg.AddEntry(jet_2t,"AK4, w.p. 0.996, 2 tags","PL")
leg.AddEntry(f_jet,"AK8, w.p. "+fj_wp,"PL")
leg.AddEntry(f_jet2,"AK8, w.p. "+fj_wp2,"PL")
leg.AddEntry(f_jet2_2m,"AK8, w.p. "+fj_wp2_2m+"; 2 b-matched","PL")

jet.SetMaximum(1.1)
jet.SetMinimum(0.)
jet.Draw("AP")
jet_2t.Draw("P,same")
f_jet.Draw("P,same")
f_jet2.Draw("P,same")
f_jet2_2m.Draw("P,same")
can.SetLogx()
##fj_den.Draw("P")
leg.Draw()
can.Print('tagger_eff_vs_dR.png')
can.Print('tagger_eff_vs_dR.pdf')
