#! /usr/bin/env python
import os, multiprocessing
import copy
import math
import uproot
import yaml
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile

cosmic_files = [
    #'/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2017/CosmicsRun2017F-CosmicSP-PromptReco-v1.root',

    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018A-CosmicSP-PromptReco-v1.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018A-CosmicSP-PromptReco-v2.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018A-CosmicSP-PromptReco-v3.root',

    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018B-CosmicSP-PromptReco-v1.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018B-CosmicSP-PromptReco-v2.root',

    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018C-CosmicSP-PromptReco-v1.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018C-CosmicSP-PromptReco-v2.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018C-CosmicSP-PromptReco-v3.root',

    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018D-CosmicSP-PromptReco-v1.root',
    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018D-CosmicSP-PromptReco-v2.root',

    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018E-CosmicSP-PromptReco-v1.root',
]

#cosmic_files = [
#    '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/CosmicsRun2018E-CosmicSP-PromptReco-v1.root',
#]


list_of_variables = ["*"]

gen = uproot.iterate(cosmic_files,"tree",list_of_variables)
hd = TH1F("","hd",1,-0.5,0.5)
hn = TH1F("","hn",1,-0.5,0.5)
hn_pass = TH1F("","hn_pass",1,-0.5,0.5)
ho = TH1F("","ho",1,-0.5,0.5)
ho_2016BF = TH1F("","ho_2016BF",1,-0.5,0.5)
ho_2016GH = TH1F("","ho_2016GH",1,-0.5,0.5)
ho_2017 = TH1F("","ho_2017",1,-0.5,0.5)
ho_2018 = TH1F("","ho_2018",1,-0.5,0.5)
hd.Sumw2()
hn.Sumw2()
hn_pass.Sumw2()
ho.Sumw2()
ho_2016BF.Sumw2()
ho_2016GH.Sumw2()
ho_2017.Sumw2()
ho_2018.Sumw2()

den = 0.
num = 0.
obs_2016BF = 2
obs_2016GH = 2
obs_2017 = 1
obs_2018 = 1 
obs = 6
for i, arrays in enumerate(gen):
    print "Reading ", cosmic_files[i]
    #n_clusters_3_stations>1 && nCosmicMuonsOneLeg==1 && CosmicMuonsOneLeg.pt>50 && CaloJets.nRecHitsEB>2
    cut_mask = arrays["n_clusters_3_stations"]>1
    cut_mask = np.logical_and(cut_mask, arrays["nCosmicMuonsOneLeg"]==1)
    cut_mask = np.logical_and(cut_mask, arrays["nCosmicMuons"]>1)
    cut_mask = np.logical_and(cut_mask, arrays["nCaloJets"]<5)

    cut_cosmic = arrays["CosmicMuonsOneLeg.pt"]>50
    cut_cosmic = np.logical_and(cut_mask,cut_cosmic)
    cut_mask = (cut_cosmic.any()==True)
    cut_jet = arrays["CaloJets.nRecHitsEB"]>2
    cut_jet = np.logical_and(cut_mask,cut_jet)
    cut_mask = (cut_jet.any()==True)
    den += cut_mask[cut_mask].sum()
    no_cut_mask = np.logical_and(cut_mask,arrays["isCosmic"]<1)
    print arrays["RunNumber"][no_cut_mask]
    print arrays["LumiNumber"][no_cut_mask]
    print arrays["EventNumber"][no_cut_mask]
    print arrays["nCaloJets"][no_cut_mask]
    print arrays["CaloJets.nRecHitsEB"][no_cut_mask]
    cut_mask = np.logical_and(cut_mask,arrays["isCosmic"]>0)
    num += cut_mask[cut_mask].sum()

for i in range(int(den)):
    hd.Fill(0)

for i in range(int(den-num)):
    hn.Fill(0)

for i in range(int(num)):
    hn_pass.Fill(0)

e = TGraphAsymmErrors()
e.BayesDivide(hn,hd)
e_unc = e.GetErrorY(0)
print "e_unc: ", e_unc


e_pass = TGraphAsymmErrors()
e_pass.BayesDivide(hn_pass,hd)
e_pass_unc = e_pass.GetErrorY(0)

for i in range(int(obs)):
    ho.Fill(0)
poiss_o = ho.GetBinError(1)/ho.GetBinContent(1)
tot_unc = math.sqrt(e_unc**2 + poiss_o**2)

#2016BF
for i in range(int(obs_2016BF)):
    ho_2016BF.Fill(0)
poiss_o_2016BF = ho_2016BF.GetBinError(1)/ho_2016BF.GetBinContent(1)
tot_unc_2016BF = math.sqrt(e_unc**2 + poiss_o_2016BF**2)
y_2016BF = obs_2016BF*(1.-num/den)

#2016GH
for i in range(int(obs_2016GH)):
    ho_2016GH.Fill(0)
poiss_o_2016GH = ho_2016GH.GetBinError(1)/ho_2016GH.GetBinContent(1)
tot_unc_2016GH = math.sqrt(e_unc**2 + poiss_o_2016GH**2)
y_2016GH = obs_2016GH*(1.-num/den)

#2017
for i in range(int(obs_2017)):
    ho_2017.Fill(0)
poiss_o_2017 = ho_2017.GetBinError(1)/ho_2017.GetBinContent(1)
tot_unc_2017 = math.sqrt(e_unc**2 + poiss_o_2017**2)
y_2017 = obs_2017*(1.-num/den)

#2018
for i in range(int(obs_2018)):
    ho_2018.Fill(0)
poiss_o_2018 = ho_2018.GetBinError(1)/ho_2018.GetBinContent(1)
print "poiss_o_2018: ", poiss_o_2018
tot_unc_2018 = math.sqrt(e_unc**2 + poiss_o_2018**2)
y_2018 = obs_2018*(1.-num/den)

print "Num: ", num
print "Den: ", den
print "Cosmic veto efficiency: %.3f +- %.3f"%(num/den,e_pass_unc)
print "Cosmic veto miss efficiency: %.3f +- %.3f"%((den-num)/den,e_unc)
print "Cosmic prediction: %.3f +- %.3f"%(obs*(1.-num/den) , tot_unc*obs*(1.-num/den))
print "\n"
print "2016BF"
print "Cosmic events vetoed in SR: ",obs_2016BF
print "Cosmic prediction: %.3f +- %.3f"%(obs_2016BF*(1.-num/den) , tot_unc_2016BF*obs_2016BF*(1.-num/den))
print "2016GH"
print "Cosmic events vetoed in SR: ",obs_2016GH
print "Cosmic prediction: %.3f +- %.3f"%(obs_2016GH*(1.-num/den) , tot_unc_2016GH*obs_2016GH*(1.-num/den))
print "2017"
print "Cosmic events vetoed in SR: ",obs_2017
print "Cosmic prediction: %.3f +- %.3f"%(obs_2017*(1.-num/den) , tot_unc_2017*obs_2017*(1.-num/den))
print "2018"
print "Cosmic events vetoed in SR: ",obs_2018
print "Cosmic prediction: %.3f +- %.3f"%(obs_2018*(1.-num/den) , tot_unc_2018*obs_2018*(1.-num/den))

#prepare yaml
BASEDIR = "plots/Yields_AN_fix/v6_calo_AOD_%s_SR/"
filename = "BkgPredResults_%s_SR_cosmic%s"
#2016BF
results = {}
OUTDIR = BASEDIR%("2016")
name = filename%("2016","_B-F")
results["bkg_cosmic"] = y_2016BF
results["unc_cosmic"] = tot_unc_2016BF
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2016GH
results = {}
OUTDIR = BASEDIR%("2016")
name = filename%("2016","_G-H")
results["bkg_cosmic"] = y_2016GH
results["unc_cosmic"] = tot_unc_2016GH
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2017
results = {}
OUTDIR = BASEDIR%("2017")
name = filename%("2017","")
results["bkg_cosmic"] = y_2017
results["unc_cosmic"] = tot_unc_2017
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2018
results = {}
OUTDIR = BASEDIR%("2018")
name = filename%("2018","")
results["bkg_cosmic"] = y_2018
results["unc_cosmic"] = tot_unc_2018
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

