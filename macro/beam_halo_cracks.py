#! /usr/bin/env python
import os, multiprocessing
import copy
import math
import uproot
import yaml
import numpy as np
import root_numpy
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

ERA="2016"
data = ["HighMET"]
INDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_no_BeamHalo_filter/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/beam_halo_cracks_"+ERA+"/"
if not os.path.isdir(OUT): os.mkdir(OUT)

LUMI = 0
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples as samples2018
from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
LUMI  += lumi2018[ data[0] ]["tot"]

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples as samples2017
from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
LUMI  += lumi2017[ data[0] ]["tot"]

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples as samples2016
from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016
LUMI  += lumi2016[ data[0] ]["tot"]

list_files = []
for ss in samples2016["HighMET"]['files']:
    list_files.append( (INDIR%"2016")+ss+'.root')
for ss in samples2017["HighMET"]['files']:
    list_files.append( (INDIR%"2017")+ss+'.root')
for ss in samples2018["HighMET"]['files']:
    list_files.append( (INDIR%"2018")+ss+'.root')


print list_files

hd = TH1F("","hd",1,-0.5,0.5)
hn = TH1F("","hn",1,-0.5,0.5)
hd.Sumw2()
hn.Sumw2()

den = 0.
num = 0.
num_csc = 0.
num_both = 0.

#EventNumberAll = np.array([])

tot_size = 0

chunk_size = 100000
list_of_variables = ["EventNumber","isSR","MinJetMetDPhi","min_dPhi*","Flag2_globalSuperTightHalo2016Filter","Jets.sigprob","Jets.nRecHitsEB"]
gen = uproot.iterate(list_files,"tree",list_of_variables,chunk_size)
i=0
for arrays in gen:
    print "Event ", chunk_size*i, " - " , chunk_size*(i+1)
    i+=1
    #EventNumberAll = np.concatenate(( EventNumberAll, arrays["EventNumber"]))

    tot_size+= arrays["isSR"].shape[0]
    cut_mask = arrays["isSR"]>0
    #cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
    den += cut_mask[cut_mask].sum()

    cut_mask_dphi = np.logical_and(cut_mask,arrays["min_dPhi_jets_0p7"]<0.05)
    cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.7 , arrays["Jets.nRecHitsEB"]<=10)
    cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
    cut_mask_bh = np.logical_and(cut_mask_low_multi_tag.any()==True,cut_mask)

    num += cut_mask[cut_mask_bh].sum()


    cut_mask_csc = np.logical_and(cut_mask,arrays["Flag2_globalSuperTightHalo2016Filter"]<1)
    num_csc += cut_mask[cut_mask_csc].sum()
    num_both += cut_mask[np.logical_and(cut_mask_csc,cut_mask_bh)].sum()

    #if i>20: break

#print RunNumber
print "total n. events:",tot_size
print "n. events:",den
print "n. events passing bh selections:",num
print "ratio ", 100.*num/den
print "n. events passing csc selections:",num_csc
print "ratio ", 100.*num_csc/den

print "n. events passing both selections:",num_both
print "ratio ", 100.*num_both/den
print "n. events passing both selections over bh:",num_both
print "ratio ", 100.*num_both/num

for i in range(int(num)):
    hd.Fill(0)

for i in range(int(num_both)):
    hn.Fill(0)

e = TGraphAsymmErrors()
e.BayesDivide(hn,hd)
e_unc = e.GetErrorY(0)
print "e +- e_unc: ", e.GetPointY(0), " +- ", e_unc

