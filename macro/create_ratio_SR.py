#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
import yaml

ERA = "2016"

inpfile = TFile("plots/v6_calo_AOD_"+ERA+"_SR_chris_hill/SR_boost/MEt_pt_calo.root","READ")
inpfile.cd()
data = inpfile.Get("HighMETGH" if ERA=="2016" else "HighMET")
back = inpfile.Get("All")
print "Loading input histograms..."
data.SetDirectory(0)
back.SetDirectory(0)
inpfile.Close()

outfile = TFile("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_chris_hill/data_MC_MEt_pt_ratio.root","RECREATE")
outfile.cd()
ratio = data.Clone("ratio")
ratio.Divide(back)
print "Performing ratio..."
ratio.Write("ratio")
outfile.Close()
print "Written /nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_chris_hill/data_MC_MEt_pt_ratio.root"

