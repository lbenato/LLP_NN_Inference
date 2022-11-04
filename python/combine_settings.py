#! /usr/bin/env python
import os, sys, getopt, multiprocessing
import copy, math, time
from array import array
from ROOT import gROOT, gSystem, gStyle, gRandom, Double
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TGaxis, TSpline, TSpline3, TGraph, TGraphErrors, TProfile, TH1D, TLine, TF1, TFitResult, TFitResultPtr, TMultiGraph
from ROOT import TStyle, TCanvas, TPad, TLegend, TLatex, TText, TPaveText, TColor
import numpy as np
from collections import defaultdict

from ROOT import RooFit, RooRealVar, RooDataHist, RooDataSet, RooAbsData, RooAbsReal, RooAbsPdf, RooPlot, RooBinning, RooCategory, RooSimultaneous, RooArgList, RooArgSet, RooWorkspace, RooMsgService, RooFitResult
from ROOT import RooFormulaVar, RooGenericPdf, RooGaussian, RooExponential, RooPolynomial, RooChebychev, RooBreitWigner, RooCBShape, RooExtendPdf, RooAddPdf, RooProdPdf, RooNumConvPdf, RooFFTConvPdf


from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import *

import optparse
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-b', '--batch', action='store_true', default=False, dest='batch')
parser.add_option('-B', '--blind', action='store_true', default=True, dest='blind')
parser.add_option('-c', '--channel', action='store', type='string', dest='channel', default='')
parser.add_option('-t', '--tagvar', action='store', type='string', dest='tagvar', default='')
parser.add_option('-F', '--fom', action='store', type='string', dest='fom', default='')
parser.add_option('-k', '--key', action='store', type='string', dest='key', default='')
parser.add_option('-a', '--abcd', action='store_true', default=False, dest='abcd')
parser.add_option('-v', '--verbose', action='store_true', default=False, dest='verbose')
(options, args) = parser.parse_args()

## Directories

#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_AK4_v2_LUMI_JJ/"
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_v3/"
#LUMI = 59.74*1000
#MAIN_OUT = "v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_AK4_v2_LUMI_JJ"
#MAIN_OUT = "v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_v3"

##Comparing Princeton with v2 tagger  and mock background:
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_tf_and_skim_v5/"
#MAIN_OUT = "v5_calo_AOD_2018_tf_and_skim_v5_MOCK"
#LUMI = 95900.0

##Comparing Princeton with v3 tagger:
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_tagger_v3_Princeton_lumi/"
#MAIN_OUT = "v4_calo_AOD_2018_tf_and_skim_tagger_v3_Princeton_lumi"
#LUMI = 95900.0

##AK4 v3, AK8 v2:
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"
MAIN_OUT = "v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2"
LUMI = 95900.0

##compare CSC
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_compare_CSC/"
MAIN_OUT =  "v4_calo_AOD_2018_compare_CSC"
LUMI = 137000

##Comparing miniAOD:
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_AK4_miniAOD_v3_LUMI_JJ/"
#MAIN_OUT = "v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_AK4_miniAOD_v3_LUMI_JJ"
#LUMI = 59.74*1000

if options.channel=="ggHeavyHiggs":
    #print "HeavyHiggs"
    #chan = "ggH"
    CHAN     = "ggHeavyHiggs"
elif options.channel=="SUSY":
    #print "HeavyHiggs"
    #chan = "ggH"
    CHAN     = "SUSY"
elif options.channel=="JetJet":
    #print "HeavyHiggs"
    #chan = "ggH"
    CHAN     = "JetJet"
elif options.channel=="splitSUSY":
    #print "HeavyHiggs"
    #chan = "ggH"
    CHAN     = "splitSUSY"
elif options.channel=="gluinoGMSB":
    #print "HeavyHiggs"
    #chan = "ggH"
    CHAN     = "gluinoGMSB"
else:
    print "Channel not recognized, aborting . . ."
    exit()





OUTPUTDIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/"
if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)

DATACARDDIR = OUTPUTDIR+"datacards/"
if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
DATACARDDIR+=CHAN+"/"
if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
DATACARDS = DATACARDDIR + options.key+"/"
if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)

RESULTDIR = OUTPUTDIR+"combine_results/"
if not os.path.isdir(RESULTDIR): os.mkdir(RESULTDIR)
RESULTDIR+=CHAN+"/"
if not os.path.isdir(RESULTDIR): os.mkdir(RESULTDIR)
RESULTS = RESULTDIR + options.key+"/"
if not os.path.isdir(RESULTS): os.mkdir(RESULTS)

## Signal specific
SIGMULTFAC = 1
ENHANCEMENT = 1
SIGNAL_SUPPRESSION_FACTOR = 1000 if (CHAN=="ggHeavyHiggs" or CHAN=="JetJet" or CHAN=="splitSUSY" or CHAN=="gluinoGMSB") else 1
WEIGHT = "EventWeight"
MH = {
    2000 : [250, 600],
    1500 : [200, 500],
    1000 : [150, 400],
    600 : [50,150],
    #400 : [50,100],

    #200 : [25,50],
    #2000 : [50],
    #1500 : [50],
    #1000 : [50],
    #600 : [150],
    #400 : [100],
    #200 : [50],
}
ctauPoint = [1000]
MChi = [127,150,200,300,400]

ctauPoints_HH = [500, 1000, 2000, 5000, 10000]
#ctauPoints_HH = [10, 100, 500, 1000, 2000, 5000, 10000]
#ctauPoints_HH = [1000, 10000]
ctauPoints_XX = [100,300,1000,3000,50000]
ctauPoints_split = [300,1000,10000,30000]
ctauPoints_gluino = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 50000]
ctauPoints = []

if CHAN=="ggHeavyHiggs":
    ctauPoints = ctauPoints_HH
elif CHAN=="JetJet":
    ctauPoints = ctauPoints_XX
    #ctauPoints_XX = [100,300,1000,50000]
elif CHAN=="splitSUSY":
    ctauPoints = ctauPoints_split
elif CHAN=="gluinoGMSB":
    ctauPoints = ctauPoints_gluino

MX = [100, 300, 1000]
Mglu = [2400]
## Background
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
back  = []

## Root initialization
gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)

h = {}
tree = {}
c2 = {}

n_D = {}
error_D = {}



#we want to write the datacards based on a cut and a tagging variable
analysis ={
    "AK4_0p994" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994 ",
        "variable" : "isTagAK4_0p994",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    "AK4_0p994_JJ" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ ",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    "AK4_0p995_JJ" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p995_JJ ",
        "variable" : "isTagAK4_0p995_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p996_JJ" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p996_JJ ",
        "variable" : "isTagAK4_0p996_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    "AK4_0p997_miniAOD" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p997_JJ ",
        "variable" : "isTagAK4_0p997_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    "AK4_0p99_JJ" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p99_JJ ",
        "variable" : "isTagAK4_0p99_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p99" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && nTagJets_0p99>1 ",
        "variable" : "isMC",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    #orthogonal
    "AK4_0p994_JJ_no_170" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && nCHSFatJetsAcceptanceCalo==0",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p994_JJ_no_170_0p9999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && !isTagAK8_0p9999_170",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p994_JJ_no_170_0p99999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && !isTagAK8_0p99999_170",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p994_JJ_no_250" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && (nCHSFatJetsAcceptanceCalo==0 || FatJets[0].pt<250)",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p994_JJ_no_250_0p9999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && !isTagAK8_0p9999_250",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },
    "AK4_0p994_JJ_no_250_0p99999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p994_JJ && !isTagAK8_0p99999_250",
        "variable" : "isTagAK4_0p994_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    "AK4_0p996_JJ_no_250_0p99999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p996_JJ && !isTagAK8_0p99999_250",
        "variable" : "isTagAK4_0p996_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },


    "AK4_0p99_JJ_no_250_0p99999" :
    {
        "name" : "tryAK4",
        "cut": selection["PreselSkimAK4JJ"] + " && isTagAK4_0p99_JJ && !isTagAK8_0p99999_250",
        "variable" : "isTagAK4_0p99_JJ",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK4",
    },

    #combinations
    "AK4_0p994_JJ_AK8_170" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK4_0p994_JJ_AK8_170_0p9999" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK4_0p994_JJ_AK8_170_0p99999" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK4_0p994_JJ_AK8_250" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK4_0p994_JJ_AK8_250_0p9999" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK4_0p994_JJ_AK8_250_0p99999" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },


    "AK4_0p99_JJ_AK8_250_0p99999" :
    {
        "name" : "comb",
        "cut": "",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },


    #AK8
    "AK8_0p9999_200_min_jet_met" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_200 && MinFatJetMetDPhiBarrel>0.5 ",
        "variable" : "isTagAK8_0p9999_200",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p9999_250_min_jet_met" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_250 && MinFatJetMetDPhiBarrel>0.5",
        "variable" : "isTagAK8_0p9999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_200_min_jet_met" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_200 && MinFatJetMetDPhiBarrel>0.5 ",
        "variable" : "isTagAK8_0p99999_200",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_250_min_jet_met" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_250 && MinFatJetMetDPhiBarrel>0.5",
        "variable" : "isTagAK8_0p99999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },



    "AK8_0p9999_170" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_170 ",
        "variable" : "isTagAK8_0p9999_170",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p9999_200" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_200 ",
        "variable" : "isTagAK8_0p9999_200",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p9999_250" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_250 ",
        "variable" : "isTagAK8_0p9999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p9999_300" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_300 ",
        "variable" : "isTagAK8_0p9999_300",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p9999_350" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_350 ",
        "variable" : "isTagAK8_0p9999_350",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },    

    "AK8_0p99999_170" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_170 ",
        "variable" : "isTagAK8_0p99999_170",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p9_double" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && nTagFatJets_0p9>1 ",
        "variable" : "isMC",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p99_double" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && nTagFatJets_0p99>1 ",
        "variable" : "isMC",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p999_double" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && nTagFatJets_0p999>1 ",
        "variable" : "isMC",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p9999_double" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && nTagFatJets_0p9999>1 ",
        "variable" : "isMC",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },


    "AK8_0p99999_200" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_200 ",
        "variable" : "isTagAK8_0p99999_200",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_250" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_250 ",
        "variable" : "isTagAK8_0p99999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p999999_250" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p999999_250 ",
        "variable" : "isTagAK8_0p999999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_300" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_300 ",
        "variable" : "isTagAK8_0p99999_300",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p99999_350" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_350 ",
        "variable" : "isTagAK8_0p99999_350",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },    

    #Orthogonal
    "AK8_0p9999_170_no_AK4" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_170 && !isTagAK4_0p994_JJ",
        "variable" : "isTagAK8_0p9999_170",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p9999_250_no_AK4" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p9999_250  && !isTagAK4_0p994_JJ",
        "variable" : "isTagAK8_0p9999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p99999_170_no_AK4" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_170 && !isTagAK4_0p994_JJ",
        "variable" : "isTagAK8_0p99999_170",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },
    "AK8_0p99999_250_no_AK4" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_250  && !isTagAK4_0p994_JJ",
        "variable" : "isTagAK8_0p99999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_250_no_AK4_0p99" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_250  && !isTagAK4_0p99_JJ",
        "variable" : "isTagAK8_0p99999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },

    "AK8_0p99999_250_no_AK4_0p996" :
    {
        "name" : "tryAK8",
        "cut": selection["PreselSkimAK8JJ"] + " && isTagAK8_0p99999_250  && !isTagAK4_0p996_JJ",
        "variable" : "isTagAK8_0p99999_250",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "AK8",
    },


    #combinations
    "AK8_170_0p9999_AK4_0p994" :
    {
        "name" : "comb",
        "cut": "",
        "description" : "If 2 tagged AK4 fails, then look for an AK8 with pT X and DNN output Y",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK8_170_0p99999_AK4_0p994" :
    {
        "name" : "comb",
        "cut": "",
        "description" : "If 2 tagged AK4 fails, then look for an AK8 with pT X and DNN output Y",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK8_250_0p9999_AK4_0p994" :
    {
        "name" : "comb",
        "cut": "",
        "description" : "If 2 tagged AK4 fails, then look for an AK8 with pT X and DNN output Y",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },
    "AK8_250_0p99999_AK4_0p994" :
    {
        "name" : "comb",
        "cut": "",
        "description" : "If 2 tagged AK4 fails, then look for an AK8 with pT X and DNN output Y",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },

    "AK8_250_0p99999_AK4_0p996" :
    {
        "name" : "comb",
        "cut": "",
        "description" : "If 2 tagged AK4 fails, then look for an AK8 with pT X and DNN output Y",
        "variable" : "",
        "binmin" : 0,
        "binmax" : 2,
        "nbins" : 2,
        "channel" : "comb",
    },

}


if not(options.key in analysis.keys()):
    print "Key not recognized, aborting . . ."
    exit()

def write_datacard(an_dict_key,mH=MH,mChi=MChi,tree_name="tree",cut="",draw_all=False,add_string="",binmin=0,binmax=2,nbins=2,title=''):
    signalMultFactor = SIGMULTFAC * ENHANCEMENT
    tagVar    = analysis[an_dict_key]["variable"]
    channel = analysis[an_dict_key]["channel"]


    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    print tagVar

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    print "    --- Cut: ", analysis[an_dict_key]["cut"]

    for b in back:
        print "\n"
        tree[b] = TChain(tree_name)

        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(NTUPLEDIR + ss + '.root'):
                print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            tree[b].Add(NTUPLEDIR + ss + '.root')
        print '  %s tree entries: %.2f' % (b,tree[b].GetEntries())
        print '  %s tree entries in SR: %.2f' % (b,tree[b].GetEntries(analysis[an_dict_key]["cut"]))

        h[b] = TH1F(str(b),str(b),nbins,binmin,binmax)
        h[b].Sumw2()
        tree[b].Project(str(b),analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+WEIGHT)
        error_D[b] = Double()
        n_D[b]   = RooRealVar('n_D_'+str(b),  str(b)+' D normalization',   h[b].IntegralAndError(h[b].GetXaxis().FindBin(binmin), h[b].GetXaxis().FindBin(binmax),error_D[b],""),   0., 1.e50)
        print "  "+str(b)+ " in SR: %.2e +- %.2e" % (n_D[b].getVal(), error_D[b])


    treeSignal = {}
    hSignal = {}
    hSignal_profX = {}
    nSignalSR = {}
    SignalYieldSR = {}
    SignalYieldErrSR = {}
    cSign = {}

    if back==[]:
        for b in ['ZJetsToNuNu','WJetsToLNu','VV','TTbar','QCD','VV']:
            n_D[b] = RooRealVar('n_D_'+str(b),  str(b)+' D normalization', 0.,   0., 1.e50)

    if CHAN=="SUSY" or CHAN=="JetJet" or CHAN=="splitSUSY" or CHAN=="gluinoGMSB":
        print CHAN
        if CHAN=="SUSY":
            m_part = mChi
            ctauPoints = ctauPoint
        elif CHAN=="JetJet":
            m_part = MX
            ctauPoints = ctauPoints_XX
        elif CHAN=="splitSUSY":
            m_part = Mglu
            ctauPoints = ctauPoints_split
        elif CHAN=="gluinoGMSB":
            m_part = Mglu
            ctauPoints = ctauPoints_gluino
        massPoints = []
        for m in m_part:
            massPoints.append(m)
            for c in ctauPoints:
                if CHAN=="SUSY":
                    #name = "SUSY_mh"+str(m)+"_pl"+str(c)
                    if m==200:
                        name = "SUSY_mh"+str(m)+"_pl"+str(c)
                    elif m==400:
                        name = "SUSY_mh"+str(m)+"_pl"+str(c)+"_XL"
                elif CHAN=="JetJet":
                    name = "XXTo4J_M"+str(m)+"_CTau"+str(c)+"mm"
                elif CHAN=="splitSUSY":
                    name = "splitSUSY_M-"+str(m)+"_CTau-"+str(c)+"mm"
                elif CHAN=="gluinoGMSB":
                    name = "gluinoGMSB_M"+str(m)+"_ctau"+str(c)
                print name

                treeSignal[name] = TChain(tree_name)
                for j, ss in enumerate(samples[name]['files']):
                    treeSignal[name].Add(NTUPLEDIR + ss + '.root')
                    #print name, treeSignal[name].GetEntries()
                hSignal[name] = TH1F(name,name,nbins,binmin,binmax)
                hSignal[name].Sumw2()
                #print "Weight w/o PU!!"
                #w = "(EventWeight/PUWeight)"
                treeSignal[name].Project(name,analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+WEIGHT+"*"+str(1./SIGNAL_SUPPRESSION_FACTOR))
                #treeSignal[name].Project(name,analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+w+"*"+str(1./SIGNAL_SUPPRESSION_FACTOR))

                SignalYieldErrSR[name]  = Double()
                SignalYieldSR[name]     = RooRealVar('SignalYield'+name,  'Signal ' + name + ' normalization',   hSignal[name].IntegralAndError(hSignal[name].GetXaxis().FindBin(binmin), hSignal[name].GetXaxis().FindBin(binmax),SignalYieldErrSR[name],""),   0., 1.e50)
                nSignalSR[name] = SignalYieldSR[name].getVal()#*signalMultFactor
                print name + " in SR: %.1f +- %.1f" % (nSignalSR[name], SignalYieldErrSR[name])

                if nSignalSR[name]==0:
                    print "No signal left for sample: ", name
                    #print "Skip to next sample!"
                    #continue

                #*******************************************************#
                #                                                       #
                #                      Datacard                         #
                #                                                       #
                #*******************************************************#

                card  = 'imax 1\n'#n of bins
                card += 'jmax *\n'#n of backgrounds
                card += 'kmax *\n'#n of nuisance parmeters
                card += '-----------------------------------------------------------------------------------\n'
                card += 'bin               %s\n' % channel
                card += 'observation       %s\n' % '-1.0'
                card += '-----------------------------------------------------------------------------------\n'

                card += 'bin                                     %-33s%-33s%-33s%-33s%-33s%-33s\n' % (channel, channel, channel, channel, channel, channel)
                #card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name+'_'+channel, 'ZJetsToNuNu_'+channel, 'WJetsToLNu_'+channel, 'TTbar_'+channel, 'QCD_'+channel, 'VV_'+channel, )
                card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name, 'ZJetsToNuNu', 'WJetsToLNu', 'TTbar', 'QCD', 'VV', )
                card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % ('0', '1', '2', '3', '4', '5',)
                card += 'rate                                    %-23f%-23f%-23f%-23f%-23f%-23f\n' % (nSignalSR[name]*signalMultFactor, n_D['ZJetsToNuNu'].getVal(), n_D['WJetsToLNu'].getVal(), n_D['TTbar'].getVal(), n_D['QCD'].getVal(), n_D['VV'].getVal())

                card += '-----------------------------------------------------------------------------------\n'
                #Syst uncertainties
                #luminosity
                card += '%-25s     lnN       %-23f%-23s%-23s%-23s%-23s%-23s\n' % ('sig_norm', 1.+SignalYieldErrSR[name]/nSignalSR[name] if nSignalSR[name]>0 else 0, '-', '-', '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23f%-23s%-23s%-23s%-23s\n' % ('ZJetsToNuNu_norm' , '-', 1. + error_D['ZJetsToNuNu']/n_D['ZJetsToNuNu'].getVal() if n_D['ZJetsToNuNu'].getVal()>0 else 1., '-', '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23f%-23s%-23s%-23s\n' % ('WJetsToLNu_norm' ,  '-', '-', 1. + error_D['WJetsToLNu']/n_D['WJetsToLNu'].getVal() if n_D['WJetsToLNu'].getVal()>0 else 1., '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23f%-23s%-23s\n' % ('TTbar_norm' ,    '-', '-', '-', 1. + error_D['TTbar']/n_D['TTbar'].getVal() if n_D['TTbar'].getVal()>0 else 1., '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23f%-23s\n' % ('QCD_norm' ,    '-', '-', '-', '-', 1. + error_D['QCD']/n_D['QCD'].getVal() if n_D['QCD'].getVal()>0 else 1., '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23s%-23f\n' % ('VV_norm' ,     '-', '-', '-', '-', '-', 1. + error_D['VV']/n_D['VV'].getVal() if n_D['VV'].getVal()>0 else 1.)

                card += '%-25s     lnN       %-23f%-23f%-23f%-23f%-23f%-23f\n' % ('lumi_13TeV', 1.025, 1.025, 1.025, 1.025, 1.025, 1.025)
                outname = DATACARDS+ name + '.txt'
                cardfile = open(outname, 'w')
                cardfile.write(card)
                cardfile.close()
                print card
                sb_yield  = '{ "S" : '+str(nSignalSR[name])+', "B" : '+str(n_D['ZJetsToNuNu'].getVal() + n_D['WJetsToLNu'].getVal() + n_D['TTbar'].getVal() + n_D['QCD'].getVal() + n_D['VV'].getVal())+'}'
                print sb_yield
                outyield = DATACARDS+ name + '_yield.txt'
                yieldfile = open(outyield, 'w')
                yieldfile.write(sb_yield)
                yieldfile.close()
                
                print 'Datacard for mass', m, 'in channel', channel, 'saved in', outname
                print 'S/B yields saved in', outyield

    elif CHAN=="ggHeavyHiggs":
        ctauPoints =  ctauPoints_HH
        for H in mH.keys():
            print H, mH[H]
            massPoints = mH[H]
            for m in massPoints:
                for c in ctauPoints:
                    name = "ggH_MH"+str(H)+"_MS"+str(m)+"_ctau"+str(c)
                    print name

                    treeSignal[name] = TChain(tree_name)
                    for j, ss in enumerate(samples[name]['files']):
                        treeSignal[name].Add(NTUPLEDIR + ss + '.root')
                        #print name, treeSignal[name].GetEntries()
                    hSignal[name] = TH1F(name,name,nbins,binmin,binmax)
                    hSignal[name].Sumw2()
                    treeSignal[name].Project(name,analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+WEIGHT+"*"+str(1./SIGNAL_SUPPRESSION_FACTOR))

                    SignalYieldErrSR[name]  = Double()
                    SignalYieldSR[name]     = RooRealVar('SignalYield'+name,  'Signal ' + name + ' normalization',   hSignal[name].IntegralAndError(hSignal[name].GetXaxis().FindBin(binmin), hSignal[name].GetXaxis().FindBin(binmax),SignalYieldErrSR[name],""),   0., 1.e50)
                    nSignalSR[name] = SignalYieldSR[name].getVal()#*signalMultFactor
                    print name + " in SR: %.1f +- %.1f" % (nSignalSR[name], SignalYieldErrSR[name])

                    if nSignalSR[name]==0:
                        print "No signal left for sample: ", name
                        print "Skip to next sample!"
                        continue

                    #*******************************************************#
                    #                                                       #
                    #                      Datacard                         #
                    #                                                       #
                    #*******************************************************#

                    card  = 'imax 1\n'#n of bins
                    card += 'jmax *\n'#n of backgrounds
                    card += 'kmax *\n'#n of nuisance parmeters
                    card += '-----------------------------------------------------------------------------------\n'
                    card += 'bin               %s\n' % channel
                    card += 'observation       %s\n' % '-1.0'
                    card += '-----------------------------------------------------------------------------------\n'

                    card += 'bin                                     %-33s%-33s%-33s%-33s%-33s%-33s\n' % (channel, channel, channel, channel, channel, channel)
                    #card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name+'_'+channel, 'ZJetsToNuNu_'+channel, 'WJetsToLNu_'+channel, 'TTbar_'+channel, 'QCD_'+channel, 'VV_'+channel, )
                    card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name, 'ZJetsToNuNu', 'WJetsToLNu', 'TTbar', 'QCD', 'VV', )
                    card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % ('0', '1', '2', '3', '4', '5',)
                    card += 'rate                                    %-23f%-23f%-23f%-23f%-23f%-23f\n' % (nSignalSR[name]*signalMultFactor, n_D['ZJetsToNuNu'].getVal(), n_D['WJetsToLNu'].getVal(), n_D['TTbar'].getVal(), n_D['QCD'].getVal(), n_D['VV'].getVal())

                    card += '-----------------------------------------------------------------------------------\n'
                    #Syst uncertainties
                    #luminosity
                    card += '%-25s     lnN       %-23f%-23s%-23s%-23s%-23s%-23s\n' % ('sig_norm', 1.+SignalYieldErrSR[name]/nSignalSR[name], '-', '-', '-', '-', '-')
                    card += '%-25s     lnN       %-23s%-23f%-23s%-23s%-23s%-23s\n' % ('ZJetsToNuNu_norm' , '-', 1. + error_D['ZJetsToNuNu']/n_D['ZJetsToNuNu'].getVal() if n_D['ZJetsToNuNu'].getVal()>0 else 1., '-', '-', '-', '-')
                    card += '%-25s     lnN       %-23s%-23s%-23f%-23s%-23s%-23s\n' % ('WJetsToLNu_norm' ,  '-', '-', 1. + error_D['WJetsToLNu']/n_D['WJetsToLNu'].getVal() if n_D['WJetsToLNu'].getVal()>0 else 1., '-', '-', '-')
                    card += '%-25s     lnN       %-23s%-23s%-23s%-23f%-23s%-23s\n' % ('TTbar_norm' ,    '-', '-', '-', 1. + error_D['TTbar']/n_D['TTbar'].getVal() if n_D['TTbar'].getVal()>0 else 1., '-', '-')
                    card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23f%-23s\n' % ('QCD_norm' ,    '-', '-', '-', '-', 1. + error_D['QCD']/n_D['QCD'].getVal() if n_D['QCD'].getVal()>0 else 1., '-')
                    card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23s%-23f\n' % ('VV_norm' ,     '-', '-', '-', '-', '-', 1. + error_D['VV']/n_D['VV'].getVal() if n_D['VV'].getVal()>0 else 1.)

                    card += '%-25s     lnN       %-23f%-23f%-23f%-23f%-23f%-23f\n' % ('lumi_13TeV', 1.025, 1.025, 1.025, 1.025, 1.025, 1.025)
                    outname = DATACARDS+ name + '.txt'
                    cardfile = open(outname, 'w')
                    cardfile.write(card)
                    cardfile.close()
                    print card
                    sb_yield  = '{ "S" : '+str(nSignalSR[name])+', "B" : '+str(n_D['ZJetsToNuNu'].getVal() + n_D['WJetsToLNu'].getVal() + n_D['TTbar'].getVal() + n_D['QCD'].getVal() + n_D['VV'].getVal())+'}'
                    print sb_yield
                    outyield = DATACARDS+ name + '_yield.txt'
                    yieldfile = open(outyield, 'w')
                    yieldfile.write(sb_yield)
                    yieldfile.close()


                
                print 'Datacard for mass', m, 'and ctau', c,  'in channel', channel, 'saved in', outname
                print 'S/B yields saved in', outyield



def write_mock_datacard(an_dict_key,mH=MH,mChi=MChi,tree_name="tree",cut="",draw_all=False,add_string="",binmin=0,binmax=2,nbins=2,title=''):
    signalMultFactor = SIGMULTFAC * ENHANCEMENT
    tagVar    = analysis[an_dict_key]["variable"]
    channel = analysis[an_dict_key]["channel"]


    if not os.path.isdir(DATACARDS): os.mkdir(DATACARDS)
    print tagVar

    if signalMultFactor == 0.001:
        print '  x-sec calculated in fb '
    elif (signalMultFactor >= 1 and signalMultFactor < 1000):
        print '  x-sec calculated in pb '
    else:
        print 'Wrong signal mult. factor, aborting...'
        exit()
    print '-'*11*2

    print "    --- Cut: ", analysis[an_dict_key]["cut"]

    for b in back:
        print "\n"
        #tree[b] = TChain(tree_name)

        #for i, ss in enumerate(samples[b]['files']):
        #    if not os.path.isfile(NTUPLEDIR + ss + '.root'):
        #        print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
        #        continue
        #    tree[b].Add(NTUPLEDIR + ss + '.root')
        #print '  %s tree entries: %.2f' % (b,tree[b].GetEntries())
        #print '  %s tree entries in SR: %.2f' % (b,tree[b].GetEntries(analysis[an_dict_key]["cut"]))

        #h[b] = TH1F(str(b),str(b),nbins,binmin,binmax)
        #h[b].Sumw2()
        #tree[b].Project(str(b),analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+WEIGHT)
        #n_D[b]   = RooRealVar('n_D_'+str(b),  str(b)+' D normalization',   h[b].IntegralAndError(h[b].GetXaxis().FindBin(binmin), h[b].GetXaxis().FindBin(binmax),error_D[b],""),   0., 1.e50)
        if "ZJetsToNuNu" in b:
            integral = 3.
        else:
            integral = 0.
        err =  0.5*integral
        error_D[b] = err
        n_D[b] = RooRealVar('n_D_'+str(b),  str(b)+' D normalization',  integral,   0., 1.e50) 
        print "  "+str(b)+ " in SR: %.2e +- %.2e" % (n_D[b].getVal(), error_D[b])


    treeSignal = {}
    hSignal = {}
    hSignal_profX = {}
    nSignalSR = {}
    SignalYieldSR = {}
    SignalYieldErrSR = {}
    cSign = {}


    if CHAN=="SUSY" or CHAN=="JetJet" or CHAN=="splitSUSY" or CHAN=="gluinoGMSB":
        print CHAN
        if CHAN=="SUSY":
            m_part = mChi
            ctauPoints = ctauPoint
        elif CHAN=="JetJet":
            m_part = MX
            ctauPoints = ctauPoints_XX
        elif CHAN=="splitSUSY":
            m_part = Mglu
            ctauPoints = ctauPoints_split
        elif CHAN=="gluinoGMSB":
            m_part = Mglu
            ctauPoints = ctauPoints_gluino
        massPoints = []
        for m in m_part:
            massPoints.append(m)
            for c in ctauPoints:
                if CHAN=="SUSY":
                    if m==200:
                        name = "SUSY_mh"+str(m)+"_pl"+str(c)
                    elif m==400:
                        name = "SUSY_mh"+str(m)+"_pl"+str(c)+"_XL"
                elif CHAN=="JetJet":
                    name = "XXTo4J_M"+str(m)+"_CTau"+str(c)+"mm"
                elif CHAN=="splitSUSY":
                    name = "splitSUSY_M-"+str(m)+"_CTau-"+str(c)+"mm"
                elif CHAN=="gluinoGMSB":
                    name = "gluinoGMSB_M"+str(m)+"_ctau"+str(c)
                print name

                treeSignal[name] = TChain(tree_name)
                for j, ss in enumerate(samples[name]['files']):
                    treeSignal[name].Add(NTUPLEDIR + ss + '.root')
                    #print name, treeSignal[name].GetEntries()
                hSignal[name] = TH1F(name,name,nbins,binmin,binmax)
                hSignal[name].Sumw2()
                treeSignal[name].Project(name,analysis[an_dict_key]["variable"],"("+analysis[an_dict_key]["cut"]+")*"+WEIGHT+"*"+str(1./SIGNAL_SUPPRESSION_FACTOR))

                SignalYieldErrSR[name]  = Double()
                SignalYieldSR[name]     = RooRealVar('SignalYield'+name,  'Signal ' + name + ' normalization',   hSignal[name].IntegralAndError(hSignal[name].GetXaxis().FindBin(binmin), hSignal[name].GetXaxis().FindBin(binmax),SignalYieldErrSR[name],""),   0., 1.e50)
                nSignalSR[name] = SignalYieldSR[name].getVal()#*signalMultFactor
                print name + " in SR: %.1f +- %.1f" % (nSignalSR[name], SignalYieldErrSR[name])

                if nSignalSR[name]==0:
                    print "No signal left for sample: ", name
                    #print "Skip to next sample!"
                    #continue

                #*******************************************************#
                #                                                       #
                #                      Datacard                         #
                #                                                       #
                #*******************************************************#

                card  = 'imax 1\n'#n of bins
                card += 'jmax *\n'#n of backgrounds
                card += 'kmax *\n'#n of nuisance parmeters
                card += '-----------------------------------------------------------------------------------\n'
                card += 'bin               %s\n' % channel
                card += 'observation       %s\n' % '-1.0'
                card += '-----------------------------------------------------------------------------------\n'

                card += 'bin                                     %-33s%-33s%-33s%-33s%-33s%-33s\n' % (channel, channel, channel, channel, channel, channel)
                #card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name+'_'+channel, 'ZJetsToNuNu_'+channel, 'WJetsToLNu_'+channel, 'TTbar_'+channel, 'QCD_'+channel, 'VV_'+channel, )
                card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % (name, 'ZJetsToNuNu', 'WJetsToLNu', 'TTbar', 'QCD', 'VV', )
                card += 'process                                 %-33s%-33s%-33s%-33s%-33s%-33s\n' % ('0', '1', '2', '3', '4', '5',)
                card += 'rate                                    %-23f%-23f%-23f%-23f%-23f%-23f\n' % (nSignalSR[name]*signalMultFactor, n_D['ZJetsToNuNu'].getVal(), n_D['WJetsToLNu'].getVal(), n_D['TTbar'].getVal(), n_D['QCD'].getVal(), n_D['VV'].getVal())

                card += '-----------------------------------------------------------------------------------\n'
                #Syst uncertainties
                #luminosity
                card += '%-25s     lnN       %-23f%-23s%-23s%-23s%-23s%-23s\n' % ('sig_norm', 1.+SignalYieldErrSR[name]/nSignalSR[name] if nSignalSR[name]>0 else 0, '-', '-', '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23f%-23s%-23s%-23s%-23s\n' % ('ZJetsToNuNu_norm' , '-', 1. + error_D['ZJetsToNuNu']/n_D['ZJetsToNuNu'].getVal() if n_D['ZJetsToNuNu'].getVal()>0 else 1., '-', '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23f%-23s%-23s%-23s\n' % ('WJetsToLNu_norm' ,  '-', '-', 1. + error_D['WJetsToLNu']/n_D['WJetsToLNu'].getVal() if n_D['WJetsToLNu'].getVal()>0 else 1., '-', '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23f%-23s%-23s\n' % ('TTbar_norm' ,    '-', '-', '-', 1. + error_D['TTbar']/n_D['TTbar'].getVal() if n_D['TTbar'].getVal()>0 else 1., '-', '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23f%-23s\n' % ('QCD_norm' ,    '-', '-', '-', '-', 1. + error_D['QCD']/n_D['QCD'].getVal() if n_D['QCD'].getVal()>0 else 1., '-')
                card += '%-25s     lnN       %-23s%-23s%-23s%-23s%-23s%-23f\n' % ('VV_norm' ,     '-', '-', '-', '-', '-', 1. + error_D['VV']/n_D['VV'].getVal() if n_D['VV'].getVal()>0 else 1.)

                card += '%-25s     lnN       %-23f%-23f%-23f%-23f%-23f%-23f\n' % ('lumi_13TeV', 1.025, 1.025, 1.025, 1.025, 1.025, 1.025)
                outname = DATACARDS+ name + '.txt'
                cardfile = open(outname, 'w')
                cardfile.write(card)
                cardfile.close()
                print card
                sb_yield  = '{ "S" : '+str(nSignalSR[name])+', "B" : '+str(n_D['ZJetsToNuNu'].getVal() + n_D['WJetsToLNu'].getVal() + n_D['TTbar'].getVal() + n_D['QCD'].getVal() + n_D['VV'].getVal())+'}'
                print sb_yield
                outyield = DATACARDS+ name + '_yield.txt'
                yieldfile = open(outyield, 'w')
                yieldfile.write(sb_yield)
                yieldfile.close()
                
                print 'Datacard for mass', m, 'in channel', channel, 'saved in', outname
                print 'S/B yields saved in', outyield

