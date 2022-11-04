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

#from Analyzer.LLPonAOD.drawUtils import *
#from Analyzer.LLP.variables import *
#from Analyzer.LLP.samples import samples

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option('-b', '--batch', action='store_true', default=False, dest='batch')
parser.add_option('-B', '--blind', action='store_true', default=True, dest='blind')
parser.add_option('-c', '--channel', action='store', type='string', dest='channel', default='ggHeavyHiggs')
parser.add_option('-t', '--tagvar', action='store', type='string', dest='tagvar', default='')
parser.add_option('-F', '--fom', action='store', type='string', dest='fom', default='')
parser.add_option('-C', '--category', action='store', type='string', dest='category', default='')
parser.add_option('-a', '--abcd', action='store_true', default=False, dest='abcd')
parser.add_option('-v', '--verbose', action='store_true', default=False, dest='verbose')
(options, args) = parser.parse_args()
if options.batch: gROOT.SetBatch(True)

########## SETTINGS ##########

# Silent RooFit
RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL)

#gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)

# Allowed channels
channelList = ["BTagCSV", "DisplacedJets", "MET","ZH","ZHMM","ZHEE","VBF","VBFH","SUSY","ggHeavyHiggs"]
if options.channel not in channelList:
    print "Warning! Channel not recognized, aborting..."
    exit()

# Allowed discriminating variables
tagvarList = ["nTracks3PixelHits","alphaMax","DNN","nTrackConstituents","nSelectedTracks","photonEFrac","nHadEFrac","cHadEFrac","MET_nDTSegments","photonEFrac_nDTSegments","photonEFrac_nDisplacedStandAloneMuons","photonEFrac_Zpt","photonEFrac_Zmass","nTrackConstituents_nDTSegments","nDTSegments","DNN_nDTSegments","nTag_pfXWP1000","pfXWP1000","ORpfXWP1000","OR3pfXWP1000","OR4pfXWP1000","OR5pfXWP1000","ORpfXWP1000_09","one_pfXWP1000"]#as per skim ntuples

tagvarList += ["cHadEFrac_pfXWP1000_j0","cHadEFrac_pfXWP1000_j1","cHadEFrac_pfXWP1000_j2"]
tagvarList += ["cHadEFrac_nSelectedTracks_j0","cHadEFrac_nSelectedTracks_j1","cHadEFrac_nSelectedTracks_j2"]

if options.tagvar not in tagvarList:
    print "Warning! Tag var not recognized and not present in ntuples, aborting..."
    exit()

# Allowed FOM
fomList = ["Limits","Significance"]
if options.fom not in fomList:
    print "Warning! FOM not recognized, aborting..."
    exit()




NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v0_SUSY_calo_MINIAOD_2018/"
from Analyzer.LLP2018.samplesMINIAOD2018 import samples

LUMI        = 35867
PRELIMINARY = True
BLIND       = True

useABCD     = False

# # # # # # # # # # # # #
 # # # # # # # # # # # #
# # # # # # # # # # # # #
MAIN_OUT = "v0_ggHeavyHiggs_calo_MINIAOD_2018"
jet_pt_min = str(5)
jet_pt_cut = ""
is_array2root_tree = False
# # # # # # # # # # # # #
 # # # # # # # # # # # #
# # # # # # # # # # # # #

#combine_results directories:
OUTPUTDIR           = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/Analyzer/LLP2018/"+MAIN_OUT+"/"
if not os.path.isdir(OUTPUTDIR): os.mkdir(OUTPUTDIR)

OUTPUTDIR_CHAN      = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/Analyzer/LLP2018/"+MAIN_OUT+"/"+options.channel+"/"
if not os.path.isdir(OUTPUTDIR_CHAN): os.mkdir(OUTPUTDIR_CHAN)


COMBINE_RESULTS_DIR = OUTPUTDIR + "/combine_results/"
if not os.path.isdir(COMBINE_RESULTS_DIR): os.mkdir(COMBINE_RESULTS_DIR)
if not os.path.isdir(COMBINE_RESULTS_DIR+options.channel+"/"): os.mkdir(COMBINE_RESULTS_DIR+options.channel+"/")
if not os.path.isdir(COMBINE_RESULTS_DIR+options.channel+"/"+options.tagvar+"/"): os.mkdir(COMBINE_RESULTS_DIR+options.channel+"/"+options.tagvar+"/")

PLOTDIR             = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/Analyzer/LLP2018/"+MAIN_OUT+"/combine_results/"+options.channel+"/"+options.tagvar+"/"+options.fom+"/"
if not os.path.isdir(PLOTDIR): os.mkdir(PLOTDIR)

#datacards directories in HiggsCombine 
DATACARDDIR_BASE = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/"
if not os.path.isdir(DATACARDDIR_BASE): os.mkdir(DATACARDDIR_BASE)
if not os.path.isdir(DATACARDDIR_BASE+"datacards/"): os.mkdir(DATACARDDIR_BASE+"datacards/")
DATACARDDIR_CHANNEL = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/datacards/"+options.channel+"/"
if not os.path.isdir(DATACARDDIR_CHANNEL): os.mkdir(DATACARDDIR_CHANNEL)
###DATACARDDIR         = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v7_calo/datacards/"+options.channel+"/"+options.tagvar+"/"
###if not os.path.isdir(DATACARDDIR): os.mkdir(DATACARDDIR)
#for launch_combine --- double check if names are equivalent!!!
DATACARDS =            "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/datacards/"+options.channel+"/"+options.tagvar+"/"
if not os.path.isdir(DATACARDS):
    print "Warning! Datacard dir does not exist, aborting..."
    os.mkdir(DATACARDS)
    exit()


#combine_results directories in HiggsCombine:
COMBINEDIR          = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/combine_results/"
if not os.path.isdir(COMBINEDIR):
    os.mkdir(COMBINEDIR)
    print "Warning! Combine results dir does not exist, aborting..."
    exit()
OUTPUTFOLDER_CHANNEL = "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/combine_results/"+options.channel+"/"
OUTPUTFOLDER_TAGVAR =  "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/combine_results/"+options.channel+"/"+options.tagvar+"/"
OUTPUTFOLDER =         "/afs/desy.de/user/l/lbenato/LLP_code_slc6/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/"+MAIN_OUT+"/combine_results/"+options.channel+"/"+options.tagvar+"/"+options.fom+"/"
if not os.path.isdir(OUTPUTFOLDER_CHANNEL): os.mkdir(OUTPUTFOLDER_CHANNEL)
if not os.path.isdir(OUTPUTFOLDER_TAGVAR): os.mkdir(OUTPUTFOLDER_TAGVAR)
if not os.path.isdir(OUTPUTFOLDER): os.mkdir(OUTPUTFOLDER)
#if not os.path.isdir(PLOTDIR): os.mkdir(PLOTDIR)

PRELIMINARY = True

signName  = "LLP"
particle = "S"
log_plot = False
j0Var = j1Var = ""

variables = {
    "Jets[0].nTrackConstituents" :
        {
        'label' : "displaced j_{0} n. track constituents",
        },
    "Jets[1].nTrackConstituents" :
        {
        'label' : "displaced j_{1} n. track constituents",
        },

    "Jets[0].nSelectedTracks" :
        {
        'label' : "displaced j_{0} n. selected tracks",
        },
    "Jets[1].nSelectedTracks" :
        {
        'label' : "displaced j_{1} n. selected tracks",
        },
    "Jets[2].nSelectedTracks" :
        {
        'label' : "displaced j_{2} n. selected tracks",
        },

    "Jets[0].nHadEFrac" :
        {
        'label' : "displaced j_{0} nHadEFrac",
        },
    "Jets[1].nHadEFrac" :
        {
        'label' : "displaced j_{1} nHadEFrac",
        },
    "Jets[0].cHadEFrac" :
        {
        'label' : "displaced j_{0} cHadEFrac",
        },
    "Jets[1].cHadEFrac" :
        {
        'label' : "displaced j_{1} cHadEFrac",
        },
    "Jets[2].cHadEFrac" :
        {
        'label' : "displaced j_{2} cHadEFrac",
        },


    "Jets[0].pfXWP1000" :
        {
        'label' : "displaced j_{0} pfXWP1000",
        },
    "Jets[1].pfXWP1000" :
        {
        'label' : "displaced j_{1} pfXWP1000",
        },
    "Jets[2].pfXWP1000" :
        {
        'label' : "displaced j_{2} pfXWP1000",
        },




   "nDTSegments" :
        {
        'label' : "nDTSegments",
        },
    "sigprob" :
        {
        'label' : "FCN score",
        },

    "j0_nTrackConstituents" :
        {
        'label' : "displaced j_{0} n. track constituents",
        },
    "j1_nTrackConstituents" :
        {
        'label' : "displaced j_{1} n. track constituents",
        },
    "j0_pfXWP1000" :
        {
        'label' : "displaced j_{0} pfXWP1000",
        },
    "j1_pfXWP1000" :
        {
        'label' : "displaced j_{1} pfXWP1000",
        },



}


#!# Define here the cuts you need for your SR!
if options.tagvar=="nTrackConstituents":
    j0_SIGMIN = 0
    j0_SIGMAX = 3#9
    j0_CRMIN = 4#10
    j0_CRMAX = 30
    j1_SIGMIN = 0
    j1_SIGMAX = 9
    j1_CRMIN = 10
    j1_CRMAX = 30
    NBINS_X = 30
    NBINS_Y = 30
elif options.tagvar=="nSelectedTracks":
    j0_SIGMIN = 0
    j0_SIGMAX = 1#9
    j0_CRMIN = 2#10
    j0_CRMAX = 30
    j1_SIGMIN = 0
    j1_SIGMAX = 1
    j1_CRMIN = 2
    j1_CRMAX = 30
    NBINS_X = 30
    NBINS_Y = 30
elif options.tagvar=="nHadEFrac":
    j0_SIGMIN = 0.9
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.9
    j1_SIGMIN = 0.9
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.9
    NBINS_X = 25
    NBINS_Y = 25
elif options.tagvar=="cHadEFrac":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0.0
    j1_SIGMAX = 0.05
    j1_CRMIN = 0.05
    j1_CRMAX = 1.0
    NBINS_X = 50
    NBINS_Y = 50

elif options.tagvar=="cHadEFrac_pfXWP1000_j0":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0.9
    j1_SIGMAX = 1.0
    j1_CRMIN = 0.
    j1_CRMAX = 0.9
    NBINS_X = 50
    NBINS_Y = 50
elif options.tagvar=="cHadEFrac_pfXWP1000_j1":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0.9
    j1_SIGMAX = 1.0
    j1_CRMIN = 0.
    j1_CRMAX = 0.9
    NBINS_X = 50
    NBINS_Y = 50
elif options.tagvar=="cHadEFrac_pfXWP1000_j2":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0.9
    j1_SIGMAX = 1.0
    j1_CRMIN = 0.
    j1_CRMAX = 0.9
    NBINS_X = 50
    NBINS_Y = 50
elif options.tagvar=="cHadEFrac_nSelectedTracks_j0":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0
    j1_SIGMAX = 3
    j1_CRMIN = 4
    j1_CRMAX = 30
    NBINS_X = 50
    NBINS_Y = 30

elif options.tagvar=="cHadEFrac_nSelectedTracks_j1":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0
    j1_SIGMAX = 3
    j1_CRMIN = 4
    j1_CRMAX = 30
    NBINS_X = 50
    NBINS_Y = 30

elif options.tagvar=="cHadEFrac_nSelectedTracks_j2":
    j0_SIGMIN = 0.0
    j0_SIGMAX = 0.05
    j0_CRMIN = 0.05
    j0_CRMAX = 1.0
    j1_SIGMIN = 0
    j1_SIGMAX = 3
    j1_CRMIN = 4
    j1_CRMAX = 30
    NBINS_X = 50
    NBINS_Y = 30

#very first attempt!
elif options.tagvar=="pfXWP1000":
    NBINS_X = 50
    NBINS_Y = 50
    j0_SIGMIN = 0.9
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.9-1./NBINS_X
    j1_SIGMIN = 0.9
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.9-1./NBINS_Y

elif options.tagvar=="one_pfXWP1000":
    j0_SIGMIN = 0.9
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.9-1./25.
    j1_SIGMIN = 0.
    j1_SIGMAX = 1.
    j1_CRMIN = 0.
    j1_CRMAX = 0.
    NBINS_X = 25
    NBINS_Y = 25



elif options.tagvar=="ORpfXWP1000":
    j0_SIGMIN = 0.
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.
    j1_SIGMIN = 0.
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.
    NBINS_X = 25
    NBINS_Y = 25

elif options.tagvar=="OR3pfXWP1000":
    j0_SIGMIN = 0
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.
    j1_SIGMIN = 0.
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.
    NBINS_X = 25
    NBINS_Y = 25

elif options.tagvar=="OR4pfXWP1000":
    j0_SIGMIN = 0.
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.
    j1_SIGMIN = 0.
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.
    NBINS_X = 25
    NBINS_Y = 25

elif options.tagvar=="OR5pfXWP1000":
    j0_SIGMIN = 0.
    j0_SIGMAX = 1.0
    j0_CRMIN = 0.0
    j0_CRMAX = 0.
    j1_SIGMIN = 0.
    j1_SIGMAX = 1.
    j1_CRMIN = 0.0
    j1_CRMAX = 0.
    NBINS_X = 25
    NBINS_Y = 25


elif options.tagvar=="nDTSegments":
    j0_SIGMIN = 20
    j0_SIGMAX = 70
    j0_CRMIN = 0
    j0_CRMAX = 19
    j1_SIGMIN = 20
    j1_SIGMAX = 70
    j1_CRMIN = 0
    j1_CRMAX = 19
    NBINS_X = 70
    NBINS_Y = 70
elif options.tagvar=="MET_nDTSegments":
    j0_SIGMIN = 500.
    j0_SIGMAX = 1100.
    j0_CRMIN = 100.
    j0_CRMAX = 500.- 1000./50.
    j1_SIGMIN = 20
    j1_SIGMAX = 70
    j1_CRMIN = 0
    j1_CRMAX = 19
    NBINS_X = 50
    NBINS_Y = 70
elif options.tagvar=="nTrackConstituents_nDTSegments":
    j0_SIGMIN = 0
    j0_SIGMAX = 7#try!
    j0_CRMIN = 8
    j0_CRMAX = 30
    j1_SIGMIN = 20#attempt!#11
    j1_SIGMAX = 70
    j1_CRMIN = 0
    j1_CRMAX = 10
    NBINS_X = 30
    NBINS_Y = 70


#!# define here your event weight
WEIGHT = "EventWeight"

SIGNAL_SUPPRESSION_FACTOR = 100
print "\n"
print "To make combine not fail, signal suppressed by a factor ", SIGNAL_SUPPRESSION_FACTOR
print "\n"

if is_array2root_tree:
    WEIGHT = "EventWeight*(1./0.2)"
    print "Multiply event weight considering test sample splitting!!"

#1# bins of your histo
#NBINS =100
weight_label = "_unweighted" if WEIGHT=="isMC" else ""

#!# These two numbers were just technical test to artificially enhance the number of signal events
SIGMULTFAC = 1#3.782#VBF x-sec
ENHANCEMENT = 1


massPoints = [x for x in range(15, 61, 5)]
ctauPoints = [100, 500, 1000, 2000, 5000, 10000]
massPoints = [15, 20, 25, 30, 40, 50]
massPoints = [15, 20, 30, 40, 50]

ctauPoints = [100, 500, 1000, 2000, 5000, 10000]
massPoints = [15, 40, 50]#TEST!

ctauPoints = [500, 1000, 2000, 5000, 10000]
massPoints = [150,400]
if is_array2root_tree:
    massPoints = [15, 20, 25]

MH = 1000

isMM = True if options.category=="MM" else False
isEE = True if options.category=="EE" else False
isComb = True if options.category=="comb" else False
