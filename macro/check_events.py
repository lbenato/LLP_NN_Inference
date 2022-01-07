#! /usr/bin/env python

import os, multiprocessing
import subprocess
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import time
import uproot
import pandas as pd
import gc
from array import array
from awkward import *
import root_numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from collections import defaultdict, OrderedDict
from itertools import chain

ERAS = ["2016","2017","2018"]


for era in ERAS:
    NTUPLEDIR          = ("/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_%s_SR/")% str(era)
    if era=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    if era=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    if era=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    print "******"
    print "Run era: ", era
    chain = TChain("tree")
    for s in samples["HighMET"]["files"]:
        #print NTUPLEDIR+s+".root"
        chain.Add(NTUPLEDIR+s+".root")
    print chain.GetEntries("nTagJets_0p996_JJ>1 && abs(Jets.eta)<1. && Jets.sigprob>0.996")
    print chain.Scan("RunNumber:LumiNumber:EventNumber:Jets.eta:Jets.sigprob","nTagJets_0p996_JJ>1 && Jets.sigprob>0.996")
