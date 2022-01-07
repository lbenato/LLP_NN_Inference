#! /usr/bin/env python 
#import os, multiprocessing
#import subprocess
#import copy
#import math
#import matplotlib.pyplot as plt
#import numpy as np
#import json
import yaml
#import time
#import uproot
#import pandas as pd
#import gc
#from array import array
#from awkward import *
#import root_numpy
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
#from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
#from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
#from ROOT import TStyle, TCanvas, TPad, TProfile
#from ROOT import TLegend, TLatex, TText, TLine, TBox
#from ROOT import RDataFrame
#from ctypes import c_double

ERA = "2016"
REGION = "WtoMN"
EXTR_REGION = "WtoMN"+"_B-F"
pred_file_name = "plots/Efficiency/v5_calo_AOD_"+ERA+"_"+REGION+"/BkgPredResults_"+ERA+"_"+REGION+"_vs_eta_B-F"

with open(pred_file_name+".yaml","r") as f:
    print "Info: opening dictionary in file "+pred_file_name+".yaml"
    print "Extrapolation region: ", REGION
    results = yaml.load(f, Loader=yaml.Loader)
    f.close()

print results
data_set = results[EXTR_REGION].keys()[0]
#print results[EXTR_REGION][data_set]
dicty = results[EXTR_REGION][data_set]

print "=== Prediction of "+REGION+ " extrapolating from "+EXTR_REGION+" ==="
table1 =  PrettyTable(['Dataset', 'Bin 1 Yield', 'Bin 1 Pred', 'Discr. %'])
row1 = [data_set+" "+ERA,round(dicty["y_1"],0),round(dicty["pred_1"],2), round(100*(dicty["y_1"]-dicty["pred_1"])/dicty["y_1"],2)]
table1.add_row(row1)
print table1

table2 = PrettyTable(['Bin 2 Yield', 'Bin 2 Pred from 0', 'Pred from 0 Discr. %', 'Bin 2 Pred from 1', 'Pred from 1 Discr. %', 'Syst. Unc. %'])
row2 = [round(dicty["y_2"],0) , round(dicty["pred_2"],4), round(100*(dicty["y_2"]-dicty["pred_2"])/dicty["y_2"] if dicty["y_2"]>0 else 100,2), round(dicty["pred_2_from_1"],4),  round(100*(dicty["y_2"]-dicty["pred_2_from_1"])/dicty["y_2"] if dicty["y_2"]>0 else 100,2), round(100*abs(dicty["pred_2"]-dicty["pred_2_from_1"])/((dicty["pred_2"]+dicty["pred_2_from_1"])/2),2) ]
table2.add_row(row2)
print table2
