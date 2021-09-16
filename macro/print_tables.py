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
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox
from ROOT import RDataFrame
from ctypes import c_double

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

BASEDIR = "plots/Efficiency_AN/v5_calo_AOD_"#+ERA+"_"+REGION+"/"

def print_tab(ERAS,inp_dict):

    y_0 = 0
    y_1 = 0
    y_2 = 0
    e_0_sq = 0
    e_1_sq = 0
    e_2_sq = 0
    pred_1 = 0
    pred_2 = 0
    pred_2_alt = 0
    e_1_comp_sq = 0
    e_1_stat_sq = 0
    e_2_meth = 0
    e_2_comp_sq = 0
    e_2_stat_sq = 0
    pred_reg_2_per_era = {}
    pred_reg_1_per_era = {}

    for era in ERAS:

        label=""
        if inp_dict["eta_cut"]==True:
            label+="_eta_1p0"
        if inp_dict["phi_cut"]==True and "2016" not in era:
            label+="_phi_cut"
        if inp_dict["eta"]:
            label+="_vs_eta"
            if inp_dict["phi"]:
                label+="_vs_phi"
        if inp_dict["clos"]:
            dnn_threshold = 0.9
            label+="_closure"+str(dnn_threshold).replace(".","p")

        more_label = ""
        if len(era)>4:
            more_label=era[4:]
        #print "more_label: ", more_label
        if inp_dict["kill_qcd"]:
            more_label += "_MinDPhi_0p5"

        year = era[:4]
        #print "Running on era ", era, " ; year ", year


        #Initialize dict at every run era
        #This is needed for the composition uncertainty
        pred_2_per_era = 0
        pred_1_per_era = 0
        for reg in inp_dict["extr_reg"]:
            pred_reg_2_per_era[reg] = 0
            pred_reg_1_per_era[reg] = 0
        
        if inp_dict["REGION"]=="ZtoLL":
            tags = ["ZtoEE","ZtoMM"]
        elif inp_dict["REGION"]=="WtoLN":
            tags = ["WtoEN","WtoMN"]
        else:
            tags = [inp_dict["REGION"]]

        #Here we need a loop on the datasets composing the REGION
        for tag in tags:
            s = tag
            if tag=="TtoEM":
                s = "MuonEG"
            if tag=="SR":
                s = "HighMET"
            if tag=="ZtoEE" or tag=="WtoEN":
                s = "EGamma" if era=="2018" else "SingleElectron"
            if tag=="ZtoMM" or tag=="WtoMN":
                s = "SingleMuon"

            PLOTDIR = BASEDIR + year + "_" + tag + "/"

            with open(PLOTDIR+"BkgPredResults_"+year+"_"+tag+ "_"+s+ label+more_label+inp_dict["label_2"]+".yaml","r") as f:
                results = yaml.load(f, Loader=yaml.Loader)
                f.close()
            #print "Info: dictionary open: "+PLOTDIR+"BkgPredResults_"+year+"_"+tag+"_"+s+label+more_label+inp_dict["label_2"]+".yaml"

            #These should be added through the datasets composition
            #Stat uncertainties go in quadrature
            y_0 += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["y_0"]
            y_1 += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["y_1"]
            y_2 += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["y_2"]
            e_0_sq += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["e_0"]**2
            e_1_sq += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["e_1"]**2
            e_2_sq += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["e_2"]**2

            pred_1 += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["pred_1"]
            pred_2 += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["pred_2_from_1"]
            pred_2_alt += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["pred_2"]

            pred_2_per_era += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["pred_2_from_1"]
            pred_1_per_era += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["pred_1"]

            e_1_stat_sq += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["e_pred_1"]**2
            e_2_stat_sq += results[inp_dict["main"]+more_label+inp_dict["label_2"]][s]["e_pred_2_from_1"]**2

            #results changes at every tag loop (depends on s)
            #add the predictions of each dataset, and of each era
            for reg in inp_dict["extr_reg"]:
                pred_reg_2_per_era[reg] += results[reg+more_label+inp_dict["label_2"]][s]["pred_2_from_1"]
                pred_reg_1_per_era[reg] += results[reg+more_label+inp_dict["label_2"]][s]["pred_1"]


        #Composition uncertainty:
        #Determine max per-era
        dev_1 = 0
        dev_2 = 0
        for reg in inp_dict["extr_reg"]:
            dev_2 = max(dev_2,abs(pred_2_per_era - pred_reg_2_per_era[reg]))
            dev_1 = max(dev_1,abs(pred_1_per_era - pred_reg_1_per_era[reg]))

        #Add in quadrature per each era separately
        e_1_comp_sq += dev_1**2
        e_2_comp_sq += dev_2**2

    #Method unc: linear
    #We have now added the predictions across datasets, here we need a simple difference
    #After the era loop
    e_2_meth = abs(pred_2 - pred_2_alt)


    round_fact = 3 if inp_dict["REGION"]=="SR" else 2

    print "\n"
    print "bin 0: ", y_0, "+-", round(math.sqrt(e_0_sq),0)
    print "bin 1: ", y_1, "+-", round(math.sqrt(e_1_sq),0)
    if inp_dict["REGION"]!="SR":
        print "bin 2: ", y_2, "+-", round(math.sqrt(e_2_sq),0)
    else:
        print "bin 2: ", "x +- x"
    print "pred 1: ", round(pred_1,2), "+-", round(math.sqrt(e_1_stat_sq),2) , "(stat.) +- " , round(math.sqrt(e_1_comp_sq),2) , "(syst.) = ", round(pred_1,2), "+-", round(math.sqrt(e_1_comp_sq + e_1_stat_sq),2), "(", round(100*math.sqrt(e_1_comp_sq + e_1_stat_sq)/pred_1,1), "%)"
    print "pred 2: ", round(pred_2,2), "+-", round(math.sqrt(e_2_stat_sq),2) , "(stat.) +- " , round(math.sqrt(e_2_comp_sq),2) , "(syst. comp.) +- " , round(e_2_meth,2) , "(syst. method) = ", round(pred_2,2), " +- ", round(math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq),2), "(", round(100*math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq)/pred_2,1),"%)"

    print "\n"
    print "Year & bin 0 yield & bin 1 yield & bin 1 prediction & stat. unc. & syst. unc. "+"\\"+"\\"
    print "\hline"
    print ERAS[0].replace("_"," ") if len(ERAS)==1 else "Tot.", " & ", round(y_0,0), " $\pm$ " , round(math.sqrt(e_0_sq),0) , " & " , round(y_1,0), " $\pm$ " , round(math.sqrt(e_1_sq),0) ,  " & " , round(pred_1,2), " $\pm$ " , round(math.sqrt(e_1_stat_sq + e_1_comp_sq),2), "(", round(100*math.sqrt(e_1_comp_sq + e_1_stat_sq)/pred_1,1), "\%)", " & ", round(math.sqrt(e_1_stat_sq),2), " & " , round(math.sqrt(e_1_comp_sq),2), "\\"+"\\"


    print "Year & bin 2 yield & bin 2 prediction & stat. unc. & syst. unc. & syst. unc "+"\\"+"\\"
    print " & &  & & (method) & (composition) "+"\\"+"\\"
    print "\hline"
    #print ERAS[0] if len(ERAS)==1 else "Tot.", " & ", str(y_2) + " $\pm$ " + str(round(math.sqrt(e_2_sq),0)) if inp_dict["REGION"]!="SR" else " - $\pm$ - " , " & ", round(pred_2,2), " $\pm$ ", round(math.sqrt(e_2_stat_sq),2) , "(stat.) $\pm$ " , round(math.sqrt(e_2_comp_sq),2) , "(syst. comp.) $\pm$ " , round(e_2_meth,2) , "(syst. method) = ", round(pred_2,2), " $\pm$ ", round(math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq),2)  , "(", round(100*math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq)/pred_2,1),"\%)", "\\"+"\\"

    print ERAS[0].replace("_"," ") if len(ERAS)==1 else "Tot.", " & ", str(y_2) + " $\pm$ " + str(round(math.sqrt(e_2_sq),0)) if inp_dict["REGION"]!="SR" else " - $\pm$ - " , " & ", round(pred_2,round_fact), " $\pm$ ", round(math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq),round_fact)  , "(", round(100*math.sqrt(e_2_comp_sq + e_2_meth**2 + e_2_stat_sq)/pred_2,1),"\%)", " & ", round(math.sqrt(e_2_stat_sq),round_fact) , " & ", round(e_2_meth,round_fact) , " & ", round(math.sqrt(e_2_comp_sq),round_fact), "\\"+"\\"


inp_dict = {
    "REGION" : "WtoLN",
    "main"   : "WtoLN",
    "extr_reg" : ["ZtoLL","WtoLN","TtoEM","JetHT"],
    "kill_qcd" : True,
    "eta" : True,
    "phi" : False,
    "eta_cut" : True,
    "phi_cut" : True,
    "clos" : True,
    "label_2" : "",
    }

print_tab(["2016_B-F"],inp_dict)
print_tab(["2016_G-H"],inp_dict)
print_tab(["2017"],inp_dict)
print_tab(["2018"],inp_dict)
print_tab(["2016_B-F","2016_G-H","2017","2018"],inp_dict)
print_tab(["2016_G-H","2017","2018"],inp_dict)
#"ERAS" : ["2016_B-F","2016_G-H","2017","2018"],
exit()

inp_dict = {
    "REGION" : "SR",
    "main"   : "WtoLN",
    "extr_reg" : ["ZtoLL","WtoLN","TtoEM","JetHT"],
    "kill_qcd" : True,
    "eta" : True,
    "phi" : False,
    "eta_cut" : True,
    "phi_cut" : True,
    "clos" : False,
    "label_2" : ""
    }
print_tab(inp_dict)
