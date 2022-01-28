#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TMultiGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
import yaml
from collections import defaultdict, OrderedDict
import tensorflow as tf
from tensorflow import keras

#import awkward1 as ak
#import matplotlib.pyplot as plt

ERA = "2018"

CHAN = "SUSY"
LUMI = -1

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["HighMET"]["tot"]    
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["HighMET"]["tot"]    
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["HighMET"]["tot"]    

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

#SEL = "TtoEM"
#SEL = "ZtoMM"
SEL = "SR"

if SEL=="ZtoMM":
    data = ["SingleMuon"]
    back = ["DYJetsToLL"]
if SEL=="ZtoEE":
    data = ["SingleElectron"]
    back = ["DYJetsToLL"]
if SEL=="TtoEM":
    data = ["MuonEG"]
    back = ["TTbarGenMET"]
if SEL=="SR":
    data = ["HighMET"]#["QCD"]
    back = ["ZJetsToNuNu"]
sign = [
    'SUSY_mh127_ctau500_HH',
    'SUSY_mh150_ctau500_HH',
    'SUSY_mh175_ctau500_HH',
    'SUSY_mh200_ctau500_HH',
    'SUSY_mh250_ctau500_HH',
    'SUSY_mh300_ctau500_HH',
    'SUSY_mh400_ctau500_HH',
    'SUSY_mh600_ctau500_HH',
    'SUSY_mh800_ctau500_HH',
    'SUSY_mh1000_ctau500_HH',
    'SUSY_mh1250_ctau500_HH',
    'SUSY_mh1500_ctau500_HH',
    'SUSY_mh1800_ctau500_HH'
]


sign += [
    'SUSY_mh127_ctau3000_HH',
    'SUSY_mh150_ctau3000_HH',
    'SUSY_mh175_ctau3000_HH',
    'SUSY_mh200_ctau3000_HH',
    'SUSY_mh250_ctau3000_HH',
    'SUSY_mh300_ctau3000_HH',
    'SUSY_mh400_ctau3000_HH',
    'SUSY_mh600_ctau3000_HH',
    'SUSY_mh800_ctau3000_HH',
    'SUSY_mh1000_ctau3000_HH',
    'SUSY_mh1250_ctau3000_HH',
    'SUSY_mh1500_ctau3000_HH',
    'SUSY_mh1800_ctau3000_HH'
]

#sign = ['SUSY_mh800_ctau3000','SUSY_mh1800_ctau3000']
#sign = ['SUSY_mh400_ctau500_HH']

#General main
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_syst_unc/"
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_syst_unc_central_values/"
MAIN_central = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_syst_unc_central_values/"
MAIN_JERUp = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_syst_unc_JERUp/"
MAIN_JERDown = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_syst_unc_JERDown/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_"+SEL+"_signal_uncertainties_fix/"
SF = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/data_MC_SF_DNN_"+ERA+"/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

    '''
    timeData = np.array([])
    WeightData = np.array([])
    timeBack = np.array([])
    WeightBack = np.array([])
    timeSign = np.array([])
    WeightSign = np.array([])

    for d in data:
        print "\n"
        print "\n"
        print "opening: ", MAIN + d + ".root"
        files_list = samples[d]["files"]
        new_lista = []
        gen_events = 0
        for l in files_list:
            new_lista.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        print new_lista
        gen = uproot.iterate(new_lista,"tree",list_of_variables)

        for arrays in gen:
            cut_mask = arrays["isTtoEM"]>0
            cut_mask = np.logical_and(cut_mask,arrays["pt"]>200)

            cut_mask_csv = arrays["JetsNegative.CSV"]>=0.8
            cut_mask_csv = np.logical_and(cut_mask,cut_mask_csv)
            cut_mask = (cut_mask_csv.any()==True)
            cut_jets = cut_mask_csv

            timeData = np.concatenate(( timeData,arrays["JetsNegative.timeRecHitsEB"][cut_jets][cut_mask] ))

        #print "gen_events: ", gen_events
        #print "events passing: ", RunNumber.size
        #print ("%d (%.2f%s) ") % (RunNumber.size, 100.*float(RunNumber.size/gen_events),'\%')

    '''

def sign_unc_PU(label="",scale=True, do_smear=True,added=""):

    prj_weight = "EventWeight*TriggerWeight"
    results = defaultdict(dict)

    for s in sign:
        #event-wise variables
        bin_2 = []
        bin_1 = []
        bin_2_up = []
        bin_2_down = []
        bin_1_up = []
        bin_1_down = []

        print "Ntuples: ", MAIN
        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            print "tree weight: ", tree_weight
            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                nTagJets = 0

                #Event level selections
                if tree.MinJetMetDPhi<=0.5:
                    continue

                #print "---------"
                #print "event ", e
                #print "nj: ", neg_jets.size()
                shift_w_up = 1.
                shift_w_down = 1.
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue

                    if jets[n].sigprob>0.996:
                        nTagJets+=1
                        
                #event-wise variables
                #print "nTagJets: ", nTagJets
                #print "shift_w: ", shift_w
                ev_weight = tree.EventWeight * tree_weight * tree.TriggerWeight
                if nTagJets==1:
                    bin_1.append(ev_weight * tree.PUReWeight)
                    bin_1_up.append(ev_weight * tree.PUReWeightUp)
                    bin_1_down.append(ev_weight * tree.PUReWeightDown)
                if nTagJets>1:
                    bin_2.append(ev_weight * tree.PUReWeight)
                    bin_2_up.append(ev_weight * tree.PUReWeightUp)
                    bin_2_down.append(ev_weight * tree.PUReWeightDown)
    
        #Here calculate differences with and without SFs; check also in all eta range
        #print "bin_2"
        #print bin_2
        #print "bin_2_shift"
        #print bin_2_shift
        y1 = np.sum(np.array(bin_1))
        y1_up = np.sum(np.array(bin_1_up))
        y1_down = np.sum(np.array(bin_1_down))
        y2 = np.sum(np.array(bin_2))
        y2_up = np.sum(np.array(bin_2_up))
        y2_down = np.sum(np.array(bin_2_down))
        print "y2: ", y2, " ; y2 up: ", y2_up, " ; diff ", 100*abs(y2_up-y2)/y2
        print "y2: ", y2, " ; y2 down: ", y2_down, " ; diff ", 100*abs(y2_down-y2)/y2
        print "y1: ", y1, " ; y1 up: ", y1_up, " ; diff ", 100*abs(y1_up-y1)/y1
        print "y1: ", y1, " ; y1 down: ", y1_down, " ; diff ", 100*abs(y1_down-y1)/y1
        results[s]['y2'] = y2
        results[s]['y2_up'] = y2_up
        results[s]['diff_y2_up'] = 100*abs(y2_up-y2)/y2
        results[s]['y2_down'] = y2_down
        results[s]['diff_y2_down'] = 100*abs(y2_down-y2)/y2

        results[s]['y1'] = y1
        results[s]['y1_up'] = y1_up
        results[s]['diff_y1_up'] = 100*abs(y1_up-y1)/y1
        results[s]['y1_down'] = y1_down
        results[s]['diff_y1_down'] = 100*abs(y1_down-y1)/y1
        #....
        #fill results

    print results
    with open(OUT+"signal_PU_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_PU_unc"+label+".yaml"


def sign_unc_QCD_scales(label="",scale=True, do_smear=True,added=""):

    prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    results = defaultdict(dict)

    for s in sign:
        #event-wise variables
        bin_2 = defaultdict(dict)
        bin_1 = defaultdict(dict)
        y1 = defaultdict(dict)
        y2 = defaultdict(dict)
        #bin_2_up = []
        #bin_2_down = []
        #bin_1_up = []
        #bin_1_down = []
        for n in [0,2,3,4,5,7,9]:
            bin_2[n] = []
            bin_1[n] = []

        print "Ntuples: ", MAIN
        for j, ss in enumerate(samples[s]['files']):
            print "\n"
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            
            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                #QCDWeights = getattr(tree,"QCDWeights")

                #Event level selections
                if tree.MinJetMetDPhi<=0.5:
                    continue

                #print "---------"
                #print "event ", e
                #print "n weights: ", tree.QCDWeights.size()
                weights_vec = {}
                weight_1 = 1.
                for n in range(tree.QCDWeights.size()):
                    #print "QCDWeight[",n,"] ", tree.QCDWeights[n]
                    #Avoid anti-correlated variations
                    if n!=6 and n!=8 and n!=1:
                        weights_vec[n] = tree.QCDWeights[n]
                    weight_1 = tree.QCDWeights[1]

                #Normalization Step
                #for n in weights_vec.keys():
                #    #weights_vec[n] = weights_vec[n]/weight_1
                    
                nTagJets = 0
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue

                    if jets[n].sigprob>0.996:
                        nTagJets+=1
                        
                #event-wise variables
                #print "nTagJets: ", nTagJets
                #print "shift_w: ", shift_w
                ev_weight = tree.EventWeight * tree_weight * tree.TriggerWeight * tree.PUReWeight
                for n in weights_vec.keys():
                    if nTagJets==1:
                        bin_1[n].append(ev_weight*weights_vec[n])
                    if nTagJets>1:
                        bin_2[n].append(ev_weight*weights_vec[n])
    
        #Here calculate differences with and without SFs; check also in all eta range
        #print "bin_2"
        #print bin_1
        #print bin_2
        #print "bin_2_shift"
        #print bin_2_shift
        #y1 = np.sum(np.array(bin_1))
        #y1_up = np.sum(np.array(bin_1_up))
        #y1_down = np.sum(np.array(bin_1_down))
        for n in weights_vec.keys():
            y1[n] = np.sum(np.array(bin_1[n]))
            y2[n] = np.sum(np.array(bin_2[n]))
            #print "Weight ", n, " yields"
            #print y1[n]
            #print y2[n]

        y1_up   = max( abs(y1[n] - y1[0]) for n in weights_vec.keys() )
        y1_down = min( abs(y1[n] - y1[0]) for n in weights_vec.keys() )
        y2_up   = max( abs(y2[n] - y2[0]) for n in weights_vec.keys() )
        y2_down = min( abs(y2[n] - y2[0]) for n in weights_vec.keys() )

        print "---------------------------------------------------------------------"
        print "   y2 nominal: ", y2[0], " ; y2 max: ", y2_up, " ; perc. ", 100*abs(y2_up)/y2[0]
        print "   y2 nominal: ", y2[0], " ; y2 min: ", y2_down, " ; perc. ", 100*abs(y2_down)/y2[0]
        print "   y1 nominal: ", y1[0], " ; y1 max: ", y1_up, " ; perc. ", 100*abs(y1_up)/y1[0]
        print "   y1 nominal: ", y1[0], " ; y1 min: ", y1_down, " ; perc. ", 100*abs(y1_down)/y1[0]
        print "---------------------------------------------------------------------"

        results[s]['y2'] = y2[0]
        results[s]['y2_up'] = y2_up
        results[s]['diff_y2_up'] = 100*abs(y2_up)/y2[0]
        results[s]['y2_down'] = y2_down
        results[s]['diff_y2_down'] = 100*abs(y2_down)/y2[0]

        results[s]['y1'] = y1[0]
        results[s]['y1_up'] = y1_up
        results[s]['diff_y1_up'] = 100*abs(y1_up)/y1[0]
        results[s]['y1_down'] = y1_down
        results[s]['diff_y1_down'] = 100*abs(y1_down)/y1[0]
        #....
        #fill results

    #print results
    with open(OUT+"signal_QCD_scales_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_QCD_scales_unc"+label+".yaml"


def sign_unc_PDF(label="",scale=True, do_smear=True,added=""):

    prj_weight = "EventWeight*PUReWeight*TriggerWeight"
    results = defaultdict(dict)

    for s in sign:
        #event-wise variables
        bin_2 = defaultdict(dict)
        bin_1 = defaultdict(dict)
        y1 = defaultdict(dict)
        y2 = defaultdict(dict)

        for n in range(103):
            bin_2[n] = []
            bin_1[n] = []

        n_pathological = 0
        
        print "Ntuples: ", MAIN
        for j, ss in enumerate(samples[s]['files']):
            print "\n"
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            
            
            for e in range(0,tree.GetEntries()):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                #QCDWeights = getattr(tree,"QCDWeights")

                #Event level selections
                if tree.MinJetMetDPhi<=0.5:
                    continue

                #print "---------"
                #print "event ", e
                #print "n weights: ", tree.PDFWeights.size()
                weights_vec = {}
                weight_1 = 1.
                max_w = 0.
                for n in range(tree.PDFWeights.size()):
                    #print "PDFWeight[",n,"] ", tree.PDFWeights[n]
                    weights_vec[n] = tree.PDFWeights[n]
                    weight_0 = tree.PDFWeights[0]
                    max_w = max(max_w,abs(weights_vec[n]))
                if max_w>50:
                    n_pathological+=1
                    print "Likely pathological PDF weights at event ", e
                    print "max_w: ", max_w
                    print "Event n. %d:%d:%d" % (tree.RunNumber,tree.LumiNumber,tree.EventNumber)
                    continue

                #Normalization Step
                for n in weights_vec.keys():
                    if n==0:
                        weights_vec[n] = weights_vec[n]/weight_0
                    
                nTagJets = 0
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue

                    if jets[n].sigprob>0.996:
                        nTagJets+=1
                        
                #event-wise variables
                #print "nTagJets: ", nTagJets
                #print "shift_w: ", shift_w
                ev_weight = tree.EventWeight * tree_weight * tree.TriggerWeight * tree.PUReWeight
                for n in weights_vec.keys():
                    if nTagJets==1:
                        bin_1[n].append(ev_weight*weights_vec[n])
                    if nTagJets>1:
                        bin_2[n].append(ev_weight*weights_vec[n])

                '''
                if nTagJets>1:
                    print "----------------------"
                    for n in weights_vec.keys():
                        if abs(weights_vec[n])>50:
                            print "In bin 2:"
                            print "   PDFWeight[",n,"] ",weights_vec[n]
                '''

        #Here calculate differences with and without SFs; check also in all eta range
        #print "bin_2"
        #print bin_1
        #print bin_2
        #print "bin_2_shift"
        #print bin_2_shift
        #y1 = np.sum(np.array(bin_1))
        #y1_up = np.sum(np.array(bin_1_up))
        #y1_down = np.sum(np.array(bin_1_down))
        for n in weights_vec.keys():
            y1[n] = np.sum(np.array(bin_1[n]))
            y2[n] = np.sum(np.array(bin_2[n]))
            #print "Weight ", n, " bin_2"
            #print bin_2[n]
            #print "Weight ", n, " yields"
            #print y1[n]
            #print y2[n]

        y1_up   = max( abs(y1[n] - y1[0]) for n in weights_vec.keys() )
        y1_down = min( abs(y1[n] - y1[0]) for n in weights_vec.keys() )
        y2_up   = max( abs(y2[n] - y2[0]) for n in weights_vec.keys() )
        y2_down = min( abs(y2[n] - y2[0]) for n in weights_vec.keys() )
        max_y2 = max(y2[n] for n in weights_vec.keys())
        min_y2 = min(y2[n] for n in weights_vec.keys())

        print "---------------------------------------------------------------------"
        print "   max_y2 : ", max_y2, " ; min_y2: ", min_y2
        print "   y2 nominal: ", y2[0], " ; y2 max: ", y2_up, " ; perc. ", 100*abs(y2_up)/y2[0]
        print "   y2 nominal: ", y2[0], " ; y2 min: ", y2_down, " ; perc. ", 100*abs(y2_down)/y2[0]
        print "   y1 nominal: ", y1[0], " ; y1 max: ", y1_up, " ; perc. ", 100*abs(y1_up)/y1[0]
        print "   y1 nominal: ", y1[0], " ; y1 min: ", y1_down, " ; perc. ", 100*abs(y1_down)/y1[0]
        if n_pathological>0:
            print "Identified",n_pathological,"events with pathological PDF weights"
        print "---------------------------------------------------------------------"

        results[s]['y2'] = y2[0]
        results[s]['y2_up'] = y2_up
        results[s]['diff_y2_up'] = 100*abs(y2_up)/y2[0]
        results[s]['y2_down'] = y2_down
        results[s]['diff_y2_down'] = 100*abs(y2_down)/y2[0]

        results[s]['y1'] = y1[0]
        results[s]['y1_up'] = y1_up
        results[s]['diff_y1_up'] = 100*abs(y1_up)/y1[0]
        results[s]['y1_down'] = y1_down
        results[s]['diff_y1_down'] = 100*abs(y1_down)/y1[0]
        #....
        #fill results

    #print results
    with open(OUT+"signal_PDF_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_PDF_unc"+label+".yaml"


def sign_unc_jet(label="",scale=True, do_smear=True,added=""):

    prj_weight = "EventWeight*PUReWeight"
    results = defaultdict(dict)

    for s in sign:
        #event-wise variables
        bin_2 = []
        bin_1 = []
        bin_2_JERUp = []
        bin_2_JERDown = []
        bin_2_JESUp = []
        bin_2_JESDown = []
        bin_2_UnclUp = []
        bin_2_UnclDown = []
        bin_1_JERUp = []
        bin_1_JERDown = []
        bin_1_JESUp = []
        bin_1_JESDown = []
        bin_1_UnclUp = []
        bin_1_UnclDown = []

        print "Ntuples: ", MAIN_central
        print MAIN_JERUp
        print MAIN_JERUp

        max_events = 1000000000000000000
        #max_events = 5000

        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN_central+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            for e in range(0,min(max_events,tree.GetEntries())):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                nTagJets = 0
                nTagJets_JESUp = 0
                nTagJets_JESDown = 0
                nTagJets_UnclUp = 0
                nTagJets_UnclDown = 0
                
                #In this loop we compute:
                #0. central
                #1. JESUp
                #2. JESDown
                #3. UnclUp
                #4. UnclDown
                
                #print "---------"
                #print "event ", e
                #print "nj: ", neg_jets.size()
                shift_w_up = 1.
                shift_w_down = 1.
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue
                        
                    #We need:
                    #selection on met
                    #selection on jet pt
                    #selection on min d phi
                    #count tagged jets
                    #choose the right trigger weight

                    #0. central
                    if met.pt>200 and jets[n].pt>30 and tree.MinJetMetDPhi>0.5 and jets[n].sigprob>0.996:
                        nTagJets+=1
                    #1. JESUp
                    if met.ptShiftJetEnUp>200 and jets[n].ptJESUp>30 and tree.MinJetMetDPhi_JESUp>0.5 and jets[n].sigprob>0.996:
                        nTagJets_JESUp+=1
                    #2. JESDown
                    if met.ptShiftJetEnDown>200 and jets[n].ptJESDown>30 and tree.MinJetMetDPhi_JESDown>0.5 and jets[n].sigprob>0.996:
                        nTagJets_JESDown+=1
                    #3. UnclUp
                    if met.ptShiftUnclusteredEnUp>200 and jets[n].pt>30 and tree.MinJetMetDPhi_UnclusteredEnUp>0.5 and jets[n].sigprob>0.996:
                        nTagJets_UnclUp+=1
                    #4. UnclDown
                    if met.ptShiftUnclusteredEnDown>200 and jets[n].pt>30 and tree.MinJetMetDPhi_UnclusteredEnDown>0.5 and jets[n].sigprob>0.996:
                        nTagJets_UnclDown+=1
                        
                #event-wise variables
                #print "nTagJets: ", nTagJets
                #print "shift_w: ", shift_w
                ev_weight = tree.EventWeight * tree_weight * tree.PUReWeight
                #0. central
                if nTagJets>1:
                    bin_2.append(ev_weight * tree.TriggerWeight)
                #1. JESUp
                if nTagJets_JESUp>1:
                    bin_2_JESUp.append(ev_weight * tree.TriggerWeight_JESUp)
                #2. JESDown
                if nTagJets_JESDown>1:
                    bin_2_JESDown.append(ev_weight * tree.TriggerWeight_JESDown)
                #3. UnclUp
                if nTagJets_UnclUp>1:
                    bin_2_UnclUp.append(ev_weight * tree.TriggerWeight_UnclusteredEnUp)
                #4. UnclDown
                if nTagJets_UnclDown>1:
                    bin_2_UnclDown.append(ev_weight * tree.TriggerWeight_UnclusteredEnDown)
    
        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN_JERUp+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            for e in range(0,min(max_events,tree.GetEntries())):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                nTagJets_JERUp = 0
                
                #In this loop we compute:
                #1. JERUp
                #print "---------"
                #print "event ", e
                #print "nj: ", neg_jets.size()
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue
                        
                    #We need:
                    #selection on met
                    #selection on jet pt
                    #selection on min d phi
                    #count tagged jets
                    #choose the right trigger weight

                    #1. JERUp
                    if met.ptShiftJetResUp>200 and jets[n].pt>30 and tree.MinJetMetDPhi_JERUp>0.5 and jets[n].sigprob>0.996:
                        nTagJets_JERUp+=1
                        
                ev_weight = tree.EventWeight * tree_weight * tree.PUReWeight
                #1. JERUp
                if nTagJets_JERUp>1:
                    bin_2_JERUp.append(ev_weight * tree.TriggerWeight_JERUp)

        for j, ss in enumerate(samples[s]['files']):
            print "Performing computation for ", ss
            tmp_file = TFile(MAIN_JERDown+ss+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()
            for e in range(0,min(max_events,tree.GetEntries())):
                tree.GetEntry(e)
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")
                nTagJets_JERDown = 0
                
                #In this loop we compute:
                #1. JERDown
                #print "---------"
                #print "event ", e
                #print "nj: ", neg_jets.size()
                for n in range(jets.size()):
                    if jets[n].eta<=-1:
                        continue
                    if jets[n].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[n].phi>=0.4 and jets[n].phi<0.9: continue
                        
                    #We need:
                    #selection on met
                    #selection on jet pt
                    #selection on min d phi
                    #count tagged jets
                    #choose the right trigger weight

                    #1. JERDown
                    if met.ptShiftJetResDown>200 and jets[n].pt>30 and tree.MinJetMetDPhi_JERDown>0.5 and jets[n].sigprob>0.996:
                        nTagJets_JERDown+=1
                        
                ev_weight = tree.EventWeight * tree_weight * tree.PUReWeight
                #1. JERDown
                if nTagJets_JERDown>1:
                    bin_2_JERDown.append(ev_weight * tree.TriggerWeight_JERDown)

        #Here calculate differences with and without SFs; check also in all eta range
        y2 = np.sum(np.array(bin_2))
        y2_JESUp = np.sum(np.array(bin_2_JESUp))
        y2_JESDown = np.sum(np.array(bin_2_JESDown))
        y2_UnclUp = np.sum(np.array(bin_2_UnclUp))
        y2_UnclDown = np.sum(np.array(bin_2_UnclDown))
        y2_JERUp = np.sum(np.array(bin_2_JERUp))
        y2_JERDown = np.sum(np.array(bin_2_JERDown))
        print "y2: ", y2, " ; y2_JESUp: ", y2_JESUp, " ; diff ", 100*abs(y2_JESUp-y2)/y2
        print "y2: ", y2, " ; y2_JESDown: ", y2_JESDown, " ; diff ", 100*abs(y2_JESDown-y2)/y2
        print "y2: ", y2, " ; y2_UnclUp: ", y2_UnclUp, " ; diff ", 100*abs(y2_UnclUp-y2)/y2
        print "y2: ", y2, " ; y2_UnclDown: ", y2_UnclDown, " ; diff ", 100*abs(y2_UnclDown-y2)/y2
        print "y2: ", y2, " ; y2_JERUp: ", y2_JERUp, " ; diff ", 100*abs(y2_JERUp-y2)/y2
        print "y2: ", y2, " ; y2_JERDown: ", y2_JERDown, " ; diff ", 100*abs(y2_JERDown-y2)/y2
        results[s]['y2'] = y2
        #JES
        results[s]['y2_JESUp'] = y2_JESUp
        results[s]['diff_y2_JESUp'] = 100*abs(y2_JESUp-y2)/y2
        results[s]['y2_JESDown'] = y2_JESDown
        results[s]['diff_y2_JESDown'] = 100*abs(y2_JESDown-y2)/y2
        #Uncl
        results[s]['y2_UnclUp'] = y2_UnclUp
        results[s]['diff_y2_UnclUp'] = 100*abs(y2_UnclUp-y2)/y2
        results[s]['y2_UnclDown'] = y2_UnclDown
        results[s]['diff_y2_UnclDown'] = 100*abs(y2_UnclDown-y2)/y2
        #JER
        results[s]['y2_JERUp'] = y2_JERUp
        results[s]['diff_y2_JERUp'] = 100*abs(y2_JERUp-y2)/y2
        results[s]['y2_JERDown'] = y2_JERDown
        results[s]['diff_y2_JERDown'] = 100*abs(y2_JERDown-y2)/y2
        #....
        #fill results

    print results
    with open(OUT+"signal_jet_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_jet_unc"+label+".yaml"

def sign_unc_lumi(label="",added=""):
    '''
    Uncorrelated 2016      1.0
    Uncorrelated 2017      2.0
    Uncorrelated 2018      1.5
    Correlated 2016,2017,2018   0.6,0.9,2.0
    Correlated 2017,2018      0.6,0.2
    '''
    results = defaultdict(dict)
    for s in sign:
        s = s.replace("_HH","")
        if ERA=="2016":
            results[s]['lumi_2016_unc'] = 1.0 
            ##results[s]['lumi_2017_unc'] = 0.0 
            ##results[s]['lumi_2018_unc'] = 0.0 
            results[s]['lumi_corr_all'] = 0.6 
            ##results[s]['lumi_corr_2017_2018'] = 0.0 
        if ERA=="2017":
            ##results[s]['lumi_2016_unc'] = 0.0 
            results[s]['lumi_2017_unc'] = 2.0 
            ##results[s]['lumi_2018_unc'] = 0.0
            results[s]['lumi_corr_all'] = 0.9 
            results[s]['lumi_corr_2017_2018'] = 0.6 
        if ERA=="2018":
            ##results[s]['lumi_2016_unc'] = 0.0 
            ##results[s]['lumi_2017_unc'] = 0.0 
            results[s]['lumi_2018_unc'] = 1.5 
            results[s]['lumi_corr_all'] = 2.0 
            results[s]['lumi_corr_2017_2018'] = 0.2 

    print results
    with open(OUT+"signal_lumi_datacard_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_lumi_datacard_unc"+label+".yaml"



def sign_unc_tau(label="",added=""):
    results = defaultdict(dict)
    for s in sign:
        s = s.replace("_HH","")
        results[s]['tau'] = 0.5 
    print results
    with open(OUT+"signal_tau_datacard_unc"+label+".yaml","w") as f:
        yaml.dump(results, f)
        f.close()
    print "Written in ", OUT+"signal_tau_datacard_unc"+label+".yaml"


def sign_unc_DNN(label=""):
    with open(OUT+"signal_DNN_unc"+label+".yaml","r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()
    uncertainties = defaultdict(dict)

    for s in sign:
        s = s.replace("_HH","")
        uncertainties[s]['DNN_'+ERA+label] = results[s]['diff_y2']

    with open(OUT+"signal_DNN_datacard_unc"+label+".yaml","w") as f:
        yaml.dump(uncertainties, f)
        f.close()
    print "Written in ", OUT+"signal_DNN_datacard_unc"+label+".yaml"



def draw_syst_unc_PU(added):
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_PU_unc"+added+".yaml","r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()
    samp = results.keys()
    masses = []
    ctaus = []
    for s in samp:
        masses.append(samples[s]['mass'])
        ctaus.append(samples[s]['ctau'])
    masses = np.unique(np.sort(np.array(masses)))
    ctaus = np.unique(np.sort(np.array(ctaus)))
    print masses
    print ctaus

    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.5, 0.6, 0.9, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff_up = {}
    diff_down = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % max(results[s]['diff_y2_up'],results[s]['diff_y2_down']))
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff_up[c] = TGraph()
        diff_down[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff_up[c].SetPoint(n,m,results[s]['diff_y2_up'])
                    diff_down[c].SetPoint(n,m,results[s]['diff_y2_down'])
                    n+=1
        diff_up[c].SetMarkerStyle(24)
        diff_up[c].SetMarkerColor(colors[count_c])
        diff_up[c].SetLineColor(colors[count_c])
        diff_up[c].SetLineWidth(2)
        diff_down[c].SetMarkerStyle(25)
        diff_down[c].SetMarkerColor(colors[count_c])
        diff_down[c].SetLineColor(colors[count_c])
        diff_down[c].SetLineWidth(2)
        diff_down[c].SetLineStyle(7)
        diff_up[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        diff_up[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff_up[c].GetYaxis().SetTitleSize(0.05)
        diff_up[c].SetMinimum(0)
        leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; PU weight up","PL")
        leg1.AddEntry(diff_down[c],"c_{#tau} = "+str(c/1000.)+" m; PU weight down","PL")
        mg.Add(diff_up[c])
        mg.Add(diff_down[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    c1.SetGrid()
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0.)
    mg.SetMaximum(16.)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_PU_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_PU_uncertainty"+added+".png")
    c1.Close()

    #Write datacards dictionary
    uncertainties = defaultdict(dict)
    for s in sign:
        s = s.replace("_HH","")
        uncertainties[s]['PU'] = max(results[s]['diff_y2_up'],results[s]['diff_y2_down'])

    with open(OUT+"signal_PU_datacard_unc"+added+".yaml","w") as f:
        yaml.dump(uncertainties, f)
        f.close()
    print "Written in ", OUT+"signal_PU_datacard_unc"+added+".yaml"



def draw_syst_unc_envelope(unc_type,added):
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_"+unc_type+"_unc"+added+".yaml","r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()
    samp = results.keys()
    masses = []
    ctaus = []
    for s in samp:
        masses.append(samples[s]['mass'])
        ctaus.append(samples[s]['ctau'])
    masses = np.unique(np.sort(np.array(masses)))
    ctaus = np.unique(np.sort(np.array(ctaus)))
    print masses
    print ctaus

    mg = TMultiGraph()
    leg1 = TLegend(0.6, 0.2, 0.9, 0.5)
    #leg1 = TLegend(0.15+0.5, 0.6+0.2, 0.3+0.5, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff_up = {}
    diff_down = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % results[s]['diff_y2_up'])
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff_up[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff_up[c].SetPoint(n,m,results[s]['diff_y2_up'])
                    n+=1
        diff_up[c].SetMarkerStyle(24)
        diff_up[c].SetMarkerColor(colors[count_c])
        diff_up[c].SetLineColor(colors[count_c])
        diff_up[c].SetLineWidth(2)
        diff_up[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        diff_up[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff_up[c].GetYaxis().SetTitleSize(0.05)
        diff_up[c].SetMinimum(0)
        if unc_type=="PDF":
            leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; PDF envelope","PL")
        if unc_type=="QCD_scales":
            leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; QCD scales envelope","PL")
        mg.Add(diff_up[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetGrid()
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0.)
    mg.SetMaximum(16.)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_"+unc_type+"_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_"+unc_type+"_uncertainty"+added+".png")
    c1.Close()

    #Write datacards dictionary
    uncertainties = defaultdict(dict)
    for s in sign:
        s = s.replace("_HH","")
        uncertainties[s][unc_type] = max(results[s]['diff_y2_up'],results[s]['diff_y2_down'])

    with open(OUT+"signal_"+unc_type+"_datacard_unc"+added+".yaml","w") as f:
        yaml.dump(uncertainties, f)
        f.close()
    print "Written in ", OUT+"signal_"+unc_type+"_datacard_unc"+added+".yaml"


def draw_syst_unc_jet(added):
    print "Drawing uncertainty vs mass"
    with open(OUT+"signal_jet_unc"+added+".yaml","r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        f.close()

    samp = results.keys()
    masses = []
    ctaus = []
    for s in samp:
        masses.append(samples[s]['mass'])
        ctaus.append(samples[s]['ctau'])
    masses = np.unique(np.sort(np.array(masses)))
    ctaus = np.unique(np.sort(np.array(ctaus)))
    print masses
    print ctaus

    ##JES
    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.5, 0.6, 0.9, 0.9)
    #leg1 = TLegend(0.15+0.5, 0.6+0.2, 0.3+0.5, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff_up = {}
    diff_down = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % max(results[s]['diff_y2_JESUp'],results[s]['diff_y2_JESDown']))
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff_up[c] = TGraph()
        diff_down[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff_up[c].SetPoint(n,m,results[s]['diff_y2_JESUp'])
                    diff_down[c].SetPoint(n,m,results[s]['diff_y2_JESDown'])
                    n+=1
        diff_up[c].SetMarkerStyle(24)
        diff_up[c].SetMarkerColor(colors[count_c])
        diff_up[c].SetLineColor(colors[count_c])
        diff_up[c].SetLineWidth(2)
        diff_down[c].SetMarkerStyle(25)
        diff_down[c].SetMarkerColor(colors[count_c])
        diff_down[c].SetLineColor(colors[count_c])
        diff_down[c].SetLineWidth(2)
        diff_down[c].SetLineStyle(7)
        diff_up[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        diff_up[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff_up[c].GetYaxis().SetTitleSize(0.05)
        diff_up[c].SetMinimum(0)
        leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; JES up","PL")
        leg1.AddEntry(diff_down[c],"c_{#tau} = "+str(c/1000.)+" m; JES down","PL")
        mg.Add(diff_up[c])
        mg.Add(diff_down[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetGrid()
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0.)
    mg.SetMaximum(16.)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_jet_JES_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_jet_JES_uncertainty"+added+".png")
    c1.Close()
    mg.Delete()


    ##JER
    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.5, 0.6, 0.9, 0.9)
    #leg1 = TLegend(0.15+0.5, 0.6+0.2, 0.3+0.5, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff_up = {}
    diff_down = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % max(results[s]['diff_y2_JERUp'],results[s]['diff_y2_JERDown']))
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff_up[c] = TGraph()
        diff_down[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff_up[c].SetPoint(n,m,results[s]['diff_y2_JERUp'])
                    diff_down[c].SetPoint(n,m,results[s]['diff_y2_JERDown'])
                    n+=1
        diff_up[c].SetMarkerStyle(24)
        diff_up[c].SetMarkerColor(colors[count_c])
        diff_up[c].SetLineColor(colors[count_c])
        diff_up[c].SetLineWidth(2)
        diff_down[c].SetMarkerStyle(25)
        diff_down[c].SetMarkerColor(colors[count_c])
        diff_down[c].SetLineColor(colors[count_c])
        diff_down[c].SetLineWidth(2)
        diff_down[c].SetLineStyle(7)
        diff_up[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        diff_up[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff_up[c].GetYaxis().SetTitleSize(0.05)
        diff_up[c].SetMinimum(0)
        leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; JER up","PL")
        leg1.AddEntry(diff_down[c],"c_{#tau} = "+str(c/1000.)+" m; JER down","PL")
        mg.Add(diff_up[c])
        mg.Add(diff_down[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetGrid()
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0.)
    mg.SetMaximum(16.)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_jet_JER_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_jet_JER_uncertainty"+added+".png")
    c1.Close()
    mg.Delete()


    ##Uncl
    mg = TMultiGraph()
    leg1 = TLegend(0.15+0.5, 0.6, 0.9, 0.9)
    #leg1 = TLegend(0.15+0.5, 0.6+0.2, 0.3+0.5, 0.9)
    colors = [2,418,801,856,602,920,881]
    diff_up = {}
    diff_down = {}
    count_c = 0
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & uncertainty (\%)"+"\\"+"\\"
    for m in masses:
        string = ""
        nt = 0
        for c in ctaus:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % max(results[s]['diff_y2_UnclUp'],results[s]['diff_y2_UnclDown']))
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    for c in ctaus:
        diff_up[c] = TGraph()
        diff_down[c] = TGraph()
        n = 0
        for m in masses:
            for s in samp:
                if "mh"+str(m)+"_ctau"+str(c) in s:
                    diff_up[c].SetPoint(n,m,results[s]['diff_y2_UnclUp'])
                    diff_down[c].SetPoint(n,m,results[s]['diff_y2_UnclDown'])
                    n+=1
        diff_up[c].SetMarkerStyle(24)
        diff_up[c].SetMarkerColor(colors[count_c])
        diff_up[c].SetLineColor(colors[count_c])
        diff_up[c].SetLineWidth(2)
        diff_down[c].SetMarkerStyle(25)
        diff_down[c].SetMarkerColor(colors[count_c])
        diff_down[c].SetLineColor(colors[count_c])
        diff_down[c].SetLineWidth(2)
        diff_down[c].SetLineStyle(7)
        diff_up[c].GetXaxis().SetTitle("m_{#chi} (GeV)")
        diff_up[c].GetYaxis().SetTitle("Uncertainty (%)")
        diff_up[c].GetYaxis().SetTitleSize(0.05)
        diff_up[c].SetMinimum(0)
        leg1.AddEntry(diff_up[c],"c_{#tau} = "+str(c/1000.)+" m; unclustered energy up","PL")
        leg1.AddEntry(diff_down[c],"c_{#tau} = "+str(c/1000.)+" m; unclustered energy down","PL")
        mg.Add(diff_up[c])
        mg.Add(diff_down[c])
        count_c+=1

    c1 = TCanvas("c1", "c1", 800, 600)
    c1.SetGrid()
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.12)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Uncertainty (%)")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0.)
    mg.SetMaximum(16.)
    mg.Draw("APL")
    leg1.Draw()
    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    #drawAnalysis("LL"+CHAN)
    drawRegion(SEL)
    #c1.SetLogx()
    c1.Print(OUT+"signal_unclustered_energy_uncertainty"+added+".pdf")
    c1.Print(OUT+"signal_unclustered_energy_uncertainty"+added+".png")
    c1.Close()
    mg.Delete()

    #Write datacards dictionary
    uncertainties = defaultdict(dict)
    for s in sign:
        if not ERA=="2018":
            s = s.replace("_HH","")
        #Don't want G-H or B-F separated
        uncertainties[s.replace("_HH","")]['JES_'+ERA] = max(results[s]['diff_y2_JESUp'],results[s]['diff_y2_JESDown'])
        uncertainties[s.replace("_HH","")]['JER_'+ERA] = max(results[s]['diff_y2_JERUp'],results[s]['diff_y2_JERDown'])
        uncertainties[s.replace("_HH","")]['uncl_energy_'+ERA] = max(results[s]['diff_y2_UnclUp'],results[s]['diff_y2_UnclDown'])

    with open(OUT+"signal_jet_datacard_unc"+added+".yaml","w") as f:
        yaml.dump(uncertainties, f)
        f.close()
    print "Written in ", OUT+"signal_jet_datacard_unc"+added+".yaml"

 
lab = "_not_scaled"#"_constant_1"
lab = ""
added = ""
if ERA=="2016":
    added="_G-H"
    #added="_B-F"

# DNN
sign_unc_DNN(label=added)

# Tau
sign_unc_tau(label=added)

# Lumi
sign_unc_lumi(label=added)

# JER JEC
#sign_unc_jet(label=added)#+"_debug")
draw_syst_unc_jet(added)

# PDF
#sign_unc_PDF(label=added)
draw_syst_unc_envelope("PDF",added)

# QCD scales
#sign_unc_QCD_scales(label=added)
draw_syst_unc_envelope("QCD_scales",added)

# Pile-Up
#sign_unc_PU(label=added+"_test")
draw_syst_unc_PU(added)
