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
#import pandas as pd
import gc
import random
from array import array
import awkward
import root_numpy
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#from prettytable import PrettyTable
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, Double
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, TH2D, THStack, TGraph, TGraphAsymmErrors, TF1, TEfficiency, TObjArray, TIter
from ROOT import TStyle, TCanvas, TPad, TProfile
from ROOT import TLegend, TLatex, TText, TLine, TBox, TRandom3
#from ROOT import RDataFrame
from ctypes import c_double
#from scipy.interpolate import Rbf, interp1d
#from scipy.interpolate import NearestNDInterpolator
#from scipy.interpolate import LinearNDInterpolator

from NNInferenceCMSSW.LLP_NN_Inference.variables import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from collections import defaultdict, OrderedDict
from itertools import chain
#import tensorflow as tf
#from tensorflow import keras

########## SETTINGS ##########

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-v", "--variable", action="store", type="string", dest="variable", default="met_pt_nomu")#"nPV")
parser.add_option("-c", "--cut", action="store", type="string", dest="cut", default="met_test")
parser.add_option("-C", "--compare", action="store", type="string", dest="compare", default="")
parser.add_option("-D", "--drawsignal", action="store_true", dest="drawsignal", default=False)
parser.add_option("-n", "--normalized", action="store_true", dest="normalized", default=False)
parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="mu")#"mu3nPV"
parser.add_option("-r", "--run", action="store", type="string", dest="run", default="G")
parser.add_option("-e", "--efficiency", action="store_true", dest="efficiency", default=False)
parser.add_option("-s", "--sample", action="store", type="string", dest="sample", default="All")
parser.add_option("-g", "--goodplots", action="store_true", default=False, dest="goodplots")#not needed in 2016
parser.add_option("-a", "--all", action="store_true", default=False, dest="all")
parser.add_option("-b", "--bash", action="store_true", default=True, dest="bash")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
parser.add_option("-f", "--final", action="store_true", default=False, dest="final")
parser.add_option("-R", "--rebin", action="store_true", default=False, dest="rebin")
parser.add_option("-p", "--public", action="store_true", default=False, dest="public")
(options, args) = parser.parse_args()
if options.bash: gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)

#ERA
ERA                = "2016"
REGION             = "SR"#"SR"#"HBHE"#"ZtoEEBoost"#"WtoMN"#"WtoEN"
CUT                = "isSR"#"isSR"#"isSRHBHE"#"isZtoEE"#"isWtoMN"#"isWtoEN"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"
KILL_QCD           = True#False
DO_ETA             = True
DO_PHI             = False#False#
if DO_PHI:
    DO_ETA = False
CUT_ETA            = True#True#True#False#True#True#False
CUT_PHI            = True
BLIND              = False
TOYS               = True

print "\n"
print "region: ", REGION
print "kill qcd: ", KILL_QCD
print "do eta: ", DO_ETA
print "do phi: ", DO_PHI
print "eta cut: ", CUT_ETA
print "phi cut: ", CUT_PHI
print "\n"


NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
if REGION=="SR":
    print "SR in v6"
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"

#JHEP referee
PLOTDIR = "plots/Efficiency_AN_fix/v6_calo_AOD_"+ERA+"_"+REGION+"_JHEP/"


CHAN               = "SUSY"
particle           = "#chi"
ctaupoint          = 500

#A bit less points:
ctaus_500          = np.array([10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 25000, 50000, 100000])
ctaus = np.unique(np.concatenate((ctaus_500,ctaus_3000)))

signalMultFactor   = 0.001#!!!
signalBRfactor     = 0.9
PRELIMINARY        = False
TAGVAR             = "nTagJets_0p996_JJ"

REBIN              = options.rebin
SAVE               = True
#LUMI               = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#8507.558453#59.74*1000#Run2018

back = ["All"]
data = ["HighMET"]
sign = []
back = []

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    #print "Fake jet tag"
    jet_tag=""
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]


COMPARE = options.compare
DRAWSIGNAL = options.drawsignal

########## SAMPLES ##########

colors = [856, 1,  634, 420, 806, 882, 401, 418, 881, 798, 602, 921]
colors_jj = [1,2,4,418,801,856]
colors_jj = [1,2,4,418,801,856]
colors = colors_jj + [881, 798, 602, 921]
lines = [2,1,3,4,1,2,2,2,2]
markers = [20,20,20,20,20,24,24,24,24]
markers = [20,21,24,25,20,24,24,24,24]
markers_MC = [20,20,20,20,20,24,24,24,24]
siz = 1.3
marker_sizes = [siz,siz,siz,siz,siz,siz,siz,siz,siz]
#markers = [20,21]#[24,25,21,20]
########## ######## ##########

#gROOT.SetBatch(True)

#cut_den = CUT
#cut_num = cut_den + " && "
chain = {}
filename = {}
hist_den = {}
hist_num = {}
hist_num_cutbased = {}
eff = {} 
eff_cutbased = {} 
dnn_bins = array('d', [0.,.00001,.0001,0.001,.01,.05,.1,.25,.5,.75,1.,])

#New version from Jiajing
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])
less_bins_pt = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500])#,10000])
less_bins_pt = array('d', [1,10,20,30,40,50,60,75,100,200,500])#,10000])#this is good
less_bins_plot = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])

#print "Warning, overflow causing nans, remove last bin"
np_bins = np.array(less_bins)
np_bins = np_bins[np_bins>=30]#only interested in non-zero bins
np_bins = np_bins[np_bins<10000]#only interested in non-nan bins

prev_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,300,500,10000])
bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,250,300,400,500,600,700,800,900,10000])
more_bins=array('d', [1,10,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,350,400,450,500,550,600,650,700,800,900,1000,2500,5000,10000,100000])
np_more_bins = np.array(more_bins)

more_bins_eta = array('d',[-1.5,-1.45,-1.4,-1.35,-1.3,-1.25,-1.2,-1.15,-1.1,-1.05,-1.,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5 ])
#homogeneous:
less_bins_eta = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])

more_bins_phi = array('d',[-3.2 , -3.04, -2.88, -2.72, -2.56, -2.4 , -2.24, -2.08, -1.92,
                           -1.76, -1.6 , -1.44, -1.28, -1.12, -0.96, -0.8 , -0.64, -0.48,
                           -0.32, -0.16,  0.  ,  0.16,  0.32,  0.48,  0.64,  0.8 ,  0.96,
                           1.12,  1.28,  1.44,  1.6 ,  1.76,  1.92,  2.08,  2.24,  2.4 ,
                           2.56,  2.72,  2.88,  3.04,  3.2                       ])
less_bins_phi = array('d',[
    -3.2 , -2.56, -1.92, -1.28, -0.64,  0.  ,  0.64,  1.28,  1.92,
    2.56,  3.2
])

np_bins_eta = np.array(less_bins_eta)
np_bins_eta = np_bins_eta[0:-1]

maxeff = 0.0015#15#08#2#15

def round_to_1(x):
    return round(x, -int(math.floor(math.log10(abs(x)))))


def get_tree_weights(sample_list,dataset_label="",main_pred_sample="HighMET"):

    #Fix LUMI
    if ERA=="2016":
        if dataset_label == "_G-H":
            LUMI  = lumi[ main_pred_sample ]["G"]+lumi[ main_pred_sample ]["H"]#["tot"]
        elif dataset_label == "_B-F":
            LUMI  = lumi[ main_pred_sample ]["B"]+lumi[ main_pred_sample ]["C"]+lumi[ main_pred_sample ]["D"]+lumi[ main_pred_sample ]["E"]+lumi[ main_pred_sample ]["F"]#["tot"]
    else:
        LUMI  = lumi[ main_pred_sample ]["tot"]

    tree_w_dict = defaultdict(dict)
    for i, s in enumerate(sample_list):
        for l, ss in enumerate(samples[s]['files']):
            #Tree weight
            if ('Run201') in ss:
                t_w = 1.
            else:
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                nevents = filename.Get("c_nEvents").GetBinContent(1)
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    print "SUSY central, consider sample dictionary for nevents!"
                    nevents = sample[ss]['nevents']
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                filename.Close()
                if('GluGluH2_H2ToSSTobbbb') in ss:
                    xs = 1.
                elif('XXTo4J') in ss:
                    xs = 1.*0.001
                elif('gluinoGMSB') in ss:
                    xs = 1.*0.001
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.*0.001
                elif ('n3n2-n1-hbb-hbb') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                    #print "Scaling SUSY to its true excluded x-sec"
                    #xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                elif ('TChiHH') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('splitSUSY') in ss:
                    print "Scaling splitSUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    #print "But consider BR!"
                    xs = 1.
                    #xs *= sample[name]['BR']
                else:
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                print "LUMI ", LUMI
                print "xs ", xs
                print "nevents ", nevents
                t_w = LUMI * xs / nevents
                if(b_skipTrain>0):
                    #print("Skip even events: ")
                    #print "n_pass: ", n_pass
                    #print "n_odd: ", n_odd
                    #if(n_pass>0):
                    #    print "ratio: ", float(n_odd/n_pass)
                    if(n_odd>0):
                        #print "even/odd weight: ", float(n_pass/n_odd)
                        t_w *= float(n_pass/n_odd)
            print("%s has tree weight %f")%(ss,t_w)
            tree_w_dict[s][ss] = t_w

    return tree_w_dict



def calculate_tag_eff(tree_weight_dict,sample_list,add_label="",check_closure=False,eta=False,phi=False,j_idx=-1,eta_cut=False,phi_cut=False):

    tree_weights = {}
    chain_entries = {}
    chain_entries_cumulative = {}
    chain = {}
    np_num = {}
    np_den = {}
    np_w_num = {}
    np_w_den = {}
    np_weight = {}
    #print np_more_bins
    #Initialize dictionary of arrays
    '''
    for b in np_more_bins:
        np_den[b] = []#np.array([])
        np_num[b] = []#np.array([])
        np_w_den[b] = []#np.array([])
        np_w_num[b] = []#np.array([])
        np_weight[b] = []
    '''

    if check_closure:
        dnn_threshold = 0.9#5
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        print  "DNN threshold: ", dnn_threshold

    if eta_cut:
        print "Apply acceptance cut |eta|<1."

    if phi_cut:
        print "Apply acceptance cut phi"

    if eta_cut==True:
        add_label+="_eta_1p0"

    if phi_cut==True:
        add_label+="_phi_cut"

    if eta==True:
        add_label+="_vs_eta"

    if phi==True:
        add_label+="_vs_phi"

    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")

    for i, s in enumerate(sample_list):
        if eta==True:
            hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins_eta)-1, more_bins_eta)
            hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins_eta)-1, more_bins_eta)
            hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins_eta)-1, more_bins_eta)
        else:
            if phi==True:
                hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins_phi)-1, more_bins_phi)
                hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins_phi)-1, more_bins_phi)
                hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins_phi)-1, more_bins_phi)
            else:
                hist_den[s] = TH1F(s+"_den", s+"_den", len(more_bins)-1, more_bins)
                hist_num[s] = TH1F(s+"_num", s+"_num", len(more_bins)-1, more_bins)
                hist_num_cutbased[s] = TH1F(s+"_num_cutbased", s+"_num_cutbased", len(more_bins)-1, more_bins)
        hist_den[s].Sumw2()
        hist_num[s].Sumw2()
        hist_num_cutbased[s].Sumw2()
        print s

        passed_num = np.array([])
        passed_den = np.array([])
        w_num = np.array([])
        w_den = np.array([])

        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight"]#"nLeptons"
        list_of_variables += ["dt_ecal_dist","min_dPhi_jets*"]
        if CUT!="isSRHBHE":
            list_of_variables += [CUT]
        else:
            list_of_variables += ["isSR"]

        if ERA=="2018" and CUT == "isSR":
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
                print "Entries per sample ", ss, " : ", chain_entries[l]
            else:
                chain_entries[l]=chain_entries_cumulative[l] - chain_entries_cumulative[l-1]
                print "Entries per sample ", ss, " : ", chain_entries[l]

            chunk_size = 100000
            n_iter = int(float(chain_entries[l])/float(chunk_size))
            c = 0
            if chain_entries[l]%chunk_size!=0:
                n_iter += 1

            print "chain_entries: ", chain_entries[l]
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter
            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            
            for arrays in gen:
                st_it = time.time()
                key_list = arrays.keys()
                array_size_tot+=len(arrays[ key_list[0] ])
                tree_w_array = tree_weight_dict[s][ss]*np.ones( len(arrays[ key_list[0] ])  )
                if CUT == "isSR":
                    cut_mask = arrays[CUT]>0
                    #cosmic
                    cosmic_veto = arrays["dt_ecal_dist"]<0.5
                    cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
                else:
                    cut_mask = (arrays[CUT]>0)


                #HEM
                if (CUT == "isSR" or CUT=="isSRHBHE") and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))

                if KILL_QCD:
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                if eta:
                    pt = arrays["Jets.eta"][cut_mask]
                else:
                    if phi:
                        pt = arrays["Jets.phi"][cut_mask]
                    else:
                        pt = arrays["Jets.pt"][cut_mask]
                sigprob = arrays["Jets.sigprob"][cut_mask]

                if phi_cut==True and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi][cut_mask]

                if eta_cut==True and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_eta][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_eta][cut_mask]

                if phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)
                    if eta:
                        pt = arrays["Jets.eta"][cut_mask_phi_eta][cut_mask]
                    else:
                        if phi:
                            pt = arrays["Jets.phi"][cut_mask_phi_eta][cut_mask]
                        else:
                            pt = arrays["Jets.pt"][cut_mask_phi_eta][cut_mask]
                    sigprob = arrays["Jets.sigprob"][cut_mask_phi_eta][cut_mask]


                tag_mask = (sigprob > dnn_threshold)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                #pt_untag = pt[untag_mask]
                pt_tag = pt[tag_mask]


                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]


                del arrays

                #Major bug... den must be pt, not pt_untag...
                passed_num = np.concatenate( (passed_num, np.hstack(pt_tag)) )
                #passed_den = np.concatenate( (passed_den, np.hstack(pt_untag)) )
                passed_den = np.concatenate( (passed_den, np.hstack(pt)) )
                w_num = np.concatenate( (w_num, np.hstack( pt_tag.astype(bool)*weight ) ) )
                #w_den = np.concatenate( (w_den, np.hstack( pt_untag.astype(bool)*weight ) ) )
                w_den = np.concatenate( (w_den, np.hstack( pt.astype(bool)*weight ) ) )


                #We'll have to flatten and remove unnecessary zeros...
                #print np_den[130.0].shape
                en_it = time.time()
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        _ = root_numpy.fill_hist( hist_den[s], passed_den, weights=w_den )
        print hist_den[s].Print()
        _ = root_numpy.fill_hist( hist_num[s], passed_num, weights=w_num )
        print hist_num[s].Print()
        
        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(hist_num[s],hist_den[s])
        eff[s].SetMarkerSize(1.)
        eff[s].SetMarkerStyle(markers[i])#(sign_sampl[s]['marker'])
        eff[s].SetMarkerColor(colors[i])#(2)
        eff[s].SetFillColor(colors[i])#(2) 
        eff[s].SetLineColor(colors[i])#(2)
        eff[s].SetLineWidth(2)
        eff[s].GetYaxis().SetRangeUser(0.,maxeff)
        eff[s].GetYaxis().SetTitle("Efficiency")#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleOffset(1.2)#("Efficiency (L1+HLT)")
        eff[s].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta==True:
            eff[s].GetXaxis().SetRangeUser(more_bins_eta[0],more_bins_eta[-1])
            eff[s].GetXaxis().SetTitle("Jet #eta")
        else:
            if phi:
                eff[s].GetXaxis().SetRangeUser(more_bins_phi[0],more_bins_phi[-1])
                eff[s].GetXaxis().SetTitle("Jet #varphi")
            else:
                eff[s].GetXaxis().SetRangeUser(more_bins[0],more_bins[-1])
                eff[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")#pt
                eff[s].GetXaxis().SetTitleSize(0.04)
                eff[s].GetXaxis().SetTitleOffset(1.1)

        eff_cutbased[s] = TGraphAsymmErrors()
        eff_cutbased[s].BayesDivide(hist_num_cutbased[s],hist_den[s])



        can = TCanvas("can","can", 1000, 800)
        can.cd()
        if i==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")
            can.Print(PLOTDIR+"TagEff_"+s+add_label+".png")
            can.Print(PLOTDIR+"TagEff_"+s+add_label+".pdf")

        outfile = TFile(PLOTDIR+"TagEff_"+s+add_label+".root","RECREATE")
        outfile.cd()
        eff[s].Write("eff_"+s)
        eff_cutbased[s].Write("eff_cutbased_"+s)
        hist_den[s].Write("den_"+s)
        hist_num[s].Write("num_"+s)
        hist_num_cutbased[s].Write("num_cutbased_"+s)
        can.Write()
        print "Info in <TFile::Write>: root file "+PLOTDIR+"TagEff_"+s+add_label+".root has been created"
        outfile.Close()




samples_to_run = data
jet_tag = ""
clos = False

#ERA
if ERA=="2016":
    jet_tag+="_G-H"#"_B-F"#"_G-H"#"_G-H"#"_B-F"#"_B-F"#
if KILL_QCD:
    jet_tag += "_MinDPhi_0p5"

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2018"
        MINPHI = 0.9 #min allowed
        MAXPHI = 0.4 #max allowed
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]#["tot"]
    if CUT_PHI==True:
        print "Here phi cuts 2017"
        MINPHI = 3.5 #min allowed
        MAXPHI = 2.7 #max allowed
    #print "Fake lumi of 100 /fb"
    #LUMI = 1000.*1000.*100
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if CUT_PHI==True:
        print "Here phi cuts 2016 --> to be re-set as false"
        CUT_PHI=False
    if "_G-H" in jet_tag:
        print "Only GH!"
        LUMI  = lumi[ data[0] ]["G"]+lumi[ data[0] ]["H"]#["tot"]
    elif "_B-F" in jet_tag:
        LUMI  = lumi[ data[0] ]["B"]+lumi[ data[0] ]["C"]+lumi[ data[0] ]["D"]+lumi[ data[0] ]["E"]+lumi[ data[0] ]["F"]#["tot"]
    else:
        LUMI  = lumi[ data[0] ]["tot"]

print "Ntupledir: ", NTUPLEDIR
#print "Luminosity: ", data[0], LUMI

sample_dict = {}
reg_comb = ""
isMC = False
if samples_to_run==back:
    isMC = True
    if "Wto" in REGION:
        sample_dict["WtoMN"] = "WJetsToLNu"
        sample_dict["WtoEN"] = "WJetsToLNu"
        reg_comb = "WtoLN"
    elif "Zto" in REGION:
        sample_dict["ZtoMM"] = "DYJetsToLL"
        sample_dict["ZtoEE"] = "DYJetsToLL"
        reg_comb = "ZtoLL"
    else:
        sample_dict[REGION] = samples_to_run[0]
        print "This won't work for lists, hence MC... keep in mind"
        reg_comb = REGION
else:
    if "Wto" in REGION:
        #sample_dict["WtoMN"] = "SingleMuon"
        sample_dict["WtoMN_MET"] = "SingleMuon"
        if ERA=="2018":
            #sample_dict["WtoEN"] = "EGamma"
            sample_dict["WtoEN_MET"] = "EGamma"
        else:
            if "_B-F" not in jet_tag:
                #sample_dict["WtoEN"] = "SingleElectron"
                sample_dict["WtoEN_MET"] = "SingleElectron"
        reg_comb = "WtoLN"
        if "MET" in REGION:
            reg_comb = "WtoLN_MET"
    elif "Zto" in REGION:
        sample_dict["ZtoMM"] = "SingleMuon"
        if ERA=="2018":
            sample_dict["ZtoEE"] = "EGamma"
        else:
            if "_B-F" not in jet_tag:
                sample_dict["ZtoEE"] = "SingleElectron"
        reg_comb = "ZtoLL"
        if "Boost" in REGION:
            sample_dict = {}
            sample_dict["ZtoMMBoost"] = "SingleMuon"
            if ERA=="2018":
                sample_dict["ZtoEEBoost"] = "EGamma"
            else:
                if "_B-F" not in jet_tag:
                    sample_dict["ZtoEEBoost"] = "SingleElectron"
            reg_comb = "ZtoLLBoost"
    else:
        sample_dict[REGION] = samples_to_run[0]
        print "This won't work for lists, hence MC... keep in mind"
        reg_comb = REGION


calculate_tag_eff(get_tree_weights(samples_to_run,LUMI),samples_to_run,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,j_idx=-1,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
##draw_tag_eff(samples_to_run,reg_label=reg_comb,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#draw_tag_eff_updated(sample_dict,reg_label=reg_comb,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#draw_tag_eff_signal(samples_to_run,reg_label=reg_comb,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#exit()

#THIS HERE!!!
'''
draw_data_combination(
#draw_data_combination_with_MC(
    ERA,
    #["WtoMN","WtoEN","ZtoMM","ZtoEE","JetHT","DiJetMET","TtoEM"],
    #["JetHT","JetHT"],
    #["ZtoMM","TtoEM",],#["ZtoMM","WtoMN","TtoEM","JetHT"],#,"JetHT",SR
    ##regions_labels=["","","",jet_tag,"","","","","","","","",""],
    #datasets=["","","","","QCD","","","","","","","",],
    #datasets=["","QCD","","","","","","","",],
    #
    # universal
    #
    #["ZtoMM","ZtoEE","WtoMN","WtoEN","TtoEM","JetHT",],#"DiJetMET"],#SR
    #
    #this
    ["ZtoLL","WtoLN","TtoEM","JetHT"],#"ZtoLLBoost","HBHE"],#"DiJetMET"],#SR
    #
    #["ZtoLL","JetHT",],#"WtoLN","TtoEM","JetHT",],#"DiJetMET"],#SR
    #2016 B-F
    #["ZtoMM","WtoMN","TtoEM","JetHT",],#"DiJetMET"],#SR
    regions_labels = [jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    #regions_labels = [jet_tag+"_MET_60_80",jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],

    # - - - - - - - - - -
    ##This: for MET bins
    #["WtoLN_MET","WtoLN_MET","WtoLN_MET","WtoLN_MET"],
    #regions_labels = [jet_tag+"_MET_40_60",jet_tag+"_MET_60_80",jet_tag+"_MET_80_100",jet_tag+"_MET_100_150",jet_tag+"_MET_150",jet_tag,jet_tag,jet_tag,jet_tag],
    #additional_legend_labels = ["; 40<E_{T}^{miss}<60 GeV","; 60<E_{T}^{miss}<80 GeV","; 80<E_{T}^{miss}<100 GeV","; 100<E_{T}^{miss}<150 GeV","; E_{T}^{miss}>150 GeV","","","","","",""],
    # - - - - - - - - - -

    #["WtoMN","WtoMN","WtoEN","WtoEN","WtoMN","WtoEN"],
    #regions_labels=["","_MinDPhiBarrel_0p5","","_MinDPhiBarrel_0p5","_MinDPhi_0p5","_MinDPhi_0p5"],
    #compare different MET cuts
    #["WtoEN","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT","WtoEN_noMT"],
    #regions_labels = ["","_noMT","_MET100","_MET150","_MET200"],
    #regions_labels=["_jet_0","_jet_1","_jet_0","_jet_1","","","","","","","","",""],
    #MC in SR
    #["TtoEM","TtoEM"],
    #datasets = ["","TTbar"],
    #["SR","SR","SR","SR","SR","SR"],
    #datasets=["ZJetsToNuNu","WJetsToLNu","QCD","TTbarGenMET","VV","All"],
    #2016
    #["ZtoMM","WtoMN","TtoEM","JetHT"],
    #regions_labels=["_B-F","_B-F","_B-F","_B-F",],
    #dphi
    #["WtoMN","WtoMN","WtoMN",],
    #regions_labels=["","_MinDPhiBarrel_0p5","_MinDPhi_0p5"],
    #regions_labels=["_B-F","_B-F_MinDPhiBarrel_0p5","_B-F_MinDPhi_0p5"],
    #["JetHT","JetHT"],
    #regions_labels = ["","_InvertBeamHalo"],
    add_label="",#"",#"_vs_QCD_MC",#"_closure"
    lab_2=jet_tag,#+"_MET_bins",#"",#"_JetHT_BeamHalo",#"_MinDPhi_and_Barrel_0p5",#"_B-F",#"_MinDPhi_and_Barrel_0p5",#"_MC",#"_vs_MC",#"_0p95",#""
    check_closure=clos,#True#False#True
    eta_cut = CUT_ETA,
    eta=DO_ETA,
    phi_cut = CUT_PHI,
    phi=DO_PHI,
    do_ratio = True,
    isMC = False
)

exit()
'''

