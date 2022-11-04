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
##gROOT.ProcessLine('.L %s/src/NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h+' % os.environ['CMSSW_BASE'])
##from ROOT import MEtType, JetType#LeptonType, JetType, FatJetType, MEtType, CandidateType, LorentzType
from collections import defaultdict, OrderedDict
from itertools import chain

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

ERA                = "2018"
REGION             = "SRpos"#"HBHE"#"ZtoEEBoost"#"WtoMN"#"WtoEN"
CUT                = "isSR"#"isSRCosmicVeto"#"isSRCosmic"#"isSRHBHE"#"isZtoEE"#"isWtoMN"#"isWtoEN"#"isJetMET_low_dPhi_MET_200_Lep"#"isJetHT_unprescaled"
KILL_QCD           = True#False
DO_ETA             = True
DO_PHI             = False#False#
if DO_PHI:
    DO_ETA = False
CUT_ETA            = True#True#True#False#True#True#False
CUT_PHI            = True

print "\n"
print "region: ", REGION
print "kill qcd: ", KILL_QCD
print "do eta: ", DO_ETA
print "do phi: ", DO_PHI
print "eta cut: ", CUT_ETA
print "phi cut: ", CUT_PHI
print "\n"

NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"

if REGION=="ZtoEE":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+REGION+"_v5_ntuples/"
if REGION=="ZtoMM":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+REGION+"_v5_ntuples/"
if REGION=="TtoEM":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_"+REGION+"_v5_ntuples_validate_timeRecHits/"

if REGION=="HBHEpos":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_positive_jets_HBHE/"
if REGION=="HBHEneg":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_negative_jets_HBHE/"
if REGION=="BHpos":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_positive_jets_BH/"
if REGION=="BHneg":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_negative_jets_BH/"
if REGION=="SRpos":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_positive_jets/"
    NTPL               = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_positive_jets/"
if REGION=="SRneg":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_negative_jets/"
    NTPL               = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_negative_jets/"

if REGION=="SR":
    NTUPLEDIR          = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"

PRE_PLOTDIR        = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"
PLOTDIR            = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#

YIELDDIR_BASE      = "plots/Yields_AN_fix/v6_calo_AOD_"+ERA+"_"
YIELDDIR           = "plots/Yields_AN_fix/v6_calo_AOD_"+ERA+"_"+REGION+"/"#"_2017_signal/"#
UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_fix/"

back = []
data = ["SingleMuon"]
data = ["EGamma"]
#data = ["SingleElectron"]
#data = [""]
data = ["HighMET"]
#data = ["HighMETHBHE"]
#data = ["MuonEG"]

sign = [
    'SUSY_mh127_ctau500',
    'SUSY_mh150_ctau500',
    'SUSY_mh175_ctau500',
    'SUSY_mh200_ctau500',
    'SUSY_mh250_ctau500',
    'SUSY_mh300_ctau500',
    'SUSY_mh400_ctau500',
    'SUSY_mh600_ctau500',
    'SUSY_mh800_ctau500',
    'SUSY_mh1000_ctau500',
    'SUSY_mh1250_ctau500',
    'SUSY_mh1500_ctau500',
    'SUSY_mh1800_ctau500',
    #'SUSY_mh127_ctau3000',
    #'SUSY_mh150_ctau3000',
    #'SUSY_mh175_ctau3000',
    #'SUSY_mh200_ctau3000',
    #'SUSY_mh250_ctau3000',
    #'SUSY_mh300_ctau3000',
    #'SUSY_mh400_ctau3000',
    #'SUSY_mh600_ctau3000',
    #'SUSY_mh800_ctau3000',
    #'SUSY_mh1000_ctau3000',
    #'SUSY_mh1250_ctau3000',
    #'SUSY_mh1500_ctau3000',
    #'SUSY_mh1800_ctau3000',
]

#sign = [
#    'SUSY_mh300_ctau500',
#    'SUSY_mh300_ctau3000',
#    'SUSY_mh1000_ctau500',
#    'SUSY_mh1000_ctau3000',
#]

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
########## ######## ##########

#gROOT.SetBatch(True)
dnn_bins = array('d', [0.,.00001,.0001,0.001,.01,.05,.1,.25,.5,.75,1.,])
less_bins = array('d', [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,10000])
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
less_bins_eta = array('d',[-1., -0.5, 0., 0.5, 1.,])
#less_bins_eta = array('d',[-1., 0., 1.,])

np_bins_eta = np.array(less_bins_eta)
np_bins_eta = np_bins_eta[0:-1]

maxeff = 0.15#0.0015#15#08#2#15

hist_den = {}
hist_num = {}
eff = {}
hist_den_cutbased = {}
hist_num_cutbased = {}
eff_cutbased = {}

def tau_weight_calc(llp_ct, new_ctau, old_ctau):
    '''
    llp_ct is a numpy array
    new_ctau and old_ctau are float
    '''
    source = np.exp(-1.0*llp_ct/old_ctau)/old_ctau**2
    weight = 1.0/new_ctau**2 * np.exp(-1.0*llp_ct/new_ctau)/source
    return weight

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = abs(p1 - p2)
    if res > math.pi:
        res -= 2*math.pi
    return res

def deltaR( e1, p1, e2, p2):
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return math.sqrt(de*de + dp*dp)

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

    if check_closure:
        dnn_threshold = 0.95#5
        dnn_max = 0.996
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        dnn_max = 1.
        print  "DNN threshold: ", dnn_threshold

    if eta_cut:
        print "Apply acceptance cut |eta|<1."

    if phi_cut:
        print "Apply acceptance cut phi"

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
        list_of_variables = ["isMC","JetsNegative.pt","JetsNegative.phi","JetsNegative.eta","JetsNegative.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","JetsNegative.pt","JetsNegative.phi","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRecHitsEB","JetsNegative.nRecHitsEB"]#"nLeptons"

        if "HBHE" in REGION:
            list_of_variables += ["Flag2_HBHEIsoNoiseFilter"]

        if CUT=="isSRHBHE" or CUT=="isSRCosmic" or CUT=="isSRCosmicVeto":
            list_of_variables += ["isSR"]
            list_of_variables += ["nCosmicMuons*","isDT_fit","dt_ecal_*"]
        else:
            list_of_variables += [CUT]

        if ERA=="2018" and "SR" in CUT:
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
                elif CUT == "isSRHBHE":
                    cut_mask = arrays["isSR"]>0
                    #cut_mask = np.logical_and(cut_mask , arrays["Flag2_HBHEIsoNoiseFilter"]==False )
                elif CUT == "isSRCosmic":
                    cut_mask = arrays["isSR"]>0
                    #enhance cosmics
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuonsOneLeg"]>0 )
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuons"]>1 )
                elif CUT == "isSRCosmicVeto":
                    cut_mask = arrays["isSR"]>0
                    #enhance cosmics
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuonsOneLeg"]>0 )
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuons"]>1 )
                else:
                    cut_mask = (arrays[CUT]>0)

                if "HBHE" in REGION:
                    print "enhance iso deposits"
                    cut_mask = np.logical_and(cut_mask , arrays["Flag2_HBHEIsoNoiseFilter"]==False )
                #HEM
                if (CUT == "isSR" or CUT=="isSRHBHE" or CUT=="isSRCosmic" or CUT=="isSRCosmicVeto" or CUT=="isSRBH") and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))

                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                #Jets --> JetsNegative
                #Default cut_jets, does nothing basically
                cut_jets = arrays["JetsNegative.pt"]>-999
                #now negative and positive will depend on the folder!
                #negative
                cut_jets = arrays["JetsNegative.timeRecHitsEB"]<=-1.
                #positive
                #cut_jets = arrays["JetsNegative.timeRecHitsEB"]>-1.
                #if CUT=="isSRCosmic" or CUT=="isSRCosmicVeto":
                #    cut_jets = np.logical_and(cut_jets,arrays["JetsNegative.nRecHitsEB"]>1)
                cut_jets = np.logical_and(cut_mask,cut_jets)
                cut_mask = (cut_jets.any()==True)

                if phi_cut==True and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_phi)#new

                if eta_cut==True and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_eta.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_eta)#new

                if phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)#new

                #Valid for all cuts
                if eta:
                    pt = arrays["JetsNegative.eta"][cut_jets][cut_mask]
                else:
                    if phi:
                        pt = arrays["JetsNegative.phi"][cut_jets][cut_mask]
                    else:
                        pt = arrays["JetsNegative.pt"][cut_jets][cut_mask]



                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]
                passed_den = np.concatenate( (passed_den, np.hstack(pt)) )
                w_den = np.concatenate( (w_den, np.hstack( pt.astype(bool)*weight ) ) )

                #print "As check, jet time"
                #print arrays["JetsNegative.timeRecHitsEB"][cut_jets][cut_mask]

                if CUT=="isSRCosmicVeto":
                    print "This must go at numerator!!"
                    #but remove cosmics causing tagged jets
                    cosmic_veto = arrays["dt_ecal_dist"]>1#0.5
                    cut_mask = np.logical_and(cut_mask,cosmic_veto)
                    cut_jets = np.logical_and(cut_jets,cut_mask)#new
                    pt = arrays["JetsNegative.eta"][cut_jets][cut_mask]

                sigprob = arrays["JetsNegative.sigprob"][cut_jets][cut_mask]
                tag_mask = np.logical_and(sigprob > dnn_threshold,sigprob <= dnn_max)#Awkward array mask, same shape as original AwkArr
                pt_tag = pt[tag_mask]
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]

                del arrays

                if pt_tag.shape[0]>0:
                    passed_num = np.concatenate( (passed_num, np.hstack(pt_tag)) )

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

        can = TCanvas("can","can", 1000, 800)
        can.cd()
        if i==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")
        #can.Print(PLOTDIR+"TagEff_"+s+add_label+".png")
        #can.Print(PLOTDIR+"TagEff_"+s+add_label+".pdf")

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

def draw_tag_eff_updated(sample_dict,reg_label,add_label="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False):
    isMC = False
    if sample_dict == back:
        isMC = True
    if sample_dict == sign:
        isMC = True
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1000, 800)
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    leg = TLegend(0.6, 0.7, 0.9, 0.9)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)

    if check_closure:
        dnn_threshold = 0.95#5
        dnn_max = 0.996
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        dnn_max = 1.
        print  "DNN threshold: ", dnn_threshold

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
    if phi:
        add_label+="_vs_phi"
    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
    #else:
    #    add_label+="_wp"+str(dnn_threshold).replace(".","p")


    print sample_dict
    print reg_label
    NEWDIR = PRE_PLOTDIR+reg_label+"/"
    if not os.path.isdir(NEWDIR): os.mkdir(NEWDIR)

    if eta==True:
        hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_eta)-1, more_bins_eta)
        hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_eta)-1, more_bins_eta)
        less_bins = less_bins_eta
    else:
        if phi==True:
            hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins_phi)-1, more_bins_phi)
            hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins_phi)-1, more_bins_phi)
            less_bins = less_bins_phi
        else:
            hden = TH1F(reg_label+"_den", reg_label+"_den", len(more_bins)-1, more_bins)
            hnum = TH1F(reg_label+"_num", reg_label+"_num", len(more_bins)-1, more_bins)
            less_bins = less_bins_pt

    for n,k in enumerate(sample_dict.keys()):
        print PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root"
        infiles = TFile(PRE_PLOTDIR+k+"/"+"TagEff_"+sample_dict[k]+add_label+".root", "READ")
        hist_den[k] = TH1F()
        hist_num[k] = TH1F()
        hist_den[k] = infiles.Get("den_"+sample_dict[k])
        hist_num[k] = infiles.Get("num_"+sample_dict[k])
        print hist_num[k].Print()
        print "now add if needed: ", n
        ##if n==0:
        ##    hnum = hist_num[k]
        ##    hden = hist_den[k]
        ##else:
        ##    print "do i crash here?"
        ##    exit()
        ##    print "hnum pre add ", hnum.Print()
        ##    hnum.Add(hist_num[k],1.)
        ##    print "hnum post add ", hnum.Print()
        ##    hden.Add(hist_den[k],1.)
        hnum.Add(hist_num[k],1.)
        hden.Add(hist_den[k],1.)
        
    graph = TGraphAsymmErrors()
    den = hden.Rebin(len(less_bins)-1,reg_label+"_den2",less_bins)
    num = hnum.Rebin(len(less_bins)-1,reg_label+"_num2",less_bins)
    graph.BayesDivide(num,den)
    eff = TEfficiency(num,den)
    #graph.BayesDivide(hnum,hden)
    #eff = TEfficiency(hnum,hden)
    eff.SetStatisticOption(TEfficiency.kBBayesian)
    eff.SetConfidenceLevel(0.68)
    graph.SetMarkerSize(1.3)
    graph.SetMarkerStyle(20)#(sign_sampl[s]['marker'])
    graph.SetMarkerColor(1)#(samples[s]['linecolor'])#(2)
    graph.SetFillColor(1)#(samples[s]['linecolor'])#(2) 
    graph.SetLineColor(1)#(samples[s]['linecolor'])#(2)
    graph.SetLineStyle(1)#(2)
    graph.SetLineWidth(2)
    graph.GetYaxis().SetRangeUser(-0.0001,0.1 if check_closure else maxeff)
    #graph.GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else 0.7)#Lisa for signal
    graph.GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
    graph.GetYaxis().SetTitleOffset(1.4)#("Efficiency (L1+HLT)")
    graph.GetYaxis().SetTitleSize(0.05)#DCMS
    if eta:
        #graph.GetXaxis().SetRangeUser(bins[0],bins[-1])
        graph.GetXaxis().SetTitle("Jet #eta")
    else:
        if phi:
            graph.GetXaxis().SetTitle("Jet #varphi")
        else:
            graph.GetXaxis().SetRangeUser(bins[2],bins[-1])
            graph.GetXaxis().SetTitle("Jet p_{T} (GeV)")
            can.SetLogx()#?
    graph.GetXaxis().SetTitleSize(0.04)
    graph.GetXaxis().SetTitleOffset(1.1)
    leg.AddEntry(graph, reg_label, "PL")
    can.SetGrid()
    graph.Draw("AP")


    outfile = TFile(NEWDIR+"TagTEfficiency_"+reg_label+add_label+".root","RECREATE")
    print "Info in <TFile::Write>: TEfficiency root file "+NEWDIR+"TagTEfficiency_"+reg_label+add_label+".root has been created"
    outfile.cd()
    graph.Write("eff_"+reg_label)
    eff.Write("TEff_"+reg_label)
    outfile.Close()

    leg.Draw()
    drawRegion(reg_label,left=True, left_marg_CMS=0.2, top=0.8)
    drawCMS_simple(LUMI, "Preliminary", onTop=True)
    can.Print(NEWDIR+"TagEff_"+reg_label+add_label+".png")
    can.Print(NEWDIR+"TagEff_"+reg_label+add_label+".pdf")

    outfile_name_check = NEWDIR+"TagEff_"+reg_label+add_label+".root"
    outfile_2 = TFile(outfile_name_check,"RECREATE")
    outfile_2.cd()
    eff.Write("eff_"+reg_label)
    hden.Write("den_"+reg_label)
    hnum.Write("num_"+reg_label)
    can.Write()
    print "Info in <TFile::Write>: root file "+outfile_name_check+" has been created"
    outfile_2.Close()


def draw_data_combination(era,regions,regions_labels=[],datasets=[],add_label="",lab_2="",check_closure=False,eta=False,phi=False,eta_cut=False,phi_cut=False,do_ratio=False,isMC=False):

    label_dict = {}
    label_dict["ZtoLL"] = "Z #rightarrow ll"
    label_dict["ZtoLLBoost"] = "Z #rightarrow ll, p_{T}^{Z}>100 GeV"
    label_dict["WtoLN"] = "W #rightarrow l#nu"
    label_dict["WtoLN_MET"] = "W #rightarrow l#nu + MET"
    label_dict["JetHT"] = "QCD"
    label_dict["TtoEM"] = "ttbar e + #mu"
    label_dict["HBHE"] = "HB HE noise"
    label_dict["HBHEpos"] = "HB HE noise, t > -1 ns"
    label_dict["HBHEneg"] = "HB HE noise, t #leq -1 ns" 
    label_dict["SR"] = "SR"
    label_dict["SRpos"] = "SR, t > -1 ns"
    label_dict["SRneg"] = "SR, t #leq -1 ns" 
    label_dict["BH"] = "Beam halo"
    label_dict["BHpos"] = "Beam halo, t > -1 ns"
    label_dict["BHneg"] = "Beam halo, t #leq -1 ns" 

    BASEDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_"+era+"_"
    OUTDIR  = "plots/Efficiency_AN_fix/v5_calo_AOD_"+era+"_combination/"
    infiles = {}
    graph =  {}
    eff   =  {}
    hist_num = {}
    hist_den = {}
    can = TCanvas("can","can", 1800, 800)#very horizontal
    can = TCanvas("can","can", 800, 800)#squared
    #can.SetRightMargin(0.1)
    can.SetLeftMargin(0.15)
    can.cd()
    #leg = TLegend(0.2, 0.6, 0.9, 0.9)#y: 0.6, 0.9

    max_len = max(len(r) for r in regions_labels )
    if eta:
        leg = TLegend(0.7-max_len/200., 0.7-0.07, 1.0-0.05, 1.0-0.07)
    else:
        leg = TLegend(0.7, 0.7-0.07, 1.0-0.05, 1.0-0.07)#(0.6, 0.7, 1.0, 1.0)
    leg.SetTextSize(0.035)#very horizontal
    leg.SetTextSize(0.025)#squared
    leg.SetTextSize(0.03)#squared
    #leg.SetBorderSize(0)
    #leg.SetFillStyle(0)
    #leg.SetFillColor(0)

    if eta_cut==True:
        add_label+="_eta_1p0"
    if phi_cut==True:
        add_label+="_phi_cut"
    if eta:
        add_label+="_vs_eta"
        less_bins_plot = less_bins_eta
    else:
        can.SetLogx()#?
        less_bins_plot = less_bins_pt


    if check_closure:
        dnn_threshold = 0.95#5
        dnn_max = 0.996
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        dnn_max = 1.
        print  "DNN threshold: ", dnn_threshold

    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")
    
    for i, r in enumerate(regions):
        if isMC:
            if r=="ZtoMM" or r=="ZtoEE" or r=="ZtoLL":
                s = "DYJetsToLL"
            elif r=="WtoMN" or r=="WtoEN" or r=="WtoLN":
                s = "WJetsToLNu"
            elif r=="TtoEM":
                s = "TTbarGenMET"
            elif r=="JetHT":
                s = "QCD"
        else:
            if r=="ZtoMM" or r=="WtoMN" or r=="WtoMN_MET" or r=="MN" or r=="WtoMN_noMT" or r=="ZtoMMBoost":
                s = "SingleMuon"
            elif r=="ZtoEE" or r=="WtoEN" or r=="WtoEN_MET" or r=="EN" or r=="WtoEN_noMT" or r=="ZtoEEBoost":
                if era=="2018":
                    s = "EGamma"
                else:
                    s = "SingleElectron"
            elif r=="TtoEM":
                s = "MuonEG"
            elif r=="JetHT":
                s = "JetHT"
            elif r=="HBHE":
                s = "HighMETHBHE"
            elif r=="HBHEpos":
                s = "HighMET"
            elif r=="HBHEneg":
                s = "HighMET"
            elif r=="SR":
                s = "HighMET"
            elif r=="SRpos":
                s = "HighMET"
            elif r=="SRneg":
                s = "HighMET"
            elif r=="BH":
                s = "HighMETBH"
            elif r=="BHpos":
                s = "HighMETBH"
            elif r=="BHneg":
                s = "HighMETBH"
            elif "ZtoLL" in r or "WtoLN" in r or "ZtoLLBoost" in r:
                s=r
            else:
                print "Invalid region, aborting..."
                exit()
        if len(datasets)>0:
            if datasets[i]!="":
                s = datasets[i]
        INPDIR  = BASEDIR + r + "/"#"_CR/"
        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
        
        colors_2016 = colors#[1,4,801,856] + [881, 798, 602, 921]

        print "Opening this file: ", INPDIR+"TagEff_"+s+reg_label+add_label+".root"
        infile = TFile(INPDIR+"TagEff_"+s+reg_label+add_label+".root", "READ")
        hist_den[r+reg_label] = TH1F()
        hist_num[r+reg_label] = TH1F()
        graph[r+reg_label] = TGraphAsymmErrors()
        hist_den[r+reg_label] = infile.Get("den_"+s)
        hist_num[r+reg_label] = infile.Get("num_"+s)
        #rebin
        #less_bins = bins#!
        den = hist_den[r+reg_label].Rebin(len(less_bins_plot)-1,r+reg_label+"_den2",less_bins_plot)
        num = hist_num[r+reg_label].Rebin(len(less_bins_plot)-1,r+reg_label+"_num2",less_bins_plot)
        graph[r+reg_label].BayesDivide(num,den)
        eff[r+reg_label] = TEfficiency(num,den)
        eff[r+reg_label].SetStatisticOption(TEfficiency.kBBayesian)
        eff[r+reg_label].SetConfidenceLevel(0.68)
        graph[r+reg_label].SetMarkerSize(marker_sizes[i])#(1.3)
        graph[r+reg_label].SetMarkerStyle(markers[i])#(21)#(sign_sampl[s]['marker'])
        if ERA=="2016" and "B-F" in reg_label:
            graph[r+reg_label].SetMarkerColor(colors_2016[i])#(samples[s]['fillcolor'])#(2)
            graph[r+reg_label].SetFillColor(colors_2016[i])#(samples[s]['fillcolor'])#(2) 
            graph[r+reg_label].SetLineColor(colors_2016[i])#(samples[s]['linecolor'])#(2)
        else:
            graph[r+reg_label].SetMarkerColor(colors[i])#(samples[s]['fillcolor'])#(2)
            graph[r+reg_label].SetFillColor(colors[i])#(samples[s]['fillcolor'])#(2) 
            graph[r+reg_label].SetLineColor(colors[i])#(samples[s]['linecolor'])#(2)
        graph[r+reg_label].SetLineStyle(lines[i])#(2)#(2)
        graph[r+reg_label].SetLineWidth(2)
        graph[r+reg_label].GetYaxis().SetRangeUser(-0.0001,0.8 if check_closure else maxeff)#good for hbhe
        graph[r+reg_label].GetYaxis().SetRangeUser(-0.0001,0.01 if check_closure else maxeff)
        graph[r+reg_label].GetYaxis().SetTitle("Mis-tag efficiency")#("Efficiency (L1+HLT)")
        graph[r+reg_label].GetYaxis().SetTitleOffset(1.5)#("Efficiency (L1+HLT)")
        graph[r+reg_label].GetYaxis().SetTitleSize(0.05)#DCMS
        if eta:
            graph[r+reg_label].GetXaxis().SetTitle("Jet #eta")
        else:
            graph[r+reg_label].GetXaxis().SetRangeUser(bins[4],bins[-1])
            graph[r+reg_label].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        graph[r+reg_label].GetXaxis().SetTitleSize(0.04)
        graph[r+reg_label].GetXaxis().SetTitleOffset(1.1)
        #leg.AddEntry(graph[r+reg_label], samples[s]['label']+"; "+r+reg_label, "PL")
        #leg.AddEntry(graph[r+reg_label], samples[s]['label']+"; "+r, "PL")
        leg.AddEntry(graph[r+reg_label], label_dict[r], "PL")
        can.SetGrid()
        if i==0:
            graph[r+reg_label].Draw("AP")
            #graph[s].Draw("P")#?
        else:
            graph[r+reg_label].Draw("P,sames")
        infile.Close()

    leg.Draw()
    #drawRegion(REGION,left=True, left_marg_CMS=0.2, top=0.8)
    era_lab = ERA
    if "B-F" in reg_label:
        era_lab+= " B-F"
    elif "G-H" in reg_label:
        era_lab+= " G-H"
    drawCMS_simple(LUMI, "Preliminary", ERA=era_lab, onTop=True)
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".png")
    can.Print(OUTDIR+"TagEffCombiData_"+era+add_label+lab_2+".pdf")
    #can.Print(OUTDIR+"buh.pdf")

    #for r in regions:
    #    print r
    #    #print eff[r].Print()
   
    if do_ratio:
        keys = graph.keys()
        ratio = {}
        h_den_r = {}
        h_num_r = {}
        basis = ""
        for k in keys:
            if "WtoLN" in k:
                basis = k
                np = graph[basis].GetN()
                print np

        print basis
        colorsr = [1,4,418]
        markersr = [20,25,24,20,24,24,24,24]
        new_den = {}
        new_num = {}
        n=0

        for k in [basis]:
            print "Doing: ",k
            r = k.replace(reg_label,"")
            INPDIR  = BASEDIR + r + "/"
            print INPDIR+"TagEff_"+k+add_label+".root"
            print "Opening this file: ", INPDIR+"TagEff_"+k+add_label+".root"
            infile = TFile(INPDIR+"TagEff_"+k+add_label+".root", "READ")
            pre_db = infile.Get("den_"+r)
            pre_nb = infile.Get("num_"+r)
            db = pre_db.Rebin(len(less_bins_plot)-1,"db",less_bins_plot)
            nb = pre_nb.Rebin(len(less_bins_plot)-1,"nb",less_bins_plot)
            db.SetDirectory(0)
            nb.SetDirectory(0)
        
        for k in keys:
            if "WtoLN" not in k:
                ratio[k] = TH1F()#TGraphAsymmErrors()
                h_den_r[k] = TH1F(k+"_den", "", len(less_bins_plot)-1, less_bins_plot)
                h_num_r[k] = TH1F(k+"_num", "", len(less_bins_plot)-1, less_bins_plot)
                h_den_r[k].Sumw2()
                h_num_r[k].Sumw2()

                print "Doing: ",k
                r = k.replace(reg_label,"")
                INPDIR  = BASEDIR + r + "/"
                print INPDIR+"TagEff_"+k+add_label+".root"
                print "Opening this file: ", INPDIR+"TagEff_"+k+add_label+".root"
                infile = TFile(INPDIR+"TagEff_"+k+add_label+".root", "READ")
                pre_dk = infile.Get("den_"+r)
                pre_nk = infile.Get("num_"+r)
                dk = pre_dk.Rebin(len(less_bins_plot)-1,"dk",less_bins_plot)
                nk = pre_nk.Rebin(len(less_bins_plot)-1,"nk",less_bins_plot)
                dk.SetDirectory(0)
                nk.SetDirectory(0)
                #we should combine these!!
                #(nk*db)/(dk*nb)
                ratio[k] = nk.Clone("r")
                ratio[k].Divide(dk)
                ratio[k].Divide(nb)
                ratio[k].Multiply(db)
                ratio[k].SetDirectory(0)
                if "ZtoLL" in k:
                    ratio[k].SetLineColor(1)
                    ratio[k].SetLineStyle(2)
                    ratio[k].SetMarkerColor(1)
                    ratio[k].SetMarkerStyle(20)
                elif "JetHT" in k:
                    ratio[k].SetLineColor(418)
                    ratio[k].SetLineStyle(2)
                    ratio[k].SetMarkerColor(418)
                    ratio[k].SetMarkerStyle(25)
                elif "TtoEM" in k:
                    ratio[k].SetLineColor(4)
                    ratio[k].SetLineStyle(2)
                    ratio[k].SetMarkerColor(4)
                    ratio[k].SetMarkerStyle(24)
                ratio[k].SetMarkerSize(marker_sizes[n])
                ratio[k].SetLineWidth(2)
                n+=1

        canr = TCanvas("canr","canr", 800, 800)#squared
        #canr.SetRightMargin(0.1)
        canr.SetLeftMargin(0.15)
        canr.cd()
        canr.SetGrid()
        if eta:
            legr = TLegend(0.7, 0.7, 1.0-0.05, 1.0-0.07)
        else:
            legr = TLegend(0.6, 0.7, 1.0, 1.0)
        legr.SetTextSize(0.03)#squared

        n=0
        for k in keys:
            if "WtoLN" not in k:
                print ratio[k]
                legr.AddEntry(ratio[k],label_dict[ k.split("_")[0] ],"PL")
                if not eta:
                    ratio[k].GetXaxis().SetRangeUser(bins[4],bins[-1])
                    canr.SetLogx()
                else:
                    ratio[k].GetXaxis().SetRangeUser(-1.1,1.1)                
                ratio[k].SetTitle("")
                if n==0:
                    ratio[k].Draw("P")
                    #ratio[k].Draw("AP")
                    ratio[k].SetMinimum(0.)
                    ratio[k].SetMaximum(2.)
                else:
                    ratio[k].Draw("P,sames")
                #ratio[k].GetYaxis().SetRangeUser(-0.0001,0.002 if check_closure else maxeff)
                n+=1

        legr.Draw()
        lineX = TLine(-1,1,1,1)
        lineX.SetLineStyle(2)
        lineX.SetLineWidth(3)
        lineX.SetLineColor(2)
        lineX.Draw("same")
        drawCMS_simple(LUMI, "Preliminary", ERA=era_lab, onTop=True)
        canr.Print(OUTDIR+"RatioTagEffCombiData_"+era+add_label+lab_2+".png")
        canr.Print(OUTDIR+"RatioTagEffCombiData_"+era+add_label+lab_2+".pdf")

def get_tree_weights(sample_list,LUMI):
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


def get_tree_weights_BR_scan(sample_list,br_h,dataset_label="",main_pred_sample="HighMET"):

    #Fix this_lumi
    if ERA=="2016":
        if dataset_label == "_G-H":
            this_lumi  = lumi[ main_pred_sample ]["G"]+lumi[ main_pred_sample ]["H"]#["tot"]
        elif dataset_label == "_B-F":
            this_lumi  = lumi[ main_pred_sample ]["B"]+lumi[ main_pred_sample ]["C"]+lumi[ main_pred_sample ]["D"]+lumi[ main_pred_sample ]["E"]+lumi[ main_pred_sample ]["F"]#["tot"]
    else:
        this_lumi  = lumi[ main_pred_sample ]["tot"]

    def eq(H,Z):
        return H**2 + 2*H*Z + Z**2

    def retZ(H):
        #x**2 + 2*H*x + (H**2 - 1.) = 0
        return -H + 1

    h = br_h/100. if br_h>1 else br_h
    z = retZ(float(h))
    w_hh = h*h
    w_hz = 2*h*z
    w_zz = z*z
    print "\n"
    print "------------------------"
    print "BR: H(", h*100,"%), Z(", z*100,"%)"
    print ("Decay composition: HH %.2f, HZ %.2f, ZZ %.2f" % (w_hh*100, w_hz*100, w_zz*100))
    print "------------------------"

    tree_w_dict = defaultdict(dict)
    for i, pr in enumerate(sample_list):
        new_list = [pr+"_HH",pr+"_HZ",pr+"_ZZ"]
        for s in new_list:
            #print s
            for l, ss in enumerate(samples[s]['files']):
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                #Tree weight
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    #print "SUSY central, consider sample dictionary for nevents!"
                    nevents = sample[ss]['nevents']
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                filename.Close()
                if('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    #print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    #print "But consider BR!"
                    xs = 1.
                    xs *= sample[ss]['BR']
                print "LUMI ", this_lumi
                print "xs ", xs
                print "nevents ", nevents
                t_w = this_lumi * xs / nevents
                if(b_skipTrain>0):
                    if(n_odd>0):
                        t_w *= float(n_pass/n_odd)
                if "_HH" in s:
                    t_w *= w_hh
                if "_HZ" in s:
                    t_w *= w_hz
                if "_ZZ" in s:
                    t_w *= w_zz
                print("%s has tree weight %f")%(ss,t_w)
                tree_w_dict[pr][ss] = t_w

    print "\n"
    return tree_w_dict
    
def background_prediction(tree_weight_dict,sample_list,extr_regions=[],regions_labels=[],datasets=[],add_label="",label_2="",check_closure=False,plot_distr="",eta=False,phi=False,eta_cut=False,phi_cut=False):

    print "\n"
    print "**************************"
    print "** Background prediction**"
    print "**************************"
    print "\n"

    mult_factor = 1.
    if sample_list==sign:
        print "\n"
        print "   ** Performing estimation on signal! Rescaling signal by signalMultFactor ", signalMultFactor, " **"
        mult_factor = signalMultFactor
        print "\n"

    if check_closure:
        dnn_threshold = 0.95#5
        dnn_max = 0.996
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        dnn_max = 1.
        print  "DNN threshold: ", dnn_threshold

    if eta_cut:
        print "Apply acceptance cut |eta|<1."
        #print "Something fishy, to be debugged" --> probably binning problem, 1 out of the range
        #exit()

    if eta_cut==True:
        add_label+="_eta_1p0"

    if phi_cut==True:
        add_label+="_phi_cut"

    if eta:
        add_label+="_vs_eta"
        if phi:
            add_label+="_vs_phi"

    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")

    print "Reading input root files: ", NTUPLEDIR

    #Region-dependent  objects
    EFFDIR = {}
    TEff = {}
    eff = {}
    effUp = {}
    effDown = {}
    table_yield = {}
    table_pred = {}
    table_pred_new = {}
    table_integral = {}
    tag_bins = np.array([0,1,2,3,4,5,6])
    uproot_tree = OrderedDict()#defaultdict(dict)
    chain = {}
    hist = {}
    dfR = {}
    h0 = {}
    h1 = {}
    h2 = {}
    results = defaultdict(dict)
    row = {}
    rowP = {}
    rowPn = {}
    infiles = {}

    if extr_regions==[]:
        extr_regions.append(REGION)

    for i,r in enumerate(extr_regions):
        reg_label = ""
        dset = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]
        results[r+reg_label+dset] = {}
        EFFDIR[r+reg_label+dset] = "plots/Efficiency_AN_fix/v5_calo_AOD_"+ERA+"_"+ r + "/"
        table_yield[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err', 'Bin 2 Yield', 'Bin 2 Err', 'Bin 2 Err combine'])
        table_pred[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 1 Pred', 'Bin 1 Pred Err', 'Bin 2 Pred', 'Bin 2 Pred Err', 'Bin 2 Pred from 1', 'Bin 2 Pred from 1 Err'])
        table_pred_new[r+reg_label+dset] =  PrettyTable(['Bin 1 Pred', 'Bin 1 Discr. %', 'Bin 2 Pred from 0', 'Bin 2 from 0 Discr. %', 'Bin 2 Pred from 1', 'Bin 2 from 1 Discr. %', 'Bin 2 Pred Stat Err', 'Bin 1 Pred Stat Err'])
        table_integral[r+reg_label+dset] =  PrettyTable(['Sample', 'Bin 0 Yield', 'Bin 0 Err', 'Bin 1 Yield', 'Bin 1 Err'])

    for i,r in enumerate(extr_regions):
        if "ZtoMM" in r or "WtoMN" in r or "MN" in r:
            eff_name="SingleMuon"
        elif "ZtoEE" in r or "WtoEN" in r or "EN" in r:
            if ERA=="2018":
                eff_name="EGamma"
            else:
                eff_name="SingleElectron"
        elif "TtoEM" in r:
            eff_name=r
        elif "SR" in r:
            eff_name=r
        elif "HBHE" in r:
            eff_name=r
        elif "JetHT" in r:
            eff_name="JetHT"
        elif "ZtoLL" in r or "WtoLN" in r:
            eff_name=r
        else:
            eff_name = r

        #HERE?
        if len(datasets)>0:
            if datasets[i]!="":
                eff_name = datasets[i]
                dset = datasets[i]
            else:
                #Otherwise it stores the one from before, wrong keys
                dset = ""

        reg_label = ""
        if len(regions_labels)>0:
            reg_label = regions_labels[i]

        infiles[r+reg_label+dset] = TFile(EFFDIR[r+reg_label+dset]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root", "READ")
        print "Opening TEff["+r+reg_label+dset+"]  " + EFFDIR[r+reg_label+dset]+"TagTEfficiency_"+eff_name+reg_label+add_label+".root"
        TEff[r+reg_label+dset] = infiles[r+reg_label+dset].Get("TEff_"+eff_name)
        for s in sample_list:
            #Define dictionaries
            results[r+reg_label+dset][s] = {}
            
    for i, s in enumerate(sample_list):
        ##Define TH1F as a cross-check for event yields
        h0[s] = TH1F(s+"_0", s+"_0", 4, -1, 2)
        h0[s].Sumw2()
        h1[s] = TH1F(s+"_1", s+"_1", 4, -1, 2)
        h1[s].Sumw2()
        h2[s] = TH1F(s+"_2", s+"_2", 4, -1, 2)
        h2[s].Sumw2()

        print s

        ##Prepare TChain and determine tree weights
        ##Note: if you open the same file twice, project doesn't work
        ##Tree weight  must be determined in a separate function
        tree_weights = {}
        tree_weights_array = np.array([])
        chain_entries = {}
        chain_entries_cumulative = {}
        chain[s] = TChain("tree")
        list_files_for_uproot = []


        start_uproot = time.time()
        array_size_tot = 0
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.phi","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","HLT*PFMETNoMu*","Jets.timeRecHitsEB","Jets.cHadEFrac","Jets.nTrackConstituents"]#"nLeptons"

        list_of_variables += ["Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRecHitsEB","Jets.nRecHitsEB","Jets.cHadEFrac","Jets.nTrackConstituents"]

        if CUT=="isSRHBHE" or CUT=="isSRCosmic" or CUT=="isSRCosmicVeto":
            list_of_variables += ["isSR"]
            list_of_variables += ["nCosmicMuons*","isDT_fit","dt_ecal_*"]
        else:
            list_of_variables += [CUT]

        if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR and CUT=="isSR":
            list_of_variables += ["dt_ecal_*","min_dPhi*"]
        if ERA=="2018" and "SR" in REGION:
            list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta"]

        #The only thing we can afford in RAM are final numbers
        #hence only a few arrays
        b0 = np.array([])
        b1 = np.array([])
        b2 = np.array([])
        pr_1 = {}
        pr_2 = {}
        pr_2_from_1 = {}

        #pr_1/2 depend on TEff, dictionaries
        for r in TEff.keys():
            pr_1[r] = np.array([])#[]
            pr_2[r] = np.array([])#[]
            pr_2_from_1[r] = np.array([])#[]


            eff[r] = []
            effUp[r] = []
            effDown[r] = []
            #0. Fill efficiencies np arrays once and forever
            if eta:
                for b in np_bins_eta:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    print(r," eta: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))

            elif eta==False and phi==True:
                for b in np_bins_phi:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    print(r," phi: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))

            else:
                for b in np_bins:
                    binN = TEff[r].GetPassedHistogram().FindBin(b)
                    #print "Making fake pt bins!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    eff[r].append(TEff[r].GetEfficiency(binN))
                    #eff[r].append(0.0005)
                    print(r," pt: ",  b, "; eff: ", TEff[r].GetEfficiency(binN))
                    effUp[r].append(TEff[r].GetEfficiencyErrorUp(binN))
                    effDown[r].append(TEff[r].GetEfficiencyErrorLow(binN))
                    
            print r, eff[r]
            #exit()

        #b0 etc filled at every chunk
        for l, ss in enumerate(samples[s]['files']):
            tree_weights[l] = tree_weight_dict[s][ss]
            chain[s].Add(NTUPLEDIR + ss + ".root")
            #print "Entries in ", ss, " : ", chain[s].GetEntries()
            chain_entries_cumulative[l] = chain[s].GetEntries()
            if l==0:
                chain_entries[l]=chain[s].GetEntries()
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
                if CUT == "isSR" or CUT=="isSR" or CUT=="isSRpos":
                    cut_mask = arrays["isSR"]>0
                    if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                        #cosmic
                        print "With cosmic veto!"
                        #cosmic_veto = arrays["dt_ecal_dist"]<0.5
                        #cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
                elif CUT == "isSRCosmic":
                    cut_mask = arrays["isSR"]>0
                    #enhance cosmics
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuonsOneLeg"]>0 )
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuons"]>1 )
                elif CUT == "isSRHBHE":
                    cut_mask = arrays["isSR"]>0
                elif CUT == "isSRCosmicVeto":
                    cut_mask = arrays["isSR"]>0
                    #enhance cosmics
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuonsOneLeg"]>0 )
                    cut_mask = np.logical_and(cut_mask , arrays["nCosmicMuons"]>1 )
                    #but remove cosmics causing tagged jets
                    cosmic_veto = arrays["dt_ecal_dist"]<0.5
                    cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
                else:
                    cut_mask = (arrays[CUT]>0)

                #HEM
                if (CUT=="isSR" or CUT=="isSRCosmic" or CUT=="isSRCosmicVeto" or CUT=="isSRHBHE" or CUT=="isSRBH") and str(ERA)=="2018":
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))


                if KILL_QCD:
                    #print "   QCD killer cut!"
                    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)


                #Default cut_jets, does nothing basically
                #Jets --> JetsNegative
                cut_jets = arrays["Jets.pt"]>-999
                #now negative and positive will depend on the folder!
                #negative
                #cut_jets = arrays["JetsNegative.timeRecHitsEB"]<=-1.
                #positive
                #cut_jets = arrays["JetsNegative.timeRecHitsEB"]>-1.
                cut_jets = np.logical_and(cut_mask,cut_jets)
                cut_mask = (cut_jets.any()==True)

                if phi_cut==True and eta_cut==False:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_phi = np.logical_and(cut_mask,cut_mask_phi)
                    cut_mask = (cut_mask_phi.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_phi)#new

                elif eta_cut==True and phi_cut==False:
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                    cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_eta.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_eta)#new

                elif phi_cut and eta_cut:
                    cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)                    
                    cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                    cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                    #This is needed to guarantee the shapes are consistent
                    cut_mask = (cut_mask_phi_eta.any()==True)
                    cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)#new


                #Beam Halo veto
                if CUT == "isSR":
                    if "v6_calo_AOD" in NTUPLEDIR and "v5_ntuples" not in NTUPLEDIR:
                        #beam halo
                        cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_"+str(dnn_threshold).replace(".","p")]<0.1#05
                        cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>dnn_threshold , arrays["Jets.nRecHitsEB"]<=10)
                        cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
                        cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)
                        cut_mask = np.logical_and(cut_mask,cut_mask_bh)
                        cut_jets = np.logical_and(cut_jets,cut_mask)

                ##Fill pt and sigprob arrays #new
                if eta:
                    pt = arrays["Jets.eta"][cut_jets][cut_mask]#new
                else:
                    if phi:
                        pt = arrays["Jets.phi"][cut_jets][cut_mask]#new
                    else:
                        pt = arrays["Jets.pt"][cut_jets][cut_mask]#new
                sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]#new

                eventweight = arrays["EventWeight"][cut_mask]
                runnumber = arrays["RunNumber"][cut_mask]
                luminumber = arrays["LumiNumber"][cut_mask]
                eventnumber = arrays["EventNumber"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                dt_ecal = arrays["dt_ecal_dist"][cut_mask]#new
                dt_ecal_acc = arrays["dt_ecal_acc_dist"][cut_mask]#new
                pt_v = arrays["Jets.pt"][cut_jets][cut_mask]#new
                eta_v = arrays["Jets.eta"][cut_jets][cut_mask]#new
                phi_v = arrays["Jets.phi"][cut_jets][cut_mask]#new
                score_v = arrays["Jets.sigprob"][cut_jets][cut_mask]#new
                time_v = arrays["Jets.timeRecHitsEB"][cut_jets][cut_mask]#new
                n_v = arrays["Jets.nRecHitsEB"][cut_jets][cut_mask]#new
                nt_v = arrays["Jets.nTrackConstituents"][cut_jets][cut_mask]#new
                cHadEFrac_v = arrays["Jets.cHadEFrac"][cut_jets][cut_mask]#new
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict[s][ss]*mult_factor
                #Default value, we'll need more masks afterwards
                n_obj = -1


                del arrays
                
                #dnn_threshold = 0.996
                tag_mask = np.logical_and(sigprob > dnn_threshold,sigprob <= dnn_max)#Awkward array mask, same shape as original AwkArr
                untag_mask = (sigprob <= dnn_threshold)
                pt_untag = pt[untag_mask]
                #test  = (tag_mask.any() == True)#at least one tag
                
                bin0_m = (sigprob[tag_mask].counts ==0)
                bin1_m = (sigprob[tag_mask].counts ==1)
                bin2_m = (sigprob[tag_mask].counts >1)
                dt_ecal = dt_ecal[bin2_m]
                dt_ecal_acc = dt_ecal_acc[bin2_m]
                runnumber = runnumber[bin2_m]
                luminumber = luminumber[bin2_m]
                eventnumber = eventnumber[bin2_m]
                pt_v = pt_v[bin2_m]
                eta_v = eta_v[bin2_m]
                phi_v = phi_v[bin2_m]
                score_v = score_v[bin2_m]
                n_v = n_v[bin2_m]
                nt_v = nt_v[bin2_m]
                time_v = time_v[bin2_m]
                cHadEFrac_v = cHadEFrac_v[bin2_m]
                if REGION=="ZtoMMBoost" or REGION=="ZtoEEBoost":
                    zpt = zpt[bin2_m]
                bin0 = np.multiply(bin0_m,weight)
                bin1 = np.multiply(bin1_m,weight)
                bin2 = np.multiply(bin2_m,weight)

                #Second: distribute per bin

                print "events in bin 2:"
                print "RunNumber: ", runnumber
                print "LumiNumber: ", luminumber
                print "EventNumber: ", eventnumber
                print "dt ecal dist: ", dt_ecal
                print "dt ecal acc dist: ", dt_ecal_acc
                print "pt: ", pt_v
                print "eta: ", eta_v
                print "phi: ", phi_v
                print "nt: ", nt_v
                print "time: ", time_v
                print "n rec hits: ", n_v
                print "score: ", score_v
                print "cHadEFrac: ", cHadEFrac_v
                #Predictions
                bin1_pred = defaultdict(dict)
                bin1_pred_up = defaultdict(dict)
                bin1_pred_low = defaultdict(dict)
                bin2_pred = defaultdict(dict)
                bin2_pred_up = defaultdict(dict)
                bin2_pred_low = defaultdict(dict)
                bin2_pred_from_1 = defaultdict(dict)
                bin2_pred_from_1_up = defaultdict(dict)
                bin2_pred_from_1_low = defaultdict(dict)
                p1 = {}
                p1Up = {}
                p1Down = {}
                p2 = {}
                p2Up = {}
                p2Down = {}
                eff1 = {}
                errUp1 = {}
                errDown1 = {}
                eff2 = {}
                errUp2 = {}
                errDown2 = {}

                BEST = True

                for r in TEff.keys():
                    #per-file, reset before any loop
                    bin1_pred[r] = np.array([]) if BEST else []
                    bin2_pred[r] = np.array([]) if BEST else []
                    bin2_pred_from_1[r] = np.array([]) if BEST else []

                
                ####
                ####  Best method
                ####
                prob_vec = {}

                for r in TEff.keys():
                    prob_vec[r] = []
                    if eta:
                        for i in range(len(np_bins_eta)):
                            #print "Here there is something wrong, bins are not being skipped...."
                            if i<len(np_bins_eta)-1:
                                prob_vec[r].append(np.logical_and(pt_untag>=np_bins_eta[i],pt_untag<np_bins_eta[i+1])*eff[r][i])#*weight)
                            else:
                                prob_vec[r].append((pt_untag>=np_bins_eta[i])*eff[r][i])#*weight)

                    else:
                        if phi:
                            for i in range(len(np_bins_phi)):
                                if i<len(np_bins_phi)-1:
                                    prob_vec[r].append(np.logical_and(pt_untag>=np_bins_phi[i],pt_untag<np_bins_phi[i+1])*eff[r][i])#*weight)
                                else:
                                    prob_vec[r].append((pt_untag>=np_bins_phi[i])*eff[r][i])#*weight)
                        else:
                            for i in range(len(np_bins)):
                                if i<len(np_bins)-1:
                                    prob_vec[r].append(np.logical_and(pt_untag>=np_bins[i],pt_untag<np_bins[i+1])*eff[r][i])#*weight)
                                else:
                                    prob_vec[r].append((pt_untag>=np_bins[i])*eff[r][i])#*weight)

                    #print "prob_vec after all bins: ", prob_vec[r]
                    prob_tot = sum(prob_vec[r])
                    #print "prob_tot: ", prob_tot
                    somma = (prob_tot*weight).sum()
                    cho = prob_tot.choose(2)
                    combi = cho.unzip()[0] * cho.unzip()[1] * weight

                    bin1_pred[r] = np.concatenate( (bin1_pred[r],somma) )
                    bin2_pred[r] = np.concatenate( (bin2_pred[r],combi.sum()) )
                    bin2_pred_from_1[r] = np.concatenate( (bin2_pred_from_1[r], somma[bin1_m]  )  )#.append(0.)
                    
                #Here: still in chunk loop
                #Here concatenate to full pred
                for r in TEff.keys():
                    pr_1[r] = np.concatenate((pr_1[r],bin1_pred[r]))
                    pr_2[r] = np.concatenate((pr_2[r],bin2_pred[r]))
                    pr_2_from_1[r] = np.concatenate((pr_2_from_1[r],bin2_pred_from_1[r]))


                b0 = np.concatenate((b0,bin0))
                b1 = np.concatenate((b1,bin1))
                b2 = np.concatenate((b2,bin2))

                #del tmp
                en_it = time.time()
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"
                    #print "uproot iteration n. ", c,"/",n_iter, " took ", en_it-st_it
                    c+=1
                    #new#gc.collect()

            del gen

        end_uproot = time.time()
        print "\n"
        print "   --- > Tot size of arrays: ", array_size_tot
        print "Size of tree_weights_array: ", len(tree_weights_array)
        print "Time elapsed to fill uproot array: ", end_uproot-start_uproot
        print "************************************"
        print "\n"

        start_chain = time.time()
        
        print("Entries in chain: %d")%(chain[s].GetEntries())
        max_n=chain[s].GetEntries()+10#100000

        ev_weight  = "EventWeight*PUReWeight"
        tagvar = "nTagJets_0p996_JJ"
        print tagvar
        cutstring_0 = ev_weight+"*("+tagvar+"==0)"
        cutstring_1 = ev_weight+"*("+tagvar+"==1)"
        cutstring_2 = ev_weight+"*("+tagvar+">1)"

        #combination single jet
        if CUT == "isJetHT":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MEt.pt<25 && (HLT_PFJet140_v))"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MEt.pt<25 && (HLT_PFJet140_v))"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MEt.pt<25 && (HLT_PFJet140_v))"

        if CUT == "isDiJetMET":
            cutstring_0 = ev_weight+"*("+tagvar+"==0 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_1 = ev_weight+"*("+tagvar+"==1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"
            cutstring_2 = ev_weight+"*("+tagvar+">1 && MinLeadingJetMetDPhi<=0.4 && MEt.pt<100 && nCHSJetsAcceptanceCalo==2 && HLT_PFJet500_v)"


        print "CUT: ", CUT
        print "cutstring bin 0: ", cutstring_0
        chain[s].Project(s+"_0", "isMC",cutstring_0,"")#,max_n)
        chain[s].Project(s+"_1", "isMC",cutstring_1,"",max_n)
        chain[s].Project(s+"_2", "isMC",cutstring_2,"",max_n)
        end_chain = time.time()
        print "Time elapsed to project TChain: ", end_chain-start_chain
        print "************************************"
        print "h0 entries: ", h0[s].GetEntries()
        print "h0 integral dummy: ", h0[s].Integral()
        print "h1 entries: ", h1[s].GetEntries()
        print "h1 integral dummy: ", h1[s].Integral()
        print "h2 entries: ", h2[s].GetEntries()
        print "h2 integral dummy: ", h2[s].Integral()
        #exit()


        #Plotting variables

        y_0 = np.sum(np.array(b0))#*tree_weight --> already in w
        y_1 = np.sum(np.array(b1))#*tree_weight --> already in w
        y_2 = np.sum(np.array(b2))#*tree_weight --> already in w
        e_0 = math.sqrt( sum(x*x for x in b0) )#*tree_weight --> already in w
        e_1 = math.sqrt( sum(x*x for x in b1) )#*tree_weight --> already in w
        e_2 = math.sqrt( sum(x*x for x in b2) )#*tree_weight --> already in w

        error_0_i = c_double()#Double()
        error_1_i = c_double()#Double()
        y_0_i = h0[s].IntegralAndError(h0[s].GetXaxis().FindBin(0),h0[s].GetXaxis().FindBin(2),error_0_i,"")
        y_1_i = h1[s].IntegralAndError(h1[s].GetXaxis().FindBin(0),h1[s].GetXaxis().FindBin(2),error_1_i,"")
        rowI = [s, round(y_0_i,2), round(error_0_i.value,2), round(y_1_i,2), round(error_1_i.value,2)]
        #rowI = [s, round(y_0_i,2), round(error_0_i,2), round(y_1_i,2), round(error_1_i,2)]

        #Predictions are now dictionaries
        pred_1 = {}
        e_pred_1 = {}
        pred_up_1 = {}
        e_pred_up_1 = {}
        pred_low_1 = {}
        e_pred_low_1 = {}
        pred_2 = {}
        e_pred_2 = {}
        pred_2_from_1 = {}
        e_pred_2_from_1 = {}
        sorted_pr_2 = {}
        sorted_pr_2_from_1 = {}

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
                if len(datasets)>0:
                    dset = datasets[i]

            sorted_pr_2[r+reg_label+dset] = -np.sort(-pr_2[r+reg_label+dset])
            sorted_pr_2_from_1[r+reg_label+dset] = -np.sort(-pr_2_from_1[r+reg_label+dset])
            pred_1[r+reg_label+dset] = np.sum(np.array(pr_1[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
            e_pred_1[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_1[r+reg_label+dset]) ).sum()
            pred_2[r+reg_label+dset] = np.sum(np.array(pr_2[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
            e_pred_2[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_2[r+reg_label+dset]) ).sum()
            pred_2_from_1[r+reg_label+dset] = np.sum(np.array(pr_2_from_1[r+reg_label+dset]))
            #print "This is wrong in BEST method!!"
            e_pred_2_from_1[r+reg_label+dset] = np.sqrt( sum(x*x for x in pr_2_from_1[r+reg_label+dset]) )

            results[r+reg_label+dset][s]["y_0"] = y_0
            results[r+reg_label+dset][s]["e_0"] = e_0
            results[r+reg_label+dset][s]["y_1"] = y_1
            results[r+reg_label+dset][s]["e_1"] = e_1
            results[r+reg_label+dset][s]["y_2"] = y_2
            results[r+reg_label+dset][s]["e_2"] = e_2
            results[r+reg_label+dset][s]["pred_1"] = pred_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_1"] = e_pred_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["pred_2"] = pred_2[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_2"] = e_pred_2[r+reg_label+dset]
            results[r+reg_label+dset][s]["pred_2_from_1"] = pred_2_from_1[r+reg_label+dset]
            results[r+reg_label+dset][s]["e_pred_2_from_1"] = e_pred_2_from_1[r+reg_label+dset]

            print "round to 5"
            round_val = 5#2
            round_pred = 5#4
            row[r+reg_label+dset] = [s, round(y_0,round_val), round(e_0,round_val), round(y_1,round_val), round(e_1,round_val), round(y_2,5), round(e_2,round_val),round( (1. + e_2/y_2),5)]
            table_yield[r+reg_label+dset].add_row(row[r+reg_label+dset])

            rowP[r+reg_label+dset] = [s, round(pred_1[r+reg_label+dset],round_val), round(e_pred_1[r+reg_label+dset],round_val), round(pred_2[r+reg_label+dset],round_pred), round(e_pred_2[r+reg_label+dset],round_pred), round(pred_2_from_1[r+reg_label+dset],round_pred), round(e_pred_2_from_1[r+reg_label+dset],round_pred)]
            table_pred[r+reg_label+dset].add_row(rowP[r+reg_label+dset])

            rowPn[r+reg_label+dset] = [round(pred_1[r+reg_label+dset],round_val), round( 100*(pred_1[r+reg_label+dset]-y_1)/y_1,round_val), round(pred_2[r+reg_label+dset],round_pred), round(100*(pred_2[r+reg_label+dset]-y_2)/y_2,round_pred), round(pred_2_from_1[r+reg_label+dset],round_pred), round(100*(pred_2_from_1[r+reg_label+dset]-y_2)/y_2,round_pred), round(e_pred_2_from_1[r+reg_label+dset],round_pred), round(e_pred_1[r+reg_label+dset],round_pred)]
            table_pred_new[r+reg_label+dset].add_row(rowPn[r+reg_label+dset])

            rowI = [s, round(y_0,round_val), round(e_0,round_val), round(y_1,round_val), round(e_1,round_val)]

            table_integral[r+reg_label+dset].add_row(rowI)


        #Here I would like to keep the name of the samples
        #Try to align here:

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
                if len(datasets)>0:
                    dset = datasets[i]

            ##if i==0:
            print('\n\n======================= From histogram  ==============================')
            print(table_integral[r+reg_label+dset])
            print '\n\n======================= Yields and predictions extrapolated from '+r+reg_label+dset+' ==============================' 
            print(table_yield[r+reg_label+dset])
            print(table_pred_new[r+reg_label+dset])
            print(table_pred[r+reg_label+dset])

        wr  = True#False#False

        for i,r in enumerate(extr_regions):
            reg_label = ""
            if len(regions_labels)>0:
                reg_label = regions_labels[i]
            if len(datasets)>0:
                dset = datasets[i]
            if wr:
                with open(PLOTDIR+'BkgPred_extr_region_'+r+reg_label+dset+add_label+label_2+'.txt', 'w') as w:
                    w.write('\n\n======================= Yields and predictions extrapolated from '+r+' ==============================\n')
                    w.write(str(table_yield[r+reg_label+dset])+"\n")
                    w.write(str(table_pred[r+reg_label+dset])+"\n")
                    w.write(str(table_pred_new[r+reg_label+dset])+"\n")
                    w.write('\n\n======================= From histogram  ==============================\n')
                    w.write(str(table_integral[r+reg_label+dset])+"\n")
                    w.close()
                    print "Info: tables written in file "+PLOTDIR+"BkgPred_extr_region_"+r+reg_label+dset+add_label+label_2+".txt"
            else:
                print "NO tables written in file !!!!!!"    

        with open(YIELDDIR+"BkgPredResults_"+ERA+"_"+REGION+ "_"+s+ add_label+label_2+".yaml","w") as f:
            yaml.dump(results, f)
            f.close()
            print "Info: dictionary written in file "+YIELDDIR+"BkgPredResults_"+ERA+"_"+REGION+"_"+s+add_label+label_2+".yaml"

samples_to_run = data
jet_tag = ""#+
clos = True#False#True#False#True
#clos = False

#jet_tag += "_B-F"#"_G-H"#"_G-H"#"_G-H"#"_B-F"#"_B-F"#

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

#print "Warning!!!!!!!"
#print "Warning!!!!!!!"
#print "Artificially setting lumi to 137*1000 /pb"
#LUMI = 137*1000
#print "Warning!!!!!!!"
#print "Warning!!!!!!!"


print "Ntupledir: ", NTUPLEDIR
#print "Luminosity: ", data[0], LUMI

sample_dict = {}
reg_comb = ""
isMC = False

#ERA
if ERA=="2016":
    #jet_tag+="_B-F"
    jet_tag+="_G-H"


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
        sample_dict["WtoMN"] = "SingleMuon"
        #sample_dict["WtoMN_MET"] = "SingleMuon"
        if ERA=="2018":
            sample_dict["WtoEN"] = "EGamma"
            #sample_dict["WtoEN_MET"] = "EGamma"
        else:
            if "_B-F" not in jet_tag:
                sample_dict["WtoEN"] = "SingleElectron"
                #sample_dict["WtoEN_MET"] = "SingleElectron"
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

def cosmic_veto_eff(eta_cut,phi_cut,check_closure=False,negative=True,add_label=""):
    if check_closure:
        dnn_threshold = 0.95#5
        dnn_max = 0.996
        print  "Performing closure test with DNN threshold: ", dnn_threshold
    else:
        dnn_threshold = 0.996#0.98#6#0.996
        dnn_max = 1.
        print  "DNN threshold: ", dnn_threshold

    all_files = [
        'HighMETRun2016B-07Aug17_ver2-v1',
        'HighMETRun2016C-07Aug17_v1',
        'HighMETRun2016D-07Aug17_v1',
        'HighMETRun2016E-07Aug17_v1',
        'HighMETRun2016F-07Aug17_v1',
        'HighMETRun2016G-07Aug17_v1',
        'HighMETRun2016H-07Aug17_v1',
        'HighMETRun2017B-17Nov2017-v1',
        'HighMETRun2017C-17Nov2017-v1',
        'HighMETRun2017D-17Nov2017-v1',
        'HighMETRun2017E-17Nov2017-v1',
        'HighMETRun2017F-17Nov2017-v1'
        'HighMETRun2018A-17Sep2018-v1',
        'HighMETRun2018B-17Sep2018-v1',
        'HighMETRun2018C-17Sep2018-v1',
        'HighMETRun2018D-PromptReco-v2',
    ]
    #load all data eras
    eras = ["2016","2017","2018"]
    root_files = []

    list_of_variables = ["isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRecHitsEB","Jets.nRecHitsEB"]#"nLeptons"
    list_of_variables += ["nCHSJets_in_HEM_pt_30_all_eta","isSR","dt_ecal_dist","dt_ecal_acc_dist","isDT_fit","nCosmic*","min_dPhi*"]

    passed_num = np.array([])
    passed_den = np.array([])
    w_num = np.array([])
    w_den = np.array([])

    for e in eras:
        print "Doing... ", e
        if e=="2018":
            MINPHI = 0.9
            MAXPHI = 0.4
        elif e=="2017":
            MINPHI = 3.5 #min allowed
            MAXPHI = 2.7 #max allowed
        else:
            MINPHI = -5
            MAXPHI = 5

        fold = (NTPL%e)
        print fold
        root_files = [fold+x+".root" for x in all_files if os.path.isfile(os.path.join(fold, x+".root"))]
        gen = uproot.iterate(root_files,"tree",list_of_variables,entrysteps=1000000)
        it = 0
        n_true_tag = 0
        n_loose_tag = 0
        for arrays in gen:
            print "------------------ iteration n. ", it
            cut_mask = arrays["isSR"]>0
            cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))

            if KILL_QCD:
                #print "   QCD killer cut!"
                cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

            #cut_mask = np.logical_and(cut_mask,arrays["nCosmicMuonsOneLeg"]>0)
            #cut_mask = np.logical_and(cut_mask,arrays["nCosmicMuons"]>1)

            #Jets --> JetsNegative
            #Default cut_jets, does nothing basically
            #look at jets with high time!
            #at least one jet tagged with large time --> induced by cosmics
            cut_jets = arrays["Jets.pt"]>-1#np.logical_and(arrays["Jets.nRecHitsEB"]>20,np.logical_and(arrays["Jets.sigprob"]> dnn_threshold,arrays["Jets.sigprob"]<= dnn_max))
            cut_jets = np.logical_and(cut_mask,cut_jets)
            cut_mask = (cut_jets.any()==True)

            #Look at bin 1-2
            sigprob = arrays["Jets.sigprob"]
            tag_mask = sigprob>dnn_threshold#np.logical_and(sigprob > dnn_threshold,sigprob <= dnn_max)
            bin1_m = (sigprob[tag_mask].counts > 1)
            cut_mask = np.logical_and(cut_mask,bin1_m)

            if phi_cut and eta_cut:
                cut_mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
                cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
                #This is needed to guarantee the shapes are consistent
                cut_mask = (cut_mask_phi_eta.any()==True)
                cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)#new

            '''
            if check_closure:
                sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
                tag_mask_blinded = np.logical_and(sigprob > dnn_threshold,sigprob <= dnn_max)
                bin2_m_blinded = (sigprob[tag_mask_blinded].counts >1)
                n_loose_tag += bin2_m_blinded.sum()
                tag_mask = (sigprob > 0.996)
                bin2_m = (sigprob[tag_mask].counts >1)
                print "Events blinded:", bin2_m_blinded.sum()
                print "Events in true SR:", bin2_m.sum()
                n_true_tag+=bin2_m.sum()

                sigprob = arrays["Jets.sigprob"]
                tag_mask = (sigprob > 0.996)
                bin2_m = (sigprob[tag_mask].counts <=1)
                cut_mask = np.logical_and(cut_mask,bin2_m)
            '''
                
            eventweight = arrays["EventWeight"][cut_mask]
            pureweight = arrays["PUReWeight"][cut_mask]
            weight = np.multiply(eventweight,pureweight)

            print "DNN: ", arrays["Jets.sigprob"][cut_jets][cut_mask]
            print "phi: ", arrays["Jets.phi"][cut_jets][cut_mask]
            print "dt_ecal_dist: ", arrays["dt_ecal_dist"][cut_mask]
            print "dt_ecal_acc_dist: ", arrays["dt_ecal_acc_dist"][cut_mask]
            print "min_dPhi_jets_eta_1p0_0p95: ", arrays["min_dPhi_jets_eta_1p0_0p95"][cut_mask]
            print "min_dPhi_jets_0p95: ", arrays["min_dPhi_jets_0p95"][cut_mask]
            print "RunNumber: ", arrays["RunNumber"][cut_mask]
            print "LumiNumber: ", arrays["LumiNumber"][cut_mask]
            print "EventNumber: ", arrays["EventNumber"][cut_mask]
            den = arrays["dt_ecal_acc_dist"][cut_mask]

            if den.shape[0]>0:
                passed_den = np.concatenate( (passed_den, np.hstack(den)) )
                w_den = np.concatenate( (w_den, np.hstack( den.astype(bool)*weight ) ) )

            cut_tag = np.logical_and(cut_mask,arrays["dt_ecal_acc_dist"]<9999)
            num = arrays["dt_ecal_acc_dist"][cut_tag]
            eventweight = arrays["EventWeight"][cut_tag]
            pureweight = arrays["PUReWeight"][cut_tag]
            weight = np.multiply(eventweight,pureweight)
            if num.shape[0]>0:
                passed_num = np.concatenate( (passed_num, np.hstack(num)) )
                w_num = np.concatenate( (w_num, np.hstack( num.astype(bool)*weight ) ) )
            it+=1

    print passed_den
    print passed_num
    print "n_loose_tag ", n_loose_tag
    print "n_true_tag (vetoed) ", n_true_tag
    bins = np.array([0.,99999.])
    #bins = np.array([-1.,1.])
    hist_den = TH1F("den", "den", len(bins)-1, bins)
    hist_num = TH1F("num", "num", len(bins)-1, bins)

    root_numpy.fill_hist( hist_den, passed_den, weights=w_den )
    print hist_den.Print()
    root_numpy.fill_hist( hist_num, passed_num, weights=w_num )
    print hist_num.Print()
        
    eff = TGraphAsymmErrors()
    eff.BayesDivide(hist_num,hist_den)

    print "EFF: "+str(eff.GetPointY(0))+" #pm "+str(eff.GetErrorY(0))
    print "EFF: "+str(eff.GetPointY(0))+" + "+str(eff.GetErrorYlow(0)) +" - "+str(eff.GetErrorYhigh(0))
    eff.SetMarkerSize(1.)
    eff.SetMarkerStyle(markers[0])#(sign_sampl[s]['marker'])
    eff.SetMarkerColor(colors[0])#(2)
    eff.SetFillColor(colors[0])#(2) 
    eff.SetLineColor(colors[0])#(2)
    eff.SetLineWidth(2)
    eff.GetYaxis().SetRangeUser(0.,1.)
    eff.SetTitle("Cosmic veto eff: "+str(eff.GetPointY(0))+" #pm "+str(eff.GetErrorY(0)))
    eff.GetYaxis().SetTitle("Efficiency")#("Efficiency (L1+HLT)")
    eff.GetYaxis().SetTitleOffset(1.2)#("Efficiency (L1+HLT)")
    eff.GetYaxis().SetTitleSize(0.05)#DCMS
    eff.GetXaxis().SetRangeUser(bins[0],bins[-1])
    eff.GetXaxis().SetTitle("")
    
    if check_closure:
        add_label+="_closure"+str(dnn_threshold).replace(".","p")

    can = TCanvas("can","can", 1000, 800)
    can.cd()
    eff.Draw("AP")
    #outfile = TFile(PLOTDIR+"TagEff_"+s+add_label+".root","RECREATE")
    #outfile.cd()
    #eff.Write("eff_"+s)
    #eff_cutbased[s].Write("eff_cutbased_"+s)
    #hist_den[s].Write("den_"+s)
    #hist_num[s].Write("num_"+s)
    #hist_num_cutbased[s].Write("num_cutbased_"+s)
    #can.Write()
    #print "Info in <TFile::Write>: root file "+PLOTDIR+"TagEff_"+s+add_label+".root has been created"
    #outfile.Close()
    can.Print(PLOTDIR+"CosmicVetoEff"+add_label+".pdf")
    can.Print(PLOTDIR+"CosmicVetoEff"+add_label+".png")


#LUMI = 137.4*1000
#print "Using full run 2 lumi!! ", LUMI


#jet_tag+="_cosmic"
#jet_tag+="_negative_cosmic"
#jet_tag+="_positive_cosmic"
#positive
#jet_tag+="_positive_cosmic"
#jet_tag+="_negative_cosmic"

#jet_tag+="_positive_cosmic_veto"
#jet_tag+="_negative_cosmic_veto"

#std
jet_tag+="_negative"
#jet_tag+="_positive"
#jet_tag+="_veto_cosmic"

##These are for checks in SR
#jet_tag += "_negative_cosmic_SR"
#jet_tag += "_positive_cosmic_SR"
#jet_tag += "_negative_cosmic_veto_SR"
#jet_tag += "_positive_cosmic_veto_SR"



##cosmic_veto_eff(eta_cut=CUT_ETA,phi_cut=CUT_PHI,check_closure=clos,negative=True if REGION=="SRneg" else False)
#exit()
#calculate_tag_eff(get_tree_weights(samples_to_run,LUMI),samples_to_run,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,j_idx=-1,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#draw_tag_eff_updated(sample_dict,reg_label=reg_comb,add_label=jet_tag,check_closure=clos,eta=DO_ETA,phi=DO_PHI,eta_cut=CUT_ETA,phi_cut=CUT_PHI)
#exit()

#THIS HERE!!!
'''
draw_data_combination(
    ERA,
    #["HBHEpos","HBHEneg","HBHEpos","HBHEneg"],
    #regions_labels = ["_MinDPhi_0p5_positive_cosmic","_MinDPhi_0p5_negative_cosmic","_MinDPhi_0p5_positive_cosmic_veto","_MinDPhi_0p5_negative_cosmic_veto"],
    #regions_labels = ["_MinDPhi_0p5_G-H_positive_cosmic","_MinDPhi_0p5_G-H_negative_cosmic"],
    #lab_2="_pos_neg_cosmic_compare_veto",
    #["SR","ZtoLL","TtoEM"],#"DiJetMET"],#SR
    #regions_labels = [jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    #lab_2=jet_tag,
    ["SRpos","SRneg","SRpos","SRneg"],
    regions_labels = ["_MinDPhi_0p5_positive_cosmic","_MinDPhi_0p5_negative_cosmic","_MinDPhi_0p5_positive_cosmic_veto","_MinDPhi_0p5_negative_cosmic_veto"],
    lab_2="_pos_neg_cosmic_compare_veto",
    add_label="",#"_vs_QCD_MC",#"_closure"
    check_closure=clos,#True#False#True
    eta_cut = CUT_ETA,
    eta=DO_ETA,
    phi_cut = CUT_PHI,
    phi=DO_PHI,
    do_ratio = False,
    isMC = False
)

exit()
'''

#Bkg pred
background_prediction(
    get_tree_weights(samples_to_run,LUMI),
    samples_to_run,
    extr_regions = ["ZtoLL","TtoEM"],#"SRneg","SRneg"],
    #all
    regions_labels = [jet_tag,jet_tag,"_MinDPhi_0p5_negative_cosmic","_MinDPhi_0p5_negative_cosmic_veto",jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    #2016
    #regions_labels = [jet_tag,jet_tag,"_MinDPhi_0p5_G-H_negative_cosmic",jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag,jet_tag],
    add_label="",#"_closure_0p95",#"",#
    label_2=jet_tag,
    check_closure=clos,#False
    plot_distr = "",
    eta = DO_ETA,
    phi = DO_PHI,
    eta_cut = CUT_ETA,
    phi_cut = CUT_PHI,
)
