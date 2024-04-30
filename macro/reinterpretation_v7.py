#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from collections import defaultdict
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, THnF, TH1D, TGraphAsymmErrors, TH2D
from ROOT import TStyle, TCanvas, TPad, gPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile, TPaveText
import uproot
#import root_numpy
import yaml
from prettytable import PrettyTable
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)


import matplotlib.pyplot as plt
#import mpl_toolkits
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D


import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
#gROOT.SetBatch(True)
#gStyle.SetOptStat(0)

#from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
#from NNInferenceCMSSW.LLP_NN_Inference.selections import *
#from NNInferenceCMSSW.LLP_NN_Inference.reinterpretation_variables import *

ERA = "2018"
LUMI = -1

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI  = lumi[ "HighMET" ]["tot"]

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_reinterpretation/"
MAIN_OUT = "plots/v6_calo_AOD_"+ERA+"_reinterpretation_v1/"

#new ntuples aftrer T2 accident
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+ERA+"_reinterpretation/"
MAIN_OUT = "plots/v7_calo_AOD_"+ERA+"_reinterpretation/"


#new ntuples after post-processing
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+ERA+"_reinterpretation/post_processed/"
MAIN_OUT = "plots/v7_calo_AOD_"+ERA+"_reinterpretation_v1/"

#new ntuples after post-processing
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+ERA+"_reinterpretation/post_processed_v2/"
MAIN_OUT = "plots/v7_calo_AOD_"+ERA+"_reinterpretation_v2/"

#new ntuples after post-processing, all eras together
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_%s_reinterpretation/post_processed_v2/"
MAIN_OUT = "plots/v7_calo_AOD_reinterpretation_v2_no_pt_bins/"
#MAIN_OUT = "plots/v7_calo_AOD_reinterpretation_v2_pt_bins/"
#MAIN_OUT = "plots/v7_calo_AOD_reinterpretation_v2_no_pt_bins_less_r_z_bins/"
#MAIN_OUT_ALL = "plots/v7_calo_AOD_reinterpretation_v2_all_events/"
MAIN_OUT_ALL = "plots/v7_calo_AOD_reinterpretation_v2_no_pt_bins_all_events/"

HEPDATA_OUT = "/afs/desy.de/user/l/lbenato/HEPData/exo-21-014-hepdata/input/"

if not os.path.isdir(MAIN_OUT): os.mkdir(MAIN_OUT)
if not os.path.isdir(MAIN_OUT_ALL): os.mkdir(MAIN_OUT_ALL)

sign = [
    #"SUSY_mh127_ctau500_",
    #"SUSY_mh127_ctau3000_",

    ##"SUSY_mh250_ctau500_HH",
    ##"SUSY_mh250_ctau3000_HH",

    #"SUSY_mh400_ctau500_",
    #"SUSY_mh400_ctau3000_",

    #"SUSY_mh1000_ctau500_",
    #"SUSY_mh1000_ctau3000_",

    #"SUSY_mh1800_ctau500_",
    #"SUSY_mh1800_ctau3000_",

    "SUSY_mh127_",
    "SUSY_mh150_",
    "SUSY_mh175_",
    "SUSY_mh200_",
    "SUSY_mh250_",
    "SUSY_mh300_",
    "SUSY_mh400_",
    "SUSY_mh600_",
    "SUSY_mh800_",
    "SUSY_mh1000_",
    "SUSY_mh1250_",
    "SUSY_mh1500_",
    "SUSY_mh1800_",
    #"SUSY_",
]


sign_500 = [

    "SUSY_mh127_ctau500_",
    "SUSY_mh150_ctau500_",
    "SUSY_mh175_ctau500_",
    "SUSY_mh200_ctau500_",
    "SUSY_mh250_ctau500_",
    "SUSY_mh300_ctau500_",
    "SUSY_mh400_ctau500_",
    "SUSY_mh600_ctau500_",
    "SUSY_mh800_ctau500_",
    "SUSY_mh1000_ctau500_",
    "SUSY_mh1250_ctau500_",
    "SUSY_mh1500_ctau500_",
    "SUSY_mh1800_ctau500_",
    #"SUSY_",
]

sign_3000 = [

    "SUSY_mh127_ctau3000_",
    "SUSY_mh150_ctau3000_",
    "SUSY_mh175_ctau3000_",
    "SUSY_mh200_ctau3000_",
    "SUSY_mh250_ctau3000_",
    "SUSY_mh300_ctau3000_",
    "SUSY_mh400_ctau3000_",
    "SUSY_mh600_ctau3000_",
    "SUSY_mh800_ctau3000_",
    "SUSY_mh1000_ctau3000_",
    "SUSY_mh1250_ctau3000_",
    "SUSY_mh1500_ctau3000_",
    "SUSY_mh1800_ctau3000_",
    #"SUSY_",
]

#sign = ["SUSY_mh127_","SUSY_mh200_",]


sign_ctau = [
    "SUSY_mh127_ctau500_",
    "SUSY_mh127_ctau3000_",

    "SUSY_mh200_ctau500_",
    "SUSY_mh200_ctau3000_",

    #"SUSY_mh300_ctau500_",
    #"SUSY_mh300_ctau3000_",

    ##"SUSY_mh250_ctau500_HH",
    ##"SUSY_mh250_ctau3000_HH",

    "SUSY_mh400_ctau500_",
    "SUSY_mh400_ctau3000_",

    #"SUSY_mh600_ctau500_",
    #"SUSY_mh600_ctau3000_",

    "SUSY_mh1000_ctau500_",
    "SUSY_mh1000_ctau3000_",

    #"SUSY_mh1800_ctau500_",
    #"SUSY_mh1800_ctau3000_",

]

########################

#bins must be defined universally
bins_r_resolved = np.array([0.,0.02,0.05,0.1,0.2,0.5,1.,1.5,500.])
bins_z_resolved = np.array([0.,0.05,0.1,0.2,1.,2.,1000.])
bins_pt_resolved_q = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])
bins_pt_resolved_h = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])

#no_pt bins
bins_pt_resolved_q = np.array([0.,5000.])
bins_pt_resolved_h = np.array([0.,5000.])

#less r z
#bins_r_resolved = np.array([0.,0.05,0.1,0.5,1.,1.5,500.])
#bins_z_resolved = np.array([0.,0.2,1.,2.,1000.])

bins_pt_merged = np.array([0.,50.,120.,180.,300.,500.,800.,1300.,2000.,5000.])
bins_r_merged = np.array([0,0.02,0.05,0.1,0.2,0.5,1.,1.5,1000.])
bins_z_merged = np.array([0,0.05,0.1,0.2,1.,2.,1000.])

#no_pt bins
bins_pt_merged = np.array([0.,5000.])

#less r z
#bins_r_merged = np.array([0.,0.05,0.1,0.5,1.,1.5,500.])
#bins_z_merged = np.array([0.,0.2,1.,2.,1000.])

pl_bins_r_resolved = np.copy(bins_r_resolved)#np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.,1.5,20.])
pl_bins_z_resolved = np.copy(bins_z_resolved)#np.array([0.01,0.05,0.1,0.2,1.,2.,20.])
pl_bins_pt_resolved = np.copy(bins_pt_resolved_q)#np.array([0.1,    40.,    70.,   120.,   200.,   300.,   500., 5000.])

pl_bins_pt_merged = np.copy(bins_pt_merged)#np.array([0.01,50.,120.,180.,300.,500.,800.,1300.,2000.,5000.])
pl_bins_r_merged = np.copy(bins_r_merged)#np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.,     1.5, 20.])#500.])
pl_bins_z_merged = np.copy(bins_z_merged)#np.array([0.01, 0.05, 0.1,  0.2, 1.,  2.,  20.])#1000.])

########################






gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)
gStyle.SetPadRightMargin(0.15)

h = {}
num = {}
den = {}
eff = {}
tree = {}
WEIGHT = "EventWeight"#"EventWeight*TriggerWeight*PUReWeight"

'''
def create_maps_prev(out_fold, sign, skip_even=False,R=0.4):

    list_of_variables = [
        "GenLLPs*","GenBquarks*","GenHiggs*","nTagJets_0p996","nTagJets_to_LLP0*","nTagJets_to_LLP1*","nJets_to_LLP0","nJets_to_LLP1","isPreselected","pt","EventNumber",
    ]

    maps = defaultdict(dict)

    #tree_weight_dict = {}
    ##first round, get tree weight from mc
    #for b in sign:
    #    for i, ss in enumerate(samples[b]['files']):
    #        if ('Run201') in ss:
    #            t_w = 1.
    #        else:
    #            if not os.path.isfile(NTUPLEDIR+ss+'.root'):
    #                print("!!!File ", NTUPLEDIR+ss+'.root', " does not exist! Continuing")
    #                continue
    #            filename = TFile(NTUPLEDIR+ss+'.root', "READ")
    #            if filename.GetListOfKeys().Contains("c_nEvents"):
    #                nevents = filename.Get("c_nEvents").GetBinContent(1)
    #                tree_entries = filename.Get("tree").GetEntries()
    #                print "tree entries: ", tree_entries
    #                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
    #                    print "SUSY central, consider sample dictionary for nevents!"
    #                    nevents = sample[ss]['nevents']
    #                    print "n gen events: ", nevents

    #                if nevents!=tree_entries:
    #                    print "There are some selections applied! Aborting...."
    #                    exit()
    #                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
    #                n_pass      = filename.Get("n_pass").GetBinContent(1)
    #                n_odd       = filename.Get("n_odd").GetBinContent(1)
    #                filename.Close()
    #                xs = sample[ss]['xsec'] * sample[ss]['kfactor']
    #                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
    #                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
    #                    xs = 1.
    #                t_w = LUMI * xs / nevents
    #                if(b_skipTrain>0):
    #                    if(n_odd>0):
    #                        t_w *= float(n_pass/n_odd)
    #            else:
    #                t_w = 0

    #        tree_weight_dict[ss] = t_w

    #print tree_weight_dict

    chunk_size = 100000
    #chunk_size = 100
    for b in sign:
        maps[b]['merged'] = {}
        maps[b]['resolved'] = {}

        preselected = np.array([])
        nTagJets    = np.array([])
        nTagJets_to_LLP0 = np.array([])
        nTagJets_to_LLP1 = np.array([])
        nTagJets_to_LLP0_q0 = np.array([])
        nTagJets_to_LLP0_q1 = np.array([])
        nTagJets_to_LLP1 = np.array([])
        nTagJets_to_LLP1_q2 = np.array([])
        nTagJets_to_LLP1_q3 = np.array([])
        nJets_to_LLP0 = np.array([])
        nJets_to_LLP1 = np.array([])
        genLLP0_merged = np.array([])
        genLLP1_merged = np.array([])
        genLLP0_resolved = np.array([])
        genLLP1_resolved = np.array([])
        genLLP0_r = np.array([])
        genLLP1_r = np.array([])
        genLLP0_z = np.array([])
        genLLP1_z = np.array([])
        genh0_pt = np.array([])
        genh1_pt = np.array([])
        genh0_eta = np.array([])
        genh1_eta = np.array([])
        genb0_pt = np.array([])
        genb1_pt = np.array([])
        genb2_pt = np.array([])
        genb3_pt = np.array([])
        genb0_eta = np.array([])
        genb1_eta = np.array([])
        genb2_eta = np.array([])
        genb3_eta = np.array([])


        for i, ss in enumerate(samples[b]['files']):
            print("Uproot iterating over ", NTUPLEDIR +ss+ '.root', " . . . ")
            filename = NTUPLEDIR + ss + '.root'
            gen = uproot.iterate(filename,"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                
                n_events = -1
            
                ##Take one event every two: to use one set as validation
                ##step = 2 if skip_even else 1
                ##and then apply the mask as follows:
                ##tmp_preselected = arrays["isPreselected"][0:n_events:step]

                #More safe when concatenating multiple things
                if skip_even:
                    skip_even_mask = (arrays["EventNumber"] % 2)==1
                else:
                    skip_even_mask = (arrays["EventNumber"] != 0 )


                #Easiest way: count how many tagged jets are associated to each LLP
                #include also an event-level flag for pre-selections
                tmp_preselected = arrays["isPreselected"][skip_even_mask]

                #number of tagged jets per LLP
                #Primary variable
                tmp_nTagJets = arrays["nTagJets_0p996"][skip_even_mask]
                tmp_nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][skip_even_mask]
                tmp_nTagJets_to_LLP0_q0 = arrays["nTagJets_to_LLP0_q0"][skip_even_mask]
                tmp_nTagJets_to_LLP0_q1 = arrays["nTagJets_to_LLP0_q1"][skip_even_mask]
                tmp_nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][skip_even_mask]
                tmp_nTagJets_to_LLP1_q2 = arrays["nTagJets_to_LLP1_q2"][skip_even_mask]
                tmp_nTagJets_to_LLP1_q3 = arrays["nTagJets_to_LLP1_q3"][skip_even_mask]
                tmp_nJets_to_LLP0 = arrays["nJets_to_LLP0"][skip_even_mask]
                tmp_nJets_to_LLP1 = arrays["nJets_to_LLP1"][skip_even_mask]

                #merged vs resolved
                #Primary variable
                tmp_genLLP0_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[0:1] < R)
                tmp_genLLP1_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[1:2] < R)
                
                #Primary variable
                tmp_genLLP0_resolved = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[0:1] >= R)
                tmp_genLLP1_resolved = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[1:2] >= R)

                #Primary variable
                tmp_genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_even_mask]/100. )))[0:1]
                tmp_genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_even_mask]/100. )))[1:2]

                #Primary variable
                tmp_genLLP0_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_even_mask]/100. )))[0:1] )
                tmp_genLLP1_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_even_mask]/100. )))[1:2] )

                #Primary variable
                tmp_genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_even_mask] )))[0:1]
                tmp_genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_even_mask] )))[1:2]

                #Primary variable
                tmp_genh0_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_even_mask] )))[0:1] )
                tmp_genh1_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_even_mask] )))[1:2] )

                #Primary variable
                tmp_genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[0:1]
                tmp_genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[1:2]
                tmp_genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[2:3]
                tmp_genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[3:]

                #Primary variable
                tmp_genb0_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[0:1] )
                tmp_genb1_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[1:2] )
                tmp_genb2_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[2:3] )
                tmp_genb3_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[3:] )

                preselected = np.concatenate((preselected,tmp_preselected))
                nTagJets    = np.concatenate((nTagJets,tmp_nTagJets))
                nTagJets_to_LLP0 = np.concatenate((nTagJets_to_LLP0,tmp_nTagJets_to_LLP0))
                nTagJets_to_LLP0_q0 = np.concatenate((nTagJets_to_LLP0_q0,tmp_nTagJets_to_LLP0_q0))
                nTagJets_to_LLP0_q1 = np.concatenate((nTagJets_to_LLP0_q1,tmp_nTagJets_to_LLP0_q1))
                nTagJets_to_LLP1 = np.concatenate((nTagJets_to_LLP1,tmp_nTagJets_to_LLP1))
                nTagJets_to_LLP1_q2 = np.concatenate((nTagJets_to_LLP1_q2,tmp_nTagJets_to_LLP1_q2))
                nTagJets_to_LLP1_q3 = np.concatenate((nTagJets_to_LLP1_q3,tmp_nTagJets_to_LLP1_q3))
                nJets_to_LLP0 = np.concatenate((nJets_to_LLP0,tmp_nJets_to_LLP0))
                nJets_to_LLP1 = np.concatenate((nJets_to_LLP1,tmp_nJets_to_LLP1))

                genLLP0_merged = np.concatenate((genLLP0_merged,tmp_genLLP0_merged.flatten()))
                genLLP1_merged = np.concatenate((genLLP1_merged,tmp_genLLP1_merged.flatten()))
                genLLP0_resolved = np.concatenate((genLLP0_resolved,tmp_genLLP0_resolved.flatten()))
                genLLP1_resolved = np.concatenate((genLLP1_resolved,tmp_genLLP1_resolved.flatten()))
                genLLP0_r = np.concatenate((genLLP0_r,tmp_genLLP0_r.flatten()))
                genLLP1_r = np.concatenate((genLLP1_r,tmp_genLLP1_r.flatten()))
                genLLP0_z = np.concatenate((genLLP0_z,tmp_genLLP0_z.flatten()))
                genLLP1_z = np.concatenate((genLLP1_z,tmp_genLLP1_z.flatten()))
                genh0_pt = np.concatenate((genh0_pt,tmp_genh0_pt.flatten()))
                genh1_pt = np.concatenate((genh1_pt,tmp_genh1_pt.flatten()))
                genh0_eta = np.concatenate((genh0_eta,tmp_genh0_eta.flatten()))
                genh1_eta = np.concatenate((genh1_eta,tmp_genh1_eta.flatten()))
                genb0_pt = np.concatenate((genb0_pt,tmp_genb0_pt.flatten()))
                genb1_pt = np.concatenate((genb1_pt,tmp_genb1_pt.flatten()))
                genb2_pt = np.concatenate((genb2_pt,tmp_genb2_pt.flatten()))
                genb3_pt = np.concatenate((genb3_pt,tmp_genb3_pt.flatten()))
                genb0_eta = np.concatenate((genb0_eta,tmp_genb0_eta.flatten()))
                genb1_eta = np.concatenate((genb1_eta,tmp_genb1_eta.flatten()))
                genb2_eta = np.concatenate((genb2_eta,tmp_genb2_eta.flatten()))
                genb3_eta = np.concatenate((genb3_eta,tmp_genb3_eta.flatten()))


        #acceptance masks
        #merged
        acc_LLP0_merged = np.logical_and( np.logical_and(genh0_pt>30.,abs(genh0_eta)<1.) , genLLP0_merged)
        acc_LLP1_merged = np.logical_and( np.logical_and(genh1_pt>30.,abs(genh1_eta)<1.) , genLLP1_merged)

        #Only q1 in acceptance
        acc_LLP0_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                np.logical_not(  np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.) )  #q2 not in acceptance!
            ), 
            genLLP0_resolved)
        #Only q2 in acceptance
        acc_LLP0_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.) ), #q1 not in acceptance
                np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
            ), 
            genLLP0_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP0_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
            ), 
            genLLP0_resolved)

        #Only q1 in acceptance
        acc_LLP1_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                np.logical_not( np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) ) #q2 not in acceptance
            ), 
            genLLP1_resolved)

        #Only q2 in acceptance
        acc_LLP1_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.) ), #q1 not in acceptance
                np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) #q2 in acceptance
            ), 
            genLLP1_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP1_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.)  #q2 in acceptance
            ), 
            genLLP1_resolved)

       
        ##update: r and z bins based on the efficiency
        ##reduce bins to avoid very sparse matrix

        #bins_r_resolved = np.array([0.,0.02,0.05,0.1,0.2,0.5,1.,1.5,500.])
        #bins_z_resolved = np.array([0.,0.05,0.1,0.2,1.,2.,1000.])
        #bins_pt_resolved_q = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])
        #bins_pt_resolved_h = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])

        ##Optimizing on cumulative sample
        #bins_pt_merged = np.array([0.,50.,120.,180.,300.,500.,800.,1300.,2000.,5000.])
        #bins_r_merged = np.array([0,0.02,0.05,0.1,0.2,0.5,1.,1.5,500.])
        #bins_z_merged = np.array([0,0.05,0.1,0.2,1.,2.,1000.])
               
        #merged:
        m_denLLP0_r = genLLP0_r[acc_LLP0_merged]
        m_denLLP1_r = genLLP1_r[acc_LLP1_merged]
        m_denLLP0_z = genLLP0_z[acc_LLP0_merged]
        m_denLLP1_z = genLLP1_z[acc_LLP1_merged]
        m_denLLP0_pt = genh0_pt[acc_LLP0_merged]
        m_denLLP1_pt = genh1_pt[acc_LLP1_merged]

        #merge LLP0 and LLP1
        m_den_r = np.concatenate((m_denLLP0_r,m_denLLP1_r)).flatten()
        m_den_z = np.concatenate((m_denLLP0_z,m_denLLP1_z)).flatten()
        m_den_pt = np.concatenate((m_denLLP0_pt,m_denLLP1_pt)).flatten()


        #w/o preselections
        #At least one tag
        #Here we do not care about the correct q1/q2 assignment
        m_numLLP0_r_T = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_r_T = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        m_numLLP0_z_T = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_z_T = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        m_numLLP0_pt_T = genh0_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_pt_T = genh1_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        #merge LLP0 and LLP1
        m_num_r_T = np.concatenate((m_numLLP0_r_T,m_numLLP1_r_T)).flatten()
        m_num_z_T = np.concatenate((m_numLLP0_z_T,m_numLLP1_z_T)).flatten()
        m_num_pt_T = np.concatenate((m_numLLP0_pt_T,m_numLLP1_pt_T)).flatten()

        #Exactly 1T
        #2T or more

        #w presel
        #At least one tag
        m_numLLP0_r_T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_r_T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        m_numLLP0_z_T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_z_T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        m_numLLP0_pt_T_presel = genh0_pt[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_pt_T_presel = genh1_pt[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        #merge LLP0 and LLP1
        m_num_r_T_presel = np.concatenate((m_numLLP0_r_T_presel,m_numLLP1_r_T_presel)).flatten()
        m_num_z_T_presel = np.concatenate((m_numLLP0_z_T_presel,m_numLLP1_z_T_presel)).flatten()
        m_num_pt_T_presel = np.concatenate((m_numLLP0_pt_T_presel,m_numLLP1_pt_T_presel)).flatten()

        #Exactly 1T
        #2T or more

        m_den, m_den_edges = np.histogramdd(np.transpose([m_den_r,m_den_z,m_den_pt]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))

        m_num_T, m_num_edges_T = np.histogramdd(np.transpose([m_num_r_T,m_num_z_T,m_num_pt_T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))
        m_num_T_presel, m_num_edges_T_presel = np.histogramdd(np.transpose([m_num_r_T_presel,m_num_z_T_presel,m_num_pt_T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))

        m_eff_T = np.divide(m_num_T,m_den,out=np.zeros_like(m_den),where=(m_den != 0))
        m_eff_T_presel = np.divide(m_num_T_presel,m_den,out=np.zeros_like(m_den),where=(m_den != 0))


        # -----

        #resolved:
        
        #Split cases where there is only 1 q in acceptance or when there are 2
        r_denLLP0_r_q1   = genLLP0_r[acc_LLP0_resolved_q1]
        r_denLLP1_r_q1   = genLLP1_r[acc_LLP1_resolved_q1]
        r_denLLP0_z_q1   = genLLP0_z[acc_LLP0_resolved_q1]
        r_denLLP1_z_q1   = genLLP1_z[acc_LLP1_resolved_q1]
        r_denLLP0_pt_q1  = genb0_pt[acc_LLP0_resolved_q1]
        r_denLLP1_pt_q1  = genb2_pt[acc_LLP1_resolved_q1]

        r_denLLP0_r_q2   = genLLP0_r[acc_LLP0_resolved_q2]
        r_denLLP1_r_q2   = genLLP1_r[acc_LLP1_resolved_q2]
        r_denLLP0_z_q2   = genLLP0_z[acc_LLP0_resolved_q2]
        r_denLLP1_z_q2   = genLLP1_z[acc_LLP1_resolved_q2]
        r_denLLP0_pt_q2  = genb1_pt[acc_LLP0_resolved_q2]
        r_denLLP1_pt_q2  = genb3_pt[acc_LLP1_resolved_q2]

        #concatenate cases of 1 quark in acceptance
        r_den_r_q = np.concatenate((r_denLLP0_r_q1,r_denLLP0_r_q2))
        r_den_r_q = np.concatenate((r_den_r_q,r_denLLP1_r_q1))
        r_den_r_q = np.concatenate((r_den_r_q,r_denLLP1_r_q2)).flatten()

        r_den_z_q = np.concatenate((r_denLLP0_z_q1,r_denLLP0_z_q2))
        r_den_z_q = np.concatenate((r_den_z_q,r_denLLP1_z_q1))
        r_den_z_q = np.concatenate((r_den_z_q,r_denLLP1_z_q2)).flatten()

        r_den_pt_q = np.concatenate((r_denLLP0_pt_q1,r_denLLP0_pt_q2))
        r_den_pt_q = np.concatenate((r_den_pt_q,r_denLLP1_pt_q1))
        r_den_pt_q = np.concatenate((r_den_pt_q,r_denLLP1_pt_q2)).flatten()

        #Both quarks in acceptance
        r_denLLP0_r_q1q2    = genLLP0_r[acc_LLP0_resolved_q1q2]
        r_denLLP1_r_q1q2    = genLLP1_r[acc_LLP1_resolved_q1q2]
        r_denLLP0_z_q1q2    = genLLP0_z[acc_LLP0_resolved_q1q2]
        r_denLLP1_z_q1q2    = genLLP1_z[acc_LLP1_resolved_q1q2]
        r_denLLP0_pt1_q1q2  = genb0_pt[acc_LLP0_resolved_q1q2]
        r_denLLP1_pt1_q1q2  = genb2_pt[acc_LLP1_resolved_q1q2]
        r_denLLP0_pt2_q1q2  = genb1_pt[acc_LLP0_resolved_q1q2]
        r_denLLP1_pt2_q1q2  = genb3_pt[acc_LLP1_resolved_q1q2]

        #concatenate LLP0 twice, LLP1 twice
        r_den_r_qq = np.concatenate((r_denLLP0_r_q1q2,r_denLLP0_r_q1q2))
        r_den_r_qq = np.concatenate((r_den_r_qq,r_denLLP1_r_q1q2))
        r_den_r_qq = np.concatenate((r_den_r_qq,r_denLLP1_r_q1q2)).flatten()

        r_den_z_qq = np.concatenate((r_denLLP0_z_q1q2,r_denLLP0_z_q1q2))
        r_den_z_qq = np.concatenate((r_den_z_qq,r_denLLP1_z_q1q2))
        r_den_z_qq = np.concatenate((r_den_z_qq,r_denLLP1_z_q1q2)).flatten()

        r_den_pt_qq = np.concatenate((r_denLLP0_pt1_q1q2,r_denLLP0_pt2_q1q2))
        r_den_pt_qq = np.concatenate((r_den_pt_qq,r_denLLP1_pt1_q1q2))
        r_den_pt_qq = np.concatenate((r_den_pt_qq,r_denLLP1_pt2_q1q2)).flatten()


        #HERE
        #Split cases when there is 1q or 2q in acceptance

        #w/o preselections
        #At least one tag
        #check if it's really due to q1/q2
        #TODO does it change?? --> yes but it's more correct!
        #Here we do care about the correct q1/q2 assignment!!
        r_numLLP0_r_q1_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_r_q1_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]
        r_numLLP0_z_q1_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_z_q1_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]
        r_numLLP0_pt_q1_T  =  genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_pt_q1_T  =  genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]

        r_numLLP0_r_q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_r_q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]
        r_numLLP0_z_q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_z_q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]
        r_numLLP0_pt_q2_T  =  genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_pt_q2_T  =  genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]

        #concatenate everything
        r_num_r_q_T = np.concatenate((r_numLLP0_r_q1_T,r_numLLP0_r_q2_T))
        r_num_r_q_T = np.concatenate((r_num_r_q_T,r_numLLP1_r_q1_T))
        r_num_r_q_T = np.concatenate((r_num_r_q_T,r_numLLP1_r_q2_T)).flatten()

        r_num_z_q_T = np.concatenate((r_numLLP0_z_q1_T,r_numLLP0_z_q2_T))
        r_num_z_q_T = np.concatenate((r_num_z_q_T,r_numLLP1_z_q1_T))
        r_num_z_q_T = np.concatenate((r_num_z_q_T,r_numLLP1_z_q2_T)).flatten()

        r_num_pt_q_T = np.concatenate((r_numLLP0_pt_q1_T,r_numLLP0_pt_q2_T))
        r_num_pt_q_T = np.concatenate((r_num_pt_q_T,r_numLLP1_pt_q1_T))
        r_num_pt_q_T = np.concatenate((r_num_pt_q_T,r_numLLP1_pt_q2_T)).flatten()


        r_numLLP0_r1_q1q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_r2_q1q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_r1_q1q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_r2_q1q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        r_numLLP0_z1_q1q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_z2_q1q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_z1_q1q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_z2_q1q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        r_numLLP0_pt1_q1q2_T  = genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_pt2_q1q2_T  = genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_pt1_q1q2_T  = genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_pt2_q1q2_T  = genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        #concatenate LLP0 twice, LLP1 twice
        r_num_r_qq_T = np.concatenate((r_numLLP0_r1_q1q2_T,r_numLLP0_r2_q1q2_T))
        r_num_r_qq_T = np.concatenate((r_num_r_qq_T,r_numLLP1_r1_q1q2_T))
        r_num_r_qq_T = np.concatenate((r_num_r_qq_T,r_numLLP1_r2_q1q2_T)).flatten()

        r_num_z_qq_T = np.concatenate((r_numLLP0_z1_q1q2_T,r_numLLP0_z2_q1q2_T))
        r_num_z_qq_T = np.concatenate((r_num_z_qq_T,r_numLLP1_z1_q1q2_T))
        r_num_z_qq_T = np.concatenate((r_num_z_qq_T,r_numLLP1_z2_q1q2_T)).flatten()

        r_num_pt_qq_T = np.concatenate((r_numLLP0_pt1_q1q2_T,r_numLLP0_pt2_q1q2_T))
        r_num_pt_qq_T = np.concatenate((r_num_pt_qq_T,r_numLLP1_pt1_q1q2_T))
        r_num_pt_qq_T = np.concatenate((r_num_pt_qq_T,r_numLLP1_pt2_q1q2_T)).flatten()



        #with preselections
        #At least one tag
        
        #print "need to carefully check the num (q1/q2 matching)"

        #print "need to carefully check LLP0-LLP1 overlaps"
        #This should be by construction, as every jet has one and only one LLP association (the closest)

        #print "need to carefully check if LLP0-LLP1 adds up to number of tagged jets"
        #This should be by construction, as every jet has one and only one LLP association (the closest)
        #We can have tagged jets that are not matched though

        r_numLLP0_r_q1_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_r_q1_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]
        r_numLLP0_z_q1_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_z_q1_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]
        r_numLLP0_pt_q1_T_presel  =  genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_pt_q1_T_presel  =  genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]

        r_numLLP0_r_q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_r_q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]
        r_numLLP0_z_q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_z_q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]
        r_numLLP0_pt_q2_T_presel  =  genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_pt_q2_T_presel  =  genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]


        #concatenate everything
        r_num_r_q_T_presel = np.concatenate((r_numLLP0_r_q1_T_presel,r_numLLP0_r_q2_T_presel))
        r_num_r_q_T_presel = np.concatenate((r_num_r_q_T_presel,r_numLLP1_r_q1_T_presel))
        r_num_r_q_T_presel = np.concatenate((r_num_r_q_T_presel,r_numLLP1_r_q2_T_presel)).flatten()

        r_num_z_q_T_presel = np.concatenate((r_numLLP0_z_q1_T_presel,r_numLLP0_z_q2_T_presel))
        r_num_z_q_T_presel = np.concatenate((r_num_z_q_T_presel,r_numLLP1_z_q1_T_presel))
        r_num_z_q_T_presel = np.concatenate((r_num_z_q_T_presel,r_numLLP1_z_q2_T_presel)).flatten()

        r_num_pt_q_T_presel = np.concatenate((r_numLLP0_pt_q1_T_presel,r_numLLP0_pt_q2_T_presel))
        r_num_pt_q_T_presel = np.concatenate((r_num_pt_q_T_presel,r_numLLP1_pt_q1_T_presel))
        r_num_pt_q_T_presel = np.concatenate((r_num_pt_q_T_presel,r_numLLP1_pt_q2_T_presel)).flatten()



        r_numLLP0_r1_q1q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_r2_q1q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_r1_q1q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_r2_q1q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        r_numLLP0_z1_q1q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_z2_q1q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_z1_q1q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_z2_q1q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        r_numLLP0_pt1_q1q2_T_presel  = genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_pt2_q1q2_T_presel  = genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_pt1_q1q2_T_presel  = genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_pt2_q1q2_T_presel  = genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        #concatenate LLP0 twice, LLP1 twice
        r_num_r_qq_T_presel = np.concatenate((r_numLLP0_r1_q1q2_T_presel,r_numLLP0_r2_q1q2_T_presel))
        r_num_r_qq_T_presel = np.concatenate((r_num_r_qq_T_presel,r_numLLP1_r1_q1q2_T_presel))
        r_num_r_qq_T_presel = np.concatenate((r_num_r_qq_T_presel,r_numLLP1_r2_q1q2_T_presel)).flatten()

        r_num_z_qq_T_presel = np.concatenate((r_numLLP0_z1_q1q2_T_presel,r_numLLP0_z2_q1q2_T_presel))
        r_num_z_qq_T_presel = np.concatenate((r_num_z_qq_T_presel,r_numLLP1_z1_q1q2_T_presel))
        r_num_z_qq_T_presel = np.concatenate((r_num_z_qq_T_presel,r_numLLP1_z2_q1q2_T_presel)).flatten()

        r_num_pt_qq_T_presel = np.concatenate((r_numLLP0_pt1_q1q2_T_presel,r_numLLP0_pt2_q1q2_T_presel))
        r_num_pt_qq_T_presel = np.concatenate((r_num_pt_qq_T_presel,r_numLLP1_pt1_q1q2_T_presel))
        r_num_pt_qq_T_presel = np.concatenate((r_num_pt_qq_T_presel,r_numLLP1_pt2_q1q2_T_presel)).flatten()


              
        #concatenate everything
        r_den_q, r_den_edges_q = np.histogramdd(np.transpose([r_den_r_q,r_den_z_q,r_den_pt_q]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))

        r_den_qq, r_den_edges_qq = np.histogramdd(np.transpose([r_den_r_qq,r_den_z_qq,r_den_pt_qq]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))


        #w/o preselections
        #At least one tag
        r_num_q_T, r_num_edges_q_T = np.histogramdd(np.transpose([r_num_r_q_T,r_num_z_q_T,r_num_pt_q_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_q_T = np.divide(r_num_q_T,r_den_q,out=np.zeros_like(r_den_q),where=(r_den_q != 0))

        r_num_qq_T, r_num_edges_qq_T = np.histogramdd(np.transpose([r_num_r_qq_T,r_num_z_qq_T,r_num_pt_qq_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_qq_T = np.divide(r_num_qq_T,r_den_qq,out=np.zeros_like(r_den_qq),where=(r_den_qq != 0))


        #with preselections

        r_num_q_T_presel, r_num_edges_q_T_presel = np.histogramdd(np.transpose([r_num_r_q_T_presel,r_num_z_q_T_presel,r_num_pt_q_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_q_T_presel = np.divide(r_num_q_T_presel,r_den_q,out=np.zeros_like(r_den_q),where=(r_den_q != 0))

        r_num_qq_T_presel, r_num_edges_qq_T_presel = np.histogramdd(np.transpose([r_num_r_qq_T_presel,r_num_z_qq_T_presel,r_num_pt_qq_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_qq_T_presel = np.divide(r_num_qq_T_presel,r_den_qq,out=np.zeros_like(r_den_qq),where=(r_den_qq != 0))

        maps[b]['merged']['eff'] = m_eff_T
        maps[b]['merged']['eff_presel'] = m_eff_T_presel
        maps[b]['merged']['bins_r'] = bins_r_merged
        maps[b]['merged']['bins_z'] = bins_z_merged
        maps[b]['merged']['bins_pt'] = bins_pt_merged


        maps[b]['resolved']['eff_q'] = r_eff_q_T
        maps[b]['resolved']['eff_q_presel'] = r_eff_q_T_presel
        maps[b]['resolved']['eff_qq'] = r_eff_qq_T
        maps[b]['resolved']['eff_qq_presel'] = r_eff_qq_T_presel
        maps[b]['resolved']['bins_r'] = bins_r_resolved
        maps[b]['resolved']['bins_z'] = bins_z_resolved
        maps[b]['resolved']['bins_pt'] = bins_pt_resolved_q

        print("merged:")
        print(maps[b]['merged']['eff'])
        print("resolved:")
        print(maps[b]['resolved']['eff_q'])

        with open(out_fold+"Tag_eff_map_"+b+"_"+str(R).replace(".","p")+".yaml","w") as f:
            yaml.dump(maps[b], f)
            f.close()
            print("Info: dictionary written in file "+out_fold+"Tag_eff_map_"+b+"_"+str(R).replace(".","p")+".yaml")

'''

def create_maps(out_fold, sign, skip_even=False,R=0.4,eras=[]):

    list_of_variables = [
        "GenLLPs*","GenBquarks*","GenHiggs*","nTagJets_0p996","nTagJets_to_LLP0*","nTagJets_to_LLP1*","nJets_to_LLP0","nJets_to_LLP1","isPreselected","pt","EventNumber",
    ]

    maps = defaultdict(dict)

    '''
    tree_weight_dict = {}
    #first round, get tree weight from mc
    for b in sign:
        for i, ss in enumerate(samples[b]['files']):
            if ('Run201') in ss:
                t_w = 1.
            else:
                if not os.path.isfile(NTUPLEDIR+ss+'.root'):
                    print("!!!File ", NTUPLEDIR+ss+'.root', " does not exist! Continuing")
                    continue
                filename = TFile(NTUPLEDIR+ss+'.root', "READ")
                if filename.GetListOfKeys().Contains("c_nEvents"):
                    nevents = filename.Get("c_nEvents").GetBinContent(1)
                    tree_entries = filename.Get("tree").GetEntries()
                    print "tree entries: ", tree_entries
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "SUSY central, consider sample dictionary for nevents!"
                        nevents = sample[ss]['nevents']
                        print "n gen events: ", nevents

                    if nevents!=tree_entries:
                        print "There are some selections applied! Aborting...."
                        exit()
                    b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                    n_pass      = filename.Get("n_pass").GetBinContent(1)
                    n_odd       = filename.Get("n_odd").GetBinContent(1)
                    filename.Close()
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                        xs = 1.
                    t_w = LUMI * xs / nevents
                    if(b_skipTrain>0):
                        if(n_odd>0):
                            t_w *= float(n_pass/n_odd)
                else:
                    t_w = 0

            tree_weight_dict[ss] = t_w

    print tree_weight_dict
    '''
    chunk_size = 100000
    #chunk_size = 100
    for b in sign:
        
        print("Doing signal ", b)

        maps[b]['merged'] = {}
        maps[b]['resolved'] = {}

        preselected = np.array([])
        nTagJets    = np.array([])
        nTagJets_to_LLP0 = np.array([])
        nTagJets_to_LLP1 = np.array([])
        nTagJets_to_LLP0_q0 = np.array([])
        nTagJets_to_LLP0_q1 = np.array([])
        nTagJets_to_LLP1 = np.array([])
        nTagJets_to_LLP1_q2 = np.array([])
        nTagJets_to_LLP1_q3 = np.array([])
        nJets_to_LLP0 = np.array([])
        nJets_to_LLP1 = np.array([])
        genLLP0_merged = np.array([])
        genLLP1_merged = np.array([])
        genLLP0_resolved = np.array([])
        genLLP1_resolved = np.array([])
        genLLP0_r = np.array([])
        genLLP1_r = np.array([])
        genLLP0_z = np.array([])
        genLLP1_z = np.array([])
        genh0_pt = np.array([])
        genh1_pt = np.array([])
        genh0_eta = np.array([])
        genh1_eta = np.array([])
        genb0_pt = np.array([])
        genb1_pt = np.array([])
        genb2_pt = np.array([])
        genb3_pt = np.array([])
        genb0_eta = np.array([])
        genb1_eta = np.array([])
        genb2_eta = np.array([])
        genb3_eta = np.array([])
        genb0_phi = np.array([])
        genb1_phi = np.array([])
        genb2_phi = np.array([])
        genb3_phi = np.array([])

        for era in eras:

            print("Doing era ", era)

            LUMI = -1

            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]
            elif era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]
            elif era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]

            for i, ss in enumerate(samples[b]['files']):
                print("HERE FIX NTUPLEDIR")
                print("Uproot iterating over ", NTUPLEDIR%era +ss+ '.root', " . . . ")
                filename = NTUPLEDIR%era + ss + '.root'
                gen = uproot.iterate(filename,"tree",list_of_variables,entrysteps=chunk_size)
                for arrays in gen:
                
                    n_events = -1
            
                    if skip_even:
                        skip_even_mask = (arrays["EventNumber"] % 2)==1
                    else:
                        skip_even_mask = (arrays["EventNumber"] != 0 )

                    tmp_preselected = arrays["isPreselected"][skip_even_mask]

                    #number of tagged jets per LLP
                    #Primary variable
                    tmp_nTagJets = arrays["nTagJets_0p996"][skip_even_mask]
                    tmp_nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][skip_even_mask]
                    tmp_nTagJets_to_LLP0_q0 = arrays["nTagJets_to_LLP0_q0"][skip_even_mask]
                    tmp_nTagJets_to_LLP0_q1 = arrays["nTagJets_to_LLP0_q1"][skip_even_mask]
                    tmp_nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][skip_even_mask]
                    tmp_nTagJets_to_LLP1_q2 = arrays["nTagJets_to_LLP1_q2"][skip_even_mask]
                    tmp_nTagJets_to_LLP1_q3 = arrays["nTagJets_to_LLP1_q3"][skip_even_mask]
                    tmp_nJets_to_LLP0 = arrays["nJets_to_LLP0"][skip_even_mask]
                    tmp_nJets_to_LLP1 = arrays["nJets_to_LLP1"][skip_even_mask]

                    #merged vs resolved
                    #Primary variable
                    tmp_genLLP0_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[0:1] < R)
                    tmp_genLLP1_merged = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[1:2] < R)
                
                    #Primary variable
                    tmp_genLLP0_resolved = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[0:1] >= R)
                    tmp_genLLP1_resolved = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_even_mask] )))[1:2] >= R)

                    #Primary variable
                    tmp_genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_even_mask]/100. )))[0:1]
                    tmp_genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_even_mask]/100. )))[1:2]

                    #Primary variable
                    tmp_genLLP0_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_even_mask]/100. )))[0:1] )
                    tmp_genLLP1_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_even_mask]/100. )))[1:2] )

                    #Primary variable
                    tmp_genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_even_mask] )))[0:1]
                    tmp_genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_even_mask] )))[1:2]
                    
                    #Primary variable
                    tmp_genh0_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_even_mask] )))[0:1] )
                    tmp_genh1_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_even_mask] )))[1:2] )

                    #Primary variable
                    tmp_genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[0:1]
                    tmp_genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[1:2]
                    tmp_genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[2:3]
                    tmp_genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_even_mask] )))[3:]

                    #Primary variable
                    tmp_genb0_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[0:1] )
                    tmp_genb1_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[1:2] )
                    tmp_genb2_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[2:3] )
                    tmp_genb3_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_even_mask] )))[3:] )

                    preselected = np.concatenate((preselected,tmp_preselected))
                    nTagJets    = np.concatenate((nTagJets,tmp_nTagJets))
                    nTagJets_to_LLP0 = np.concatenate((nTagJets_to_LLP0,tmp_nTagJets_to_LLP0))
                    nTagJets_to_LLP0_q0 = np.concatenate((nTagJets_to_LLP0_q0,tmp_nTagJets_to_LLP0_q0))
                    nTagJets_to_LLP0_q1 = np.concatenate((nTagJets_to_LLP0_q1,tmp_nTagJets_to_LLP0_q1))
                    nTagJets_to_LLP1 = np.concatenate((nTagJets_to_LLP1,tmp_nTagJets_to_LLP1))
                    nTagJets_to_LLP1_q2 = np.concatenate((nTagJets_to_LLP1_q2,tmp_nTagJets_to_LLP1_q2))
                    nTagJets_to_LLP1_q3 = np.concatenate((nTagJets_to_LLP1_q3,tmp_nTagJets_to_LLP1_q3))
                    nJets_to_LLP0 = np.concatenate((nJets_to_LLP0,tmp_nJets_to_LLP0))
                    nJets_to_LLP1 = np.concatenate((nJets_to_LLP1,tmp_nJets_to_LLP1))

                    genLLP0_merged = np.concatenate((genLLP0_merged,tmp_genLLP0_merged.flatten()))
                    genLLP1_merged = np.concatenate((genLLP1_merged,tmp_genLLP1_merged.flatten()))
                    genLLP0_resolved = np.concatenate((genLLP0_resolved,tmp_genLLP0_resolved.flatten()))
                    genLLP1_resolved = np.concatenate((genLLP1_resolved,tmp_genLLP1_resolved.flatten()))
                    genLLP0_r = np.concatenate((genLLP0_r,tmp_genLLP0_r.flatten()))
                    genLLP1_r = np.concatenate((genLLP1_r,tmp_genLLP1_r.flatten()))
                    genLLP0_z = np.concatenate((genLLP0_z,tmp_genLLP0_z.flatten()))
                    genLLP1_z = np.concatenate((genLLP1_z,tmp_genLLP1_z.flatten()))
                    genh0_pt = np.concatenate((genh0_pt,tmp_genh0_pt.flatten()))
                    genh1_pt = np.concatenate((genh1_pt,tmp_genh1_pt.flatten()))
                    genh0_eta = np.concatenate((genh0_eta,tmp_genh0_eta.flatten()))
                    genh1_eta = np.concatenate((genh1_eta,tmp_genh1_eta.flatten()))
                    genb0_pt = np.concatenate((genb0_pt,tmp_genb0_pt.flatten()))
                    genb1_pt = np.concatenate((genb1_pt,tmp_genb1_pt.flatten()))
                    genb2_pt = np.concatenate((genb2_pt,tmp_genb2_pt.flatten()))
                    genb3_pt = np.concatenate((genb3_pt,tmp_genb3_pt.flatten()))
                    genb0_eta = np.concatenate((genb0_eta,tmp_genb0_eta.flatten()))
                    genb1_eta = np.concatenate((genb1_eta,tmp_genb1_eta.flatten()))
                    genb2_eta = np.concatenate((genb2_eta,tmp_genb2_eta.flatten()))
                    genb3_eta = np.concatenate((genb3_eta,tmp_genb3_eta.flatten()))


        #acceptance masks
        #merged
        acc_LLP0_merged = np.logical_and( np.logical_and(genh0_pt>30.,abs(genh0_eta)<1.) , genLLP0_merged)
        acc_LLP1_merged = np.logical_and( np.logical_and(genh1_pt>30.,abs(genh1_eta)<1.) , genLLP1_merged)

        #Only q1 in acceptance
        acc_LLP0_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                np.logical_not(  np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.) )  #q2 not in acceptance!
            ), 
            genLLP0_resolved)
        #Only q2 in acceptance
        acc_LLP0_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.) ), #q1 not in acceptance
                np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
            ), 
            genLLP0_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP0_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb0_pt>30.,abs(genb0_eta)<1.), #q1 in acceptance
                np.logical_and(genb1_pt>30.,abs(genb1_eta)<1.)  #q2 in acceptance
            ), 
            genLLP0_resolved)

        #Only q1 in acceptance
        acc_LLP1_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                np.logical_not( np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) ) #q2 not in acceptance
            ), 
            genLLP1_resolved)

        #Only q2 in acceptance
        acc_LLP1_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.) ), #q1 not in acceptance
                np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.) #q2 in acceptance
            ), 
            genLLP1_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP1_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(genb2_pt>30.,abs(genb2_eta)<1.), #q1 in acceptance
                np.logical_and(genb3_pt>30.,abs(genb3_eta)<1.)  #q2 in acceptance
            ), 
            genLLP1_resolved)

        '''       
        #update: r and z bins based on the efficiency
        #reduce bins to avoid very sparse matrix
        bins_r_resolved = np.array([0.,0.02,0.05,0.1,0.2,0.5,1.,1.5,500.])
        bins_z_resolved = np.array([0.,0.05,0.1,0.2,1.,2.,1000.])
        bins_pt_resolved_q = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])
        bins_pt_resolved_h = np.array([0.,    40.,    70.,   120.,   200.,   300.,   500., 5000.])

        #Optimizing on cumulative sample
        bins_pt_merged = np.array([0.,50.,120.,180.,300.,500.,800.,1300.,2000.,5000.])
        bins_r_merged = np.array([0,0.02,0.05,0.1,0.2,0.5,1.,1.5,500.])
        bins_z_merged = np.array([0,0.05,0.1,0.2,1.,2.,1000.])
        '''
    
        #merged:
        m_denLLP0_r = genLLP0_r[acc_LLP0_merged]
        m_denLLP1_r = genLLP1_r[acc_LLP1_merged]
        m_denLLP0_z = genLLP0_z[acc_LLP0_merged]
        m_denLLP1_z = genLLP1_z[acc_LLP1_merged]
        m_denLLP0_pt = genh0_pt[acc_LLP0_merged]
        m_denLLP1_pt = genh1_pt[acc_LLP1_merged]

        #merge LLP0 and LLP1
        m_den_r = np.concatenate((m_denLLP0_r,m_denLLP1_r)).flatten()
        m_den_z = np.concatenate((m_denLLP0_z,m_denLLP1_z)).flatten()
        m_den_pt = np.concatenate((m_denLLP0_pt,m_denLLP1_pt)).flatten()


        #w/o preselections
        #At least one tag
        #Here we do not care about the correct q1/q2 assignment
        m_numLLP0_r_T = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_r_T = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        m_numLLP0_z_T = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_z_T = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        m_numLLP0_pt_T = genh0_pt[np.logical_and(nTagJets_to_LLP0>0,acc_LLP0_merged)]
        m_numLLP1_pt_T = genh1_pt[np.logical_and(nTagJets_to_LLP1>0,acc_LLP1_merged)]
        #merge LLP0 and LLP1
        m_num_r_T = np.concatenate((m_numLLP0_r_T,m_numLLP1_r_T)).flatten()
        m_num_z_T = np.concatenate((m_numLLP0_z_T,m_numLLP1_z_T)).flatten()
        m_num_pt_T = np.concatenate((m_numLLP0_pt_T,m_numLLP1_pt_T)).flatten()

        #Exactly 1T
        #2T or more

        #w presel
        #At least one tag
        m_numLLP0_r_T_presel = genLLP0_r[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_r_T_presel = genLLP1_r[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        m_numLLP0_z_T_presel = genLLP0_z[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_z_T_presel = genLLP1_z[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        m_numLLP0_pt_T_presel = genh0_pt[np.logical_and(nTagJets_to_LLP0>0, np.logical_and(preselected,acc_LLP0_merged) )]
        m_numLLP1_pt_T_presel = genh1_pt[np.logical_and(nTagJets_to_LLP1>0, np.logical_and(preselected,acc_LLP1_merged) )]
        #merge LLP0 and LLP1
        m_num_r_T_presel = np.concatenate((m_numLLP0_r_T_presel,m_numLLP1_r_T_presel)).flatten()
        m_num_z_T_presel = np.concatenate((m_numLLP0_z_T_presel,m_numLLP1_z_T_presel)).flatten()
        m_num_pt_T_presel = np.concatenate((m_numLLP0_pt_T_presel,m_numLLP1_pt_T_presel)).flatten()

        #Exactly 1T
        #2T or more

        m_den, m_den_edges = np.histogramdd(np.transpose([m_den_r,m_den_z,m_den_pt]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))

        m_num_T, m_num_edges_T = np.histogramdd(np.transpose([m_num_r_T,m_num_z_T,m_num_pt_T]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))
        m_num_T_presel, m_num_edges_T_presel = np.histogramdd(np.transpose([m_num_r_T_presel,m_num_z_T_presel,m_num_pt_T_presel]), bins = (bins_r_merged,bins_z_merged,bins_pt_merged))

        m_eff_T = np.divide(m_num_T,m_den,out=np.zeros_like(m_den),where=(m_den != 0))
        m_eff_T_presel = np.divide(m_num_T_presel,m_den,out=np.zeros_like(m_den),where=(m_den != 0))


        print("Need to transform this in a 2D map")
        print("m_den shape: ", m_den.shape)
        print("pt bins: ")
        #print("m_den pt0: ", m_den[:,:,0])
        #print("m_den pt1: ", m_den[:,:,1])
        #print("m_den pt2: ", m_den[:,:,2])
        #print("m_den pt3: ", m_den[:,:,3])
        #print("m_den pt4: ", m_den[:,:,4])
        #print("m_den pt5: ", m_den[:,:,5])
        #print("m_den pt6: ", m_den[:,:,6])
        #print("m_den pt7: ", m_den[:,:,7])
        #print("m_den pt8: ", m_den[:,:,8])

        # -----

        #resolved:
        
        #Split cases where there is only 1 q in acceptance or when there are 2
        r_denLLP0_r_q1   = genLLP0_r[acc_LLP0_resolved_q1]
        r_denLLP1_r_q1   = genLLP1_r[acc_LLP1_resolved_q1]
        r_denLLP0_z_q1   = genLLP0_z[acc_LLP0_resolved_q1]
        r_denLLP1_z_q1   = genLLP1_z[acc_LLP1_resolved_q1]
        r_denLLP0_pt_q1  = genb0_pt[acc_LLP0_resolved_q1]
        r_denLLP1_pt_q1  = genb2_pt[acc_LLP1_resolved_q1]

        r_denLLP0_r_q2   = genLLP0_r[acc_LLP0_resolved_q2]
        r_denLLP1_r_q2   = genLLP1_r[acc_LLP1_resolved_q2]
        r_denLLP0_z_q2   = genLLP0_z[acc_LLP0_resolved_q2]
        r_denLLP1_z_q2   = genLLP1_z[acc_LLP1_resolved_q2]
        r_denLLP0_pt_q2  = genb1_pt[acc_LLP0_resolved_q2]
        r_denLLP1_pt_q2  = genb3_pt[acc_LLP1_resolved_q2]

        #concatenate cases of 1 quark in acceptance
        r_den_r_q = np.concatenate((r_denLLP0_r_q1,r_denLLP0_r_q2))
        r_den_r_q = np.concatenate((r_den_r_q,r_denLLP1_r_q1))
        r_den_r_q = np.concatenate((r_den_r_q,r_denLLP1_r_q2)).flatten()

        r_den_z_q = np.concatenate((r_denLLP0_z_q1,r_denLLP0_z_q2))
        r_den_z_q = np.concatenate((r_den_z_q,r_denLLP1_z_q1))
        r_den_z_q = np.concatenate((r_den_z_q,r_denLLP1_z_q2)).flatten()

        r_den_pt_q = np.concatenate((r_denLLP0_pt_q1,r_denLLP0_pt_q2))
        r_den_pt_q = np.concatenate((r_den_pt_q,r_denLLP1_pt_q1))
        r_den_pt_q = np.concatenate((r_den_pt_q,r_denLLP1_pt_q2)).flatten()

        #Both quarks in acceptance
        r_denLLP0_r_q1q2    = genLLP0_r[acc_LLP0_resolved_q1q2]
        r_denLLP1_r_q1q2    = genLLP1_r[acc_LLP1_resolved_q1q2]
        r_denLLP0_z_q1q2    = genLLP0_z[acc_LLP0_resolved_q1q2]
        r_denLLP1_z_q1q2    = genLLP1_z[acc_LLP1_resolved_q1q2]
        r_denLLP0_pt1_q1q2  = genb0_pt[acc_LLP0_resolved_q1q2]
        r_denLLP1_pt1_q1q2  = genb2_pt[acc_LLP1_resolved_q1q2]
        r_denLLP0_pt2_q1q2  = genb1_pt[acc_LLP0_resolved_q1q2]
        r_denLLP1_pt2_q1q2  = genb3_pt[acc_LLP1_resolved_q1q2]

        #concatenate LLP0 twice, LLP1 twice
        r_den_r_qq = np.concatenate((r_denLLP0_r_q1q2,r_denLLP0_r_q1q2))
        r_den_r_qq = np.concatenate((r_den_r_qq,r_denLLP1_r_q1q2))
        r_den_r_qq = np.concatenate((r_den_r_qq,r_denLLP1_r_q1q2)).flatten()

        r_den_z_qq = np.concatenate((r_denLLP0_z_q1q2,r_denLLP0_z_q1q2))
        r_den_z_qq = np.concatenate((r_den_z_qq,r_denLLP1_z_q1q2))
        r_den_z_qq = np.concatenate((r_den_z_qq,r_denLLP1_z_q1q2)).flatten()

        r_den_pt_qq = np.concatenate((r_denLLP0_pt1_q1q2,r_denLLP0_pt2_q1q2))
        r_den_pt_qq = np.concatenate((r_den_pt_qq,r_denLLP1_pt1_q1q2))
        r_den_pt_qq = np.concatenate((r_den_pt_qq,r_denLLP1_pt2_q1q2)).flatten()


        #HERE
        #Split cases when there is 1q or 2q in acceptance

        #w/o preselections
        #At least one tag
        #check if it's really due to q1/q2
        #TODO does it change?? --> yes but it's more correct!
        #Here we do care about the correct q1/q2 assignment!!
        r_numLLP0_r_q1_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_r_q1_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]
        r_numLLP0_z_q1_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_z_q1_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]
        r_numLLP0_pt_q1_T  =  genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>0,acc_LLP0_resolved_q1)]
        r_numLLP1_pt_q1_T  =  genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>0,acc_LLP1_resolved_q1)]

        r_numLLP0_r_q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_r_q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]
        r_numLLP0_z_q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_z_q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]
        r_numLLP0_pt_q2_T  =  genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>0,acc_LLP0_resolved_q2)]
        r_numLLP1_pt_q2_T  =  genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>0,acc_LLP1_resolved_q2)]

        #concatenate everything
        r_num_r_q_T = np.concatenate((r_numLLP0_r_q1_T,r_numLLP0_r_q2_T))
        r_num_r_q_T = np.concatenate((r_num_r_q_T,r_numLLP1_r_q1_T))
        r_num_r_q_T = np.concatenate((r_num_r_q_T,r_numLLP1_r_q2_T)).flatten()

        r_num_z_q_T = np.concatenate((r_numLLP0_z_q1_T,r_numLLP0_z_q2_T))
        r_num_z_q_T = np.concatenate((r_num_z_q_T,r_numLLP1_z_q1_T))
        r_num_z_q_T = np.concatenate((r_num_z_q_T,r_numLLP1_z_q2_T)).flatten()

        r_num_pt_q_T = np.concatenate((r_numLLP0_pt_q1_T,r_numLLP0_pt_q2_T))
        r_num_pt_q_T = np.concatenate((r_num_pt_q_T,r_numLLP1_pt_q1_T))
        r_num_pt_q_T = np.concatenate((r_num_pt_q_T,r_numLLP1_pt_q2_T)).flatten()


        r_numLLP0_r1_q1q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_r2_q1q2_T   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_r1_q1q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_r2_q1q2_T   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        r_numLLP0_z1_q1q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_z2_q1q2_T   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_z1_q1q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_z2_q1q2_T   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        r_numLLP0_pt1_q1q2_T  = genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>=1,acc_LLP0_resolved_q1q2)]
        r_numLLP0_pt2_q1q2_T  = genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>=1,acc_LLP0_resolved_q1q2)]

        r_numLLP1_pt1_q1q2_T  = genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>=1,acc_LLP1_resolved_q1q2)]
        r_numLLP1_pt2_q1q2_T  = genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>=1,acc_LLP1_resolved_q1q2)]

        #concatenate LLP0 twice, LLP1 twice
        r_num_r_qq_T = np.concatenate((r_numLLP0_r1_q1q2_T,r_numLLP0_r2_q1q2_T))
        r_num_r_qq_T = np.concatenate((r_num_r_qq_T,r_numLLP1_r1_q1q2_T))
        r_num_r_qq_T = np.concatenate((r_num_r_qq_T,r_numLLP1_r2_q1q2_T)).flatten()

        r_num_z_qq_T = np.concatenate((r_numLLP0_z1_q1q2_T,r_numLLP0_z2_q1q2_T))
        r_num_z_qq_T = np.concatenate((r_num_z_qq_T,r_numLLP1_z1_q1q2_T))
        r_num_z_qq_T = np.concatenate((r_num_z_qq_T,r_numLLP1_z2_q1q2_T)).flatten()

        r_num_pt_qq_T = np.concatenate((r_numLLP0_pt1_q1q2_T,r_numLLP0_pt2_q1q2_T))
        r_num_pt_qq_T = np.concatenate((r_num_pt_qq_T,r_numLLP1_pt1_q1q2_T))
        r_num_pt_qq_T = np.concatenate((r_num_pt_qq_T,r_numLLP1_pt2_q1q2_T)).flatten()



        #with preselections
        #At least one tag
        
        #print "need to carefully check the num (q1/q2 matching)"

        #print "need to carefully check LLP0-LLP1 overlaps"
        #This should be by construction, as every jet has one and only one LLP association (the closest)

        #print "need to carefully check if LLP0-LLP1 adds up to number of tagged jets"
        #This should be by construction, as every jet has one and only one LLP association (the closest)
        #We can have tagged jets that are not matched though

        r_numLLP0_r_q1_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_r_q1_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]
        r_numLLP0_z_q1_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_z_q1_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]
        r_numLLP0_pt_q1_T_presel  =  genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>0,np.logical_and(preselected,acc_LLP0_resolved_q1)) ]
        r_numLLP1_pt_q1_T_presel  =  genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>0,np.logical_and(preselected,acc_LLP1_resolved_q1)) ]

        r_numLLP0_r_q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_r_q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]
        r_numLLP0_z_q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_z_q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]
        r_numLLP0_pt_q2_T_presel  =  genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>0,np.logical_and(preselected,acc_LLP0_resolved_q2)) ]
        r_numLLP1_pt_q2_T_presel  =  genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>0,np.logical_and(preselected,acc_LLP1_resolved_q2)) ]


        #concatenate everything
        r_num_r_q_T_presel = np.concatenate((r_numLLP0_r_q1_T_presel,r_numLLP0_r_q2_T_presel))
        r_num_r_q_T_presel = np.concatenate((r_num_r_q_T_presel,r_numLLP1_r_q1_T_presel))
        r_num_r_q_T_presel = np.concatenate((r_num_r_q_T_presel,r_numLLP1_r_q2_T_presel)).flatten()

        r_num_z_q_T_presel = np.concatenate((r_numLLP0_z_q1_T_presel,r_numLLP0_z_q2_T_presel))
        r_num_z_q_T_presel = np.concatenate((r_num_z_q_T_presel,r_numLLP1_z_q1_T_presel))
        r_num_z_q_T_presel = np.concatenate((r_num_z_q_T_presel,r_numLLP1_z_q2_T_presel)).flatten()

        r_num_pt_q_T_presel = np.concatenate((r_numLLP0_pt_q1_T_presel,r_numLLP0_pt_q2_T_presel))
        r_num_pt_q_T_presel = np.concatenate((r_num_pt_q_T_presel,r_numLLP1_pt_q1_T_presel))
        r_num_pt_q_T_presel = np.concatenate((r_num_pt_q_T_presel,r_numLLP1_pt_q2_T_presel)).flatten()



        r_numLLP0_r1_q1q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_r2_q1q2_T_presel   = genLLP0_r[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_r1_q1q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_r2_q1q2_T_presel   = genLLP1_r[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        r_numLLP0_z1_q1q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_z2_q1q2_T_presel   = genLLP0_z[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_z1_q1q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_z2_q1q2_T_presel   = genLLP1_z[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        r_numLLP0_pt1_q1q2_T_presel  = genb0_pt[np.logical_and(nTagJets_to_LLP0_q0>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]
        r_numLLP0_pt2_q1q2_T_presel  = genb1_pt[np.logical_and(nTagJets_to_LLP0_q1>=1,np.logical_and(preselected,acc_LLP0_resolved_q1q2)) ]

        r_numLLP1_pt1_q1q2_T_presel  = genb2_pt[np.logical_and(nTagJets_to_LLP1_q2>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]
        r_numLLP1_pt2_q1q2_T_presel  = genb3_pt[np.logical_and(nTagJets_to_LLP1_q3>=1,np.logical_and(preselected,acc_LLP1_resolved_q1q2)) ]

        #concatenate LLP0 twice, LLP1 twice
        r_num_r_qq_T_presel = np.concatenate((r_numLLP0_r1_q1q2_T_presel,r_numLLP0_r2_q1q2_T_presel))
        r_num_r_qq_T_presel = np.concatenate((r_num_r_qq_T_presel,r_numLLP1_r1_q1q2_T_presel))
        r_num_r_qq_T_presel = np.concatenate((r_num_r_qq_T_presel,r_numLLP1_r2_q1q2_T_presel)).flatten()

        r_num_z_qq_T_presel = np.concatenate((r_numLLP0_z1_q1q2_T_presel,r_numLLP0_z2_q1q2_T_presel))
        r_num_z_qq_T_presel = np.concatenate((r_num_z_qq_T_presel,r_numLLP1_z1_q1q2_T_presel))
        r_num_z_qq_T_presel = np.concatenate((r_num_z_qq_T_presel,r_numLLP1_z2_q1q2_T_presel)).flatten()

        r_num_pt_qq_T_presel = np.concatenate((r_numLLP0_pt1_q1q2_T_presel,r_numLLP0_pt2_q1q2_T_presel))
        r_num_pt_qq_T_presel = np.concatenate((r_num_pt_qq_T_presel,r_numLLP1_pt1_q1q2_T_presel))
        r_num_pt_qq_T_presel = np.concatenate((r_num_pt_qq_T_presel,r_numLLP1_pt2_q1q2_T_presel)).flatten()


              
        #concatenate everything
        r_den_q, r_den_edges_q = np.histogramdd(np.transpose([r_den_r_q,r_den_z_q,r_den_pt_q]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))

        r_den_qq, r_den_edges_qq = np.histogramdd(np.transpose([r_den_r_qq,r_den_z_qq,r_den_pt_qq]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))


        #w/o preselections
        #At least one tag
        r_num_q_T, r_num_edges_q_T = np.histogramdd(np.transpose([r_num_r_q_T,r_num_z_q_T,r_num_pt_q_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_q_T = np.divide(r_num_q_T,r_den_q,out=np.zeros_like(r_den_q),where=(r_den_q != 0))

        r_num_qq_T, r_num_edges_qq_T = np.histogramdd(np.transpose([r_num_r_qq_T,r_num_z_qq_T,r_num_pt_qq_T]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_qq_T = np.divide(r_num_qq_T,r_den_qq,out=np.zeros_like(r_den_qq),where=(r_den_qq != 0))


        #with preselections

        r_num_q_T_presel, r_num_edges_q_T_presel = np.histogramdd(np.transpose([r_num_r_q_T_presel,r_num_z_q_T_presel,r_num_pt_q_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_q_T_presel = np.divide(r_num_q_T_presel,r_den_q,out=np.zeros_like(r_den_q),where=(r_den_q != 0))

        r_num_qq_T_presel, r_num_edges_qq_T_presel = np.histogramdd(np.transpose([r_num_r_qq_T_presel,r_num_z_qq_T_presel,r_num_pt_qq_T_presel]), bins = (bins_r_resolved,bins_z_resolved,bins_pt_resolved_q))
        r_eff_qq_T_presel = np.divide(r_num_qq_T_presel,r_den_qq,out=np.zeros_like(r_den_qq),where=(r_den_qq != 0))


        maps[b]['merged']['eff'] = m_eff_T
        maps[b]['merged']['eff_presel'] = m_eff_T_presel
        maps[b]['merged']['bins_r'] = bins_r_merged
        maps[b]['merged']['bins_z'] = bins_z_merged
        maps[b]['merged']['bins_pt'] = bins_pt_merged


        maps[b]['resolved']['eff_q'] = r_eff_q_T
        maps[b]['resolved']['eff_q_presel'] = r_eff_q_T_presel
        maps[b]['resolved']['eff_qq'] = r_eff_qq_T
        maps[b]['resolved']['eff_qq_presel'] = r_eff_qq_T_presel
        maps[b]['resolved']['bins_r'] = bins_r_resolved
        maps[b]['resolved']['bins_z'] = bins_z_resolved
        maps[b]['resolved']['bins_pt'] = bins_pt_resolved_q

        print("merged:")
        print(maps[b]['merged']['eff'])
        print("resolved:")
        print(maps[b]['resolved']['eff_q'])

        print("bins r merged:")
        print(bins_r_merged)
        print(maps[b]['merged']['bins_r'])

        with open(out_fold+"Tag_eff_map_"+b+"_"+str(R).replace(".","p")+".yaml","w") as f:
            yaml.dump(maps[b], f)
            f.close()
            print("Info: dictionary written in file "+out_fold+"Tag_eff_map_"+b+"_"+str(R).replace(".","p")+".yaml")

        #store arrays themselves

        np.save(out_fold+"m_den_"+b+"_"+str(R).replace(".","p")+".npy",m_den)
        np.save(out_fold+"m_den_edges_"+b+"_"+str(R).replace(".","p")+".npy",m_den_edges)
        np.save(out_fold+"m_num_T_"+b+"_"+str(R).replace(".","p")+".npy",m_num_T)
        np.save(out_fold+"m_num_edges_T_"+b+"_"+str(R).replace(".","p")+".npy",m_num_edges_T)
        np.save(out_fold+"m_num_T_presel_"+b+"_"+str(R).replace(".","p")+".npy",m_num_T_presel)
        np.save(out_fold+"m_num_edges_T_presel"+b+"_"+str(R).replace(".","p")+".npy",m_num_edges_T_presel)

        np.save(out_fold+"r_den_q_"+b+"_"+str(R).replace(".","p")+".npy",r_den_q)
        np.save(out_fold+"r_den_edges_q_"+b+"_"+str(R).replace(".","p")+".npy",r_den_edges_q)
        np.save(out_fold+"r_num_q_T_"+b+"_"+str(R).replace(".","p")+".npy",r_num_q_T)
        np.save(out_fold+"r_num_edges_q_T"+b+"_"+str(R).replace(".","p")+".npy",r_num_edges_q_T)
        np.save(out_fold+"r_num_q_T_presel_"+b+"_"+str(R).replace(".","p")+".npy",r_num_q_T_presel)
        np.save(out_fold+"r_num_edges_q_T_presel"+b+"_"+str(R).replace(".","p")+".npy",r_num_edges_q_T_presel)

        np.save(out_fold+"r_den_qq_"+b+"_"+str(R).replace(".","p")+".npy",r_den_qq)
        np.save(out_fold+"r_den_edges_qq_"+b+"_"+str(R).replace(".","p")+".npy",r_den_edges_qq)
        np.save(out_fold+"r_num_qq_T_"+b+"_"+str(R).replace(".","p")+".npy",r_num_qq_T)
        np.save(out_fold+"r_num_edges_qq_T"+b+"_"+str(R).replace(".","p")+".npy",r_num_edges_qq_T)
        np.save(out_fold+"r_num_qq_T_presel_"+b+"_"+str(R).replace(".","p")+".npy",r_num_qq_T_presel)
        np.save(out_fold+"r_num_edges_qq_T_presel"+b+"_"+str(R).replace(".","p")+".npy",r_num_edges_qq_T_presel)


##updated
def use_maps_updated(out_fold, sign, skip_odd=False, cross_maps="", full_map_name="",R=0.4,eras=[]):
    
    list_of_variables = [
        "GenLLPs*","GenBquarks*","GenHiggs*","nTagJets_0p996","nTagJets_to_LLP0*","nTagJets_to_LLP1*","nJets_to_LLP0","nJets_to_LLP1","isPreselected","pt","EventNumber",
    ]


    #Define tables
    table_m = PrettyTable([
        'signal',
        'true LLP0',
        'pred LLP0',
        'diff LLP0',
        'true LLP1',
        'pred LLP1',
        'diff LLP1',
        'pred both',
        'diff both'
    ])


    table_m_presel = PrettyTable([
        'signal',
        'true LLP0',
        'pred LLP0',
        'diff LLP0',
        'true LLP1',
        'pred LLP1',
        'diff LLP1',
        'pred both',
        'diff both'
    ])


    table_r_q_LLP0 = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1 LLP0',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2 LLP0',
    ])

    table_r_q_LLP0_presel = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1 LLP0',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2 LLP0',
    ])



    table_r_qq_LLP0 = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2',
        'true sum LLP0',
        'pred sum LLP0',
        'diff sum',
    ])

    table_r_qq_LLP0_presel = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2',
        'true sum LLP0',
        'pred sum LLP0',
        'diff sum',
    ])




    #LLP1
    table_r_q_LLP1 = PrettyTable([
        'signal',
        'true q1 LLP1',
        'pred q1 LLP1',
        'diff q1 LLP1',
        'true q2 LLP1',
        'pred q2 LLP1',
        'diff q2 LLP1',
    ])

    table_r_q_LLP1_presel = PrettyTable([
        'signal',
        'true q1 LLP1',
        'pred q1 LLP1',
        'diff q1 LLP1',
        'true q2 LLP1',
        'pred q2 LLP1',
        'diff q2 LLP1',
    ])



    table_r_qq_LLP1 = PrettyTable([
        'signal',
        'true q1 LLP1',
        'pred q1 LLP1',
        'diff q1',
        'true q2 LLP1',
        'pred q2 LLP1',
        'diff q2',
        'true sum LLP1',
        'pred sum LLP1',
        'diff sum',
    ])

    table_r_qq_LLP1_presel = PrettyTable([
        'signal',
        'true q1 LLP1',
        'pred q1 LLP1',
        'diff q1',
        'true q2 LLP1',
        'pred q2 LLP1',
        'diff q2',
        'true sum LLP1',
        'pred sum LLP1',
        'diff sum',
    ])


    table_r_qq_LLP0_cross = PrettyTable([
        'signal',
        'true q1 LLP0',
        'pred q1 LLP0',
        'diff q1',
        'true q2 LLP0',
        'pred q2 LLP0',
        'diff q2',
        'true sum LLP0',
        'pred sum LLP0',
        'diff sum',
    ])



    chunk_size = 100000
    for b in sign:

        if cross_maps=="":
            map_file = out_fold+"Tag_eff_map_"+b+"_"+str(R).replace(".","p")+".yaml" 
        else:
            if cross_maps=="_":
                map_name = b.replace("_HH",cross_maps).replace("_HZ",cross_maps).replace("_ZZ",cross_maps)
            else:
                map_name = b.replace("HH",cross_maps).replace("HZ",cross_maps).replace("ZZ",cross_maps)
            map_file = out_fold+"Tag_eff_map_"+map_name+"_"+str(R).replace(".","p")+".yaml" 
            
        if full_map_name!="":
            map_file = out_fold+"Tag_eff_map_"+full_map_name+"_"+str(R).replace(".","p")+".yaml" 

        print("Opening... ", map_file)
        with open(map_file) as f:
            maps = yaml.load(f, Loader=yaml.Loader)
            f.close()

        #open two maps, one for H and one for Z
        #map_name_H = b.replace("HZ","HH").replace("ZZ","HH")
        #map_file_H = out_fold+"Tag_eff_map_"+map_name_H+".yaml" 

        #map_name_Z = b.replace("HZ","ZZ").replace("HH","ZZ")
        #map_file_Z = out_fold+"Tag_eff_map_"+map_name_Z+".yaml" 

        #with open(map_file_H) as f:
        #    maps_H = yaml.load(f, Loader=yaml.Loader)
        #    f.close()

        #with open(map_file_Z) as f:
        #    maps_Z = yaml.load(f, Loader=yaml.Loader)
        #    f.close()


        preselected = np.array([])
        nTagJets    = np.array([])
        nTagJets_to_LLP0 = np.array([])
        nTagJets_to_LLP0_q0 = np.array([])
        nTagJets_to_LLP0_q1 = np.array([])
        nTagJets_to_LLP1 = np.array([])
        nTagJets_to_LLP1_q2 = np.array([])
        nTagJets_to_LLP1_q3 = np.array([])
        nJets_to_LLP0 = np.array([])
        nJets_to_LLP1 = np.array([])
        genLLP0_merged = np.array([])
        genLLP1_merged = np.array([])
        genLLP0_resolved = np.array([])
        genLLP1_resolved = np.array([])
        genLLP0_dR = np.array([])
        genLLP1_dR = np.array([])
        genLLP0_r = np.array([])
        genLLP1_r = np.array([])
        genLLP0_z = np.array([])
        genLLP1_z = np.array([])
        genh0_pt = np.array([])
        genh1_pt = np.array([])
        genh0_eta = np.array([])
        genh1_eta = np.array([])
        genb0_pt = np.array([])
        genb1_pt = np.array([])
        genb2_pt = np.array([])
        genb3_pt = np.array([])
        genb0_eta = np.array([])
        genb1_eta = np.array([])
        genb2_eta = np.array([])
        genb3_eta = np.array([])
        genb0_phi = np.array([])
        genb1_phi = np.array([])
        genb2_phi = np.array([])
        genb3_phi = np.array([])

        genLLP0_m = np.array([])
        genLLP1_m = np.array([])

        for era in eras:

            print("Doing era ", era)

            LUMI = -1

            if era=="2018":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]
            elif era=="2017":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]
            elif era=="2016":
                from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
                from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
                LUMI  = lumi[ "HighMET" ]["tot"]


            for i, ss in enumerate(samples[b]['files']):
                print("Uproot iterating over ", NTUPLEDIR%era +ss+ '.root', " . . . ")
                filename = NTUPLEDIR%era + ss + '.root'
                gen = uproot.iterate(filename,"tree",list_of_variables,entrysteps=chunk_size)
                for arrays in gen:
                
                    n_events = -1            
                    ##Take one event every two: to use one set as validation
                    ##step = 2 if skip_odd else 1
                    ##start = 1 if skip_odd else 0
                    ##and then apply as
                    ##tmp_preselected = arrays["isPreselected"][start:n_events:step]

                    #More safe when concatenating multiple things
                    if skip_odd:
                        skip_odd_mask = (arrays["EventNumber"] % 2)==0
                    else:
                        skip_odd_mask = (arrays["EventNumber"] != 0 )

                    tmp_preselected = arrays["isPreselected"][skip_odd_mask]

                    tmp_genLLP0_dR = np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_odd_mask] )))[0:1]
                    tmp_genLLP1_dR = np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"][skip_odd_mask] )))[1:2]

                    tmp_genLLP0_merged = (tmp_genLLP0_dR < R)
                    tmp_genLLP1_merged = (tmp_genLLP1_dR < R)

                    tmp_genLLP0_resolved = (tmp_genLLP0_dR >= R)#np.logical_not(genLLP0_merged)
                    tmp_genLLP1_resolved = (tmp_genLLP1_dR >= R)#np.logical_not(genLLP1_merged)

                    tmp_genLLP0_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_odd_mask]/100. )))[0:1]
                    tmp_genLLP1_r = np.transpose(np.stack(( arrays["GenLLPs.travelRadius"][skip_odd_mask]/100. )))[1:2]

                    tmp_genLLP0_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_odd_mask]/100. )))[0:1] )
                    tmp_genLLP1_z = np.absolute( np.transpose(np.stack(( arrays["GenLLPs.travelZ"][skip_odd_mask]/100. )))[1:2] )

                    tmp_genh0_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_odd_mask] )))[0:1]
                    tmp_genh1_pt = np.transpose(np.stack(( arrays["GenHiggs.pt"][skip_odd_mask] )))[1:2]

                    tmp_genh0_id = np.transpose(np.stack(( arrays["GenHiggs.pdgId"][skip_odd_mask] )))[0:1]
                    tmp_genh1_id = np.transpose(np.stack(( arrays["GenHiggs.pdgId"][skip_odd_mask] )))[1:2]

                    tmp_genh0_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_odd_mask] )))[0:1] )
                    tmp_genh1_eta = np.absolute( np.transpose(np.stack(( arrays["GenHiggs.eta"][skip_odd_mask] )))[1:2] )

                    tmp_genb0_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_odd_mask] )))[0:1]
                    tmp_genb1_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_odd_mask] )))[1:2]
                    tmp_genb2_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_odd_mask] )))[2:3]
                    tmp_genb3_pt = np.transpose(np.stack(( arrays["GenBquarks.pt"][skip_odd_mask] )))[3:]

                    tmp_genb0_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_odd_mask] )))[0:1] )
                    tmp_genb1_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_odd_mask] )))[1:2] )
                    tmp_genb2_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_odd_mask] )))[2:3] )
                    tmp_genb3_eta = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.eta"][skip_odd_mask] )))[3:] )
 
                    tmp_genb0_phi = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.phi"][skip_odd_mask] )))[0:1] )
                    tmp_genb1_phi = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.phi"][skip_odd_mask] )))[1:2] )
                    tmp_genb2_phi = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.phi"][skip_odd_mask] )))[2:3] )
                    tmp_genb3_phi = np.absolute( np.transpose(np.stack(( arrays["GenBquarks.phi"][skip_odd_mask] )))[3:] )
                    

                    tmp_nTagJets = arrays["nTagJets_0p996"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP0 = arrays["nTagJets_to_LLP0"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP0_q0 = arrays["nTagJets_to_LLP0_q0"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP0_q1 = arrays["nTagJets_to_LLP0_q1"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP1 = arrays["nTagJets_to_LLP1"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP1_q2 = arrays["nTagJets_to_LLP1_q2"][skip_odd_mask].flatten()
                    tmp_nTagJets_to_LLP1_q3 = arrays["nTagJets_to_LLP1_q3"][skip_odd_mask].flatten()
                    tmp_nJets_to_LLP0 = arrays["nJets_to_LLP0"][skip_odd_mask].flatten()
                    tmp_nJets_to_LLP1 = arrays["nJets_to_LLP1"][skip_odd_mask].flatten()
                    
                    tmp_genLLP0_m = np.transpose(np.stack(( arrays["GenLLPs.mass"][skip_odd_mask] )))[0:1].flatten()
                    tmp_genLLP1_m = np.transpose(np.stack(( arrays["GenLLPs.mass"][skip_odd_mask] )))[1:2].flatten()

                    preselected = np.concatenate((preselected,tmp_preselected))

                    nTagJets    = np.concatenate((nTagJets,tmp_nTagJets))
                    nTagJets_to_LLP0 = np.concatenate((nTagJets_to_LLP0,tmp_nTagJets_to_LLP0))
                    nTagJets_to_LLP0_q0 = np.concatenate((nTagJets_to_LLP0_q0,tmp_nTagJets_to_LLP0_q0))
                    nTagJets_to_LLP0_q1 = np.concatenate((nTagJets_to_LLP0_q1,tmp_nTagJets_to_LLP0_q1))
                    nTagJets_to_LLP1 = np.concatenate((nTagJets_to_LLP1,tmp_nTagJets_to_LLP1))
                    nTagJets_to_LLP1_q2 = np.concatenate((nTagJets_to_LLP1_q2,tmp_nTagJets_to_LLP1_q2))
                    nTagJets_to_LLP1_q3 = np.concatenate((nTagJets_to_LLP1_q3,tmp_nTagJets_to_LLP1_q3))
                    nJets_to_LLP0 = np.concatenate((nJets_to_LLP0,tmp_nJets_to_LLP0))
                    nJets_to_LLP1 = np.concatenate((nJets_to_LLP1,tmp_nJets_to_LLP1))

                    genLLP0_merged = np.concatenate((genLLP0_merged,tmp_genLLP0_merged.flatten()))
                    genLLP1_merged = np.concatenate((genLLP1_merged,tmp_genLLP1_merged.flatten()))
                    genLLP0_resolved = np.concatenate((genLLP0_resolved,tmp_genLLP0_resolved.flatten()))
                    genLLP1_resolved = np.concatenate((genLLP1_resolved,tmp_genLLP1_resolved.flatten()))
                    genLLP0_dR = np.concatenate((genLLP0_dR,tmp_genLLP0_dR.flatten()))
                    genLLP1_dR = np.concatenate((genLLP1_dR,tmp_genLLP1_dR.flatten()))
                    genLLP0_r = np.concatenate((genLLP0_r,tmp_genLLP0_r.flatten()))
                    genLLP1_r = np.concatenate((genLLP1_r,tmp_genLLP1_r.flatten()))
                    genLLP0_z = np.concatenate((genLLP0_z,tmp_genLLP0_z.flatten()))
                    genLLP1_z = np.concatenate((genLLP1_z,tmp_genLLP1_z.flatten()))
                    genh0_pt = np.concatenate((genh0_pt,tmp_genh0_pt.flatten()))
                    genh1_pt = np.concatenate((genh1_pt,tmp_genh1_pt.flatten()))
                    genh0_eta = np.concatenate((genh0_eta,tmp_genh0_eta.flatten()))
                    genh1_eta = np.concatenate((genh1_eta,tmp_genh1_eta.flatten()))
                    genb0_pt = np.concatenate((genb0_pt,tmp_genb0_pt.flatten()))
                    genb1_pt = np.concatenate((genb1_pt,tmp_genb1_pt.flatten()))
                    genb2_pt = np.concatenate((genb2_pt,tmp_genb2_pt.flatten()))
                    genb3_pt = np.concatenate((genb3_pt,tmp_genb3_pt.flatten()))
                    genb0_eta = np.concatenate((genb0_eta,tmp_genb0_eta.flatten()))
                    genb1_eta = np.concatenate((genb1_eta,tmp_genb1_eta.flatten()))
                    genb2_eta = np.concatenate((genb2_eta,tmp_genb2_eta.flatten()))
                    genb3_eta = np.concatenate((genb3_eta,tmp_genb3_eta.flatten()))

                    genb0_phi = np.concatenate((genb0_phi,tmp_genb0_phi.flatten()))
                    genb1_phi = np.concatenate((genb1_phi,tmp_genb1_phi.flatten()))
                    genb2_phi = np.concatenate((genb2_phi,tmp_genb2_phi.flatten()))
                    genb3_phi = np.concatenate((genb3_phi,tmp_genb3_phi.flatten()))

                    genLLP0_m = np.concatenate((genLLP0_m,tmp_genLLP0_m.flatten()))
                    genLLP1_m = np.concatenate((genLLP1_m,tmp_genLLP1_m.flatten()))

        #Easiest way: count how many tagged jets are associated to each LLP
        #include also an event-level flag for pre-selections
        LLP0_m = genLLP0_m.flatten()#np.transpose(np.stack(( arrays["GenLLPs.mass"][skip_odd_mask] )))[0:1].flatten()
        LLP0_dR = genLLP0_dR.flatten()
        LLP0_r = genLLP0_r.flatten()
        LLP0_z = genLLP0_z.flatten()
        LLP0_pt = genh0_pt.flatten()
        LLP0_eta = genh0_eta.flatten()
        LLP0_pt1 = genb0_pt.flatten()
        LLP0_eta1 = genb0_eta.flatten()
        LLP0_phi1 = genb0_phi.flatten()
        LLP0_pt2 = genb1_pt.flatten()
        LLP0_eta2 = genb1_eta.flatten()
        LLP0_phi2 = genb1_phi.flatten()


        LLP1_m = genLLP1_m.flatten()#np.transpose(np.stack(( arrays["GenLLPs.mass"][skip_odd_mask] )))[1:2].flatten()
        LLP1_dR = genLLP1_dR.flatten()
        LLP1_r = genLLP1_r.flatten()
        LLP1_z = genLLP1_z.flatten()
        LLP1_pt = genh1_pt.flatten()
        LLP1_eta = genh1_eta.flatten()
        LLP1_pt1 = genb2_pt.flatten()
        LLP1_eta1 = genb2_eta.flatten()
        LLP1_pt2 = genb3_pt.flatten()
        LLP1_eta2 = genb3_eta.flatten()


        #Here use the maps!
        #Inputs are simple arrays
        
        LLP0_merged = (LLP0_dR < R)
        LLP1_merged = (LLP1_dR < R)

        LLP0_resolved = (LLP0_dR >= R)#np.logical_not(LLP0_merged)
        LLP1_resolved = (LLP1_dR >= R)#np.logical_not(LLP1_merged)

        #print "Acceptance cut missing!!!"
        acc_LLP0_merged = np.logical_and( np.logical_and(LLP0_pt>30.,abs(LLP0_eta)<1.) , LLP0_merged)
        acc_LLP1_merged = np.logical_and( np.logical_and(LLP1_pt>30.,abs(LLP1_eta)<1.) , LLP1_merged)

        LLP0_merged = np.logical_and(LLP0_merged,acc_LLP0_merged)
        LLP1_merged = np.logical_and(LLP1_merged,acc_LLP1_merged)

        nTagLLP0_m = nTagJets_to_LLP0[LLP0_merged]
        nTagLLP1_m = nTagJets_to_LLP1[LLP1_merged]

        nTagLLP0_m_presel = nTagJets_to_LLP0[np.logical_and(preselected,LLP0_merged)]
        nTagLLP1_m_presel = nTagJets_to_LLP1[np.logical_and(preselected,LLP1_merged)]

        #resolved acceptance here....
        #LLP0
        #Only q1 in acceptance
        acc_LLP0_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.), 
                np.logical_not( np.logical_and( LLP0_pt2>30.,abs(LLP0_eta2)<1.) )
            ), 
            LLP0_resolved)
        #Only q2 in acceptance
        acc_LLP0_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.) ), 
                np.logical_and(LLP0_pt2>30.,abs(LLP0_eta2)<1.)
            ), 
            LLP0_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP0_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP0_pt1>30.,abs(LLP0_eta1)<1.), 
                np.logical_and(LLP0_pt2>30.,abs(LLP0_eta2)<1.)
            ), 
            LLP0_resolved)

        #LLP1
        #Only q1 in acceptance
        acc_LLP1_resolved_q1 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.), 
                np.logical_not( np.logical_and( LLP1_pt2>30.,abs(LLP1_eta2)<1.) )
            ), 
            LLP1_resolved)
        #Only q2 in acceptance
        acc_LLP1_resolved_q2 = np.logical_and( 
            np.logical_and( 
                np.logical_not( np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.) ), 
                np.logical_and(LLP1_pt2>30.,abs(LLP1_eta2)<1.)
            ), 
            LLP1_resolved)
        #Both q1 and q2 in acceptance
        acc_LLP1_resolved_q1q2 = np.logical_and( 
            np.logical_and( 
                np.logical_and(LLP1_pt1>30.,abs(LLP1_eta1)<1.), 
                np.logical_and(LLP1_pt2>30.,abs(LLP1_eta2)<1.)
            ), 
            LLP1_resolved)

        LLP0_resolved_q1 = np.logical_and(acc_LLP0_resolved_q1,LLP0_resolved)
        LLP0_resolved_q2 = np.logical_and(acc_LLP0_resolved_q2,LLP0_resolved)
        LLP0_resolved_q1q2 = np.logical_and(acc_LLP0_resolved_q1q2,LLP0_resolved)

        LLP1_resolved_q1 = np.logical_and(acc_LLP1_resolved_q1,LLP1_resolved)
        LLP1_resolved_q2 = np.logical_and(acc_LLP1_resolved_q2,LLP1_resolved)
        LLP1_resolved_q1q2 = np.logical_and(acc_LLP1_resolved_q1q2,LLP1_resolved)

        #nTagLLP0_r_q1_old = nTagJets_to_LLP0[LLP0_resolved_q1]
        #nTagLLP0_r_q2_old = nTagJets_to_LLP0[LLP0_resolved_q2]
        nTagLLP0_r_q_q1 = nTagJets_to_LLP0_q0[LLP0_resolved_q1]
        nTagLLP0_r_q_q2 = nTagJets_to_LLP0_q1[LLP0_resolved_q2]

        #nTagLLP1_r_q1_old = nTagJets_to_LLP1[LLP1_resolved_q1]
        #nTagLLP1_r_q2_old = nTagJets_to_LLP1[LLP1_resolved_q2]
        nTagLLP1_r_q_q1 = nTagJets_to_LLP1_q2[LLP1_resolved_q1]
        nTagLLP1_r_q_q2 = nTagJets_to_LLP1_q3[LLP1_resolved_q2]

        nTagLLP0_r_q_q1_presel    = nTagJets_to_LLP0_q0[np.logical_and(preselected,LLP0_resolved_q1)]
        nTagLLP0_r_q_q2_presel    = nTagJets_to_LLP0_q1[np.logical_and(preselected,LLP0_resolved_q2)]

        nTagLLP1_r_q_q1_presel    = nTagJets_to_LLP1[np.logical_and(preselected,LLP1_resolved_q1)]
        nTagLLP1_r_q_q2_presel    = nTagJets_to_LLP1[np.logical_and(preselected,LLP1_resolved_q2)]

        nTagLLP0_r_qq_q1 = nTagJets_to_LLP0_q0[LLP0_resolved_q1q2]
        nTagLLP0_r_qq_q2 = nTagJets_to_LLP0_q1[LLP0_resolved_q1q2]
        nTagLLP1_r_qq_q1 = nTagJets_to_LLP1_q2[LLP1_resolved_q1q2]
        nTagLLP1_r_qq_q2 = nTagJets_to_LLP1_q3[LLP1_resolved_q1q2]

        nTagLLP0_r_qq_q1_presel = nTagJets_to_LLP0_q0[np.logical_and(preselected,LLP0_resolved_q1q2)]
        nTagLLP0_r_qq_q2_presel = nTagJets_to_LLP0_q1[np.logical_and(preselected,LLP0_resolved_q1q2)]
        nTagLLP1_r_qq_q1_presel = nTagJets_to_LLP1_q2[np.logical_and(preselected,LLP1_resolved_q1q2)]
        nTagLLP1_r_qq_q2_presel = nTagJets_to_LLP1_q3[np.logical_and(preselected,LLP1_resolved_q1q2)]

        ##printouts

        print "m"
        print LLP0_r[LLP0_merged]
        print LLP0_z[LLP0_merged]
        print LLP0_pt[LLP0_merged]
        print LLP0_eta[LLP0_merged]
        print LLP0_pt1[LLP0_merged]
        print LLP0_eta1[LLP0_merged]
        print LLP0_phi1[LLP0_merged]
        print LLP0_pt2[LLP0_merged]
        print LLP0_eta2[LLP0_merged]
        print LLP0_phi2[LLP0_merged]

        print "q1"
        print LLP0_r[LLP0_resolved_q1]
        print LLP0_z[LLP0_resolved_q1]
        print LLP0_pt[LLP0_resolved_q1]
        print LLP0_eta[LLP0_resolved_q1]
        print LLP0_pt1[LLP0_resolved_q1]
        print LLP0_eta1[LLP0_resolved_q1]
        print LLP0_phi1[LLP0_resolved_q1]
        print LLP0_pt2[LLP0_resolved_q1]
        print LLP0_eta2[LLP0_resolved_q1]
        print LLP0_phi2[LLP0_resolved_q1]

        print "q1q2"
        print LLP0_r[LLP0_resolved_q1q2]
        print LLP0_z[LLP0_resolved_q1q2]
        print LLP0_pt[LLP0_resolved_q1q2]
        print LLP0_eta[LLP0_resolved_q1q2]
        print LLP0_pt1[LLP0_resolved_q1q2]
        print LLP0_eta1[LLP0_resolved_q1q2]
        print LLP0_phi1[LLP0_resolved_q1q2]
        print LLP0_pt2[LLP0_resolved_q1q2]
        print LLP0_eta2[LLP0_resolved_q1q2]
        print LLP0_phi2[LLP0_resolved_q1q2]

        exit()


        #Merged pred.


        print("\n")
        print("   M e r g e d")
        print("\n")

        m_LLP0_r = LLP0_r[LLP0_merged]
        m_LLP0_z = LLP0_z[LLP0_merged]
        m_LLP0_pt = LLP0_pt[LLP0_merged]

        m_LLP1_r = LLP1_r[LLP1_merged]
        m_LLP1_z = LLP1_z[LLP1_merged]
        m_LLP1_pt = LLP1_pt[LLP1_merged]

        #TODO
        bins_r_merged =  maps['merged']['bins_r']
        bins_z_merged = maps['merged']['bins_z']
        bins_pt_merged = maps['merged']['bins_pt']

        print(bins_r_merged)

        #bins_r_merged_H =  maps_H['merged']['bins_r']
        #bins_z_merged_H = maps_H['merged']['bins_z']
        #bins_pt_merged_H = maps_H['merged']['bins_pt']

        #bins_r_merged_Z =  maps_Z['merged']['bins_r']
        #bins_z_merged_Z = maps_Z['merged']['bins_z']
        #bins_pt_merged_Z = maps_Z['merged']['bins_pt']

        #digitize is the right function to use
        m_LLP0_r_idx = np.digitize(m_LLP0_r,bins_r_merged)-1
        m_LLP0_z_idx = np.digitize(m_LLP0_z,bins_z_merged)-1
        m_LLP0_pt_idx = np.digitize(m_LLP0_pt,bins_pt_merged)-1


        #m_LLP0_r_idx_H = np.digitize(m_LLP0_r,bins_r_merged_H)-1
        #m_LLP0_z_idx_H = np.digitize(m_LLP0_z,bins_z_merged_H)-1
        #m_LLP0_pt_idx_H = np.digitize(m_LLP0_pt,bins_pt_merged_H)-1

        m_LLP1_r_idx = np.digitize(m_LLP1_r,bins_r_merged)-1
        m_LLP1_z_idx = np.digitize(m_LLP1_z,bins_z_merged)-1
        m_LLP1_pt_idx = np.digitize(m_LLP1_pt,bins_pt_merged)-1

        #m_LLP1_r_idx_Z = np.digitize(m_LLP1_r,bins_r_merged_Z)-1
        #m_LLP1_z_idx_Z = np.digitize(m_LLP1_z,bins_z_merged_Z)-1
        #m_LLP1_pt_idx_Z = np.digitize(m_LLP1_pt,bins_pt_merged_Z)-1


        nTagLLP0_pred_m = 0
        nTagLLP1_pred_m = 0

        nTagLLP0_pred_m_presel = 0
        nTagLLP1_pred_m_presel = 0

        #for plotting purposes
        m_LLP0_eff = []
        m_LLP0_eff_presel = []

        print("bug!!")
        print(m_LLP0_r_idx)
        print(m_LLP0_z_idx)
        print(m_LLP0_pt_idx)
        print(len(m_LLP0_r_idx))
        print(len(m_LLP0_z_idx))
        print(len(m_LLP0_pt_idx))

        for it in range(len(m_LLP0_r_idx)):
            #print('r',  m_LLP0_r_idx[it])
            #print(m_LLP0_r  [ it ])
            #print(bins_r_merged)
            #print('z',  m_LLP0_z_idx[it])
            #print('pt', m_LLP0_pt_idx[it])
            pred = maps['merged']['eff'][ m_LLP0_r_idx[it] ][  m_LLP0_z_idx[it] ][  m_LLP0_pt_idx[it] ]
            m_LLP0_eff.append(pred)
            nTagLLP0_pred_m+= pred

            pred_presel = maps['merged']['eff_presel'][ m_LLP0_r_idx[it] ][  m_LLP0_z_idx[it] ][  m_LLP0_pt_idx[it] ]
            m_LLP0_eff_presel.append(pred_presel)
            nTagLLP0_pred_m_presel+= pred_presel

        m_LLP0_eff = np.array(m_LLP0_eff)
        m_LLP0_eff_presel = np.array(m_LLP0_eff_presel)

        #cond = np.logical_and(m_LLP0_r>0, m_LLP0_z>0)
        #X = m_LLP0_r[cond]#[0:1000]
        #Y = m_LLP0_z[cond]#[0:1000]
        #Z = m_LLP0_pt[cond]#[0:1000]
        #V = m_LLP0_eff[cond]#[0:1000]

        #np.save(out_fold+"X_m_"+b+"_"+str(R).replace(".","p")+".npy",X)
        #np.save(out_fold+"Y_m_"+b+"_"+str(R).replace(".","p")+".npy",Y)
        #np.save(out_fold+"Z_m_"+b+"_"+str(R).replace(".","p")+".npy",Z)
        #np.save(out_fold+"V_m_"+b+"_"+str(R).replace(".","p")+".npy",V)

        #Why have we done this?

        #exit()


        '''
        # here are the x,y and respective z values
        X, Y = np.meshgrid(m_LLP0_r[0:1000], m_LLP0_z[0:1000])
        Z = m_LLP0_pt[0:1000]
        # this is the value to use for the color
        V = m_LLP0_eff[0:1000]

        # create the figure, add a 3d axis, set the viewing angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45,60)

        # here we create the surface plot, but pass V through a colormap
        # to create a different color for each patch
        ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X, Y, Z, c=V)
        
        plt.savefig("test_maps.pdf")
        plt.savefig("test_maps.png")
        '''

        for it in range(len(m_LLP1_r_idx)):
            pred = maps['merged']['eff'][ m_LLP1_r_idx[it] ][  m_LLP1_z_idx[it] ][  m_LLP1_pt_idx[it] ]
            nTagLLP1_pred_m+= pred

            pred_presel = maps['merged']['eff_presel'][ m_LLP1_r_idx[it] ][  m_LLP1_z_idx[it] ][  m_LLP1_pt_idx[it] ]
            nTagLLP1_pred_m_presel+= pred_presel


        t_nTagLLP0_m = nTagLLP0_m[nTagLLP0_m>0].astype(bool).sum()
        t_nTagLLP0_m_unc = 100./np.sqrt(t_nTagLLP0_m) if t_nTagLLP0_m>0 else 0
        t_diff_nTagLLP0_m = 100*( (nTagLLP0_pred_m-t_nTagLLP0_m)/ ( t_nTagLLP0_m ) ) if t_nTagLLP0_m>0 else 0

        t_nTagLLP1_m = nTagLLP1_m[nTagLLP1_m>0].astype(bool).sum()
        t_nTagLLP1_m_unc = 100./np.sqrt(t_nTagLLP1_m) if t_nTagLLP1_m>0 else 0
        t_diff_nTagLLP1_m = 100*( (nTagLLP1_pred_m-t_nTagLLP1_m)/ ( t_nTagLLP1_m ) ) if t_nTagLLP1_m>0 else 0

        t_nTag_m = t_nTagLLP0_m + t_nTagLLP1_m
        t_nTag_m_unc = 100./np.sqrt(t_nTag_m) if t_nTag_m>0 else 0
        t_nTag_m_diff = 100.*(nTagLLP0_pred_m + nTagLLP1_pred_m - t_nTag_m)/t_nTag_m if t_nTag_m>0 else 0

        row = [
            b,
            str(t_nTagLLP0_m) + " +- " + str( round(t_nTagLLP0_m_unc, 2) ) + "%",
            str( round(nTagLLP0_pred_m, 2) ),
            str( round(t_diff_nTagLLP0_m, 2) ) + "%",


            str(t_nTagLLP1_m) + " +- " + str( round(t_nTagLLP1_m_unc, 2) ) + "%",
            str( round(nTagLLP1_pred_m, 2) ),
            str( round(t_diff_nTagLLP1_m, 2) ) + "%",
        
            str( round(nTagLLP0_pred_m+nTagLLP1_pred_m,2) ),
            str( round( t_nTag_m_diff, 2) ) + "%",
        ]
        table_m.add_row(row)

        #Preselections

        t_nTagLLP0_m_presel = nTagLLP0_m_presel[nTagLLP0_m_presel>0].astype(bool).sum()
        t_nTagLLP0_m_presel_unc = 100./np.sqrt(t_nTagLLP0_m_presel) if t_nTagLLP0_m_presel>0 else 0
        t_diff_nTagLLP0_m_presel = 100*( (nTagLLP0_pred_m_presel-t_nTagLLP0_m_presel)/ ( t_nTagLLP0_m_presel ) ) if t_nTagLLP0_m_presel>0 else 0

        t_nTagLLP1_m_presel = nTagLLP1_m_presel[nTagLLP1_m_presel>0].astype(bool).sum()
        t_nTagLLP1_m_presel_unc = 100./np.sqrt(t_nTagLLP1_m_presel) if t_nTagLLP1_m_presel>0 else 0
        t_diff_nTagLLP1_m_presel = 100*( (nTagLLP1_pred_m_presel-t_nTagLLP1_m_presel)/ ( t_nTagLLP1_m_presel ) ) if t_nTagLLP1_m_presel>0 else 0

        t_nTag_m_presel = t_nTagLLP0_m_presel + t_nTagLLP1_m_presel
        t_nTag_m_presel_unc = 100./np.sqrt(t_nTag_m_presel) if t_nTag_m_presel>0 else 0
        t_nTag_m_presel_diff = 100.*(nTagLLP0_pred_m_presel + nTagLLP1_pred_m_presel - t_nTag_m_presel)/t_nTag_m_presel if t_nTag_m_presel>0 else 0

        row_presel = [
            b,
            str(t_nTagLLP0_m_presel) + " +- " + str( round(t_nTagLLP0_m_presel_unc, 2) ) + "%",
            str( round(nTagLLP0_pred_m_presel, 2) ),
            str( round(t_diff_nTagLLP0_m_presel, 2) ) + "%",


            str(t_nTagLLP1_m_presel) + " +- " + str( round(t_nTagLLP1_m_presel_unc, 2) ) + "%",
            str( round(nTagLLP1_pred_m_presel, 2) ),
            str( round(t_diff_nTagLLP1_m_presel, 2) ) + "%",
        
            str( round(nTagLLP0_pred_m_presel+nTagLLP1_pred_m_presel,2) ),
            str( round( t_nTag_m_presel_diff, 2) ) + "%",
        ]
        table_m_presel.add_row(row_presel)



        print("\n")
        print("   R e s o l v e d")
        print("\n")

        #resolved pred 
        r_LLP0_r_q1 = LLP0_r[LLP0_resolved_q1]
        r_LLP0_z_q1 = LLP0_z[LLP0_resolved_q1]
        r_LLP0_pt_q1 = LLP0_pt1[LLP0_resolved_q1]

        r_LLP0_r_q2 = LLP0_r[LLP0_resolved_q2]
        r_LLP0_z_q2 = LLP0_z[LLP0_resolved_q2]
        r_LLP0_pt_q2 = LLP0_pt2[LLP0_resolved_q2]

        r_LLP0_r_q1q2 = LLP0_r[LLP0_resolved_q1q2]
        r_LLP0_z_q1q2 = LLP0_z[LLP0_resolved_q1q2]
        r_LLP0_pt1_q1q2 = LLP0_pt1[LLP0_resolved_q1q2]
        r_LLP0_pt2_q1q2 = LLP0_pt2[LLP0_resolved_q1q2]

        r_LLP1_r_q1 = LLP1_r[LLP1_resolved_q1]
        r_LLP1_z_q1 = LLP1_z[LLP1_resolved_q1]
        r_LLP1_pt_q1 = LLP1_pt1[LLP1_resolved_q1]

        r_LLP1_r_q2 = LLP1_r[LLP1_resolved_q2]
        r_LLP1_z_q2 = LLP1_z[LLP1_resolved_q2]
        r_LLP1_pt_q2 = LLP1_pt2[LLP1_resolved_q2]

        r_LLP1_r_q1q2 = LLP1_r[LLP1_resolved_q1q2]
        r_LLP1_z_q1q2 = LLP1_z[LLP1_resolved_q1q2]
        r_LLP1_pt1_q1q2 = LLP1_pt1[LLP1_resolved_q1q2]
        r_LLP1_pt2_q1q2 = LLP1_pt2[LLP1_resolved_q1q2]

        bins_r_resolved =  maps['resolved']['bins_r']
        bins_z_resolved = maps['resolved']['bins_z']
        bins_pt_resolved_q = maps['resolved']['bins_pt']

        #digitize is the right function to use
        r_LLP0_r_idx_q1    = np.digitize(r_LLP0_r_q1,bins_r_resolved)-1
        r_LLP0_z_idx_q1    = np.digitize(r_LLP0_z_q1,bins_z_resolved)-1
        r_LLP0_pt_idx_q1   = np.digitize(r_LLP0_pt_q1,bins_pt_resolved_q)-1

        r_LLP0_r_idx_q2    = np.digitize(r_LLP0_r_q2,bins_r_resolved)-1
        r_LLP0_z_idx_q2    = np.digitize(r_LLP0_z_q2,bins_z_resolved)-1
        r_LLP0_pt_idx_q2   = np.digitize(r_LLP0_pt_q2,bins_pt_resolved_q)-1

        r_LLP0_r_idx_q1q2     = np.digitize(r_LLP0_r_q1q2,bins_r_resolved)-1
        r_LLP0_z_idx_q1q2     = np.digitize(r_LLP0_z_q1q2,bins_z_resolved)-1
        r_LLP0_pt1_idx_q1q2   = np.digitize(r_LLP0_pt1_q1q2,bins_pt_resolved_q)-1
        r_LLP0_pt2_idx_q1q2   = np.digitize(r_LLP0_pt2_q1q2,bins_pt_resolved_q)-1

        r_LLP1_r_idx_q1    = np.digitize(r_LLP1_r_q1,bins_r_resolved)-1
        r_LLP1_z_idx_q1    = np.digitize(r_LLP1_z_q1,bins_z_resolved)-1
        r_LLP1_pt_idx_q1   = np.digitize(r_LLP1_pt_q1,bins_pt_resolved_q)-1

        r_LLP1_r_idx_q2    = np.digitize(r_LLP1_r_q2,bins_r_resolved)-1
        r_LLP1_z_idx_q2    = np.digitize(r_LLP1_z_q2,bins_z_resolved)-1
        r_LLP1_pt_idx_q2   = np.digitize(r_LLP1_pt_q2,bins_pt_resolved_q)-1

        r_LLP1_r_idx_q1q2     = np.digitize(r_LLP1_r_q1q2,bins_r_resolved)-1
        r_LLP1_z_idx_q1q2     = np.digitize(r_LLP1_z_q1q2,bins_z_resolved)-1
        r_LLP1_pt1_idx_q1q2   = np.digitize(r_LLP1_pt1_q1q2,bins_pt_resolved_q)-1
        r_LLP1_pt2_idx_q1q2   = np.digitize(r_LLP1_pt2_q1q2,bins_pt_resolved_q)-1

        #LLP0 predictions
        nTagLLP0_pred_r_q1 = 0
        nTagLLP0_pred_r_q2 = 0
        nTagLLP0_pred_r_qq_q1 = 0
        nTagLLP0_pred_r_qq_q2 = 0
        nTagLLP0_pred_r_qq_q1_cross = 0
        nTagLLP0_pred_r_qq_q2_cross = 0

        #presel
        nTagLLP0_pred_r_q1_presel = 0
        nTagLLP0_pred_r_q2_presel = 0
        nTagLLP0_pred_r_qq_q1_presel = 0
        nTagLLP0_pred_r_qq_q2_presel = 0

        for it in range(len(r_LLP0_r_idx_q1)):
            nTagLLP0_pred_r_q1 += maps['resolved']['eff_q'][ r_LLP0_r_idx_q1[it] ][  r_LLP0_z_idx_q1[it] ][  r_LLP0_pt_idx_q1[it] ]
            pred_presel = maps['resolved']['eff_q_presel'] [ r_LLP0_r_idx_q1[it] ][  r_LLP0_z_idx_q1[it] ][  r_LLP0_pt_idx_q1[it] ]
            nTagLLP0_pred_r_q1_presel+= pred_presel


        for it in range(len(r_LLP0_r_idx_q2)):
            nTagLLP0_pred_r_q2 += maps['resolved']['eff_q'][ r_LLP0_r_idx_q2[it] ][  r_LLP0_z_idx_q2[it] ][  r_LLP0_pt_idx_q2[it] ]
            pred_presel = maps['resolved']['eff_q_presel'] [ r_LLP0_r_idx_q2[it] ][  r_LLP0_z_idx_q2[it] ][  r_LLP0_pt_idx_q2[it] ]
            nTagLLP0_pred_r_q2_presel+= pred_presel


        for it in range(len(r_LLP0_r_idx_q1q2)):
            pred_q1 = maps['resolved']['eff_qq'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt1_idx_q1q2[it] ]
            pred_q2 = maps['resolved']['eff_qq'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt2_idx_q1q2[it] ]
            nTagLLP0_pred_r_qq_q1 += pred_q1
            nTagLLP0_pred_r_qq_q2 += pred_q2

            pred_q1_presel = maps['resolved']['eff_qq_presel'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt1_idx_q1q2[it] ]
            pred_q2_presel = maps['resolved']['eff_qq_presel'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt2_idx_q1q2[it] ]
            nTagLLP0_pred_r_qq_q1_presel += pred_q1_presel
            nTagLLP0_pred_r_qq_q2_presel += pred_q2_presel

            #cross
            pred_q1_cross = maps['resolved']['eff_q'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt1_idx_q1q2[it] ]
            pred_q2_cross = maps['resolved']['eff_q'][ r_LLP0_r_idx_q1q2[it] ][  r_LLP0_z_idx_q1q2[it] ][  r_LLP0_pt2_idx_q1q2[it] ]
            nTagLLP0_pred_r_qq_q1_cross += pred_q1_cross
            nTagLLP0_pred_r_qq_q2_cross += pred_q2_cross
            



        #with correct numerator!
        t_nTagLLP0_r_q_q1 = nTagLLP0_r_q_q1[nTagLLP0_r_q_q1>0].astype(bool).sum()
        t_nTagLLP0_r_q_q1_unc = 100./np.sqrt(t_nTagLLP0_r_q_q1) if t_nTagLLP0_r_q_q1>0 else 0
        t_diff_nTagLLP0_r_q_q1 = 100*( (nTagLLP0_pred_r_q1-t_nTagLLP0_r_q_q1)/ ( t_nTagLLP0_r_q_q1 ) )

        #with correct numerator!
        t_nTagLLP0_r_q_q2 = nTagLLP0_r_q_q2[nTagLLP0_r_q_q2>0].astype(bool).sum()
        t_nTagLLP0_r_q_q2_unc = 100./np.sqrt(t_nTagLLP0_r_q_q2) if t_nTagLLP0_r_q_q2>0 else 0
        t_diff_nTagLLP0_r_q_q2 = 100*( (nTagLLP0_pred_r_q2-t_nTagLLP0_r_q_q2)/ ( t_nTagLLP0_r_q_q2 ) )

        t_nTagLLP0_r_q_tot = t_nTagLLP0_r_q_q1 + t_nTagLLP0_r_q_q2
        t_nTagLLP0_r_q_tot_unc = 100./np.sqrt(t_nTagLLP0_r_q_tot) if t_nTagLLP0_r_q_tot>0 else 0
        t_diff_nTagLLP0_r_q_tot = 100*( ( nTagLLP0_pred_r_q1 + nTagLLP0_pred_r_q2  -t_nTagLLP0_r_q_tot)/t_nTagLLP0_r_q_tot  )

        row = [
            b,
            str(t_nTagLLP0_r_q_q1) + " +- " + str( round( t_nTagLLP0_r_q_q1_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q1,2) ),
            str( round(t_diff_nTagLLP0_r_q_q1, 2) ) + "%",

            str(t_nTagLLP0_r_q_q2) + " +- " + str( round( t_nTagLLP0_r_q_q2_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q2,2) ),
            str( round(t_diff_nTagLLP0_r_q_q2, 2) ) + "%",

        ]

        table_r_q_LLP0.add_row(row)


        t_nTagLLP0_r_qq_q1 = nTagLLP0_r_qq_q1[nTagLLP0_r_qq_q1>0].astype(bool).sum()
        t_nTagLLP0_r_qq_q1_unc = 100./np.sqrt(t_nTagLLP0_r_qq_q1) if t_nTagLLP0_r_qq_q1>0 else 0
        t_diff_nTagLLP0_r_qq_q1 = 100*( (nTagLLP0_pred_r_qq_q1 - t_nTagLLP0_r_qq_q1)/t_nTagLLP0_r_qq_q1 )
        t_diff_nTagLLP0_r_qq_q1_cross = 100*( (nTagLLP0_pred_r_qq_q1_cross - t_nTagLLP0_r_qq_q1)/t_nTagLLP0_r_qq_q1 )

        t_nTagLLP0_r_qq_q2 = nTagLLP0_r_qq_q2[nTagLLP0_r_qq_q2>0].astype(bool).sum()
        t_nTagLLP0_r_qq_q2_unc = 100./np.sqrt(t_nTagLLP0_r_qq_q2) if t_nTagLLP0_r_qq_q2>0 else 0
        t_diff_nTagLLP0_r_qq_q2 = 100*( (nTagLLP0_pred_r_qq_q2 - t_nTagLLP0_r_qq_q2)/t_nTagLLP0_r_qq_q2 )
        t_diff_nTagLLP0_r_qq_q2_cross = 100*( (nTagLLP0_pred_r_qq_q2_cross - t_nTagLLP0_r_qq_q2)/t_nTagLLP0_r_qq_q2 )

        t_nTagLLP0_r_qq_tot = t_nTagLLP0_r_qq_q1 + t_nTagLLP0_r_qq_q2
        t_nTagLLP0_r_qq_tot_unc = 100./np.sqrt(t_nTagLLP0_r_qq_tot) if t_nTagLLP0_r_qq_tot>0 else 0
        t_diff_nTagLLP0_r_qq_tot = 100*( (-t_nTagLLP0_r_qq_tot + nTagLLP0_pred_r_qq_q1 + nTagLLP0_pred_r_qq_q2)/t_nTagLLP0_r_qq_tot   )
        t_diff_nTagLLP0_r_qq_tot_cross = 100*( (-t_nTagLLP0_r_qq_tot + nTagLLP0_pred_r_qq_q1_cross + nTagLLP0_pred_r_qq_q2_cross)/t_nTagLLP0_r_qq_tot   )



        row = [
            b,
            #'true q1 only LLP0',
            str(t_nTagLLP0_r_qq_q1) + " +- " + str( round( t_nTagLLP0_r_qq_q1_unc, 2) ) + "%",
            #'pred q1 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q1,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_qq_q1,2)) + "%",
            #'true q2 only LLP0',
            str(t_nTagLLP0_r_qq_q2) + " +- " + str( round( t_nTagLLP0_r_qq_q2_unc, 2) ) + "%",
            #'pred q2 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q2,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_qq_q2,2)) + "%",
            #true sum
            str(t_nTagLLP0_r_qq_tot) + " +- " + str( round( t_nTagLLP0_r_qq_tot_unc,2) ) + "%",
            #'pred sum LLP0',
            str( round(nTagLLP0_pred_r_qq_q1 + nTagLLP0_pred_r_qq_q2 ,2) ),
            #diff
            str( round(t_diff_nTagLLP0_r_qq_tot,2)) + "%",
        ]

        table_r_qq_LLP0.add_row(row)


        row_cross = [
            b,
            #'true q1 only LLP0',
            str(t_nTagLLP0_r_qq_q1) + " +- " + str( round( t_nTagLLP0_r_qq_q1_unc, 2) ) + "%",
            #'pred q1 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q1_cross,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_qq_q1_cross,2)) + "%",
            #'true q2 only LLP0',
            str(t_nTagLLP0_r_qq_q2) + " +- " + str( round( t_nTagLLP0_r_qq_q2_unc, 2) ) + "%",
            #'pred q2 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q2_cross,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_qq_q2_cross,2)) + "%",
            #true sum
            str(t_nTagLLP0_r_qq_tot) + " +- " + str( round( t_nTagLLP0_r_qq_tot_unc,2) ) + "%",
            #'pred sum LLP0',
            str( round(nTagLLP0_pred_r_qq_q1_cross + nTagLLP0_pred_r_qq_q2_cross ,2) ),
            #diff
            str( round(t_diff_nTagLLP0_r_qq_tot_cross,2)) + "%",
        ]

        #table_r_qq_LLP0_cross.add_row(row_cross)


        #w preselections
        #TODO

        #with correct numerator!
        t_nTagLLP0_r_presel_q_q1 = nTagLLP0_r_q_q1_presel[nTagLLP0_r_q_q1_presel>0].astype(bool).sum()
        t_nTagLLP0_r_presel_q_q1_unc = 100./np.sqrt(t_nTagLLP0_r_presel_q_q1) if t_nTagLLP0_r_presel_q_q1>0 else 0
        t_diff_nTagLLP0_r_presel_q_q1 = 100*( (nTagLLP0_pred_r_q1_presel-t_nTagLLP0_r_presel_q_q1)/ ( t_nTagLLP0_r_presel_q_q1 ) )

        #with correct numerator!
        t_nTagLLP0_r_presel_q_q2 = nTagLLP0_r_q_q2_presel[nTagLLP0_r_q_q2_presel>0].astype(bool).sum()
        t_nTagLLP0_r_presel_q_q2_unc = 100./np.sqrt(t_nTagLLP0_r_presel_q_q2) if t_nTagLLP0_r_presel_q_q2>0 else 0
        t_diff_nTagLLP0_r_presel_q_q2 = 100*( (nTagLLP0_pred_r_q2_presel-t_nTagLLP0_r_presel_q_q2)/ ( t_nTagLLP0_r_presel_q_q2 ) )

        t_nTagLLP0_r_presel_q_tot = t_nTagLLP0_r_presel_q_q1 + t_nTagLLP0_r_presel_q_q2
        t_nTagLLP0_r_presel_q_tot_unc = 100./np.sqrt(t_nTagLLP0_r_presel_q_tot) if t_nTagLLP0_r_presel_q_tot>0 else 0
        t_diff_nTagLLP0_r_presel_q_tot = 100*( ( nTagLLP0_pred_r_q1_presel + nTagLLP0_pred_r_q2_presel  -t_nTagLLP0_r_presel_q_tot)/t_nTagLLP0_r_presel_q_tot  )

        row = [
            b,
            str(t_nTagLLP0_r_presel_q_q1) + " +- " + str( round( t_nTagLLP0_r_presel_q_q1_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q1_presel,2) ),
            str( round(t_diff_nTagLLP0_r_presel_q_q1, 2) ) + "%",

            str(t_nTagLLP0_r_presel_q_q2) + " +- " + str( round( t_nTagLLP0_r_presel_q_q2_unc,2) ) + "%",
            str(round(nTagLLP0_pred_r_q2_presel,2) ),
            str( round(t_diff_nTagLLP0_r_presel_q_q2, 2) ) + "%",

        ]

        table_r_q_LLP0_presel.add_row(row)




        t_nTagLLP0_r_presel_qq_q1 = nTagLLP0_r_qq_q1_presel[nTagLLP0_r_qq_q1_presel>0].astype(bool).sum()
        t_nTagLLP0_r_presel_qq_q1_unc = 100./np.sqrt(t_nTagLLP0_r_presel_qq_q1) if t_nTagLLP0_r_presel_qq_q1>0 else 0
        t_diff_nTagLLP0_r_presel_qq_q1 = 100*( (nTagLLP0_pred_r_qq_q1_presel - t_nTagLLP0_r_presel_qq_q1)/t_nTagLLP0_r_presel_qq_q1 )

        t_nTagLLP0_r_presel_qq_q2 = nTagLLP0_r_qq_q2_presel[nTagLLP0_r_qq_q2_presel>0].astype(bool).sum()
        t_nTagLLP0_r_presel_qq_q2_unc = 100./np.sqrt(t_nTagLLP0_r_presel_qq_q2) if t_nTagLLP0_r_presel_qq_q2>0 else 0
        t_diff_nTagLLP0_r_presel_qq_q2 = 100*( (nTagLLP0_pred_r_qq_q2_presel - t_nTagLLP0_r_presel_qq_q2)/t_nTagLLP0_r_presel_qq_q2 )

        t_nTagLLP0_r_presel_qq_tot = t_nTagLLP0_r_presel_qq_q1 + t_nTagLLP0_r_presel_qq_q2
        t_nTagLLP0_r_presel_qq_tot_unc = 100./np.sqrt(t_nTagLLP0_r_presel_qq_tot) if t_nTagLLP0_r_presel_qq_tot>0 else 0
        t_diff_nTagLLP0_r_presel_qq_tot = 100*( (-t_nTagLLP0_r_presel_qq_tot + nTagLLP0_pred_r_qq_q1_presel + nTagLLP0_pred_r_qq_q2_presel)/t_nTagLLP0_r_presel_qq_tot   )

        row = [
            b,
            #'true q1 only LLP0',
            str(t_nTagLLP0_r_presel_qq_q1) + " +- " + str( round( t_nTagLLP0_r_presel_qq_q1_unc, 2) ) + "%",
            #'pred q1 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q1_presel,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_presel_qq_q1,2)) + "%",
            #'true q2 only LLP0',
            str(t_nTagLLP0_r_presel_qq_q2) + " +- " + str( round( t_nTagLLP0_r_presel_qq_q2_unc, 2) ) + "%",
            #'pred q2 only LLP0',
            str( round(nTagLLP0_pred_r_qq_q2_presel,2) ),
            #diff
            str(round(t_diff_nTagLLP0_r_presel_qq_q2,2)) + "%",
            #true sum
            str(t_nTagLLP0_r_presel_qq_tot) + " +- " + str( round( t_nTagLLP0_r_presel_qq_tot_unc,2) ) + "%",
            #'pred sum LLP0',
            str( round(nTagLLP0_pred_r_qq_q1_presel + nTagLLP0_pred_r_qq_q2_presel ,2) ),
            #diff
            str( round(t_diff_nTagLLP0_r_presel_qq_tot,2)) + "%",
        ]

        table_r_qq_LLP0_presel.add_row(row)



        #LLP1
        #LLP1 predictions
        nTagLLP1_pred_r_q1 = 0
        nTagLLP1_pred_r_q2 = 0
        nTagLLP1_pred_r_qq_q1 = 0
        nTagLLP1_pred_r_qq_q2 = 0

        #presel
        nTagLLP1_pred_r_q1_presel = 0
        nTagLLP1_pred_r_q2_presel = 0
        nTagLLP1_pred_r_qq_q1_presel = 0
        nTagLLP1_pred_r_qq_q2_presel = 0

        for it in range(len(r_LLP1_r_idx_q1)):
            nTagLLP1_pred_r_q1 += maps['resolved']['eff_q'][ r_LLP1_r_idx_q1[it] ][  r_LLP1_z_idx_q1[it] ][  r_LLP1_pt_idx_q1[it] ]
            pred_presel = maps['resolved']['eff_q_presel'] [ r_LLP1_r_idx_q1[it] ][  r_LLP1_z_idx_q1[it] ][  r_LLP1_pt_idx_q1[it] ]
            nTagLLP1_pred_r_q1_presel+= pred_presel


        for it in range(len(r_LLP1_r_idx_q2)):
            nTagLLP1_pred_r_q2 += maps['resolved']['eff_q'][ r_LLP1_r_idx_q2[it] ][  r_LLP1_z_idx_q2[it] ][  r_LLP1_pt_idx_q2[it] ]
            pred_presel = maps['resolved']['eff_q_presel'] [ r_LLP1_r_idx_q2[it] ][  r_LLP1_z_idx_q2[it] ][  r_LLP1_pt_idx_q2[it] ]
            nTagLLP1_pred_r_q2_presel+= pred_presel


        for it in range(len(r_LLP1_r_idx_q1q2)):
            pred_q1 = maps['resolved']['eff_qq'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt1_idx_q1q2[it] ]
            pred_q2 = maps['resolved']['eff_qq'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt2_idx_q1q2[it] ]
            nTagLLP1_pred_r_qq_q1 += pred_q1
            nTagLLP1_pred_r_qq_q2 += pred_q2

            pred_q1_presel = maps['resolved']['eff_qq_presel'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt1_idx_q1q2[it] ]
            pred_q2_presel = maps['resolved']['eff_qq_presel'][ r_LLP1_r_idx_q1q2[it] ][  r_LLP1_z_idx_q1q2[it] ][  r_LLP1_pt2_idx_q1q2[it] ]
            nTagLLP1_pred_r_qq_q1_presel += pred_q1_presel
            nTagLLP1_pred_r_qq_q2_presel += pred_q2_presel


        #with correct numerator!
        t_nTagLLP1_r_q_q1 = nTagLLP1_r_q_q1[nTagLLP1_r_q_q1>0].astype(bool).sum()
        t_nTagLLP1_r_q_q1_unc = 100./np.sqrt(t_nTagLLP1_r_q_q1) if t_nTagLLP1_r_q_q1>0 else 0
        t_diff_nTagLLP1_r_q_q1 = 100*( (nTagLLP1_pred_r_q1-t_nTagLLP1_r_q_q1)/ ( t_nTagLLP1_r_q_q1 ) )

        #with correct numerator!
        t_nTagLLP1_r_q_q2 = nTagLLP1_r_q_q2[nTagLLP1_r_q_q2>0].astype(bool).sum()
        t_nTagLLP1_r_q_q2_unc = 100./np.sqrt(t_nTagLLP1_r_q_q2) if t_nTagLLP1_r_q_q2>0 else 0
        t_diff_nTagLLP1_r_q_q2 = 100*( (nTagLLP1_pred_r_q2-t_nTagLLP1_r_q_q2)/ ( t_nTagLLP1_r_q_q2 ) )


        t_nTagLLP1_r_q_tot = t_nTagLLP1_r_q_q1 + t_nTagLLP1_r_q_q2
        t_nTagLLP1_r_q_tot_unc = 100./np.sqrt(t_nTagLLP1_r_q_tot) if t_nTagLLP1_r_q_tot>0 else 0
        t_diff_nTagLLP1_r_q_tot = 100*( ( nTagLLP1_pred_r_q1 + nTagLLP1_pred_r_q2  -t_nTagLLP1_r_q_tot)/t_nTagLLP1_r_q_tot  )

        row = [
            b,
            str(t_nTagLLP1_r_q_q1) + " +- " + str( round( t_nTagLLP1_r_q_q1_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q1,2) ),
            str( round(t_diff_nTagLLP1_r_q_q1, 2) ) + "%",

            str(t_nTagLLP1_r_q_q2) + " +- " + str( round( t_nTagLLP1_r_q_q2_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q2,2) ),
            str( round(t_diff_nTagLLP1_r_q_q2, 2) ) + "%",

        ]

        table_r_q_LLP1.add_row(row)


        t_nTagLLP1_r_qq_q1 = nTagLLP1_r_qq_q1[nTagLLP1_r_qq_q1>0].astype(bool).sum()
        t_nTagLLP1_r_qq_q1_unc = 100./np.sqrt(t_nTagLLP1_r_qq_q1) if t_nTagLLP1_r_qq_q1>0 else 0
        t_diff_nTagLLP1_r_qq_q1 = 100*( (nTagLLP1_pred_r_qq_q1 - t_nTagLLP1_r_qq_q1)/t_nTagLLP1_r_qq_q1 )

        t_nTagLLP1_r_qq_q2 = nTagLLP1_r_qq_q2[nTagLLP1_r_qq_q2>0].astype(bool).sum()
        t_nTagLLP1_r_qq_q2_unc = 100./np.sqrt(t_nTagLLP1_r_qq_q2) if t_nTagLLP1_r_qq_q2>0 else 0
        t_diff_nTagLLP1_r_qq_q2 = 100*( (nTagLLP1_pred_r_qq_q2 - t_nTagLLP1_r_qq_q2)/t_nTagLLP1_r_qq_q2 )

        t_nTagLLP1_r_qq_tot = t_nTagLLP1_r_qq_q1 + t_nTagLLP1_r_qq_q2
        t_nTagLLP1_r_qq_tot_unc = 100./np.sqrt(t_nTagLLP1_r_qq_tot) if t_nTagLLP1_r_qq_tot>0 else 0
        t_diff_nTagLLP1_r_qq_tot = 100*( (-t_nTagLLP1_r_qq_tot + nTagLLP1_pred_r_qq_q1 + nTagLLP1_pred_r_qq_q2)/t_nTagLLP1_r_qq_tot   )

        row = [
            b,
            #'true q1 only LLP1',
            str(t_nTagLLP1_r_qq_q1) + " +- " + str( round( t_nTagLLP1_r_qq_q1_unc, 2) ) + "%",
            #'pred q1 only LLP1',
            str( round(nTagLLP1_pred_r_qq_q1,2) ),
            #diff
            str(round(t_diff_nTagLLP1_r_qq_q1,2)) + "%",
            #'true q2 only LLP1',
            str(t_nTagLLP1_r_qq_q2) + " +- " + str( round( t_nTagLLP1_r_qq_q2_unc, 2) ) + "%",
            #'pred q2 only LLP1',
            str( round(nTagLLP1_pred_r_qq_q2,2) ),
            #diff
            str(round(t_diff_nTagLLP1_r_qq_q2,2)) + "%",
            #true sum
            str(t_nTagLLP1_r_qq_tot) + " +- " + str( round( t_nTagLLP1_r_qq_tot_unc,2) ) + "%",
            #'pred sum LLP1',
            str( round(nTagLLP1_pred_r_qq_q1 + nTagLLP1_pred_r_qq_q2 ,2) ),
            #diff
            str( round(t_diff_nTagLLP1_r_qq_tot,2)) + "%",
        ]

        table_r_qq_LLP1.add_row(row)

        #w preselections
        #TODO

        #with correct numerator!
        t_nTagLLP1_r_presel_q_q1 = nTagLLP1_r_q_q1_presel[nTagLLP1_r_q_q1_presel>0].astype(bool).sum()
        t_nTagLLP1_r_presel_q_q1_unc = 100./np.sqrt(t_nTagLLP1_r_presel_q_q1) if t_nTagLLP1_r_presel_q_q1>0 else 0
        t_diff_nTagLLP1_r_presel_q_q1 = 100*( (nTagLLP1_pred_r_q1_presel-t_nTagLLP1_r_presel_q_q1)/ ( t_nTagLLP1_r_presel_q_q1 ) )

        #with correct numerator!
        t_nTagLLP1_r_presel_q_q2 = nTagLLP1_r_q_q2_presel[nTagLLP1_r_q_q2_presel>0].astype(bool).sum()
        t_nTagLLP1_r_presel_q_q2_unc = 100./np.sqrt(t_nTagLLP1_r_presel_q_q2) if t_nTagLLP1_r_presel_q_q2>0 else 0
        t_diff_nTagLLP1_r_presel_q_q2 = 100*( (nTagLLP1_pred_r_q2_presel-t_nTagLLP1_r_presel_q_q2)/ ( t_nTagLLP1_r_presel_q_q2 ) )

        t_nTagLLP1_r_presel_q_tot = t_nTagLLP1_r_presel_q_q1 + t_nTagLLP1_r_presel_q_q2
        t_nTagLLP1_r_presel_q_tot_unc = 100./np.sqrt(t_nTagLLP1_r_presel_q_tot) if t_nTagLLP1_r_presel_q_tot>0 else 0
        t_diff_nTagLLP1_r_presel_q_tot = 100*( ( nTagLLP1_pred_r_q1_presel + nTagLLP1_pred_r_q2_presel  -t_nTagLLP1_r_presel_q_tot)/t_nTagLLP1_r_presel_q_tot  )


        row = [
            b,
            str(t_nTagLLP1_r_presel_q_q1) + " +- " + str( round( t_nTagLLP1_r_presel_q_q1_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q1_presel,2) ),
            str( round(t_diff_nTagLLP1_r_presel_q_q1, 2) ) + "%",

            str(t_nTagLLP1_r_presel_q_q2) + " +- " + str( round( t_nTagLLP1_r_presel_q_q2_unc,2) ) + "%",
            str(round(nTagLLP1_pred_r_q2_presel,2) ),
            str( round(t_diff_nTagLLP1_r_presel_q_q2, 2) ) + "%",

        ]

        table_r_q_LLP1_presel.add_row(row)

        t_nTagLLP1_r_presel_qq_q1 = nTagLLP1_r_qq_q1_presel[nTagLLP1_r_qq_q1_presel>0].astype(bool).sum()
        t_nTagLLP1_r_presel_qq_q1_unc = 100./np.sqrt(t_nTagLLP1_r_presel_qq_q1) if t_nTagLLP1_r_presel_qq_q1>0 else 0
        t_diff_nTagLLP1_r_presel_qq_q1 = 100*( (nTagLLP1_pred_r_qq_q1_presel - t_nTagLLP1_r_presel_qq_q1)/t_nTagLLP1_r_presel_qq_q1 )

        t_nTagLLP1_r_presel_qq_q2 = nTagLLP1_r_qq_q2_presel[nTagLLP1_r_qq_q2_presel>0].astype(bool).sum()
        t_nTagLLP1_r_presel_qq_q2_unc = 100./np.sqrt(t_nTagLLP1_r_presel_qq_q2) if t_nTagLLP1_r_presel_qq_q2>0 else 0
        t_diff_nTagLLP1_r_presel_qq_q2 = 100*( (nTagLLP1_pred_r_qq_q2_presel - t_nTagLLP1_r_presel_qq_q2)/t_nTagLLP1_r_presel_qq_q2 )

        t_nTagLLP1_r_presel_qq_tot = t_nTagLLP1_r_presel_qq_q1 + t_nTagLLP1_r_presel_qq_q2
        t_nTagLLP1_r_presel_qq_tot_unc = 100./np.sqrt(t_nTagLLP1_r_presel_qq_tot) if t_nTagLLP1_r_presel_qq_tot>0 else 0
        t_diff_nTagLLP1_r_presel_qq_tot = 100*( (-t_nTagLLP1_r_presel_qq_tot + nTagLLP1_pred_r_qq_q1_presel + nTagLLP1_pred_r_qq_q2_presel)/t_nTagLLP1_r_presel_qq_tot   )

        row = [
            b,
            #'true q1 only LLP1',
            str(t_nTagLLP1_r_presel_qq_q1) + " +- " + str( round( t_nTagLLP1_r_presel_qq_q1_unc, 2) ) + "%",
            #'pred q1 only LLP1',
            str( round(nTagLLP1_pred_r_qq_q1_presel,2) ),
            #diff
            str(round(t_diff_nTagLLP1_r_presel_qq_q1,2)) + "%",
            #'true q2 only LLP1',
            str(t_nTagLLP1_r_presel_qq_q2) + " +- " + str( round( t_nTagLLP1_r_presel_qq_q2_unc, 2) ) + "%",
            #'pred q2 only LLP1',
            str( round(nTagLLP1_pred_r_qq_q2_presel,2) ),
            #diff
            str(round(t_diff_nTagLLP1_r_presel_qq_q2,2)) + "%",
            #true sum
            str(t_nTagLLP1_r_presel_qq_tot) + " +- " + str( round( t_nTagLLP1_r_presel_qq_tot_unc,2) ) + "%",
            #'pred sum LLP1',
            str( round(nTagLLP1_pred_r_qq_q1_presel + nTagLLP1_pred_r_qq_q2_presel ,2) ),
            #diff
            str( round(t_diff_nTagLLP1_r_presel_qq_tot,2)) + "%",
        ]

        table_r_qq_LLP1_presel.add_row(row)


        pred_map = {}
        pred_map['m'] = samples[b]['mass']

        #Merged
        pred_map['true_m_LLP0']     = t_nTagLLP0_m
        pred_map['true_unc_m_LLP0'] = t_nTagLLP0_m_unc
        pred_map['pred_m_LLP0']     = nTagLLP0_pred_m
        pred_map['diff_m_LLP0']     = t_diff_nTagLLP0_m

        pred_map['true_m_presel_LLP0'] = t_nTagLLP0_m_presel
        pred_map['true_unc_m_presel_LLP0'] = t_nTagLLP0_m_presel_unc
        pred_map['pred_m_presel_LLP0'] = nTagLLP0_pred_m_presel
        pred_map['diff_m_presel_LLP0'] = t_diff_nTagLLP0_m_presel

        pred_map['true_m_LLP1']     = t_nTagLLP1_m
        pred_map['true_unc_m_LLP1'] = t_nTagLLP1_m_unc
        pred_map['pred_m_LLP1']     = nTagLLP1_pred_m
        pred_map['diff_m_LLP1']     = t_diff_nTagLLP1_m

        pred_map['true_m_presel_LLP1'] = t_nTagLLP1_m_presel
        pred_map['true_unc_m_presel_LLP1'] = t_nTagLLP1_m_presel_unc
        pred_map['pred_m_presel_LLP1'] = nTagLLP1_pred_m_presel
        pred_map['diff_m_presel_LLP1'] = t_diff_nTagLLP1_m_presel

        #Resolved
        pred_map['true_r_q_LLP0']     = t_nTagLLP0_r_q_q1 + t_nTagLLP0_r_q_q2
        pred_map['true_unc_r_q_LLP0'] = 100./np.sqrt(t_nTagLLP0_r_q_q1+ t_nTagLLP0_r_q_q2) if (t_nTagLLP0_r_q_q1+ t_nTagLLP0_r_q_q2)>0 else 0
        pred_map['pred_r_q_LLP0']     = nTagLLP0_pred_r_q1 + nTagLLP0_pred_r_q2
        pred_map['diff_r_q_LLP0']     = t_diff_nTagLLP0_r_q_tot

        pred_map['true_r_q_presel_LLP0']     = t_nTagLLP0_r_presel_q_q1 + t_nTagLLP0_r_presel_q_q2
        pred_map['true_unc_r_q_presel_LLP0'] = 100./np.sqrt(t_nTagLLP0_r_presel_q_q1+ t_nTagLLP0_r_presel_q_q2) if (t_nTagLLP0_r_presel_q_q1+ t_nTagLLP0_r_presel_q_q2)>0 else 0
        pred_map['pred_r_q_presel_LLP0']     = nTagLLP0_pred_r_q1_presel + nTagLLP0_pred_r_q2_presel
        pred_map['diff_r_q_presel_LLP0']     = t_diff_nTagLLP0_r_presel_q_tot

        pred_map['true_r_qq_LLP0']     = t_nTagLLP0_r_qq_tot
        pred_map['true_unc_r_qq_LLP0'] = t_nTagLLP0_r_qq_tot_unc
        pred_map['pred_r_qq_LLP0']     = nTagLLP0_pred_r_qq_q1 + nTagLLP0_pred_r_qq_q2
        pred_map['diff_r_qq_LLP0']     = t_diff_nTagLLP0_r_qq_tot

        pred_map['true_r_qq_presel_LLP0']     = t_nTagLLP0_r_presel_qq_tot
        pred_map['true_unc_r_qq_presel_LLP0'] = t_nTagLLP0_r_presel_qq_tot_unc
        pred_map['pred_r_qq_presel_LLP0']     = nTagLLP0_pred_r_qq_q1_presel + nTagLLP0_pred_r_qq_q2_presel
        pred_map['diff_r_qq_presel_LLP0']     = t_diff_nTagLLP0_r_presel_qq_tot

        pred_map['true_r_q_LLP1']     = t_nTagLLP1_r_q_q1 + t_nTagLLP1_r_q_q2
        pred_map['true_unc_r_q_LLP1'] = 100./np.sqrt(t_nTagLLP1_r_q_q1+ t_nTagLLP1_r_q_q2) if (t_nTagLLP1_r_q_q1+ t_nTagLLP1_r_q_q2)>0 else 0
        pred_map['pred_r_q_LLP1']     = nTagLLP1_pred_r_q1 + nTagLLP1_pred_r_q2
        pred_map['diff_r_q_LLP1']     = t_diff_nTagLLP1_r_q_tot

        pred_map['true_r_q_presel_LLP1']     = t_nTagLLP1_r_presel_q_q1 + t_nTagLLP1_r_presel_q_q2
        pred_map['true_unc_r_q_presel_LLP1'] = 100./np.sqrt(t_nTagLLP1_r_presel_q_q1+ t_nTagLLP1_r_presel_q_q2) if (t_nTagLLP1_r_presel_q_q1+ t_nTagLLP1_r_presel_q_q2)>0 else 0
        pred_map['pred_r_q_presel_LLP1']     = nTagLLP1_pred_r_q1_presel + nTagLLP1_pred_r_q2_presel
        pred_map['diff_r_q_presel_LLP1']     = t_diff_nTagLLP1_r_presel_q_tot

        pred_map['true_r_qq_LLP1']     = t_nTagLLP1_r_qq_tot
        pred_map['true_unc_r_qq_LLP1'] = t_nTagLLP1_r_qq_tot_unc
        pred_map['pred_r_qq_LLP1']     = nTagLLP1_pred_r_qq_q1 + nTagLLP1_pred_r_qq_q2
        pred_map['diff_r_qq_LLP1']     = t_diff_nTagLLP1_r_qq_tot

        pred_map['true_r_qq_presel_LLP1']     = t_nTagLLP1_r_presel_qq_tot
        pred_map['true_unc_r_qq_presel_LLP1'] = t_nTagLLP1_r_presel_qq_tot_unc
        pred_map['pred_r_qq_presel_LLP1']     = nTagLLP1_pred_r_qq_q1_presel + nTagLLP1_pred_r_qq_q2_presel
        pred_map['diff_r_qq_presel_LLP1']     = t_diff_nTagLLP1_r_presel_qq_tot



        with open(out_fold+"Tag_pred_"+b+"_"+str(R).replace(".","p")+".yaml","w") as f:
            yaml.dump(pred_map, f)
            f.close()
            print("Info: dictionary written in file "+out_fold+"Tag_pred_"+b+"_"+str(R).replace(".","p")+".yaml")



    #HERE
    print("\n")
    print("   Merged   ")
    print("   using LLP0+LLP1 efficiency map")
    print(table_m)
    print("   with preselection ")
    print(table_m_presel)

    print("\n")
    print("   Resolved   ")
    print("   using LLP0+LLP1 efficiency map")
    print("   1 acceptance ")
    print(table_r_q_LLP0)
    print("   2 acceptance ")
    print(table_r_qq_LLP0)
    print("   1 acceptance ")
    print(table_r_q_LLP1)
    print("   2 acceptance ")
    print(table_r_qq_LLP1)
    #print "   double check with 1 acceptance maps"
    #print(table_r_qq_LLP0_cross)
    print("   ")
    print("   Preselections: ")
    print("   1 acceptance ")
    print(table_r_q_LLP0_presel)
    print("   2 acceptance ")
    print(table_r_qq_LLP0_presel)
    print("   1 acceptance ")
    print(table_r_q_LLP1_presel)
    print("   2 acceptance ")
    print(table_r_qq_LLP1_presel)



def plot_maps(out_fold, sign, R=0.8,add_unc=False,unc_fold="",start_figure=0):

    tmp_bins_r_resolved=np.copy(pl_bins_r_resolved)
    tmp_bins_z_resolved=np.copy(pl_bins_z_resolved)
    tmp_bins_pt_resolved=np.copy(pl_bins_pt_resolved)

    tmp_bins_pt_merged=np.copy(pl_bins_pt_merged)
    tmp_bins_r_merged=np.copy(pl_bins_r_merged)
    tmp_bins_z_merged=np.copy(pl_bins_z_merged)

    tmp_bins_r_resolved[0] = 0.01
    tmp_bins_r_resolved[-1] = 20.
    tmp_bins_z_resolved[0] = 0.01
    tmp_bins_z_resolved[-1] = 20.
    tmp_bins_r_merged[0] = 0.01
    tmp_bins_r_merged[-1] = 20.
    tmp_bins_z_merged[0] = 0.01
    tmp_bins_z_merged[-1] = 20.



    unc_m = {}
    unc_r_q = {}
    unc_r_qq = {}
    unc_m_presel = {}
    unc_r_q_presel = {}
    unc_r_qq_presel = {}

    max_unc_m = 0
    max_unc_r_q = 0
    max_unc_r_qq = 0

    max_unc_m_presel = 0
    max_unc_r_q_presel = 0
    max_unc_r_qq_presel = 0

    if add_unc and unc_fold!="":
        print("Adding in quadrature nonclosure uncertainty")
        masses =  np.load(unc_fold+"Prediction_nonclosure_unc_masses.npy")
        diff_m = {}
        diff_r_q = {}
        diff_r_qq = {}
        diff_m_presel = {}
        diff_r_q_presel = {}
        diff_r_qq_presel = {}

        
        for lf in ["_ctau500","_ctau3000"]:
            diff_m[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_m_"+str(R).replace(".","p")+lf+".npy")/100.)
            diff_r_q[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_r_q_"+str(R).replace(".","p")+lf+".npy")/100.)
            diff_r_qq[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_r_qq_"+str(R).replace(".","p")+lf+".npy")/100.)
            diff_m_presel[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_presel_m_"+str(R).replace(".","p")+lf+".npy")/100.)
            diff_r_q_presel[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_presel_r_q_"+str(R).replace(".","p")+lf+".npy")/100.)
            diff_r_qq_presel[lf] = np.abs(np.load(unc_fold+"Prediction_nonclosure_unc_presel_r_qq_"+str(R).replace(".","p")+lf+".npy")/100.)

        #print masses
        #print diff_m["_ctau500"]
        #print diff_m["_ctau3000"]
        

        for i,m in enumerate(masses):
            unc_m[m]    = np.maximum(diff_m["_ctau500"][i],diff_m["_ctau3000"][i])
            unc_r_q[m]  = np.maximum(diff_r_q["_ctau500"][i],diff_r_q["_ctau3000"][i])
            unc_r_qq[m] = np.maximum(diff_r_qq["_ctau500"][i],diff_r_qq["_ctau3000"][i])

            max_unc_m = max(max_unc_m,abs(unc_m[m]))
            max_unc_r_q = max(max_unc_r_q,abs(unc_r_q[m]))
            max_unc_r_qq = max(max_unc_r_qq,abs(unc_r_qq[m]))

            unc_m_presel[m]    = np.maximum(diff_m_presel["_ctau500"][i],diff_m_presel["_ctau3000"][i])
            unc_r_q_presel[m]  = np.maximum(diff_r_q_presel["_ctau500"][i],diff_r_q_presel["_ctau3000"][i])
            unc_r_qq_presel[m] = np.maximum(diff_r_qq_presel["_ctau500"][i],diff_r_qq_presel["_ctau3000"][i])

            max_unc_m_presel = max(max_unc_m_presel,abs(unc_m_presel[m]))
            max_unc_r_q_presel = max(max_unc_r_q_presel,abs(unc_r_q_presel[m]))
            max_unc_r_qq_presel = max(max_unc_r_qq_presel,abs(unc_r_qq_presel[m]))

    print("Unc m:")
    print(unc_m)
    print(max_unc_m)

    print("Unc r_q:")
    print(unc_r_q)
    print(max_unc_r_q)

    print("Unc r_qq:")
    print(unc_r_qq)
    print(max_unc_r_qq)

    print("Opening ",out_fold,"...")

    #index for hepdata figures: increase by one unit at every mass and every topology
    i_hep = start_figure

    for b in sign:

        m_den = np.load(out_fold+"m_den_"+b+"_"+str(R).replace(".","p")+".npy")
        m_num_T = np.load(out_fold+"m_num_T_"+b+"_"+str(R).replace(".","p")+".npy")
        m_num_T_presel = np.load(out_fold+"m_num_T_presel_"+b+"_"+str(R).replace(".","p")+".npy")

        r_den_q = np.load(out_fold+"r_den_q_"+b+"_"+str(R).replace(".","p")+".npy")
        r_num_q_T = np.load(out_fold+"r_num_q_T_"+b+"_"+str(R).replace(".","p")+".npy")
        r_num_q_T_presel = np.load(out_fold+"r_num_q_T_presel_"+b+"_"+str(R).replace(".","p")+".npy")

        r_den_qq = np.load(out_fold+"r_den_qq_"+b+"_"+str(R).replace(".","p")+".npy")
        r_num_qq_T = np.load(out_fold+"r_num_qq_T_"+b+"_"+str(R).replace(".","p")+".npy")
        r_num_qq_T_presel = np.load(out_fold+"r_num_qq_T_presel_"+b+"_"+str(R).replace(".","p")+".npy")

        m_eff_T = np.divide(m_num_T,m_den,out=np.zeros_like(m_den),where=(m_den != 0))
        m_eff_T_presel = np.divide(m_num_T_presel,m_den,out=np.zeros_like(m_den),where=(m_den != 0))




        '''

        print("Need to transform this in a 2D map")
        print("m_den shape: ", m_den.shape)
        #shape is (8,6,9)

        h2_den_m_pt_b_8 = TH2D(b+"_den_m_pt_b_8",b+"_den_m_pt_b_8", len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
        h2_den_m_pt_b_8.Sumw2()

        h2_num_m_pt_b_8 = TH2D(b+"_num_m_pt_b_8",b+"_num_m_pt_b_8", len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
        h2_num_m_pt_b_8.Sumw2()


        pt_b = 8
        for i_z in range(m_den.shape[1]):
            #Looping over z direction
            print("z bin: ", i_z, tmp_bins_z_merged[i_z])

            #den
            #h_m_den = TH1D(b+"_m_den_"+str(i), b+"_m_den_"+str(i), len(tmp_bins_r_merged)-1, tmp_bins_r_merged)
            #h_m_den.Sumw2()
            to_put_m_den = m_den[:,i_z,pt_b] 
            for p in range(to_put_m_den.shape[0]):
                print("r bin: ", p, tmp_bins_r_merged[p])
                print("counts: ", to_put_m_den[p])
                for c in range(int(to_put_m_den[p])):
                    #h_m_den.Fill(tmp_bins_r_merged[p])
                    h2_den_m_pt_b_8.Fill(tmp_bins_r_merged[p],tmp_bins_z_merged[i_z])
                    c+=1

            #num
            #h_m_num_T = TH1D(b+"_m_num_T_"+str(i), b+"_m_num_T_"+str(i), len(tmp_bins_r_merged)-1, tmp_bins_r_merged)
            #h_m_num_T.Sumw2()
            to_put_m_num_T = m_num_T[:,i_z,pt_b] 
            for p in range(to_put_m_num_T.shape[0]):
                for c in range(int(to_put_m_num_T[p])):
                    #h_m_num_T.Fill(tmp_bins_r_merged[p])
                    h2_num_m_pt_b_8.Fill(tmp_bins_r_merged[p],tmp_bins_z_merged[i_z])
                    c+=1

            #eff_m_T[i] = TGraphAsymmErrors()
            #eff_m_T[i].BayesDivide(h_m_num_T,h_m_den)   

            #print(eff_m_T[i].Print())


            #num presel
            i_z+=1

            #h_m_den.Delete()
            #h_m_num_T.Delete()

            
        #Try a canvas
        print(h2_den_m_pt_b_8.Print())
        print(h2_num_m_pt_b_8.Print())
            
        h2_num_m_pt_b_8.Divide(h2_num_m_pt_b_8,h2_den_m_pt_b_8,1,1,"cl=0.683 b(1,1) mode")

        print("Post divide")
        print(h2_num_m_pt_b_8.Print())

        list_palette = [100,55,104,57]
        gStyle.SetPalette(list_palette[0])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can = TCanvas("can","can",1400,800)
        can.cd()
        #can.DrawFrame(0.01,0.01,20,20)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)

        h2_num_m_pt_b_8.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");

        can.SetLogx()
        can.SetLogy()
        can.SetLogz()
        h2_num_m_pt_b_8.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_m_pt_b_8.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_m_pt_b_8.GetXaxis().SetTitleSize(0.04)
        h2_num_m_pt_b_8.GetXaxis().SetTitleOffset(1.3)
        h2_num_m_pt_b_8.GetYaxis().SetTitleSize(0.04)
        h2_num_m_pt_b_8.GetYaxis().SetTitleOffset(1.2)
        h2_num_m_pt_b_8.GetZaxis().SetTitle("Efficiency")
        h2_num_m_pt_b_8.GetZaxis().SetTitleSize(0.04)
        h2_num_m_pt_b_8.GetZaxis().SetTitleOffset(1.2)

        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can.Update()

        can.Print(out_fold+'test2_pt_b8.png')
        can.Print(out_fold+'test2_pt_b8.pdf')
        can.Close()

        '''



        #Do it for each pt bin and do the inclusive one


        #Inclusive
        #m
        h2_den_m = TH2D(b+"_den_m",b+"_den_m", len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
        h2_den_m.Sumw2()
        h2_num_m = TH2D(b+"_num_m",b+"_num_m", len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
        h2_num_m.Sumw2()
        h2_num_m_presel = TH2D(b+"_num_m_presel",b+"_num_m_presel", len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
        h2_num_m_presel.Sumw2()
        
        h2_den_m_pt = {}
        h2_num_m_pt = {}
        h2_num_m_presel_pt = {}

        for i_pt in range(m_den.shape[2]):
            h2_den_m_pt[i_pt] = TH2D(b+"_den_m_pt"+str(i_pt),b+"_den_m_pt"+str(i_pt), len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
            h2_den_m_pt[i_pt].Sumw2()
            h2_num_m_pt[i_pt] = TH2D(b+"_num_m_pt"+str(i_pt),b+"_num_m_pt"+str(i_pt), len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
            h2_num_m_pt[i_pt].Sumw2()
            h2_num_m_presel_pt[i_pt] = TH2D(b+"_num_m_presel_pt"+str(i_pt),b+"_num_m_presel_pt"+str(i_pt), len(tmp_bins_r_merged)-1, tmp_bins_r_merged,len(tmp_bins_z_merged)-1, tmp_bins_z_merged)
            h2_num_m_presel_pt[i_pt].Sumw2()

        #q
        h2_den_r_q = TH2D(b+"_den_r_q",b+"_den_r_q", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_den_r_q.Sumw2()
        h2_num_r_q = TH2D(b+"_num_r_q",b+"_num_r_q", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_num_r_q.Sumw2()
        h2_num_r_q_presel = TH2D(b+"_num_r_q_presel",b+"_num_r_q_presel", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_num_r_q_presel.Sumw2()

        h2_den_r_q_pt = {}
        h2_num_r_q_pt = {}
        h2_num_r_q_presel_pt = {}

        for i_pt in range(r_den_q.shape[2]):
            h2_den_r_q_pt[i_pt] = TH2D(b+"_den_r_q_pt"+str(i_pt),b+"_den_r_q_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_den_r_q_pt[i_pt].Sumw2()
            h2_num_r_q_pt[i_pt] = TH2D(b+"_num_r_q_pt"+str(i_pt),b+"_num_r_q_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_num_r_q_pt[i_pt].Sumw2()
            h2_num_r_q_presel_pt[i_pt] = TH2D(b+"_num_r_q_presel_pt"+str(i_pt),b+"_num_r_q_presel_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_num_r_q_presel_pt[i_pt].Sumw2()


        #qq
        h2_den_r_qq = TH2D(b+"_den_r_qq",b+"_den_r_qq", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_den_r_qq.Sumw2()
        h2_num_r_qq = TH2D(b+"_num_r_qq",b+"_num_r_qq", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_num_r_qq.Sumw2()
        h2_num_r_qq_presel = TH2D(b+"_num_r_qq_presel",b+"_num_r_qq_presel", len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
        h2_num_r_qq_presel.Sumw2()

        h2_den_r_qq_pt = {}
        h2_num_r_qq_pt = {}
        h2_num_r_qq_presel_pt = {}

        for i_pt in range(r_den_qq.shape[2]):
            h2_den_r_qq_pt[i_pt] = TH2D(b+"_den_r_qq_pt"+str(i_pt),b+"_den_r_qq_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_den_r_qq_pt[i_pt].Sumw2()
            h2_num_r_qq_pt[i_pt] = TH2D(b+"_num_r_qq_pt"+str(i_pt),b+"_num_r_qq_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_num_r_qq_pt[i_pt].Sumw2()
            h2_num_r_qq_presel_pt[i_pt] = TH2D(b+"_num_r_qq_presel_pt"+str(i_pt),b+"_num_r_qq_presel_pt"+str(i_pt), len(tmp_bins_r_resolved)-1, tmp_bins_r_resolved,len(tmp_bins_z_resolved)-1, tmp_bins_z_resolved)
            h2_num_r_q_presel_pt[i_pt].Sumw2()

        #m
        for i_r in range(m_den.shape[0]):
            for i_z in range(m_den.shape[1]):
                #print("---")
                for i_pt in range(m_den.shape[2]):
                    #print("r bin: ", i_r, tmp_bins_r_merged[i_r])
                    #print("z bin: ", i_z, tmp_bins_z_merged[i_z])
                    #print("pt bin: ", i_pt, tmp_bins_pt_merged[i_pt])
                    #print("content: ", m_den[i_r][i_z][i_pt])

                    to_put_m_den = m_den[i_r][i_z][i_pt]
                    for c in range(int(to_put_m_den)):
                        h2_den_m.Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])
                        h2_den_m_pt[i_pt].Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])

                    to_put_m_num_T = m_num_T[i_r][i_z][i_pt]
                    for c in range(int(to_put_m_num_T)):
                        h2_num_m.Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])
                        h2_num_m_pt[i_pt].Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])
                        
                    to_put_m_num_T_presel = m_num_T_presel[i_r][i_z][i_pt]
                    for c in range(int(to_put_m_num_T_presel)):
                        h2_num_m_presel.Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])
                        h2_num_m_presel_pt[i_pt].Fill(tmp_bins_r_merged[i_r],tmp_bins_z_merged[i_z])



        for i_pt in range(m_den.shape[2]):
            h2_num_m_pt[i_pt].Divide(h2_num_m_pt[i_pt],h2_den_m_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")
            h2_num_m_presel_pt[i_pt].Divide(h2_num_m_presel_pt[i_pt],h2_den_m_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")

        h2_num_m.Divide(h2_num_m,h2_den_m,1,1,"cl=0.683 b(1,1) mode")
        h2_num_m_presel.Divide(h2_num_m_presel,h2_den_m,1,1,"cl=0.683 b(1,1) mode")


        #Nonclosure uncertainty 
        print(b)
        if add_unc and unc_fold!="":
            for i_r in range(m_den.shape[0]+1):
                for i_z in range(m_den.shape[1]+1):
                    print("m bin: ", i_r, i_z)
                    print(h2_num_m.GetBinContent(i_r,i_z)," +- ", h2_num_m.GetBinError(i_r,i_z))
                    print(h2_num_m.GetBinContent(i_r,i_z)," +- ", math.sqrt(h2_num_m.GetBinError(i_r,i_z)**2 + (max_unc_m*h2_num_m.GetBinContent(i_r,i_z) )**2))
                    ##mass-dependent uncertainty:
                    #h2_num_m.SetBinError(i_r, i_z,math.sqrt(h2_num_m.GetBinError(i_r,i_z)**2 + (unc_m[ samples[b]['mass'] ]*h2_num_m.GetBinContent(i_r,i_z) )**2))
                    #mass-independent uncertainty:
                    h2_num_m.SetBinError(i_r, i_z,math.sqrt(h2_num_m.GetBinError(i_r,i_z)**2 + (max_unc_m*h2_num_m.GetBinContent(i_r,i_z) )**2))


                    ##print(h2_num_m_presel.GetBinContent(i_r,i_z)," +- ", h2_num_m_presel.GetBinError(i_r,i_z))
                    ##print(h2_num_m_presel.GetBinContent(i_r,i_z)," +- ", math.sqrt(h2_num_m_presel.GetBinError(i_r,i_z)**2 + (unc_m_presel[ samples[b]['mass'] ]*h2_num_m_presel.GetBinContent(i_r,i_z) )**2))
                    ##mass-dependent uncertainty:
                    #h2_num_m_presel.SetBinError(i_r, i_z,math.sqrt(h2_num_m_presel.GetBinError(i_r,i_z)**2 + (unc_m_presel[ samples[b]['mass'] ]*h2_num_m_presel.GetBinContent(i_r,i_z) )**2))
                    #mass-independent uncertainty:
                    h2_num_m_presel.SetBinError(i_r, i_z,math.sqrt(h2_num_m_presel.GetBinError(i_r,i_z)**2 + (max_unc_m_presel*h2_num_m_presel.GetBinContent(i_r,i_z) )**2))


        #Plot
        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,55,104,57]
        gStyle.SetPalette(list_palette[0])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_m.Draw("COLZTEXTE sames")
        #h2_den_m.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_m.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_m.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_m.GetXaxis().SetTitleSize(0.04)
        h2_num_m.GetXaxis().SetTitleOffset(1.3)
        h2_num_m.GetYaxis().SetTitleSize(0.04)
        h2_num_m.GetYaxis().SetTitleOffset(1.2)
        h2_num_m.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_m.GetZaxis().SetTitleSize(0.04)
        h2_num_m.GetZaxis().SetTitleOffset(1.2)
        h2_num_m.SetMaximum(1.)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        '''
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(13)
        latex.SetTextFont(62)#42, 52, 62, 72
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.15, 0.8, "Merged")
        latex.DrawLatex(0.15, 0.75, "m_{#chi} = "+str(samples[b]['mass'])+" GeV")
        '''

        pavetext = TPaveText(-1.85,2.5,-1.05,7.)
        #pavetext.SetNDC()
        pavetext.SetTextAlign(12)
        pavetext.SetTextFont(62)#42, 52, 62, 72
        pavetext.SetTextSize(0.04)
        pavetext.AddText("Merged")
        pavetext.AddText("m_{LLP} = "+str(samples[b]['mass'])+" GeV")
        pavetext.Draw()

        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        can.Print(out_fold+'Efficiency_map_m_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        can.Print(out_fold+'Efficiency_map_m_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')

        if start_figure>0:
            num_str = "00"+str(i_hep) if i_hep<10 else"0"+str(i_hep)
            can.Print(HEPDATA_OUT+"Figure-aux_"+num_str+".pdf")

        can.Close()

        out_file = TFile(out_fold+'Efficiency_map_m_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        out_file.cd()
        h2_num_m.Write("h2")
        print "Writing "+out_fold+'Efficiency_map_m_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        out_file.Write()
        out_file.Close()

        if start_figure>0:
            out_file = TFile(HEPDATA_OUT+"Figure-aux_"+num_str+".root","RECREATE")
            out_file.cd()
            h2_num_m.Write("h2")
            print "Writing "+HEPDATA_OUT+"Figure-aux_"+num_str+".root"
            out_file.Write()
            out_file.Close()
            i_hep+=1


        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,57,109]
        gStyle.SetPalette(list_palette[1])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_m_presel.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_m_presel.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_m_presel.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_m_presel.GetXaxis().SetTitleSize(0.04)
        h2_num_m_presel.GetXaxis().SetTitleOffset(1.3)
        h2_num_m_presel.GetYaxis().SetTitleSize(0.04)
        h2_num_m_presel.GetYaxis().SetTitleOffset(1.2)
        h2_num_m_presel.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_m_presel.GetZaxis().SetTitleSize(0.04)
        h2_num_m_presel.GetZaxis().SetTitleOffset(1.2)
        h2_num_m_presel.SetMaximum(.5)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)


        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        #can.Print(out_fold+'Efficiency_map_m_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        #can.Print(out_fold+'Efficiency_map_m_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')
        can.Close()

        #out_file = TFile(out_fold+'Efficiency_map_m_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        #out_file.cd()
        #h2_num_m_presel.Write("h2")
        #print "Writing "+out_fold+'Efficiency_map_m_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        #out_file.Write()
        #out_file.Close()



        #Plot each pt bin!
        for i_pt in range(m_den.shape[2]):
            #Plot
            can = TCanvas("can","can",1400,800)
            can.cd()

            list_palette = [100,55,104,57]
            gStyle.SetPalette(list_palette[0])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_m_pt[i_pt].Draw("COLZTEXTE sames")
            #h2_den_m_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_m_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_m_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_m_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_m_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_m_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_m_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_m_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_m_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_m_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_m_pt[i_pt].SetMaximum(1.)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_m_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_m_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()

            can = TCanvas("can","can",1400,800)
            can.cd()
            
            list_palette = [100,57,109]
            gStyle.SetPalette(list_palette[1])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_m_presel_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_m_presel_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_m_presel_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_m_presel_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_m_presel_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_m_presel_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_m_presel_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_m_presel_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_m_presel_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_m_presel_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_m_presel_pt[i_pt].SetMaximum(.5)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_m_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_m_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()



        #q
        for i_r in range(r_den_q.shape[0]):
            for i_z in range(r_den_q.shape[1]):
                #print("---")
                for i_pt in range(r_den_q.shape[2]):
                    #print("r bin: ", i_r, tmp_bins_r_resolved[i_r])
                    #print("z bin: ", i_z, tmp_bins_z_resolved[i_z])
                    #print("pt bin: ", i_pt, tmp_bins_pt_resolved[i_pt])
                    #print("content: ", r_den_q[i_r][i_z][i_pt])

                    to_put_r_den_q = r_den_q[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_den_q)):
                        h2_den_r_q.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_den_r_q_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        

                    to_put_r_num_q_T = r_num_q_T[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_num_q_T)):
                        h2_num_r_q.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_num_r_q_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        

                    to_put_r_num_q_T_presel = r_num_q_T_presel[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_num_q_T_presel)):
                        h2_num_r_q_presel.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_num_r_q_presel_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        

        for i_pt in range(r_den_q.shape[2]):
            h2_num_r_q_pt[i_pt].Divide(h2_num_r_q_pt[i_pt],h2_den_r_q_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")
            h2_num_r_q_presel_pt[i_pt].Divide(h2_num_r_q_presel_pt[i_pt],h2_den_r_q_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")

        h2_num_r_q.Divide(h2_num_r_q,h2_den_r_q,1,1,"cl=0.683 b(1,1) mode")
        h2_num_r_q_presel.Divide(h2_num_r_q_presel,h2_den_r_q,1,1,"cl=0.683 b(1,1) mode")



        #Nonclosure uncertainty 
        if add_unc and unc_fold!="":
            for i_r in range(r_den_q.shape[0]+1):
                for i_z in range(r_den_q.shape[1]+1):
                    h2_num_r_q.SetBinError(i_r, i_z,math.sqrt(h2_num_r_q.GetBinError(i_r,i_z)**2 + (max_unc_r_q*h2_num_r_q.GetBinContent(i_r,i_z) )**2))
                    h2_num_r_q_presel.SetBinError(i_r, i_z,math.sqrt(h2_num_r_q_presel.GetBinError(i_r,i_z)**2 + (max_unc_r_q_presel*h2_num_r_q_presel.GetBinContent(i_r,i_z) )**2))



        #Plot
        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,55,104,57]
        gStyle.SetPalette(list_palette[0])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_r_q.Draw("COLZTEXTE sames")
        #h2_den_r_q.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_r_q.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_r_q.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_r_q.GetXaxis().SetTitleSize(0.04)
        h2_num_r_q.GetXaxis().SetTitleOffset(1.3)
        h2_num_r_q.GetYaxis().SetTitleSize(0.04)
        h2_num_r_q.GetYaxis().SetTitleOffset(1.2)
        h2_num_r_q.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_r_q.GetZaxis().SetTitleSize(0.04)
        h2_num_r_q.GetZaxis().SetTitleOffset(1.2)
        h2_num_r_q.SetMaximum(1.)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)


        pavetext = TPaveText(-1.85,2.5,-1.05,7.)
        #pavetext.SetNDC()
        pavetext.SetTextAlign(12)
        pavetext.SetTextFont(62)#42, 52, 62, 72
        pavetext.SetTextSize(0.04)
        pavetext.AddText("Resolved 1q")
        pavetext.AddText("m_{LLP} = "+str(samples[b]['mass'])+" GeV")
        pavetext.Draw()

        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        can.Print(out_fold+'Efficiency_map_r_q_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        can.Print(out_fold+'Efficiency_map_r_q_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')

        if start_figure>0:
            num_str = "00"+str(i_hep) if i_hep<10 else"0"+str(i_hep)
            can.Print(HEPDATA_OUT+"Figure-aux_"+num_str+".pdf")

        can.Close()

        out_file = TFile(out_fold+'Efficiency_map_r_q_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        out_file.cd()
        h2_num_r_q.Write("h2")
        print "Writing "+out_fold+'Efficiency_map_r_q_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        out_file.Write()
        out_file.Close()

        if start_figure>0:
            out_file = TFile(HEPDATA_OUT+"Figure-aux_"+num_str+".root","RECREATE")
            out_file.cd()
            h2_num_m.Write("h2")
            print "Writing "+HEPDATA_OUT+"Figure-aux_"+num_str+".root"
            out_file.Write()
            out_file.Close()
            i_hep+=1

        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,57,109]
        gStyle.SetPalette(list_palette[1])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_r_q_presel.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_r_q_presel.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_r_q_presel.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_r_q_presel.GetXaxis().SetTitleSize(0.04)
        h2_num_r_q_presel.GetXaxis().SetTitleOffset(1.3)
        h2_num_r_q_presel.GetYaxis().SetTitleSize(0.04)
        h2_num_r_q_presel.GetYaxis().SetTitleOffset(1.2)
        h2_num_r_q_presel.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_r_q_presel.GetZaxis().SetTitleSize(0.04)
        h2_num_r_q_presel.GetZaxis().SetTitleOffset(1.2)
        h2_num_r_q_presel.SetMaximum(.5)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        #can.Print(out_fold+'Efficiency_map_r_q_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        #can.Print(out_fold+'Efficiency_map_r_q_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')
        can.Close()

        #out_file = TFile(out_fold+'Efficiency_map_r_q_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        #out_file.cd()
        #h2_num_r_q_presel.Write("h2")
        #print "Writing "+out_fold+'Efficiency_map_r_q_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        #out_file.Write()
        #out_file.Close()

        #Plot each pt bin!
        for i_pt in range(r_den_q.shape[2]):
            #Plot
            can = TCanvas("can","can",1400,800)
            can.cd()

            list_palette = [100,55,104,57]
            gStyle.SetPalette(list_palette[0])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_r_q_pt[i_pt].Draw("COLZTEXTE sames")
            #h2_den_r_q_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_r_q_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_r_q_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_r_q_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_r_q_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_r_q_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_r_q_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_r_q_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_r_q_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_r_q_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_r_q_pt[i_pt].SetMaximum(1.)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_r_q_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_r_q_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()

            can = TCanvas("can","can",1400,800)
            can.cd()
            
            list_palette = [100,57,109]
            gStyle.SetPalette(list_palette[1])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_r_q_presel_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_r_q_presel_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_r_q_presel_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_r_q_presel_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_r_q_presel_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_r_q_presel_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_r_q_presel_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_r_q_presel_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_r_q_presel_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_r_q_presel_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_r_q_presel_pt[i_pt].SetMaximum(.5)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_r_q_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_r_q_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()




        #qq
        for i_r in range(r_den_qq.shape[0]):
            for i_z in range(r_den_qq.shape[1]):
                #print("---")
                for i_pt in range(r_den_qq.shape[2]):
                    #print("r bin: ", i_r, tmp_bins_r_resolved[i_r])
                    #print("z bin: ", i_z, tmp_bins_z_resolved[i_z])
                    #print("pt bin: ", i_pt, tmp_bins_pt_resolved[i_pt])
                    #print("content: ", r_den_qq[i_r][i_z][i_pt])

                    to_put_r_den_qq = r_den_qq[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_den_qq)):
                        h2_den_r_qq.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_den_r_qq_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])

                    to_put_r_num_qq_T = r_num_qq_T[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_num_qq_T)):
                        h2_num_r_qq.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_num_r_qq_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])

                    to_put_r_num_qq_T_presel = r_num_qq_T_presel[i_r][i_z][i_pt]
                    for c in range(int(to_put_r_num_qq_T_presel)):
                        h2_num_r_qq_presel.Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        h2_num_r_qq_presel_pt[i_pt].Fill(tmp_bins_r_resolved[i_r],tmp_bins_z_resolved[i_z])
                        

        for i_pt in range(r_den_qq.shape[2]):
            h2_num_r_qq_pt[i_pt].Divide(h2_num_r_qq_pt[i_pt],h2_den_r_qq_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")
            h2_num_r_qq_presel_pt[i_pt].Divide(h2_num_r_qq_presel_pt[i_pt],h2_den_r_qq_pt[i_pt],1,1,"cl=0.683 b(1,1) mode")
                    

        h2_num_r_qq.Divide(h2_num_r_qq,h2_den_r_qq,1,1,"cl=0.683 b(1,1) mode")
        h2_num_r_qq_presel.Divide(h2_num_r_qq_presel,h2_den_r_qq,1,1,"cl=0.683 b(1,1) mode")


        #Nonclosure uncertainty 
        if add_unc and unc_fold!="":
            for i_r in range(r_den_qq.shape[0]+1):
                for i_z in range(r_den_qq.shape[1]+1):
                    h2_num_r_qq.SetBinError(i_r, i_z,math.sqrt(h2_num_r_qq.GetBinError(i_r,i_z)**2 + (max_unc_r_qq*h2_num_r_qq.GetBinContent(i_r,i_z) )**2))
                    h2_num_r_qq_presel.SetBinError(i_r, i_z,math.sqrt(h2_num_r_qq_presel.GetBinError(i_r,i_z)**2 + (max_unc_r_qq_presel*h2_num_r_qq_presel.GetBinContent(i_r,i_z) )**2))

        #Plot
        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,55,104,57]
        gStyle.SetPalette(list_palette[0])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_r_qq.Draw("COLZTEXTE sames")
        #h2_den_r_qq.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_r_qq.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_r_qq.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_r_qq.GetXaxis().SetTitleSize(0.04)
        h2_num_r_qq.GetXaxis().SetTitleOffset(1.3)
        h2_num_r_qq.GetYaxis().SetTitleSize(0.04)
        h2_num_r_qq.GetYaxis().SetTitleOffset(1.2)
        h2_num_r_qq.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_r_qq.GetZaxis().SetTitleSize(0.04)
        h2_num_r_qq.GetZaxis().SetTitleOffset(1.2)
        h2_num_r_qq.SetMaximum(1.)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        pavetext = TPaveText(-1.85,2.5,-1.05,7.)
        #pavetext.SetNDC()
        pavetext.SetTextAlign(12)
        pavetext.SetTextFont(62)#42, 52, 62, 72
        pavetext.SetTextSize(0.04)
        pavetext.AddText("Resolved 2q")
        pavetext.AddText("m_{LLP} = "+str(samples[b]['mass'])+" GeV")
        pavetext.Draw()

        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        can.Print(out_fold+'Efficiency_map_r_qq_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        can.Print(out_fold+'Efficiency_map_r_qq_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')

        if start_figure>0:
            num_str = "00"+str(i_hep) if i_hep<10 else"0"+str(i_hep)
            can.Print(HEPDATA_OUT+"Figure-aux_"+num_str+".pdf")

        can.Close()

        out_file = TFile(out_fold+'Efficiency_map_r_qq_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        out_file.cd()
        h2_num_r_qq.Write("h2")
        print "Writing "+out_fold+'Efficiency_map_r_qq_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        out_file.Write()
        out_file.Close()

        if start_figure>0:
            out_file = TFile(HEPDATA_OUT+"Figure-aux_"+num_str+".root","RECREATE")
            out_file.cd()
            h2_num_m.Write("h2")
            print "Writing "+HEPDATA_OUT+"Figure-aux_"+num_str+".root"
            out_file.Write()
            out_file.Close()
            i_hep+=1

        can = TCanvas("can","can",1400,800)
        can.cd()

        list_palette = [100,57,109]
        gStyle.SetPalette(list_palette[1])
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)

        #can.SetGrid()
        can.SetRightMargin(0.16)
        can.SetTopMargin(0.12)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.12)
        h2_num_r_qq_presel.Draw("COLZTEXTE sames")
        gStyle.SetPaintTextFormat("1.3f");
        can.SetLogx()
        can.SetLogy()
        #can.SetLogz()
        h2_num_r_qq_presel.GetYaxis().SetTitle("LLP longitudinal decay length (m)")
        h2_num_r_qq_presel.GetXaxis().SetTitle("LLP transverse decay length (m)")
        h2_num_r_qq_presel.GetXaxis().SetTitleSize(0.04)
        h2_num_r_qq_presel.GetXaxis().SetTitleOffset(1.3)
        h2_num_r_qq_presel.GetYaxis().SetTitleSize(0.04)
        h2_num_r_qq_presel.GetYaxis().SetTitleOffset(1.2)
        h2_num_r_qq_presel.GetZaxis().SetTitle("TD-tagged jet efficiency")
        h2_num_r_qq_presel.GetZaxis().SetTitleSize(0.04)
        h2_num_r_qq_presel.GetZaxis().SetTitleOffset(1.2)
        h2_num_r_qq_presel.SetMaximum(.5)
        #Forcing axis range:
        gPad.RedrawAxis()
        can.RangeAxis(0.01,0.01,20,20)
        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can.Update()
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        #can.Print(out_fold+'Efficiency_map_r_qq_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.png')
        #can.Print(out_fold+'Efficiency_map_r_qq_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+'.pdf')
        can.Close()

        #out_file = TFile(out_fold+'Efficiency_map_r_qq_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root","RECREATE")
        #out_file.cd()
        #h2_num_r_qq_presel.Write("h2")
        #print "Writing "+out_fold+'Efficiency_map_r_qq_presel_pt_inclusive_'+b+"_"+str(R).replace(".","p")+".root"
        #out_file.Write()
        #out_file.Close()


        #Plot each pt bin!
        for i_pt in range(r_den_qq.shape[2]):
            #Plot
            can = TCanvas("can","can",1400,800)
            can.cd()

            list_palette = [100,55,104,57]
            gStyle.SetPalette(list_palette[0])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_r_qq_pt[i_pt].Draw("COLZTEXTE sames")
            #h2_den_r_qq_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_r_qq_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_r_qq_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_r_qq_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_r_qq_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_r_qq_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_r_qq_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_r_qq_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_r_qq_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_r_qq_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_r_qq_pt[i_pt].SetMaximum(1.)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_r_qq_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_r_qq_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()

            can = TCanvas("can","can",1400,800)
            can.cd()
            
            list_palette = [100,57,109]
            gStyle.SetPalette(list_palette[1])
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)

            #can.SetGrid()
            can.SetRightMargin(0.16)
            can.SetTopMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)
            h2_num_r_qq_presel_pt[i_pt].Draw("COLZTEXTE sames")
            gStyle.SetPaintTextFormat("1.3f");
            can.SetLogx()
            can.SetLogy()
            #can.SetLogz()
            h2_num_r_qq_presel_pt[i_pt].GetYaxis().SetTitle("LLP longitudinal decay length (m)")
            h2_num_r_qq_presel_pt[i_pt].GetXaxis().SetTitle("LLP transverse decay length (m)")
            h2_num_r_qq_presel_pt[i_pt].GetXaxis().SetTitleSize(0.04)
            h2_num_r_qq_presel_pt[i_pt].GetXaxis().SetTitleOffset(1.3)
            h2_num_r_qq_presel_pt[i_pt].GetYaxis().SetTitleSize(0.04)
            h2_num_r_qq_presel_pt[i_pt].GetYaxis().SetTitleOffset(1.2)
            h2_num_r_qq_presel_pt[i_pt].GetZaxis().SetTitle("TD-tagged jet efficiency")
            h2_num_r_qq_presel_pt[i_pt].GetZaxis().SetTitleSize(0.04)
            h2_num_r_qq_presel_pt[i_pt].GetZaxis().SetTitleOffset(1.2)
            h2_num_r_qq_presel_pt[i_pt].SetMaximum(.5)
            #Forcing axis range:
            gPad.RedrawAxis()
            can.RangeAxis(0.01,0.01,20,20)
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can.Update()
            drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.2,custom_spacing=0.4,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
            #can.Print(out_fold+'Efficiency_map_r_qq_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.png')
            #can.Print(out_fold+'Efficiency_map_r_qq_presel_pt_bin_'+str(i_pt)+'_'+b+"_"+str(R).replace(".","p")+'.pdf')
            can.Close()


        #print("pt bins: ")
        #print("m_den pt0: ", m_den[:,0,0])
        #print("m_den pt1: ", m_den[:,:,1])
        #print("m_den pt2: ", m_den[:,:,2])
        #print("m_den pt3: ", m_den[:,:,3])
        #print("m_den pt4: ", m_den[:,:,4])
        #print("m_den pt5: ", m_den[:,:,5])
        #print("m_den pt6: ", m_den[:,:,6])
        #print("m_den pt7: ", m_den[:,:,7])
        #print("m_den pt8: ", m_den[:,:,8])


def plot_maps_vs_mass(out_fold, sign, full_map_name="",R=0.4,label="",store_unc=False):

    masses = []

    diff_m_LLP0 = []
    true_unc_m_LLP0 = []
    diff_r_q_LLP0 = []
    true_unc_r_q_LLP0 = []
    diff_r_qq_LLP0 = []
    true_unc_r_qq_LLP0 = []

    diff_m_LLP1 = []
    true_unc_m_LLP1 = []
    diff_r_q_LLP1 = []
    true_unc_r_q_LLP1 = []
    diff_r_qq_LLP1 = []
    true_unc_r_qq_LLP1 = []

    diff_m     = []
    true_unc_m = []
    diff_r_q     = []
    true_unc_r_q = []
    diff_r_qq     = []
    true_unc_r_qq = []

    diff_m_presel_LLP0 = []
    true_unc_m_presel_LLP0 = []
    diff_r_q_presel_LLP0 = []
    true_unc_r_q_presel_LLP0 = []
    diff_r_qq_presel_LLP0 = []
    true_unc_r_qq_presel_LLP0 = []

    diff_m_presel_LLP1 = []
    true_unc_m_presel_LLP1 = []
    diff_r_q_presel_LLP1 = []
    true_unc_r_q_presel_LLP1 = []
    diff_r_qq_presel_LLP1 = []
    true_unc_r_qq_presel_LLP1 = []

    diff_m_presel     = []
    true_unc_m_presel = []
    diff_r_q_presel     = []
    true_unc_r_q_presel = []
    diff_r_qq_presel     = []
    true_unc_r_qq_presel = []


    #pred_maps = {}
    for b in sign:

        map_file = out_fold+"Tag_pred_"+b+"_"+str(R).replace(".","p")+".yaml"
        print("Opening... ", map_file)
        with open(map_file) as f:
            pred_map = yaml.load(f, Loader=yaml.Loader)
            f.close()

        masses.append(pred_map['m'])

        diff_m_LLP0.append(abs(pred_map['diff_m_LLP0']))
        true_unc_m_LLP0.append(pred_map['true_unc_m_LLP0'])
        diff_r_q_LLP0.append(abs(pred_map['diff_r_q_LLP0']))
        true_unc_r_q_LLP0.append(pred_map['true_unc_r_q_LLP0'])
        diff_r_qq_LLP0.append(abs(pred_map['diff_r_qq_LLP0']))
        true_unc_r_qq_LLP0.append(pred_map['true_unc_r_qq_LLP0'])

        diff_m_LLP1.append(abs(pred_map['diff_m_LLP1']))
        true_unc_m_LLP1.append(pred_map['true_unc_m_LLP1'])
        diff_r_q_LLP1.append(abs(pred_map['diff_r_q_LLP1']))
        true_unc_r_q_LLP1.append(pred_map['true_unc_r_q_LLP1'])
        diff_r_qq_LLP1.append(abs(pred_map['diff_r_qq_LLP1']))
        true_unc_r_qq_LLP1.append(pred_map['true_unc_r_qq_LLP1'])

        true_m = pred_map['true_m_LLP0'] + pred_map['true_m_LLP1']
        pred_m = pred_map['pred_m_LLP0'] + pred_map['pred_m_LLP1']
        diff_m.append( 100.*(pred_m - true_m)/true_m if true_m>0 else 0  )
        true_unc_m.append( 100./np.sqrt( true_m  ) if true_m>0 else 0  )

        true_r_q = pred_map['true_r_q_LLP0'] + pred_map['true_r_q_LLP1']
        pred_r_q = pred_map['pred_r_q_LLP0'] + pred_map['pred_r_q_LLP1']
        diff_r_q.append( 100.*(pred_r_q - true_r_q)/true_r_q if true_r_q>0 else 0  )
        true_unc_r_q.append( 100./np.sqrt( true_r_q  ) if true_r_q>0 else 0  )

        true_r_qq = pred_map['true_r_qq_LLP0'] + pred_map['true_r_qq_LLP1']
        pred_r_qq = pred_map['pred_r_qq_LLP0'] + pred_map['pred_r_qq_LLP1']
        diff_r_qq.append( 100.*(pred_r_qq - true_r_qq)/true_r_qq if true_r_qq>0 else 0  )
        true_unc_r_qq.append( 100./np.sqrt( true_r_qq  ) if true_r_qq>0 else 0  )

        #presel
        diff_m_presel_LLP0.append(abs(pred_map['diff_m_presel_LLP0']))
        true_unc_m_presel_LLP0.append(pred_map['true_unc_m_presel_LLP0'])
        diff_r_q_presel_LLP0.append(abs(pred_map['diff_r_q_presel_LLP0']))
        true_unc_r_q_presel_LLP0.append(pred_map['true_unc_r_q_presel_LLP0'])
        diff_r_qq_presel_LLP0.append(abs(pred_map['diff_r_qq_presel_LLP0']))
        true_unc_r_qq_presel_LLP0.append(pred_map['true_unc_r_qq_presel_LLP0'])

        diff_m_presel_LLP1.append(abs(pred_map['diff_m_presel_LLP1']))
        true_unc_m_presel_LLP1.append(pred_map['true_unc_m_presel_LLP1'])
        diff_r_q_presel_LLP1.append(abs(pred_map['diff_r_q_presel_LLP1']))
        true_unc_r_q_presel_LLP1.append(pred_map['true_unc_r_q_presel_LLP1'])
        diff_r_qq_presel_LLP1.append(abs(pred_map['diff_r_qq_presel_LLP1']))
        true_unc_r_qq_presel_LLP1.append(pred_map['true_unc_r_qq_presel_LLP1'])

        true_m_presel = pred_map['true_m_presel_LLP0'] + pred_map['true_m_presel_LLP1']
        pred_m_presel = pred_map['pred_m_presel_LLP0'] + pred_map['pred_m_presel_LLP1']
        diff_m_presel.append( 100.*(pred_m_presel - true_m_presel)/true_m_presel if true_m_presel>0 else 0  )
        true_unc_m_presel.append( 100./np.sqrt( true_m_presel  ) if true_m_presel>0 else 0  )

        true_r_q_presel = pred_map['true_r_q_presel_LLP0'] + pred_map['true_r_q_presel_LLP1']
        pred_r_q_presel = pred_map['pred_r_q_presel_LLP0'] + pred_map['pred_r_q_presel_LLP1']
        diff_r_q_presel.append( 100.*(pred_r_q_presel - true_r_q_presel)/true_r_q_presel if true_r_q_presel>0 else 0  )
        true_unc_r_q_presel.append( 100./np.sqrt( true_r_q_presel  ) if true_r_q_presel>0 else 0  )

        true_r_qq_presel = pred_map['true_r_qq_presel_LLP0'] + pred_map['true_r_qq_presel_LLP1']
        pred_r_qq_presel = pred_map['pred_r_qq_presel_LLP0'] + pred_map['pred_r_qq_presel_LLP1']
        diff_r_qq_presel.append( 100.*(pred_r_qq_presel - true_r_qq_presel)/true_r_qq_presel if true_r_qq_presel>0 else 0  )
        true_unc_r_qq_presel.append( 100./np.sqrt( true_r_qq_presel  ) if true_r_qq_presel>0 else 0  )




    masses = np.array(masses)

    #diff_m_LLP0 = np.array(diff_m_LLP0)
    #true_unc_m_LLP0 = np.array(true_unc_m_LLP0)
    #diff_r_q_LLP0 = np.array(diff_r_q_LLP0)
    #true_unc_r_q_LLP0 = np.array(true_unc_r_q_LLP0)
    #diff_r_qq_LLP0 = np.array(diff_r_qq_LLP0)
    #true_unc_r_qq_LLP0 = np.array(true_unc_r_qq_LLP0)

    #diff_m_LLP1 = np.array(diff_m_LLP1)
    #true_unc_m_LLP1 = np.array(true_unc_m_LLP1)
    #diff_r_q_LLP1 = np.array(diff_r_q_LLP1)
    #true_unc_r_q_LLP1 = np.array(true_unc_r_q_LLP1)
    #diff_r_qq_LLP1 = np.array(diff_r_qq_LLP1)
    #true_unc_r_qq_LLP1 = np.array(true_unc_r_qq_LLP1)

    diff_m = np.array(diff_m)
    true_unc_m = np.array(true_unc_m)
    true_unc_m_n = -np.array(true_unc_m)

    diff_r_q = np.array(diff_r_q)
    true_unc_r_q = np.array(true_unc_r_q)
    true_unc_r_q_n = -np.array(true_unc_r_q)

    diff_r_qq = np.array(diff_r_qq)
    true_unc_r_qq = np.array(true_unc_r_qq)
    true_unc_r_qq_n = -np.array(true_unc_r_qq)


    diff_m_presel = np.array(diff_m_presel)
    true_unc_m_presel = np.array(true_unc_m_presel)
    true_unc_m_presel_n = -np.array(true_unc_m_presel)

    diff_r_q_presel = np.array(diff_r_q_presel)
    true_unc_r_q_presel = np.array(true_unc_r_q_presel)
    true_unc_r_q_presel_n = -np.array(true_unc_r_q_presel)

    diff_r_qq_presel = np.array(diff_r_qq_presel)
    true_unc_r_qq_presel = np.array(true_unc_r_qq_presel)
    true_unc_r_qq_presel_n = -np.array(true_unc_r_qq_presel)

    colors = ['blue','blue','crimson','crimson','green','green','darkorange','darkorange','gray','gray','magenta','magenta','chocolate','chocolate','yellow','yellow','black','black','olive','olive']
    linestyles = ['-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-','--',]
    pointstyles = ['o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s',]
    facecolors= ['blue','none','crimson','none','green','none','darkorange','none','gray','none','magenta','none','chocolate','none','yellow','none','black','none','olive','none']

    plt.rcParams.update({"font.size": 20})  

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(masses, diff_m, color='black', lw=0, marker='o',label='diff pred-true merged')
    plt.plot(masses, true_unc_m, color='black', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_m_n, color='black', lw=3, linestyle='-')

    plt.plot(masses, diff_r_q, color='red', lw=0, marker='o',label='diff pred-true resolved 1q')
    plt.plot(masses, true_unc_r_q, color='red', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_r_q_n, color='red', lw=3, linestyle='-')

    plt.plot(masses, diff_r_qq, color='green', lw=0, marker='o',label='diff pred-true resolved 2q')
    plt.plot(masses, true_unc_r_qq, color='green', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_r_qq_n, color='green', lw=3, linestyle='-')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-6., 6.])
    if "_ctau500" in label:
        plt.ylim([-20., 20.])
    elif "_ctau3000" in label:
        plt.ylim([-20., 20.])
    else:
        plt.ylim([-6., 6.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP mass (GeV)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('R = '+str(R))
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+label+".pdf")
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+label+".png")
    print("Written ", out_fold+"Prediction_closure_"+str(R).replace(".","p")+label+".pdf")



    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    plt.plot(masses, diff_m_presel, color='black', lw=0, marker='o',label='diff merged')
    plt.plot(masses, true_unc_m_presel, color='black', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_m_presel_n, color='black', lw=3, linestyle='-')

    plt.plot(masses, diff_r_q_presel, color='red', lw=0, marker='o',label='diff resolved 1q')
    plt.plot(masses, true_unc_r_q_presel, color='red', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_r_q_presel_n, color='red', lw=3, linestyle='-')

    plt.plot(masses, diff_r_qq_presel, color='green', lw=0, marker='o',label='diff resolved 2q')
    plt.plot(masses, true_unc_r_qq_presel, color='green', lw=3, linestyle='-',label='stat. unc.')
    plt.plot(masses, true_unc_r_qq_presel_n, color='green', lw=3, linestyle='-')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-6., 6.])
    if "_ctau500" in label:
        plt.ylim([-150., 150.])
    elif "_ctau3000" in label:
        plt.ylim([-150., 150.])
    else:
        plt.ylim([-6., 6.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP mass (GeV)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R))
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+label+".pdf")
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+label+".png")
    print("Written ", out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+label+".pdf")

    if store_unc:
        print("Storing per-mass nonclosure uncertainties: Prediction_nonclosure_*npy")
        print(masses)
        np.save(out_fold+"Prediction_nonclosure_unc_masses.npy",masses)
        print(diff_m)
        print(diff_r_q)
        print(diff_r_qq)
        np.save(out_fold+"Prediction_nonclosure_unc_m_"+str(R).replace(".","p")+label+".npy",diff_m)
        np.save(out_fold+"Prediction_nonclosure_unc_r_q_"+str(R).replace(".","p")+label+".npy",diff_r_q)
        np.save(out_fold+"Prediction_nonclosure_unc_r_qq_"+str(R).replace(".","p")+label+".npy",diff_r_qq)

        print("---")
        print(diff_m_presel)
        print(diff_r_q_presel)
        print(diff_r_qq_presel)

        np.save(out_fold+"Prediction_nonclosure_unc_presel_m_"+str(R).replace(".","p")+label+".npy",diff_m_presel)
        np.save(out_fold+"Prediction_nonclosure_unc_presel_r_q_"+str(R).replace(".","p")+label+".npy",diff_r_q_presel)
        np.save(out_fold+"Prediction_nonclosure_unc_presel_r_qq_"+str(R).replace(".","p")+label+".npy",diff_r_qq_presel)



def plot_maps_vs_ctau(out_fold, sign, full_map_name="",R=0.4):

    masses = []
    ctaus = []

    diff_m_LLP0 = {}
    true_unc_m_LLP0 = {}
    diff_r_q_LLP0 = {}
    true_unc_r_q_LLP0 = {}
    diff_r_qq_LLP0 = {}
    true_unc_r_qq_LLP0 = {}

    diff_m_LLP1 = {}
    true_unc_m_LLP1 = {}
    diff_r_q_LLP1 = {}
    true_unc_r_q_LLP1 = {}
    diff_r_qq_LLP1 = {}
    true_unc_r_qq_LLP1 = {}

    diff_m     = {}
    true_unc_m = {}
    true_unc_m_n = {}
    diff_r_q     = {}
    true_unc_r_q = {}
    true_unc_r_q_n = {}
    diff_r_qq     = {}
    true_unc_r_qq = {}
    true_unc_r_qq_n = {}

    #presel
    diff_m_presel_LLP0 = {}
    true_unc_m_presel_LLP0 = {}
    diff_r_q_presel_LLP0 = {}
    true_unc_r_q_presel_LLP0 = {}
    diff_r_qq_presel_LLP0 = {}
    true_unc_r_qq_presel_LLP0 = {}

    diff_m_presel_LLP1 = {}
    true_unc_m_presel_LLP1 = {}
    diff_r_q_presel_LLP1 = {}
    true_unc_r_q_presel_LLP1 = {}
    diff_r_qq_presel_LLP1 = {}
    true_unc_r_qq_presel_LLP1 = {}

    diff_m_presel     = {}
    true_unc_m_presel = {}
    true_unc_m_presel_n = {}
    diff_r_q_presel     = {}
    true_unc_r_q_presel = {}
    true_unc_r_q_presel_n = {}
    diff_r_qq_presel     = {}
    true_unc_r_qq_presel = {}
    true_unc_r_qq_presel_n = {}



    #pred_maps = {}
    for b in sign:

        map_file = out_fold+"Tag_pred_"+b+"_"+str(R).replace(".","p")+".yaml"
        print("Opening... ", map_file)
        with open(map_file) as f:
            pred_map = yaml.load(f, Loader=yaml.Loader)
            f.close()

        masses.append(pred_map['m'])

        diff_m_LLP0[pred_map['m']] = []
        true_unc_m_LLP0[pred_map['m']] = []
        diff_r_q_LLP0[pred_map['m']] = []
        true_unc_r_q_LLP0[pred_map['m']] = []
        diff_r_qq_LLP0[pred_map['m']] = []
        true_unc_r_qq_LLP0[pred_map['m']] = []
        
        diff_m_LLP1[pred_map['m']] = []
        true_unc_m_LLP1[pred_map['m']] = []
        diff_r_q_LLP1[pred_map['m']] = []
        true_unc_r_q_LLP1[pred_map['m']] = []
        diff_r_qq_LLP1[pred_map['m']] = []
        true_unc_r_qq_LLP1[pred_map['m']] = []
        
        diff_m    [pred_map['m']] = []
        true_unc_m[pred_map['m']] = []
        true_unc_m_n[pred_map['m']] = []
        diff_r_q    [pred_map['m']] = []
        true_unc_r_q[pred_map['m']] = []
        true_unc_r_q_n[pred_map['m']] = []
        diff_r_qq    [pred_map['m']] = []
        true_unc_r_qq[pred_map['m']] = []
        true_unc_r_qq_n[pred_map['m']] = []



        diff_m_presel_LLP0[pred_map['m']] = []
        true_unc_m_presel_LLP0[pred_map['m']] = []
        diff_r_q_presel_LLP0[pred_map['m']] = []
        true_unc_r_q_presel_LLP0[pred_map['m']] = []
        diff_r_qq_presel_LLP0[pred_map['m']] = []
        true_unc_r_qq_presel_LLP0[pred_map['m']] = []
        
        diff_m_presel_LLP1[pred_map['m']] = []
        true_unc_m_presel_LLP1[pred_map['m']] = []
        diff_r_q_presel_LLP1[pred_map['m']] = []
        true_unc_r_q_presel_LLP1[pred_map['m']] = []
        diff_r_qq_presel_LLP1[pred_map['m']] = []
        true_unc_r_qq_presel_LLP1[pred_map['m']] = []
        
        diff_m_presel    [pred_map['m']] = []
        true_unc_m_presel[pred_map['m']] = []
        true_unc_m_presel_n[pred_map['m']] = []
        diff_r_q_presel    [pred_map['m']] = []
        true_unc_r_q_presel[pred_map['m']] = []
        true_unc_r_q_presel_n[pred_map['m']] = []
        diff_r_qq_presel    [pred_map['m']] = []
        true_unc_r_qq_presel[pred_map['m']] = []
        true_unc_r_qq_presel_n[pred_map['m']] = []


    for b in sign:

        map_file = out_fold+"Tag_pred_"+b+"_"+str(R).replace(".","p")+".yaml"
        print("Opening... ", map_file)
        with open(map_file) as f:
            pred_map = yaml.load(f, Loader=yaml.Loader)
            f.close()

        m = pred_map['m']
        ctaus.append(samples[b]['ctau'])

        diff_m_LLP0[m].append(abs(pred_map['diff_m_LLP0']))
        true_unc_m_LLP0[m].append(pred_map['true_unc_m_LLP0'])
        diff_r_q_LLP0[m].append(abs(pred_map['diff_r_q_LLP0']))
        true_unc_r_q_LLP0[m].append(pred_map['true_unc_r_q_LLP0'])
        diff_r_qq_LLP0[m].append(abs(pred_map['diff_r_qq_LLP0']))
        true_unc_r_qq_LLP0[m].append(pred_map['true_unc_r_qq_LLP0'])

        diff_m_LLP1[m].append(abs(pred_map['diff_m_LLP1']))
        true_unc_m_LLP1[m].append(pred_map['true_unc_m_LLP1'])
        diff_r_q_LLP1[m].append(abs(pred_map['diff_r_q_LLP1']))
        true_unc_r_q_LLP1[m].append(pred_map['true_unc_r_q_LLP1'])
        diff_r_qq_LLP1[m].append(abs(pred_map['diff_r_qq_LLP1']))
        true_unc_r_qq_LLP1[m].append(pred_map['true_unc_r_qq_LLP1'])

        true_m = pred_map['true_m_LLP0'] + pred_map['true_m_LLP1']
        pred_m = pred_map['pred_m_LLP0'] + pred_map['pred_m_LLP1']
        diff_m[m].append( 100.*(pred_m - true_m)/true_m if true_m>0 else 0  )
        true_unc_m[m].append( 100./np.sqrt( true_m  ) if true_m>0 else 0  )

        true_r_q = pred_map['true_r_q_LLP0'] + pred_map['true_r_q_LLP1']
        pred_r_q = pred_map['pred_r_q_LLP0'] + pred_map['pred_r_q_LLP1']
        diff_r_q[m].append( 100.*(pred_r_q - true_r_q)/true_r_q if true_r_q>0 else 0  )
        true_unc_r_q[m].append( 100./np.sqrt( true_r_q  ) if true_r_q>0 else 0  )

        true_r_qq = pred_map['true_r_qq_LLP0'] + pred_map['true_r_qq_LLP1']
        pred_r_qq = pred_map['pred_r_qq_LLP0'] + pred_map['pred_r_qq_LLP1']
        diff_r_qq[m].append( 100.*(pred_r_qq - true_r_qq)/true_r_qq if true_r_qq>0 else 0  )
        true_unc_r_qq[m].append( 100./np.sqrt( true_r_qq  ) if true_r_qq>0 else 0  )

        #presel
        diff_m_presel_LLP0[m].append(abs(pred_map['diff_m_presel_LLP0']))
        true_unc_m_presel_LLP0[m].append(pred_map['true_unc_m_presel_LLP0'])
        diff_r_q_presel_LLP0[m].append(abs(pred_map['diff_r_q_presel_LLP0']))
        true_unc_r_q_presel_LLP0[m].append(pred_map['true_unc_r_q_presel_LLP0'])
        diff_r_qq_presel_LLP0[m].append(abs(pred_map['diff_r_qq_presel_LLP0']))
        true_unc_r_qq_presel_LLP0[m].append(pred_map['true_unc_r_qq_presel_LLP0'])

        diff_m_presel_LLP1[m].append(abs(pred_map['diff_m_presel_LLP1']))
        true_unc_m_presel_LLP1[m].append(pred_map['true_unc_m_presel_LLP1'])
        diff_r_q_presel_LLP1[m].append(abs(pred_map['diff_r_q_presel_LLP1']))
        true_unc_r_q_presel_LLP1[m].append(pred_map['true_unc_r_q_presel_LLP1'])
        diff_r_qq_presel_LLP1[m].append(abs(pred_map['diff_r_qq_presel_LLP1']))
        true_unc_r_qq_presel_LLP1[m].append(pred_map['true_unc_r_qq_presel_LLP1'])

        true_m_presel = pred_map['true_m_presel_LLP0'] + pred_map['true_m_presel_LLP1']
        pred_m_presel = pred_map['pred_m_presel_LLP0'] + pred_map['pred_m_presel_LLP1']
        diff_m_presel[m].append( 100.*(pred_m_presel - true_m_presel)/true_m_presel if true_m_presel>0 else 0  )
        true_unc_m_presel[m].append( 100./np.sqrt( true_m_presel  ) if true_m_presel>0 else 0  )

        true_r_q_presel = pred_map['true_r_q_presel_LLP0'] + pred_map['true_r_q_presel_LLP1']
        pred_r_q_presel = pred_map['pred_r_q_presel_LLP0'] + pred_map['pred_r_q_presel_LLP1']
        diff_r_q_presel[m].append( 100.*(pred_r_q_presel - true_r_q_presel)/true_r_q_presel if true_r_q_presel>0 else 0  )
        true_unc_r_q_presel[m].append( 100./np.sqrt( true_r_q_presel  ) if true_r_q_presel>0 else 0  )

        true_r_qq_presel = pred_map['true_r_qq_presel_LLP0'] + pred_map['true_r_qq_presel_LLP1']
        pred_r_qq_presel = pred_map['pred_r_qq_presel_LLP0'] + pred_map['pred_r_qq_presel_LLP1']
        diff_r_qq_presel[m].append( 100.*(pred_r_qq_presel - true_r_qq_presel)/true_r_qq_presel if true_r_qq_presel>0 else 0  )
        true_unc_r_qq_presel[m].append( 100./np.sqrt( true_r_qq_presel  ) if true_r_qq_presel>0 else 0  )
        

    print("Now organize the plots")
    masses = np.unique(np.array(masses))
    ctaus = np.unique(np.array(ctaus)/1000.)


    colors = ['blue','darkorange','crimson','green','gray','magenta','chocolate','yellow','black','olive']
    linestyles = ['-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-','--',]
    pointstyles = ['o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s','o','s',]
    facecolors= ['blue','none','crimson','none','green','none','darkorange','none','gray','none','magenta','none','chocolate','none','yellow','none','black','none','olive','none']

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):
        print(m)
        print(diff_m[m])
        #m
        diff_m[m] = np.array(diff_m[m])
        true_unc_m[m] = np.array(true_unc_m[m])
        true_unc_m_n[m] = -np.array(true_unc_m[m])

        plt.plot(ctaus, diff_m[m], color=colors[i], lw=0, linestyle='--', marker='o')#label='diff merged')
        plt.plot(ctaus, true_unc_m[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')#'stat. unc.')
        plt.plot(ctaus, true_unc_m_n[m], color=colors[i], lw=3, linestyle='-')

        #plt.plot(ctaus, diff_r_q, color='red', lw=0, marker='o',label='diff resolved 1q')
        #plt.plot(ctaus, true_unc_r_q, color='red', lw=3, linestyle='-',label='stat. unc.')
        #plt.plot(ctaus, true_unc_r_q_n, color='red', lw=3, linestyle='-')

        #plt.plot(ctaus, diff_r_qq, color='green', lw=0, marker='o',label='diff resolved 2q')
        #plt.plot(ctaus, true_unc_r_qq, color='green', lw=3, linestyle='-',label='stat. unc.')
        #plt.plot(ctaus, true_unc_r_qq_n, color='green', lw=3, linestyle='-')

    plt.plot(dummy_x, dummy_y, color='black', lw=0, marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-10., 10.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', merged topology')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_m_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_m_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_m_vs_ctau.pdf")

    #m presel
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):
        diff_m_presel[m] = np.array(diff_m_presel[m])
        true_unc_m_presel[m] = np.array(true_unc_m_presel[m])
        true_unc_m_presel_n[m] = -np.array(true_unc_m_presel[m])

        plt.plot(ctaus, diff_m_presel[m], color=colors[i], lw=1, linestyle='--', marker='o')#label='diff merged')
        plt.plot(ctaus, true_unc_m_presel[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')#'stat. unc.')
        plt.plot(ctaus, true_unc_m_presel_n[m], color=colors[i], lw=3, linestyle='-')


    plt.plot(dummy_x, dummy_y, color='black', lw=0, marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-150., 150.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', merged topology')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_m_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_m_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_m_vs_ctau.pdf")



    #q
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):
        print(m)
        print(diff_m[m])
        #m
        diff_r_q[m] = np.array(diff_r_q[m])
        true_unc_r_q[m] = np.array(true_unc_r_q[m])
        true_unc_r_q_n[m] = -np.array(true_unc_r_q[m])

        plt.plot(ctaus, diff_r_q[m], color=colors[i], lw=0, linestyle='--', marker='o')
        plt.plot(ctaus, true_unc_r_q[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')
        plt.plot(ctaus, true_unc_r_q_n[m], color=colors[i], lw=3, linestyle='-')

    plt.plot(dummy_x, dummy_y, color='black', lw=0, linestyle='--', marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-10., 10.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', resolved topology, 1 quark in acceptance')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_q_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_q_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_q_vs_ctau.pdf")

    #q presel
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):

        diff_r_q_presel[m] = np.array(diff_r_q_presel[m])
        true_unc_r_q_presel[m] = np.array(true_unc_r_q_presel[m])
        true_unc_r_q_presel_n[m] = -np.array(true_unc_r_q_presel[m])

        plt.plot(ctaus, diff_r_q_presel[m], color=colors[i], lw=1, linestyle='--', marker='o')
        plt.plot(ctaus, true_unc_r_q_presel[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')
        plt.plot(ctaus, true_unc_r_q_presel_n[m], color=colors[i], lw=3, linestyle='-')

    plt.plot(dummy_x, dummy_y, color='black', lw=0, linestyle='--', marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-150., 150.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', resolved topology, 1 quark in acceptance')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_q_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_q_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_q_vs_ctau.pdf")
    
    #qq
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):
        print(m)
        print(diff_m[m])
        #m
        diff_r_qq[m] = np.array(diff_r_qq[m])
        true_unc_r_qq[m] = np.array(true_unc_r_qq[m])
        true_unc_r_qq_n[m] = -np.array(true_unc_r_qq[m])

        plt.plot(ctaus, diff_r_qq[m], color=colors[i], lw=0, linestyle='--', marker='o')#label='diff merged')
        plt.plot(ctaus, true_unc_r_qq[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')#'stat. unc.')
        plt.plot(ctaus, true_unc_r_qq_n[m], color=colors[i], lw=3, linestyle='-')

    plt.plot(dummy_x, dummy_y, color='black', lw=0, linestyle='--', marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-10., 10.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', resolved topology, 2 quarks in acceptance')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_qq_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_qq_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_"+str(R).replace(".","p")+"_qq_vs_ctau.pdf")


    #qq presel
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 20})  
    dummy_x = np.array([])
    dummy_y = np.array([])

    for i,m in enumerate(masses):
        print(m)
        print(diff_m[m])
        #m
        diff_r_qq_presel[m] = np.array(diff_r_qq_presel[m])
        true_unc_r_qq_presel[m] = np.array(true_unc_r_qq_presel[m])
        true_unc_r_qq_presel_n[m] = -np.array(true_unc_r_qq_presel[m])

        plt.plot(ctaus, diff_r_qq_presel[m], color=colors[i], lw=1, linestyle='--', marker='o')#label='diff merged')
        plt.plot(ctaus, true_unc_r_qq_presel[m], color=colors[i], lw=3, linestyle='-',label='m = '+str(m)+ ' GeV')#'stat. unc.')
        plt.plot(ctaus, true_unc_r_qq_presel_n[m], color=colors[i], lw=3, linestyle='-')

    plt.plot(dummy_x, dummy_y, color='black', lw=0, linestyle='--', marker='o', label='diff pred. - true')
    plt.plot(dummy_x, dummy_y, color='black', lw=3, linestyle='-',label='stat. unc.')

    #plt.ylim([0.57, 1.01])
    plt.ylim([-150., 150.])
    #plt.xscale("log")
    plt.ylabel("Closure (%)")
    plt.xlabel("LLP ctau (m)")
    plt.legend(loc="upper right", title="")
    plt.grid(True)
    plt.title('dR='+str(R)+', resolved topology, 2 quarks in acceptance')
    #hep.cms.label("Supplementary",data=False, year=int(ERA))
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_qq_vs_ctau.pdf")
    fig.savefig(out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_qq_vs_ctau.png")
    print("Written ", out_fold+"Prediction_closure_presel_"+str(R).replace(".","p")+"_qq_vs_ctau.pdf")


def predict_merged():
    print("merged")

def predict_resolved():
    print("resolved")


#create_maps(sign,skip_even=True,R=0.8)
#use_maps_updated(sign,skip_odd=True,cross_maps="",R=0.8)
#plot_maps(sign,R=0.8)

r_list = [0.4,0.6,0.8,1.0,1.2]
#r_list = [0.4,1.0,1.2]
#r_list = [0.6,0.8]
r_list = [0.8]
#r_list = [0.4,0.8,1.0,1.2]
for r in r_list:

    #mix all eras together
    #also maps with all events!
    #create_maps(MAIN_OUT_ALL,sign,skip_even=False,R=r,eras=["2016","2017","2018"])
    #plot_maps(MAIN_OUT_ALL,["SUSY_mh1800_"],R=r)
    plot_maps(MAIN_OUT_ALL,sign,R=r,start_figure = 6)

    ###
    #Only odd/even events:
    #create_maps(MAIN_OUT,sign,skip_even=True,R=r,eras=["2016","2017","2018"])
    ###create_maps_prev(sign,skip_even=True,R=r)
    #use_maps_updated(MAIN_OUT,["SUSY_mh400_"],skip_odd=True,cross_maps="",R=r,eras=["2016","2017","2018"])
    #plot_maps_vs_mass(MAIN_OUT,sign,R=r)

    #Remove mass too?
    #create_maps(MAIN_OUT,["SUSY_",],skip_even=True,R=r,eras=["2016","2017","2018"])
    #use_maps_updated(MAIN_OUT,["SUSY_",],skip_odd=True,cross_maps="",R=r,eras=["2016","2017","2018"])
    #plot_maps_vs_mass(MAIN_OUT,["SUSY_",],R=r)

    ##vs ctau: need to recompute the outcome
    ##for s in sign_ctau:
    #tmp_sign_ctau = sign_500+sign_3000#["SUSY_mh127_ctau500_","SUSY_mh127_ctau3000_","SUSY_mh200_ctau500_","SUSY_mh200_ctau3000_",]
    #for s in tmp_sign_ctau:
    #    ##print r, s
    #    ##print(s.split('_'))
    #    new_s = s.split('_')[0]+"_"+s.split('_')[1]+"_"
    #    ##print(new_s)
    #    use_maps_updated(MAIN_OUT,[s],skip_odd=True,cross_maps="",full_map_name=new_s,R=r,eras=["2016","2017","2018"])
    ##plot_maps_vs_ctau(MAIN_OUT,tmp_sign_ctau,R=r)
    #plot_maps_vs_mass(MAIN_OUT,sign,R=r,label="")
    #if r==0.8:
    #    plot_maps_vs_mass(MAIN_OUT,sign_500,R=r,label="_ctau500",store_unc=True)
    #    plot_maps_vs_mass(MAIN_OUT,sign_3000,R=r,label="_ctau3000",store_unc=True)

    #maps with nonclosure uncertainties:
    #plot_maps(MAIN_OUT_ALL,sign,R=r,add_unc=True,unc_fold=MAIN_OUT)

    ###plot maps with only even/odd events as a cross-check
    #plot_maps(MAIN_OUT,["SUSY_mh127_","SUSY_mh200_","SUSY_mh400_","SUSY_mh1000_","SUSY_mh1800_"],R=r)


#OLD
#    #create_maps(sign,skip_even=True,R=r)
#    #use_maps_updated(sign,skip_odd=True,cross_maps="",R=r)
#
#    #separate prediction for each m/ct combination
#    '''
#    for s in sign_ctau:
#        print r, s
#        print(s.split('_'))
#        new_s = s.split('_')[0]+"_"+s.split('_')[1]+"_"
#        print(new_s)
#        use_maps_updated([s],skip_odd=True,cross_maps="",full_map_name=new_s,R=r)
#        exit()
#    '''
#    #plot_maps_vs_ctau(sign_ctau,R=r)

#use_maps_updated(sign,skip_odd=True,cross_maps="",R=1.0)
#use_maps_updated(sign,skip_odd=True,cross_maps="",R=1.2)

#1. Tune R such that in resolved topology and 2 quarks in acceptance we have mostly 2 tags
########################################
