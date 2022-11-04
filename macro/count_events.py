#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
import yaml
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TMultiGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
#import matplotlib.pyplot as plt
from collections import defaultdict
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
import tensorflow as tf
from tensorflow import keras
from prettytable import PrettyTable

ERA = "2017"

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi["HighMET"]["tot"]
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi["HighMET"]["tot"]
if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI  = lumi["HighMET"]["tot"]
#from Analyzer.LLP2017.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = [
    #"skim_pickevents_275309_875415530",#Run2016B
    #"skim_pickevents_276811_244034371",#Run2016D##this fails the opposite emisphere requirement
    #"skim_pickevents_277096_153555117",#Run2016E #beam halo
    #"n3n2-n1-hbb-hbb_mh400_pl1000",
    #"n3n2-n1-hbb-hbb_mh300_pl1000",
    #"n3n2-n1-hbb-hbb_mh200_pl1000",
    #"n3n2-n1-hbb-hbb_mh127_pl1000",
    #"skim_pickevents_279029_387083289",#Run2016G
    #"skim_pickevents_279931_4073593513",#Run2016G
    #"skim_pickevents_302485_987393715",
    #"skim_pickevents_319910_457404349",
    #"test_signal",
    #"SUSY_mh127_pl1000",
    #"SUSY_mh150_pl1000",
    #"SUSY_mh175_pl1000",
    #"SUSY_mh200_pl1000",
    #"SUSY_mh250_pl1000",
    #"SUSY_mh300_pl1000",
    #"SUSY_mh400_pl1000",
    ##central
    #'SUSY_mh127_ctau500_HH', 
    #'SUSY_mh127_ctau3000_HH',
    #'SUSY_mh150_ctau500_HH', 
    #'SUSY_mh150_ctau3000_HH',
    #'SUSY_mh175_ctau500_HH', 
    #'SUSY_mh175_ctau3000_HH',
    'SUSY_mh200_ctau500_HH', 
    #'SUSY_mh200_ctau3000_HH',
    #'SUSY_mh250_ctau500_HH', 
    #'SUSY_mh250_ctau3000_HH',
    #'SUSY_mh300_ctau500_HH', 
    #'SUSY_mh300_ctau3000_HH',
    #'SUSY_mh400_ctau500_HH', 
    #'SUSY_mh400_ctau3000_HH',
    #'SUSY_mh600_ctau500_HH', 
    #'SUSY_mh600_ctau3000_HH',
    #'SUSY_mh800_ctau500_HH', 
    #'SUSY_mh800_ctau3000_HH',
    'SUSY_mh1000_ctau500_HH',
    #'SUSY_mh1000_ctau3000_HH',
    #'SUSY_mh1250_ctau500_HH',
    #'SUSY_mh1250_ctau3000_HH',
    #'SUSY_mh1500_ctau500_HH',
    #'SUSY_mh1500_ctau3000_HH',
    #'SUSY_mh1800_ctau500_HH',
    #'SUSY_mh1800_ctau3000_HH',
    #"HighMET",
    #"HighMETBH",
    #"SingleMuon"
]


#data = [
#    'SUSY_mh400_ctau500_HH', 
#    'SUSY_mh400_ctau3000_HH',
#]

#data = ["HighMET"]
##MAIN = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2017/"
#MAIN = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR/"
##MAIN = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_ZtoMM/"
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_Gen/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_bin_1_2/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_time_smearing_no_cuts/"
#this for debug:
#go even before!
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared_correlated/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared_uncorrelated/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen_time_smeared_no_cuts_debug/"

#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_positive_jets_BH/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_bin_1_2/"
#OUT = "plots/v6_calo_AOD_2017_BeamHalo_AN/"
#OUT = "plots/v6_calo_AOD_2017_MinDPhi/"
OUT = "plots/v6_calo_AOD_2017_compare_princeton/"
#OUT = "plots/v6_calo_AOD_"+ERA+"_compare_time_smearing/"
OUT = "plots/v6_calo_AOD_"+ERA+"_debug/"

OUT = "plots/v6_calo_AOD_"+ERA+"_time_smeared_correlation_studies/"

##Keep this as a reference!
#OUT = "plots/v6_calo_AOD_"+ERA+"_time_smeared_correlated/"



def count_gen_central(force_HH=True):
    print "\n"
    print "Ntuples: ", MAIN
    print "\n"
    for d in data:
        for j, ss in enumerate(samples[d]['files']):
            tmp_file = TFile(MAIN+ss+".root","READ")
            tree = tmp_file.Get("tree")
            nentries = tree.GetEntries()
        if "_HZ" not in d and "_ZZ" not in d and force_HH:
            print "    '"+samples[d]['files'][0]+"_HH' : {"
        else:
            print "    '"+samples[d]['files'][0]+"' : {"
        print "        'nevents' : "+str(nentries)+","
        if "_HZ" in d:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0].replace("_HZ","_HH")  ]['xsec'] )+","
            print "        'BR'      : "+ "0.5824*0.69911"  +","
        elif "_ZZ" in d:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0].replace("_ZZ","_HH")  ]['xsec'] )+","
            print "        'BR'      : "+"0.69911*0.69911"+","
        else:
            print "        'xsec'    : "+str( sample[ samples[d]['files'][0]+"_HH" ]['xsec'] / (0.5824*0.5824))+","#remove HH BR
            print "        'BR'      : "+"0.5824*0.5824"+","
        print "        'matcheff': 1.,"
        print "        'kfactor' : 1.,"
        print "    },"
        print "\n"

def debug(label,do_eta=False):

    OUT_pho = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_ZtoLLPho_v5_ntuples_updated/smearing/"
    OUT_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_E_v5_ntuples_reweighted/weighted/smearing/"

    sf_pho_file = TFile(OUT_pho+"data_MC_SF"+label+"_1bin.root","READ")     
    sf_pho_file.cd()     
    sf_pho_1ns = sf_pho_file.Get("ratio_1ns")     
    sf_pho_2ns = sf_pho_file.Get("ratio_2ns")     
    sf_pho_1ns.SetDirectory(0)     
    sf_pho_2ns.SetDirectory(0)     
    sf_pho_file.Close()     
    sf_ele_file = TFile(OUT_ele+"data_MC_SF"+label+"_1bin.root","READ")     
    sf_ele_file.cd()     
    sf_ele_1ns = sf_ele_file.Get("ratio_1ns")     
    sf_ele_2ns = sf_ele_file.Get("ratio_2ns")     
    sf_ele_1ns.SetDirectory(0)     
    sf_ele_2ns.SetDirectory(0)     
    sf_ele_file.Close()

    #Get the highest correction among 1 or 2 ns shift
    sf_pho = sf_pho_1ns.GetBinContent(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinContent(1)
    sf_unc_pho = sf_pho_1ns.GetBinError(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinError(1)
    sf_pho_up = sf_pho+sf_unc_pho
    sf_pho_down = sf_pho-sf_unc_pho

    #Get the highest correction among 1 or 2 ns shift
    sf_ele = sf_ele_1ns.GetBinContent(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinContent(1)
    sf_unc_ele = sf_ele_1ns.GetBinError(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinError(1)
    sf_ele_up = sf_ele+sf_unc_ele
    sf_ele_down = sf_ele-sf_unc_ele

    dnn_threshold = 0.996
    list_of_variables = [ "EventNumber","RunNumber","LumiNumber",
                          "dt_ecal_dist","min_dPhi_jets_*",
                          "JetsNegative.pt","JetsNegative.eta","JetsNegative.phi","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","PUReWeight","TriggerWeight","JetsNegative.timeRecHitsEB","JetsNegative.nRecHitsEB","isSR","isWtoMN","isZtoMM","isMC",
                          "Jets.pt","Jets.eta","Jets.phi","Jets.timeRecHitsEB","Jets.nRecHitsEB","Jets.sigprob"
                      ]

    for d in data:
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]

        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']

        #Here: tree weight
        xs = 0.33918976
        tree_weight = LUMI * xs / gen_events
        print "tree weight: ", tree_weight

        gen = uproot.iterate(new_list,"tree",list_of_variables,100)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        eta = np.array([])
        nRecHitsEB = np.array([])
        timeRecHitsEB = np.array([])
        timeRMSRecHitsEB = np.array([])
        bin2 = np.array([])
        bin2_up_ele = np.array([])
        bin2_up_pho = np.array([])
        bin2_down_ele = np.array([])
        bin2_down_pho = np.array([])

        for arrays in gen:
            print "~~~~~~~~~~~"
            ################################################################
            #Study effect of vetos
            cut_mask = arrays["isSR"]>0
            ### Cosmics
            cosmic_veto = arrays["dt_ecal_dist"]<0.5
            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)

            print "check qcd: ", (cut_mask).sum()

            '''
            ###remove me:
            cut_jets = arrays["JetsNegative.pt"]>-999
            cut_jets = np.logical_and(cut_mask,cut_jets)
            cut_mask = (cut_jets.any()==True)
            print "check entries valid jet negative: ", (cut_mask).sum()
            MINPHI = 3.5
            MAXPHI = 2.7
            cut_mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
            cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
            cut_mask_phi_eta = np.logical_and(cut_mask,np.logical_and(cut_mask_phi,cut_mask_eta))
            cut_mask_phi_eta = np.logical_and(cut_mask,cut_mask_phi_eta)
            cut_mask = (cut_mask_phi_eta.any()==True)
            cut_jets = np.logical_and(cut_jets,cut_mask_phi_eta)
            print "check entries valid jet negative eta/phi cut: ", (cut_mask).sum()

            cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
            cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
            cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)
            cut_mask = np.logical_and(cut_mask,cut_mask_bh)
            cut_jets = np.logical_and(cut_jets,cut_mask)

            print "check entries beam halo: ", (cut_mask).sum()
            continue
            #####
            '''

            mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
            mask_time = arrays["Jets.timeRecHitsEB"]>-1.
            mask_eta_time = np.logical_and(mask_eta,mask_time)

            if ERA=="2017":
                MINPHI = 3.5
                MAXPHI = 2.7
                mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                cut_jets = np.logical_and(mask_eta_time,mask_phi)

            if ERA=="2018":
                MINPHI = 0.9
                MAXPHI = 0.4
                mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                cut_jets = np.logical_and(mask_eta_time,mask_phi)

            if ERA=="2016":
                #There are no noisy regions in 2016, no need to cut on phi
                cut_jets = mask_eta_time

            cut_mask = np.logical_and(cut_mask,(cut_jets.any()==True))
            print "check entries eta/phi/time: ", cut_mask.sum()
            
            cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
            cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
            cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)

            cut_mask = np.logical_and(cut_mask,cut_mask_bh)
            cut_jets = np.logical_and(cut_jets,cut_mask)
            print "check entries beam halo: ", cut_mask.sum()

            eventweight = arrays["EventWeight"][cut_mask]
            pureweight = arrays["PUReWeight"][cut_mask]
            triggerweight = arrays["TriggerWeight"][cut_mask]
            
            #Total weight: multiply all the previous
            weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight*0.001*0.9
            sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
            pt      = arrays["Jets.pt"][cut_jets][cut_mask]

            tag_mask = sigprob > 0.996
            bin2_m = (sigprob[tag_mask].counts>1)


            #Select the jets in bin 2
            sigprob_bin2 = sigprob[bin2_m]

            #Store the pt of the jets in bin 2
            pt_bin2 = pt[bin2_m]
            #Store only the tagged jets in bin 2 (the only ones we must correct)
            pt_tag_bin2 = pt_bin2[sigprob_bin2>dnn_threshold]

            #Create a mutually exclusive categorization
            pt_ele_mask_bin2 = (pt_tag_bin2 > 70)
            pt_pho_mask_bin2 = (pt_tag_bin2 <= 70)

            #Calculate the eventweight due to DNN corrections
            dnnweight = (sf_ele*pt_ele_mask_bin2 + sf_pho*pt_pho_mask_bin2).prod()

            #Scale the DNN corrections up and down for the uncertainties
            dnnweight_up_ele   = (sf_ele_up*pt_ele_mask_bin2 + sf_pho*pt_pho_mask_bin2).prod()
            dnnweight_up_pho   = (sf_ele*pt_ele_mask_bin2 + sf_pho_up*pt_pho_mask_bin2).prod()
            dnnweight_down_ele = (sf_ele_down*pt_ele_mask_bin2 + sf_pho*pt_pho_mask_bin2).prod()
            dnnweight_down_pho = (sf_ele*pt_ele_mask_bin2 + sf_pho_down*pt_pho_mask_bin2).prod()

            #Calculate bin2 yield, and bin2 with DNN shifts up and down for the DNN uncertainty
            #Keep in mind: now weight and dnn_weight have different size, so apply the bin2_m accordingly
            print "bin2 in this chunk: ", np.multiply(bin2_m[bin2_m], weight[bin2_m]  ).sum()
            #with dnn weight
            #bin2 = np.concatenate( (bin2, np.multiply(bin2_m[bin2_m], np.multiply(weight[bin2_m],dnnweight)  )  ) )
            bin2 = np.concatenate( (bin2, np.multiply(bin2_m[bin2_m], weight[bin2_m]  )  ) )


            #print "debug, only entries wo weight"
            #bin2 = np.concatenate( (bin2, np.ones(weight.shape[0]) ) )

        print "************************************"

        #Now we have bin2 and up/down shifts due to DNN
        #Calculate effect of shifts
        ele_shift_up = abs(bin2_up_ele.sum()-bin2.sum())/bin2.sum() if bin2.sum()!=0 else 0
        ele_shift_down = abs(bin2_down_ele.sum()-bin2.sum())/bin2.sum() if bin2.sum()!=0 else 0
        pho_shift_up = abs(bin2_up_pho.sum()-bin2.sum())/bin2.sum() if bin2.sum()!=0 else 0
        pho_shift_down = abs(bin2_down_pho.sum()-bin2.sum())/bin2.sum() if bin2.sum()!=0 else 0
        #Sum in quadrature the effects of up/down shifts
        #ele and pho are independent measurement
        shift_up = math.sqrt( ele_shift_up**2 + pho_shift_up**2  )
        shift_down = math.sqrt( ele_shift_down**2 + pho_shift_down**2  )

        # ******** #
        #Final numbers I need
        bin2_yield = bin2.sum() 
        bin2_stat_unc = np.sqrt( sum(x*x for x in bin2) ).sum()
        #The final DNN uncertainty will be the max between the up and down shifts
        #I prefer to store percentage uncertainty
        dnn_unc = (100*max(shift_up,shift_down))

        print "bin2: ", bin2_yield, " +- ", bin2_stat_unc/bin2_yield


def count(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             #"nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced","dt_ecal_acc_dist",
                             #,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",#"PUReweight",
                             "TriggerWeight","Jets.phi",
                             "JetsNegative.pt","JetsNegative.eta","JetsNegative.sigprob",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "Z_pt",
                             "pt",
                             "min_dPhi_jets_*",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        eta = np.array([])
        nRecHitsEB = np.array([])
        timeRecHitsEB = np.array([])
        timeRMSRecHitsEB = np.array([])
        pr = np.array([])
        b2_pre = np.array([])
        b2_post = np.array([])
        b0_pre  = np.array([])
        b1_pre  = np.array([])
        b1_post = np.array([])

        Z_pt = np.array([])
        DT_x = np.array([])
        DT_y = np.array([])
        DT_z = np.array([])
        CSC_x = np.array([])
        CSC_y = np.array([])
        CSC_z = np.array([])
        ECAL_x = np.array([])
        ECAL_y = np.array([])
        ECAL_z = np.array([])
        ECAL_tag_x  = np.array([])
        ECAL_tag_y  = np.array([])
        ECAL_tag_z  = np.array([])
        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            cut_mask = arrays["isSR"]>0
            cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
            ### Cosmics
            ##cosmic_veto = np.logical_and(arrays["dt_ecal_dist"]<0.5,arrays["dt_fit_chi2_reduced"]<0.15)            
            cosmic_veto = arrays["dt_ecal_dist"]<0.5
            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
            ### Very old BH
            ##cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            ##cut_mask_x = np.logical_and(cut_mask_v,arrays["JetsCaloAdd.spreadXEB"]<1)
            ##cut_mask_additional = (cut_mask_x.any()==True) 
            ##cut_mask_v = np.logical_and(cut_mask_v,cut_mask_additional)
            ##cut_mask_v = arrays["min_dPhi_jets"]<0.05
            ##cut_mask = np.logical_not(cut_mask_v)
            pr = np.concatenate((pr, np.multiply(cut_mask[cut_mask],1.) ))

            #Default cut_jets, does nothing basically
            cut_jets = arrays["Jets.pt"]>-999
            cut_jets = np.logical_and(cut_mask,cut_jets)
            cut_mask = (cut_jets.any()==True)

            ##print "inspect this event: 7192052"
            ##cut_mask = np.logical_and(cut_mask,arrays["EventNumber"]==7192052)

            if do_eta:
                cut_mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                cut_mask = (cut_mask_eta.any()==True)
                cut_jets = np.logical_and(cut_jets,cut_mask_eta)

                #print "passing eta"
                #print arrays["Jets.eta"][cut_jets][cut_mask]
                #print cut_mask[cut_mask].shape
            
            #This selects events with a jet at least --> we care about bin 0 or 1
            sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
            tag_pre_mask = np.logical_and(sigprob > 0.996,sigprob<=1.)
            tag_b0_mask = (sigprob > 0)
            bin2_pre_m = (sigprob[tag_pre_mask].counts>1)
            bin1_pre_m = (sigprob[tag_pre_mask].counts==1)
            bin0_pre_m = (sigprob[tag_b0_mask].counts>0)
            #deb#
            #print "passing bin 2 before BH veto"
            #print bin2_m[bin2_m].shape
            b2_pre = np.concatenate((b2_pre, np.multiply(bin2_pre_m,1.) ))
            b1_pre = np.concatenate((b1_pre, np.multiply(bin1_pre_m,1.) ))
            b0_pre = np.concatenate((b0_pre, np.multiply(bin0_pre_m,1.) ))
            
            #print "bin2 pre:"
            #print arrays["RunNumber"][cut_mask][bin2_pre_m]
            #print arrays["LumiNumber"][cut_mask][bin2_pre_m]
            #print arrays["EventNumber"][cut_mask][bin2_pre_m]
            #print arrays["dt_ecal_acc_dist"][cut_mask][bin2_pre_m]
            #print arrays["min_dPhi_jets_eta_1p0_0p996"][cut_mask][bin2_pre_m]
            #print arrays["Jets.phi"][cut_jets][cut_mask][bin2_pre_m]
            #print arrays["Jets.timeRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]
            #print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]

            '''
            print arrays["Jets.timeRMSRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]
            print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]
            print "bin1 pre beam halo:"
            print arrays["RunNumber"][cut_mask][bin1_pre_m].shape
            print arrays["RunNumber"][cut_mask][bin1_pre_m]
            print arrays["LumiNumber"][cut_mask][bin1_pre_m]
            print arrays["EventNumber"][cut_mask][bin1_pre_m]
            print arrays["Jets.timeRMSRecHitsEB"][cut_jets][cut_mask][bin1_pre_m]
            print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin1_pre_m]
            '''

            ###probe cosmic
            ###L
            ###cosmic_veto = arrays["dt_ecal_dist"]<0.5
            ###cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            ### Beam halo
            
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]>=0))#any
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]<=3))
            
            #13.01.2022: not good
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]<=10))
            #cut_mask_and = np.logical_and(cut_mask,cut_mask_and)
            #cut_mask = (cut_mask_and.any()==True)
            #cut_jets = np.logical_and(cut_jets,cut_mask_and)
            
            #deb#
            #print "passing timeRMSRecHitsEB"
            #print cut_mask[cut_mask].shape

            #Additional on size?
            #cut_mask_n = arrays["Jets.nRecHitsEB"]>1
            #cut_mask_n = np.logical_and(cut_mask,cut_mask_n)
            #cut_mask = (cut_mask_n.any()==True)
            #cut_jets = np.logical_and(cut_jets,cut_mask_n)
            
            #deb#
            #print "passing nRecHitsEB"
            #print cut_mask[cut_mask].shape

            #Test cosmic
            #cosmic_veto = arrays["dt_ecal_dist"]<0.5
            #cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            

            #old BH cut:
            ##cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            ##cut_mask = np.logical_and(cut_mask,np.logical_not(cut_mask_v))

            #new BH cut min_dPhi_jets_eta_1p0_0p9
            cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            #cut_mask = np.logical_and(cut_mask,np.logical_not(cut_mask_dphi))
            #cut_jets = np.logical_and(cut_jets,cut_mask)
            cut_mask_low_multi_tag = np.logical_and(np.logical_and(arrays["Jets.sigprob"]>0.996,arrays["Jets.sigprob"]<=1.) , arrays["Jets.nRecHitsEB"]<=10)
            cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
            cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)

            cut_mask = np.logical_and(cut_mask,cut_mask_bh)
            
            cut_jets = np.logical_and(cut_jets,cut_mask)
            #This selects events with a jet at least --> we care about bin 0 or 1
            sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]

            tag_mask = np.logical_and(sigprob > 0.996,sigprob<=1.)
            bin2_m = (sigprob[tag_mask].counts>1)
            bin2 = np.multiply(bin2_m,1.)

            b2_post = np.concatenate((b2_post,bin2))

            bin1_m = (sigprob[tag_mask].counts==1)
            bin1 = np.multiply(bin1_m,1.)

            b1_post = np.concatenate((b1_post,bin1))

            #deb#
            #print "passing bin 2 with BH veto"
            #print np.sum(bin2)
            #print bin2_m[bin2_m].shape

            isMC = np.concatenate((isMC,np.array(arrays["isMC"][cut_mask][bin2_m])))
            Z_pt = np.concatenate((Z_pt,np.array(arrays["Z_pt"][cut_mask][bin2_m])))
            RunNumber = np.concatenate((RunNumber,np.array(arrays["RunNumber"][cut_mask][bin2_m])))
            LumiNumber = np.concatenate((LumiNumber,np.array(arrays["LumiNumber"][cut_mask][bin2_m])))
            EventNumber = np.concatenate((EventNumber,np.array(arrays["EventNumber"][cut_mask][bin2_m])))

            #print "bin2 POST beam halo:"
            #print arrays["RunNumber"][cut_mask][bin2_m].shape
            #print arrays["RunNumber"][cut_mask][bin2_m]
            #print arrays["LumiNumber"][cut_mask][bin2_m]
            #print arrays["EventNumber"][cut_mask][bin2_m]
            #print arrays["dt_ecal_acc_dist"][cut_mask][bin2_m]
            #print arrays["min_dPhi_jets_eta_1p0_0p996"][cut_mask][bin2_m]
            #print arrays["Jets.phi"][cut_jets][cut_mask][bin2_m]
            #print arrays["Jets.timeRecHitsEB"][cut_jets][cut_mask][bin2_m]
            #print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin2_m]

            ################################################################
            #Study effect of trigger and MET cuts
            #cut_mask = arrays["isMC"]>0
            #cut_mask = arrays["pt"]>200

            ################################################################
            #Study ZtoMM
            #cut_mask = arrays["isZtoMM"]>0
            #cut_mask = np.logical_and(cut_mask,arrays["Z_pt"]>100)

        print "gen_events: ", gen_events
        print "events passing preselections: ", pr.sum()
        print ("%d (%.2f%s) ") % (pr.sum(), 100.*float(pr.sum()/gen_events),'\%')

        
        print "************************************"
        print "events passing bin 0 before BH: ", b0_pre.sum()
        print ("%d (%.2f%s) ") % (b0_pre.sum(), 100.*float(b0_pre.sum()/gen_events),'\%')
        print "events passing bin 1 before BH: ", b1_pre.sum()
        print ("%d (%.2f%s) ") % (b1_pre.sum(), 100.*float(b1_pre.sum()/gen_events),'\%')
        print "events passing bin 1 after BH: ", b1_post.sum()
        print ("%d (%.2f%s) ") % (b1_post.sum(), 100.*float(b1_post.sum()/gen_events),'\%')
        print "relative loss: "
        print ("%d (%.2f%s) ") % (b1_post.sum()-b1_pre.sum(), 100.*float(abs(b1_post.sum()-b1_pre.sum())/(b1_post.sum()+b1_pre.sum())),'\%')


        print "************************************"
        print "events passing bin 2 before BH: ", b2_pre.sum()
        print ("%d (%.2f%s) ") % (b2_pre.sum(), 100.*float(b2_pre.sum()/gen_events),'\%')
        print "events passing bin 2 after BH: ", b2_post.sum()
        print ("%d (%.2f%s) ") % (b2_post.sum(), 100.*float(b2_post.sum()/gen_events),'\%')
        print "relative loss: "
        print ("%d (%.2f%s) ") % (b2_post.sum()-b2_pre.sum(), 100.*float(abs(b2_post.sum()-b2_pre.sum())/(b2_post.sum()+b2_pre.sum())),'\%')
        
        eff[d]['pr'] = pr.sum()
        eff[d]['b2_pre'] = b2_pre.sum()
        eff[d]['b2_post'] = b2_post.sum()
        eff[d]['b1_pre'] = b1_pre.sum()
        eff[d]['b1_post'] = b1_post.sum()

        ##print "events in bin 2: ", np.sum(b2)
        ##print ("%d (%.2f%s) ") % (np.sum(b2), 100.*float(np.sum(b2)/gen_events),'\%')
        ###print Z_pt
        #print np.sort(RunNumber)
        #print LumiNumber
        #print EventNumber

    print eff
    with open(MAIN+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"



def compare_princeton(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["*"]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables,10000)
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        met= np.array([])
        ht = np.array([])
        met_acc= np.array([])
        ht_acc = np.array([])
        #met_tr= np.array([])
        #dijet_tr = np.array([])
        #met_met_tr= np.array([])
        #ht_dijet_tr = np.array([])

        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            met_mask = arrays["pt"]>200
            ht_mask = arrays["HT"]>500
            track_acc = arrays["GenBquarks.travelRadiusLLP"]<10 
            track_mask = (arrays["GenBquarks.travelRadiusLLP"][track_acc].counts)>0
            track_mask = np.logical_and(ht_mask,track_mask)
            calo_acc = np.logical_and( arrays["GenBquarks.travelRadiusLLP"]>30, arrays["GenBquarks.travelRadiusLLP"]<184)
            calo_mask = (arrays["GenBquarks.travelRadiusLLP"][calo_acc].counts)>0
            calo_mask = np.logical_and(met_mask,calo_mask)
            #print arrays["GenBquarks.travelRadiusLLP"]
            #print arrays["GenBquarks.travelRadiusLLP"][track_acc]
            #print arrays["GenBquarks.travelRadiusLLP"][calo_acc]
            #met_tr_mask = np.logical_or(np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v"]>0,arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v"]>0),arrays["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v"]>0)
            #dijet_tr_mask = 
            met = np.concatenate((met, np.multiply(met_mask[met_mask],1.) ))
            ht = np.concatenate((ht, np.multiply(ht_mask[ht_mask],1.) ))
            met_acc = np.concatenate((met_acc, np.multiply(calo_mask[calo_mask],1.) ))
            ht_acc = np.concatenate((ht_acc, np.multiply(track_mask[track_mask],1.) ))


        print "gen_events: ", gen_events
        print "events passing met: ", met.sum()
        print ("%d (%.2f%s) ") % (met.sum(), 100.*float(met.sum()/gen_events),'\%')
        print "events passing ht: ", ht.sum()
        print ("%d (%.2f%s) ") % (ht.sum(), 100.*float(ht.sum()/gen_events),'\%')

        print "events passing met_acc: ", met_acc.sum()
        print ("%d (%.2f%s) ") % (met_acc.sum(), 100.*float(met_acc.sum()/gen_events),'\%')
        print "events passing ht_acc: ", ht_acc.sum()
        print ("%d (%.2f%s) ") % (ht_acc.sum(), 100.*float(ht_acc.sum()/gen_events),'\%')

        eff[d]['met'] = met.sum()
        eff[d]['ht'] = ht.sum()
        eff[d]['met_acc'] = met_acc.sum()
        eff[d]['ht_acc'] = ht_acc.sum()

    print eff
    with open(MAIN+"Dict_compare_princeton"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"


def compare_time_smearing_v2(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["*"]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']

        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)#100000
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])
        #same sequence as Jiajing
        met = np.array([])
        trigger = np.array([])
        lep_veto = np.array([])
        pho_veto = np.array([])
        tau_veto = np.array([])
        d_phi = np.array([])
        met_filter = np.array([])
        cosmic_veto = np.array([])
        bh_veto = np.array([])
        jet_acc = np.array([])
        jet_acc_smear = np.array([])
        bin2 = np.array([])
        bin2_smear = np.array([])
        bin2_smear_isonoise = np.array([])

        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            isMC = arrays["isMC"]

            met_mask = arrays["pt"]>200
            met = np.concatenate(( met, np.multiply(isMC[met_mask],1.) ))

            met_tr_mask = np.logical_or(np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v"]>0,arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v"]>0),arrays["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v"]>0)
            met_tr_mask = np.logical_and(met_mask,met_tr_mask)
            trigger = np.concatenate(( trigger, np.multiply(isMC[met_tr_mask],1.) ))
            
            lep_mask = np.logical_and(arrays["nElectronsPassing"]==0,arrays["nMuonsPassing"]==0)
            lep_mask = np.logical_and(met_tr_mask,lep_mask)
            lep_veto = np.concatenate(( lep_veto, np.multiply(isMC[lep_mask],1.) ))

            pho_mask = arrays["nPhotonsPassing"]==0
            pho_mask = np.logical_and(lep_mask,pho_mask)
            pho_veto = np.concatenate(( pho_veto, np.multiply(isMC[pho_mask],1.) ))

            tau_mask = arrays["nTausPassing"]==0
            tau_mask = np.logical_and(pho_mask,tau_mask)
            tau_veto = np.concatenate(( tau_veto, np.multiply(isMC[tau_mask],1.) ))

            d_phi_mask = arrays["MinJetMetDPhi"]>0.5
            d_phi_mask = np.logical_and(tau_mask,d_phi_mask)
            d_phi = np.concatenate(( d_phi, np.multiply(isMC[d_phi_mask],1.) ))


            met_filter_mask =  arrays["Flag2_globalSuperTightHalo2016Filter"]>0
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_EcalDeadCellTriggerPrimitiveFilter"]>0)
            #met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_HBHENoiseFilter"]>0)
            #met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_HBHEIsoNoiseFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_ecalBadCalibFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_eeBadScFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_BadPFMuonFilter"]>0)
            met_filter_mask = np.logical_and(d_phi_mask,met_filter_mask)
            met_filter = np.concatenate(( met_filter, np.multiply(isMC[met_filter_mask],1.) ))

            #cosmic veto
            cosmic_mask = arrays["dt_ecal_dist"]>0.5
            cosmic_mask = np.logical_and(met_filter_mask,cosmic_mask)
            cosmic_veto = np.concatenate(( cosmic_veto, np.multiply(isMC[cosmic_mask],1.) ))

            #beam halo
            mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
            mask_low_multi_tag = np.logical_and(mask_dphi,mask_low_multi_tag)
            bh_mask = np.logical_not(mask_low_multi_tag.any()==True)
            bh_mask = np.logical_and(cosmic_mask,bh_mask)
            bh_veto = np.concatenate(( bh_veto, np.multiply(isMC[bh_mask],1.) ))

            mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
            cut_mask = (mask_eta.any()==True)
            #jet in acceptance, including eta/phi
            if ERA=="2017":
                MINPHI = 3.5
                MAXPHI = 2.7
                mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                mask_phi = np.logical_and(mask_eta,mask_phi)
                cut_mask = (mask_phi.any()==True)
            if ERA=="2018":
                MINPHI = 0.9
                MAXPHI = 0.4
                mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                mask_phi = np.logical_and(mask_eta,mask_phi)
                cut_mask = (mask_phi.any()==True)

            cut_mask = np.logical_and(cut_mask,bh_mask)
            cut_mask_smear = cut_mask
            
            time_mask = arrays["JetsNegative.timeRecHitsHB"]>-1
            cut_mask = np.logical_and(cut_mask, (time_mask.any()==True)  )

            smear_time_mask = arrays["JetsNegative.timeRecHitsEB"]>-1
            cut_mask_smear = np.logical_and(cut_mask_smear, (smear_time_mask.any()==True)  )

            jet_acc = np.concatenate(( jet_acc, np.multiply(isMC[cut_mask],1.) ))
            jet_acc_smear = np.concatenate(( jet_acc_smear, np.multiply(isMC[cut_mask_smear],1.) ))

            dnn_threshold = 0.996

            sigprob = arrays["JetsNegative.pfXWP1000"][time_mask]
            tag_mask = (sigprob > dnn_threshold)
            bin2_m = (sigprob[tag_mask].counts >1)
            bin2_m = np.logical_and(bin2_m,cut_mask)
            bin2 = np.concatenate( (bin2, np.multiply(isMC[bin2_m],1.)  ) ) 

            sigprob_smear = arrays["JetsNegative.sigprob"][smear_time_mask]
            tag_mask_smear = (sigprob_smear > dnn_threshold)
            bin2_m_smear = (sigprob_smear[tag_mask_smear].counts >1)
            bin2_m_smear = np.logical_and(bin2_m_smear,cut_mask_smear)
            bin2_smear = np.concatenate( (bin2_smear, np.multiply(isMC[bin2_m_smear],1.)  ) ) 

            isonoise_met_filter_mask = np.logical_and(bin2_m_smear,arrays["Flag2_HBHENoiseFilter"]>0)
            isonoise_met_filter_mask = np.logical_and(isonoise_met_filter_mask,arrays["Flag2_HBHEIsoNoiseFilter"]>0)
            bin2_smear_isonoise = np.concatenate( (bin2_smear_isonoise, np.multiply(isMC[isonoise_met_filter_mask],1.)  ) ) 

        print "gen_events: ", gen_events
        print "events passing: ", bin2.sum()
        print ("%d (%.2f%s) ") % (bin2.sum(), 100.*float(bin2.sum()/gen_events),'\%')
        print "events passing with smearing: ", bin2_smear.sum()
        print ("%d (%.2f%s) ") % (bin2_smear.sum(), 100.*float(bin2_smear.sum()/gen_events),'\%')


        eff[d]['met'] = met.sum()
        eff[d]['trigger'] = trigger.sum()
        eff[d]['lep_veto'] = lep_veto.sum()
        eff[d]['pho_veto'] = pho_veto.sum()
        eff[d]['tau_veto'] = tau_veto.sum()
        eff[d]['d_phi'] = d_phi.sum()
        eff[d]['met_filter'] = met_filter.sum()
        eff[d]['cosmic_veto'] = cosmic_veto.sum()
        eff[d]['bh_veto'] = bh_veto.sum()
        eff[d]['jet_acc'] = jet_acc.sum()
        eff[d]['jet_acc_smear'] = jet_acc_smear.sum()
        eff[d]['bin2'] = bin2.sum()
        eff[d]['bin2_smear'] = bin2_smear.sum()
        eff[d]['bin2_smear_isonoise'] = bin2_smear_isonoise.sum()

    print eff
    with open(OUT+"Dict_compare_time_smearing_v2"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+OUT+"Dict"+label+".yaml"

def compare_time_smearing(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["*"]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']

        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)#100000
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])
        #same sequence as Jiajing
        met = np.array([])
        trigger = np.array([])
        lep_veto = np.array([])
        pho_veto = np.array([])
        tau_veto = np.array([])
        d_phi = np.array([])
        met_filter = np.array([])
        cosmic_veto = np.array([])
        bh_veto = np.array([])
        jet_acc = np.array([])
        jet_acc_smear = np.array([])
        bin2 = np.array([])
        bin2_smear = np.array([])

        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            isMC = arrays["isMC"]

            met_mask = arrays["pt"]>200
            met = np.concatenate(( met, np.multiply(isMC[met_mask],1.) ))

            met_tr_mask = np.logical_or(np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v"]>0,arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v"]>0),arrays["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v"]>0)
            met_tr_mask = np.logical_and(met_mask,met_tr_mask)
            trigger = np.concatenate(( trigger, np.multiply(isMC[met_tr_mask],1.) ))
            
            lep_mask = np.logical_and(arrays["nElectronsPassing"]==0,arrays["nMuonsPassing"]==0)
            lep_mask = np.logical_and(met_tr_mask,lep_mask)
            lep_veto = np.concatenate(( lep_veto, np.multiply(isMC[lep_mask],1.) ))

            pho_mask = arrays["nPhotonsPassing"]==0
            pho_mask = np.logical_and(lep_mask,pho_mask)
            pho_veto = np.concatenate(( pho_veto, np.multiply(isMC[pho_mask],1.) ))

            tau_mask = arrays["nTausPassing"]==0
            tau_mask = np.logical_and(pho_mask,tau_mask)
            tau_veto = np.concatenate(( tau_veto, np.multiply(isMC[tau_mask],1.) ))

            d_phi_mask = arrays["MinJetMetDPhi"]>0.5
            d_phi_mask = np.logical_and(tau_mask,d_phi_mask)
            d_phi = np.concatenate(( d_phi, np.multiply(isMC[d_phi_mask],1.) ))


            met_filter_mask =  arrays["Flag2_globalSuperTightHalo2016Filter"]>0
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_EcalDeadCellTriggerPrimitiveFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_HBHENoiseFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_HBHEIsoNoiseFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_ecalBadCalibFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_eeBadScFilter"]>0)
            met_filter_mask = np.logical_and(met_filter_mask,arrays["Flag2_BadPFMuonFilter"]>0)
            met_filter_mask = np.logical_and(d_phi_mask,met_filter_mask)
            met_filter = np.concatenate(( met_filter, np.multiply(isMC[met_filter_mask],1.) ))

            #cosmic veto
            cosmic_mask = arrays["dt_ecal_dist"]>0.5
            cosmic_mask = np.logical_and(met_filter_mask,cosmic_mask)
            cosmic_veto = np.concatenate(( cosmic_veto, np.multiply(isMC[cosmic_mask],1.) ))

            #beam halo
            mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
            mask_low_multi_tag = np.logical_and(mask_dphi,mask_low_multi_tag)
            bh_mask = np.logical_not(mask_low_multi_tag.any()==True)
            bh_mask = np.logical_and(cosmic_mask,bh_mask)
            bh_veto = np.concatenate(( bh_veto, np.multiply(isMC[bh_mask],1.) ))

            mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
            cut_mask = (mask_eta.any()==True)
            #jet in acceptance, including eta/phi
            if ERA=="2017":
                MINPHI = 3.5
                MAXPHI = 2.7
                mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                mask_phi = np.logical_and(mask_eta,mask_phi)
                cut_mask = (mask_phi.any()==True)
            if ERA=="2018":
                MINPHI = 0.9
                MAXPHI = 0.4
                mask_phi = np.logical_or(arrays["JetsNegative.phi"]>MINPHI , arrays["JetsNegative.phi"]<MAXPHI)
                mask_phi = np.logical_and(mask_eta,mask_phi)
                cut_mask = (mask_phi.any()==True)

            cut_mask = np.logical_and(cut_mask,bh_mask)
            cut_mask_smear = cut_mask

            print "passing up to beam halo: ", cut_mask_smear.sum()
            
            time_mask = arrays["JetsNegative.timeRecHitsHB"]>-1
            time_mask = np.logical_and(np.logical_and(mask_phi,mask_eta),time_mask)
            cut_mask = np.logical_and(cut_mask, (time_mask.any()==True)  )

            smear_time_mask = arrays["JetsNegative.timeRecHitsEB"]>-1
            smear_time_mask = np.logical_and(np.logical_and(mask_phi,mask_eta),smear_time_mask)
            cut_mask_smear = np.logical_and(cut_mask_smear, (smear_time_mask.any()==True)  )

            jet_acc = np.concatenate(( jet_acc, np.multiply(isMC[cut_mask],1.) ))
            jet_acc_smear = np.concatenate(( jet_acc_smear, np.multiply(isMC[cut_mask_smear],1.) ))

            dnn_threshold = 0.996

            sigprob = arrays["JetsNegative.pfXWP1000"][time_mask][cut_mask]
            tag_mask = (sigprob > dnn_threshold)
            bin2_m = (sigprob[tag_mask].counts >1)
            #bin2_m = np.logical_and(bin2_m,cut_mask)
            bin2 = np.concatenate( (bin2, bin2_m))#np.multiply(isMC[bin2_m],1.)  ) ) 

            eta = arrays["JetsNegative.eta"][smear_time_mask][cut_mask_smear]
            sigprob_smear = arrays["JetsNegative.sigprob"][smear_time_mask][cut_mask_smear]
            timeRecHitsEB = arrays["JetsNegative.timeRecHitsEB"][smear_time_mask][cut_mask_smear]
            print "eta: ", eta
            print "timeRecHitsEB: ", timeRecHitsEB
            print "sigprob: ", sigprob_smear
            tag_mask_smear = (sigprob_smear > dnn_threshold)
            print "passing acceptance: ", cut_mask_smear.sum()
            bin2_m_smear = (sigprob_smear[tag_mask_smear].counts >1)
            #bin2_m_smear = np.logical_and(bin2_m_smear,cut_mask_smear)
            bin2_smear = np.concatenate( (bin2_smear, bin2_m_smear))#np.multiply(isMC[bin2_m_smear],1.)  ) ) 
            print "passing bin2: ", bin2_m_smear.sum()


        print "gen_events: ", gen_events
        print "events passing: ", bin2.sum()
        print ("%d (%.2f%s) ") % (bin2.sum(), 100.*float(bin2.sum()/gen_events),'\%')
        print "events passing with smearing: ", bin2_smear.sum()
        print ("%d (%.2f%s) ") % (bin2_smear.sum(), 100.*float(bin2_smear.sum()/gen_events),'\%')


        eff[d]['met'] = met.sum()
        eff[d]['trigger'] = trigger.sum()
        eff[d]['lep_veto'] = lep_veto.sum()
        eff[d]['pho_veto'] = pho_veto.sum()
        eff[d]['tau_veto'] = tau_veto.sum()
        eff[d]['d_phi'] = d_phi.sum()
        eff[d]['met_filter'] = met_filter.sum()
        eff[d]['cosmic_veto'] = cosmic_veto.sum()
        eff[d]['bh_veto'] = bh_veto.sum()
        eff[d]['jet_acc'] = jet_acc.sum()
        eff[d]['jet_acc_smear'] = jet_acc_smear.sum()
        eff[d]['bin2'] = bin2.sum()
        eff[d]['bin2_smear'] = bin2_smear.sum()

    print eff
    with open(OUT+"Dict_compare_time_smearing"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+OUT+"Dict"+label+".yaml"


def compare_met_filters(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["*"]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']

        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)#100000
        filt1 = np.array([])
        filt2 = np.array([])
        filt3 = np.array([])
        filt4 = np.array([])
        filt5 = np.array([])
        filt6 = np.array([])
        filt7 = np.array([])

        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            isMC = arrays["isMC"]

            mask1 = arrays["Flag2_globalSuperTightHalo2016Filter"]>0
            filt1 = np.concatenate(( filt1, np.multiply(isMC[mask1],1.) ))
            mask2 = arrays["Flag2_EcalDeadCellTriggerPrimitiveFilter"]>0
            filt2 = np.concatenate(( filt2, np.multiply(isMC[mask2],1.) ))
            mask3 = arrays["Flag2_HBHENoiseFilter"]>0
            filt3 = np.concatenate(( filt3, np.multiply(isMC[mask3],1.) ))
            mask4 = arrays["Flag2_HBHEIsoNoiseFilter"]>0
            filt4 = np.concatenate(( filt4, np.multiply(isMC[mask4],1.) ))
            mask5 = arrays["Flag2_ecalBadCalibFilter"]>0
            filt5 = np.concatenate(( filt5, np.multiply(isMC[mask5],1.) ))
            mask6 = arrays["Flag2_eeBadScFilter"]>0
            filt6 = np.concatenate(( filt6, np.multiply(isMC[mask6],1.) ))
            mask7 = arrays["Flag2_BadPFMuonFilter"]>0
            filt7 = np.concatenate(( filt7, np.multiply(isMC[mask7],1.) ))


        print "gen_events: ", gen_events
        print "events passing filter 1: ", filt1.sum()
        print ("%d (%.2f%s) ") % (filt1.sum(), 100.*float(filt1.sum()/gen_events),'\%')
        print "events passing filter 2: ", filt2.sum()
        print ("%d (%.2f%s) ") % (filt2.sum(), 100.*float(filt2.sum()/gen_events),'\%')
        print "events passing filter 3: ", filt3.sum()
        print ("%d (%.2f%s) ") % (filt3.sum(), 100.*float(filt3.sum()/gen_events),'\%')
        print "events passing filter 4: ", filt4.sum()
        print ("%d (%.2f%s) ") % (filt4.sum(), 100.*float(filt4.sum()/gen_events),'\%')
        print "events passing filter 5: ", filt5.sum()
        print ("%d (%.2f%s) ") % (filt5.sum(), 100.*float(filt5.sum()/gen_events),'\%')
        print "events passing filter 6: ", filt6.sum()
        print ("%d (%.2f%s) ") % (filt6.sum(), 100.*float(filt6.sum()/gen_events),'\%')
        print "events passing filter 7: ", filt7.sum()
        print ("%d (%.2f%s) ") % (filt7.sum(), 100.*float(filt7.sum()/gen_events),'\%')


        eff[d]['globalSuperTightHalo2016Filter'] = filt1.sum()#/gen_events
        eff[d]['EcalDeadCellTriggerPrimitiveFilter'] = filt2.sum()#/gen_events
        eff[d]['HBHENoiseFilter'] = filt3.sum()#/gen_events
        eff[d]['HBHEIsoNoiseFilter'] = filt4.sum()#/gen_events
        eff[d]['ecalBadCalibFilter'] = filt5.sum()#/gen_events
        eff[d]['eeBadScFilter'] = filt6.sum()#/gen_events
        eff[d]['BadPFMuonFilter'] = filt7.sum()#/gen_events

    print eff
    with open(OUT+"Dict_compare_met_filters"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+OUT+"Dict"+label+".yaml"


def count_jet_negative(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             #"nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced",
                             #,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",
                             "JetsNegative.pt","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRecHitsEB",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "Z_pt",
                             "pt",
                             "min_dPhi_jets_*",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        eta = np.array([])
        nRecHitsEB = np.array([])
        timeRecHitsEB = np.array([])
        timeRMSRecHitsEB = np.array([])
        pr = np.array([])
        b2_pre = np.array([])
        b2_post = np.array([])
        b1_pre  = np.array([])
        b1_post = np.array([])

        Z_pt = np.array([])
        DT_x = np.array([])
        DT_y = np.array([])
        DT_z = np.array([])
        CSC_x = np.array([])
        CSC_y = np.array([])
        CSC_z = np.array([])
        ECAL_x = np.array([])
        ECAL_y = np.array([])
        ECAL_z = np.array([])
        ECAL_tag_x  = np.array([])
        ECAL_tag_y  = np.array([])
        ECAL_tag_z  = np.array([])
        #print gen
        for arrays in gen:
            ################################################################
            #Study effect of vetos
            cut_mask = arrays["isSR"]>0
            cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
            ### Cosmics
            ##cosmic_veto = np.logical_and(arrays["dt_ecal_dist"]<0.5,arrays["dt_fit_chi2_reduced"]<0.15)            
            '''
            cosmic_veto = arrays["dt_ecal_dist"]<0.5
            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))
            '''
            ### Very old BH
            ##cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            ##cut_mask_x = np.logical_and(cut_mask_v,arrays["JetsCaloAdd.spreadXEB"]<1)
            ##cut_mask_additional = (cut_mask_x.any()==True) 
            ##cut_mask_v = np.logical_and(cut_mask_v,cut_mask_additional)
            ##cut_mask_v = arrays["min_dPhi_jets"]<0.05
            ##cut_mask = np.logical_not(cut_mask_v)
            pr = np.concatenate((pr, np.multiply(cut_mask[cut_mask],1.) ))

            #Default cut_jets, does nothing basically
            cut_jets = arrays["JetsNegative.timeRecHitsEB"]<=-1.
            cut_jets = np.logical_and(cut_mask,cut_jets)
            cut_mask = (cut_jets.any()==True)

            ##print "inspect this event: 7192052"
            ##cut_mask = np.logical_and(cut_mask,arrays["EventNumber"]==7192052)

            if do_eta:
                cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                cut_mask = (cut_mask_eta.any()==True)
                cut_jets = np.logical_and(cut_jets,cut_mask_eta)

                #print "passing eta"
                #print arrays["Jets.eta"][cut_jets][cut_mask]
                #print cut_mask[cut_mask].shape
            
            #This selects events with a jet at least --> we care about bin 0 or 1
            sigprob = arrays["JetsNegative.sigprob"][cut_jets][cut_mask]
            tag_pre_mask = (sigprob > 0.996)
            bin2_pre_m = (sigprob[tag_pre_mask].counts>1)
            bin1_pre_m = (sigprob[tag_pre_mask].counts==1)
            #deb#
            #print "passing bin 2 before BH veto"
            #print bin2_m[bin2_m].shape
            b2_pre = np.concatenate((b2_pre, np.multiply(bin2_pre_m,1.) ))
            b1_pre = np.concatenate((b1_pre, np.multiply(bin1_pre_m,1.) ))
            
            print "bin2 pre beam halo:"
            print arrays["RunNumber"][cut_mask][bin2_pre_m]
            print arrays["LumiNumber"][cut_mask][bin2_pre_m]
            print arrays["EventNumber"][cut_mask][bin2_pre_m]
            '''
            print arrays["Jets.timeRMSRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]
            print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin2_pre_m]

            print "bin1 pre beam halo:"
            print arrays["RunNumber"][cut_mask][bin1_pre_m].shape
            print arrays["RunNumber"][cut_mask][bin1_pre_m]
            print arrays["LumiNumber"][cut_mask][bin1_pre_m]
            print arrays["EventNumber"][cut_mask][bin1_pre_m]
            print arrays["Jets.timeRMSRecHitsEB"][cut_jets][cut_mask][bin1_pre_m]
            print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin1_pre_m]
            '''

            ###probe cosmic
            ###L
            ###cosmic_veto = arrays["dt_ecal_dist"]<0.5
            ###cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            ### Beam halo
            
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]>=0))#any
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]<=3))
            
            #13.01.2022: not good
            #cut_mask_and = np.logical_not(np.logical_and(arrays["Jets.timeRMSRecHitsEB"]>=8 , arrays["Jets.nRecHitsEB"]<=10))
            #cut_mask_and = np.logical_and(cut_mask,cut_mask_and)
            #cut_mask = (cut_mask_and.any()==True)
            #cut_jets = np.logical_and(cut_jets,cut_mask_and)
            
            #deb#
            #print "passing timeRMSRecHitsEB"
            #print cut_mask[cut_mask].shape

            #Additional on size?
            #cut_mask_n = arrays["Jets.nRecHitsEB"]>1
            #cut_mask_n = np.logical_and(cut_mask,cut_mask_n)
            #cut_mask = (cut_mask_n.any()==True)
            #cut_jets = np.logical_and(cut_jets,cut_mask_n)
            
            #deb#
            #print "passing nRecHitsEB"
            #print cut_mask[cut_mask].shape
            

            #old BH cut:
            ##cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            ##cut_mask = np.logical_and(cut_mask,np.logical_not(cut_mask_v))

            #new BH cut min_dPhi_jets_eta_1p0_0p9
            '''
            cut_mask_dphi = arrays["min_dPhi_jets_eta_1p0_0p996"]<0.05
            #cut_mask = np.logical_and(cut_mask,np.logical_not(cut_mask_dphi))
            #cut_jets = np.logical_and(cut_jets,cut_mask)

            cut_mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
            cut_mask_low_multi_tag = np.logical_and(cut_mask_dphi,cut_mask_low_multi_tag)
            cut_mask_bh = np.logical_not(cut_mask_low_multi_tag.any()==True)
            cut_mask = np.logical_and(cut_mask,cut_mask_bh)
            cut_jets = np.logical_and(cut_jets,cut_mask)
            '''
            #This selects events with a jet at least --> we care about bin 0 or 1
            sigprob = arrays["JetsNegative.sigprob"][cut_jets][cut_mask]

            tag_mask = (sigprob > 0.996)
            bin2_m = (sigprob[tag_mask].counts>1)
            bin2 = np.multiply(bin2_m,1.)

            b2_post = np.concatenate((b2_post,bin2))

            bin1_m = (sigprob[tag_mask].counts==1)
            bin1 = np.multiply(bin1_m,1.)

            b1_post = np.concatenate((b1_post,bin1))
            #deb#
            #print "passing bin 2 with BH veto"
            #print np.sum(bin2)
            #print bin2_m[bin2_m].shape

            isMC = np.concatenate((isMC,np.array(arrays["isMC"][cut_mask][bin2_m])))
            Z_pt = np.concatenate((Z_pt,np.array(arrays["Z_pt"][cut_mask][bin2_m])))
            RunNumber = np.concatenate((RunNumber,np.array(arrays["RunNumber"][cut_mask][bin2_m])))
            LumiNumber = np.concatenate((LumiNumber,np.array(arrays["LumiNumber"][cut_mask][bin2_m])))
            EventNumber = np.concatenate((EventNumber,np.array(arrays["EventNumber"][cut_mask][bin2_m])))

            '''
            print "bin1 POST beam halo:"
            print arrays["RunNumber"][cut_mask][bin1_m].shape
            print arrays["RunNumber"][cut_mask][bin1_m]
            print arrays["LumiNumber"][cut_mask][bin1_m]
            print arrays["EventNumber"][cut_mask][bin1_m]
            print arrays["Jets.timeRMSRecHitsEB"][cut_jets][cut_mask][bin1_m]
            print arrays["Jets.nRecHitsEB"][cut_jets][cut_mask][bin1_m]
            '''

            ################################################################
            #Study effect of trigger and MET cuts
            #cut_mask = arrays["isMC"]>0
            #cut_mask = arrays["pt"]>200

            ################################################################
            #Study ZtoMM
            #cut_mask = arrays["isZtoMM"]>0
            #cut_mask = np.logical_and(cut_mask,arrays["Z_pt"]>100)

        print "gen_events: ", gen_events
        print "events passing preselections: ", pr.sum()
        print ("%d (%.2f%s) ") % (pr.sum(), 100.*float(pr.sum()/gen_events),'\%')

        
        print "************************************"
        print "events passing bin 1 before BH: ", b1_pre.sum()
        print ("%d (%.2f%s) ") % (b1_pre.sum(), 100.*float(b1_pre.sum()/gen_events),'\%')
        print "events passing bin 1 after BH: ", b1_post.sum()
        print ("%d (%.2f%s) ") % (b1_post.sum(), 100.*float(b1_post.sum()/gen_events),'\%')
        print "relative loss: "
        print ("%d (%.2f%s) ") % (b1_post.sum()-b1_pre.sum(), 100.*float(abs(b1_post.sum()-b1_pre.sum())/(b1_post.sum()+b1_pre.sum())),'\%')


        print "************************************"
        print "events passing bin 2 before BH: ", b2_pre.sum()
        print ("%d (%.2f%s) ") % (b2_pre.sum(), 100.*float(b2_pre.sum()/gen_events),'\%')
        print "events passing bin 2 after BH: ", b2_post.sum()
        print ("%d (%.2f%s) ") % (b2_post.sum(), 100.*float(b2_post.sum()/gen_events),'\%')
        print "relative loss: "
        print ("%d (%.2f%s) ") % (b2_post.sum()-b2_pre.sum(), 100.*float(abs(b2_post.sum()-b2_pre.sum())/(b2_post.sum()+b2_pre.sum())),'\%')
        
        eff[d]['pr'] = pr.sum()
        eff[d]['b2_pre'] = b2_pre.sum()
        eff[d]['b2_post'] = b2_post.sum()
        eff[d]['b1_pre'] = b1_pre.sum()
        eff[d]['b1_post'] = b1_post.sum()

        ##print "events in bin 2: ", np.sum(b2)
        ##print ("%d (%.2f%s) ") % (np.sum(b2), 100.*float(np.sum(b2)/gen_events),'\%')
        ###print Z_pt
        #print np.sort(RunNumber)
        #print LumiNumber
        #print EventNumber

    '''
    print eff
    with open(MAIN+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"
    '''



def smear_correlation(label,do_eta=False):

    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced",
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",
                             "JetsNegative.pt","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRecHitsEB","JetsNegative.timeRecHitsHB","JetsNegative.pfXWP1000","JetsNegative.nTrackConstituents",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "pt",
                             "min_dPhi_jets_*",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list




        gen = uproot.iterate(new_list,"tree",list_of_variables)

        pr = np.array([])
        b2_post = np.array([])
        b1_post = np.array([])
        b0_post = np.array([])


        for arrays in gen:
            ################################################################
            cut_mask = arrays["isSR"]>0

            cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)

            #cosmic_veto = arrays["dt_ecal_dist"]<0.5
            #cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))


            ### Very old BH
            ##cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            ##cut_mask_x = np.logical_and(cut_mask_v,arrays["JetsCaloAdd.spreadXEB"]<1)
            ##cut_mask_additional = (cut_mask_x.any()==True) 
            ##cut_mask_v = np.logical_and(cut_mask_v,cut_mask_additional)
            ##cut_mask_v = arrays["min_dPhi_jets"]<0.05
            ##cut_mask = np.logical_not(cut_mask_v)
            pr = np.concatenate((pr, np.multiply(cut_mask[cut_mask],1.) ))

            #Default cut_jets, does nothing basically

            #Unsmeared!
            #cut_jets = arrays["JetsNegative.timeRecHitsHB"]>=-1.

            #Smeared!
            cut_jets = arrays["JetsNegative.timeRecHitsEB"]>=-1.

            cut_jets = np.logical_and(cut_mask,cut_jets)
            cut_mask = (cut_jets.any()==True)

            if do_eta:
                cut_mask_eta = np.logical_and(arrays["JetsNegative.eta"]>-1. , arrays["JetsNegative.eta"]<1.)
                cut_mask_eta = np.logical_and(cut_mask,cut_mask_eta)
                cut_mask = (cut_mask_eta.any()==True)
                cut_jets = np.logical_and(cut_jets,cut_mask_eta)


            #HERE beam halo missing

            #Unsmeared!
            #sigprob = arrays["JetsNegative.pfXWP1000"][cut_jets][cut_mask]

            #Smeared!
            sigprob = arrays["JetsNegative.sigprob"][cut_jets][cut_mask]

            print sigprob

            tag_mask = (sigprob > 0.996)
            print tag_mask[tag_mask]

            bin2_m = (sigprob[tag_mask].counts>1)
            bin2 = np.multiply(bin2_m,1.)
            b2_post = np.concatenate((b2_post,bin2))

            bin1_m = (sigprob[tag_mask].counts==1)
            bin1 = np.multiply(bin1_m,1.)
            b1_post = np.concatenate((b1_post,bin1))

            bin0_m = (sigprob[tag_mask].counts==0)
            bin0 = np.multiply(bin0_m,1.)
            b0_post = np.concatenate((b0_post,bin0))


        print "gen_events: ", gen_events
        print "events passing preselections: ", pr.sum()
        print ("%d (%.2f%s) ") % (pr.sum(), 100.*float(pr.sum()/gen_events),'\%')

        
        print "************************************"
        print "events passing bin 0: ", b0_post.sum()
        print ("%d (%.2f%s) ") % (b0_post.sum(), 100.*float(b0_post.sum()/gen_events),'\%')

        print "************************************"
        print "events passing bin 1: ", b1_post.sum()
        print ("%d (%.2f%s) ") % (b1_post.sum(), 100.*float(b1_post.sum()/gen_events),'\%')

        print "************************************"
        print "events passing bin 2: ", b2_post.sum()
        print ("%d (%.2f%s) ") % (b2_post.sum(), 100.*float(b2_post.sum()/gen_events),'\%')


        print "************************************"
        print "bin 0-1-2 sum: ", b2_post.sum()+b1_post.sum()+b0_post.sum()


        #Load data-MC SB for time smearing
        smearing_file = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+str(ERA)+"_TtoEM_v5_ntuples_validate_timeRecHits/data_smear_file_CSV_0p8_all_jets.root"
        timeCBFile = TFile(smearing_file,"READ")
        dataCB = timeCBFile.Get("data_CB");
        mcCB = timeCBFile.Get("back_CB");
        smearCB = dataCB.Clone("smear_cb");
        smearCB.SetParameter(0,dataCB.GetParameter(0));
        smearCB.SetParameter(1,dataCB.GetParameter(1) - mcCB.GetParameter(1));
        smearCB.SetParameter(2, math.sqrt( abs( (dataCB.GetParameter(2)**2) - (mcCB.GetParameter(2)**2) )) );
        smearCB.SetParameter(3,dataCB.GetParameter(3));
        smearCB.SetParameter(4,dataCB.GetParameter(4));

        #Generate sufficient number of smearing numbers before the loop
        global_smear_vec = []
        jet_smear_vec = []
        for g in range(gen_events*10):
            if g<gen_events:
                global_smear_vec.append(smearCB.GetRandom()) 
            else:
                jet_smear_vec.append(smearCB.GetRandom())

        '''
        outfile = TFile(OUT+"smearing_factors.root","RECREATE")
        outfile.cd()
        v_g = ROOT.std.vector('float')()
        v_j = ROOT.std.vector('float')()
        t = TTree('tree','tree')
        t.Branch('global_smear',v_g)
        t.Branch('jet_smear',v_j)
        for g in global_smear_vec:
            v_g.push_back(g)
        for j in jet_smear_vec:
            v_j.push_back(j)
        t.Fill()
        t.Write()
        outfile.Close()
        print "Please check ", OUT+"smearing_factors.root"
        '''

        #Load tensorflow model
        model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')


        #Define variables
        v_n_jet_acceptance = []
        v_nTrackConstituents = []
        v_nSelectedTracks = []
        v_timeRecHitsHB = []
        v_pre_sigprob = []
        v_timeSmearedCorr = []
        v_timeSmeared = []
        v_eta = []
        v_pt = []
        v_eFracRecHitsEB = []
        v_nRecHitsEB = []
        v_sig1EB = []
        v_sig2EB = []
        v_ptDEB = []
        v_cHadEFrac = []
        v_nHadEFrac = []
        v_eleEFrac = []
        v_photonEFrac = []
        v_ptAllTracks = []
        v_ptAllPVTracks = []
        v_alphaMax = []
        v_betaMax = []
        v_gammaMax = []
        v_gammaMaxEM = []
        v_gammaMaxHadronic = []
        v_gammaMaxET = []
        v_minDeltaRAllTracks = []
        v_minDeltaRPVTracks = []
        v_EventWeight = []

        nTag = []
        nTagSmeared = []
        nTagSmearedCorr = []

        #need some zero padding to be able to count stuff again..

        
        #Do the time smearing
        #uproot is a nightmare, do it in slow old fashioned python
        tmp_file = TFile(MAIN+l+".root","READ")
        tree = tmp_file.Get("tree")

        glob_counter = 0 
        jet_counter = gen_events

        for e in range(0,tree.GetEntries()):
            tree.GetEntry(e)
            #print "----------------------------"
            neg_jets = getattr(tree,"JetsNegative")
            met = getattr(tree,"MEt")

            if e%100 == 0:
                print "Event n. ", e, "/", tree.GetEntries()

            #print "glob_counter ", glob_counter
            
            if tree.MinJetMetDPhi<=0.5:
                continue

            tmp_n_tag = 0
            tmp_n_tag_smeared = 0
            tmp_n_tag_smeared_corr = 0

            for n in range(neg_jets.size()):
                if neg_jets[n].eta<=-1:
                    continue
                if neg_jets[n].eta>=1:
                    continue

                #if ERA=="2017":
                #    if neg_jets[n].phi>=2.7: continue
                #if ERA=="2018":
                #    if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue
                
                jet_counter+=1

                v_tmp_n_jet_acceptance = []
                v_tmp_nTrackConstituents = []
                v_tmp_nSelectedTracks = []
                v_tmp_timeRecHitsHB = []
                v_tmp_pre_sigprob = []
                v_tmp_timeSmearedCorr = []
                v_tmp_timeSmeared = []
                v_tmp_eta = []
                v_tmp_pt = []
                v_tmp_eFracRecHitsEB = []
                v_tmp_nRecHitsEB = []
                v_tmp_sig1EB = []
                v_tmp_sig2EB = []
                v_tmp_ptDEB = []
                v_tmp_cHadEFrac = []
                v_tmp_nHadEFrac = []
                v_tmp_eleEFrac = []
                v_tmp_photonEFrac = []
                v_tmp_ptAllTracks = []
                v_tmp_ptAllPVTracks = []
                v_tmp_alphaMax = []
                v_tmp_betaMax = []
                v_tmp_gammaMax = []
                v_tmp_gammaMaxEM = []
                v_tmp_gammaMaxHadronic = []
                v_tmp_gammaMaxET = []
                v_tmp_minDeltaRAllTracks = []
                v_tmp_minDeltaRPVTracks = []
                v_tmp_EventWeight = []
                
                print "Only for unsmeared ntuples!!! Use EB!!!"
                time_rnd_smear_corr = neg_jets[n].timeRecHitsEB + global_smear_vec[glob_counter]
                time_rnd_smear = neg_jets[n].timeRecHitsEB + jet_smear_vec[jet_counter]

                #print "global/jet smearer ", global_smear_vec[glob_counter], " / " , jet_smear_vec[jet_counter]
                #print "time_rnd_corr ", time_rnd_smear_corr
                #print "time_rnd_uncorr ", time_rnd_smear

                v_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                v_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)

                print "Only for unsmeared ntuples!!! Use EB!!!"
                v_tmp_timeRecHitsHB.append(neg_jets[n].timeRecHitsEB)
                v_tmp_timeSmearedCorr.append(time_rnd_smear_corr)
                v_tmp_timeSmeared.append(time_rnd_smear)

                v_tmp_pre_sigprob.append(neg_jets[n].pfXWP1000)

                v_tmp_eta.append(neg_jets[n].eta)
                v_tmp_pt.append(neg_jets[n].pt)
                v_tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                v_tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                v_tmp_sig1EB.append(neg_jets[n].sig1EB)
                v_tmp_sig2EB.append(neg_jets[n].sig2EB)
                v_tmp_ptDEB.append(neg_jets[n].ptDEB)
                v_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                v_tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                v_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                v_tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                v_tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                v_tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                v_tmp_alphaMax.append(neg_jets[n].alphaMax)
                v_tmp_betaMax.append(neg_jets[n].betaMax)
                v_tmp_gammaMax.append(neg_jets[n].gammaMax)
                v_tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                v_tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                v_tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                v_tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                v_tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)

                #Here define v_tmp_tmp_ ...

                v_tmp_dat_list = [
                    v_tmp_nTrackConstituents,
                    v_tmp_nSelectedTracks,
                    v_tmp_timeRecHitsHB,
                    v_tmp_eFracRecHitsEB,
                    v_tmp_nRecHitsEB,
                    v_tmp_sig1EB,
                    v_tmp_sig2EB,
                    v_tmp_ptDEB,
                    v_tmp_cHadEFrac,
                    v_tmp_nHadEFrac,
                    v_tmp_eleEFrac,
                    v_tmp_photonEFrac,
                    v_tmp_ptAllTracks,
                    v_tmp_ptAllPVTracks,
                    v_tmp_alphaMax,
                    v_tmp_betaMax,
                    v_tmp_gammaMax,
                    v_tmp_gammaMaxEM,
                    v_tmp_gammaMaxHadronic,
                    v_tmp_gammaMaxET,
                    v_tmp_minDeltaRAllTracks,
                    v_tmp_minDeltaRPVTracks,
                ]

                v_tmp_dat_list_smeared = [
                    v_tmp_nTrackConstituents,
                    v_tmp_nSelectedTracks,
                    v_tmp_timeSmeared,
                    v_tmp_eFracRecHitsEB,
                    v_tmp_nRecHitsEB,
                    v_tmp_sig1EB,
                    v_tmp_sig2EB,
                    v_tmp_ptDEB,
                    v_tmp_cHadEFrac,
                    v_tmp_nHadEFrac,
                    v_tmp_eleEFrac,
                    v_tmp_photonEFrac,
                    v_tmp_ptAllTracks,
                    v_tmp_ptAllPVTracks,
                    v_tmp_alphaMax,
                    v_tmp_betaMax,
                    v_tmp_gammaMax,
                    v_tmp_gammaMaxEM,
                    v_tmp_gammaMaxHadronic,
                    v_tmp_gammaMaxET,
                    v_tmp_minDeltaRAllTracks,
                    v_tmp_minDeltaRPVTracks,
                ]

                v_tmp_dat_list_smeared_correlated = [
                    v_tmp_nTrackConstituents,
                    v_tmp_nSelectedTracks,
                    v_tmp_timeSmearedCorr,
                    v_tmp_eFracRecHitsEB,
                    v_tmp_nRecHitsEB,
                    v_tmp_sig1EB,
                    v_tmp_sig2EB,
                    v_tmp_ptDEB,
                    v_tmp_cHadEFrac,
                    v_tmp_nHadEFrac,
                    v_tmp_eleEFrac,
                    v_tmp_photonEFrac,
                    v_tmp_ptAllTracks,
                    v_tmp_ptAllPVTracks,
                    v_tmp_alphaMax,
                    v_tmp_betaMax,
                    v_tmp_gammaMax,
                    v_tmp_gammaMaxEM,
                    v_tmp_gammaMaxHadronic,
                    v_tmp_gammaMaxET,
                    v_tmp_minDeltaRAllTracks,
                    v_tmp_minDeltaRPVTracks,
                ]

                v_tmp_X = np.transpose(np.stack((v_tmp_dat_list)))
                v_tmp_X_smear = np.transpose(np.stack((v_tmp_dat_list_smeared)))
                v_tmp_X_smear_corr = np.transpose(np.stack((v_tmp_dat_list_smeared_correlated)))
                
                v_tmp_probs = model.predict(v_tmp_X)
                v_tmp_probs_smear = model.predict(v_tmp_X_smear)
                v_tmp_probs_smear_corr = model.predict(v_tmp_X_smear_corr)

                #print v_tmp_probs_smear[:,1][0]
                #print v_tmp_sigprob

                if v_tmp_timeRecHitsHB[0]>-1 and v_tmp_probs[:,1][0]>0.996:
                    tmp_n_tag += 1
                if time_rnd_smear>-1 and v_tmp_probs_smear[:,1][0]>0.996:
                    tmp_n_tag_smeared+=1
                if time_rnd_smear_corr>-1 and v_tmp_probs_smear_corr[:,1][0]>0.996:
                    tmp_n_tag_smeared_corr+=1

            nTag.append(tmp_n_tag)
            nTagSmeared.append(tmp_n_tag_smeared)
            nTagSmearedCorr.append(tmp_n_tag_smeared_corr)

            glob_counter+=1

        nTag = np.array(nTag)
        nTagSmeared = np.array(nTagSmeared)
        nTagSmearedCorr = np.array(nTagSmearedCorr)

        bin2 = (nTag>1).sum()
        bin1 = (nTag==1).sum()
        bin0 = (nTag==0).sum()

        bin2Smeared = (nTagSmeared>1).sum()
        bin1Smeared = (nTagSmeared==1).sum()
        bin0Smeared = (nTagSmeared==0).sum()

        bin2SmearedCorr = (nTagSmearedCorr>1).sum()
        bin1SmearedCorr = (nTagSmearedCorr==1).sum()
        bin0SmearedCorr = (nTagSmearedCorr==0).sum()

        print bin0, bin1, bin2, bin0 + bin1 + bin2 
        print bin0Smeared, bin1Smeared, bin2Smeared, bin0Smeared + bin1Smeared + bin2Smeared
        print bin0SmearedCorr, bin1SmearedCorr, bin2SmearedCorr, bin0SmearedCorr + bin1SmearedCorr + bin2SmearedCorr

        
        eff[d]['b0'] = bin0
        eff[d]['b1'] = bin1
        eff[d]['b2'] = bin2

        eff[d]['b0Smeared'] = bin0Smeared
        eff[d]['b1Smeared'] = bin1Smeared
        eff[d]['b2Smeared'] = bin2Smeared

        eff[d]['b0SmearedCorr'] = bin0SmearedCorr
        eff[d]['b1SmearedCorr'] = bin1SmearedCorr
        eff[d]['b2SmearedCorr'] = bin2SmearedCorr


    print eff
    with open(OUT+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"



def print_smear_correlation(label):

    with open(OUT+"Dict"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()


    print "Reading the following dictionary: ", OUT+"Dict"+label+".yaml"

    mass = []#np.array([])
    ctau = []#np.array([])

    eff_bin2 = defaultdict(dict)
    eff_bin2_smear = defaultdict(dict)
    eff_bin2_smear_corr = defaultdict(dict)
    ratio = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]



    table = PrettyTable(['m', 'ct','bin 2','bin 2 uncorr.','bin 2 corr.', 'diff. corr-uncorr (\%)', 'diff. corr-unsmeared (\%)', 'diff. uncorr-unsmeared (\%)'])
    #table.add(['','','','','','corr-uncorr (\%)', 'corr-unsmeared (\%)', 'uncorr-unsmeared (\%)'])
    for m in mass:
        row = []
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    row = [m,c, eff[d]['b2'],eff[d]['b2Smeared'],eff[d]['b2SmearedCorr'],   round( 100*(eff[d]['b2Smeared']-eff[d]['b2SmearedCorr'])/ ((eff[d]['b2Smeared'] + eff[d]['b2SmearedCorr'])*0.5) , 2 ), round( 100*(eff[d]['b2Smeared']-eff[d]['b2'])/ ((eff[d]['b2'] + eff[d]['b2Smeared'])*0.5) , 2 ), round( 100*(eff[d]['b2SmearedCorr']-eff[d]['b2'])/ ((eff[d]['b2'] + eff[d]['b2SmearedCorr'])*0.5) , 2 ) ]
                    table.add_row(row)

    print(table)
    exit()

    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & difference corr-uncorr (\%)"+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['b2Smeared']-eff[d]['b2SmearedCorr'])/ eff[d]['b2Smeared'] ) )
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

    '''
    for s in data:
        print "\n"
        print "\\begin{table}[!h]"
        print "\\begin{center}"
        print "\caption{$m_{\\chi}$="+str(samples[s]['mass'])+" (GeV), $c \\tau$ = "+str(samples[s]['ctau']/1000.)+" (m)}"
    
        print "\\begin{tabular}{|c|c|}"
        print "\hline"
        print "MET filter & efficiency (\%)\\\\"
        print "\hline"
        #####print str("%.2f" % ( 100.*eff[s]["bin2_smear"]/eff[s]['gen']))+ " & "+str("%.2f" % ( 100.*eff[s]["bin2_smear_isonoise"]/eff[s]['gen']))+ " & " +  str(  "%.2f" % (100*(eff[s]["bin2_smear_isonoise"]/eff[s]['gen'] - eff[s]["bin2_smear"]/eff[s]['gen']))   )   + " & " + str("%.2f" % ( 100*(eff[s]["bin2_smear_isonoise"] - eff[s]["bin2_smear"])/ (eff[s]["bin2_smear"])))# + eff[s]["bin2_smear_isonoise"])/2. ) ) 
        #for k in eff[s].keys():
        #   #if k!='gen' and k!="c" and k!="m":
        #    #f "bin2_smear" in k:
        #        print k + " & "+str("%.2f" % ( 100*eff[s][k]/eff[s]['gen']))+" \\\\"
        print "\hline"
        print "\end{tabular}"
        print "\end{center}"
        print "\end{table}"
    '''

def draw(label):
    with open(MAIN+"Dict"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()
    #print eff
    #print eff.keys()
    #print data
    mass = []#np.array([])
    ctau = []#np.array([])

    eff_pr_m = defaultdict(dict)
    eff_pr_c = defaultdict(dict)
    ratio = defaultdict(dict)
    eff_b2_pre_m = defaultdict(dict)
    eff_b2_pre_c = defaultdict(dict)
    eff_b2_post_m = defaultdict(dict)
    eff_b2_post_c = defaultdict(dict)
    eff_b1_pre_m = defaultdict(dict)
    eff_b1_pre_c = defaultdict(dict)
    eff_b1_post_m = defaultdict(dict)
    eff_b1_post_c = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    #eff_pr.append(eff[d]['pr'])
    #eff_b2_pre.append(eff[d]['b2_pre'])
    #eff_b2_post.append(eff[d]['b2_post'])

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]
        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    #c1.Divide(1, 2)
    #setTopPad(c1.GetPad(1))
    #setBotPad(c1.GetPad(2))
    #c1.cd(1)
    #c1.GetPad(bool(1)).SetTopMargin(0.06)
    #c1.GetPad(bool(1)).SetRightMargin(0.05)
    #c1.GetPad(bool(1)).SetTicks(1, 1)
    c1.SetTopMargin(0.06)
    c1.SetRightMargin(0.05)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    mg_ratio = TMultiGraph()
    leg1 = TLegend(0.1, 0.68, 0.3, 0.94)

    for c in ctau:
        eff_pr_m[c] = TGraph()
        eff_b2_pre_m[c] = TGraph()
        eff_b2_post_m[c] = TGraph()
        eff_b1_pre_m[c] = TGraph()
        eff_b1_post_m[c] = TGraph()
        ratio[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print m, c, d, eff[d]['pr']
                    print "point n. ", n
                    eff_pr_m[c].SetPoint(n,m,100*float(eff[d]['pr']/eff[d]['gen']))
                    eff_b2_pre_m[c].SetPoint(n,m,100*float(eff[d]['b2_pre']/eff[d]['gen']))
                    eff_b2_post_m[c].SetPoint(n,m,100*float(eff[d]['b2_post']/eff[d]['gen']))
                    eff_b1_pre_m[c].SetPoint(n,m,100*float(eff[d]['b1_pre']/eff[d]['gen']))
                    eff_b1_post_m[c].SetPoint(n,m,100*float(eff[d]['b1_post']/eff[d]['gen']))
                    ratio[c].SetPoint(n,m,float(eff[d]['b2_post']/eff[d]['b2_pre']))
                    n+=1
        
        print c
        print eff_pr_m[c].Print()
        print eff_b2_pre_m[c].Print()
        print eff_b2_post_m[c].Print()
        eff_pr_m[c].SetLineColor(colors[count_c])
        eff_b2_pre_m[c].SetLineColor(colors[count_c])
        eff_b2_post_m[c].SetLineColor(colors[count_c])
        eff_b1_pre_m[c].SetLineColor(colors[count_c])
        eff_b1_post_m[c].SetLineColor(colors[count_c])
        eff_pr_m[c].SetLineWidth(3)
        eff_b2_pre_m[c].SetLineWidth(3)
        eff_b2_post_m[c].SetLineWidth(3)
        eff_b1_pre_m[c].SetLineWidth(1)
        eff_b1_post_m[c].SetLineWidth(1)
        eff_pr_m[c].SetLineStyle(1)
        eff_b2_pre_m[c].SetLineStyle(2)
        eff_b2_post_m[c].SetLineStyle(3)
        eff_b1_pre_m[c].SetLineStyle(2)
        eff_b1_post_m[c].SetLineStyle(2)
        eff_b1_pre_m[c].SetMarkerStyle(21)
        eff_b1_post_m[c].SetMarkerStyle(25)
        eff_b1_pre_m[c].SetMarkerColor(colors[count_c])
        eff_b1_post_m[c].SetMarkerColor(colors[count_c])

        leg1.AddEntry(eff_pr_m[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        eff_b2_pre_m[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        eff_b2_pre_m[c].GetYaxis().SetTitle("Efficiency (%)")
        eff_b2_pre_m[c].GetYaxis().SetTitleSize(0.05)
        eff_b2_pre_m[c].SetMinimum(0)
        eff_b2_pre_m[c].SetMaximum(101)
        #mg.Add(eff_pr_m[c])
        mg.Add(eff_b2_pre_m[c])
        mg.Add(eff_b2_post_m[c])
        #mg.Add(eff_b1_pre_m[c])
        #mg.Add(eff_b1_post_m[c])

        ratio[c].SetMarkerStyle(20)
        ratio[c].SetMarkerColor(colors[count_c])
        ratio[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        ratio[c].GetXaxis().SetTitleOffset(1.2)
        ratio[c].GetXaxis().SetTitleSize(0.15)
        ratio[c].GetXaxis().SetLabelSize(0.15)
        ratio[c].GetYaxis().SetTitle("ratio")
        ratio[c].GetYaxis().SetTitleSize(0.15)
        ratio[c].GetYaxis().SetTitleOffset(1.)
        ratio[c].GetYaxis().SetLabelSize(0.1)
        #if count_c==0:
        #    ratio[c].Draw("AP")
        #else:
        #    ratio[c].Draw("P,sames")
        mg_ratio.Add(ratio[c])
        count_c+=1

    c1.cd(1)
    leg1.AddEntry(eff_pr_m[ctau[0]],"Presel.","L")
    leg1.AddEntry(eff_b2_pre_m[ctau[0]],"bin2","L")
    leg1.AddEntry(eff_b2_post_m[ctau[0]],"bin2 + BH","L")
    #leg1.AddEntry(eff_b1_pre_m[ctau[0]],"bin1","L")
    #leg1.AddEntry(eff_b1_post_m[ctau[0]],"bin1 + BH","L")
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0)
    #mg.SetMaximum(101)
    #c1.SetLogy()
    c1.SetGrid()
    mg.Draw("AL")
    leg1.Draw()
    LUMI = 1.
    drawCMS_simple(LUMI, "Simulation", ERA="2017", onTop=True)
    '''
    c1.cd(2)
    mg_ratio.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg_ratio.GetXaxis().SetTitleOffset(1.2)
    mg_ratio.GetXaxis().SetTitleSize(0.15)
    mg_ratio.GetXaxis().SetLabelSize(0.15)
    mg_ratio.GetYaxis().SetTitle("ratio")
    mg_ratio.GetYaxis().SetTitleSize(0.15)
    mg_ratio.GetYaxis().SetTitleOffset(1.)
    mg_ratio.GetYaxis().SetLabelSize(0.1)
    mg_ratio.Draw("AP")
    '''
    c1.Print(OUT+"SignEff_BeamHalo"+label+".pdf")
    c1.Print(OUT+"SignEff_BeamHalo"+label+".png")
    #c1.Print(OUT+"SignEff_MinDPhi"+label+".pdf")
    #c1.Print(OUT+"SignEff_MinDPhi"+label+".png")
    c1.Clear()


    #Print table
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & difference (\%)"+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['b2_pre']-eff[d]['b2_post'])/ eff[d]['b2_pre'] ) )
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"


    '''
    count_m = 0
    for m in mass:
        eff_pr_c[m] = TGraph()
        eff_b2_pre_c[m] = TGraph()
        eff_b2_post_c[m] = TGraph()
        n = 0
        for c in ctau:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print m, c, d, eff[d]['pr']
                    print "point n. ", n
                    eff_pr_c[m].SetPoint(n,c,100*float(eff[d]['pr']/eff[d]['gen']))
                    eff_b2_pre_c[m].SetPoint(n,c,100*float(eff[d]['b2_pre']/eff[d]['gen']))
                    eff_b2_post_c[m].SetPoint(n,c,100*float(eff[d]['b2_post']/eff[d]['gen']))
                    n+=1
        print eff_pr_c[m].Print()
        print eff_b2_pre_c[m].Print()
        print eff_b2_post_c[m].Print()
        count_m+=1
    '''



def draw_princeton(label):
    with open(MAIN+"Dict_compare_princeton"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()
    mass = []#np.array([])
    ctau = []#np.array([])

    eff_met_m = defaultdict(dict)
    eff_ht_m = defaultdict(dict)
    eff_met_acc_m = defaultdict(dict)
    eff_ht_acc_m = defaultdict(dict)
    ratio = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]
        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    #c1.Divide(1, 2)
    #setTopPad(c1.GetPad(1))
    #setBotPad(c1.GetPad(2))
    #c1.cd(1)
    #c1.GetPad(bool(1)).SetTopMargin(0.06)
    #c1.GetPad(bool(1)).SetRightMargin(0.05)
    #c1.GetPad(bool(1)).SetTicks(1, 1)
    c1.SetTopMargin(0.06)
    c1.SetRightMargin(0.05)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    mg_ratio = TMultiGraph()
    leg1 = TLegend(0.1, 0.62, 0.35, 0.94)

    for c in ctau:
        eff_met_m[c] = TGraph()
        eff_ht_m[c] = TGraph()
        eff_met_acc_m[c] = TGraph()
        eff_ht_acc_m[c] = TGraph()
        ratio[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print "point n. ", n
                    eff_met_m[c].SetPoint(n,m,100*float(eff[d]['met']/eff[d]['gen']))
                    eff_ht_m[c].SetPoint(n,m,100*float(eff[d]['ht']/eff[d]['gen']))
                    eff_met_acc_m[c].SetPoint(n,m,100*float(eff[d]['met_acc']/eff[d]['gen']))
                    eff_ht_acc_m[c].SetPoint(n,m,100*float(eff[d]['ht_acc']/eff[d]['gen']))
                    ratio[c].SetPoint(n,m,float(eff[d]['met']/eff[d]['ht']))
                    n+=1
        
        eff_met_m[c].SetLineColor(colors[count_c])
        eff_ht_m[c].SetLineColor(colors[count_c+1])
        eff_met_acc_m[c].SetLineColor(colors[count_c])
        eff_ht_acc_m[c].SetLineColor(colors[count_c+1])

        eff_met_m[c].SetLineWidth(3)
        eff_ht_m[c].SetLineWidth(3)
        eff_met_acc_m[c].SetLineWidth(2)
        eff_ht_acc_m[c].SetLineWidth(2)

        eff_met_m[c].SetLineStyle(1)
        eff_ht_m[c].SetLineStyle(2)
        eff_met_acc_m[c].SetLineStyle(1)
        eff_ht_acc_m[c].SetLineStyle(2)

        #eff_met_m[c].SetMarkerStyle(21)
        #eff_ht_m[c].SetMarkerStyle(21)
        eff_met_acc_m[c].SetMarkerStyle(20)
        eff_ht_acc_m[c].SetMarkerStyle(21)

        #eff_met_m[c].SetMarkerColor(colors[count_c])
        #eff_ht_m[c].SetMarkerColor(colors[count_c])
        eff_met_acc_m[c].SetMarkerColor(colors[count_c])
        eff_ht_acc_m[c].SetMarkerColor(colors[count_c+1])

        #leg1.AddEntry(eff_met_m[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        leg1.SetHeader("c_{#tau} = "+str(c/1000.)+" m")
        eff_met_m[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        eff_met_m[c].GetYaxis().SetTitle("Efficiency (%)")
        eff_met_m[c].GetYaxis().SetTitleSize(0.05)
        eff_met_m[c].SetMinimum(0)
        eff_met_m[c].SetMaximum(101)
        mg.Add(eff_met_m[c])
        mg.Add(eff_ht_m[c])
        mg.Add(eff_met_acc_m[c])
        mg.Add(eff_ht_acc_m[c])

        #mg_ratio.Add(ratio[c])
        count_c+=1

    c1.cd(1)
    leg1.AddEntry(eff_met_m[ctau[0]],"MET","L")
    leg1.AddEntry(eff_ht_m[ctau[0]],"HT","L")
    leg1.AddEntry(eff_met_acc_m[ctau[0]],"MET + calo acc.","PL")
    leg1.AddEntry(eff_ht_acc_m[ctau[0]],"HT + tracker acc.","PL")
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.SetMinimum(0)
    #mg.SetMaximum(101)
    #c1.SetLogy()
    c1.SetGrid()
    mg.Draw("APL")
    leg1.Draw()
    LUMI = 1.
    drawCMS_simple(LUMI, "Simulation", ERA="2017", onTop=True)
    c1.Print(OUT+"SignEff_ComparePrinceton"+label+".pdf")
    c1.Print(OUT+"SignEff_ComparePrinceton"+label+".png")
    c1.Clear()


    #Print table
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & difference (\%)"+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['met']-eff[d]['ht'])/ eff[d]['met'] ) )
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"

def draw_time_smearing(label,ct):
    with open(OUT+"Dict_compare_time_smearing"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()
    mass = []#np.array([])
    ctau = []#np.array([])

    eff_jet_acc = defaultdict(dict)
    eff_jet_acc_smear = defaultdict(dict)
    eff_bin2 = defaultdict(dict)
    eff_bin2_smear = defaultdict(dict)
    ratio = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]
        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    #c1.Divide(1, 2)
    #setTopPad(c1.GetPad(1))
    #setBotPad(c1.GetPad(2))
    #c1.cd(1)
    #c1.GetPad(bool(1)).SetTopMargin(0.06)
    #c1.GetPad(bool(1)).SetRightMargin(0.05)
    #c1.GetPad(bool(1)).SetTicks(1, 1)
    c1.SetTopMargin(0.06)
    c1.SetBottomMargin(0.125)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.12)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    mg_ratio = TMultiGraph()
    #if log
    leg1 = TLegend(0.1+0.6, 0.62-0.45, 0.35+0.6, 0.94-0.45)
    #else
    leg1 = TLegend(0.1+0.6, 0.62-0.1, 0.35+0.6, 0.94-0.1)

    for c in ctau:
        if c!=ct:
            continue
        eff_jet_acc[c] = TGraph()
        eff_jet_acc_smear[c] = TGraph()
        eff_bin2[c] = TGraph()
        eff_bin2_smear[c] = TGraph()
        ratio[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print "point n. ", n
                    eff_jet_acc[c].SetPoint(n,m,100*float(eff[d]['jet_acc']/eff[d]['gen']))
                    eff_jet_acc_smear[c].SetPoint(n,m,100*float(eff[d]['jet_acc_smear']/eff[d]['gen']))
                    eff_bin2[c].SetPoint(n,m,100*float(eff[d]['bin2']/eff[d]['gen']))
                    eff_bin2_smear[c].SetPoint(n,m,100*float(eff[d]['bin2_smear']/eff[d]['gen']))
                    ratio[c].SetPoint(n,m,float(eff[d]['bin2_smear']/eff[d]['bin2']))
                    n+=1
        
        eff_jet_acc[c].SetLineColor(colors[count_c])
        eff_jet_acc_smear[c].SetLineColor(colors[count_c+1])
        eff_bin2[c].SetLineColor(colors[count_c])
        eff_bin2_smear[c].SetLineColor(colors[count_c+1])

        eff_jet_acc[c].SetLineWidth(3)
        eff_jet_acc_smear[c].SetLineWidth(3)
        eff_bin2[c].SetLineWidth(2)
        eff_bin2_smear[c].SetLineWidth(2)

        eff_jet_acc[c].SetLineStyle(1)
        eff_jet_acc_smear[c].SetLineStyle(2)
        eff_bin2[c].SetLineStyle(1)
        eff_bin2_smear[c].SetLineStyle(2)

        #eff_jet_acc[c].SetMarkerStyle(21)
        #eff_jet_acc_smear[c].SetMarkerStyle(21)
        eff_bin2[c].SetMarkerStyle(20)
        eff_bin2_smear[c].SetMarkerStyle(21)

        #eff_jet_acc[c].SetMarkerColor(colors[count_c])
        #eff_jet_acc_smear[c].SetMarkerColor(colors[count_c])
        eff_bin2[c].SetMarkerColor(colors[count_c])
        eff_bin2_smear[c].SetMarkerColor(colors[count_c+1])

        #leg1.AddEntry(eff_jet_acc[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        leg1.SetHeader("c_{#tau} = "+str(ct/1000.)+" m")
        eff_jet_acc[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        eff_jet_acc[c].GetYaxis().SetTitle("Efficiency (%)")
        eff_jet_acc[c].GetYaxis().SetTitleSize(0.05)
        eff_jet_acc[c].SetMinimum(0)
        eff_jet_acc[c].SetMaximum(101)
        mg.Add(eff_jet_acc[c])
        mg.Add(eff_jet_acc_smear[c])
        mg.Add(eff_bin2[c])
        mg.Add(eff_bin2_smear[c])

        #mg_ratio.Add(ratio[c])
        count_c+=1

    c1.cd(1)
    leg1.AddEntry(eff_jet_acc[ctau[0]],"jet acc.","L")
    leg1.AddEntry(eff_jet_acc_smear[ctau[0]],"jet acc. smear","L")
    leg1.AddEntry(eff_bin2[ctau[0]],"bin2","PL")
    leg1.AddEntry(eff_bin2_smear[ctau[0]],"bin2 smear","PL")
    mg.GetXaxis().SetTitle("m_{#chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.GetXaxis().SetTitleSize(0.05)
    #if log:
    #c1.SetLogy()
    #mg.SetMinimum(0.01)
    #mg.SetMaximum(110)
    #else
    mg.SetMinimum(0.)
    mg.SetMaximum(60)
    c1.SetGrid()
    mg.Draw("APL")
    leg1.Draw()
    LUMI = 1.
    drawCMS_simple(LUMI, "Simulation", ERA=ERA, onTop=True)
    c1.Print(OUT+"SignEff_CompareTimeSmearing_ct"+str(ct/1000.)+label+".pdf")
    c1.Print(OUT+"SignEff_CompareTimeSmearing_ct"+str(ct/1000.)+label+".png")
    c1.Clear()


    #Print table 
    print "$c \\tau$ (m) & $m_{\chi}$ (GeV) & difference (\%)"+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['bin2_smear']-eff[d]['bin2'])/ (eff[d]['bin2']+eff[d]['bin2_smear'])/2. ) )
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"


    #Print table 
    print "$m_{\\chi} (GeV) & c \\tau$ (m) "+"\\"+"\\"
    for m in mass:
        string = ""
        nt = 0
        string+= str(m) + " & "
        for c in ctau:
            for d in data:
                if "mh"+str(m)+"_ctau"+str(c) in d:
                    #string += str(c/1000.) + " & " + str(m)+" & "+str("%.1f" % ( 100*(eff[d]['bin2_smear']-eff[d]['bin2'])/ (eff[d]['bin2']+eff[d]['bin2_smear'])/2. ) )
                    string += str("%.2f" % ( 100*eff[d]['bin2_smear'] / eff[d]['gen']))
                    if nt==0: string += " & "
            nt+=1
        print string +"\\"+"\\"



def print_met_filters(label):

    with open(OUT+"Dict_compare_met_filters"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()

    mass = []#np.array([])
    ctau = []#np.array([])

    eff_jet_acc = defaultdict(dict)
    eff_jet_acc_smear = defaultdict(dict)
    eff_bin2 = defaultdict(dict)
    eff_bin2_smear = defaultdict(dict)
    ratio = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]


    for s in data:
        print "\n"
        print "\\begin{table}[!h]"
        print "\\begin{center}"
        print "\caption{$m_{\\chi}$="+str(samples[s]['mass'])+" (GeV), $c \\tau$ = "+str(samples[s]['ctau']/1000.)+" (m)}"
    
        print "\\begin{tabular}{|c|c|}"
        print "\hline"
        print "MET filter & efficiency (\%)\\\\"
        print "\hline"
        for k in eff[s].keys():
            if k!='gen' and k!="c" and k!="m":
                print k + " & "+str("%.2f" % ( 100*eff[s][k]))+" \\\\"
        print "\hline"
        print "\end{tabular}"
        print "\end{center}"
        print "\end{table}"


def print_time_smearing(label):

    with open(OUT+"Dict_compare_time_smearing"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()

    mass = []#np.array([])
    ctau = []#np.array([])

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]


    order_list = ["gen","met","trigger","lep_veto","pho_veto","tau_veto","d_phi","met_filter","cosmic_veto","bh_veto","jet_acc","bin2","jet_acc_smear","bin2_smear"]
    label_list = ["No selections","$E_T^{miss}$","Trigger","Lepton veto","Photon veto","Tau veto","Min $\Delta \\varphi(j,E_T^{miss})$","$E_T^{miss}$ filters","Cosmic Veto","Beam halo veto","Jet acceptance","SR","Jet acceptance smear","SR smear"]

    for s in data:
        print "\n"
        print "\\begin{table}[!h]"
        print "\\begin{center}"
        print "\caption{$m_{\\chi}$="+str(samples[s]['mass'])+" (GeV), $c \\tau$ = "+str(samples[s]['ctau']/1000.)+" (m)}"
    
        print "\\begin{tabular}{|c|c|c|}"
        print "\hline"
        print "Selection & Entries & Efficiency (\%)\\\\"
        print "\hline"
        #print str("%.2f" % ( 100.*eff[s]["bin2_smear"]/eff[s]['gen']))+ " & "+str("%.2f" % ( 100.*eff[s]["bin2_smear_isonoise"]/eff[s]['gen']))+ " & " +  str(  "%.2f" % (100*(eff[s]["bin2_smear_isonoise"]/eff[s]['gen'] - eff[s]["bin2_smear"]/eff[s]['gen']))   )   + " & " + str("%.2f" % ( 100*(eff[s]["bin2_smear_isonoise"] - eff[s]["bin2_smear"])/ (eff[s]["bin2_smear"])))# + eff[s]["bin2_smear_isonoise"])/2. ) ) 
        #for k in eff[s].keys():
        #   if k!='gen' and k!="c" and k!="m":
        #    #f "bin2_smear" in k:
        #       print k +  " & " + str("%d" % ( eff[s][k]))  + " & "+str("%.2f" % ( 100*eff[s][k]/eff[s]['gen']))+" \\\\"
        c = 0
        for k in order_list:
            print label_list[c] +  " & " + str("%d" % ( eff[s][k]))  + " & "+str("%.2f" % ( 100*eff[s][k]/eff[s]['gen']))+" \\\\"
            c+=1
        print "\hline"
        print "\end{tabular}"
        print "\end{center}"
        print "\end{table}"

def print_time_smearing_v2(label):

    with open(OUT+"Dict_compare_time_smearing_v2"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()

    mass = []#np.array([])
    ctau = []#np.array([])

    eff_jet_acc = defaultdict(dict)
    eff_jet_acc_smear = defaultdict(dict)
    eff_bin2 = defaultdict(dict)
    eff_bin2_smear = defaultdict(dict)
    ratio = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    colors = [1,2,418,801,856,602,920,881]
    colors = [2,418,801,856,602,920,881]


    for s in data:
        print "\n"
        print "\\begin{table}[!h]"
        print "\\begin{center}"
        print "\caption{$m_{\\chi}$="+str(samples[s]['mass'])+" (GeV), $c \\tau$ = "+str(samples[s]['ctau']/1000.)+" (m)}"
    
        print "\\begin{tabular}{|c|c|}"
        print "\hline"
        print "MET filter & efficiency (\%)\\\\"
        print "\hline"
        print str("%.2f" % ( 100.*eff[s]["bin2_smear"]/eff[s]['gen']))+ " & "+str("%.2f" % ( 100.*eff[s]["bin2_smear_isonoise"]/eff[s]['gen']))+ " & " +  str(  "%.2f" % (100*(eff[s]["bin2_smear_isonoise"]/eff[s]['gen'] - eff[s]["bin2_smear"]/eff[s]['gen']))   )   + " & " + str("%.2f" % ( 100*(eff[s]["bin2_smear_isonoise"] - eff[s]["bin2_smear"])/ (eff[s]["bin2_smear"])))# + eff[s]["bin2_smear_isonoise"])/2. ) ) 
        #for k in eff[s].keys():
        #   #if k!='gen' and k!="c" and k!="m":
        #    #f "bin2_smear" in k:
        #        print k + " & "+str("%.2f" % ( 100*eff[s][k]/eff[s]['gen']))+" \\\\"
        print "\hline"
        print "\end{tabular}"
        print "\end{center}"
        print "\end{table}"


def calc_eff(label,do_eta=False):
    eff = defaultdict(dict)
    for d in data:
        list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber",
                             #"nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB",
                             "isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced",
                             #,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"
                             "Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight",
                             "Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",
                             "TriggerWeight","PUWeight",
                             "PUReWeight",
                             "Jets.isGenMatchedCaloCorrLLPAccept","isSR","isWtoMN","isZtoMM","isMC",
                             "Z_pt",
                             "pt",
                             "min_dPhi_jets_*",
                             "n*Passing",
                         ]
        print "\n"
        print "\n"
        print "opening: ", MAIN
        list_files = samples[d]["files"]
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c
            print m, c
        new_list = []
        gen_events = 0
        for l in list_files:
            new_list.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        if "SMS" in samples[d]["files"][0]:
            name = samples[d]["files"][0]
            gen_events = sample[ name  ]['nevents']
        eff[d]['gen'] = gen_events
        print new_list
        gen = uproot.iterate(new_list,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])

        eta = np.array([])
        nRecHitsEB = np.array([])
        timeRecHitsEB = np.array([])
        timeRMSRecHitsEB = np.array([])
        sel1 = np.array([])
        sel2 = np.array([])
        sel3 = np.array([])

        for arrays in gen:
            cut_mask = arrays["isMC"]>0
            cut_mask = np.logical_and(cut_mask, np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v"],np.logical_or(arrays["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v"],arrays["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v"] )) )
            sel1 = np.concatenate((sel1, np.multiply(cut_mask[cut_mask],1.) ))
            cut_mask = np.logical_and(cut_mask, arrays["pt"]>200)
            sel2 = np.concatenate((sel2, np.multiply(cut_mask[cut_mask],1.) ))
            cut_mask = np.logical_and(cut_mask, np.logical_and( arrays["nMuonsPassing"]==0, np.logical_and(arrays["nElectronsPassing"]==0, np.logical_and(arrays["nPhotonsPassing"]==0, arrays["nTausPassing"]==0) ) ) )
            sel3 = np.concatenate((sel3,np.multiply(cut_mask[cut_mask],1.)))

        print "gen_events: ", gen_events
        print "************************************"
        print "events passing sel1: ", sel1.sum()
        print ("%d (%.2f%s) ") % (sel1.sum(), 100.*float(sel1.sum()/gen_events),'\%')        
        print "events passing sel2: ", sel2.sum()
        print ("%d (%.2f%s) ") % (sel2.sum(), 100.*float(sel2.sum()/gen_events),'\%')
        print "events passing sel3: ", sel3.sum()
        print ("%d (%.2f%s) ") % (sel3.sum(), 100.*float(sel3.sum()/gen_events),'\%')
        
        eff[d]['sel1'] = sel1.sum()
        eff[d]['sel2'] = sel2.sum()
        eff[d]['sel3'] = sel3.sum()

    print eff
    with open(MAIN+"Dict"+label+".yaml","w") as f:
        yaml.dump(eff, f)
        f.close()
        print "Info: dictionary written in file "+MAIN+"Dict"+label+".yaml"

def draw_eff(label,do_eta=False):
    with open(MAIN+"Dict"+label+".yaml","r") as f:
        print "\n"
        eff = yaml.load(f, Loader=yaml.Loader)
        f.close()


    mass = []#np.array([])
    ctau = []#np.array([])

    eff_sel1_m = defaultdict(dict)
    eff_sel1_c = defaultdict(dict)
    ratio = defaultdict(dict)
    eff_sel2_m = defaultdict(dict)
    eff_sel2_c = defaultdict(dict)
    eff_sel3_m = defaultdict(dict)
    eff_sel3_c = defaultdict(dict)

    for d in data:
        mass.append(eff[d]['m'])
        ctau.append(eff[d]['c'])

    mass = np.unique(np.array(mass))
    ctau = np.unique(np.array(ctau))

    #eff_sel1.append(eff[d]['sel1'])
    #eff_sel2.append(eff[d]['sel2'])
    #eff_sel3.append(eff[d]['sel3'])

    colors = [418,2,856,800,602,920,881,1]
        
    count_c = 0

    c1 = TCanvas("c1", "c1", 800, 600)
    #c1.Divide(1, 2)
    #setTopPad(c1.GetPad(1))
    #setBotPad(c1.GetPad(2))
    c1.cd(1)
    #c1.GetPad(bool(1)).SetTopMargin(0.06)
    #c1.GetPad(bool(1)).SetRightMargin(0.05)
    #c1.GetPad(bool(1)).SetTicks(1, 1)
    c1.SetTopMargin(0.1)
    c1.SetBottomMargin(0.125)
    c1.SetRightMargin(0.05)
    c1.SetLeftMargin(0.125)
    c1.SetTicks(1, 1)

    mg = TMultiGraph()
    mg_ratio = TMultiGraph()
    leg1 = TLegend(0.15, 0.6, 0.3, 0.9)
    leg1 = TLegend(0.15+0.5, 0.6-0.4, 0.3+0.6, 0.9-0.4)

    for c in ctau:
        eff_sel1_m[c] = TGraph()
        eff_sel2_m[c] = TGraph()
        eff_sel3_m[c] = TGraph()
        ratio[c] = TGraph()
        n = 0
        for m in mass:
            for d in data:
                if eff[d]['m']==m and eff[d]['c']==c:                    
                    print m, c, d, eff[d]['sel1']
                    print "point n. ", n
                    eff_sel1_m[c].SetPoint(n,m,100*float(eff[d]['sel1']/eff[d]['gen']))
                    eff_sel2_m[c].SetPoint(n,m,100*float(eff[d]['sel2']/eff[d]['gen']))
                    eff_sel3_m[c].SetPoint(n,m,100*float(eff[d]['sel3']/eff[d]['gen']))
                    ratio[c].SetPoint(n,m,float(eff[d]['sel3']/eff[d]['sel2']))
                    n+=1
        
        print c
        print eff_sel1_m[c].Print()
        print eff_sel2_m[c].Print()
        print eff_sel3_m[c].Print()
        eff_sel1_m[c].SetLineColor(colors[count_c])
        eff_sel2_m[c].SetLineColor(colors[count_c])
        eff_sel3_m[c].SetLineColor(colors[count_c])
        eff_sel1_m[c].SetLineWidth(3)
        eff_sel2_m[c].SetLineWidth(3)
        eff_sel3_m[c].SetLineWidth(3)
        eff_sel1_m[c].SetLineStyle(1)
        eff_sel2_m[c].SetLineStyle(2)
        eff_sel3_m[c].SetLineStyle(3)

        leg1.AddEntry(eff_sel1_m[c],"c_{#tau} = "+str(c/1000.)+" m","F")
        eff_sel2_m[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        eff_sel2_m[c].GetXaxis().SetTitleSize(0.05)
        eff_sel2_m[c].GetYaxis().SetTitle("Efficiency (%)")
        eff_sel2_m[c].GetYaxis().SetTitleSize(0.05)
        eff_sel2_m[c].SetMinimum(0)
        mg.Add(eff_sel1_m[c])
        mg.Add(eff_sel2_m[c])
        #mg.Add(eff_sel3_m[c])

        ratio[c].SetMarkerStyle(20)
        ratio[c].SetMarkerColor(colors[count_c])
        ratio[c].GetXaxis().SetTitle("m_{\chi} (GeV)")
        ratio[c].GetXaxis().SetTitleOffset(1.2)
        ratio[c].GetXaxis().SetTitleSize(0.15)
        ratio[c].GetXaxis().SetLabelSize(0.15)
        ratio[c].GetYaxis().SetTitle("ratio")
        ratio[c].GetYaxis().SetTitleSize(0.15)
        ratio[c].GetYaxis().SetTitleOffset(1.)
        ratio[c].GetYaxis().SetLabelSize(0.1)
        #if count_c==0:
        #    ratio[c].Draw("AP")
        #else:
        #    ratio[c].Draw("P,sames")
        mg_ratio.Add(ratio[c])
        count_c+=1

    c1.cd(1)
    leg1.AddEntry(eff_sel1_m[ctau[0]],"Trigger","L")
    leg1.AddEntry(eff_sel2_m[ctau[0]],"+ offline MET","L")
    #leg1.AddEntry(eff_sel3_m[ctau[0]],"+ lep. veto","L")
    mg.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg.GetYaxis().SetTitle("Efficiency (%)")
    mg.GetYaxis().SetTitleSize(0.05)
    mg.GetXaxis().SetTitleSize(0.05)
    mg.SetMinimum(0)
    mg.SetMaximum(101)
    mg.Draw("AL")
    LUMI = 0.
    drawCMS_simple(LUMI, "Simulation", ERA="2017", onTop=True)
    leg1.Draw()
    '''
    c1.cd(2)
    mg_ratio.GetXaxis().SetTitle("m_{\chi} (GeV)")
    mg_ratio.GetXaxis().SetTitleOffset(1.2)
    mg_ratio.GetXaxis().SetTitleSize(0.15)
    mg_ratio.GetXaxis().SetLabelSize(0.15)
    mg_ratio.GetYaxis().SetTitle("ratio")
    mg_ratio.GetYaxis().SetTitleSize(0.15)
    mg_ratio.GetYaxis().SetTitleOffset(1.)
    mg_ratio.GetYaxis().SetLabelSize(0.1)
    mg_ratio.Draw("AP")
    c1.SetLogy()
    '''
    c1.SetLogy()
    c1.Print("plots/Efficiency/test"+label+".pdf")
    c1.Print("plots/Efficiency/test"+label+".png")
    c1.Delete()

def print_todd_tables(era,label=""):
    #execution here
    YAML_DIR = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+era+"_SR_unblinding_ARC/SUSY/BR_h100_z0/datacards/"
    order_list = ["eff","stat","syst"]
    
    masses = []
    ctaus = []
    for s in data:
        masses.append(samples[s]["mass"])
    for s in data:
        ctaus.append(samples[s]["ctau"])
    
    masses = np.unique(np.array(masses))
    ctaus = np.unique(np.array(ctaus))

    print era, era, era, era, era
    print "\n"
    print "\\begin{table}[!h]"
    print "\\begin{center}"
    
    print "\\begin{tabular}{|c|c|c|}"
    print "\hline"
    print "Efficiency (\%) & Stat. unc. (\%) & Syst. unc (\%) \\\\"
    print "\hline"
    for m in masses:
        print_string = ""
        for c in ctaus:
            name = "SUSY_mh"+str(m)+"_ctau"+str(c)
            extra_label = ""
            if era=="2016":
                extra_label = "_G-H"
            with open(YAML_DIR+name.replace("_HH","")+extra_label+".yaml","r") as f:
                y_sign = yaml.load(f, Loader=yaml.Loader)
                f.close()
            if c==500 and print_string=="":
                print_string += str(samples[name]["mass"]) + " & " + str("%.2f" % ( y_sign["eff"]))  + " & "+str("%.2f" % (y_sign["stat"] ) ) + " & " +str("%.2f" % (y_sign["syst"] ) ) 
            if c==3000 and print_string!="":
                print_string += " & " + str("%.2f" % ( y_sign["eff"]))  + " & "+str("%.2f" % (y_sign["stat"] ) ) + " & " +str("%.2f" % (y_sign["syst"] ) ) +" \\\\"
        print print_string
    
    print "\hline"
    print "\end{tabular}"
    print "\end{center}"
    print "\end{table}"

###
# count events in SR with correlated time smearing (1 random number per event) and uncorrelated time smearing (1 random number per jet)
label ="_time_smeared_from_v6_SR_no_time_smearing"

#Keep this as a reference!!
#label ="time_smeared_uncorrelated"

smear_correlation(label,do_eta=True)
print_smear_correlation(label)
exit()
###

label = ""
for era in ["2016","2017","2018"]:
    print_todd_tables(era)
exit()

#label = "debug_with_SR_applied"
label = "debug_from_gen"
#label = "debug_from_gen_new"
compare_time_smearing(label,do_eta=False)
print_time_smearing(label)
exit()

#debug what's wrong
label = ""
debug(label)
exit()

#compare time smearing
label = ""
#compare_time_smearing(label,do_eta=False)
#compare_time_smearing_v2(label,do_eta=False)
print_time_smearing(label)
#print_time_smearing_v2(label)
#draw_time_smearing(label,ct=500)
exit()


#compare met filters
label = ""
#compare_met_filters(label,do_eta=False)
print_met_filters(label)
exit()

#compare princeton
label = "_ct0p5m"
compare_princeton(label,do_eta=False)
draw_princeton(label)
exit()

label ="bh_updated_0p1"
count(label,do_eta=True)
draw(label)
exit()



label ="neg_bh_impact"
count(label,do_eta=True)
exit()

label ="neg_cosmic_veto_impact"
count(label,do_eta=True)
exit()

label ="neg"
count_jet_negative(label,do_eta=True)
exit()


label = "_BH_min_dPhi_rec_hits"
count(label,do_eta=True)
#draw(label)
exit()

label = "_BH_min_dPhi"
count(label,do_eta=True)
draw(label)
exit()


label = "_MinDPhi0p5"
#count(label,do_eta=True)
draw(label)
exit()

count_gen_central()
exit()
#label = "_cosmic_veto"
#count(label)
#exit()

#label = "_BH_RMS_8"
#label = "_BH_RMS_8_nHits_3"#basically no effect!
#label = "_BH_RMS_8_nHits_10"
#label = "_min_dPhi_jets_eta_1p0_0p996_0p05"

#label = "_min_dPhi_jets_eta_1p0_0p996_0p05_eta_1p0"
#label = "_BH_RMS_8_eta_1p0"
#label = "_BH_RMS_8_nHits_3_eta_1p0"#basically no effect!
label = "_BH_RMS_8_nHits_10_eta_1p0_data"
count(label,do_eta=True)
draw(label)
exit()

#label = "_test_no_dphi"
#label = "_gen"
label = "_SR_minjetmetdphi"
label = "_SR_minjetmetdphi_new_BH"
#label = "_SR_minjetmetdphi_old_BH"
#label = "_SR_minjetmetdphi_old_BH_eta_1p0"
#label = "_SR_minjetmetdphi_new_BH_only_timeRMS"
#count(label)
#draw(label)

label = "_MET_trigger_lep_veto"
#calc_eff(label)
draw_eff(label)
