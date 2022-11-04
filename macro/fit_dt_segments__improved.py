#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
#import matplotlib.pyplot as plt

#from Analyzer.LLP2017.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = [
    "skim_pickevents_275309_875415530",
    "skim_pickevents_276811_244034371",#this fails the opposite emisphere requirement
    #"skim_pickevents_277096_153555117",#pure noise
    "skim_pickevents_279029_387083289",
    "skim_pickevents_279931_4073593513",
    "skim_pickevents_302485_987393715",
    "skim_pickevents_319910_457404349",
    #"test_signal",
    #'HighMETRun2016B-07Aug17_ver2-v1',  'HighMETRun2016E-07Aug17_v1',  'HighMETRun2016H-07Aug17_v1',
    #'HighMETRun2016C-07Aug17_v1',       'HighMETRun2016F-07Aug17_v1',
    #'HighMETRun2016D-07Aug17_v1',       'HighMETRun2016G-07Aug17_v1',
    #"HighMETRun2017E-17Nov2017-v1",
    #"HighMETRun2018A-17Sep2018-v1",
    #"HighMETRun2018D-PromptReco-v2"
]
#MAIN = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/"
MAIN = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/"
##MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2016_SR/"
#MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_negative_jets_HBHE/"
##OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_v3/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2017_BeamHalo_studies/"
##OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_bin_1/"
#OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_moar/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/check_cosmics_HBHE/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/cosmics_fit_on_cosmics/"

for d in data:

    '''
    in_file = TFile(MAIN+d+".root")
    tree = in_file.Get("tree")
    arr_run = []
    arr_lumi = []
    arr_event = []
    for entry in tree:
        print entry.RunNumber
        print entry.EventNumber
        print entry.LumiNumber
        arr_run.append(entry.RunNumber)
        arr_event.append(entry.EventNumber)
        arr_lumi.append(entry.LumiNumber)

    exit()
    '''

    chain = TChain("tree")
    chain.Add(MAIN+d+".root")


    #h_3D = TH1F("rh","", 300, 0, 3000)
    #gh.Sumw2()
    #rh.Sumw2()

    list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber","nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept","isSR","isMC","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets"]#,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"]
    print "opening: ", MAIN + d + ".root"
    gen = uproot.iterate([MAIN + d + ".root"],"tree",list_of_variables)
    print gen
    for arrays in gen:
        cut_mask = arrays["isSR"]>0
        cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
        #print "SELECT bin 1!!!"
        #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)
        #print "SELECT events close in dphi"
        #cut_mask = np.logical_and(cut_mask,arrays["min_dPhi_jets"]<0.05)

        print "SELECT affected event"
        #cut_mask = np.logical_and(cut_mask,arrays["EventNumber"]==195197216)
        #cut_mask = np.logical_and(cut_mask,arrays["EventNumber"]==1329879583)
        #cut_mask = np.logical_and(cut_mask,arrays["EventNumber"]==270123341)
        isMC = arrays["isMC"][cut_mask]
        RunNumber = arrays["RunNumber"][cut_mask]
        LumiNumber = arrays["LumiNumber"][cut_mask]
        EventNumber = arrays["EventNumber"][cut_mask]
        EventWeight = arrays["EventWeight"][cut_mask]
        nDTSegments=arrays["nDTSegments"][cut_mask]
        nCosmics=arrays["nCosmicMuons"][cut_mask]
        nCosmicsOneLeg=arrays["nCosmicMuonsOneLeg"][cut_mask]
        ##Cosmics_eta=arrays["CosmicMuons.eta"][cut_mask]
        ##Cosmics_phi=arrays["CosmicMuons.phi"][cut_mask]
        ##CosmicsOneLeg_eta=arrays["CosmicMuonsOneLeg.eta"][cut_mask]
        ##CosmicsOneLeg_phi=arrays["CosmicMuonsOneLeg.phi"][cut_mask]
        #DT_x = np.divide(arrays["DTSegments.x"].flatten(),100.)
        #DT_y = np.divide(arrays["DTSegments.y"].flatten(),100.)
        #DT_z = np.divide(arrays["DTSegments.z"].flatten(),100.)
        #ECAL_tag_x = np.divide(arrays["taggedEcalRecHitsAK4.x"].flatten(),100.)
        #ECAL_tag_y = np.divide(arrays["taggedEcalRecHitsAK4.y"].flatten(),100.)
        #ECAL_tag_z = np.divide(arrays["taggedEcalRecHitsAK4.z"].flatten(),100.)
        #ECAL_x = np.divide(arrays["skimmedEcalRecHitsAK4.x"].flatten(),100.)
        #ECAL_y = np.divide(arrays["skimmedEcalRecHitsAK4.y"].flatten(),100.)
        #ECAL_z = np.divide(arrays["skimmedEcalRecHitsAK4.z"].flatten(),100.)
        ##DT_x = arrays["DTSegments.eta"][cut_mask]#
        DT_x = np.divide(arrays["DTSegments.x"][cut_mask],100.)
        DT_y = np.divide(arrays["DTSegments.y"][cut_mask],100.)#
        DT_z = np.divide(arrays["DTSegments.z"][cut_mask],100.)#
        DT_t = arrays["DTSegments.time"][cut_mask]
        DT_eta = arrays["DTSegments.eta"][cut_mask]
        DT_phi = arrays["DTSegments.phi"][cut_mask]
        CSC_x = np.divide(arrays["CSCSegments.x"][cut_mask],100.)
        CSC_y = np.divide(arrays["CSCSegments.y"][cut_mask],100.)#
        CSC_z = np.divide(arrays["CSCSegments.z"][cut_mask],100.)#
        CSC_eta = arrays["CSCSegments.eta"][cut_mask]
        CSC_phi = arrays["CSCSegments.phi"][cut_mask]
        ECAL_tag_x = np.divide(arrays["taggedEcalRecHitsAK4.x"][cut_mask],100.)
        ECAL_tag_y = np.divide(arrays["taggedEcalRecHitsAK4.y"][cut_mask],100.)
        ECAL_tag_z = np.divide(arrays["taggedEcalRecHitsAK4.z"][cut_mask],100.)
        ECAL_tag_eta = arrays["taggedEcalRecHitsAK4.eta"][cut_mask]
        ECAL_tag_phi = arrays["taggedEcalRecHitsAK4.phi"][cut_mask]
        ECAL_x = np.divide(arrays["skimmedEcalRecHitsAK4.x"][cut_mask],100.)
        ECAL_y = np.divide(arrays["skimmedEcalRecHitsAK4.y"][cut_mask],100.)
        ECAL_z = np.divide(arrays["skimmedEcalRecHitsAK4.z"][cut_mask],100.)
        ECAL_eta = arrays["skimmedEcalRecHitsAK4.eta"][cut_mask]
        ECAL_phi = arrays["skimmedEcalRecHitsAK4.phi"][cut_mask]

    print "This should be done in uproot because this puts together all the events"
    print "1. make one plot per event -- > rewmove the flatten"
    print "2. look at signal"
    print "3. define a metric to decide overlapping ecal and dt"

    #print DT_x.shape
    #print "DT_x: ", DT_x
    #print " DT_x[0] ", DT_x[0]
    #print " DT_x[0][0] ", DT_x[0][0]
    #print DT_z

    for ev in range(EventNumber.shape[0]):
        #print ev, DT_x[ev], DT_x[ev].size
        if DT_x[ev].size == 0:
            continue
        if ECAL_tag_x[ev].size == 0:
            continue


        DT       = np.dstack((DT_x[ev],DT_y[ev],DT_z[ev])).reshape(-1,3)
        DT_cyl   = np.dstack((DT_eta[ev],DT_phi[ev],DT_z[ev])).reshape(-1,3)
        ECAL     = np.dstack((ECAL_x[ev],ECAL_y[ev],ECAL_z[ev])).reshape(-1,3)
        ECAL_cyl     = np.dstack((ECAL_eta[ev],ECAL_phi[ev],ECAL_z[ev])).reshape(-1,3)
        ECAL_tag = np.dstack((ECAL_tag_x[ev],ECAL_tag_y[ev],ECAL_tag_z[ev])).reshape(-1,3)
        ECAL_tag_cyl = np.dstack((ECAL_tag_eta[ev],ECAL_tag_phi[ev],ECAL_tag_z[ev])).reshape(-1,3)
        DT_time      = DT_t[ev]
        CSC       = np.dstack((CSC_x[ev],CSC_y[ev],CSC_z[ev])).reshape(-1,3)
        CSC_cyl   = np.dstack((CSC_eta[ev],CSC_phi[ev],CSC_z[ev])).reshape(-1,3)


        ##Hist definition
        h_xy_dt   = TH2F("h_xy_dt","",     100, -10, 10, 100, -10, 10)
        h_xy_dt_n   = TH2F("h_xy_dt_n","",     100, -10, 10, 100, -10, 10)
        h_xy_dt_0   = TH2F("h_xy_dt_0","",     100, -10, 10, 100, -10, 10)
        h_xy_dt_1   = TH2F("h_xy_dt_1","",     100, -10, 10, 100, -10, 10)
        h_xy_dt_2   = TH2F("h_xy_dt_2","",     100, -10, 10, 100, -10, 10)
        h_xy_dt_3   = TH2F("h_xy_dt_3","",     100, -10, 10, 100, -10, 10)
        h_xy_ecal = TH2F("h_xy_ecal","",   100, -10, 10, 100, -10, 10)
        h_xy_ecal_tag = TH2F("h_xy_ecal_tag","",   100, -10, 10, 100, -10, 10)
        h_xy_dt_sum   = TH2F("h_xy_dt_sum","",     100, -10, 10, 100, -10, 10)

        h_etaphi_dt     = TH2F("h_etaphi_dt","",         100, -3, 3, 100, -4, 4)
        h_etaphi_dt_n   = TH2F("h_etaphi_dt_n","",       100, -3, 3, 100, -4, 4)
        h_etaphi_dt_0   = TH2F("h_etaphi_dt_0","",       100, -3, 3, 100, -4, 4)
        h_etaphi_dt_1   = TH2F("h_etaphi_dt_1","",       100, -3, 3, 100, -4, 4)
        h_etaphi_dt_2   = TH2F("h_etaphi_dt_2","",       100, -3, 3, 100, -4, 4)
        h_etaphi_dt_3   = TH2F("h_etaphi_dt_3","",       100, -3, 3, 100, -4, 4)
        h_etaphi_ecal   = TH2F("h_etaphi_ecal","",       100, -3, 3, 100, -4, 4)
        h_etaphi_ecal_tag = TH2F("h_etaphi_ecal_tag","", 100, -3, 3, 100, -4, 4)
        h_etaphi_dt_sum   = TH2F("h_etaphi_dt_sum","",   100, -3, 3, 100, -4, 4)
        #h_etaphi_cos     = TH2F("h_etaphi_cos","",         100, -3, 3, 100, -4, 4)
        #h_etaphi_cos1L   = TH2F("h_etaphi_cos1L","",         100, -3, 3, 100, -4, 4)

        h_yz_dt   = TH2F("h_yz_dt","",     100, -10, 10, 100, -10, 10)
        h_yz_dt_n   = TH2F("h_yz_dt_n","",     100, -10, 10, 100, -10, 10)
        h_yz_dt_0   = TH2F("h_yz_dt_0","",     100, -10, 10, 100, -10, 10)
        h_yz_dt_1   = TH2F("h_yz_dt_1","",     100, -10, 10, 100, -10, 10)
        h_yz_dt_2   = TH2F("h_yz_dt_2","",     100, -10, 10, 100, -10, 10)
        h_yz_dt_3   = TH2F("h_yz_dt_3","",     100, -10, 10, 100, -10, 10)
        h_yz_ecal = TH2F("h_yz_ecal","",   100, -10, 10, 100, -10, 10)
        h_yz_ecal_tag = TH2F("h_yz_ecal_tag","",   100, -10, 10, 100, -10, 10)
        h_yz_dt_sum   = TH2F("h_yz_dt_sum","",     100, -10, 10, 100, -10, 10)

        h_phiz_dt   = TH2F("h_phiz_dt","",     100, -10, 10, 100, -10, 10)
        h_phiz_dt_n   = TH2F("h_phiz_dt_n","",     100, -10, 10, 100, -10, 10)
        h_phiz_dt_0   = TH2F("h_phiz_dt_0","",     100, -10, 10, 100, -10, 10)
        h_phiz_dt_1   = TH2F("h_phiz_dt_1","",     100, -10, 10, 100, -10, 10)
        h_phiz_dt_2   = TH2F("h_phiz_dt_2","",     100, -10, 10, 100, -10, 10)
        h_phiz_dt_3   = TH2F("h_phiz_dt_3","",     100, -10, 10, 100, -10, 10)
        h_phiz_ecal = TH2F("h_phiz_ecal","",   100, -10, 10, 100, -10, 10)
        h_phiz_ecal_tag = TH2F("h_phiz_ecal_tag","",   100, -10, 10, 100, -10, 10)
        h_phiz_dt_sum   = TH2F("h_phiz_dt_sum","",     100, -10, 10, 100, -10, 10)


        h_xz_dt   = TH2F("h_xz_dt","",     100, -10, 10, 100, -10, 10)
        h_xz_dt_n   = TH2F("h_xz_dt_n","",     100, -10, 10, 100, -10, 10)
        h_xz_dt_0   = TH2F("h_xz_dt_0","",     100, -10, 10, 100, -10, 10)
        h_xz_dt_1   = TH2F("h_xz_dt_1","",     100, -10, 10, 100, -10, 10)
        h_xz_dt_2   = TH2F("h_xz_dt_2","",     100, -10, 10, 100, -10, 10)
        h_xz_dt_3   = TH2F("h_xz_dt_3","",     100, -10, 10, 100, -10, 10)
        h_xz_ecal = TH2F("h_xz_ecal","",   100, -10, 10, 100, -10, 10)
        h_xz_ecal_tag = TH2F("h_xz_ecal_tag","",   100, -10, 10, 100, -10, 10)
        h_xz_dt_sum   = TH2F("h_xz_dt_sum","",     100, -10, 10, 100, -10, 10)

        h_etaz_dt   = TH2F("h_etaz_dt","",     100, -10, 10, 100, -10, 10)
        h_etaz_dt_n   = TH2F("h_etaz_dt_n","",     100, -10, 10, 100, -10, 10)
        h_etaz_dt_0   = TH2F("h_etaz_dt_0","",     100, -10, 10, 100, -10, 10)
        h_etaz_dt_1   = TH2F("h_etaz_dt_1","",     100, -10, 10, 100, -10, 10)
        h_etaz_dt_2   = TH2F("h_etaz_dt_2","",     100, -10, 10, 100, -10, 10)
        h_etaz_dt_3   = TH2F("h_etaz_dt_3","",     100, -10, 10, 100, -10, 10)
        h_etaz_ecal = TH2F("h_etaz_ecal","",   100, -10, 10, 100, -10, 10)
        h_etaz_ecal_tag = TH2F("h_etaz_ecal_tag","",   100, -10, 10, 100, -10, 10)
        h_etaz_dt_sum   = TH2F("h_etaz_dt_sum","",     100, -10, 10, 100, -10, 10)

        h_xyz_dt   = TH3F("h_xyz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_n   = TH3F("h_xyz_dt_n","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_0   = TH3F("h_xyz_dt_0","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_1   = TH3F("h_xyz_dt_1","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_2   = TH3F("h_xyz_dt_2","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_3   = TH3F("h_xyz_dt_3","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_ecal = TH3F("h_xyz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_ecal_tag = TH3F("h_xyz_ecal_tag","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_xyz_dt_sum   = TH3F("h_xyz_dt_sum","",   100, -10, 10, 100, -10, 10, 100, -10, 10)

        h_etaphiz_dt   = TH3F("h_etaphiz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_n   = TH3F("h_etaphiz_dt_n","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_0   = TH3F("h_etaphiz_dt_0","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_1   = TH3F("h_etaphiz_dt_1","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_2   = TH3F("h_etaphiz_dt_2","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_3   = TH3F("h_etaphiz_dt_3","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_ecal = TH3F("h_etaphiz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_ecal_tag = TH3F("h_etaphiz_ecal_tag","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphiz_dt_sum   = TH3F("h_etaphiz_dt_sum","",   100, -10, 10, 100, -10, 10, 100, -10, 10)

        fit_dt_3d   = TH3F("fit_dt_3d","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        fit_dt_xy   = TH2F("fit_dt_xy","",   100, -10, 10, 100, -10, 10)
        fit_dt_xz   = TH2F("fit_dt_xz","",   100, -10, 10, 100, -10, 10)
        fit_dt_yz   = TH2F("fit_dt_yz","",   100, -10, 10, 100, -10, 10)

        fit_ecal_3d   = TH3F("fit_ecal_3d","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
        fit_ecal_xy   = TH2F("fit_ecal_xy","",   100, -10, 10, 100, -10, 10)
        fit_ecal_xz   = TH2F("fit_ecal_xz","",   100, -10, 10, 100, -10, 10)
        fit_ecal_yz   = TH2F("fit_ecal_yz","",   100, -10, 10, 100, -10, 10)

        h_xy_csc   = TH2F("h_xy_csc","",     100, -10, 10, 100, -10, 10)
        h_yz_csc   = TH2F("h_yz_csc","",     100, -10, 10, 100, -10, 10)
        h_xz_csc   = TH2F("h_xz_csc","",     100, -10, 10, 100, -10, 10)
        h_xyz_csc   = TH3F("h_xyz_csc","",     100, -10, 10, 100, -10, 10, 100, -10, 10)
        h_etaphi_csc     = TH2F("h_etaphi_csc","",         100, -3, 3, 100, -4, 4)

        ##DBSCAN section
        dbscan = DBSCAN(eps=1.4,min_samples=3).fit(DT)
        #dbscan = DBSCAN(eps=1.4,min_samples=3).fit(DT)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print "DB scan: "
        print "labels", labels
        print "n_clusters_", n_clusters_
        print "n_noise_", n_noise_
        m_noise = (labels==-1)
        m = {}
        for l in labels:
            if l!=-1:
                m[l] = (labels==l)
        #m[0] = (labels==0)
        #m[1] = (labels==1)
        #m[2] = (labels==2)
        #m[3] = (labels==3)
        m_cl_0 = (labels==0)
        m_cl_1 = (labels==1)
        m_cl_2 = (labels==2)
        m_cl_3 = (labels==3)

        DT_cl = {}
        DT_cl_cyl = {}
        DT_cl_time = {}
        DT_n = DT[m_noise]
        DT_n_cyl = DT_cyl[m_noise]

        for k in m.keys():
            DT_cl[k] = DT[m[k]]
            DT_cl_cyl[k] = DT_cyl[m[k]]
            #print DT_time
            #print m[k]
            DT_cl_time[k] = DT_time[m[k]]

        #print DT_cl
        if n_clusters_>=2:
            n_pairs = int(n_clusters_/2)
            if (n_clusters_ % 2 )!=0:
                print "Odd number of DT clusters, strange..."
                #exit()

            mean_x = {}
            mean_y = {}
            mean_z = {}
            mean_t = {}
            std_t = {}
            for k in m.keys():
                mean_x[k] = np.sum(DT_cl[k][:,0])/np.size( DT_cl[k][:,0] )
                mean_y[k] = np.sum(DT_cl[k][:,1])/np.size( DT_cl[k][:,1] )
                mean_z[k] = np.sum(DT_cl[k][:,2])/np.size( DT_cl[k][:,2] )
                mean_t[k] = np.mean(DT_cl_time[k])
                std_t[k]  = np.std(DT_cl_time[k])

            mean_ECAL_tag_z = np.sum(ECAL_tag_z[ev].flatten())/np.size(ECAL_tag_z[ev])
            print "mean_ECAL_tag_z: ", mean_ECAL_tag_z

            pairs = {}
            #for n in range(n_pairs):
            #    pairs[n] = np.array([-1,1])

            #This handles two cosmics. If there are more, we'll have to deal with this somehow...
            dz_DT = 1000.
            dz_ECAL = 1000.
            ch_k1 = -1
            ch_k2 = -1
            ch_t_k1 = 0
            ch_t_k2 = 0
            for k1 in range(len(m.keys())):
                print "k1: ", k1
                print mean_x[k1], mean_y[k1], mean_z[k1]
                for k2 in range(len(m.keys())):
                    if k2!=k1: print "k2: ", k2
                    if k2!=k1: print mean_x[k2], mean_x[k2], mean_z[k2]
                    #print "time sparsity: ", std_t[k1], std_t[k2]
                    ##if k2>k1 and ( (mean_x[k1]*mean_x[k2]<0 and mean_y[k1]*mean_y[k2]<0 and mean_t[k1]*mean_t[k2]<0) or (mean_t[k1]*mean_t[k2]<0 and mean_x[k1]*mean_x[k2]<0 and (abs(mean_y[k1]<0.2) or abs(mean_y[k2]<0.2))) ) and nCosmicsOneLeg[ev]>0 and nCosmics[ev]>1:
                    #if k2>k1 and ( (mean_x[k1]*mean_x[k2]<0 and mean_y[k1]*mean_y[k2]<0) or (mean_x[k1]*mean_x[k2]<0 and (abs(mean_y[k1]<0.2) or abs(mean_y[k2]<0.2))) ) and nCosmicsOneLeg[ev]>0 and nCosmics[ev]>1:

                    if k2!=k1 and ( (mean_x[k1]*mean_x[k2])<0 or (mean_y[k1]*mean_y[k2])<0 or (mean_z[k1]*mean_z[k2])<0 ) and nCosmicsOneLeg[ev]>0:# and nCosmics[ev]>1:
                        ##opposite emispheres
                        print "pair: ", k1, k2
                        print "z average: ", mean_z[k1], mean_z[k2]
                        tmp_z = abs(mean_z[k1] - mean_z[k2])
                        dz_DT = min(dz_DT,tmp_z)
                        print "tmp hits distance in z: ", tmp_z
                        print "best hits distance in z: ", dz_DT
                        print "do I need this?"

                        tmp_ECAL = abs((mean_z[k1]+mean_z[k2])/2. - mean_ECAL_tag_z)
                        dz_ECAL = min(dz_ECAL,tmp_ECAL)
                        print "tmp ECAL distance in z: ", tmp_ECAL
                        print "best ECAL distance in z: ", dz_ECAL

                        #if dz_DT==tmp_z and dz_ECAL==tmp_ECAL:
                        if dz_ECAL==tmp_ECAL:
                            ch_k1 = k1
                            ch_k2 = k2
                            ch_t_k1 = mean_t[k1]
                            ch_t_k2 = mean_t[k2]
                            print "tmp chosen pairs: ", k1, k2

            #print "Overwrite DT dict"
            if ch_k1>-1 and ch_k2>-1:
                print "Pair: (",ch_k1,",",ch_k2,")"
                print "Time sparsity: ", ch_t_k1, " ; ", ch_t_k2
                tmp_DT_cl_0 = DT_cl[ch_k1]
                tmp_DT_cl_1 = DT_cl[ch_k2]
                tmp_DT_cl_cyl_0 = DT_cl_cyl[ch_k1]
                tmp_DT_cl_cyl_1 = DT_cl_cyl[ch_k2]

                new_keys = m.keys()
                new_keys = np.setdiff1d(np.array(m.keys()),np.array([ch_k1,ch_k2]))
                tmp_cl_discarded = {}
                tmp_cl_cyl_discarded = {}
                for a in range(len(new_keys)):
                    tmp_cl_discarded[a] = DT_cl[new_keys[a]]
                    tmp_cl_cyl_discarded[a] = DT_cl_cyl[new_keys[a]]

                DT_cl[0] = tmp_DT_cl_0
                DT_cl[1] = tmp_DT_cl_1
                DT_cl_cyl[0] = tmp_DT_cl_cyl_0
                DT_cl_cyl[1] = tmp_DT_cl_cyl_1

                #Add discarded cosmics to noise histogram
                for a in range(len(new_keys)):
                    DT_n = np.vstack((DT_n,tmp_cl_discarded[a]))
                    DT_n_cyl = np.vstack((DT_n_cyl,tmp_cl_cyl_discarded[a]))
                    #DT_n.vstack(tmp_cl_discarded[a])
                    #DT_cl[2+a] = tmp_cl_discarded[a]
            else:
                print "No good cosmic identified, all noise"
                for a in (m.keys()):
                    DT_n = np.vstack((DT_n,DT_cl[a]))
                    DT_n_cyl = np.vstack((DT_n_cyl,DT_cl_cyl[a]))
                for a in (m.keys()):
                    DT_cl[a] = np.array([])
                    DT_cl_cyl[a] = np.array([])
                continue

        if n_clusters_==1:
            print "Only 1 cluster, no cosmic identified, all noise"
            for a in (m.keys()):
                DT_n = np.vstack((DT_n,DT_cl[a]))
                DT_n_cyl = np.vstack((DT_n_cyl,DT_cl_cyl[a]))
            for a in (m.keys()):
                DT_cl[a] = np.array([])
                DT_cl_cyl[a] = np.array([])
            continue

        if n_clusters_==0:
            print "No clusters, no cosmic identified, all noise"
            continue



        root_numpy.fill_hist(h_xy_dt_n,DT_n[:,[0,1]])
        root_numpy.fill_hist(h_xz_dt_n,DT_n[:,[2,0]])
        root_numpy.fill_hist(h_yz_dt_n,DT_n[:,[2,1]])
        root_numpy.fill_hist(h_xyz_dt_n,DT_n[:,[2,0,1]])

        root_numpy.fill_hist(h_etaphi_dt_n,DT_n_cyl[:,[0,1]])
        root_numpy.fill_hist(h_phiz_dt_n,DT_n_cyl[:,[1,2]])
        root_numpy.fill_hist(h_etaz_dt_n,DT_n_cyl[:,[0,2]])
        root_numpy.fill_hist(h_etaphiz_dt_n,DT_n_cyl[:,[0,1,2]])

        #Initialize objects
        DT_to_fit = np.array([])
        DT_post_fit = np.array([])

        if len(DT_cl.keys())>0:
            if DT_cl[0].size>0:
                root_numpy.fill_hist(h_xy_dt_0,DT_cl[0][:,[0,1]])
                root_numpy.fill_hist(h_xz_dt_0,DT_cl[0][:,[2,0]])
                root_numpy.fill_hist(h_yz_dt_0,DT_cl[0][:,[2,1]])
                root_numpy.fill_hist(h_xyz_dt_0,DT_cl[0][:,[2,0,1]])

                root_numpy.fill_hist(h_etaphi_dt_0,DT_cl_cyl[0][:,[0,1]])
                root_numpy.fill_hist(h_etaz_dt_0,DT_cl_cyl[0][:,[0,2]])
                root_numpy.fill_hist(h_phiz_dt_0,DT_cl_cyl[0][:,[1,2]])
                root_numpy.fill_hist(h_etaphiz_dt_0,DT_cl_cyl[0][:,[0,1,2]])
                print "DT_cl_cyl[0]"
                print DT_cl_cyl[0]
                
        if len(DT_cl.keys())>1:
            if DT_cl[1].size>0:
                root_numpy.fill_hist(h_xy_dt_1,DT_cl[1][:,[0,1]])
                root_numpy.fill_hist(h_xz_dt_1,DT_cl[1][:,[2,0]])
                root_numpy.fill_hist(h_yz_dt_1,DT_cl[1][:,[2,1]])
                root_numpy.fill_hist(h_xyz_dt_1,DT_cl[1][:,[2,0,1]])

                root_numpy.fill_hist(h_etaphi_dt_1,DT_cl_cyl[1][:,[0,1]])
                root_numpy.fill_hist(h_etaz_dt_1,DT_cl_cyl[1][:,[0,2]])
                root_numpy.fill_hist(h_phiz_dt_1,DT_cl_cyl[1][:,[1,2]])
                root_numpy.fill_hist(h_etaphiz_dt_1,DT_cl_cyl[1][:,[0,1,2]])

                print "DT_cl_cyl[1]"
                print DT_cl_cyl[1]

        if len(DT_cl.keys())>1:
            DT_to_fit = np.concatenate((DT_cl[0],DT_cl[1]))
        else:
            DT_to_fit = np.array([])

        #print "full ECAL_tag"
        #print ECAL_tag
        ECAL_to_fit = ECAL_tag
        ssq = 1000.

        root_numpy.fill_hist(h_xy_ecal,ECAL[:,[0,1]])
        root_numpy.fill_hist(h_xz_ecal,ECAL[:,[2,0]])
        root_numpy.fill_hist(h_yz_ecal,ECAL[:,[2,1]])
        root_numpy.fill_hist(h_xyz_ecal,ECAL[:,[2,0,1]])

        #Cos_cyl   = np.dstack((Cosmics_eta[ev],Cosmics_phi[ev])).reshape(-1,2)
        #print "cos cyl"
        #print Cos_cyl
        #Cos1L_cyl   = np.dstack((CosmicsOneLeg_eta[ev],CosmicsOneLeg_phi[ev])).reshape(-1,2)
        #print "cos cyl 1L"
        #print Cos1L_cyl

        #root_numpy.fill_hist(h_etaphi_cos,Cos_cyl)
        #root_numpy.fill_hist(h_etaphi_cos1L,Cos1L_cyl)

        root_numpy.fill_hist(h_etaphi_ecal,ECAL_cyl[:,[0,1]])
        root_numpy.fill_hist(h_etaz_ecal,ECAL_cyl[:,[0,2]])
        root_numpy.fill_hist(h_phiz_ecal,ECAL_cyl[:,[1,2]])
        root_numpy.fill_hist(h_etaphiz_ecal,ECAL_cyl[:,[0,1,2]])

        root_numpy.fill_hist(h_xy_ecal_tag,ECAL_tag[:,[0,1]])
        root_numpy.fill_hist(h_xz_ecal_tag,ECAL_tag[:,[2,0]])
        root_numpy.fill_hist(h_yz_ecal_tag,ECAL_tag[:,[2,1]])
        root_numpy.fill_hist(h_xyz_ecal_tag,ECAL_tag[:,[2,0,1]])

        root_numpy.fill_hist(h_etaphi_ecal_tag,ECAL_tag_cyl[:,[0,1]])
        root_numpy.fill_hist(h_etaz_ecal_tag,ECAL_tag_cyl[:,[0,2]])
        root_numpy.fill_hist(h_phiz_ecal_tag,ECAL_tag_cyl[:,[1,2]])
        root_numpy.fill_hist(h_etaphiz_ecal_tag,ECAL_tag_cyl[:,[0,1,2]])

        #CSC plot
        root_numpy.fill_hist(h_xy_csc,CSC[:,[0,1]])
        root_numpy.fill_hist(h_xz_csc,CSC[:,[2,0]])
        root_numpy.fill_hist(h_yz_csc,CSC[:,[2,1]])
        root_numpy.fill_hist(h_xyz_csc,CSC[:,[2,0,1]])
        root_numpy.fill_hist(h_etaphi_csc,CSC_cyl[:,[0,1]])

        ##3D line fit
        if DT_to_fit.size > 0:
            
            ## Method 1.
            ## Using linalg
            ## Pro: looks fast. Cons: wants zz as input
            
            x,y,z = DT_to_fit.T
            A_xz = np.vstack((x, np.ones(len(x)))).T
            m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]
            A_yz = np.vstack((y, np.ones(len(y)))).T
            m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]
            def lin(z):
                x = (z - c_xz)/m_xz
                y = (z - c_yz)/m_yz
                return x,y
            zz = np.linspace(-50.,50.,2000)
            ##zz = np.linspace(-5.,5.,5)
            xx,yy = lin(zz)
            print "\n"
            print "\n"
            print (("    LLS solution: m_xz %.6f, c_xz %.6f, m_yz %.6f, c_yz %.6f") % (m_xz,c_xz,m_yz,c_yz))
            print "\n"
            print "\n"
            ##print xx
            ##print yy
            ##print zz
            ##exit()
            DT_post_fit   = np.dstack((zz,np.dstack((xx,yy)))).reshape(xx.size,3)
            zz_ecal = ECAL_to_fit[:,2]
            xx_ecal,yy_ecal = lin(zz_ecal)
            ECAL_post_fit = np.dstack((zz_ecal,np.dstack((xx_ecal,yy_ecal)))).reshape(xx_ecal.size,3)
            ECAL_diff = ECAL_to_fit - np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3)#np.divide( (ECAL_to_fit - np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3))**2 , np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3))
            #print "Real points"
            #print ECAL_to_fit
            #print "Expected at the same z"
            #print -ECAL_diff + ECAL_to_fit
            #print "ECAL_diff as 1 d chi square"
            #print ECAL_diff
            ##exit()
            #print "ECAL_diff**2"
            #print ECAL_diff**2
            #print "ECAL_diff**2 sum"
            #print np.sum(ECAL_diff**2)
            #print "ECAL_diff**2 sum axis 0"
            #print np.sum(ECAL_diff**2,axis=0)
            #print "ECAL_diff**2 sum axis -1"
            #print np.sum(ECAL_diff**2,axis=-1)
            #print "sqrt ECAL_diff**2 sum axis -1"
            #print np.sqrt(np.sum(ECAL_diff**2,axis=-1))
            #print "media distanza euclidea"
            #print np.mean(np.sqrt(np.sum(ECAL_diff**2,axis=-1)))
            ssq = np.mean(np.sqrt(np.sum(ECAL_diff**2,axis=-1)))
            ##ssq = np.sum(ECAL_diff**2)/ECAL_diff.shape[0]
            #print "n. points", ECAL_diff.shape[0]
            #print ssq
            #print "x ecal"
            #print ECAL_to_fit[:,0]
            #print xx_ecal
            #print "y ecal"
            #print ECAL_to_fit[:,1]
            #print yy_ecal
            #print "z ecal"
            #print ECAL_to_fit[:,2]
            #print zz_ecal

            '''
            ## Method 2.
            ## Using linear regression
            ## Pro: can take any (x,y), con: does not give a line but exactly the points
            
            reg = LinearRegression().fit(DT_to_fit[:,[0,1]], DT_to_fit[:,2].flatten())
            dim = DT_to_fit[:,[0,1]].shape[0]
            z_pred = reg.intercept_ + np.matmul( DT_to_fit[:,[0,1]] , reg.coef_.reshape(-1,1)).reshape(dim)
            z_pred_ECAL = reg.intercept_ + np.matmul( ECAL_to_fit[:,[0,1]] , reg.coef_.reshape(-1,1)).reshape(ECAL_to_fit[:,[0,1]].shape[0])
            print "ECAL_to_fit"
            print "x"
            print ECAL_to_fit[:,0]

            print "ECAL_to_fit"
            print "y"
            print ECAL_to_fit[:,1]

            print "ECAL_to_fit"
            print "z"
            print ECAL_to_fit[:,2]#.shape

            print "prediction: "
            print z_pred_ECAL#.shape

            print "again check, weird"
            print ECAL_to_fit[:,0][0]
            print ECAL_to_fit[:,1][0]
            print ECAL_to_fit[:,2][0]
            print z_pred_ECAL[0]
            
            print ECAL_to_fit[:,[0,1]]

            #print ECAL_to_fit[:,2]#.shape
            #print z_pred_ECAL#.shape
            #DT_post_fit = np.dstack(( np.dstack((z_pred,DT_to_fit[:,0])), DT_to_fit[:,1])).reshape(dim,3)
            DT_post_fit = np.dstack(( z_pred, np.dstack((DT_to_fit[:,0], DT_to_fit[:,1])) )).reshape(dim,3)
            ECAL_post_fit = np.dstack(( z_pred_ECAL, np.dstack((ECAL_to_fit[:,0], ECAL_to_fit[:,1])) )).reshape(ECAL_to_fit[:,[0,1]].shape[0],3)
            ##DT_post_fit = np.dstack(( np.dstack((DT_to_fit[:,0],DT_to_fit[:,1])), z_pred)).reshape(dim,3)
            #print DT_to_fit
            #exit()
            '''

            '''
            ## Method 3.
            datamean = np.mean(DT_to_fit,axis=0)
            #print datamean
            uu, dd, vv = np.linalg.svd(DT_to_fit - datamean)
            #print "uu: ", uu
            #print "dd: ", dd
            #print "vv: ", vv
            print datamean
            print ECAL_to_fit[:,0]
            prova = ECAL_to_fit[:,0]
            print prova[:,np.newaxis]
            print "grid?"
            print np.mgrid[-1:1:4j][:, np.newaxis]
            print "vv[0]? "
            print vv[0]
            linepts = vv[0] * np.mgrid[-1:1:4j][:, np.newaxis]
            print linepts.shape
            print linepts
            #exit()
            linepts = vv[0] * np.mgrid[-10:10:100j][:, np.newaxis]
            linepts+=datamean
            DT_post_fit = linepts.reshape(100,3)
            DT_post_fit = DT_post_fit[:,[2,0,1]]
            print DT_post_fit[0]
            ECAL_post_fit = (vv[0] * ECAL_to_fit[:,0][:,np.newaxis] + datamean).reshape(ECAL_to_fit[:,0].size,3)
            ECAL_post_fit = ECAL_post_fit[:,[2,0,1]]
            #ECAL_to_fit#np.array([]).reshape(1,3)
            print "\n"
            print DT_to_fit.shape
            print linepts.shape
            '''

            root_numpy.fill_hist(fit_dt_3d, DT_post_fit)
            root_numpy.fill_hist(fit_dt_xy, DT_post_fit[:,[1,2]])
            root_numpy.fill_hist(fit_dt_xz, DT_post_fit[:,[0,1]])
            root_numpy.fill_hist(fit_dt_yz, DT_post_fit[:,[0,2]])

            root_numpy.fill_hist(fit_ecal_3d, ECAL_post_fit)
            root_numpy.fill_hist(fit_ecal_xy, ECAL_post_fit[:,[1,2]])
            root_numpy.fill_hist(fit_ecal_xz, ECAL_post_fit[:,[0,1]])
            root_numpy.fill_hist(fit_ecal_yz, ECAL_post_fit[:,[0,2]])
        else:
            DT_post_fit = DT_to_fit
            ECAL_post_fit = np.array([])

            
        #Not indented --> push right
        #chain.Project("h_xy_dt", "DTSegments.y/100.:DTSegments.x/100.", "")
        #chain.Project("h_xy_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.x/100.", "")
        #chain.Project("h_xy_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.x/100.", "")
        #chain.Project("h_xyz_dt", "DTSegments.y/100.:DTSegments.x/100.:DTSegments.z/100.", "")
        #chain.Project("h_xyz_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.x/100.:skimmedEcalRecHitsAK4.z/100.", "")
        #chain.Project("h_xyz_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.x/100.:taggedEcalRecHitsAK4.z/100.", "")
        #chain.Project("h_xz_dt", "DTSegments.x/100.:DTSegments.z/100.", "")
        #chain.Project("h_xz_ecal", "skimmedEcalRecHitsAK4.x/100.:skimmedEcalRecHitsAK4.z/100.", "")
        #chain.Project("h_xz_ecal_tag", "taggedEcalRecHitsAK4.x/100.:taggedEcalRecHitsAK4.z/100.", "")
        #chain.Project("h_yz_dt", "DTSegments.y/100.:DTSegments.z/100.", "")
        #chain.Project("h_yz_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.z/100.", "")
        #chain.Project("h_yz_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.z/100.", "")


        #root_numpy.fill_hist(h_xy_dt_2,DT_2[:,[0,1]])
        #root_numpy.fill_hist(h_xz_dt_2,DT_2[:,[2,0]])
        #root_numpy.fill_hist(h_yz_dt_2,DT_2[:,[2,1]])
        #root_numpy.fill_hist(h_xyz_dt_2,DT_2[:,[2,0,1]])

        #root_numpy.fill_hist(h_xy_dt_3,DT_3[:,[0,1]])
        #root_numpy.fill_hist(h_xz_dt_3,DT_3[:,[2,0]])
        #root_numpy.fill_hist(h_yz_dt_3,DT_3[:,[2,1]])
        #root_numpy.fill_hist(h_xyz_dt_3,DT_3[:,[2,0,1]])
    
        h_xy_dt_sum.Add(h_xy_dt_0)
        h_xy_dt_sum.Add(h_xy_dt_1)
        h_yz_dt_sum.Add(h_yz_dt_0)
        h_yz_dt_sum.Add(h_yz_dt_1)
        h_xz_dt_sum.Add(h_xz_dt_0)
        h_xz_dt_sum.Add(h_xz_dt_1)

        h_etaphi_dt_sum.Add(h_etaphi_dt_0)
        h_etaphi_dt_sum.Add(h_etaphi_dt_1)
        h_etaz_dt_sum.Add(h_etaz_dt_0)
        h_etaz_dt_sum.Add(h_etaz_dt_1)
        h_phiz_dt_sum.Add(h_phiz_dt_0)
        h_phiz_dt_sum.Add(h_phiz_dt_1)
        #h_xy_dt_sum.Add(h_xy_dt_2)
        #h_xy_dt_sum.Add(h_xy_dt_3)


        h_xy_dt.SetMarkerColor(4)
        h_xy_dt.SetMarkerStyle(20)
        h_xy_dt_n.SetMarkerColor(1)
        h_xy_dt_n.SetMarkerStyle(20)
        h_xy_dt_0.SetMarkerColor(4)
        h_xy_dt_0.SetMarkerStyle(20)
        h_xy_dt_1.SetMarkerColor(4)
        h_xy_dt_1.SetMarkerStyle(21)

        h_etaphi_dt.SetMarkerColor(4)
        h_etaphi_dt.SetMarkerStyle(20)
        h_etaphi_dt_n.SetMarkerColor(1)
        h_etaphi_dt_n.SetMarkerStyle(20)
        h_etaphi_dt_0.SetMarkerColor(4)
        h_etaphi_dt_0.SetMarkerStyle(20)
        h_etaphi_dt_1.SetMarkerColor(4)
        h_etaphi_dt_1.SetMarkerStyle(21)

        #h_etaphi_cos.SetMarkerColor(801)
        #h_etaphi_cos.SetMarkerStyle(20)
        #h_etaphi_cos1L.SetMarkerColor(6)
        #h_etaphi_cos1L.SetMarkerStyle(20)
        #h_xy_dt_2.SetMarkerColor(7)
        #h_xy_dt_2.SetMarkerStyle(20)
        #h_xy_dt_3.SetMarkerColor(7)
        #h_xy_dt_3.SetMarkerStyle(21)

        h_xy_ecal.SetMarkerColor(2)
        h_xy_ecal.SetMarkerStyle(20)
        h_xy_ecal_tag.SetMarkerColor(8)
        h_xy_ecal_tag.SetMarkerStyle(24)

        h_etaphi_ecal.SetMarkerColor(2)
        h_etaphi_ecal.SetMarkerStyle(20)
        h_etaphi_ecal_tag.SetMarkerColor(8)
        h_etaphi_ecal_tag.SetMarkerStyle(24)

        #Fit section
        fit_xy = TF1("lin_fit_xy","pol1",0.,0.)
        print "FIT xy"
        fit_dt_xy.Fit(fit_xy,"E")
        #h_xy_dt_sum.Fit(fit_xy,"E")
        h_xy_dt_sum.SetMarkerColor(0)

        fit_yz = TF1("lin_fit_yz","pol1",0.,0.)
        print "FIT yz"
        fit_dt_yz.Fit(fit_yz,"E")
        #h_yz_dt_sum.Fit(fit_yz,"E")
        h_yz_dt_sum.SetMarkerColor(0)

        fit_xz = TF1("lin_fit_xz","pol1",0.,0.)
        print "FIT xz"
        fit_dt_xz.Fit(fit_xz,"E")
        #h_xz_dt_sum.Fit(fit_xz,"E")
        h_xz_dt_sum.SetMarkerColor(0)

        h_xyz_dt.SetMarkerColor(4)
        h_xyz_dt.SetMarkerStyle(20)
        h_xyz_ecal.SetMarkerColor(2)
        h_xyz_ecal.SetMarkerStyle(20)
        h_xyz_ecal_tag.SetMarkerColor(8)
        h_xyz_ecal_tag.SetMarkerStyle(24)

        h_xyz_dt_n.SetMarkerColor(1)
        h_xyz_dt_n.SetMarkerStyle(20)
        h_xyz_dt_0.SetMarkerColor(4)
        h_xyz_dt_0.SetMarkerStyle(20)
        h_xyz_dt_1.SetMarkerColor(4)
        h_xyz_dt_1.SetMarkerStyle(21)
        h_xyz_dt_2.SetMarkerColor(7)
        h_xyz_dt_2.SetMarkerStyle(20)
        h_xyz_dt_3.SetMarkerColor(7)
        h_xyz_dt_3.SetMarkerStyle(21)

        h_xz_dt.SetMarkerColor(4)
        h_xz_dt.SetMarkerStyle(20)
        h_xz_ecal.SetMarkerColor(2)
        h_xz_ecal.SetMarkerStyle(20)
        h_xz_ecal_tag.SetMarkerColor(8)
        h_xz_ecal_tag.SetMarkerStyle(24)

        h_xz_dt_n.SetMarkerColor(1)
        h_xz_dt_n.SetMarkerStyle(20)
        h_xz_dt_0.SetMarkerColor(4)
        h_xz_dt_0.SetMarkerStyle(20)
        h_xz_dt_1.SetMarkerColor(4)
        h_xz_dt_1.SetMarkerStyle(21)
        h_xz_dt_2.SetMarkerColor(7)
        h_xz_dt_2.SetMarkerStyle(20)
        h_xz_dt_3.SetMarkerColor(7)
        h_xz_dt_3.SetMarkerStyle(21)

        
        #h_xz_dt.Fit("pol1")

        h_yz_dt.SetMarkerColor(4)
        h_yz_dt.SetMarkerStyle(20)
        h_yz_ecal.SetMarkerColor(2)
        h_yz_ecal.SetMarkerStyle(20)
        h_yz_ecal_tag.SetMarkerColor(8)
        h_yz_ecal_tag.SetMarkerStyle(24)

        h_yz_dt_n.SetMarkerColor(1)
        h_yz_dt_n.SetMarkerStyle(20)
        h_yz_dt_0.SetMarkerColor(4)
        h_yz_dt_0.SetMarkerStyle(20)
        h_yz_dt_1.SetMarkerColor(4)
        h_yz_dt_1.SetMarkerStyle(21)
        h_yz_dt_2.SetMarkerColor(7)
        h_yz_dt_2.SetMarkerStyle(20)
        h_yz_dt_3.SetMarkerColor(7)
        h_yz_dt_3.SetMarkerStyle(21)

        #h_yz_dt.Fit("pol1")

        leg = TLegend(0.75, 0.8, 1., 1.)
        leg = TLegend(0.75, 0.7, 1., 1.)
        leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
        leg.AddEntry(h_xy_dt,"DT segments","P")
        leg.AddEntry(h_xy_csc,"CSC segments","P")
        leg.AddEntry(h_xy_ecal,"EB rec hits","P")
        leg.AddEntry(h_xy_ecal_tag,"EB rec hits; tag jets","P")
        #leg.AddEntry(h_xy_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
        leg.AddEntry(fit_dt_xy,"3D linear fit","PF")
        chi_xy = h_xy_ecal_tag.Chisquare(fit_xy)
        #leg.AddEntry(fit_xy,("2D line chi_sq:%.2f"%chi_xy),"L")
        leg.AddEntry(fit_xy,"2D line projection","L")
        chi_test_xy = fit_ecal_xy.Chi2Test(h_xy_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_xy_ecal_tag)#Chi2Test(h_xy_ecal_tag, "CHI2/NDF")#square(fit_xy)
        #leg.AddEntry(h_xy_ecal_tag,("2D chi_test:%.5f"%chi_test_xy),"L")
        leg.AddEntry(h_xy_dt_n,"DBSCAN noise: "+str(n_noise_),"P")

        can_xy = TCanvas("can_xy","can_xy",900,800)
        can_xy.cd()
        can_xy.SetGrid()
        can_xy.SetRightMargin(0.05)
        #can_xy.SetLogy()
        #h_xy_dt.Draw("")

        fit_dt_xy.GetXaxis().SetTitle("x (m)")
        fit_dt_xy.GetYaxis().SetTitle("y (m)")
        h_xy_dt_sum.GetXaxis().SetTitle("x (m)")
        h_xy_dt_sum.GetYaxis().SetTitle("y (m)")
        h_xy_dt_n.GetXaxis().SetTitle("x (m)")
        h_xy_dt_n.GetYaxis().SetTitle("y (m)")
        h_xy_dt.GetYaxis().SetTitleOffset(1.4)
        h_xy_ecal.GetXaxis().SetTitle("x (m)")
        h_xy_ecal.GetYaxis().SetTitle("y (m)")

        fit_dt_xy.SetLineColor(856)
        fit_dt_xy.SetMarkerColor(856)
        fit_dt_xy.SetMarkerStyle(20)
        fit_dt_xy.SetMarkerSize(0.4)#(0.4)
        fit_dt_xy.SetLineWidth(2)

        fit_ecal_xy.SetLineColor(801)
        fit_ecal_xy.SetMarkerColor(801)
        fit_ecal_xy.SetMarkerStyle(20)
        #fit_ecal_xy.SetMarkerSize(2.)#(0.4)
        fit_ecal_xy.SetLineWidth(2)

        fit_dt_xy.Draw("")
        #fit_ecal_xy.Draw("sames")
        h_xy_dt_sum.Draw("sames")
        h_xy_dt_n.Draw("sames")
        h_xy_dt_0.Draw("sames")
        h_xy_dt_1.Draw("sames")
        #h_xy_dt_2.Draw("sames")
        #h_xy_dt_3.Draw("sames")
        h_xy_ecal.Draw("sames")
        h_xy_ecal_tag.Draw("sames")
        #fit_xy.Draw("sames,L")

        h_xy_csc.SetMarkerStyle(20)
        h_xy_csc.SetMarkerColor(800)
        h_xy_csc.Draw("sames")

        OUTSTRING = OUT
        if isMC[ev]:
            OUTSTRING+="mc_"
        else:
            OUTSTRING+="data_"
        OUTSTRING += "run_"+str(RunNumber[ev])+"_lumi_"+str(LumiNumber[ev])+"_ev_"+str(EventNumber[ev])        

        leg.Draw()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        latex.DrawLatex(0.36, 0.96, "Simulation")
        can_xy.Update()
        can_xy.Print(OUTSTRING+'_xy.png')
        can_xy.Print(OUTSTRING+'_xy.pdf')
        can_xy.Close()
        h_xy_dt.Delete()
        h_xy_dt_sum.Delete()
        h_xy_dt_n.Delete()
        h_xy_dt_0.Delete()
        h_xy_dt_1.Delete()
        h_xy_dt_2.Delete()
        h_xy_dt_3.Delete()
        fit_xy.Delete()
        h_xy_ecal.Delete()
        h_xy_ecal_tag.Delete()
        leg.Delete()


        ##etaphi
        leg = TLegend(0.75, 0.7, 1., 1.)
        leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
        leg.AddEntry(h_etaphi_dt,"DT segments","P")
        leg.AddEntry(h_etaphi_csc,"CSC segments","P")
        leg.AddEntry(h_etaphi_ecal,"EB rec hits","P")
        leg.AddEntry(h_etaphi_ecal_tag,"EB rec hits; tag jets","P")
        #leg.AddEntry(h_etaphi_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
        leg.AddEntry(h_etaphi_dt_n,"DBSCAN noise: "+str(n_noise_),"P")

        can_etaphi = TCanvas("can_etaphi","can_etaphi",900,800)
        can_etaphi.cd()
        can_etaphi.SetGrid()
        can_etaphi.SetRightMargin(0.05)
        #can_etaphi.SetLogy()
        #h_etaphi_dt.Draw("")

        h_etaphi_dt_sum.GetXaxis().SetTitle("#eta")
        h_etaphi_dt_sum.GetYaxis().SetTitle("#phi")

        h_etaphi_dt_sum.Draw("sames")
        h_etaphi_dt_n.Draw("sames")
        h_etaphi_dt_0.Draw("sames")
        h_etaphi_dt_1.Draw("sames")
        #h_etaphi_dt_2.Draw("sames")
        #h_etaphi_dt_3.Draw("sames")
        h_etaphi_ecal.Draw("sames")
        h_etaphi_ecal_tag.Draw("sames")
        #h_etaphi_cos.Draw("sames")
        #h_etaphi_cos1L.Draw("sames")
        #fit_etaphi.Draw("sames,L")
        h_etaphi_csc.SetMarkerStyle(20)
        h_etaphi_csc.SetMarkerColor(800)
        h_etaphi_csc.Draw("sames")


        OUTSTRING = OUT
        if isMC[ev]:
            OUTSTRING+="mc_"
        else:
            OUTSTRING+="data_"
        OUTSTRING += "run_"+str(RunNumber[ev])+"_lumi_"+str(LumiNumber[ev])+"_ev_"+str(EventNumber[ev])        

        leg.Draw()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        latex.DrawLatex(0.36, 0.96, "Simulation")
        can_etaphi.Update()
        can_etaphi.Print(OUTSTRING+'_etaphi.png')
        can_etaphi.Print(OUTSTRING+'_etaphi.pdf')
        can_etaphi.Close()
        h_etaphi_dt.Delete()
        h_etaphi_dt_sum.Delete()
        h_etaphi_dt_n.Delete()
        h_etaphi_dt_0.Delete()
        h_etaphi_dt_1.Delete()
        h_etaphi_dt_2.Delete()
        h_etaphi_dt_3.Delete()
        h_etaphi_ecal.Delete()
        h_etaphi_ecal_tag.Delete()
        leg.Delete()

        ##etaphi

        can_xz = TCanvas("can_xz","can_xz",900,800)
        can_xz.cd()
        can_xz.SetGrid()
        can_xz.SetRightMargin(0.05)
        #can_xz.SetLogy()
        #h_xz_dt.Draw("")

        h_xz_dt_sum.GetXaxis().SetTitle("z (m)")
        h_xz_dt_sum.GetYaxis().SetTitle("x (m)")
        h_xz_dt_n.GetXaxis().SetTitle("z (m)")
        h_xz_dt_n.GetYaxis().SetTitle("x (m)")
        h_xz_dt.GetYaxis().SetTitleOffset(1.4)
        h_xz_ecal.GetXaxis().SetTitle("z (m)")
        h_xz_ecal.GetYaxis().SetTitle("x (m)")

        h_xz_dt_sum.Draw("")
        h_xz_dt_n.Draw("sames")
        h_xz_dt_0.Draw("sames")
        h_xz_dt_1.Draw("sames")
        #h_xz_dt_2.Draw("sames")
        #h_xz_dt_3.Draw("sames")
        h_xz_ecal.Draw("sames")
        h_xz_ecal_tag.Draw("sames")
        #fit_xz.Draw("sames,L")
        fit_dt_xz.SetLineColor(856)
        fit_dt_xz.SetMarkerColor(856)
        fit_dt_xz.SetMarkerStyle(20)
        fit_dt_xz.SetMarkerSize(0.4)
        fit_dt_xz.SetLineWidth(2)
        fit_ecal_xz.SetLineColor(801)
        fit_ecal_xz.SetMarkerColor(801)
        fit_ecal_xz.SetMarkerStyle(20)
        #fit_ecal_xz.SetMarkerSize(2.)#(0.4)
        fit_ecal_xz.SetLineWidth(2)

        fit_dt_xz.Draw("sames,PL")
        #fit_ecal_xz.Draw("sames,PL")

        h_xz_csc.SetMarkerStyle(20)
        h_xz_csc.SetMarkerColor(800)
        h_xz_csc.Draw("sames")

        h_xz_dt.GetXaxis().SetTitle("z (m)")
        h_xz_dt.GetYaxis().SetTitle("x (m)")
        h_xz_ecal.GetXaxis().SetTitle("z (m)")
        h_xz_ecal.GetYaxis().SetTitle("x (m)")
        h_xz_dt.GetYaxis().SetTitleOffset(1.4)
        leg = TLegend(0.75, 0.8, 1., 1.)
        leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
        #leg.SetHeader(d)
        leg.AddEntry(h_xz_dt,"DT segments","P")
        leg.AddEntry(h_xz_csc,"CSC segments","P")
        leg.AddEntry(h_xz_ecal,"EB rec hits","P")
        leg.AddEntry(h_xz_ecal_tag,"EB rec hits; tag jets","P")
        #leg.AddEntry(fit_dt_xz,("3D lin. fit; ssq:%.3f"%ssq),"PF")
        leg.AddEntry(fit_dt_xz,"3D linear fit","PF")
        chi_xz = h_xz_ecal_tag.Chisquare(fit_xz)
        leg.AddEntry(fit_xz,"2D line projection","L")
        #leg.AddEntry(fit_xz,("2D chi_sq:%.2f"%chi_xz),"L")
        chi_test_xz = fit_ecal_xz.Chi2Test(h_xz_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_xz_ecal_tag)#Chi2Test(h_xz_ecal_tag, "CHI2/NDF")#square(fit_xz)
        leg.AddEntry(h_xz_ecal_tag,("2D chi_test:%.5f"%chi_test_xz),"L")
        leg.AddEntry(h_xz_dt_n,"DBSCAN noise: "+str(n_noise_),"P")
        leg.Draw()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        latex.DrawLatex(0.36, 0.96, "Simulation")
        can_xz.Update()
        can_xz.Print(OUTSTRING+'_xz.png')
        can_xz.Print(OUTSTRING+'_xz.pdf')
        can_xz.Close()
        h_xz_dt.Delete()
        h_xz_ecal.Delete()
        h_xz_ecal_tag.Delete()
        leg.Delete()

        can_yz = TCanvas("can_yz","can_yz",900,800)
        can_yz.cd()
        can_yz.SetGrid()
        can_yz.SetRightMargin(0.05)
        #can_yz.SetLogy()
        #h_yz_dt.Draw("")

        h_yz_dt_sum.GetXaxis().SetTitle("z (m)")
        h_yz_dt_sum.GetYaxis().SetTitle("y (m)")
        h_yz_dt_n.GetXaxis().SetTitle("z (m)")
        h_yz_dt_n.GetYaxis().SetTitle("y (m)")
        h_yz_ecal.GetXaxis().SetTitle("z (m)")
        h_yz_ecal.GetYaxis().SetTitle("y (m)")

        h_yz_dt_sum.Draw("")
        h_yz_dt_n.Draw("sames")
        h_yz_dt_0.Draw("sames")
        h_yz_dt_1.Draw("sames")
        #h_yz_dt_2.Draw("sames")
        #h_yz_dt_3.Draw("sames")
        h_yz_ecal.Draw("sames")
        h_yz_ecal_tag.Draw("sames")

        h_yz_csc.SetMarkerStyle(20)
        h_yz_csc.SetMarkerColor(800)
        h_yz_csc.Draw("sames")
        #fit_yz.Draw("sames,L")
        fit_dt_yz.SetLineColor(856)
        fit_dt_yz.SetMarkerColor(856)
        fit_dt_yz.SetMarkerStyle(20)
        fit_dt_yz.SetMarkerSize(0.4)
        fit_dt_yz.SetLineWidth(2)
        fit_dt_yz.Draw("sames,PL")

        fit_ecal_yz.SetLineColor(801)
        fit_ecal_yz.SetMarkerColor(801)
        fit_ecal_yz.SetMarkerStyle(20)
        #fit_ecal_yz.SetMarkerSize(2.)#(0.4)
        fit_ecal_yz.SetLineWidth(2)
        #fit_ecal_yz.Draw("sames,PL")

        h_yz_dt.GetXaxis().SetTitle("z (m)")
        h_yz_dt.GetYaxis().SetTitle("y (m)")
        h_yz_ecal.GetXaxis().SetTitle("z (m)")
        h_yz_ecal.GetYaxis().SetTitle("y (m)")
        h_yz_dt.GetYaxis().SetTitleOffset(1.4)
        leg = TLegend(0.75, 0.8, 1., 1.)
        leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
        #leg.SetHeader(d)
        leg.AddEntry(h_yz_dt,"DT segments","P")
        leg.AddEntry(h_yz_csc,"CSC segments","P")
        leg.AddEntry(h_yz_ecal,"EB rec hits","P")
        leg.AddEntry(h_yz_ecal_tag,"EB rec hits; tag jets","P")
        leg.AddEntry(fit_dt_yz,"3D linear fit","PF")
        #leg.AddEntry(fit_dt_yz,("3D lin. fit; ssq:%.3f"%ssq),"PF")
        chi_yz = h_yz_ecal_tag.Chisquare(fit_yz)
        #leg.AddEntry(fit_yz,("2D chi_sq:%.2f"%chi_yz),"L")
        leg.AddEntry(fit_yz,"2D line projection","L")
        chi_test_yz = fit_ecal_yz.Chi2Test(h_yz_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_yz_ecal_tag)#Chi2Test(h_yz_ecal_tag, "CHI2/NDF")#square(fit_xy)
        leg.AddEntry(h_yz_ecal_tag,("2D chi_test:%.5f"%chi_test_yz),"L")
        leg.AddEntry(h_yz_dt_n,"DBSCAN noise: "+str(n_noise_),"P")
        leg.Draw()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        latex.DrawLatex(0.36, 0.96, "Simulation")
        can_yz.Update()
        can_yz.Print(OUTSTRING+'_yz.png')
        can_yz.Print(OUTSTRING+'_yz.pdf')
        can_yz.Close()
        h_yz_dt.Delete()
        h_yz_ecal.Delete()
        h_yz_ecal_tag.Delete()
        leg.Delete()

        '''
        print DT_to_fit
        datamean = np.mean(DT_to_fit,axis=0)
        print datamean
        uu, dd, vv = np.linalg.svd(DT_to_fit - datamean)
        print "uu: ", uu
        print "dd: ", dd
        print "vv: ", vv
        linepts = vv[0] * np.mgrid[-10:10:2j][:, np.newaxis]
        linepts+=datamean
        print linepts
        '''
        
        '''
        reg = LinearRegression().fit(DT_to_fit[:,[0,1]], DT_to_fit[:,2].flatten())
        dim = DT_to_fit[:,[0,1]].shape[0]
        z_pred = reg.intercept_ + np.matmul( DT_to_fit[:,[0,1]] , reg.coef_.reshape(-1,1)).reshape(dim)
        #print DT_to_fit[:,[0,1]].shape
        print DT_to_fit[:,2]#.shape
        print z_pred#.shape
        print "lol"
        DT_post_fit = np.dstack(( np.dstack((z_pred,DT_to_fit[:,0])), DT_to_fit[:,1])).reshape(dim,3)
        print DT_to_fit
        print DT_post_fit
        root_numpy.fill_hist(fit_dt_3d, DT_post_fit)
        '''

        can_xyz = TCanvas("can_xyz","can_xyz",900,800)
        can_xyz.cd()
        can_xyz.SetGrid()
        #can_xyz.SetRightMargin(0.05)
        #can_xyz.SetLogy()
        #h_xyz_dt.Draw("")
        fit_dt_3d.SetLineColor(856)
        fit_dt_3d.SetMarkerColor(856)
        fit_dt_3d.SetMarkerStyle(20)
        fit_dt_3d.SetMarkerSize(0.4)
        fit_dt_3d.SetLineWidth(2)
        fit_ecal_3d.SetLineColor(801)
        fit_ecal_3d.SetMarkerColor(801)
        fit_ecal_3d.SetMarkerStyle(20)
        #fit_ecal_3d.SetMarkerSize(2.)#(0.4)
        fit_ecal_3d.SetLineWidth(2)

        fit_dt_3d.Draw("")
        #fit_ecal_3d.Draw("sames")
        h_xyz_dt_n.Draw("sames")
        h_xyz_dt_0.Draw("sames")
        h_xyz_dt_1.Draw("sames")
        #h_xyz_dt_2.Draw("sames")
        #h_xyz_dt_3.Draw("sames")
        h_xyz_ecal.Draw("sames")
        h_xyz_ecal_tag.Draw("sames")

        h_xyz_csc.SetMarkerStyle(20)
        h_xyz_csc.SetMarkerColor(800)
        h_xyz_csc.Draw("sames")

        fit_dt_3d.GetXaxis().SetTitle("z (m)")
        fit_dt_3d.GetYaxis().SetTitle("x (m)")
        fit_dt_3d.GetZaxis().SetTitle("y (m)")
        h_xyz_ecal.GetXaxis().SetTitle("z (m)")
        h_xyz_ecal.GetYaxis().SetTitle("x (m)")
        h_xyz_ecal.GetZaxis().SetTitle("y (m)")
        fit_dt_3d.GetXaxis().SetTitleOffset(1.4)
        fit_dt_3d.GetYaxis().SetTitleOffset(1.8)
        fit_dt_3d.GetZaxis().SetTitleOffset(1.4)
        leg = TLegend(0.75, 0.7, 1., 1.)
        leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
        #leg.SetHeader(d)
        leg.AddEntry(h_xyz_dt,"DT segments","P")
        leg.AddEntry(h_xyz_csc,"CSC segments","P")
        leg.AddEntry(h_xyz_ecal,"EB rec hits","P")
        leg.AddEntry(h_xyz_ecal_tag,"EB rec hits; tag jets","P")
        #leg.AddEntry(h_xyz_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
        #leg.AddEntry(h_xyz_dt_n,"n. noise: "+str(n_noise_),"P")
        #leg.AddEntry(fit_dt_3d,("3D lin. fit; ssq:%.3f"%ssq),"PF")
        leg.AddEntry(fit_dt_3d,"3D linear fit","PF")
        #leg_string = (("2D chi_sq:(%.2f,%.2f,%.2f)")%(chi_xy,chi_xz,chi_yz))
        #leg.AddEntry(fit_dt_3d,str(leg_string),"L")
        leg.AddEntry(h_xyz_dt_n,"DBSCAN noise: "+str(n_noise_),"P")
        leg.Draw()
        latex = TLatex()
        latex.SetNDC()
        latex.SetTextAlign(33)
        latex.SetTextSize(0.04)
        latex.SetTextFont(62)
        latex.DrawLatex(0.20, 0.96, "CMS")
        latex.SetTextFont(52)
        latex.DrawLatex(0.36, 0.96, "Simulation")
        can_xyz.Update()
        can_xyz.Print(OUTSTRING+'_xyz.png')
        can_xyz.Print(OUTSTRING+'_xyz.pdf')
        can_xyz.Close()
        h_xyz_dt.Delete()
        h_xyz_ecal.Delete()
        h_xyz_ecal_tag.Delete()
        #raw_input("Press Enter to continue...")
        leg.Delete()
        h_yz_dt_n.Delete()
        h_yz_dt_0.Delete()
        h_yz_dt_1.Delete()
        h_yz_dt_2.Delete()
        h_yz_dt_3.Delete()
        h_yz_dt_sum.Delete()
        h_xz_dt_n.Delete()
        h_xz_dt_0.Delete()
        h_xz_dt_1.Delete()
        h_xz_dt_2.Delete()
        h_xz_dt_3.Delete()
        h_xz_dt_sum.Delete()
        h_xyz_dt_n.Delete()
        h_xyz_dt_0.Delete()
        h_xyz_dt_1.Delete()
        h_xyz_dt_2.Delete()
        h_xyz_dt_3.Delete()
        h_xyz_dt_sum.Delete()
        fit_dt_3d.Delete()
        fit_dt_xy.Delete()
        fit_dt_xz.Delete()
        fit_dt_yz.Delete()
        #exit()




def altro():
    #This is good at a per event level
    DT = np.dstack((DT_x,DT_y,DT_z)).reshape(-1,3)
    #print DT
    
    #reconstructs 2 legs
    #3 works, 2 works, 1.5 more stringent
    dbscan = DBSCAN(eps=1.4,min_samples=3).fit(DT)

    #reconstructs 1 single cosmic
    #dbscan = DBSCAN(eps=1.4,min_samples=3).fit(DT)
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print "First DB scan: get the 2 clusters"
    print "labels", labels
    print "n_clusters_", n_clusters_
    print "n_noise_", n_noise_
    m_noise = (labels==-1)

    m = {}
    m[0] = (labels==0)
    m[1] = (labels==1)
    m[2] = (labels==2)
    m[3] = (labels==3)
    m_cl_0 = (labels==0)
    m_cl_1 = (labels==1)
    m_cl_2 = (labels==2)
    m_cl_3 = (labels==3)


    DT_cl = {}
    DT_n = DT[m_noise]

    for k in m.keys():
        DT_cl[k] = DT[m[k]]

    if n_clusters_>2:
        n_pairs = int(n_clusters_/2)
        if (n_clusters_ % 2 )!=0:
            print "Odd number of DT clusters, strange..."
            exit()

        mean_x = {}
        mean_y = {}
        mean_z = {}
        for k in m.keys():
            mean_x[k] = np.sum(DT_cl[k][:,0])/np.size( DT_cl[k][:,0] )
            mean_y[k] = np.sum(DT_cl[k][:,1])/np.size( DT_cl[k][:,1] )
            mean_z[k] = np.sum(DT_cl[k][:,2])/np.size( DT_cl[k][:,2] )

        mean_ECAL_tag_z = np.sum(ECAL_tag_z.flatten())/np.size(ECAL_tag_z)
        print "mean_ECAL_tag_z: ", mean_ECAL_tag_z

        pairs = {}
        #for n in range(n_pairs):
        #    pairs[n] = np.array([-1,1])

        #This handles two cosmics. If there are more, we'll have to deal with this somehow...
        dz_DT = 1000.
        dz_ECAL = 1000.
        ch_k1 = -1
        ch_k2 = -1
        for k1 in range(len(m.keys())):
            for k2 in range(len(m.keys())):
                if k2>k1 and mean_x[k1]*mean_x[k2]<0 and mean_y[k1]*mean_y[k2]<0:
                    #opposite emispheres
                    #print "pair: ", k1, k2
                    #print "z average: ", mean_z[k1], mean_z[k2]
                    tmp_z = abs(mean_z[k1] - mean_z[k2])
                    dz_DT = min(dz_DT,tmp_z)

                    tmp_ECAL = abs((mean_z[k1]+mean_z[k2])/2. - mean_ECAL_tag_z)
                    dz_ECAL = min(dz_ECAL,tmp_ECAL)

                    if dz_DT==tmp_z and dz_ECAL==tmp_ECAL:
                        ch_k1 = k1
                        ch_k2 = k2
                        #print "I choose you: ", k1, k2

        #print "final pair: ", ch_k1, ch_k2
        #print DT_cl[0]
        #print DT_cl[1]
        #print "Overwrite DT dict"
        tmp_DT_cl_0 = DT_cl[ch_k1]
        tmp_DT_cl_1 = DT_cl[ch_k2]
        new_keys = m.keys()
        new_keys = np.setdiff1d(np.array(m.keys()),np.array([ch_k1,ch_k2]))
        #new_keys = np.delete(new_keys,ch_k2,-1)
        #print "new keys", new_keys
        #exit()
        tmp_cl_discarded = {}
        for a in range(len(new_keys)):
            tmp_cl_discarded[a] = DT_cl[new_keys[a]]

        DT_cl[0] = tmp_DT_cl_0
        DT_cl[1] = tmp_DT_cl_1

        #Add discarded cosmics to noise histogram
        for a in range(len(new_keys)):
            DT_n = np.vstack((DT_n,tmp_cl_discarded[a]))
            #DT_n.vstack(tmp_cl_discarded[a])
            #DT_cl[2+a] = tmp_cl_discarded[a]

    chain.Project("h_xy_dt", "DTSegments.y/100.:DTSegments.x/100.", "")
    chain.Project("h_xy_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.x/100.", "")
    chain.Project("h_xy_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.x/100.", "")
    chain.Project("h_xyz_dt", "DTSegments.y/100.:DTSegments.x/100.:DTSegments.z/100.", "")
    chain.Project("h_xyz_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.x/100.:skimmedEcalRecHitsAK4.z/100.", "")
    chain.Project("h_xyz_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.x/100.:taggedEcalRecHitsAK4.z/100.", "")
    chain.Project("h_xz_dt", "DTSegments.x/100.:DTSegments.z/100.", "")
    chain.Project("h_xz_ecal", "skimmedEcalRecHitsAK4.x/100.:skimmedEcalRecHitsAK4.z/100.", "")
    chain.Project("h_xz_ecal_tag", "taggedEcalRecHitsAK4.x/100.:taggedEcalRecHitsAK4.z/100.", "")
    chain.Project("h_yz_dt", "DTSegments.y/100.:DTSegments.z/100.", "")
    chain.Project("h_yz_ecal", "skimmedEcalRecHitsAK4.y/100.:skimmedEcalRecHitsAK4.z/100.", "")
    chain.Project("h_yz_ecal_tag", "taggedEcalRecHitsAK4.y/100.:taggedEcalRecHitsAK4.z/100.", "")
    #chain.Project("rh", "RecHigMass", "")

    root_numpy.fill_hist(h_xy_dt_n,DT_n[:,[0,1]])
    root_numpy.fill_hist(h_xz_dt_n,DT_n[:,[2,0]])
    root_numpy.fill_hist(h_yz_dt_n,DT_n[:,[2,1]])
    root_numpy.fill_hist(h_xyz_dt_n,DT_n[:,[2,0,1]])

    root_numpy.fill_hist(h_xy_dt_0,DT_cl[0][:,[0,1]])
    root_numpy.fill_hist(h_xz_dt_0,DT_cl[0][:,[2,0]])
    root_numpy.fill_hist(h_yz_dt_0,DT_cl[0][:,[2,1]])
    root_numpy.fill_hist(h_xyz_dt_0,DT_cl[0][:,[2,0,1]])

    root_numpy.fill_hist(h_xy_dt_1,DT_cl[1][:,[0,1]])
    root_numpy.fill_hist(h_xz_dt_1,DT_cl[1][:,[2,0]])
    root_numpy.fill_hist(h_yz_dt_1,DT_cl[1][:,[2,1]])
    root_numpy.fill_hist(h_xyz_dt_1,DT_cl[1][:,[2,0,1]])

    #root_numpy.fill_hist(h_xy_dt_2,DT_2[:,[0,1]])
    #root_numpy.fill_hist(h_xz_dt_2,DT_2[:,[2,0]])
    #root_numpy.fill_hist(h_yz_dt_2,DT_2[:,[2,1]])
    #root_numpy.fill_hist(h_xyz_dt_2,DT_2[:,[2,0,1]])

    #root_numpy.fill_hist(h_xy_dt_3,DT_3[:,[0,1]])
    #root_numpy.fill_hist(h_xz_dt_3,DT_3[:,[2,0]])
    #root_numpy.fill_hist(h_yz_dt_3,DT_3[:,[2,1]])
    #root_numpy.fill_hist(h_xyz_dt_3,DT_3[:,[2,0,1]])
    
    h_xy_dt_sum.Add(h_xy_dt_0)
    h_xy_dt_sum.Add(h_xy_dt_1)
    #h_xy_dt_sum.Add(h_xy_dt_2)
    #h_xy_dt_sum.Add(h_xy_dt_3)


    h_xy_dt.SetMarkerColor(4)
    h_xy_dt.SetMarkerStyle(20)

    h_xy_dt_n.SetMarkerColor(1)
    h_xy_dt_n.SetMarkerStyle(20)
    h_xy_dt_0.SetMarkerColor(4)
    h_xy_dt_0.SetMarkerStyle(20)
    h_xy_dt_1.SetMarkerColor(4)
    h_xy_dt_1.SetMarkerStyle(21)
    h_xy_dt_2.SetMarkerColor(7)
    h_xy_dt_2.SetMarkerStyle(20)
    h_xy_dt_3.SetMarkerColor(7)
    h_xy_dt_3.SetMarkerStyle(21)

    h_xy_ecal.SetMarkerColor(2)
    h_xy_ecal.SetMarkerStyle(20)
    h_xy_ecal_tag.SetMarkerColor(8)
    h_xy_ecal_tag.SetMarkerStyle(24)

    fit_xy = TF1("lin_fit","pol1",0.,0.)

    h_xy_dt_sum.Fit(fit_xy,"E")

    h_xy_dt_sum.SetMarkerColor(0)

    h_xyz_dt.SetMarkerColor(4)
    h_xyz_dt.SetMarkerStyle(20)
    h_xyz_ecal.SetMarkerColor(2)
    h_xyz_ecal.SetMarkerStyle(20)
    h_xyz_ecal_tag.SetMarkerColor(8)
    h_xyz_ecal_tag.SetMarkerStyle(24)

    h_xyz_dt_n.SetMarkerColor(1)
    h_xyz_dt_n.SetMarkerStyle(20)
    h_xyz_dt_0.SetMarkerColor(4)
    h_xyz_dt_0.SetMarkerStyle(20)
    h_xyz_dt_1.SetMarkerColor(4)
    h_xyz_dt_1.SetMarkerStyle(21)
    h_xyz_dt_2.SetMarkerColor(7)
    h_xyz_dt_2.SetMarkerStyle(20)
    h_xyz_dt_3.SetMarkerColor(7)
    h_xyz_dt_3.SetMarkerStyle(21)

    h_xz_dt.SetMarkerColor(4)
    h_xz_dt.SetMarkerStyle(20)
    h_xz_ecal.SetMarkerColor(2)
    h_xz_ecal.SetMarkerStyle(20)
    h_xz_ecal_tag.SetMarkerColor(8)
    h_xz_ecal_tag.SetMarkerStyle(24)

    h_xz_dt_n.SetMarkerColor(1)
    h_xz_dt_n.SetMarkerStyle(20)
    h_xz_dt_0.SetMarkerColor(4)
    h_xz_dt_0.SetMarkerStyle(20)
    h_xz_dt_1.SetMarkerColor(4)
    h_xz_dt_1.SetMarkerStyle(21)
    h_xz_dt_2.SetMarkerColor(7)
    h_xz_dt_2.SetMarkerStyle(20)
    h_xz_dt_3.SetMarkerColor(7)
    h_xz_dt_3.SetMarkerStyle(21)


    #h_xz_dt.Fit("pol1")

    h_yz_dt.SetMarkerColor(4)
    h_yz_dt.SetMarkerStyle(20)
    h_yz_ecal.SetMarkerColor(2)
    h_yz_ecal.SetMarkerStyle(20)
    h_yz_ecal_tag.SetMarkerColor(8)
    h_yz_ecal_tag.SetMarkerStyle(24)

    h_yz_dt_n.SetMarkerColor(1)
    h_yz_dt_n.SetMarkerStyle(20)
    h_yz_dt_0.SetMarkerColor(4)
    h_yz_dt_0.SetMarkerStyle(20)
    h_yz_dt_1.SetMarkerColor(4)
    h_yz_dt_1.SetMarkerStyle(21)
    h_yz_dt_2.SetMarkerColor(7)
    h_yz_dt_2.SetMarkerStyle(20)
    h_yz_dt_3.SetMarkerColor(7)
    h_yz_dt_3.SetMarkerStyle(21)

    #h_yz_dt.Fit("pol1")

    leg = TLegend(0.75, 0.8, 1., 1.)
    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.SetHeader("nCosmics: "+str(nCosmics[0])+"; one leg: "+str(nCosmicsOneLeg[0]))
    leg.AddEntry(h_xy_dt,"DT segments","P")
    leg.AddEntry(h_xy_ecal,"EB rec hits","P")
    leg.AddEntry(h_xy_ecal_tag,"EB rec hits; tag jets","P")
    leg.AddEntry(h_xy_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
    leg.AddEntry(h_xy_dt_n,"n. noise: "+str(n_noise_),"P")

    can_xy = TCanvas("can_xy","can_xy",900,800)
    can_xy.cd()
    can_xy.SetRightMargin(0.05)
    #can_xy.SetLogy()
    #h_xy_dt.Draw("")
    h_xy_dt_sum.Draw("")
    h_xy_dt_n.Draw("sames")
    h_xy_dt_0.Draw("sames")
    h_xy_dt_1.Draw("sames")
    #h_xy_dt_2.Draw("sames")
    #h_xy_dt_3.Draw("sames")
    h_xy_ecal.Draw("sames")
    h_xy_ecal_tag.Draw("sames")
    fit_xy.Draw("sames,L")

    h_xy_dt.GetXaxis().SetTitle("x (m)")
    h_xy_dt.GetYaxis().SetTitle("y (m)")
    h_xy_dt.GetYaxis().SetTitleOffset(1.4)
    h_xy_ecal.GetXaxis().SetTitle("x (m)")
    h_xy_ecal.GetYaxis().SetTitle("y (m)")

    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xy.Update()
    can_xy.Print(MAIN+d+'_xy.png')
    can_xy.Print(MAIN+d+'_xy.pdf')
    can_xy.Close()
    h_xy_dt.Delete()
    h_xy_dt_sum.Delete()
    h_xy_dt_n.Delete()
    h_xy_dt_0.Delete()
    h_xy_dt_1.Delete()
    h_xy_dt_2.Delete()
    h_xy_dt_3.Delete()
    fit_xy.Delete()
    h_xy_ecal.Delete()
    h_xy_ecal_tag.Delete()
    leg.Delete()


    can_xz = TCanvas("can_xz","can_xz",900,800)
    can_xz.cd()
    can_xz.SetRightMargin(0.05)
    #can_xz.SetLogy()
    #h_xz_dt.Draw("")
    h_xz_dt_n.Draw("")
    h_xz_dt_0.Draw("sames")
    h_xz_dt_1.Draw("sames")
    #h_xz_dt_2.Draw("sames")
    #h_xz_dt_3.Draw("sames")
    h_xz_ecal.Draw("sames")
    h_xz_ecal_tag.Draw("sames")
    h_xz_dt.GetXaxis().SetTitle("z (m)")
    h_xz_dt.GetYaxis().SetTitle("x (m)")
    h_xz_ecal.GetXaxis().SetTitle("z (m)")
    h_xz_ecal.GetYaxis().SetTitle("x (m)")
    h_xz_dt.GetYaxis().SetTitleOffset(1.4)
    leg = TLegend(0.75, 0.8, 1., 1.)
    leg.SetHeader("nCosmics: "+str(nCosmics[0])+"; one leg: "+str(nCosmicsOneLeg[0]))
    #leg.SetHeader(d)
    leg.AddEntry(h_xz_dt,"DT segments","P")
    leg.AddEntry(h_xz_ecal,"EB rec hits","P")
    leg.AddEntry(h_xz_ecal_tag,"EB rec hits; tag jets","P")
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xz.Update()
    can_xz.Print(MAIN+d+'_xz.png')
    can_xz.Print(MAIN+d+'_xz.pdf')
    can_xz.Close()
    h_xz_dt.Delete()
    h_xz_ecal.Delete()
    h_xz_ecal_tag.Delete()
    leg.Delete()

    can_yz = TCanvas("can_yz","can_yz",900,800)
    can_yz.cd()
    can_yz.SetRightMargin(0.05)
    #can_yz.SetLogy()
    #h_yz_dt.Draw("")
    h_yz_dt_n.Draw("")
    h_yz_dt_0.Draw("sames")
    h_yz_dt_1.Draw("sames")
    #h_yz_dt_2.Draw("sames")
    #h_yz_dt_3.Draw("sames")
    h_yz_ecal.Draw("sames")
    h_yz_ecal_tag.Draw("sames")
    h_yz_dt.GetXaxis().SetTitle("z (m)")
    h_yz_dt.GetYaxis().SetTitle("y (m)")
    h_yz_ecal.GetXaxis().SetTitle("z (m)")
    h_yz_ecal.GetYaxis().SetTitle("y (m)")
    h_yz_dt.GetYaxis().SetTitleOffset(1.4)
    leg = TLegend(0.75, 0.8, 1., 1.)
    leg.SetHeader("nCosmics: "+str(nCosmics[0])+"; one leg: "+str(nCosmicsOneLeg[0]))
    #leg.SetHeader(d)
    leg.AddEntry(h_yz_dt,"DT segments","P")
    leg.AddEntry(h_yz_ecal,"EB rec hits","P")
    leg.AddEntry(h_yz_ecal_tag,"EB rec hits; tag jets","P")
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_yz.Update()
    can_yz.Print(MAIN+d+'_yz.png')
    can_yz.Print(MAIN+d+'_yz.pdf')
    can_yz.Close()
    h_yz_dt.Delete()
    h_yz_ecal.Delete()
    h_yz_ecal_tag.Delete()
    leg.Delete()


    can_xyz = TCanvas("can_xyz","can_xyz",900,800)
    can_xyz.cd()
    #can_xyz.SetRightMargin(0.05)
    #can_xyz.SetLogy()
    #h_xyz_dt.Draw("")
    h_xyz_dt_n.Draw("")
    h_xyz_dt_0.Draw("sames")
    h_xyz_dt_1.Draw("sames")
    #h_xyz_dt_2.Draw("sames")
    #h_xyz_dt_3.Draw("sames")
    h_xyz_ecal.Draw("sames")
    h_xyz_ecal_tag.Draw("sames")
    h_xyz_dt.GetXaxis().SetTitle("z (m)")
    h_xyz_dt.GetYaxis().SetTitle("x (m)")
    h_xyz_dt.GetZaxis().SetTitle("y (m)")
    h_xyz_ecal.GetXaxis().SetTitle("z (m)")
    h_xyz_ecal.GetYaxis().SetTitle("x (m)")
    h_xyz_ecal.GetZaxis().SetTitle("y (m)")
    h_xyz_dt.GetXaxis().SetTitleOffset(1.4)
    h_xyz_dt.GetYaxis().SetTitleOffset(1.8)
    h_xyz_dt.GetZaxis().SetTitleOffset(1.4)
    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.SetHeader("nCosmics: "+str(nCosmics[0])+"; one leg: "+str(nCosmicsOneLeg[0]))
    #leg.SetHeader(d)
    leg.AddEntry(h_xyz_dt,"DT segments","P")
    leg.AddEntry(h_xyz_ecal,"EB rec hits","P")
    leg.AddEntry(h_xyz_ecal_tag,"EB rec hits; tag jets","P")
    leg.AddEntry(h_xyz_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
    leg.AddEntry(h_xyz_dt_n,"n. noise: "+str(n_noise_),"P")
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xyz.Update()
    can_xyz.Print(MAIN+d+'_xyz.png')
    can_xyz.Print(MAIN+d+'_xyz.pdf')
    can_xyz.Close()
    h_xyz_dt.Delete()
    h_xyz_ecal.Delete()
    h_xyz_ecal_tag.Delete()
    #raw_input("Press Enter to continue...")
    leg.Delete()
