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

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples
#from Analyzer.LLP2018.drawUtils import *

#import optparse
#usage = "usage: %prog [options]"
#parser = optparse.OptionParser(usage)
#parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
#(options, args) = parser.parse_args()
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = [
    #"skim_pickevents_275309_875415530",
    #"skim_pickevents_276811_244034371",#this fails the opposite emisphere requirement
    #"skim_pickevents_277096_153555117",#pure noise
    #"n3n2-n1-hbb-hbb_mh400_pl1000",
    #"n3n2-n1-hbb-hbb_mh300_pl1000",
    #"n3n2-n1-hbb-hbb_mh200_pl1000",
    #"n3n2-n1-hbb-hbb_mh127_pl1000",
    #"skim_pickevents_279029_387083289",
    #"skim_pickevents_279931_4073593513",
    #"skim_pickevents_302485_987393715",
    #"skim_pickevents_319910_457404349",
    #"test_signal",
    #"SUSY_mh400_pl1000",
    #"SUSY_mh300_pl1000",
    #"SUSY_mh200_pl1000",
    #"SUSY_mh150_pl1000",
    #"SUSY_mh127_pl1000",
    "HighMET",
    #"HighMETBH",
]
##MAIN = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/"
#MAIN = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/"
##MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_SR/"
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_SR_wrong_cosmic_veto/"
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2018_SR_bin_1_2/"
##OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_v3/"

OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_moar_2018/"
OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_bin_1_2_2018/"

OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_bin_1_2_2018__bin_1/"
#OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_BeamHalo_studies/BeamHaloEventDisplaysSmallDphi/"
#OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_BeamHalo_studies/Bin1EventDisplaysSmallDphi/"
#OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_BeamHalo_studies/BeamHaloEventDisplaysSmallDphiNarrowX/"
#OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_2018_BeamHalo_studies/Bin1EventDisplaysSmallDphiNarrowX/"

##OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_cosmic_veto_bin_1/"

def deltaPhi_np( a1, v2):
    a2 = np.multiply(np.ones(a1.size),v2)
    pi = np.multiply(np.ones(a1.size),math.pi)
    diff = np.abs(a1 - a2)
    mask = diff>pi
    M = np.ma.masked_array(data=diff, mask = mask)
    new = diff - 2*math.pi*np.ones(diff.size)
    M.filled(new)
    return np.array(M)

def event_display():
    list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber","nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept","isSR","isMC","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB","isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced","Jets.timeRecHitsEB","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB",]#,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"]
    for d in data:
        print "\n"
        print "\n"
        print "opening: ", MAIN + d + ".root"
        lista_files = samples[d]["files"]
        nuova_lista = []
        gen_events = 0
        for l in lista_files:
            nuova_lista.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        print nuova_lista
        gen = uproot.iterate(nuova_lista,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])
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
            cut_mask = arrays["isSR"]>0
            #cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
            #print "SELECT events with jets matched to CSC"
            #cut_mask_csc = np.logical_and(cut_mask,arrays["JetsCaloAdd.eFracEB2CSC0p1"]>0.8)
            #cut_mask_additional = (cut_mask_csc.any()==True)
            #cut_mask = np.logical_and(cut_mask,cut_mask_additional)
            #print "Narrox X"
            #cut_mask_x = np.logical_and(cut_mask,arrays["JetsCaloAdd.spreadXEB"]<1)
            #cut_mask_additional = (cut_mask_x.any()==True) 
            #cut_mask = np.logical_and(cut_mask,cut_mask_additional)
            #print "SELECT bin 1!!!"
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)
            #print "SELECT events close in dphi"
            #cut_mask = np.logical_and(cut_mask,arrays["min_dPhi_jets"]<0.02)
            
            ### Beam halo
            #cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            #cut_mask_x = np.logical_and(cut_mask_v,arrays["JetsCaloAdd.spreadXEB"]<1)
            #cut_mask_additional = (cut_mask_x.any()==True) 
            #cut_mask_v = np.logical_and(cut_mask_v,cut_mask_additional)
            ##cut_mask = cut_mask_v#
            #cut_mask_v = arrays["min_dPhi_jets"]<0.05
            #cut_mask = np.logical_not(cut_mask_v)

            #New beam halo
            #Invert it!!
            cut_mask_rms = arrays["Jets.timeRMSRecHitsEB"]>=8
            cut_mask_rms = np.logical_and(cut_mask,cut_mask_rms)
            cut_mask = (cut_mask_rms.any()==True)
            cut_jets = cut_mask_rms#np.logical_and(cut_jets,cut_mask_rms)

            #cut_mask_n = arrays["Jets.nRecHitsEB"]>1
            #cut_mask_n = np.logical_and(cut_mask,cut_mask_n)
            #cut_mask = (cut_mask_n.any()==True)
            #cut_jets = np.logical_and(cut_jets,cut_mask_n)
            
            
            ### Cosmics
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)
            #cosmic_veto = np.logical_and(arrays["dt_ecal_dist"]<0.5,arrays["dt_fit_chi2_reduced"]<0.15)
            cosmic_veto = arrays["dt_ecal_dist"]<0.5
            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            #Bin 2
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]>1)
            #Bin 1
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)


            #Proper bin1
            sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
            tag_mask = (sigprob > 0.996)
            bin1_m = (sigprob[tag_mask].counts==1)

            #isMC = arrays["isMC"][cut_mask]
            isMC = np.concatenate((isMC,np.array(arrays["isMC"][cut_mask][bin1_m])))
            RunNumber = np.concatenate((RunNumber,np.array(arrays["RunNumber"][cut_mask][bin1_m])))
            LumiNumber = np.concatenate((LumiNumber,np.array(arrays["LumiNumber"][cut_mask][bin1_m])))
            EventNumber = np.concatenate((EventNumber,np.array(arrays["EventNumber"][cut_mask][bin1_m])))
            Weight = np.concatenate((Weight, np.array( arrays["EventWeight"][cut_mask][bin1_m] * arrays["PUReWeight"][cut_mask][bin1_m] * arrays["TriggerWeight"][cut_mask][bin1_m]    )   ))
            #LumiNumber = arrays["LumiNumber"][cut_mask]
            #EventNumber = arrays["EventNumber"][cut_mask]
            #EventWeight = arrays["EventWeight"][cut_mask]
            #nDTSegments=arrays["nDTSegments"][cut_mask]
            #nCosmics=arrays["nCosmicMuons"][cut_mask]
            #nCosmicsOneLeg=arrays["nCosmicMuonsOneLeg"][cut_mask]
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
            DT_x = np.concatenate(( DT_x,np.divide(arrays["DTSegments.x"][cut_mask][bin1_m],100.) ))
            DT_y = np.concatenate(( DT_y,np.divide(arrays["DTSegments.y"][cut_mask][bin1_m],100.) ))
            DT_z = np.concatenate(( DT_z,np.divide(arrays["DTSegments.z"][cut_mask][bin1_m],100.) ))
            #DT_t = arrays["DTSegments.time"][cut_mask]
            #DT_eta = arrays["DTSegments.eta"][cut_mask]
            #DT_phi = arrays["DTSegments.phi"][cut_mask]
            CSC_x = np.concatenate(( CSC_x,np.divide(arrays["CSCSegments.x"][cut_mask][bin1_m],100.) ))
            CSC_y = np.concatenate(( CSC_y,np.divide(arrays["CSCSegments.y"][cut_mask][bin1_m],100.) ))
            CSC_z = np.concatenate(( CSC_z,np.divide(arrays["CSCSegments.z"][cut_mask][bin1_m],100.) ))
            #CSC_eta = arrays["CSCSegments.eta"][cut_mask]
            #CSC_phi = arrays["CSCSegments.phi"][cut_mask]
            #CSC_t = arrays["CSCSegments.time"][cut_mask]
            ECAL_tag_x = np.concatenate(( ECAL_tag_x,np.divide(arrays["taggedEcalRecHitsAK4.x"][cut_mask][bin1_m],100.) ))
            ECAL_tag_y = np.concatenate(( ECAL_tag_y,np.divide(arrays["taggedEcalRecHitsAK4.y"][cut_mask][bin1_m],100.) ))
            ECAL_tag_z = np.concatenate(( ECAL_tag_z,np.divide(arrays["taggedEcalRecHitsAK4.z"][cut_mask][bin1_m],100.) ))
            #ECAL_tag_eta = arrays["taggedEcalRecHitsAK4.eta"][cut_mask]
            #ECAL_tag_phi = arrays["taggedEcalRecHitsAK4.phi"][cut_mask]
            ECAL_x = np.concatenate(( ECAL_x,np.divide(arrays["skimmedEcalRecHitsAK4.x"][cut_mask][bin1_m],100.) ))
            ECAL_y = np.concatenate(( ECAL_y,np.divide(arrays["skimmedEcalRecHitsAK4.y"][cut_mask][bin1_m],100.) ))
            ECAL_z = np.concatenate(( ECAL_z,np.divide(arrays["skimmedEcalRecHitsAK4.z"][cut_mask][bin1_m],100.) ))
            #ECAL_eta = arrays["skimmedEcalRecHitsAK4.eta"][cut_mask]
            #ECAL_phi = arrays["skimmedEcalRecHitsAK4.phi"][cut_mask]
            #print "temp size of run number: ", RunNumber.shape


        print "gen_events: ", gen_events
        print "events passing: ", RunNumber.size
        print ("%d (%.2f%s) ") % (RunNumber.size, 100.*float(RunNumber.size/gen_events),'\%')

        for ev in range(EventNumber.shape[0]):
            
            DT        = np.dstack((DT_x[ev],DT_y[ev],DT_z[ev])).reshape(-1,3)
            ECAL      = np.dstack((ECAL_x[ev],ECAL_y[ev],ECAL_z[ev])).reshape(-1,3)
            ECAL_tag  = np.dstack((ECAL_tag_x[ev],ECAL_tag_y[ev],ECAL_tag_z[ev])).reshape(-1,3)
            CSC       = np.dstack((CSC_x[ev],CSC_y[ev],CSC_z[ev])).reshape(-1,3)

            ##Hist definition
            h_xy_dt       = TH2F("h_xy_dt","",     100, -10, 10, 100, -10, 10)
            h_xy_ecal     = TH2F("h_xy_ecal","",   100, -10, 10, 100, -10, 10)
            h_xy_ecal_tag = TH2F("h_xy_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_xy_csc      = TH2F("h_xy_csc","",     100, -10, 10, 100, -10, 10)
            
            h_yz_dt       = TH2F("h_yz_dt","",     100, -10, 10, 100, -10, 10)
            h_yz_ecal     = TH2F("h_yz_ecal","",   100, -10, 10, 100, -10, 10)
            h_yz_ecal_tag = TH2F("h_yz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_yz_csc      = TH2F("h_yz_csc","",     100, -10, 10, 100, -10, 10)

            h_xz_dt       = TH2F("h_xz_dt","",     100, -10, 10, 100, -10, 10)
            h_xz_ecal     = TH2F("h_xz_ecal","",   100, -10, 10, 100, -10, 10)
            h_xz_ecal_tag = TH2F("h_xz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_xz_csc      = TH2F("h_xz_csc","",     100, -10, 10, 100, -10, 10)

            h_xyz_dt      = TH3F("h_xyz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_ecal    = TH3F("h_xyz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_ecal_tag = TH3F("h_xyz_ecal_tag","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc     = TH3F("h_xyz_csc","",   100, -10, 10, 100, -10, 10, 100, -10, 10)

            root_numpy.fill_hist(h_xy_ecal,ECAL[:,[0,1]])
            root_numpy.fill_hist(h_xz_ecal,ECAL[:,[2,0]])
            root_numpy.fill_hist(h_yz_ecal,ECAL[:,[2,1]])
            root_numpy.fill_hist(h_xyz_ecal,ECAL[:,[2,0,1]])

            h_xy_ecal.SetMarkerColor(2)
            h_xy_ecal.SetMarkerStyle(20)
            h_xz_ecal.SetMarkerColor(2)
            h_xz_ecal.SetMarkerStyle(20)
            h_yz_ecal.SetMarkerColor(2)
            h_yz_ecal.SetMarkerStyle(20)
            h_xyz_ecal.SetMarkerColor(2)
            h_xyz_ecal.SetMarkerStyle(20)

            root_numpy.fill_hist(h_xy_ecal_tag,ECAL_tag[:,[0,1]])
            root_numpy.fill_hist(h_xz_ecal_tag,ECAL_tag[:,[2,0]])
            root_numpy.fill_hist(h_yz_ecal_tag,ECAL_tag[:,[2,1]])
            root_numpy.fill_hist(h_xyz_ecal_tag,ECAL_tag[:,[2,0,1]])
            
            h_xy_ecal_tag.SetMarkerColor(8)
            h_xy_ecal_tag.SetMarkerStyle(24)
            h_yz_ecal_tag.SetMarkerColor(8)
            h_yz_ecal_tag.SetMarkerStyle(24)
            h_xz_ecal_tag.SetMarkerColor(8)
            h_xz_ecal_tag.SetMarkerStyle(24)
            h_xyz_ecal_tag.SetMarkerColor(8)
            h_xyz_ecal_tag.SetMarkerStyle(24)

            root_numpy.fill_hist(h_xy_csc,CSC[:,[0,1]])
            root_numpy.fill_hist(h_xz_csc,CSC[:,[2,0]])
            root_numpy.fill_hist(h_yz_csc,CSC[:,[2,1]])
            root_numpy.fill_hist(h_xyz_csc,CSC[:,[2,0,1]])

            h_xy_csc.SetMarkerColor(800)
            h_xy_csc.SetMarkerStyle(20)
            h_yz_csc.SetMarkerColor(800)
            h_yz_csc.SetMarkerStyle(20)
            h_xz_csc.SetMarkerColor(800)
            h_xz_csc.SetMarkerStyle(20)
            h_xyz_csc.SetMarkerColor(800)
            h_xyz_csc.SetMarkerStyle(20)

            root_numpy.fill_hist(h_xy_dt,DT[:,[0,1]])
            root_numpy.fill_hist(h_xz_dt,DT[:,[2,0]])
            root_numpy.fill_hist(h_yz_dt,DT[:,[2,1]])
            root_numpy.fill_hist(h_xyz_dt,DT[:,[2,0,1]])

            h_xy_dt.SetMarkerColor(4)
            h_xy_dt.SetMarkerStyle(20)
            h_yz_dt.SetMarkerColor(4)
            h_yz_dt.SetMarkerStyle(20)
            h_xz_dt.SetMarkerColor(4)
            h_xz_dt.SetMarkerStyle(20)
            h_xyz_dt.SetMarkerColor(4)
            h_xyz_dt.SetMarkerStyle(20)

            #xy
            
            leg = TLegend(0.75, 0.7, 1., 1.)
            leg.AddEntry(h_xy_dt,"DT segments","P")
            leg.AddEntry(h_xy_csc,"CSC segments","P")
            leg.AddEntry(h_xy_ecal,"EB rec hits","P")
            leg.AddEntry(h_xy_ecal_tag,"EB rec hits; tag jets","P")

            can_xy = TCanvas("can_xy","can_xy",900,800)
            can_xy.cd()
            can_xy.SetGrid()
            can_xy.SetRightMargin(0.05)
            h_xy_dt.GetXaxis().SetTitle("x (m)")
            h_xy_dt.GetYaxis().SetTitle("y (m)")
            h_xy_dt.GetYaxis().SetTitleOffset(1.4)

            h_xy_dt.Draw("")
            h_xy_csc.Draw("sames")
            h_xy_ecal.Draw("sames")
            h_xy_ecal_tag.Draw("sames")

            OUTSTRING = OUT
            if isMC[ev]:
                OUTSTRING+="mc_"
            else:
                OUTSTRING+="data_"
            OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  

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
            h_xy_csc.Delete()
            h_xy_ecal.Delete()
            h_xy_ecal_tag.Delete()


            #xz
            leg = TLegend(0.75, 0.7, 1., 1.)
            leg.AddEntry(h_xz_dt,"DT segments","P")
            leg.AddEntry(h_xz_csc,"CSC segments","P")
            leg.AddEntry(h_xz_ecal,"EB rec hits","P")
            leg.AddEntry(h_xz_ecal_tag,"EB rec hits; tag jets","P")
            
            can_xz = TCanvas("can_xz","can_xz",900,800)
            can_xz.cd()
            can_xz.SetGrid()
            can_xz.SetRightMargin(0.05)
            h_xz_dt.GetXaxis().SetTitle("z (m)")
            h_xz_dt.GetYaxis().SetTitle("x (m)")
            h_xz_dt.GetYaxis().SetTitleOffset(1.4)

            h_xz_dt.Draw("")
            h_xz_csc.Draw("sames")
            h_xz_ecal.Draw("sames")
            h_xz_ecal_tag.Draw("sames")

            OUTSTRING = OUT
            if isMC[ev]:
                OUTSTRING+="mc_"
            else:
                OUTSTRING+="data_"
            OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  

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
            h_xz_csc.Delete()
            h_xz_ecal.Delete()
            h_xz_ecal_tag.Delete()
            
            #yz
            leg = TLegend(0.75, 0.7, 1., 1.)
            leg.AddEntry(h_yz_dt,"DT segments","P")
            leg.AddEntry(h_yz_csc,"CSC segments","P")
            leg.AddEntry(h_yz_ecal,"EB rec hits","P")
            leg.AddEntry(h_yz_ecal_tag,"EB rec hits; tag jets","P")
            
            can_yz = TCanvas("can_yz","can_yz",900,800)
            can_yz.cd()
            can_yz.SetGrid()
            can_yz.SetRightMargin(0.05)
            h_yz_dt.GetXaxis().SetTitle("z (m)")
            h_yz_dt.GetYaxis().SetTitle("y (m)")
            h_yz_dt.GetYaxis().SetTitleOffset(1.4)
            
            h_yz_dt.Draw("")
            h_yz_csc.Draw("sames")
            h_yz_ecal.Draw("sames")
            h_yz_ecal_tag.Draw("sames")
            
            OUTSTRING = OUT
            if isMC[ev]:
                OUTSTRING+="mc_"
            else:
                OUTSTRING+="data_"
            OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  

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
            h_yz_csc.Delete()
            h_yz_ecal.Delete()
            h_yz_ecal_tag.Delete()

            #xyz
            leg = TLegend(0.75, 0.7, 1., 1.)
            leg.AddEntry(h_xyz_dt,"DT segments","P")
            leg.AddEntry(h_xyz_csc,"CSC segments","P")
            leg.AddEntry(h_xyz_ecal,"EB rec hits","P")
            leg.AddEntry(h_xyz_ecal_tag,"EB rec hits; tag jets","P")
            
            can_xyz = TCanvas("can_xyz","can_xyz",900,800)
            can_xyz.cd()
            can_xyz.SetGrid()
            can_xyz.SetRightMargin(0.05)
            h_xyz_dt.GetXaxis().SetTitle("z (m)")
            h_xyz_dt.GetYaxis().SetTitle("x (m)")
            h_xyz_dt.GetZaxis().SetTitle("y (m)")
            h_xyz_dt.GetYaxis().SetTitleOffset(1.4)
            h_xyz_dt.GetZaxis().SetTitleOffset(1.4)
            
            h_xyz_dt.Draw("")
            h_xyz_csc.Draw("sames")
            h_xyz_ecal.Draw("sames")
            h_xyz_ecal_tag.Draw("sames")

            OUTSTRING = OUT
            if isMC[ev]:
                OUTSTRING+="mc_"
            else:
                OUTSTRING+="data_"
            OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  

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
            h_xyz_csc.Delete()
            h_xyz_ecal.Delete()
            h_xyz_ecal_tag.Delete()


def csc_fit():
    list_of_variables = ["nTagJets_0p996_JJ","EventNumber","LumiNumber","RunNumber","nDTSegments","nCosmicMuons","nCosmicMuonsOneLeg","DTSegments.x","DTSegments.y","DTSegments.z","CSCSegments.x","CSCSegments.y","CSCSegments.z","DTSegments.time","CSCSegments.time","skimmedEcalRecHitsAK4.x","skimmedEcalRecHitsAK4.y","skimmedEcalRecHitsAK4.z","taggedEcalRecHitsAK4.x","taggedEcalRecHitsAK4.phi","taggedEcalRecHitsAK4.y","taggedEcalRecHitsAK4.z","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventWeight","TriggerWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept","isSR","isMC","DTSegments.eta","DTSegments.phi","CSCSegments.eta","CSCSegments.phi","taggedEcalRecHitsAK4.eta","taggedEcalRecHitsAK4.phi","skimmedEcalRecHitsAK4.eta","skimmedEcalRecHitsAK4.phi","min_dPhi_jets","min_dPhi_jets_0p996","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.spreadXEB","isDT_fit","dt_ecal_dist","dt_fit_chi2_reduced"]#,"CosmicMuons.eta","CosmicMuons.phi","CosmicMuonsOneLeg.eta","CosmicMuonsOneLeg.phi"]#,"CosmicMuonsPropagated.eta","CosmicMuonsPropagated.phi","CosmicMuonsPropagatedOneLeg.eta","CosmicMuonsPropagatedOneLeg.phi"]
    for d in data:
        print "\n"
        print "\n"
        print "opening: ", MAIN + d + ".root"
        lista_files = samples[d]["files"]
        nuova_lista = []
        gen_events = 0
        for l in lista_files:
            nuova_lista.append(MAIN+l+".root")
            file_temp = uproot.open(MAIN+l+".root")
            gen_events+=file_temp["c_nEvents"].values[0]
        print nuova_lista
        gen = uproot.iterate(nuova_lista,"tree",list_of_variables)
        isMC = np.array([])
        Weight = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])
        DT_x = np.array([])
        DT_y = np.array([])
        DT_z = np.array([])
        CSC_x = np.array([])
        CSC_y = np.array([])
        CSC_z = np.array([])
        CSC_time = np.array([])
        CSC_phi = np.array([])
        ECAL_x = np.array([])
        ECAL_y = np.array([])
        ECAL_z = np.array([])
        ECAL_tag_x  = np.array([])
        ECAL_tag_y  = np.array([])
        ECAL_tag_z  = np.array([])
        ECAL_tag_phi  = np.array([])
        #print gen
        for arrays in gen:
            cut_mask = arrays["isSR"]>0
            #cut_mask = np.logical_and(cut_mask,arrays["MinJetMetDPhi"]>0.5)
            #print "SELECT events with jets matched to CSC"
            #cut_mask_csc = np.logical_and(cut_mask,arrays["JetsCaloAdd.eFracEB2CSC0p1"]>0.8)
            #cut_mask_additional = (cut_mask_csc.any()==True)
            #cut_mask = np.logical_and(cut_mask,cut_mask_additional)
            #print "Narrox X"
            #cut_mask_x = np.logical_and(cut_mask,arrays["JetsCaloAdd.spreadXEB"]<1)
            #cut_mask_additional = (cut_mask_x.any()==True) 
            #cut_mask = np.logical_and(cut_mask,cut_mask_additional)
            #print "SELECT bin 1!!!"
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)
            #print "SELECT events close in dphi"
            #cut_mask = np.logical_and(cut_mask,arrays["min_dPhi_jets"]<0.02)
            
            ### Beam halo
            
            #cut_mask_v = arrays["min_dPhi_jets_0p996"]<0.05
            #cut_mask_x = np.logical_and(cut_mask_v,arrays["JetsCaloAdd.spreadXEB"]<1)
            #cut_mask_additional = (cut_mask_x.any()==True) 
            #cut_mask_v = np.logical_and(cut_mask_v,cut_mask_additional)
            ##cut_mask = cut_mask_v#
            #cut_mask_v = arrays["min_dPhi_jets"]<0.05
            #cut_mask = np.logical_not(cut_mask_v)
            
            ### Cosmics
            #cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]==1)
            #cosmic_veto = np.logical_and(arrays["dt_ecal_dist"]<0.5,arrays["dt_fit_chi2_reduced"]<0.15)
            cosmic_veto = arrays["dt_ecal_dist"]<0.5
            cut_mask = np.logical_and(cut_mask,np.logical_not(cosmic_veto))

            #Bin 2
            cut_mask = np.logical_and(cut_mask,arrays["nTagJets_0p996_JJ"]>1)

            #isMC = arrays["isMC"][cut_mask]
            isMC = np.concatenate((isMC,np.array(arrays["isMC"][cut_mask])))
            RunNumber = np.concatenate((RunNumber,np.array(arrays["RunNumber"][cut_mask])))
            LumiNumber = np.concatenate((LumiNumber,np.array(arrays["LumiNumber"][cut_mask])))
            EventNumber = np.concatenate((EventNumber,np.array(arrays["EventNumber"][cut_mask])))
            Weight = np.concatenate((Weight, np.array( arrays["EventWeight"][cut_mask] * arrays["PUReWeight"][cut_mask] * arrays["TriggerWeight"][cut_mask]    )   ))
            #LumiNumber = arrays["LumiNumber"][cut_mask]
            #EventNumber = arrays["EventNumber"][cut_mask]
            #EventWeight = arrays["EventWeight"][cut_mask]
            #nDTSegments=arrays["nDTSegments"][cut_mask]
            #nCosmics=arrays["nCosmicMuons"][cut_mask]
            #nCosmicsOneLeg=arrays["nCosmicMuonsOneLeg"][cut_mask]
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
            DT_x = np.concatenate(( DT_x,np.divide(arrays["DTSegments.x"][cut_mask],100.) ))
            DT_y = np.concatenate(( DT_y,np.divide(arrays["DTSegments.y"][cut_mask],100.) ))
            DT_z = np.concatenate(( DT_z,np.divide(arrays["DTSegments.z"][cut_mask],100.) ))
            #DT_t = arrays["DTSegments.time"][cut_mask]
            #DT_eta = arrays["DTSegments.eta"][cut_mask]
            #DT_phi = arrays["DTSegments.phi"][cut_mask]
            CSC_x = np.concatenate(( CSC_x,np.divide(arrays["CSCSegments.x"][cut_mask],100.) ))
            CSC_y = np.concatenate(( CSC_y,np.divide(arrays["CSCSegments.y"][cut_mask],100.) ))
            CSC_z = np.concatenate(( CSC_z,np.divide(arrays["CSCSegments.z"][cut_mask],100.) ))
            CSC_time = np.concatenate(( CSC_time,arrays["CSCSegments.time"][cut_mask] ))
            CSC_phi = np.concatenate(( CSC_phi,arrays["CSCSegments.phi"][cut_mask] ))
            #CSC_eta = arrays["CSCSegments.eta"][cut_mask]
            #CSC_phi = arrays["CSCSegments.phi"][cut_mask]
            #CSC_t = arrays["CSCSegments.time"][cut_mask]
            ECAL_tag_x = np.concatenate(( ECAL_tag_x,np.divide(arrays["taggedEcalRecHitsAK4.x"][cut_mask],100.) ))
            ECAL_tag_y = np.concatenate(( ECAL_tag_y,np.divide(arrays["taggedEcalRecHitsAK4.y"][cut_mask],100.) ))
            ECAL_tag_z = np.concatenate(( ECAL_tag_z,np.divide(arrays["taggedEcalRecHitsAK4.z"][cut_mask],100.) ))
            ECAL_tag_phi = np.concatenate(( ECAL_tag_z,arrays["taggedEcalRecHitsAK4.phi"][cut_mask] ))
            #ECAL_tag_eta = arrays["taggedEcalRecHitsAK4.eta"][cut_mask]
            #ECAL_tag_phi = arrays["taggedEcalRecHitsAK4.phi"][cut_mask]
            ECAL_x = np.concatenate(( ECAL_x,np.divide(arrays["skimmedEcalRecHitsAK4.x"][cut_mask],100.) ))
            ECAL_y = np.concatenate(( ECAL_y,np.divide(arrays["skimmedEcalRecHitsAK4.y"][cut_mask],100.) ))
            ECAL_z = np.concatenate(( ECAL_z,np.divide(arrays["skimmedEcalRecHitsAK4.z"][cut_mask],100.) ))
            #ECAL_eta = arrays["skimmedEcalRecHitsAK4.eta"][cut_mask]
            #ECAL_phi = arrays["skimmedEcalRecHitsAK4.phi"][cut_mask]
            #print "temp size of run number: ", RunNumber.shape

        print "gen_events: ", gen_events
        print "events passing: ", RunNumber.size
        print ("%d (%.2f%s) ") % (RunNumber.size, 100.*float(RunNumber.size/gen_events),'\%')

        for ev in range(EventNumber.shape[0]):
            print ev, CSC_x[ev], CSC_x[ev].size
            if CSC_x[ev].size == 0:
                continue
            if ECAL_tag_x[ev].size == 0:
                continue

            print "******"
            print "processing event: ", ev

            DT       = np.dstack((DT_x[ev],DT_y[ev],DT_z[ev])).reshape(-1,3)
            ECAL     = np.dstack((ECAL_x[ev],ECAL_y[ev],ECAL_z[ev])).reshape(-1,3)
            ECAL_tag = np.dstack((ECAL_tag_x[ev],ECAL_tag_y[ev],ECAL_tag_z[ev])).reshape(-1,3)
            CSC       = np.dstack((CSC_x[ev],CSC_y[ev],CSC_z[ev])).reshape(-1,3)
            #print "CSC"
            #print CSC
            #print CSC_time[ev]

            ##Hist definition
            h_xy_dt   = TH2F("h_xy_dt","",     100, -10, 10, 100, -10, 10)
            h_xy_csc_n   = TH2F("h_xy_csc_n","",     100, -10, 10, 100, -10, 10)
            h_xy_csc_0   = TH2F("h_xy_csc_0","",     100, -10, 10, 100, -10, 10)
            h_xy_csc_1   = TH2F("h_xy_csc_1","",     100, -10, 10, 100, -10, 10)
            h_xy_csc_2   = TH2F("h_xy_csc_2","",     100, -10, 10, 100, -10, 10)
            h_xy_csc_3   = TH2F("h_xy_csc_3","",     100, -10, 10, 100, -10, 10)
            h_xy_ecal = TH2F("h_xy_ecal","",   100, -10, 10, 100, -10, 10)
            h_xy_ecal_tag = TH2F("h_xy_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_xy_csc_sum   = TH2F("h_xy_csc_sum","",     100, -10, 10, 100, -10, 10)

            h_etaphi_dt     = TH2F("h_etaphi_dt","",         100, -3, 3, 100, -4, 4)
            h_etaphi_csc_n   = TH2F("h_etaphi_csc_n","",       100, -3, 3, 100, -4, 4)
            h_etaphi_csc_0   = TH2F("h_etaphi_csc_0","",       100, -3, 3, 100, -4, 4)
            h_etaphi_csc_1   = TH2F("h_etaphi_csc_1","",       100, -3, 3, 100, -4, 4)
            h_etaphi_csc_2   = TH2F("h_etaphi_csc_2","",       100, -3, 3, 100, -4, 4)
            h_etaphi_csc_3   = TH2F("h_etaphi_csc_3","",       100, -3, 3, 100, -4, 4)
            h_etaphi_ecal   = TH2F("h_etaphi_ecal","",       100, -3, 3, 100, -4, 4)
            h_etaphi_ecal_tag = TH2F("h_etaphi_ecal_tag","", 100, -3, 3, 100, -4, 4)
            h_etaphi_csc_sum   = TH2F("h_etaphi_csc_sum","",   100, -3, 3, 100, -4, 4)
            #h_etaphi_cos     = TH2F("h_etaphi_cos","",         100, -3, 3, 100, -4, 4)
            #h_etaphi_cos1L   = TH2F("h_etaphi_cos1L","",         100, -3, 3, 100, -4, 4)

            h_yz_dt   = TH2F("h_yz_dt","",     100, -10, 10, 100, -10, 10)
            h_yz_csc_n   = TH2F("h_yz_csc_n","",     100, -10, 10, 100, -10, 10)
            h_yz_csc_0   = TH2F("h_yz_csc_0","",     100, -10, 10, 100, -10, 10)
            h_yz_csc_1   = TH2F("h_yz_csc_1","",     100, -10, 10, 100, -10, 10)
            h_yz_csc_2   = TH2F("h_yz_csc_2","",     100, -10, 10, 100, -10, 10)
            h_yz_csc_3   = TH2F("h_yz_csc_3","",     100, -10, 10, 100, -10, 10)
            h_yz_ecal = TH2F("h_yz_ecal","",   100, -10, 10, 100, -10, 10)
            h_yz_ecal_tag = TH2F("h_yz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_yz_csc_sum   = TH2F("h_yz_csc_sum","",     100, -10, 10, 100, -10, 10)

            h_phiz_dt   = TH2F("h_phiz_dt","",     100, -10, 10, 100, -10, 10)
            h_phiz_csc_n   = TH2F("h_phiz_csc_n","",     100, -10, 10, 100, -10, 10)
            h_phiz_csc_0   = TH2F("h_phiz_csc_0","",     100, -10, 10, 100, -10, 10)
            h_phiz_csc_1   = TH2F("h_phiz_csc_1","",     100, -10, 10, 100, -10, 10)
            h_phiz_csc_2   = TH2F("h_phiz_csc_2","",     100, -10, 10, 100, -10, 10)
            h_phiz_csc_3   = TH2F("h_phiz_csc_3","",     100, -10, 10, 100, -10, 10)
            h_phiz_ecal = TH2F("h_phiz_ecal","",   100, -10, 10, 100, -10, 10)
            h_phiz_ecal_tag = TH2F("h_phiz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_phiz_csc_sum   = TH2F("h_phiz_csc_sum","",     100, -10, 10, 100, -10, 10)


            h_xz_dt   = TH2F("h_xz_dt","",     100, -10, 10, 100, -10, 10)
            h_xz_csc_n   = TH2F("h_xz_csc_n","",     100, -10, 10, 100, -10, 10)
            h_xz_csc_0   = TH2F("h_xz_csc_0","",     100, -10, 10, 100, -10, 10)
            h_xz_csc_1   = TH2F("h_xz_csc_1","",     100, -10, 10, 100, -10, 10)
            h_xz_csc_2   = TH2F("h_xz_csc_2","",     100, -10, 10, 100, -10, 10)
            h_xz_csc_3   = TH2F("h_xz_csc_3","",     100, -10, 10, 100, -10, 10)
            h_xz_ecal = TH2F("h_xz_ecal","",   100, -10, 10, 100, -10, 10)
            h_xz_ecal_tag = TH2F("h_xz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_xz_csc_sum   = TH2F("h_xz_csc_sum","",     100, -10, 10, 100, -10, 10)

            h_etaz_dt   = TH2F("h_etaz_dt","",     100, -10, 10, 100, -10, 10)
            h_etaz_csc_n   = TH2F("h_etaz_csc_n","",     100, -10, 10, 100, -10, 10)
            h_etaz_csc_0   = TH2F("h_etaz_csc_0","",     100, -10, 10, 100, -10, 10)
            h_etaz_csc_1   = TH2F("h_etaz_csc_1","",     100, -10, 10, 100, -10, 10)
            h_etaz_csc_2   = TH2F("h_etaz_csc_2","",     100, -10, 10, 100, -10, 10)
            h_etaz_csc_3   = TH2F("h_etaz_csc_3","",     100, -10, 10, 100, -10, 10)
            h_etaz_ecal = TH2F("h_etaz_ecal","",   100, -10, 10, 100, -10, 10)
            h_etaz_ecal_tag = TH2F("h_etaz_ecal_tag","",   100, -10, 10, 100, -10, 10)
            h_etaz_csc_sum   = TH2F("h_etaz_csc_sum","",     100, -10, 10, 100, -10, 10)

            h_xyz_dt   = TH3F("h_xyz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_n   = TH3F("h_xyz_csc_n","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_0   = TH3F("h_xyz_csc_0","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_1   = TH3F("h_xyz_csc_1","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_2   = TH3F("h_xyz_csc_2","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_3   = TH3F("h_xyz_csc_3","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_ecal = TH3F("h_xyz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_ecal_tag = TH3F("h_xyz_ecal_tag","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_xyz_csc_sum   = TH3F("h_xyz_csc_sum","",   100, -10, 10, 100, -10, 10, 100, -10, 10)

            h_etaphiz_dt   = TH3F("h_etaphiz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_n   = TH3F("h_etaphiz_csc_n","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_0   = TH3F("h_etaphiz_csc_0","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_1   = TH3F("h_etaphiz_csc_1","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_2   = TH3F("h_etaphiz_csc_2","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_3   = TH3F("h_etaphiz_csc_3","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_ecal = TH3F("h_etaphiz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_ecal_tag = TH3F("h_etaphiz_ecal_tag","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
            h_etaphiz_csc_sum   = TH3F("h_etaphiz_csc_sum","",   100, -10, 10, 100, -10, 10, 100, -10, 10)

            fit_csc_3d   = TH3F("fit_csc_3d","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            fit_csc_xy   = TH2F("fit_csc_xy","",   100, -10, 10, 100, -10, 10)
            fit_csc_xz   = TH2F("fit_csc_xz","",   100, -10, 10, 100, -10, 10)
            fit_csc_yz   = TH2F("fit_csc_yz","",   100, -10, 10, 100, -10, 10)

            fit_ecal_3d   = TH3F("fit_ecal_3d","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
            fit_ecal_xy   = TH2F("fit_ecal_xy","",   100, -10, 10, 100, -10, 10)
            fit_ecal_xz   = TH2F("fit_ecal_xz","",   100, -10, 10, 100, -10, 10)
            fit_ecal_yz   = TH2F("fit_ecal_yz","",   100, -10, 10, 100, -10, 10)

            ##I should select the CSC that are close in z to the direction of the tag
            ##DBSCAN section
            if CSC.size>0:
                print RunNumber[ev], LumiNumber[ev]
                mean_ECAL_tag_x = np.sum(ECAL_tag_x[ev].flatten())/np.size(ECAL_tag_x[ev])
                mean_ECAL_tag_y = np.sum(ECAL_tag_y[ev].flatten())/np.size(ECAL_tag_y[ev])
                mean_ECAL_tag_z = np.sum(ECAL_tag_z[ev].flatten())/np.size(ECAL_tag_z[ev])

                mean_ECAL_tag_phi = np.sum(ECAL_tag_phi[ev].flatten())/np.size(ECAL_tag_phi[ev])

                mean_ECAL_tag_r = math.sqrt(mean_ECAL_tag_x**2 + mean_ECAL_tag_y**2)
                #print "mean_ECAL_tag_z: ", mean_ECAL_tag_z
                #print CSC_x[ev], mean_ECAL_tag_x
                #print CSC_y[ev], mean_ECAL_tag_y
                #print CSC_z[ev], mean_ECAL_tag_z
                print CSC_phi[ev], mean_ECAL_tag_phi
                print "delta phi"
                print deltaPhi_np(CSC_phi[ev], mean_ECAL_tag_phi)
                print "All CSC:"
                print CSC
                print "CSC considered:"
                print CSC[deltaPhi_np(CSC_phi[ev], mean_ECAL_tag_phi)<0.4]
                print "And their time:"
                print CSC_time[ev][deltaPhi_np(CSC_phi[ev], mean_ECAL_tag_phi)<0.4]
                CSC = CSC[deltaPhi_np(CSC_phi[ev], mean_ECAL_tag_phi)<0.5]
            ##DBSCAN section
            if CSC.size>0:
                print "Attempt DBSCAN"
                dbscan = DBSCAN(eps=0.8,min_samples=1).fit(CSC)#1 and 2 #also 0.8 and 3 worked
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
                m_cl_0 = (labels==0)
                m_cl_1 = (labels==1)
                m_cl_2 = (labels==2)
                m_cl_3 = (labels==3)

                CSC_cl = {}
                CSC_n = CSC[m_noise]

                for k in m.keys():
                    CSC_cl[k] = CSC[m[k]]
                    #print CSC_time
                    #print m[k]

                #print CSC_cl
                if n_clusters_>=2:
                    n_pairs = int(n_clusters_/2)
                    if (n_clusters_ % 2 )!=0:
                        print "Odd number of CSC clusters, strange..."
                        #exit()

                    mean_x = {}
                    mean_y = {}
                    mean_z = {}
                    mean_t = {}
                    std_t = {}
                    for k in m.keys():
                        mean_x[k] = np.sum(CSC_cl[k][:,0])/np.size( CSC_cl[k][:,0] )
                        mean_y[k] = np.sum(CSC_cl[k][:,1])/np.size( CSC_cl[k][:,1] )
                        mean_z[k] = np.sum(CSC_cl[k][:,2])/np.size( CSC_cl[k][:,2] )

                    mean_ECAL_tag_x = np.sum(ECAL_tag_x[ev].flatten())/np.size(ECAL_tag_x[ev])
                    mean_ECAL_tag_y = np.sum(ECAL_tag_y[ev].flatten())/np.size(ECAL_tag_y[ev])
                    mean_ECAL_tag_r = math.sqrt(mean_ECAL_tag_x**2 + mean_ECAL_tag_y**2)
                    #print "mean_ECAL_tag_z: ", mean_ECAL_tag_z

                    pairs = {}
                    #for n in range(n_pairs):
                    #    pairs[n] = np.array([-1,1])

                    #This handles two cosmics. If there are more, we'll have to deal with this somehow...
                    dx_CSC = 1000.
                    dx_ECAL = 1000.
                    dr_CSC = 1000.
                    dr_ECAL = 1000.
                    ch_k1 = -1
                    ch_k2 = -1
                    ch_t_k1 = 0
                    ch_t_k2 = 0
                    for k1 in range(len(m.keys())):
                        #print k1
                        #print mean_x[k1], mean_y[k1]
                        for k2 in range(len(m.keys())):
                            if k2>k1 and ( (mean_z[k1]*mean_z[k2])<0 ):# and nCosmicsOneLeg[ev]>0 and nCosmics[ev]>1:
                                ##opposite emispheres
                                #print "z average: ", mean_z[k1], mean_z[k2]
                                tmp_x = abs(mean_x[k1] - mean_x[k2])
                                tmp_y = abs(mean_y[k1] - mean_y[k2])
                                tmp_r = math.sqrt(tmp_x**2 + tmp_y**2)
                                #dx_CSC = min(dx_CSC,tmp_x)
                                dr_CSC = min(dr_CSC,tmp_r)

                                tmp_x_ECAL = abs((mean_x[k1]+mean_x[k2])/2. - mean_ECAL_tag_x)
                                tmp_y_ECAL = abs((mean_y[k1]+mean_y[k2])/2. - mean_ECAL_tag_y)
                                tmp_r_ECAL = math.sqrt(tmp_x_ECAL**2 + tmp_y_ECAL**2)
                                dx_ECAL = min(dx_ECAL,tmp_x_ECAL)
                                dr_ECAL = min(dr_ECAL,tmp_r_ECAL)

                                #if dx_CSC==tmp_x and dx_ECAL==tmp_x_ECAL:
                                if dr_CSC==tmp_r and dr_ECAL==tmp_r_ECAL:
                                    ch_k1 = k1
                                    ch_k2 = k2
                                    #print "I choose you: ", k1, k2

                    #print "Overwrite CSC dict"
                    if ch_k1>-1 and ch_k2>-1:
                        #print "Pair: (",k1,",",k2,")"
                        tmp_CSC_cl_0 = CSC_cl[ch_k1]
                        tmp_CSC_cl_1 = CSC_cl[ch_k2]
                        
                        new_keys = m.keys()
                        new_keys = np.setdiff1d(np.array(m.keys()),np.array([ch_k1,ch_k2]))
                        tmp_cl_discarded = {}
                        for a in range(len(new_keys)):
                            tmp_cl_discarded[a] = CSC_cl[new_keys[a]]

                        CSC_cl[0] = tmp_CSC_cl_0
                        CSC_cl[1] = tmp_CSC_cl_1

                        #Add discarded cosmics to noise histogram
                        for a in range(len(new_keys)):
                            CSC_n = np.vstack((CSC_n,tmp_cl_discarded[a]))
                    else:
                        print "No cosmic identified, all noise"
                        for a in (m.keys()):
                            CSC_n = np.vstack((CSC_n,CSC_cl[a]))
                        for a in (m.keys()):
                            CSC_cl[a] = np.array([])
                        continue

                if n_clusters_==1:
                    print "Only 1 cluster, no cosmic identified, all noise"
                    for a in (m.keys()):
                        CSC_n = np.vstack((CSC_n,CSC_cl[a]))
                    for a in (m.keys()):
                        CSC_cl[a] = np.array([])
                    continue

                if n_clusters_==0:
                    print "No clusters, no cosmic identified, all noise"
                    continue


            if CSC.size>0:
                if CSC_n.size>0:
                    root_numpy.fill_hist(h_xy_csc_n,CSC_n[:,[0,1]])
                    root_numpy.fill_hist(h_xz_csc_n,CSC_n[:,[2,0]])
                    root_numpy.fill_hist(h_yz_csc_n,CSC_n[:,[2,1]])
                    root_numpy.fill_hist(h_xyz_csc_n,CSC_n[:,[2,0,1]])


            #Initialize objects
            CSC_to_fit = np.array([])
            CSC_post_fit = np.array([])

            if CSC.size>0:
                if len(CSC_cl.keys())>0:
                    if CSC_cl[0].size>0:
                        root_numpy.fill_hist(h_xy_csc_0,CSC_cl[0][:,[0,1]])
                        root_numpy.fill_hist(h_xz_csc_0,CSC_cl[0][:,[2,0]])
                        root_numpy.fill_hist(h_yz_csc_0,CSC_cl[0][:,[2,1]])
                        root_numpy.fill_hist(h_xyz_csc_0,CSC_cl[0][:,[2,0,1]])

                
                if len(CSC_cl.keys())>1:
                    if CSC_cl[1].size>0:
                        root_numpy.fill_hist(h_xy_csc_1,CSC_cl[1][:,[0,1]])
                        root_numpy.fill_hist(h_xz_csc_1,CSC_cl[1][:,[2,0]])
                        root_numpy.fill_hist(h_yz_csc_1,CSC_cl[1][:,[2,1]])
                        root_numpy.fill_hist(h_xyz_csc_1,CSC_cl[1][:,[2,0,1]])
                    

                if len(CSC_cl.keys())>1:
                    CSC_to_fit = np.concatenate((CSC_cl[0],CSC_cl[1]))
                else:
                    CSC_to_fit = np.array([])

            else:
                n_noise_=-1
            #print "full ECAL_tag"
            #print ECAL_tag
            ECAL_to_fit = ECAL_tag
            ssq = 1000.

            root_numpy.fill_hist(h_xy_ecal,ECAL[:,[0,1]])
            root_numpy.fill_hist(h_xz_ecal,ECAL[:,[2,0]])
            root_numpy.fill_hist(h_yz_ecal,ECAL[:,[2,1]])
            root_numpy.fill_hist(h_xyz_ecal,ECAL[:,[2,0,1]])

            root_numpy.fill_hist(h_xy_ecal_tag,ECAL_tag[:,[0,1]])
            root_numpy.fill_hist(h_xz_ecal_tag,ECAL_tag[:,[2,0]])
            root_numpy.fill_hist(h_yz_ecal_tag,ECAL_tag[:,[2,1]])
            root_numpy.fill_hist(h_xyz_ecal_tag,ECAL_tag[:,[2,0,1]])


            ##3D line fit
            if CSC_to_fit.size > 0:
            
                ## Method 1.
                ## Using linalg
                ## Pro: looks fast. Cons: wants zz as input
            
                x,y,z = CSC_to_fit.T
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
                CSC_post_fit   = np.dstack((zz,np.dstack((xx,yy)))).reshape(xx.size,3)
                zz_ecal = ECAL_to_fit[:,2]
                xx_ecal,yy_ecal = lin(zz_ecal)
                ECAL_post_fit = np.dstack((zz_ecal,np.dstack((xx_ecal,yy_ecal)))).reshape(xx_ecal.size,3)
                ECAL_diff = ECAL_to_fit - np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3)#np.divide( (ECAL_to_fit - np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3))**2 , np.dstack((xx_ecal,np.dstack((yy_ecal,zz_ecal)))).reshape(xx_ecal.size,3))
                ssq = np.mean(np.sqrt(np.sum(ECAL_diff**2,axis=-1)))
                root_numpy.fill_hist(fit_csc_3d, CSC_post_fit)
                root_numpy.fill_hist(fit_csc_xy, CSC_post_fit[:,[1,2]])
                root_numpy.fill_hist(fit_csc_xz, CSC_post_fit[:,[0,1]])
                root_numpy.fill_hist(fit_csc_yz, CSC_post_fit[:,[0,2]])

                root_numpy.fill_hist(fit_ecal_3d, ECAL_post_fit)
                root_numpy.fill_hist(fit_ecal_xy, ECAL_post_fit[:,[1,2]])
                root_numpy.fill_hist(fit_ecal_xz, ECAL_post_fit[:,[0,1]])
                root_numpy.fill_hist(fit_ecal_yz, ECAL_post_fit[:,[0,2]])
            else:
                CSC_post_fit = CSC_to_fit
                ECAL_post_fit = np.array([])


            h_xy_csc_sum.Add(h_xy_csc_0)
            h_xy_csc_sum.Add(h_xy_csc_1)
            h_yz_csc_sum.Add(h_yz_csc_0)
            h_yz_csc_sum.Add(h_yz_csc_1)
            h_xz_csc_sum.Add(h_xz_csc_0)
            h_xz_csc_sum.Add(h_xz_csc_1)

            h_etaphi_csc_sum.Add(h_etaphi_csc_0)
            h_etaphi_csc_sum.Add(h_etaphi_csc_1)
            h_etaz_csc_sum.Add(h_etaz_csc_0)
            h_etaz_csc_sum.Add(h_etaz_csc_1)
            h_phiz_csc_sum.Add(h_phiz_csc_0)
            h_phiz_csc_sum.Add(h_phiz_csc_1)


            h_xy_dt.SetMarkerColor(4)
            h_xy_dt.SetMarkerStyle(20)
            h_xy_csc_n.SetMarkerColor(1)
            h_xy_csc_n.SetMarkerStyle(20)
            h_xy_csc_0.SetMarkerColor(800)
            h_xy_csc_0.SetMarkerStyle(20)
            h_xy_csc_1.SetMarkerColor(800)
            h_xy_csc_1.SetMarkerStyle(21)

            h_etaphi_dt.SetMarkerColor(4)
            h_etaphi_dt.SetMarkerStyle(20)
            h_etaphi_csc_n.SetMarkerColor(1)
            h_etaphi_csc_n.SetMarkerStyle(20)
            h_etaphi_csc_0.SetMarkerColor(800)
            h_etaphi_csc_0.SetMarkerStyle(20)
            h_etaphi_csc_1.SetMarkerColor(800)
            h_etaphi_csc_1.SetMarkerStyle(21)

            #h_etaphi_cos.SetMarkerColor(801)
            #h_etaphi_cos.SetMarkerStyle(20)
            #h_etaphi_cos1L.SetMarkerColor(6)
            #h_etaphi_cos1L.SetMarkerStyle(20)
            #h_xy_csc_2.SetMarkerColor(7)
            #h_xy_csc_2.SetMarkerStyle(20)
            #h_xy_csc_3.SetMarkerColor(7)
            #h_xy_csc_3.SetMarkerStyle(21)

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
            fit_csc_xy.Fit(fit_xy,"E")
            #h_xy_csc_sum.Fit(fit_xy,"E")
            h_xy_csc_sum.SetMarkerColor(0)

            fit_yz = TF1("lin_fit_yz","pol1",0.,0.)
            print "FIT yz"
            fit_csc_yz.Fit(fit_yz,"E")
            #h_yz_csc_sum.Fit(fit_yz,"E")
            h_yz_csc_sum.SetMarkerColor(0)

            fit_xz = TF1("lin_fit_xz","pol1",0.,0.)
            print "FIT xz"
            fit_csc_xz.Fit(fit_xz,"E")
            #h_xz_csc_sum.Fit(fit_xz,"E")
            h_xz_csc_sum.SetMarkerColor(0)

            h_xyz_dt.SetMarkerColor(4)
            h_xyz_dt.SetMarkerStyle(20)
            h_xyz_ecal.SetMarkerColor(2)
            h_xyz_ecal.SetMarkerStyle(20)
            h_xyz_ecal_tag.SetMarkerColor(8)
            h_xyz_ecal_tag.SetMarkerStyle(24)

            h_xyz_csc_n.SetMarkerColor(1)
            h_xyz_csc_n.SetMarkerStyle(20)
            h_xyz_csc_0.SetMarkerColor(800)
            h_xyz_csc_0.SetMarkerStyle(20)
            h_xyz_csc_1.SetMarkerColor(800)
            h_xyz_csc_1.SetMarkerStyle(21)
            h_xyz_csc_2.SetMarkerColor(7)
            h_xyz_csc_2.SetMarkerStyle(20)
            h_xyz_csc_3.SetMarkerColor(7)
            h_xyz_csc_3.SetMarkerStyle(21)
            
            h_xz_dt.SetMarkerColor(4)
            h_xz_dt.SetMarkerStyle(20)
            h_xz_ecal.SetMarkerColor(2)
            h_xz_ecal.SetMarkerStyle(20)
            h_xz_ecal_tag.SetMarkerColor(8)
            h_xz_ecal_tag.SetMarkerStyle(24)

            h_xz_csc_n.SetMarkerColor(1)
            h_xz_csc_n.SetMarkerStyle(20)
            h_xz_csc_0.SetMarkerColor(800)
            h_xz_csc_0.SetMarkerStyle(20)
            h_xz_csc_1.SetMarkerColor(800)
            h_xz_csc_1.SetMarkerStyle(21)
            h_xz_csc_2.SetMarkerColor(7)
            h_xz_csc_2.SetMarkerStyle(20)
            h_xz_csc_3.SetMarkerColor(7)
            h_xz_csc_3.SetMarkerStyle(21)

        
            #h_xz_dt.Fit("pol1")

            h_yz_dt.SetMarkerColor(4)
            h_yz_dt.SetMarkerStyle(20)
            h_yz_ecal.SetMarkerColor(2)
            h_yz_ecal.SetMarkerStyle(20)
            h_yz_ecal_tag.SetMarkerColor(8)
            h_yz_ecal_tag.SetMarkerStyle(24)

            h_yz_csc_n.SetMarkerColor(1)
            h_yz_csc_n.SetMarkerStyle(20)
            h_yz_csc_0.SetMarkerColor(800)
            h_yz_csc_0.SetMarkerStyle(20)
            h_yz_csc_1.SetMarkerColor(800)
            h_yz_csc_1.SetMarkerStyle(21)
            h_yz_csc_2.SetMarkerColor(7)
            h_yz_csc_2.SetMarkerStyle(20)
            h_yz_csc_3.SetMarkerColor(7)
            h_yz_csc_3.SetMarkerStyle(21)

            #h_yz_dt.Fit("pol1")

            leg = TLegend(0.75, 0.8, 1., 1.)
            leg = TLegend(0.75, 0.7, 1., 1.)
            #leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
            leg.AddEntry(h_xy_dt,"CSC segments","P")
            leg.AddEntry(h_xy_ecal,"EB rec hits","P")
            leg.AddEntry(h_xy_ecal_tag,"EB rec hits; tag jets","P")
            #leg.AddEntry(h_xy_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
            leg.AddEntry(fit_csc_xy,"3D linear fit","PF")
            chi_xy = h_xy_ecal_tag.Chisquare(fit_xy)
            #leg.AddEntry(fit_xy,("2D line chi_sq:%.2f"%chi_xy),"L")
            leg.AddEntry(fit_xy,"2D line projection","L")
            chi_test_xy = fit_ecal_xy.Chi2Test(h_xy_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_xy_ecal_tag)#Chi2Test(h_xy_ecal_tag, "CHI2/NDF")#square(fit_xy)
            #leg.AddEntry(h_xy_ecal_tag,("2D chi_test:%.5f"%chi_test_xy),"L")
            leg.AddEntry(h_xy_csc_n,"DBSCAN noise: "+str(n_noise_),"P")

            can_xy = TCanvas("can_xy","can_xy",900,800)
            can_xy.cd()
            can_xy.SetGrid()
            can_xy.SetRightMargin(0.05)
            #can_xy.SetLogy()
            #h_xy_dt.Draw("")

            fit_csc_xy.GetXaxis().SetTitle("x (m)")
            fit_csc_xy.GetYaxis().SetTitle("y (m)")
            h_xy_csc_sum.GetXaxis().SetTitle("x (m)")
            h_xy_csc_sum.GetYaxis().SetTitle("y (m)")
            h_xy_csc_n.GetXaxis().SetTitle("x (m)")
            h_xy_csc_n.GetYaxis().SetTitle("y (m)")
            h_xy_dt.GetYaxis().SetTitleOffset(1.4)
            h_xy_ecal.GetXaxis().SetTitle("x (m)")
            h_xy_ecal.GetYaxis().SetTitle("y (m)")
            
            fit_csc_xy.SetLineColor(856)
            fit_csc_xy.SetMarkerColor(856)
            fit_csc_xy.SetMarkerStyle(20)
            fit_csc_xy.SetMarkerSize(0.4)#(0.4)
            fit_csc_xy.SetLineWidth(2)

            fit_ecal_xy.SetLineColor(801)
            fit_ecal_xy.SetMarkerColor(801)
            fit_ecal_xy.SetMarkerStyle(20)
            #fit_ecal_xy.SetMarkerSize(2.)#(0.4)
            fit_ecal_xy.SetLineWidth(2)

            fit_csc_xy.Draw("")
            #fit_ecal_xy.Draw("sames")
            h_xy_csc_sum.Draw("sames")
            h_xy_csc_n.Draw("sames")
            h_xy_csc_0.Draw("sames")
            h_xy_csc_1.Draw("sames")
            #h_xy_csc_2.Draw("sames")
            #h_xy_csc_3.Draw("sames")
            h_xy_ecal.Draw("sames")
            h_xy_ecal_tag.Draw("sames")
            #fit_xy.Draw("sames,L")

            
            OUTSTRING = OUT
            if isMC[ev]:
                OUTSTRING+="mc_"
            else:
                OUTSTRING+="data_"
            OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))+"_csc_fit_dr" 

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
            h_xy_csc_sum.Delete()
            h_xy_csc_n.Delete()
            h_xy_csc_0.Delete()
            h_xy_csc_1.Delete()
            h_xy_csc_2.Delete()
            h_xy_csc_3.Delete()
            fit_xy.Delete()
            h_xy_ecal.Delete()
            h_xy_ecal_tag.Delete()
            leg.Delete()

            can_xz = TCanvas("can_xz","can_xz",900,800)
            can_xz.cd()
            can_xz.SetGrid()
            can_xz.SetRightMargin(0.05)
            #can_xz.SetLogy()
            #h_xz_dt.Draw("")
            
            h_xz_csc_sum.GetXaxis().SetTitle("z (m)")
            h_xz_csc_sum.GetYaxis().SetTitle("x (m)")
            h_xz_csc_n.GetXaxis().SetTitle("z (m)")
            h_xz_csc_n.GetYaxis().SetTitle("x (m)")
            h_xz_dt.GetYaxis().SetTitleOffset(1.4)
            h_xz_ecal.GetXaxis().SetTitle("z (m)")
            h_xz_ecal.GetYaxis().SetTitle("x (m)")

            h_xz_csc_sum.Draw("")
            h_xz_csc_n.Draw("sames")
            h_xz_csc_0.Draw("sames")
            h_xz_csc_1.Draw("sames")
            #h_xz_csc_2.Draw("sames")
            #h_xz_csc_3.Draw("sames")
            h_xz_ecal.Draw("sames")
            h_xz_ecal_tag.Draw("sames")
            #fit_xz.Draw("sames,L")
            fit_csc_xz.SetLineColor(856)
            fit_csc_xz.SetMarkerColor(856)
            fit_csc_xz.SetMarkerStyle(20)
            fit_csc_xz.SetMarkerSize(0.4)
            fit_csc_xz.SetLineWidth(2)
            fit_ecal_xz.SetLineColor(801)
            fit_ecal_xz.SetMarkerColor(801)
            fit_ecal_xz.SetMarkerStyle(20)
            #fit_ecal_xz.SetMarkerSize(2.)#(0.4)
            fit_ecal_xz.SetLineWidth(2)

            fit_csc_xz.Draw("sames,PL")
            #fit_ecal_xz.Draw("sames,PL")


            h_xz_dt.GetXaxis().SetTitle("z (m)")
            h_xz_dt.GetYaxis().SetTitle("x (m)")
            h_xz_ecal.GetXaxis().SetTitle("z (m)")
            h_xz_ecal.GetYaxis().SetTitle("x (m)")
            h_xz_dt.GetYaxis().SetTitleOffset(1.4)
            leg = TLegend(0.75, 0.8, 1., 1.)
            #leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
            #leg.SetHeader(d)
            leg.AddEntry(h_xz_dt,"CSC segments","P")
            leg.AddEntry(h_xz_ecal,"EB rec hits","P")
            leg.AddEntry(h_xz_ecal_tag,"EB rec hits; tag jets","P")
            #leg.AddEntry(fit_csc_xz,("3D lin. fit; ssq:%.3f"%ssq),"PF")
            leg.AddEntry(fit_csc_xz,"3D linear fit","PF")
            chi_xz = h_xz_ecal_tag.Chisquare(fit_xz)
            leg.AddEntry(fit_xz,"2D line projection","L")
            #leg.AddEntry(fit_xz,("2D chi_sq:%.2f"%chi_xz),"L")
            chi_test_xz = fit_ecal_xz.Chi2Test(h_xz_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_xz_ecal_tag)#Chi2Test(h_xz_ecal_tag, "CHI2/NDF")#square(fit_xz)
            leg.AddEntry(h_xz_ecal_tag,("2D chi_test:%.5f"%chi_test_xz),"L")
            leg.AddEntry(h_xz_csc_n,"DBSCAN noise: "+str(n_noise_),"P")
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
            
            h_yz_csc_sum.GetXaxis().SetTitle("z (m)")
            h_yz_csc_sum.GetYaxis().SetTitle("y (m)")
            h_yz_csc_n.GetXaxis().SetTitle("z (m)")
            h_yz_csc_n.GetYaxis().SetTitle("y (m)")
            h_yz_ecal.GetXaxis().SetTitle("z (m)")
            h_yz_ecal.GetYaxis().SetTitle("y (m)")
            
            h_yz_csc_sum.Draw("")
            h_yz_csc_n.Draw("sames")
            h_yz_csc_0.Draw("sames")
            h_yz_csc_1.Draw("sames")
            #h_yz_csc_2.Draw("sames")
            #h_yz_csc_3.Draw("sames")
            h_yz_ecal.Draw("sames")
            h_yz_ecal_tag.Draw("sames")
            
            #fit_yz.Draw("sames,L")
            fit_csc_yz.SetLineColor(856)
            fit_csc_yz.SetMarkerColor(856)
            fit_csc_yz.SetMarkerStyle(20)
            fit_csc_yz.SetMarkerSize(0.4)
            fit_csc_yz.SetLineWidth(2)
            fit_csc_yz.Draw("sames,PL")
            
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
            #leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
            #leg.SetHeader(d)
            leg.AddEntry(h_yz_dt,"CSC segments","P")
            leg.AddEntry(h_yz_ecal,"EB rec hits","P")
            leg.AddEntry(h_yz_ecal_tag,"EB rec hits; tag jets","P")
            leg.AddEntry(fit_csc_yz,"3D linear fit","PF")
            #leg.AddEntry(fit_csc_yz,("3D lin. fit; ssq:%.3f"%ssq),"PF")
            chi_yz = h_yz_ecal_tag.Chisquare(fit_yz)
            #leg.AddEntry(fit_yz,("2D chi_sq:%.2f"%chi_yz),"L")
            leg.AddEntry(fit_yz,"2D line projection","L")
            chi_test_yz = fit_ecal_yz.Chi2Test(h_yz_ecal_tag,"CHI2/NDF")#KolmogorovTest(h_yz_ecal_tag)#Chi2Test(h_yz_ecal_tag, "CHI2/NDF")#square(fit_xy)
            leg.AddEntry(h_yz_ecal_tag,("2D chi_test:%.5f"%chi_test_yz),"L")
            leg.AddEntry(h_yz_csc_n,"DBSCAN noise: "+str(n_noise_),"P")
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

            can_xyz = TCanvas("can_xyz","can_xyz",900,800)
            can_xyz.cd()
            can_xyz.SetGrid()
            #can_xyz.SetRightMargin(0.05)
            #can_xyz.SetLogy()
            #h_xyz_dt.Draw("")
            fit_csc_3d.SetLineColor(856)
            fit_csc_3d.SetMarkerColor(856)
            fit_csc_3d.SetMarkerStyle(20)
            fit_csc_3d.SetMarkerSize(0.4)
            fit_csc_3d.SetLineWidth(2)
            fit_ecal_3d.SetLineColor(801)
            fit_ecal_3d.SetMarkerColor(801)
            fit_ecal_3d.SetMarkerStyle(20)
            #fit_ecal_3d.SetMarkerSize(2.)#(0.4)
            fit_ecal_3d.SetLineWidth(2)
            
            fit_csc_3d.Draw("")
            #fit_ecal_3d.Draw("sames")
            h_xyz_csc_n.Draw("sames")
            h_xyz_csc_0.Draw("sames")
            h_xyz_csc_1.Draw("sames")
            #h_xyz_csc_2.Draw("sames")
            #h_xyz_csc_3.Draw("sames")
            h_xyz_ecal.Draw("sames")
            h_xyz_ecal_tag.Draw("sames")

            
            fit_csc_3d.GetXaxis().SetTitle("z (m)")
            fit_csc_3d.GetYaxis().SetTitle("x (m)")
            fit_csc_3d.GetZaxis().SetTitle("y (m)")
            h_xyz_ecal.GetXaxis().SetTitle("z (m)")
            h_xyz_ecal.GetYaxis().SetTitle("x (m)")
            h_xyz_ecal.GetZaxis().SetTitle("y (m)")
            fit_csc_3d.GetXaxis().SetTitleOffset(1.4)
            fit_csc_3d.GetYaxis().SetTitleOffset(1.8)
            fit_csc_3d.GetZaxis().SetTitleOffset(1.4)
            leg = TLegend(0.75, 0.7, 1., 1.)
            #leg.SetHeader("nCosmics: "+str(nCosmics[ev])+"; one leg: "+str(nCosmicsOneLeg[ev]))
            #leg.SetHeader(d)
            leg.AddEntry(h_xyz_dt,"CSC segments","P")
            leg.AddEntry(h_xyz_ecal,"EB rec hits","P")
            leg.AddEntry(h_xyz_ecal_tag,"EB rec hits; tag jets","P")
            #leg.AddEntry(h_xyz_dt,"n. DBSCAN clusters: "+str(n_clusters_),"")
            #leg.AddEntry(h_xyz_csc_n,"n. noise: "+str(n_noise_),"P")
            #leg.AddEntry(fit_csc_3d,("3D lin. fit; ssq:%.3f"%ssq),"PF")
            leg.AddEntry(fit_csc_3d,"3D linear fit","PF")
            #leg_string = (("2D chi_sq:(%.2f,%.2f,%.2f)")%(chi_xy,chi_xz,chi_yz))
            #leg.AddEntry(fit_csc_3d,str(leg_string),"L")
            leg.AddEntry(h_xyz_csc_n,"DBSCAN noise: "+str(n_noise_),"P")
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
            h_yz_csc_n.Delete()
            h_yz_csc_0.Delete()
            h_yz_csc_1.Delete()
            h_yz_csc_2.Delete()
            h_yz_csc_3.Delete()
            h_yz_csc_sum.Delete()
            h_xz_csc_n.Delete()
            h_xz_csc_0.Delete()
            h_xz_csc_1.Delete()
            h_xz_csc_2.Delete()
            h_xz_csc_3.Delete()
            h_xz_csc_sum.Delete()
            h_xyz_csc_n.Delete()
            h_xyz_csc_0.Delete()
            h_xyz_csc_1.Delete()
            h_xyz_csc_2.Delete()
            h_xyz_csc_3.Delete()
            h_xyz_csc_sum.Delete()
            fit_csc_3d.Delete()
            fit_csc_xy.Delete()
            fit_csc_xz.Delete()
            fit_csc_yz.Delete()

event_display()
#csc_fit()
