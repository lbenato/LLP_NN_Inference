#!/usr/bin/env python

import os
import ROOT as ROOT

ERA="2016"

#region = ["WtoEN_noMT","ZtoEE"]
#region = ["EN"]
#region = ["WtoEN_noMT_InvertBeamHalo"]
#data = ["SingleElectron"]
#data =  ["EGamma"]

#region = ["TtoEM"]
#data  = ["MuonEG"]

#region = ["Pho"]
#data  = ["EGamma"]

#region = ["DiJetMET"]
#region = ["JetHT_InvertBeamHalo"]
#region = ["JetHT"]
#data  = ["JetHT"]

#region = ["WtoMN_noMT","ZtoMM"]
#region=["ZtoMMPho"]
#region = ["MN"]
#region = ["WtoMN_noMT"]
#region = ["WtoMN_noMT_InvertBeamHalo"]
#data = ["SingleMuon"]

data = {}

#data["SingleMuon"] = ["ZtoMMPho"]
#if ERA=="2018":
#    data["EGamma"] = ["ZtoEEPho"]
#else:
#    data["SingleElectron"] = ["ZtoEEPho"]

#data["SingleElectron"] = ["E"]
#data["EGamma"] = ["E"]

data["SingleElectron"] = ["ZtoEEPho"]
#data["EGamma"] = ["ZtoEEPho"]
#data["SingleMuon"] = ["ZtoMMPho"]

DATADIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_%s_v5_ntuples_updated/"

#s = REGION_resubmission_x if data
#s = REGION if MC

if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples

for d in data.keys():
    for r in data[d]:
        #DIR = DATADIR + r #+ "_tranche4"#_debug_condor_again_"
        for s in samples[d]["files"]:
            print "\n"
            print s
            DEST = (DATADIR % r)
            INP = (DATADIR % (r+"_resubmission_*"))
            ###os.system("echo hadd -fk207 "+DIR+"/"+s+".root "+DIR+"_resubmission_*/"+s+".root \n")
            os.system("echo hadd -f "+DEST+s+".root "+INP+s+".root \n")
            os.system("hadd -f "+DEST+s+".root "+INP+s+".root \n")
