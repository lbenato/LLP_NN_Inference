#!/usr/bin/env python

import os
import ROOT as ROOT

ERA = "2017"

#region = ["EN"]
#data = ["EGamma"]
#region = ["WtoEN_noMT_BeamHalo"]
#region = ["WtoEN_noMT"]
#region = ["ZtoEE"]
#data  = ["SingleElectron"]

#region = ["WtoMN_noMT"]
#region = ["WtoMN_noMT_InvertBeamHalo"]
#data  = ["SingleMuon"]

#region = ["MN"]
#region = ["ZtoMM"]
region = ["WtoMN"]
#region = ["WtoMN_noMT_"]
data  = ["SingleMuon"]

#region = ["DiJetMET"]
#region = ["JetHT_BeamHalo"]
#region = ["JetHT"]
#data  = ["JetHT"]
#region = ["JetHT_InvertBeamHalo"]

#region = ["TtoEM"]
#data = ["MuonEG"]

#resub = ["5","4","3","2","1","0"]

DATADIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_"+ERA+"_"

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-r", "--resubm", action="store", type="string", dest="resubm", default="6")
(options, args) = parser.parse_args()

for d in data:
    for r in region:
        DIR = DATADIR + r #+ "_tranche4"#
        print "\n"
        os.system("echo python utils/merge_outputs_condor.py -l v5_"+d+"_"+ERA+" -g "+d+" -i "+DIR+"_resubmission_"+options.resubm+"/ -o "+DIR+"_resubmission_"+options.resubm+"/ \n")
        os.system("python utils/merge_outputs_condor.py -l v5_"+d+"_"+ERA+" -g "+d+" -i "+DIR+"_resubmission_"+options.resubm+"/ -o "+DIR+"_resubmission_"+options.resubm+"/ \n")
        print "\n"
        #os.system("echo hadd -fk207 "+DIR+"/"+s+".root "+DIR+"_resubmission_*/"+s+".root \n")
        #os.system("hadd -fk207 "+DIR+"/"+s+".root "+DIR+"_resubmission_*/"+s+".root \n")
