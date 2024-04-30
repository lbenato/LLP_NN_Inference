#!/usr/bin/env python
import os, re
import multiprocessing
import logging
import commands
import math, time
import sys
from ROOT import TObject, TFile, TH1, TH1F
#from Analyzer.LLP2018.samples import sample
from array import array


# use the following lists to include/exclude samples to be merged

blacklist = []
whitelist = []

#TIP = "/pnfs/desy.de/cms/tier2/store/user/lbenato/"
#DEST = "/nfs/dust/cms/user/lbenato/v1/"


########## DO NOT TOUCH BELOW THIS POINT ##########
###optparse
import optparse

usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-l", "--lists", action="store", type="string", dest="lists", default="v4_calo_AOD_2018")
parser.add_option("-i", "--input_folder", action="store", type="string", dest="input_folder", default="", help='the input folder containing the CRAB outputs')
parser.add_option("-o", "--output_folder", action="store", type="string", dest="output_folder", default="", help='the output folder containing the hadd of CRAB outputs')
parser.add_option("-g", "--groupofsamples", action="store", type="string", dest="groupofsamples", default="")
(options, args) = parser.parse_args()


if options.lists == "v4_calo_AOD_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    #LUMI = 13967.832433416# METRun2018A-17Sep2018-v1 
    LUMI = 59.74*1000
    #LUMI = 59690#2018 lumi with normtag, from pdvm2018 twiki, PromptReco
elif options.lists == "debug":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    LUMI = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#.....
elif options.lists == "v5_SingleMuon_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["SingleMuon"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
    #LUMI = (2.090881769 + 3.121721023 + 2.909790309 + 0.385165352)*1000#resubm 0
    #LUMI+= (7.662822707 + 0.028465764 + 3.969519137 + 0.426230555)*1000#resubm_1
    #LUMI+= (0.061722871 + 0.016999521 + 0.015461525 + 9.167168975)*1000#resubm_2
    #LUMI+= (4.161906683 +                             5.726058201)*1000#resubm_3
    #LUMI+= (                                          5.604608939)*1000#resubm_4

elif options.lists == "v5_EGamma_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["EGamma"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
 
    #total of D should be 31.687229183

    #D resubm0 from crab:
    #to process: 31.687229183
    #processed: 0.884947674
    #not finished: 30.802281509
    #stored in dataJSON: 16.760030375 #will  be picked from resubm1 --> this seems random!

    #in dataJSON resubm1:
    #first part of not finished: 8.279307279
    #second part of not finished: 14.042251135
    #thing passed to 2: 8.279307279
    #thing passed to 4: 14.04

    #D resubm1 from crab: dead
    #input from dataJSON: 16.760030375?
    #should include resubm0 fraction

    #D resubm2 from crab:
    #input from dataJSON: 8.279307279
    #should include resubm1 failures
    #to process: 8.279307279
    #processed: 3.105890490
    #not finished: 5.173416789

    #D resubm3 from crab:
    #input from dataJSON: 5.173416789 
    #should include resubm2 failures

    #D resubm4 from crab:
    #input from dataJSON: 14.042251135
    #includes  resubm0 second part

    #D resubm5 from crab:
    #from dataJSON: 5.645825019
    #includes failures from resubm4

    #from dataAOD: 16.760030375 vs 

    #             A              B           C          D
    #LUMI = ( 3.062045650 + 3.517958038 + 3.185107178 + 0.884947674)*1000#resubm 0
    #LUMI+= ( 4.310108316 + 3.522423496 + 3.685531373 + 8.480723096)*1000#resubm_1  --> crab folder lost, extrapolated from input_1 - processed23 
    #LUMI+= ( 2.463000080 + 0.017443625 + 0.024132420 + 3.105890490)*1000#resubm_2
    #LUMI+= ( 3.519520778 +                             5.173416789)*1000#resubm_3
    #LUMI+= ( 0.441589640 +                             8.396426116)*1000#resubm_4
    #LUMI+= (                                           5.645825019)*1000#resubm_5
    #LUMI_A = (3.062045650+4.310108316+2.463000080+3.519520778+0.441589640)*1000
    #print "total LUMI: ", LUMI
elif options.lists == "v5_MuonEG_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["MuonEG"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
    #LUMI = ( 13.977334030 + 7.043373660 + 6.894770971 + 31.436052907)*1000#resubm 0
    #LUMI+= (                0.014149433 +               0.306926432)*1000#resubm_1
    #print "total LUMI: ", LUMI
elif options.lists == "v5_MET_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["MET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
    #LUMI = ( 11.883435811 + 4.185775977 + 6.790485785 + 22.398936830 )*1000#resubm 0
    #LUMI+= ( 2.091027226 +  2.872196207 + 0.104296295 + 9.341251370)*1000#resubm_1
    #LUMI+= (                                            0.002792445 )*1000#resubm_2
    #print "total LUMI: ", LUMI

elif options.lists == "v5_HighMET_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI

elif options.lists == "v5_JetHT_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["JetHT"]["tot"]#["tot"]
    print "total LUMI: ", LUMI

elif options.lists == "v5_SingleMuon_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["SingleMuon"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_SingleElectron_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["SingleElectron"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_MuonEG_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["MuonEG"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_HighMET_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_JetHT_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["JetHT"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_SinglePhoton_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["SinglePhoton"]["tot"]#["tot"]
    print "total LUMI: ", LUMI

##
elif options.lists == "v5_SingleMuon_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["SingleMuon"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_SingleElectron_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["SingleElectron"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_MuonEG_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["MuonEG"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_HighMET_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_JetHT_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["JetHT"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v5_SinglePhoton_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["SinglePhoton"]["tot"]#["tot"]
    print "total LUMI: ", LUMI

elif options.lists == "v5_calo_AOD_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    #LUMI = 13967.832433416# METRun2018A-17Sep2018-v1 
    LUMI = 95.9*1000#Princeton
    #LUMI = 59.74*1000
    #LUMI = 59690#2018 lumi with normtag, from pdvm2018 twiki, PromptReco
elif options.lists == "v5_calo_AOD_2017_compare_CSC":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    #LUMI = 13967.832433416# METRun2018A-17Sep2018-v1 
    LUMI = 137*1000#Princeton
    #LUMI = 59.74*1000
    #LUMI = 59690#2018 lumi with normtag, from pdvm2018 twiki, PromptReco


elif options.lists == "v6_HighMET_2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    print "!! Using v5 lumi!!!"
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v6_HighMET_2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import *
    print "!! Using v5 lumi!!!"
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI
elif options.lists == "v6_HighMET_2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import *
    print "!! Using v5 lumi!!!"
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI = lumi["HighMET"]["tot"]#["tot"]
    print "total LUMI: ", LUMI

elif options.lists == "v6_cosmics":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import *
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    LUMI = 1.

else:
    print("No sample list indicated, aborting!")
    exit()


print("*****************************************************************************")
print("\n")
print("Please input the correct lumi!!!")
print("\n")
print(LUMI, " fb -1")
print("*****************************************************************************")


list_of_samples = ["SM_Higgs","VV","WJetsToQQ","WJetsToLNu","WJetsToLNu_Pt","DYJetsToQQ","DYJetsToNuNu","DYJetsToLL","ST","TTbar","QCD","signal_VBF","signal_ggH","all","data_obs","SingleMuon","ZJetsToNuNu","DYJets","WJets","signal_ZH","ZJetsToNuNuRed","SUSY","TTbarSemiLep","TTbarNu","ggHeavyHiggs","WJetsToLNu_HT","WJetsToLNuIncl","JetJet","splitSUSY","gluinoGMSB","EGamma","TTbarGenMET","MuonEG","MET","HighMET","JetHT","SingleElectron","SinglePhoton","SUSY_HH","SUSY_HZ","SUSY_ZZ","Cosmics","CosmicsMC","HeavyHiggsToLLP","HeavyHiggsToLLPTo4b_mH_400","zPrime","Stealth"]
print("Possible subgroups of samples:")
for a in list_of_samples:
    print(a)
    print("---------------")

#print requests.keys()


selected_requests = {}
if options.groupofsamples not in list_of_samples:
    print("Invalid subgroup of samples, aborting!")
    exit()

for b, k in enumerate(requests.keys()):
    if options.groupofsamples=="signal_VBF":
        if "VBFH_HToSSTobb" in k:
            print(k)
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="signal_ggH":
        if "GluGluH_HToSSTobb" in k:
            print(k)
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="signal_ZH":
        if "ZH_HToSSTobb" in k:
            print(k)
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="SUSY":
        if "n3n2-n1-hbb-hbb" in k:
            print(k)
            selected_requests[k] = requests[k]
        elif "TChi" in k:
            print(k)
            selected_requests[k] = requests[k]
        elif "SMS" in k:
            print(k)
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="SUSY_HH":
        if "00_HH" in k:
            if "TChi" in k:
                print(k)
                selected_requests[k] = requests[k]
    elif options.groupofsamples=="SUSY_HZ":
        if "00_HZ" in k:
            if "TChi" in k:
                print(k)
                selected_requests[k] = requests[k]
    elif options.groupofsamples=="SUSY_ZZ":
        if "00_ZZ" in k:
            if "TChi" in k:
                print(k)
                selected_requests[k] = requests[k]
    elif options.groupofsamples=="ggHeavyHiggs":
        if "GluGluH2_H2ToSSTobb" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="JetJet":
        if "XXTo4J" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="splitSUSY":
        if "GluinoGluinoToNeutralinoNeutralinoTo2T2B2S" in k:
            print k
            selected_requests[k] = requests[k]
        if "splitSUSY" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="gluinoGMSB":
        if "gluinoGMSB" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="HeavyHiggsToLLP":
        if "HeavyHiggsToLLP" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="Stealth":
        if "Stealth" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="HeavyHiggsToLLPTo4b_mH_400":
        if "HeavyHiggsToLLPTo4b_mH_400" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="zPrime":
        if "zPrime" in k:
            print k
            selected_requests[k] = requests[k]
    elif options.groupofsamples=="all":
        print("All samples considered")
        selected_requests[k] = requests[k]
    else:
        if k in samples[options.groupofsamples]["files"]:
            print(k)
            selected_requests[k] = requests[k]

if options.output_folder == "":
    DEST = "/nfs/dust/cms/user/lbenato/v3/"
else:
    DEST = options.output_folder+'/'

if not os.path.exists(os.path.expandvars(options.input_folder)):
    print('--- ERROR: INPUT FOLDER ---')
    print('  \''+options.input_folder+'\' path not found')
    print('  please point to the correct path to the folder containing the CRAB output' )
    exit()

if not os.path.exists(os.path.expandvars(DEST)):
    print('--- ERROR: OUTPUT FOLDER ---')
    print('  \''+DEST+'\' path not found')
    print('  please point to the correct output path' )
    exit()


#########

jobs = []
names = []

def hadd_outputs(fold,name):
    if "_PRIVATE-MC" in name:
        short_name = name[:-11]
    else:
        short_name = name

######################This blocks naf machines
    #print name
    #os.system('hadd -fk207 '+DEST+name+'.root ' + fold + "/*/*.root")#timestamp for calo_signal!
    #no compression, goes faster
    os.system('hadd -f '+DEST+name+'.root ' + fold + "/*/*.root")#timestamp for calo_signal!
pass

def weight(name):
    weight = 1.
    filename = TFile(DEST+name+'.root', "UPDATE")
    if ('Run201') in name: weight = 1.
    ###
    else:
        nevents     = filename.Get("c_nEvents").GetBinContent(1)
        if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in name:
            print "SUSY central, consider sample dictionary for nevents!"
            nevents = sample[name]['nevents']
            #will be: sample[name]['nevents']
        b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
        n_pass      = filename.Get("n_pass").GetBinContent(1)
        n_odd       = filename.Get("n_odd").GetBinContent(1)            

        if ('VBFH_HToSSTobbbb') in name:
            #We take into account VBF Higgs production x-sec
            xs = 3.782
        elif('GluGluH_HToSSTobbbb') in name:
            #We take into account ggH Higgs production x-sec
            xs = 48.58
        elif('ZH_HToSSTobbbb') in name:
            #We take into account ZH Higgs production x-sec times branching fraction into leptons
            xs = 0.8839*(3.3658/100.)
        elif('n3n2-n1-hbb-hbb') in name:
            print "Scaling SUSY to 1. for absolute x-sec sensitivity"
            #Don't know this x-sec actually...
            print "This is susy name: ", name
            xs = 1.
        elif('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in name:
            print "Scaling SUSY to 1. for absolute x-sec sensitivity"
            #print "But consider BR!"
            xs = 1.
            #xs *= sample[name]['BR']
        elif('GluGluH2_H2ToSSTobbbb') in name:
            #We do not take into account ggH Higgs production x-sec! Absolute x-sec needed!
            xs = 1.#48.58
        elif('XXTo4J') in name:
            xs = 1.
        elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in name:
            xs = 1. 
        elif('gluinoGMSB') in name:
            xs = 1.
        elif('splitSUSY') in name:
            xs = 1.
        elif('HeavyHiggsToLLP') in name:
            xs = 1.
        elif('Stealth') in name:
            xs = 1.
        elif('zPrime') in name:
            xs = 1.
            print "\n"
            print "HERE!!"
            print "Remember MAtthew's corrections!!!"
            print "\n"
        else:
            xs = sample[name]['xsec'] * sample[name]['kfactor']#to correct MET phase-space
        weight = LUMI * xs / nevents

        
        if(b_skipTrain>0):
            print("Skip even events: ")
            print "n_pass: ", n_pass
            print "n_odd: ", n_odd
            if(n_pass>0):
                print "ratio: ", float(n_odd/n_pass)
            #weight: divide by this factor above
            if(n_odd>0):
                print "even/odd weight: ", float(n_pass/n_odd)
                weight *= float(n_pass/n_odd)
        

        #tree = filename.Get("ntuple/tree")
        tree = filename.Get("tree")
        print(name)
        print "LUMI ", LUMI
        print "xs ", xs
        print "nevents ", nevents
        print "weight ", weight
        tree.SetWeight(weight)
        tree.AutoSave()


subdirs = [x for x in os.listdir(options.input_folder) if os.path.isdir(os.path.join(options.input_folder, x))]

#subdir_renamed = []
#for l in subdirs:
#    if l=="DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8-v1":
#        l = "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia-v1"
#    subdir_renamed.append(l)
#subdirs = subdir_renamed
    
#subdirs have the names of the samples without v1, etc

#print "PRE: ", selected_requests.keys()

#for naming purposes, we have to include v1, etc. Additional loop###
os.chdir(options.input_folder)

#crab_subdirs = []
#for l in subdirs:
#    #no need to ignore crab_
#    #crab_subdirs += [x[5:] for x in os.listdir(l) if os.path.isdir(os.path.join(l, x))]
#    crab_subdirs += [x for x in os.listdir(l) if os.path.isdir(os.path.join(l, x))]

print "subdirs:"
print subdirs
#print crab_subdirs
#exit()
#here they have the proper names, including v1

#print "selected requests"
#print selected_requests.keys()
os.chdir(options.input_folder)
for a in subdirs:
    fold = ""
    name = ""
    print a
    if a in selected_requests.keys():
        fold = a
        name = a
        print("Being added....")
        print(name)
        #print("Not being added....")
        hadd_outputs(fold,name)
        print("##################################")

#exit()

######################
#
#    Multiprocessing stucked naf machines, avoid - also, not tested with optparse
#
#    p = multiprocessing.Process(target=hadd_outputs, args=(fold,name))
#    jobs.append(p)
#    p.start()
######################
            
    #hadd_outputs(fold,name)

print "Ntuples ready in ", DEST 
os.system('cd '+DEST+".. ")


onlyfiles = [f for f in os.listdir(DEST) if (os.path.isfile(os.path.join(DEST, f)))]
os.chdir(DEST)

for b in onlyfiles:
    #print b
    if b[:-5] in selected_requests.keys():
        #print b
        print("I am going to weight:")
        weight(b[:-5])

        #print "I am just pretending to weight:"

        ##q = multiprocessing.Process(target=weight, args=(b[:-5],))
        ##jobs.append(q)
        ##q.start()                                                                                                                                                                         
print "Ntuples weighted in ", DEST
