#!/usr/bin/env python

import os
import subprocess
import ROOT as ROOT
import multiprocessing
from collections import defaultdict
import numpy as np
import uproot
import root_numpy

run_condor = True

# # # # # #
# Run era #
# # # # # #
RUN_ERA = 2017#6#7#8

doRegion = "doSR"#"doGen"#"doSR"#"doSR"#"doTtoEM"#"doZtoEE"#MN/EN
resubm_label = ""
#resubm_label = "_resubmission_4"

##
if resubm_label=="_resubmission_0" or resubm_label=="":
    if RUN_ERA == 2018:
        INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_HEM/")% str(RUN_ERA)
        #JERUp
        #INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_HEM_JERUp/")% str(RUN_ERA)
        ##INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_15November2021_slimmedJets_HEM/")% str(RUN_ERA)
    else:
        INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021/")% str(RUN_ERA)
        #JERUp
        #INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_JERUp/")% str(RUN_ERA)
        #Stealth SHH:
        #INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SHH/")% str(RUN_ERA)

else:
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_07October2021"+resubm_label+"/")% str(RUN_ERA)


OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR/")%(RUN_ERA)
OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_time_smeared_compare_SUSY_HH/")%(RUN_ERA)
OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_time_smeared/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_time_smeared_no_cuts_debug/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_no_cosmicOneLeg/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_signal_smearing_test/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_tagger_v3_pt_flat/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_positive_jets_0p7/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_BeamHalo/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_no_photonEFrac_eleEFrac_cut/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_syst_unc_JERUp/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_syst_unc_central_values/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_no_cuts_syst_unc_central_values/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_PDF_QCD_syst_unc/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_bin_1_2/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen_HZ/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_xcheck_slimmedJets/")%(RUN_ERA)

if not(os.path.exists(OUTPUTDIR)):
    os.mkdir(OUTPUTDIR)

data_file = ("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_69200_%s.root") % str(RUN_ERA)
data_file_up = ("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_72380_%s.root") % str(RUN_ERA)
data_file_down = ("/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/dataAOD/PU_66020_%s.root") % str(RUN_ERA)

if RUN_ERA==2018:
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2018 import *
    pu_tag = "Fall18_2018_calo"
    tune = "TuneCP5"
    print samples.keys()

elif RUN_ERA==2017:
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2017 import requests
    pu_tag = "Fall17_2017_calo"
    tune = "TuneCP5"

elif RUN_ERA==2016:
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.crab_requests_lists_calo_AOD_2016 import requests
    pu_tag = "Summer16_2016_calo"
    tune = "TuneCUETP8M1"

else:
    print "Invalid run era, aborting..."
    exit()


dicty_o = defaultdict()


#data = ["EGamma"]
#data = ["SingleMuon"]
#data = ["SingleElectron"]
#data = ["SinglePhoton"]
#data = ["MuonEG"]
#data = ["MET"]
data = ["HighMET"]
#data = ["JetHT"]
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbarGenMET","QCD"]
#back = ["TTbar"]
#back = ["DYJetsToLL"]
#back = ["TTbarGenMET"]
#back = ["QCD"]
#back = ["ZJetsToNuNu"]
#back = ["VV"]#["TTbar"]#"VV",
#back = ["WJetsToLNu"]#done


sign = [
        #'SUSY_mh200_pl1000','SUSY_mh400_pl1000_XL',
        #'splitSUSY_M-2400_CTau-300mm','splitSUSY_M-2400_CTau-1000mm','splitSUSY_M-2400_CTau-10000mm','splitSUSY_M-2400_CTau-30000mm',
        #'gluinoGMSB_M2400_ctau1','gluinoGMSB_M2400_ctau3','gluinoGMSB_M2400_ctau10','gluinoGMSB_M2400_ctau30','gluinoGMSB_M2400_ctau100','gluinoGMSB_M2400_ctau300','gluinoGMSB_M2400_ctau1000','gluinoGMSB_M2400_ctau3000','gluinoGMSB_M2400_ctau10000','gluinoGMSB_M2400_ctau30000','gluinoGMSB_M2400_ctau50000',
        #'XXTo4J_M100_CTau100mm','XXTo4J_M100_CTau300mm','XXTo4J_M100_CTau1000mm','XXTo4J_M100_CTau3000mm','XXTo4J_M100_CTau50000mm',
        #'XXTo4J_M300_CTau100mm','XXTo4J_M300_CTau300mm','XXTo4J_M300_CTau1000mm','XXTo4J_M300_CTau3000mm','XXTo4J_M300_CTau50000mm',
        #'XXTo4J_M1000_CTau100mm','XXTo4J_M1000_CTau300mm','XXTo4J_M1000_CTau1000mm','XXTo4J_M1000_CTau3000mm','XXTo4J_M1000_CTau50000mm',
        #2017:
        #'SUSY_mh400_pl1000', 'SUSY_mh300_pl1000', 'SUSY_mh250_pl1000', 'SUSY_mh200_pl1000', 'SUSY_mh175_pl1000', 'SUSY_mh150_pl1000', 'SUSY_mh127_pl1000',
        #'SUSY_central',
        'SUSY_mh127_ctau500', 'SUSY_mh127_ctau3000',
        'SUSY_mh150_ctau500', 'SUSY_mh150_ctau3000',
        'SUSY_mh175_ctau500', 'SUSY_mh175_ctau3000',
        'SUSY_mh200_ctau500', 'SUSY_mh200_ctau3000',
        'SUSY_mh250_ctau500', 'SUSY_mh250_ctau3000',
        'SUSY_mh300_ctau500', 'SUSY_mh300_ctau3000',
        'SUSY_mh400_ctau500', 'SUSY_mh400_ctau3000',
        'SUSY_mh600_ctau500', 'SUSY_mh600_ctau3000',
        'SUSY_mh800_ctau500', 'SUSY_mh800_ctau3000',
        'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
        'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
        'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
        'SUSY_mh1800_ctau500','SUSY_mh1800_ctau3000',

        #'SUSY_mh127_ctau500_HZ', 'SUSY_mh127_ctau3000_HZ',
        #'SUSY_mh150_ctau500_HZ', 'SUSY_mh150_ctau3000_HZ',
        #'SUSY_mh175_ctau500_HZ', 'SUSY_mh175_ctau3000_HZ',
        #'SUSY_mh200_ctau500_HZ', 'SUSY_mh200_ctau3000_HZ',
        #'SUSY_mh250_ctau500_HZ', 'SUSY_mh250_ctau3000_HZ',
        #'SUSY_mh300_ctau500_HZ', 'SUSY_mh300_ctau3000_HZ',
        #'SUSY_mh400_ctau500_HZ', 'SUSY_mh400_ctau3000_HZ',
        #'SUSY_mh600_ctau500_HZ', 'SUSY_mh600_ctau3000_HZ',
        #'SUSY_mh800_ctau500_HZ', 'SUSY_mh800_ctau3000_HZ',
        #'SUSY_mh1000_ctau500_HZ','SUSY_mh1000_ctau3000_HZ',
        #'SUSY_mh1250_ctau500_HZ','SUSY_mh1250_ctau3000_HZ',
        #'SUSY_mh1500_ctau500_HZ','SUSY_mh1500_ctau3000_HZ',
        #'SUSY_mh1800_ctau500_HZ','SUSY_mh1800_ctau3000_HZ',

        #'SUSY_mh127_ctau500_ZZ', 'SUSY_mh127_ctau3000_ZZ',
        #'SUSY_mh150_ctau500_ZZ', 'SUSY_mh150_ctau3000_ZZ',
        #'SUSY_mh175_ctau500_ZZ', 'SUSY_mh175_ctau3000_ZZ',
        #'SUSY_mh200_ctau500_ZZ', 'SUSY_mh200_ctau3000_ZZ',
        #'SUSY_mh250_ctau500_ZZ', 'SUSY_mh250_ctau3000_ZZ',
        #'SUSY_mh300_ctau500_ZZ', 'SUSY_mh300_ctau3000_ZZ',
        #'SUSY_mh400_ctau500_ZZ', 'SUSY_mh400_ctau3000_ZZ',
        #'SUSY_mh600_ctau500_ZZ', 'SUSY_mh600_ctau3000_ZZ',
        #'SUSY_mh800_ctau500_ZZ', 'SUSY_mh800_ctau3000_ZZ',
        #'SUSY_mh1000_ctau500_ZZ','SUSY_mh1000_ctau3000_ZZ',
        #'SUSY_mh1250_ctau500_ZZ','SUSY_mh1250_ctau3000_ZZ',
        #'SUSY_mh1500_ctau500_ZZ','SUSY_mh1500_ctau3000_ZZ',
        #'SUSY_mh1800_ctau500_ZZ','SUSY_mh1800_ctau3000_ZZ',

        #'SUSY_mh400_prompt', 'SUSY_mh300_prompt', 'SUSY_mh200_prompt',
        #'splitSUSY_M2400_100_ctau0p1','splitSUSY_M2400_100_ctau1p0','splitSUSY_M2400_100_ctau10p0','splitSUSY_M2400_100_ctau100p0','splitSUSY_M2400_100_ctau1000p0','splitSUSY_M2400_100_ctau10000p0','splitSUSY_M2400_100_ctau100000p0',
        #'SUSY_mh1800_ctau500',
        #'ggH_MH125_MS25_ctau500',  'ggH_MH125_MS25_ctau1000',  'ggH_MH125_MS25_ctau2000',  'ggH_MH125_MS25_ctau5000',  'ggH_MH125_MS25_ctau10000', 
        #'ggH_MH125_MS55_ctau500',  'ggH_MH125_MS55_ctau1000',  'ggH_MH125_MS55_ctau2000',  'ggH_MH125_MS55_ctau5000',  'ggH_MH125_MS55_ctau10000', 
        #'ggH_MH200_MS50_ctau500',  'ggH_MH200_MS50_ctau1000',  'ggH_MH200_MS50_ctau2000',  'ggH_MH200_MS50_ctau5000',  'ggH_MH200_MS50_ctau10000', 
        #'ggH_MH200_MS25_ctau500',  'ggH_MH200_MS25_ctau1000',  'ggH_MH200_MS25_ctau2000',  'ggH_MH200_MS25_ctau5000',  'ggH_MH200_MS25_ctau10000', 
        #'ggH_MH400_MS100_ctau500', 'ggH_MH400_MS100_ctau1000', 'ggH_MH400_MS100_ctau2000', 'ggH_MH400_MS100_ctau5000', 'ggH_MH400_MS100_ctau10000',
        #'ggH_MH400_MS50_ctau500',  'ggH_MH400_MS50_ctau1000',  'ggH_MH400_MS50_ctau2000',  'ggH_MH400_MS50_ctau5000',  'ggH_MH400_MS50_ctau10000',
        #'ggH_MH600_MS150_ctau500', 'ggH_MH600_MS150_ctau1000', 'ggH_MH600_MS150_ctau2000', 'ggH_MH600_MS150_ctau5000', 'ggH_MH600_MS150_ctau10000',
        #'ggH_MH600_MS50_ctau500',  'ggH_MH600_MS50_ctau1000',  'ggH_MH600_MS50_ctau2000',  'ggH_MH600_MS50_ctau5000',  'ggH_MH600_MS50_ctau10000',
	#'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000',
	#'ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
	#'ggH_MH1500_MS200_ctau500','ggH_MH1500_MS200_ctau1000','ggH_MH1500_MS200_ctau2000','ggH_MH1500_MS200_ctau5000','ggH_MH1500_MS200_ctau10000',
	#'ggH_MH1500_MS500_ctau500','ggH_MH1500_MS500_ctau1000','ggH_MH1500_MS500_ctau2000','ggH_MH1500_MS500_ctau5000','ggH_MH1500_MS500_ctau10000',
	#'ggH_MH2000_MS250_ctau500','ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS250_ctau2000','ggH_MH2000_MS250_ctau5000','ggH_MH2000_MS250_ctau10000',
	#'ggH_MH2000_MS600_ctau500','ggH_MH2000_MS600_ctau1000','ggH_MH2000_MS600_ctau2000','ggH_MH2000_MS600_ctau5000','ggH_MH2000_MS600_ctau10000',
        ]

#sign = ['SUSY_mh400_ctau500','SUSY_mh400_ctau3000',]


sign+= [
        'SUSY_mh127_ctau500', 'SUSY_mh127_ctau3000',
        'SUSY_mh150_ctau500', 'SUSY_mh150_ctau3000',
        'SUSY_mh175_ctau500', 'SUSY_mh175_ctau3000',
        'SUSY_mh200_ctau500', 'SUSY_mh200_ctau3000',
        'SUSY_mh250_ctau500', 'SUSY_mh250_ctau3000',
        'SUSY_mh300_ctau500', 'SUSY_mh300_ctau3000',
        'SUSY_mh400_ctau500', 'SUSY_mh400_ctau3000',
        'SUSY_mh600_ctau500', 'SUSY_mh600_ctau3000',
        'SUSY_mh800_ctau500', 'SUSY_mh800_ctau3000',
        'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
        'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
        'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
        'SUSY_mh1800_ctau500','SUSY_mh1800_ctau3000',
]

sign = [
    'zPrimeTo4b_mZ3000_mX1450_ct0p1', 'zPrimeTo4b_mZ3000_mX1450_ct1', 'zPrimeTo4b_mZ3000_mX1450_ct10', 'zPrimeTo4b_mZ3000_mX1450_ct100', 'zPrimeTo4b_mZ3000_mX1450_ct1000', 'zPrimeTo4b_mZ3000_mX1450_ct10000', 'zPrimeTo4b_mZ3000_mX300_ct0p1', 'zPrimeTo4b_mZ3000_mX300_ct1', 'zPrimeTo4b_mZ3000_mX300_ct10', 'zPrimeTo4b_mZ3000_mX300_ct100', 'zPrimeTo4b_mZ3000_mX300_ct1000', 'zPrimeTo4b_mZ3000_mX300_ct10000', 'zPrimeTo4b_mZ4500_mX2200_ct0p1', 'zPrimeTo4b_mZ4500_mX2200_ct1', 'zPrimeTo4b_mZ4500_mX2200_ct10', 'zPrimeTo4b_mZ4500_mX2200_ct100', 'zPrimeTo4b_mZ4500_mX2200_ct1000', 'zPrimeTo4b_mZ4500_mX2200_ct10000', 'zPrimeTo4b_mZ4500_mX450_ct0p1', 'zPrimeTo4b_mZ4500_mX450_ct1', 'zPrimeTo4b_mZ4500_mX450_ct10', 'zPrimeTo4b_mZ4500_mX450_ct100', 'zPrimeTo4b_mZ4500_mX450_ct1000', 'zPrimeTo4b_mZ4500_mX450_ct10000',
    'zPrimeTo2b2nu_mZ3000_mX1450_ct0p1', 'zPrimeTo2b2nu_mZ3000_mX1450_ct1', 'zPrimeTo2b2nu_mZ3000_mX1450_ct10', 'zPrimeTo2b2nu_mZ3000_mX1450_ct100', 'zPrimeTo2b2nu_mZ3000_mX1450_ct1000', 'zPrimeTo2b2nu_mZ3000_mX1450_ct10000', 'zPrimeTo2b2nu_mZ3000_mX300_ct0p1', 'zPrimeTo2b2nu_mZ3000_mX300_ct1', 'zPrimeTo2b2nu_mZ3000_mX300_ct10', 'zPrimeTo2b2nu_mZ3000_mX300_ct100', 'zPrimeTo2b2nu_mZ3000_mX300_ct1000', 'zPrimeTo2b2nu_mZ3000_mX300_ct10000', 'zPrimeTo2b2nu_mZ4500_mX2200_ct0p1', 'zPrimeTo2b2nu_mZ4500_mX2200_ct1', 'zPrimeTo2b2nu_mZ4500_mX2200_ct10', 'zPrimeTo2b2nu_mZ4500_mX2200_ct100', 'zPrimeTo2b2nu_mZ4500_mX2200_ct1000', 'zPrimeTo2b2nu_mZ4500_mX2200_ct10000', 'zPrimeTo2b2nu_mZ4500_mX450_ct0p1', 'zPrimeTo2b2nu_mZ4500_mX450_ct1', 'zPrimeTo2b2nu_mZ4500_mX450_ct10', 'zPrimeTo2b2nu_mZ4500_mX450_ct100', 'zPrimeTo2b2nu_mZ4500_mX450_ct1000', 'zPrimeTo2b2nu_mZ4500_mX450_ct10000'
]


sign = [
    'HeavyHiggsToLLPTo4b_mH400_mX150_ct0p1', 'HeavyHiggsToLLPTo4b_mH400_mX150_ct1', 'HeavyHiggsToLLPTo4b_mH400_mX150_ct10', 'HeavyHiggsToLLPTo4b_mH400_mX150_ct100', 'HeavyHiggsToLLPTo4b_mH400_mX150_ct1000', 'HeavyHiggsToLLPTo4b_mH400_mX150_ct10000', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct0p1', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct1', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct10', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct100', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct1000', 'HeavyHiggsToLLPTo4b_mH400_mX40_ct10000', 

    #'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct0p1', 'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct1', 'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct10', 'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct100', 'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct1000', 'HeavyHiggsToLLPTo2b2nu_mH400_mX150_ct10000', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct0p1', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct1', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct10', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct100', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct1000', 'HeavyHiggsToLLPTo2b2nu_mH400_mX40_ct10000', 

    #'HeavyHiggsToLLPTo4b_mH800_mX350_ct0p1', 'HeavyHiggsToLLPTo4b_mH800_mX350_ct1', 'HeavyHiggsToLLPTo4b_mH800_mX350_ct10', 'HeavyHiggsToLLPTo4b_mH800_mX350_ct100', 'HeavyHiggsToLLPTo4b_mH800_mX350_ct1000', 'HeavyHiggsToLLPTo4b_mH800_mX350_ct10000', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct0p1', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct1', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct10', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct100', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct1000', 'HeavyHiggsToLLPTo4b_mH800_mX80_ct10000', 

    #'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct0p1', 'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct1', 'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct10', 'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct100', 'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct1000', 'HeavyHiggsToLLPTo2b2nu_mH800_mX350_ct10000', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct0p1', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct1', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct10', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct100', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct1000', 'HeavyHiggsToLLPTo2b2nu_mH800_mX80_ct10000', 
]

#sign = ['StealthSHH_2t6j_mstop1100_ms875_ctau1000_SHH',]

#sign = ['StealthSHH_2t6j_mstop300_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop300_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop300_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop300_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop300_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop300_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop500_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau0p01_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau0p1_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau1_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau10_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau100_SHH', 'StealthSHH_2t6j_mstop500_ms275_ctau1000_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop700_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau0p01_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau0p1_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau1_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau10_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau100_SHH', 'StealthSHH_2t6j_mstop700_ms475_ctau1000_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop900_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau0p01_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau0p1_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau1_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau10_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau100_SHH', 'StealthSHH_2t6j_mstop900_ms675_ctau1000_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop1100_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau1_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau10_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau100_SHH', 'StealthSHH_2t6j_mstop1100_ms875_ctau1000_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop1300_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau1_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau10_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau100_SHH', 'StealthSHH_2t6j_mstop1300_ms1075_ctau1000_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau1_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau10_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau100_SHH', 'StealthSHH_2t6j_mstop1500_ms100_ctau1000_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau0p01_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau0p1_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau1_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau10_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau100_SHH', 'StealthSHH_2t6j_mstop1500_ms1275_ctau1000_SHH']

#sign = ['StealthSYY_2t6j_mstop300_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop300_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop300_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop300_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop300_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop300_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop500_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau0p01_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau0p1_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau1_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau10_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau100_SYY', 'StealthSYY_2t6j_mstop500_ms275_ctau1000_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop700_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau0p01_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau0p1_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau1_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau10_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau100_SYY', 'StealthSYY_2t6j_mstop700_ms475_ctau1000_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop900_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau0p01_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau0p1_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau1_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau10_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau100_SYY', 'StealthSYY_2t6j_mstop900_ms675_ctau1000_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop1100_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau1_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau10_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau100_SYY', 'StealthSYY_2t6j_mstop1100_ms875_ctau1000_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop1300_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau1_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau10_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau100_SYY', 'StealthSYY_2t6j_mstop1300_ms1075_ctau1000_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau1_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau10_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau100_SYY', 'StealthSYY_2t6j_mstop1500_ms100_ctau1000_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau0p01_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau0p1_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau1_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau10_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau100_SYY', 'StealthSYY_2t6j_mstop1500_ms1275_ctau1000_SYY']

sign = ['StealthSYY_2t6j_mstop500_ms100_ctau0p01_SYY',]


#Stealth SHH:
if RUN_ERA==2017 and "StealthSHH" in sign[0]:
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SHH/")% str(RUN_ERA)
if RUN_ERA==2018 and "StealthSHH" in sign[0]:
    print "ebbene?"
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SHH/")% str(RUN_ERA)
if RUN_ERA==2018 and "StealthSYY" in sign[0]:
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SYY/")% str(RUN_ERA)
if RUN_ERA==2016 and "StealthSHH" in sign[0]:
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SHH/")% str(RUN_ERA)
if RUN_ERA==2016 and "StealthSYY" in sign[0]:
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_AOD_%s_07October2021_fix_SYY/")% str(RUN_ERA)

print INPUTDIR

sample_list = sign#data#sign#data#sign#data#data#sign#data#sign#data#sign+data#data#sign#data#sign#data#sign#data#sign#back#sign#data#sign#data#back#sign#data#back#sign#data#sign#back#data#back#sign#sign#data#back#sign#data#back#data#data+back

dicty = {}
#for s in sign:

for s in sample_list:
    for ss in samples[s]["files"]:
        print ss
        print requests[ss]
        s1 = requests[ss][1:].split('/')[0]
        if 'splitSUSY_M2400_100' in ss:
            s1 = "CRAB_UserFiles"
        if 'HeavyHiggsToLLP' in ss:
            s1 = "CRAB_UserFiles"
        if 'zPrime' in ss:
            s1 = "CRAB_UserFiles"
        print s1
        dicty[ss] = s1+'/crab_'+ss+'/'
        if s=="DYJetsToLL" and RUN_ERA==2018:
            new_ss = ss.replace('pythia8','pythia')
            dicty[ss] = s1+'8/crab_'+new_ss+'/'
        if "JER" in INPUTDIR:
            dicty[ss] = s1+'/crab_'+ss+'_HH/'

if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 512*2#4000#5000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 3600*20#*12#86400
    root_files_per_job = 100#Stealth!#100#40#20#40#
    
    sample_to_loop = dicty.keys()
    #print sample_to_loop
    for s in sample_to_loop:
        print s, ": . . ."
        skipTrain = False
        #bkg
        if (('QCD_HT' in s) and RUN_ERA==2018): skipTrain = True
        if (('WW_TuneCP5' in s) and RUN_ERA==2018): skipTrain = True
        if (('WZ_TuneCP5' in s) and RUN_ERA==2018): skipTrain = True
        if (('ZZ_TuneCP5' in s) and RUN_ERA==2018): skipTrain = True
        if (('ZJetsToNuNu_HT' in s) and RUN_ERA==2018): skipTrain = True
        if (('WJetsToLNu_HT' in s) and RUN_ERA==2018): skipTrain = True
        if (('TTJets_TuneCP5' in s) and RUN_ERA==2018): skipTrain = True
        #sgn
        if (('TChiHH_mass400_pl1000' in s) and RUN_ERA==2018): skipTrain = True 
        if (('GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2' in s) and RUN_ERA==2018): skipTrain = True 

        if(skipTrain):
            print("Sample used for DNN training, keeping only odd events...")
            
        if(skipTrain):
            skip_string = ' yes '
        else:
            skip_string = ' no '


        #isSignal: decide what triggers to store
        isSignal = False
        isData   = True

        #mc trigger file
        mc_trigger_file = ("/nfs/dust/cms/group/cms-llp/MET_trigger_SF_Caltech/METTriggers_SF.root")
        mc_trigger_string = ""
        if RUN_ERA==2018:
            mc_trigger_string = "trigger_efficiency_Fall18"
        elif RUN_ERA==2017:
            mc_trigger_string = "trigger_efficiency_Fall17"
        elif RUN_ERA==2016:
            mc_trigger_string = "trigger_efficiency_Summer16"

        #mc PU file
        mc_PU_file = ("/nfs/dust/cms/group/cms-llp/PU_histograms_Caltech/")
        if ('QCD_HT' in s):
            if RUN_ERA==2018:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            elif RUN_ERA==2016:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            else:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraph-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('WW_Tune' in s):
            #mc_PU_file+=('VV_%s.root')%(pu_tag)
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            isData = False
        if ('WZ_Tune' in s):
            #mc_PU_file+=('VV_%s.root')%(pu_tag)
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            isData = False
        if ('ZZ_Tune' in s):
            #mc_PU_file+=('VV_%s.root')%(pu_tag)
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            isData = False
        if ('ZJetsToNuNu_HT' in s):
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            isData = False
        if ('WJetsToLNu_HT' in s):
            mc_PU_file+=('PileupReweight_WJetsToLNu_HT-70ToInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_TuneCP5_13TeV' in s):
            mc_PU_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_DiLept_genMET' in s):
            mc_PU_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_SingleLeptFromT_genMET' in s):
            mc_PU_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_SingleLeptFromTbar_genMET' in s):
            mc_PU_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('DYJetsToLL' in s):
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            #mc_PU_file+=('DYJetsToLL_%s.root')%(pu_tag)
            isData = False
        #sgn
        if ('n3n2-n1-hbb-hbb' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('TChiHH_mass400_pl1000' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('SMS' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('splitSUSY' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('HeavyHiggsToLLP' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('Stealth' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('GluGluH2_H2ToSSTobbbb' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('gluinoGMSB' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('XXTo4J' in s):
            isSignal = True
            isData = False
            mc_PU_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)


        #Time smearing data file
        data_smearing_file = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+str(RUN_ERA)+"_TtoEM_v5_ntuples_validate_timeRecHits/data_smear_file_CSV_0p8_all_jets.root"
        #Time smearing signal file
        signal_smearing_file = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+str(RUN_ERA)+"_SR_v5_ntuples_validate_timeRecHits/sign_smear_file.root"
        #Name of the corresponding crystal ball
        signal_CB_string = "sign_CB_"+s.replace("_HH","").replace("_HZ","").replace("_ZZ","")

        if isSignal:
            sign_str = "true"
        else:
            sign_str = "false"

        #mock, for data
        if isData:
            if RUN_ERA==2018:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            elif RUN_ERA==2016:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            else:
                mc_PU_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraph-pythia8_%s.root')%(tune,pu_tag)

        #read input files in crab folder
        IN = INPUTDIR + dicty[s]
        print(IN)

        if not(os.path.exists(IN)):
            print IN , " : empty dir, go to next..."
            continue

        date_subdirs = [x for x in os.listdir(IN) if os.path.isdir(os.path.join(IN, x))]

        if(len(date_subdirs)>1):
            print("Multiple date/time subdirs, aborting...")
            exit()


        IN += date_subdirs[0]
        #print(IN)
        num_subdirs = [x for x in os.listdir(IN) if os.path.isdir(os.path.join(IN, x))]
        #print(num_subdirs)

        #Here must implement having multiple subdirs
        for subd in num_subdirs:
            INPS = IN + "/"+subd+"/"
            #print(INPS)
            root_files = [x for x in os.listdir(INPS) if os.path.isfile(os.path.join(INPS, x))]

            #create out subdir
            OUT = OUTPUTDIR + s
            if "JER" in INPUTDIR:
                OUT += '_HH'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            print "Writing results in ", OUT
            OUT += "/" + subd + '/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)

            cond_name = "skim_"+str(RUN_ERA)+"_v6_smear"+doRegion#+"_positive0p7"#"_syst_unc_JERUp"##+"Gen"#+"_nophoton"#+"_BeamHalo"##+"_bin_1_2"#+"_test"#
            cond_name = "skim_"+str(RUN_ERA)+"_v6_syst_unc"+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_PDF_QCD_syst_unc"#+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_heavy_higgs"#+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_compare_SUSY_HH"#+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_zPrime"#+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_HeavyHiggs"#+doRegion
            cond_name = "skim_"+str(RUN_ERA)+"_v6_stealth"#+doRegion
            COND_DIR = '/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_'+cond_name+resubm_label
            #COND_DIR = '/nfs/dust/cms/user/lbenato/condor_'+cond_name+resubm_label
            if not(os.path.exists(COND_DIR)):
                os.mkdir(COND_DIR)

            COND_DIR += '/'+s
            if not(os.path.exists(COND_DIR)):
                os.mkdir(COND_DIR)
                
            COND_DIR += '/'+subd
            if not(os.path.exists(COND_DIR)):
                os.mkdir(COND_DIR)
            else:
                print "Warning, directory exists, deleting old condor outputs ... "
                os.system("ls " +COND_DIR)
                os.system("rm " + COND_DIR + "/*sh")
                os.system("rm " + COND_DIR + "/*submit")
                os.system("rm " + COND_DIR + "/*txt")

            os.chdir(COND_DIR)

            #start loop
            print "\n"
            print s
            print "subdir: ", subd
            print "len root files: ", len(root_files)
            print "root_files_per_job: ", root_files_per_job
            #print "%: ", len(root_files)%root_files_per_job

            j_num = 0
            for a in range(0,len(root_files),root_files_per_job):
                start = a
                stop = min(a+root_files_per_job-1,len(root_files)-1)
                print "Submitting job n. : ", j_num
                

                #BASH
                with open('job_skim_'+str(j_num)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('module use -a /afs/desy.de/group/cms/modulefiles/ \n')
                    fout.write('module load cmssw \n')
                    fout.write('cd /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src \n')
                    fout.write('cmsenv \n')
                    #here loop over the files
                    for b in np.arange(start,stop+1):
                        #print b, root_files[b]
                        fout.write('echo "Processing '+ root_files[b]  +' . . ." \n')
                        #HEM
                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')
                        fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_signal_time_smearing ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' ' + data_smearing_file + ' ' + signal_smearing_file + ' ' + signal_CB_string + ' \n')
                        fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_signal_time_smearing ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' ' + data_smearing_file + ' ' + signal_smearing_file + ' ' + signal_CB_string + ' \n')
                        '''
                        #We are not ready for this
                        if isSignal:
                            fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_signal_time_smearing ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' ' + data_smearing_file + ' ' + signal_smearing_file + ' ' + signal_CB_string + ' \n')
                            fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_signal_time_smearing ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' ' + data_smearing_file + ' ' + signal_smearing_file + ' ' + signal_CB_string + ' \n')
                        else:
                            fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                            fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')
                        '''

                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_tagger_v3_no_time ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_tagger_v3_no_time ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')


                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_positive_jets_0p7 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_positive_jets_0p7 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')


                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_syst_unc ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_syst_unc ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')


                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_PDF_QCD_syst_unc ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_PDF_QCD_syst_unc ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')

                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_no_photonEFrac_cut ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_no_photonEFrac_cut ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')

                    fout.write('echo \n')
                os.system('chmod 755 job_skim_'+str(j_num)+'.sh')

                #CONDOR
                with open('submit_skim_'+str(j_num)+'.submit', 'w') as fout:
                    fout.write('executable   = ' + COND_DIR + '/job_skim_'+ str(j_num) + '.sh \n')
                    fout.write('output       = ' + COND_DIR + '/out_skim_'+ str(j_num) + '.txt \n')
                    fout.write('error        = ' + COND_DIR + '/error_skim_'+ str(j_num) + '.txt \n')
                    fout.write('log          = ' + COND_DIR + '/log_skim_'+ str(j_num) + '.txt \n')
                    fout.write(' \n')
                    fout.write('Requirements = OpSysAndVer == "CentOS7" \n')
                    fout.write('##Requirements = OpSysAndVer == "CentOS7" && CUDADeviceName == "GeForce GTX 1080 Ti" \n')
                    fout.write('##Request_GPUs = 1 \n')#to check if it fails less!!
                    fout.write(' \n')
                    fout.write('## uncomment this if you want to use the job specific variables $CLUSTER and $PROCESS inside your batchjob \n')
                    fout.write('##environment = "CLUSTER=$(Cluster) PROCESS=$(Process)" \n')
                    fout.write(' \n')
                    fout.write('## uncomment this to specify a runtime longer than 3 hours (time in seconds) \n')
                    fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                    fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                    fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                    fout.write('batch_name = skim_'+s[:6]+str(j_num)+' \n')
                    fout.write('queue 1 \n')
               
                ##submit condor
                #os.system('python /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_skim_v5/'+s+'/skim_macro_'+str(j_num)+'.py'  +' \n')            
                os.system('condor_submit ' + 'submit_skim_'+str(j_num)+'.submit' + ' \n')            
                #os.system('sh job_skim_'+str(j_num)+'.sh \n')

                #if j_num<9:
                #    print "job n. ", j_num, " non lo faccio"
                #else:
                #    print "job n. ", j_num, " DA FARE"
                #    os.system('sh job_skim_'+str(j_num)+'.sh \n')
                j_num +=1
                #exit()

            os.chdir('../../')

else:
    ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/skimJetsAcceptanceCaloFast.cc")
    for s in (back):
        for ss in samples[s]["files"]:
            print ss
            filename = ss + ".root"
            ROOT.skimJetsAcceptanceCaloFast(INPUTDIR+filename,OUTPUTDIR+filename,0,-1,True)
