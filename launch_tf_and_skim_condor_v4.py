#!/usr/bin/env python

import os
import subprocess
import ROOT as ROOT
import multiprocessing
from collections import defaultdict
import numpy as np

#INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2018_31December2020/"
#INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2017_31December2020/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_HT_cut/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_tf_and_skim_v5/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_gen/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_gen/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"

run_condor = True

# # # # # #
# Run era # 
# # # # # #
RUN_ERA = 2018

doRegion = "doSR"#"doTtoEM"#"doZtoEE"
resubm_label = ""
#resubm_label = "_resubmission_2"
#INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_31December2020/")% str(RUN_ERA)
INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v4_calo_AOD_%s_18October2020"+resubm_label+"/")% str(RUN_ERA)
OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_%s_debug_old"+resubm_label+"/")% str(RUN_ERA)

#INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_01March2021_94X_mc2017_realistic_v17/")% str(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v5_calo_AOD_%s_94X_mc2017_realistic_v17/")% str(RUN_ERA)
#INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_01March2021_102X_upgrade2018_realistic_v19/")% str(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v5_calo_AOD_%s_102X_upgrade2018_realistic_v19/")% str(RUN_ERA)

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

#if RUN_ERA not in [2016,2017,2018]:
else:
    print "Invalid run era, aborting..."
    exit()


dicty_o = defaultdict()
dicty_o = {
    #'XXTo4J_M100_CTau100mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M100_CTau100mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M100_CTau100mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M100_CTau300mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M100_CTau300mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M100_CTau300mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M100_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M100_CTau1000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M100_CTau1000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M100_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M100_CTau3000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M100_CTau3000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M100_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M100_CTau50000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M100_CTau50000mm_TuneCP2_13TeV_pythia8-v1/',

    #'XXTo4J_M300_CTau100mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M300_CTau100mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M300_CTau100mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M300_CTau300mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M300_CTau300mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M300_CTau300mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M300_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M300_CTau1000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M300_CTau1000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M300_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M300_CTau3000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M300_CTau3000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M300_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M300_CTau50000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M300_CTau50000mm_TuneCP2_13TeV_pythia8-v1/',

    #'XXTo4J_M1000_CTau100mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M1000_CTau100mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M1000_CTau100mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M1000_CTau300mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M1000_CTau300mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M1000_CTau300mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M1000_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M1000_CTau1000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M1000_CTau1000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M1000_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M1000_CTau3000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M1000_CTau3000mm_TuneCP2_13TeV_pythia8-v1/',
    #'XXTo4J_M1000_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : 'XXTo4J_M1000_CTau50000mm_TuneCP2_13TeV_pythia8/crab_XXTo4J_M1000_CTau50000mm_TuneCP2_13TeV_pythia8-v1/',

    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10000mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1000mm_TuneCP2_13TeV-pythia8-v1/',
    ##'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-100um_TuneCP2_13TeV_2018-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-100um_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-100um_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30000mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10000mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1000mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30000mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30000mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30000mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300mm_TuneCP2_13TeV-pythia8-v1' : 'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300mm_TuneCP2_13TeV-pythia8/crab_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300mm_TuneCP2_13TeV-pythia8-v1/',
    #'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30mm_TuneCP2_13TeV_2018-pythia8-v1' : '',

    #gluino GMSB
    #'gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau3000p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau3000p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau3000p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau100p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau100p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau100p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau30p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau30p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau30p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau10p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau10p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau10p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau3p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau3p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau3p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau1p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau1p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau1p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau50000p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau50000p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau50000p0_TuneCP2_13TeV_pythia8-v1/',
    #'gluinoGMSB_M2400_ctau30000p0_TuneCP2_13TeV_pythia8-v1' : 'gluinoGMSB_M2400_ctau30000p0_TuneCP2_13TeV_pythia8/crab_gluinoGMSB_M2400_ctau30000p0_TuneCP2_13TeV_pythia8-v1/',

    ##'TChiHH_mass400_pl1000' : 'TChiHH_mass400_pl1000/crab_TChiHH_mass400_pl1000/',
    #'n3n2-n1-hbb-hbb_mh200_pl1000' : 'n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh200_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh127_pl1000' : 'n3n2-n1-hbb-hbb_mh127_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh127_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh150_pl1000' : 'n3n2-n1-hbb-hbb_mh150_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh150_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh175_pl1000' : 'n3n2-n1-hbb-hbb_mh175_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh175_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh200_pl1000' : 'n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh200_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh250_pl1000' : 'n3n2-n1-hbb-hbb_mh250_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh250_pl1000/',
    ###'n3n2-n1-hbb-hbb_mh300_pl1000' : 'n3n2-n1-hbb-hbb_mh300_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh300_pl1000/',
    #'n3n2-n1-hbb-hbb_mh400_pl1000' : 'n3n2-n1-hbb-hbb_mh400_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh400_pl1000/',#!!
    
    #Very boosted
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',

    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    ##'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    ##'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',

    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    ##'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4/',

    #MH 2000
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',#201018_094940/0000/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 1500
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC': 'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 1000
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2/',

    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 600
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2/',

    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',


    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2/',

    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 400
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2/',

    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 200
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #MH 125
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    

    #'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-100To200_13TeV-madgraph/crab_ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-200To400_13TeV-madgraph/crab_ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-400To600_13TeV-madgraph/crab_ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-600To800_13TeV-madgraph/crab_ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-800To1200_13TeV-madgraph/crab_ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph/crab_ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1/',
    #'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph/crab_ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1/',

    #'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',

    #VV
    #'WW_TuneCP5_13TeV-pythia8-v2' : 'WW_TuneCP5_13TeV-pythia8/crab_WW_TuneCP5_13TeV-pythia8-v2/',
    #'WZ_TuneCP5_13TeV-pythia8-v3' : 'WZ_TuneCP5_13TeV-pythia8/crab_WZ_TuneCP5_13TeV-pythia8-v3/',
    #'ZZ_TuneCP5_13TeV-pythia8-v2' : 'ZZ_TuneCP5_13TeV-pythia8/crab_ZZ_TuneCP5_13TeV-pythia8-v2/',
    
    #QCD
    #'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',#
    #'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',#also 0001
    #'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',

    ##'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1-v2' : 'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/crab_TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1-v2/',

    #'TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    #'TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/', 

    #'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/crab_DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia-v1/',

    #'METRun2018A-17Sep2018-v1' : 'MET/crab_METRun2018A-17Sep2018-v1/',
    #'SingleMuonRun2018B-17Sep2018-v1' : 'SingleMuon/crab_SingleMuonRun2018B-17Sep2018-v1/',

}


#data = ["EGamma"]
#data = ["SingleMuon"]
#data = ["MuonEG"]
data = ["MET"]
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
back = ["DYJetsToLL"]
back = ["TTbarGenMET"]
back = ["QCD"]
#back = ["ZJetsToNuNu"]
#back = ["WJetsToLNu"]#done
sign = [
        'SUSY_mh200_pl1000'
        #'SUSY_mh400_pl1000',# 'SUSY_mh300_pl1000', 'SUSY_mh250_pl1000', 'SUSY_mh200_pl1000', 'SUSY_mh175_pl1000', 'SUSY_mh150_pl1000', 'SUSY_mh127_pl1000',
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

dicty = {}
#for s in sign:

sample_list = back#data
for s in sample_list:
    for ss in samples[s]["files"]:
        print ss
        print requests[ss]
        s1 = requests[ss][1:].split('/')[0]
        print s1
        dicty[ss] = s1+'/crab_'+ss+'/'
        if s=="DYJetsToLL":
            new_ss = ss.replace('pythia8','pythia')
            dicty[ss] = s1+'8/crab_'+new_ss+'/'

#print dicty_o
#print "new:"
#print dicty
#exit()

if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 4000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 3600*12#86400
    root_files_per_job = 40#3#20#50
    
    sample_to_loop = dicty.keys()
    print sample_to_loop
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
        #mc PU file
        mc_file = ("/nfs/dust/cms/group/cms-llp/PU_histograms_Caltech/")
        if ('QCD_HT' in s):
            mc_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('WW_Tune' in s):
            mc_file+=('VV_%s.root')%(pu_tag)
            isData = False
        if ('WZ_Tune' in s):
            mc_file+=('VV_%s.root')%(pu_tag)
            isData = False
        if ('ZZ_Tune' in s):
            mc_file+=('VV_%s.root')%(pu_tag)
            isData = False
        if ('ZJetsToNuNu_HT' in s):
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            isData = False
        if ('WJetsToLNu_HT' in s):
            mc_file+=('PileupReweight_WJetsToLNu_HT-70ToInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_DiLept_genMET' in s):
            mc_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_SingleLeptFromT_genMET' in s):
            mc_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('TTJets_SingleLeptFromTbar_genMET' in s):
            mc_file+=('PileupReweight_TTJets_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)
            isData = False
        if ('DYJetsToLL' in s):
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
            #mc_file+=('DYJetsToLL_%s.root')%(pu_tag)
            isData = False
        #sgn
        if ('n3n2-n1-hbb-hbb' in s):
            isSignal = True
            isData = False
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('TChiHH_mass400_pl1000' in s):
            isSignal = True
            isData = False
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('GluGluH2_H2ToSSTobbbb' in s):
            isSignal = True
            isData = False
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('gluinoGMSB' in s):
            isSignal = True
            isData = False
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S' in s):
            isSignal = True
            isData = False
            mcfile+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)
        if ('XXTo4J' in s):
            isSignal = True
            isData = False
            mc_file+=('PileupReweight_ZJetsToNuNu_HT-100ToInf_13TeV-madgraph_%s.root')%(pu_tag)

        if isSignal:
            sign_str = "true"
        else:
            sign_str = "false"

        #mock, for data
        if isData:
            mc_file+=('PileupReweight_QCD_HT50toInf_%s_13TeV-madgraphMLM-pythia8_%s.root')%(tune,pu_tag)

        #read input files in crab folder
        IN = INPUTDIR + dicty[s]
        print(IN)
        date_subdirs = [x for x in os.listdir(IN) if os.path.isdir(os.path.join(IN, x))]

        if(len(date_subdirs)>1):
            print("Multiple date/time subdirs, aborting...")
            exit()

        IN += date_subdirs[0]
        print(IN)
        num_subdirs = [x for x in os.listdir(IN) if os.path.isdir(os.path.join(IN, x))]
        print(num_subdirs)

        #Here must implement having multiple subdirs
        for subd in num_subdirs:
            print(subd)
            INPS = IN + "/"+subd+"/"
            print(INPS)
            root_files = [x for x in os.listdir(INPS) if os.path.isfile(os.path.join(INPS, x))]

            #create out subdir
            OUT = OUTPUTDIR + s
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            OUT += "/" + subd + '/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)

            cond_name = "skim_v5"
            cond_name = "debug_v4_old"
            COND_DIR = '/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_'+cond_name+resubm_label
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
                    #fout.write('echo   1. scram \n')
                    #fout.write('echo \n')
                    #fout.write('scram b -j 32 \n')
                    #fout.write('echo \n')
                    #fout.write('echo   2. submitting: '+s+' \n')
                    #fout.write('echo \n')
                    #here loop over the files
                    for b in np.arange(start,stop+1):
                        #print b, root_files[b]
                        fout.write('echo "Processing '+ root_files[b]  +' . . ." \n')
                        #With PU reweight
                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v5_debug_new ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_file+ ' ' + doRegion + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v5_debug_new ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_file+ ' ' + doRegion + ' \n')
                        #Old approach for debuggin
                        fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v4_debug_old ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' \n')
                        fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v4_debug_old ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + '  \n')



                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_AK4_v3__AK8_v2__v5 ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + isSignal  + ' ' + mc_file+' ' + data_file  + ' ' + data_file_up + ' ' + data_file_down + ' \n')
                        ##fout.write('../bin/slc7_amd64_gcc820/gen_studies ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' \n')
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
                    fout.write('##Request_GPUs = 1 \n')
                    fout.write(' \n')
                    fout.write('## uncomment this if you want to use the job specific variables $CLUSTER and $PROCESS inside your batchjob \n')
                    fout.write('##environment = "CLUSTER=$(Cluster) PROCESS=$(Process)" \n')
                    fout.write(' \n')
                    fout.write('## uncomment this to specify a runtime longer than 3 hours (time in seconds) \n')
                    fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                    fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                    fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                    fout.write('batch_name = skim_'+s[:6]+str(j_num)+' \n')
                    #fout.write('SCHEDD_NAME = bird-htc-sched11.desy.de \n')
                    fout.write('queue 1 \n')
               
                ##submit condor
                #os.system('python /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_skim_v5/'+s+'/skim_macro_'+str(j_num)+'.py'  +' \n')            
                #os.system('condor_submit ' + 'submit_skim_'+str(j_num)+'.submit' + ' \n')            
                os.system('sh job_skim_'+str(j_num)+'.sh \n')
                j_num +=1

            os.chdir('../../')

else:
    ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/skimJetsAcceptanceCaloFast.cc")
    for s in (back):
        for ss in samples[s]["files"]:
            print ss
            filename = ss + ".root"
            ROOT.skimJetsAcceptanceCaloFast(INPUTDIR+filename,OUTPUTDIR+filename,0,-1,True)
