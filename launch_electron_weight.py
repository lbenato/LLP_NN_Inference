import os
import subprocess
import ROOT as ROOT
import multiprocessing
from collections import defaultdict
import numpy as np
import uproot
import root_numpy

RUN_ERA = 2018
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


#electron

#INPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples/weighted/"
#HT_ratio_file = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples/data_MC_HT_ratio.root"
#ele_ratio_file = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples/data_MC_Electrons_pt_ratio.root"

INPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples_reweighted/"
OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples_reweighted/weighted/"
HT_ratio_file = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples_reweighted/data_MC_HT_ratio.root"
ele_ratio_file = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(RUN_ERA)+"_E_v5_ntuples_reweighted/data_MC_Electrons_pt_ratio.root"

back = ["Wtop"]

for b in back:
    os.chdir('/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src')
    for ss in samples[b]["files"]:
        print ss
        os.system("echo ../bin/slc7_amd64_gcc820/v5_electron_reweight "+INPUTDIR+ss+".root "+OUTPUTDIR+ss+".root "+HT_ratio_file+" " + ele_ratio_file +" \n")
        os.system("../bin/slc7_amd64_gcc820/v5_electron_reweight "+INPUTDIR+ss+".root "+OUTPUTDIR+ss+".root "+HT_ratio_file+ " " + ele_ratio_file +" \n")
