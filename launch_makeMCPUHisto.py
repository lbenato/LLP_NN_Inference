#!/usr/bin/env python

import os
import subprocess
import ROOT as ROOT
import multiprocessing
from collections import defaultdict
from samplesAOD2018 import sample, samples
import numpy as np

RUN_ERA = 2018
run_condor = True

if RUN_ERA==2018:
    INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2018_31December2020/"
    OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_31December2020_PU/"
    #data_file = ""
    #data_file_up = ""
    #data_file_down = ""
elif RUN_ERA==2017:
    INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2017_31December2020/"
    OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_31December2020_PU/"
    #data_file = ""
    #data_file_up = ""
    #data_file_down = ""
elif RUN_ERA==2016:
    INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_2016_31December2020/"
    OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2016_31December2020_PU/"
    #data_file = ""
    #data_file_up = ""
    #data_file_down = ""
else:
    print "Invalid run era, aborting..."
    exit()

dicty = defaultdict()
dicty = {
    #'TChiHH_mass400_pl1000' : 'TChiHH_mass400_pl1000/crab_TChiHH_mass400_pl1000/',
    #'n3n2-n1-hbb-hbb_mh200_pl1000' : 'n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh200_pl1000/',
    ##'n3n2-n1-hbb-hbb_mh127_pl1000' : 'n3n2-n1-hbb-hbb_mh127_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh127_pl1000//0000/',
    ##'n3n2-n1-hbb-hbb_mh150_pl1000' : 'n3n2-n1-hbb-hbb_mh150_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh150_pl1000//0000/',
    ##'n3n2-n1-hbb-hbb_mh175_pl1000' : 'n3n2-n1-hbb-hbb_mh175_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh175_pl1000//0000/',
    ##'n3n2-n1-hbb-hbb_mh200_pl1000' : 'n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh200_pl1000//0000/',
    ##'n3n2-n1-hbb-hbb_mh250_pl1000' : 'n3n2-n1-hbb-hbb_mh250_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh250_pl1000/200630_173814/0000/',
    ##'n3n2-n1-hbb-hbb_mh300_pl1000' : 'n3n2-n1-hbb-hbb_mh300_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh300_pl1000/200630_175000/0000/',
    ##'n3n2-n1-hbb-hbb_mh400_pl1000' : 'n3n2-n1-hbb-hbb_mh400_pl1000_ev100000/crab_n3n2-n1-hbb-hbb_mh400_pl1000/200630_174204/0000/',
    
    #Very boosted
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : 'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC':'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC':'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC':'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',
    #'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC':'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/',

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
    

    'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-100To200_13TeV-madgraph/crab_ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-200To400_13TeV-madgraph/crab_ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-400To600_13TeV-madgraph/crab_ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-600To800_13TeV-madgraph/crab_ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-800To1200_13TeV-madgraph/crab_ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph/crab_ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1/',
    'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1' : 'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph/crab_ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1/',

    'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/crab_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',

    #VV
    'WW_TuneCP5_13TeV-pythia8-v2' : 'WW_TuneCP5_13TeV-pythia8/crab_WW_TuneCP5_13TeV-pythia8-v2/',
    'WZ_TuneCP5_13TeV-pythia8-v3' : 'WZ_TuneCP5_13TeV-pythia8/crab_WZ_TuneCP5_13TeV-pythia8-v3/',
    'ZZ_TuneCP5_13TeV-pythia8-v2' : 'ZZ_TuneCP5_13TeV-pythia8/crab_ZZ_TuneCP5_13TeV-pythia8-v2/',
    
    #QCD
    'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',#201018_195308/0000/',#also 0001
    'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',#
    'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',#also 0001
    'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : 'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/crab_QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',

    ###'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1-v2' : 'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/crab_TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1-v2/',
    
    'TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',
    'TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/crab_TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1/',

    'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8-v1'  : 'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/crab_DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia-v1/',

    #'METRun2018A-17Sep2018-v1' : 'MET/crab_METRun2018A-17Sep2018-v1/',
}


data = ["data_obs"]
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
sign = [
        'SUSY_mh400_pl1000', 'SUSY_mh300_pl1000', 'SUSY_mh250_pl1000', 'SUSY_mh200_pl1000', 'SUSY_mh175_pl1000', 'SUSY_mh150_pl1000', 'SUSY_mh127_pl1000',
        'ggH_MH125_MS25_ctau500',  'ggH_MH125_MS25_ctau1000',  'ggH_MH125_MS25_ctau2000',  'ggH_MH125_MS25_ctau5000',  'ggH_MH125_MS25_ctau10000', 
        'ggH_MH125_MS55_ctau500',  'ggH_MH125_MS55_ctau1000',  'ggH_MH125_MS55_ctau2000',  'ggH_MH125_MS55_ctau5000',  'ggH_MH125_MS55_ctau10000', 
        'ggH_MH200_MS50_ctau500',  'ggH_MH200_MS50_ctau1000',  'ggH_MH200_MS50_ctau2000',  'ggH_MH200_MS50_ctau5000',  'ggH_MH200_MS50_ctau10000', 
        'ggH_MH200_MS25_ctau500',  'ggH_MH200_MS25_ctau1000',  'ggH_MH200_MS25_ctau2000',  'ggH_MH200_MS25_ctau5000',  'ggH_MH200_MS25_ctau10000', 
        'ggH_MH400_MS100_ctau500', 'ggH_MH400_MS100_ctau1000', 'ggH_MH400_MS100_ctau2000', 'ggH_MH400_MS100_ctau5000', 'ggH_MH400_MS100_ctau10000',
        'ggH_MH400_MS50_ctau500',  'ggH_MH400_MS50_ctau1000',  'ggH_MH400_MS50_ctau2000',  'ggH_MH400_MS50_ctau5000',  'ggH_MH400_MS50_ctau10000',
        'ggH_MH600_MS150_ctau500', 'ggH_MH600_MS150_ctau1000', 'ggH_MH600_MS150_ctau2000', 'ggH_MH600_MS150_ctau5000', 'ggH_MH600_MS150_ctau10000',
        'ggH_MH600_MS50_ctau500',  'ggH_MH600_MS50_ctau1000',  'ggH_MH600_MS50_ctau2000',  'ggH_MH600_MS50_ctau5000',  'ggH_MH600_MS50_ctau10000',
	'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000',
	'ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
	'ggH_MH1500_MS200_ctau500','ggH_MH1500_MS200_ctau1000','ggH_MH1500_MS200_ctau2000','ggH_MH1500_MS200_ctau5000','ggH_MH1500_MS200_ctau10000',
	'ggH_MH1500_MS500_ctau500','ggH_MH1500_MS500_ctau1000','ggH_MH1500_MS500_ctau2000','ggH_MH1500_MS500_ctau5000','ggH_MH1500_MS500_ctau10000',
	'ggH_MH2000_MS250_ctau500','ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS250_ctau2000','ggH_MH2000_MS250_ctau5000','ggH_MH2000_MS250_ctau10000',
	'ggH_MH2000_MS600_ctau500','ggH_MH2000_MS600_ctau1000','ggH_MH2000_MS600_ctau2000','ggH_MH2000_MS600_ctau5000','ggH_MH2000_MS600_ctau10000',
        ]



def makeMCPU(in_filename,out_filename,min_ev,max_ev):
    ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/makeMCPUHisto.cc")
    print "Calculating MC PC hist of " + in_filename+".root . . ."
    ROOT.makeMCPUHisto(in_filename+".root",out_filename+".root",min_ev,max_ev)
    print "File saved in " + out_filename + ".root"
    print "min: ", min_ev, " max: ", max_ev
    #print "\n"


if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 1000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 3600*12#86400
    root_files_per_job = 20#3#20#50
    
    sample_to_loop = dicty.keys()
    for s in sample_to_loop:
        print s, ": . . ."

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

            #here: condor
            '''
            OUT += "/" + subd + '/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            COND_DIR = '/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_MCPU/'+s
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
            print "len root files: ", len(root_files)
            print "root_files_per_job: ", root_files_per_job
            #print "%: ", len(root_files)%root_files_per_job

            j_num = 0
            for a in range(0,len(root_files),root_files_per_job):
                start = a
                stop = min(a+root_files_per_job-1,len(root_files)-1)
                print "Submitting job n. : ", j_num

                #PYTHON MACRO
                with open('MCPU_macro_'+str(j_num)+'.py', 'w') as fout:
                    fout.write('#!/usr/bin/env python \n')
                    fout.write('import os \n')
                    fout.write('import subprocess \n')
                    fout.write('import ROOT as ROOT \n')
                    fout.write('import multiprocessing \n')
                    fout.write(' \n')
                    fout.write('ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/makeMCPUHisto.cc") \n')
                    fout.write('def makeMCPU(in_filename,out_filename,min_ev,max_ev): \n')
                    fout.write('    print "Calculating MC PC hist of  "+in_filename+". . ." \n')
                    fout.write('    ROOT.makeMCPUHisto(in_filename,out_filename,min_ev,max_ev) \n')
                    fout.write('    print "File saved in " + out_filename  \n')
                    fout.write(' \n')
                    for b in np.arange(start,stop+1):
                        #print b, root_files[b]
                        fout.write('print "Processing '+ root_files[b]  +' . . . " ')
                        fout.write(' \n')
                        fout.write('makeMCPU("'+INPS+root_files[b]+'","'+OUT+root_files[b]+'",0,-1) \n')
                        fout.write(' \n')

                #BASH
                with open('job_MCPU_'+str(j_num)+'.sh', 'w') as fout:
                    fout.write('#!/bin/sh \n')
                    fout.write('source /etc/profile.d/modules.sh \n')
                    fout.write('module use -a /afs/desy.de/group/cms/modulefiles/ \n')
                    fout.write('module load cmssw \n')
                    #fout.write('cd /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference \n')
                    fout.write('cmsenv \n')
                    fout.write('echo submitting: \n')
                    
                    fout.write('python '+COND_DIR+'/MCPU_macro_'+str(j_num)+'.py'  +' \n')
                os.system('chmod 755 job_MCPU_'+str(j_num)+'.sh')

                #CONDOR
                with open('submit_MCPU_'+str(j_num)+'.submit', 'w') as fout:
                    fout.write('executable   = ' + COND_DIR + '/job_MCPU_'+ str(j_num) + '.sh \n')
                    fout.write('output       = ' + COND_DIR + '/out_MCPU_'+ str(j_num) + '.txt \n')
                    fout.write('error        = ' + COND_DIR + '/error_MCPU_'+ str(j_num) + '.txt \n')
                    fout.write('log          = ' + COND_DIR + '/log_MCPU_'+ str(j_num) + '.txt \n')
                    fout.write(' \n')
                    fout.write('Requirements = OpSysAndVer == "CentOS7" \n')
                    fout.write('##Requirements = OpSysAndVer == "CentOS7" && CUDADeviceName == "GeForce GTX 1080 Ti" \n')
                    fout.write('#Request_GPUs = 1 \n')
                    fout.write(' \n')
                    fout.write('## uncomment this if you want to use the job specific variables $CLUSTER and $PROCESS inside your batchjob \n')
                    fout.write('##environment = "CLUSTER=$(Cluster) PROCESS=$(Process)" \n')
                    fout.write(' \n')
                    fout.write('## uncomment this to specify a runtime longer than 3 hours (time in seconds) \n')
                    fout.write('Request_Cpus = ' + str(NCPUS) + ' \n')
                    fout.write('Request_Memory = ' + str(MEMORY) + ' \n')
                    fout.write('+RequestRuntime = ' + str(RUNTIME) + ' \n')
                    fout.write('batch_name = MCPU_'+s[:6]+str(j_num)+' \n')
                    fout.write('queue 1 \n')
               
                ##submit condor
                #os.system('python /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_MCPU/'+s+'/MCPU_macro_'+str(j_num)+'.py'  +' \n')            
                os.system('condor_submit ' + 'submit_MCPU_'+str(j_num)+'.submit' + ' \n')            
                #Local:
                #os.system('sh job_MCPU_'+str(j_num)+'.sh \n')            
                j_num +=1

            os.chdir('../../')
            '''

            #Here hadd
            print OUT + " content:"
            os.system('ls '+OUT)
            print s
            print OUT+"/*/*.root "
            print OUTPUTDIR+ "/"+s+"_PU.root"

            print "hadd PU histogram "
            os.system("hadd -fk207 "+ OUTPUTDIR + "/"+s+"_PU.root " + OUT + "/*/*.root " )
            ##print "rm counter histograms "
            ##os.system("rm "+OUT+"/*.root ")


#else:
#    ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/makeMCPUHisto.cc")
#    for s in (back):
#        for ss in samples[s]["files"]:
#            print ss
#            filename = ss + ".root"
#            ROOT.skimJetsAcceptanceCaloFast(INPUTDIR+filename,OUTPUTDIR+filename,0,-1,True)

