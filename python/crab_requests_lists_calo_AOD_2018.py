#! /usr/bin/env python 

#Here you can find a useful list of our privately produced VBF dark pion samples

requests = {
#############################
## VBF ###
#############################
#############################
## VBF Tranche 2 ###
#############################

#QCD
###QCD, HT50to100
'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',# at T1_UK_RAL
###QCD, HT100to200
'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
###QCD, HT200to300
'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
###QCD, HT300to500
'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
###QCD, HT500to700
'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#partially at T2
###QCD, HT700to1000
'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#partially at T2
###QCD, HT1000to1500
'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
###QCD, HT1500to2000
'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
###QCD, HT2000toInf
'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE

##############################
###TTbar inclusive
#/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM #TAPE
#/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM #TAPE
#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM #partially present at T2_UK_London_IC and T2_IN_TIFR
#/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM #T1_IT_CNAF_Disk
'TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8_ext1-v2' : '/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2/AODSIM',#T1_US_FNAL_Disk
#JJ:
'TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/TTJets_SingleLeptFromTbar_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/TTJets_SingleLeptFromT_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/TTJets_DiLept_genMET-80_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##############################
###Single top
'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1' : '/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1/AODSIM',
'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_ext1-v1' : '/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1/AODSIM',#TAPE

'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8_ext1-v3' : '/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v3/AODSIM',#TAPE
'ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-madgraph-pythia8_ext1-v1' : '/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-madgraph-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1/AODSIM',#TAPE

'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1' : '/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#pieces at T2 FR GRIF LLR
'ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8-v1' : '/ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#pieces at T2_RU
'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8-v1' : '/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#FNAL disk, pieces at Aachen
'ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8-v1' : '/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#SPRACE

##############################
###VBF Higgs to b bbar
'VBFHToBB_M-125_13TeV_powheg_pythia8-v1' : '',

##############################
#Dibosons
'WW_TuneCP5_13TeV-pythia8-v2' : '/WW_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',#partially at T2
'WZ_TuneCP5_13TeV-pythia8-v3' : '/WZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3/AODSIM',#T2_UK_London_IC
'ZZ_TuneCP5_13TeV-pythia8-v2' : '/ZZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',#T2_RU_JINR

##############################
#DYJetsToNuNu
# not produced in 2018
#'DYJetsToNuNu_PtZ-50To100_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-v1' : '',
#'DYJetsToNuNu_PtZ-100To250_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-v1' : '',
#'DYJetsToNuNu_PtZ-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-v1' : '',
#'DYJetsToNuNu_PtZ-400To650_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-v1' : '',
#'DYJetsToNuNu_PtZ-650ToInf_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-v1' : '',

##############################
#ZJetsToNuNu
'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-100To200_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-200To400_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
#from now on, not checked anymore, it's clearly all on tape...
'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-400To600_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',#TAPE
'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-600To800_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#T2_BE_IIHE
'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-800To1200_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#TAPE
'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#T1_RU_JINR_Disk
'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1' : '/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#T2_BE_UCL

##############################
#DYJetsToQQ
'DYJetsToQQ_HT180_13TeV-madgraphMLM-pythia8-v1' : '',

##############################
#DYJetsToLL inclusive
'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',#T2_FR_IPHC

##############################
### WJets
#WJetsToLNu
#'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8-v2' : '/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',#T2_EE_Estonia
'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : '/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',

#WJetsToQQ
'WJetsToQQ_HT180_13TeV-madgraphMLM-pythia8-v1' : '',

###################### Higgs
##Gluon fusion Higgs
#GluGluHToBB
'GluGluHToBB_M125_13TeV_powheg_pythia8-v1' : '',

##Higgs associated production
#ZH
'ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8-v1' : '',
'ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_ext1-v1' : '',
'ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8-v1' : '',
#WH
'WminusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8-v1' : '',
'WplusH_HToBB_WToQQ_M125_13TeV_powheg_pythia8-v1' : '',
'WminusH_HToBB_WToLNu_M125_13TeV_powheg_pythia8-v1' : '',
'WplusH_HToBB_WToLNu_M125_13TeV_powheg_pythia8-v1' : '',

##ttH
#ttH to bb
'ttHTobb_M125_13TeV_powheg_pythia8-v1' : '',

##bbH
#bbH...
'bbHToBB_M-125_4FS_yb2_13TeV_amcatnlo-v1' : '',

##Double Higgs production
#VBF HH
'VBFHHTo4B_CV_1_C2V_1_C3_1_13TeV-madgraph-v1' : '',

###############################################
#Data
###!!
### Let's start with RunG, that was one of the best in 2016 (no trigger inefficiencies, no HIP issue, better calibrations, but starting to have a bit of pre-firing!)
###!!
#2018: available at T2 aachen
#MET skim
'HighMETRun2018A-17Sep2018-v1' : '/MET/Run2018A-HighMET-17Sep2018-v1/RAW-RECO',
'HighMETRun2018B-17Sep2018-v1' : '/MET/Run2018B-HighMET-17Sep2018-v1/RAW-RECO',
'HighMETRun2018C-17Sep2018-v1' : '/MET/Run2018C-HighMET-17Sep2018-v1/RAW-RECO',
'HighMETRun2018D-PromptReco-v1' : '/MET/Run2018D-HighMET-PromptReco-v1/RAW-RECO',
'HighMETRun2018D-PromptReco-v2' : '/MET/Run2018D-HighMET-PromptReco-v2/RAW-RECO',
'HighMETRun2018E-PromptReco-v1' : '/MET/Run2018E-HighMET-PromptReco-v1/RAW-RECO',
#MET
'METRun2018A-17Sep2018-v1' : '/MET/Run2018A-17Sep2018-v1/AOD',
'METRun2018B-17Sep2018-v1' : '/MET/Run2018B-17Sep2018-v1/AOD',
'METRun2018C-17Sep2018-v1' : '/MET/Run2018C-17Sep2018-v1/AOD',
#'METRun2018D-PromptReco-v1' : '/MET/Run2018D-PromptReco-v1/AOD',#creates 0 jobs
'METRun2018D-PromptReco-v2' : '/MET/Run2018D-PromptReco-v2/AOD',
#'METRun2018E-PromptReco-v1' : '/MET/Run2018E-PromptReco-v1/AOD',
#SingleMuon
'SingleMuonRun2018A-17Sep2018-v2' : '/SingleMuon/Run2018A-17Sep2018-v2/AOD',
'SingleMuonRun2018B-17Sep2018-v1' : '/SingleMuon/Run2018B-17Sep2018-v1/AOD',
'SingleMuonRun2018C-17Sep2018-v1' : '/SingleMuon/Run2018C-17Sep2018-v1/AOD',
'SingleMuonRun2018D-PromptReco-v2' : '/SingleMuon/Run2018D-PromptReco-v2/AOD',
#EGamma
'EGammaRun2018A-17Sep2018-v2' : '/EGamma/Run2018A-17Sep2018-v2/AOD',
'EGammaRun2018B-17Sep2018-v1' : '/EGamma/Run2018B-17Sep2018-v1/AOD',
'EGammaRun2018C-17Sep2018-v1' : '/EGamma/Run2018C-17Sep2018-v1/AOD',
#'EGammaRun2018D-PromptReco-v1' : '/EGamma/Run2018D-PromptReco-v1/AOD',#creates 0 jobs
'EGammaRun2018D-PromptReco-v2' : '/EGamma/Run2018D-PromptReco-v2/AOD',
#'EGammaRun2018E-PromptReco-v1' : '/EGamma/Run2018E-PromptReco-v1/AOD',#creates 0 jobs
#JetHT
'JetHTRun2018A-17Sep2018-v1' : '/JetHT/Run2018A-17Sep2018-v1/AOD',
'JetHTRun2018B-17Sep2018-v1' : '/JetHT/Run2018B-17Sep2018-v1/AOD',
'JetHTRun2018C-17Sep2018-v1' : '/JetHT/Run2018C-17Sep2018-v1/AOD',
#'JetHTRun2018D-PromptReco-v1' : '/JetHT/Run2018D-PromptReco-v1/AOD',#creates 0 jobs
'JetHTRun2018D-PromptReco-v2' : '/JetHT/Run2018D-PromptReco-v2/AOD',
#'JetHTRun2018E-PromptReco-v1' : '/JetHT/Run2018E-PromptReco-v1/AOD',#creates 0 jobs
#MuonEG
'MuonEGRun2018A-17Sep2018-v1' : '/MuonEG/Run2018A-17Sep2018-v1/AOD',#all done
'MuonEGRun2018B-17Sep2018-v1' : '/MuonEG/Run2018B-17Sep2018-v1/AOD',
'MuonEGRun2018C-17Sep2018-v1' : '/MuonEG/Run2018C-17Sep2018-v1/AOD',
'MuonEGRun2018D-PromptReco-v2' : '/MuonEG/Run2018D-PromptReco-v2/AOD',
###############################################
#Jiajing's
'n3n2-n1-hbb-hbb_mh400_pl1000' : '/n3n2-n1-hbb-hbb_mh400_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi400_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh300_pl1000' : '/n3n2-n1-hbb-hbb_mh300_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi300_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh250_pl1000' : '/n3n2-n1-hbb-hbb_mh250_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi250_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh200_pl1000' : '/n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi200_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh175_pl1000' : '/n3n2-n1-hbb-hbb_mh175_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi175_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh150_pl1000' : '/n3n2-n1-hbb-hbb_mh150_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi150_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',
'n3n2-n1-hbb-hbb_mh127_pl1000' : '/n3n2-n1-hbb-hbb_mh127_pl1000_ev100000/jmao-crab_CMSSW_9_4_12_n3n2-n1-hbb-hbb_mchi127_pl1000_ev100000_AODSIM_CaltechT2-37ac8d442b9840be1abba4facfa67e70/USER',

###High stat
'TChiHH_mass400_pl1000' : '/TChiHH_mass400_pl1000/apresyan-crab_PrivateProduction_Fall18_DR_step2_TChiHH_mass400_pl1000_batch1_v1-2fd59cbde119ecab78af65e08efe8aae/USER',
'n3n2-n1-hbb-hbb_mh200_pl1000' : '/n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/jmao-crab_CMSSW_10_2_11_n3n2-n1-hbb-hbb_mchi200_pl1000_ev100000_AODSIM_CaltechT2-05ab6a21d925342611e50bb80c2191f7/USER',
##'n3n2-n1-hbb-hbb_mh200_pl1000_2018' : '/n3n2-n1-hbb-hbb_mh200_pl1000_ev100000/jmao-crab_CMSSW_10_2_11_n3n2-n1-hbb-hbb_mchi200_pl1000_ev100000_GENSIM_CaltechT2-f30b642cff340931c61e0cda79b85df0/USER',

##Central prod
#'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
#split by sample mass/ctau
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',


#split by sample mass/ctau
#HH
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HH' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',

#HZ
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',

#ZZ
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',
'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_ZZ' : '/SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_mC2-127to1800_TuneCP2_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-GridpackScan_102X_upgrade2018_realistic_v15-v1/AODSIM',

###############################################
#Heavy Higgs
#MH 2 TeV, MS 600 GeV
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-600_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 2 TeV, MS 250 GeV
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 2 TeV, MS 50 GeV
##'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 2 TeV, MS 50 GeV, Tranche 4
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-2000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',

##MH 1.5 TeV, MS 500 GeV
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-500_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1.5 TeV, MS 200 GeV
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-200_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1.5 TeV, MS 50 GeV
##'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1.5 TeV, MS 50 GeV, Tranche 4
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1500_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',

##MH 1 TeV, MS 400 GeV
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1 TeV, MS 150 GeV
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1 TeV, MS 50 GeV
##'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 1 TeV, MS 50 GeV, Tranche 4
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-100_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',
'GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche4' : '/GluGluH2_H2ToSSTobbbb_MH-1000_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche4_AODSIM-b6b7c7e4bbcffb31da65095d455b824f/USER',

####MH 600 GeV, MS 150 GeV
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 600 GeV, MS 50 GeV
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-600_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_Tranche2_partII_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 400 GeV, MS 100 GeV
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-100_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 400 GeV, MS 50 GeV
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-400_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 200 GeV, MS 50 GeV
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-50_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
##MH 200 GeV, MS 25 GeV
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-200_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#MH 125 GeV, MS 55 GeV
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-55_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#MH 125 GeV, MS 25 GeV
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',
#'GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC' : '/GluGluH2_H2ToSSTobbbb_MH-125_MS-25_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC/lbenato-RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_AODSIM-51dbace3c073d03a4eeb2fde7d5a06e2/USER',

#split susy
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-100um_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-10mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-1mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-300mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-300mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-300um_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-1600_CTau-30mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-100um_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-10mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-1mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30000mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30000mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300mm_TuneCP2_13TeV-pythia8-v1':'/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300mm_TuneCP2_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-300um_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
#/GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-2400_CTau-30mm_TuneCP2_13TeV_2018-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM

#Jet-Jet
'XXTo4J_M1000_CTau10000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau10000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M1000_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau1000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M1000_CTau100mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau100mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M1000_CTau10mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M1000_CTau1mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M1000_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau3000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M1000_CTau300mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau300mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M1000_CTau30mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M1000_CTau3mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M1000_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M1000_CTau50000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M100_CTau10000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau10000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M100_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau1000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M100_CTau100mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau100mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M100_CTau10mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M100_CTau1mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M100_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau3000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M100_CTau300mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau300mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M100_CTau30mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M100_CTau3mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M100_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M100_CTau50000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M300_CTau10000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau10000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M300_CTau1000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau1000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M300_CTau100mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau100mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M300_CTau10mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M300_CTau1mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M300_CTau3000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau3000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'XXTo4J_M300_CTau300mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau300mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
##/XXTo4J_M300_CTau30mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
##/XXTo4J_M300_CTau3mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM
'XXTo4J_M300_CTau50000mm_TuneCP2_13TeV_pythia8-v1' : '/XXTo4J_M300_CTau50000mm_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',

#gluinoGMSB
'gluinoGMSB_M2400_ctau50000p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau50000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau3p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau3p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau30p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau30p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8-v2' : '/gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau300p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau3000p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau3000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau30000p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau30000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau1p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau1p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau10p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau10p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau100p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau100p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8_ext1-v1' : '/gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1/AODSIM',
'gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau1000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8-v2' : '/gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8-v1' : '/gluinoGMSB_M2400_ctau10000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',


##'CosmicsRun2018A-06Jun2018-v1' : '/Cosmics/Run2018A-06Jun2018-v1/AOD',
'CosmicsRun2018A-CosmicSP-PromptReco-v1' : '/Cosmics/Run2018A-CosmicSP-PromptReco-v1/RAW-RECO',
'CosmicsRun2018A-CosmicSP-PromptReco-v2' : '/Cosmics/Run2018A-CosmicSP-PromptReco-v2/RAW-RECO',
'CosmicsRun2018A-CosmicSP-PromptReco-v3' : '/Cosmics/Run2018A-CosmicSP-PromptReco-v3/RAW-RECO',
'CosmicsRun2018B-CosmicSP-PromptReco-v1' : '/Cosmics/Run2018B-CosmicSP-PromptReco-v1/RAW-RECO',
'CosmicsRun2018B-CosmicSP-PromptReco-v2' : '/Cosmics/Run2018B-CosmicSP-PromptReco-v2/RAW-RECO',
'CosmicsRun2018C-CosmicSP-PromptReco-v1' : '/Cosmics/Run2018C-CosmicSP-PromptReco-v1/RAW-RECO',
'CosmicsRun2018C-CosmicSP-PromptReco-v2' : '/Cosmics/Run2018C-CosmicSP-PromptReco-v2/RAW-RECO',
'CosmicsRun2018C-CosmicSP-PromptReco-v3' : '/Cosmics/Run2018C-CosmicSP-PromptReco-v3/RAW-RECO',
'CosmicsRun2018D-CosmicSP-PromptReco-v1' : '/Cosmics/Run2018D-CosmicSP-PromptReco-v1/RAW-RECO',
'CosmicsRun2018D-CosmicSP-PromptReco-v2' : '/Cosmics/Run2018D-CosmicSP-PromptReco-v2/RAW-RECO',
'CosmicsRun2018E-CosmicSP-PromptReco-v1' : '/Cosmics/Run2018E-CosmicSP-PromptReco-v1/RAW-RECO',

#Cosmics MC
'LooseMuCosmic_38T_p5-3000-RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v9' : '/LooseMuCosmic_38T_p5-3000/RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v9/AODSIM',
'TKCosmic_38T_p20-3000-RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v8' : '/TKCosmic_38T_p20-3000/RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v8/AODSIM',

#TwinHiggs
#VBF
#'VBFH_HToSSTo4b_MH-125_TuneCP5_13TeV-powheg-pythia8-v2' : '/VBFH_HToSSTo4b_MH-125_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-rp_102X_upgrade2018_realistic_v15-v2/GEN-SIM-RECO',
#ggH
#'ggH_HToSSTobbbb_MH-125_TuneCP5_13TeV-powheg-pythia8-v1' : '/ggH_HToSSTobbbb_MH-125_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-rp_102X_upgrade2018_realistic_v15-v1/GEN-SIM-RECO',

#split SUSY private
'splitSUSY_M2400_100_ctau0p1_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau0p1_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau1p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau1p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau10p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau10p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau100p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau100p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau1000p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau1000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau10000p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau10000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',
'splitSUSY_M2400_100_ctau100000p0_TuneCP2_13TeV_pythia8-v1' : '/splitSUSY_M2400_100_ctau100000p0_TuneCP2_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM',

#HeavyHiggs
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
'HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_250_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_600_mX_60_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',

#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_350_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_800_mX_80_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',


#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_450_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4b_mH_1000_mX_100_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_250_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_600_mX_60_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_450_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_0p1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_1_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_10_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_100_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_1000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH' : '/HeavyHiggsToLLPTo4j_mH_1000_mX_100_ctau_10000_2018_privateMC_LLPNTUPLE_v5_generationHH/XXX/AODSIM',

#zPrime to 4b
'zPrime_mZ_3000_mX_1450_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_0p1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_1_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_10_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_100_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_1000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_10000_4b_2018_privateMC_RECO_v5_generationForDSReview/XXX/AODSIM',

#zPrime to 2b2nu
'zPrime_mZ_3000_mX_1450_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_1450_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_1450_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_3000_mX_300_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_3000_mX_300_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_2200_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_2200_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_0p1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_1_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_10_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_100_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_1000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',
'zPrime_mZ_4500_mX_450_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview' : '/zPrime_mZ_4500_mX_450_ctau_10000_2b2nu_2018_privateMC_RECO_v4_generationForDSReview/XXX/AODSIM',


#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_150_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_400_mX_40_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_350_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_0p1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_1_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_10_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_100_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_1000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',
#'HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH' : '/HeavyHiggsToLLPTo2b2nu_mH_800_mX_80_ctau_10000_2018_privateMC_LLPNTUPLE_v4_generationHH/XXX/AODSIM',

#Stealth
'StealthSHH_2t6j_mstop300_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop300_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop300_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop300_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop300_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop300_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop500_ms275_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop700_ms475_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop900_ms675_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1100_ms875_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1300_ms1075_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms100_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau0p01_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau0p1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau1_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau10_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau100_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSHH_2t6j_mstop1500_ms1275_ctau1000_SHH' : '/StealthSHH_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',

'StealthSYY_2t6j_mstop300_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop300_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop300_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop300_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop300_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop300_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop500_ms275_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop700_ms475_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop900_ms675_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1100_ms875_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1300_ms1075_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms100_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau0p01_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau0p1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau1_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau10_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau100_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',
'StealthSYY_2t6j_mstop1500_ms1275_ctau1000_SYY' : '/StealthSYY_2t6j_mStop-300to1500_mSo-lowandhigh_ctau-0p01to1000_TuneCP5_13TeV_madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2/AODSIM',


}
