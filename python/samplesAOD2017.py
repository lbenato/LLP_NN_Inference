#! /usr/bin/env python

#voms-proxy-init -voms cms
#

#Higgs production cross sections: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHXSWG#Production_cross_sections_and_de
#mH = 125 GeV (check if it' better 125.09!)
    #ggH xsec: 48.58
    #VBF xsec: 3.782
    #WH xsec: 1.373
    #ZH xsec: 0.8839

sample = {

    ########## BACKGROUNDS ##########
    #QCD
    #cross-sections taken from mcm:
    'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1': {
        'nevents' : 40471637,
        'xsec'    : 246300000.0,#pb, not found on das/mcm? from 2016 samples
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT100to200_TuneCP5_13TeV-madgraph-pythia8-v2': {
        'nevents' : 93954381,
        #'xsec' : 28060000.0,#pb#2016 numbers
        'xsec'    : 23700000.0,#XSDB
        #'xsec' : 19380000, #mcm, seems completely wrong
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8-v1' : {
        'nevents' : 18722416,
        #'xsec' : 1710000.0,#pb#2016
        'xsec'    : 1547000.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8-v1' : {
        'nevents' : 17035891,
        #'xsec'   : 351300,#pb#B2G-17-005
        'xsec'    : 322600.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8-v2' : {
        'nevents' : 18929951,
        #'xsec'   : 31630,#pb#B2G-17-005
        'xsec'    : 29980.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8-v1' : {
        'nevents' : 15629253,
        #'xsec'   : 6802,#pb#B2G-17-005
        'xsec'    : 6334.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8-v1' : {
        'nevents' : 4767100,
        #'xsec' : 1206,#pb#B2G-17-005
        'xsec'    : 1088.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8-v2' : {
        'nevents' : 3970819,
        #'xsec' : 120.4,#pb
        'xsec'    : 99.11,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8-v2' : {
        'nevents' : 1991645,
        #'xsec' : 25.25,#pb
        'xsec'    : 20.23,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    #TTbar
    'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 496.1,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTJets_SingleLeptFromT_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 6.212,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTJets_SingleLeptFromTbar_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 6.167,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTJets_DiLept_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 3.655,#x-sec DB
        'matcheff': 1.,
        'kfactor' : 1.,
    },    
    #Single Top
    #x-sec taken from B2G-17-005
    #'ST_s-channel_4f_leptonDecays_13TeV-amcatnlo-pythia8_TuneCUETP8M1-v1' : {
    #    'nevents' : 1000000.,
#        'xsec'    : 3.365,#x-sec DB
    #    'xsec'    : 10.32*(1.-0.6760),#B2G-17-005
    #    'kfactor' : 1.,
    #},
    #'ST_t-channel_antitop_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1-v1' : {
    #    'nevents' : 38811017.,
#        'xsec'    : 80.95,#x-sec DB is 0!
    #    'xsec'    : 80.95,#B2G-17-005
    #    'kfactor' : 1.,
    #},
    #'ST_t-channel_top_4f_inclusiveDecays_13TeV-powhegV2-madspin-pythia8_TuneCUETP8M1-v1' : {
    #    'nevents' : 67240808.,
#        'xsec'    : 136.02,#x-sec DB is 0!
    #    'xsec'    : 136.02,#B2G-17-005
    #    'kfactor' : 1.,
    #},
    #'ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_ext1-v1' : {
    #    'nevents' : 6933094,
#        'xsec'    : 38.06,#x-sec DB
    #    'xsec'    : 71.7/2., #B2G-17-005
    #    'matcheff': 1.,
    #    'kfactor' : 1.,
    #},
    #'ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_ext1-v1' : {
    #    'nevents' : 6952830,
#        'xsec'    : 38.09,#x-sec DB
    #    'xsec'    : 71.7/2.,#B2G-17-005
    #    'matcheff': 1.,
    #    'kfactor' : 1.,
    #},

    ##Dibosons
    ##WW
    'WW_TuneCP5_13TeV-pythia8-v1' : {
        'nevents' : 994012,
        'xsec'    : 75.8,#XSDB
        #'xsec'    : 118.7,#B2G-17-005
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    ##WZ
    'WZ_TuneCP5_13TeV-pythia8-v1' : {
        'nevents' : 1000000,
        'xsec'    : 27.6,#XSDB
        #'xsec'    : 47.2,#B2G-17-005
        'matcheff': 1.,
        'kfactor' : 1.,
    }, 
    #ZZ
    'ZZ_TuneCP5_13TeV-pythia8-v1' : {
        'nevents' : 990064,
        'xsec'    : 12.14,#XSDB
        #'xsec'    : 16.6,#B2G-17-005
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    #ZJetsToNuNu
    'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 384.1,
        'xsec'    : 302.8,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.37,
    },
    'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 118.1,
        'xsec'    : 92.59,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.52,
    },
    'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 14.7,
        'xsec'    : 13.18,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.37,
    },
    'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 3.35,
        'xsec'    : 3.257,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.04,
    },
    'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 1.68,
        'xsec'    : 1.49,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.14,
    },
    'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 0.316,
        'xsec'    : 0.3419,#XSDB
        'matcheff': 1.,
        'kfactor' : 0.88,
    },
    'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1' : {
        'nevents' : 1,
        #'xsec'    : 0.0072,
        'xsec'    : 0.005146,#XSDB
        'matcheff': 1.,
        'kfactor' : 0.88,
    },


    #DYJetsToLL
    'DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 146.7,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 161.1,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 48.66,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 6.968,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1.743,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 0.8052,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 0.1933,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 0.003468,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    
    
    #WJetsToLNu LO inclusive
    'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 52850.0,#second entry on XSDB, 50260.0,
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    #WJetsToLNu HT binned
    'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1292.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v2' : {
        'nevents' : 1,
        'xsec'    : 1395.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 407.9,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 57.48,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 12.87,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 5.366,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1.074,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v3' : {
        'nevents' : 1,
        'xsec'    : 0.008001,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    'n3n2-n1-hbb-hbb_mh400_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*88.7372*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh300_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*284.855*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh250_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*810.24*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh200_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*1335.62*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh175_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*2583.96*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh150_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1e-03*3832.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh127_pl1000' : {
        'nevents' : 1,
        'xsec'    : 10.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },  

    'n3n2-n1-hbb-hbb_mh400_prompt' : {
        'nevents' : 1,
        'xsec'    : 1e-03*88.7372*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh300_prompt' : {
        'nevents' : 1,
        'xsec'    : 1e-03*284.855*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh200_prompt' : {
        'nevents' : 1,
        'xsec'    : 1e-03*1335.62*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
}



samples = {

    #DATA
    'data_obs' : {
        'order' : 0,
        'files' : ['METRun2016B-03Feb2017_ver2-v2'],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Data",
        'weight': 1.,
        'plot': True,
    },
    'HighMET' : {
        'order' : 0,
        'files' : [
            'HighMETRun2017B-17Nov2017-v1',
            'HighMETRun2017C-17Nov2017-v1',
            'HighMETRun2017D-17Nov2017-v1',
            'HighMETRun2017E-17Nov2017-v1',
            'HighMETRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET 2017",
        'weight': 1.,
        'plot': True,
    },

    'HighMETCopy' : {
        'order' : 0,
        'files' : [
            'HighMETRun2017B-17Nov2017-v1',
            'HighMETRun2017C-17Nov2017-v1',
            'HighMETRun2017D-17Nov2017-v1',
            'HighMETRun2017E-17Nov2017-v1',
            'HighMETRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 8,
        'fillstyle' : 1001,
        'linecolor' : 8,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET 2017, bin1",
        'weight': 1.,
        'plot': True,
    },

    'HighMETBH' : {
        'order' : 0,
        'files' : [
            'HighMETRun2017B-17Nov2017-v1BH',
            'HighMETRun2017C-17Nov2017-v1BH',
            'HighMETRun2017D-17Nov2017-v1BH',
            'HighMETRun2017E-17Nov2017-v1BH',
            'HighMETRun2017F-17Nov2017-v1BH'
        ],
        'fillcolor' : 1,#632-7,
        'fillstyle' : 1001,
        'linecolor' : 1,#632-7,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET with Beam Halo",#bin1",#with Beam Halo",
        'weight': 1.,
        'plot': True,
    },


    'MET' : {
        'order' : 0,
        'files' : ['METRun2017B-17Nov2017-v1','METRun2017C-17Nov2017-v1','METRun2017D-17Nov2017-v1','METRun2017E-17Nov2017-v1','METRun2017F-17Nov2017-v1'],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Data",
        'weight': 1.,
        'plot': True,
    },
    'SingleMuon' : {
        'order' : 0,
        'files' : [
            'SingleMuonRun2017B-17Nov2017-v1',
            'SingleMuonRun2017C-17Nov2017-v1',
            'SingleMuonRun2017D-17Nov2017-v1',
            'SingleMuonRun2017E-17Nov2017-v1',
            'SingleMuonRun2017F-17Nov2017-v1',
            ##'SingleMuonRun2017G-17Nov2017-v1',
            ##'SingleMuonRun2017H-17Nov2017-v2'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon 2017",
        'weight': 1.,
        'plot': True,
    },

    'SingleMuonBH' : {
        'order' : 0,
        'files' : [
            'SingleMuonRun2017B-17Nov2017-v1BH',
            'SingleMuonRun2017C-17Nov2017-v1BH',
            'SingleMuonRun2017D-17Nov2017-v1BH',
            'SingleMuonRun2017E-17Nov2017-v1BH',
            'SingleMuonRun2017F-17Nov2017-v1BH',
            ##'SingleMuonRun2017G-17Nov2017-v1',
            ##'SingleMuonRun2017H-17Nov2017-v2'
        ],
        'fillcolor' : 632-7,
        'fillstyle' : 1001,
        'linecolor' : 632-7,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon with Beam Halo",
        'weight': 1.,
        'plot': True,
    },


    'Bin2' : {
        'order' : 0,
        'files' : [
            'SingleMuonRun2017B-17Nov2017-v1_skim',
            'SingleMuonRun2017C-17Nov2017-v1_skim',
            'SingleMuonRun2017D-17Nov2017-v1_skim',
            'SingleMuonRun2017E-17Nov2017-v1_skim',
            'SingleMuonRun2017F-17Nov2017-v1_skim',

            #'SingleElectronRun2017B-17Nov2017-v1_skim',
            #'SingleElectronRun2017C-17Nov2017-v1_skim',
            #'SingleElectronRun2017D-17Nov2017-v1_skim',
            #'SingleElectronRun2017E-17Nov2017-v1_skim',
            #'SingleElectronRun2017F-17Nov2017-v1_skim'
        ],
        'fillcolor' : 2,
        'fillstyle' : 1001,
        'linecolor' : 2,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon 2017 bin 2",
        'weight': 1.,
        'plot': True,
    },
    'SingleElectron' : {
        'order' : 0,
        'files' : [
            'SingleElectronRun2017B-17Nov2017-v1',
            'SingleElectronRun2017C-17Nov2017-v1',
            'SingleElectronRun2017D-17Nov2017-v1',
            'SingleElectronRun2017E-17Nov2017-v1',
            'SingleElectronRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleEle 2017",
        'weight': 1.,
        'plot': True,
    },

    'SingleElectronBH' : {
        'order' : 0,
        'files' : [
            'SingleElectronRun2017B-17Nov2017-v1BH',
            'SingleElectronRun2017C-17Nov2017-v1BH',
            'SingleElectronRun2017D-17Nov2017-v1BH',
            'SingleElectronRun2017E-17Nov2017-v1BH',
            'SingleElectronRun2017F-17Nov2017-v1BH'
        ],
        'fillcolor' : 632-7,
        'fillstyle' : 1001,
        'linecolor' : 632-7,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleEle with Beam Halo",
        'weight': 1.,
        'plot': True,
    },

    'JetHT' : {
        'order' : 0,
        'files' : [
            'JetHTRun2017B-17Nov2017-v1',
            'JetHTRun2017C-17Nov2017-v1',
            'JetHTRun2017D-17Nov2017-v1',
            'JetHTRun2017E-17Nov2017-v1',
            'JetHTRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,#4,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "JetHT 2017",
        'weight': 1.,
        'plot': True,
    },

    'JetHTMC' : {
        'order' : 0,
        'files' : [
            'JetHTRun2017B-17Nov2017-v1',
            'JetHTRun2017C-17Nov2017-v1',
            'JetHTRun2017D-17Nov2017-v1',
            'JetHTRun2017E-17Nov2017-v1',
            'JetHTRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 860-9,
        'fillstyle' : 1001,
        'linecolor' : 860-9,#4,#1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "JetHT 2017",
        'weight': 1.,
        'plot': True,
    },

    'MuonEG' : {
        'order' : 0,
        'files' : [
            'MuonEGRun2017B-17Nov2017-v1',
            'MuonEGRun2017C-17Nov2017-v1',
            'MuonEGRun2017D-17Nov2017-v1',
            'MuonEGRun2017E-17Nov2017-v1',
            'MuonEGRun2017F-17Nov2017-v1',
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "MuonEG 2017",
        'weight': 1.,
        'plot': True,
    },
    'SinglePhoton' : {
        'order' : 0,
        'files' : [
            'SinglePhotonRun2017B-17Nov2017-v1',
            'SinglePhotonRun2017C-17Nov2017-v1',
            'SinglePhotonRun2017D-17Nov2017-v1',
            'SinglePhotonRun2017E-17Nov2017-v1',
            'SinglePhotonRun2017F-17Nov2017-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Data",
        'weight': 1.,
        'plot': True,
    },

    # Dummy entry for background sum
    'BkgSum' : {
        'order' : 0,
        'files' : [],
        'fillcolor' : 1,
        'fillstyle' : 3003,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Bkg stat.",
        'weight': 1.,
        'plot': True,
    },

    #BACKGROUNDS
    'All' : {
        'files' : [
            'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'QCD_HT100to200_TuneCP5_13TeV-madgraph-pythia8-v2',
            'QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8-v1',
            'QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8-v1', 
            'QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8-v2', 
            'QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8-v1', 
            'QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8-v1', 
            'QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8-v2', 
            'QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8-v2',
            'TTJets_SingleLeptFromT_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'TTJets_SingleLeptFromTbar_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v2',
            'TTJets_DiLept_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1',
            'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v2',
            'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v3',
            'WW_TuneCP5_13TeV-pythia8-v1', 
            'WZ_TuneCP5_13TeV-pythia8-v1', 
            'ZZ_TuneCP5_13TeV-pythia8-v1'
        ],
        'fillcolor' : 856,
        'fillstyle' : 1001,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : 'All bkg MC samples',
        'title' : 'All',
        'weight': 1.,
        'plot': True,
    },

    #QCD
    'QCD' : {
        'files' : [
             'QCD_HT50to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
             'QCD_HT100to200_TuneCP5_13TeV-madgraph-pythia8-v2',
             'QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8-v1',
             'QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8-v1', 
             'QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8-v2', 
             'QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8-v1', 
             'QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8-v1', 
             'QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8-v2', 
             'QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8-v2'
        ],
        'fillcolor' : 920,
        'fillstyle' : 1001,
        'linecolor' : 920,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : 'QCD',
        'title' : 'QCD',
        'weight': 1.,
        'plot': True,
    },

    #TTbar
    'TTbar' : {
        'files' : [
             'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
        ],
        'fillcolor' : 798,
        'fillstyle' : 1001,
        'linecolor' : 798,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "t#bar{t}",
        'weight': 1.,
        'plot': True,
    },

    #TTbar
    'TTbarGenMET' : {
        'files' : [
             'TTJets_SingleLeptFromT_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
             'TTJets_SingleLeptFromTbar_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v2',
             'TTJets_DiLept_genMET-150_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
        ],
        'fillcolor' : 798,
        'fillstyle' : 1001,
        'linecolor' : 798,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "t#bar{t}",
        'weight': 1.,
        'plot': True,
    },

    #Single top
    'ST' : {
        'files' : [],  
        'fillcolor' : 801,
        'fillstyle' : 1001,
        'linecolor' : 801,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Single t",
        'weight': 1.,
        'plot': True,
    },

    #ZJetsToNuNu
    'ZJetsToNuNu' : {
        'files' : [
            'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1'
            ],  
        'fillcolor' : 856,
        'fillstyle' : 1001,
        'linecolor' : 856,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Z #rightarrow #nu #nu + jets",
        'weight': 1.,
        'plot': True,
    },
    'ZJetsToNuNuRed' : {
        'files' : [
            'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1', 
            #'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1', 
            #'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1', 
            #'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1', 
            #'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1'
            ],  
        'fillcolor' : 856,
        'fillstyle' : 1001,
        'linecolor' : 856,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Z #rightarrow #nu #nu + jets",
        'weight': 1.,
        'plot': True,
    },

    #WJets
    'WJetsToLNuIncl' : { 
        'files' : ['WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8-v2'],
        'fillcolor' : 881,
        'fillstyle' : 1001,
        'linecolor' : 881,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "W jets to LNu",
        'weight': 1.,
        'plot': True,
    },
    #HT binned
    'WJetsToLNu' : { 
        'files' : [
            'WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8-v2',
            'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8-v3',
            ],
        'fillcolor' : 881,
        'fillstyle' : 1001,
        'linecolor' : 881,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "W #rightarrow l #nu + jets",
        'weight': 1.,
        'plot': True,
    },
    #DYJetsToLL
    'DYJetsToLL' : {
        'files' : [
            'DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
            'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8-v1',
        ],
        'fillcolor' : 418,
        'fillstyle' : 1001,
        'linecolor' : 418,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Z/#gamma #rightarrow ll + jets",
        'weight': 1.,
        'plot': True,
    },
    #Dibosons
    'VV' : {
        'files' : [
            'WW_TuneCP5_13TeV-pythia8-v1', 
            'WZ_TuneCP5_13TeV-pythia8-v1', 
            'ZZ_TuneCP5_13TeV-pythia8-v1'
            ],  
        'fillcolor' : 602,
        'fillstyle' : 1001,
        'linecolor' : 602,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "WW, WZ, ZZ",
        'weight': 1.,
        'plot': True,
    },
   
    #SIGNAL
######################################
#
#  JiaJing's AOD

    'SUSY_all' : {
        'files' : [
            'n3n2-n1-hbb-hbb_mh400_pl1000',
            'n3n2-n1-hbb-hbb_mh300_pl1000',
            'n3n2-n1-hbb-hbb_mh250_pl1000',
            'n3n2-n1-hbb-hbb_mh200_pl1000',
            'n3n2-n1-hbb-hbb_mh175_pl1000',
            'n3n2-n1-hbb-hbb_mh127_pl1000',
        ],
        'mass' : 400,
        'ctau' : 1000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127-400 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh400_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh400_pl1000'],
        'mass' : 400,
        'ctau' : 1000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh300_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh300_pl1000'],
        'mass' : 300,
        'ctau' : 1000,
        'fillcolor' : 801,
        'fillstyle' : 0,
        'linecolor' : 801,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh250_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh250_pl1000'],
        'mass' : 250,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh200_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh200_pl1000'],
        'mass' : 200,
        'ctau' : 1000,
        'fillcolor' : 8,
        'fillstyle' : 0,
        'linecolor' : 8,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh175_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh175_pl1000'],
        'mass' : 175,
        'ctau' : 1000,
        'fillcolor' : 826,
        'fillstyle' : 0,
        'linecolor' : 826,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh150_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh150_pl1000'],
        'mass' : 150,
        'ctau' : 1000,
        'fillcolor' : 4,
        'fillstyle' : 0,
        'linecolor' : 4,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh127_pl1000' : {
        'files' : ['n3n2-n1-hbb-hbb_mh127_pl1000'],
        'mass' : 127,
        'ctau' : 1000,
        'fillcolor' : 8,
        'fillstyle' : 0,
        'linecolor' : 8,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },

    #Prompt
    'SUSY_mh400_prompt' : {
        'files' : ['n3n2-n1-hbb-hbb_mh400_prompt'],
        'mass' : 400,
        'ctau' : 0,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 2,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 0 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh300_prompt' : {
        'files' : ['n3n2-n1-hbb-hbb_mh300_prompt'],
        'mass' : 300,
        'ctau' : 0,
        'fillcolor' : 801,
        'fillstyle' : 0,
        'linecolor' : 801,
        'linewidth' : 3,
        'linestyle' : 2,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 0 m",
        'weight': 1.,
        'plot': True,
    },

    'SUSY_mh200_prompt' : {
        'files' : ['n3n2-n1-hbb-hbb_mh200_prompt'],
        'mass' : 200,
        'ctau' : 0,
        'fillcolor' : 8,
        'fillstyle' : 0,
        'linecolor' : 8,
        'linewidth' : 3,
        'linestyle' : 2,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 0 m",
        'weight': 1.,
        'plot': True,
    },
######################################
#
#  Heavy Higgs AOD
    #MH 1000, MS 400
    'ggH_MH1000_MS400_ctau500' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 400,
        'ctau' : 500,
        'fillcolor' : 9,
        'fillstyle' : 0,
        'linecolor' : 9,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H;S} = 1000;400 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS400_ctau1000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 400,
        'ctau' : 1000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 400 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS400_ctau2000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 400,        
        'ctau' : 2000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 400 GeV, c#tau_{0} = 2 m",
        'weight': 1.,
        'plot': True,
    },    
    'ggH_MH1000_MS400_ctau5000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 400,        
        'ctau' : 5000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 400 GeV, c#tau_{0} = 5 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS400_ctau10000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-400_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 400,
        'ctau' : 10000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 400 GeV, c#tau_{0} = 10 m",
        'weight': 1.,
        'plot': True,
    },

    #MH 1000, MS 150
    'ggH_MH1000_MS150_ctau500' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-500_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 150,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 150 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS150_ctau1000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 150,
        'ctau' : 1000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 150 GeV, c#tau_{0} = 1 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS150_ctau2000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-2000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 150,
        'ctau' : 2000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 150 GeV, c#tau_{0} = 2 m",
        'weight': 1.,
        'plot': True,
    },    
    'ggH_MH1000_MS150_ctau5000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-5000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 150,
        'ctau' : 5000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 150 GeV, c#tau_{0} = 5 m",
        'weight': 1.,
        'plot': True,
    },
    'ggH_MH1000_MS150_ctau10000' : {
        'files' : ['GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-10000_TuneCP5_13TeV-pythia8_PRIVATE-MC'],
        'MH' : 1000,
        'MS' : 150,
        'ctau' : 10000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{H} = 1 TeV, m_{S} = 150 GeV, c#tau_{0} = 10 m",
        'weight': 1.,
        'plot': True,
    },

}
