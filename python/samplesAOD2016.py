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
    'QCD_HT50to100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1': {
        'nevents' : 40471637,
        'xsec'    : 246300000.0,#pb, not found on das/mcm? from 2016 samples
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1': {
        'nevents' : 93954381,
        'xsec' : 28060000.0,#pb#2016 numbers
        #'xsec'    : 28060000.0,#XSDB
        #'xsec' : 19380000, #mcm, seems completely wrong
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 18722416,
        'xsec' : 1710000.0,#pb#2016
        #'xsec'    : 1547000.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 17035891,
        #'xsec'   : 351300,#pb#B2G-17-005
        'xsec'    : 347500.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 18929951,
        #'xsec'   : 31630,#pb#B2G-17-005
        'xsec'    : 32060.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 15629253,
        #'xsec'   : 6802,#pb#B2G-17-005
        'xsec'    : 6829.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 4767100,
        #'xsec' : 1206,#pb#B2G-17-005
        'xsec'    : 1207.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 3970819,
        #'xsec' : 120.4,#pb
        'xsec'    : 120.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1991645,
        #'xsec' : 25.25,#pb
        'xsec'    : 25.25,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    #TTbar
    'TTJets_SingleLeptFromT_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 6.212,#x-sec DB 2017! not existing in 2016
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTJets_SingleLeptFromTbar_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 6.167,#x-sec DB 2017! not existing in 2016 
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'TTJets_DiLept_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 3.655,#x-sec DB 2017! not existing in 2016 
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
    'WW_TuneCUETP8M1_13TeV-pythia8-v1' : {
        'nevents' : 994012,
        'xsec'    : 64.3,#XSDB
        #'xsec'    : 118.7,#B2G-17-005
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    ##WZ
    'WZ_TuneCUETP8M1_13TeV-pythia8-v1' : {
        'nevents' : 1000000,
        'xsec'    : 23.43,#XSDB
        #'xsec'    : 47.2,#B2G-17-005
        'matcheff': 1.,
        'kfactor' : 1.,
    }, 
    #ZZ
    'ZZ_TuneCUETP8M1_13TeV-pythia8-v1' : {
        'nevents' : 990064,
        'xsec'    : 10.16,#XSDB
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


    #DYJetsToLL inclusive madgraph
    'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia_ext1-v1' : {
        'nevents' : 1,
        'xsec'    : 4963.0,#XSDB
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
    'WJetsToLNu_HT-70To100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1353.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1346.0,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 360.1,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 48.8,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 12.07,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 5.497,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 1.329,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1' : {
        'nevents' : 1,
        'xsec'    : 0.03209,#XSDB
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3' : {
        'nevents' : 1,
        'xsec'    : 1,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    #HH old approach
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500' : {
        'nevents' : 164114,
        'xsec'    : 10.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000' : {
        'nevents' : 162306,
        'xsec'    : 10.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500' : {
        'nevents' : 135983,
        'xsec'    : 1e-03*3832.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000' : {
        'nevents' : 136349,
        'xsec'    : 1e-03*3832.31*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500' : {
        'nevents' : 89471,
        'xsec'    : 1e-03*2583.96*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000' : {
        'nevents' : 94055,
        'xsec'    : 1e-03*2583.96*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500' : {
        'nevents' : 67193,
        'xsec'    : 1e-03*1335.62*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000' : {
        'nevents' : 65703,
        'xsec'    : 1e-03*1335.62*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500' : {
        'nevents' : 37517,
        'xsec'    : 1e-03*810.24*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000' : {
        'nevents' : 36228,
        'xsec'    : 1e-03*810.24*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500' : {
        'nevents' : 24403,
        'xsec'    : 1e-03*284.855*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000' : {
        'nevents' : 26652,
        'xsec'    : 1e-03*284.855*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500' : {
        'nevents' : 11962,
        'xsec'    : 1e-03*88.7372*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000' : {
        'nevents' : 14107,
        'xsec'    : 1e-03*88.7372*0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500' : {
        'nevents' : 12831,
        'xsec'    : 1e-03*0.5824*0.5824*14.6677,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000' : {
        'nevents' : 13069,
        'xsec'    : 1e-03*0.5824*0.5824*14.6677,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500' : {
        'nevents' : 12391,
        'xsec'    : 1e-03*0.5824*0.5824*3.46143,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000' : {
        'nevents' : 12560,
        'xsec'    : 1e-03*0.5824*0.5824*3.46143,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500' : {
        'nevents' : 12986,
        'xsec'    : 1e-03*0.5824*0.5824*0.968853,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000' : {
        'nevents' : 13069,
        'xsec'    : 1e-03*0.5824*0.5824*0.968853,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500' : {
        'nevents' : 12265,
        'xsec'    : 1e-03*0.5824*0.5824*0.240471,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000' : {
        'nevents' : 11783,
        'xsec'    : 1e-03*0.5824*0.5824*0.240471,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500' : {
        'nevents' : 12644,
        'xsec'    : 1e-03*0.5824*0.5824*0.,#not done yet!!
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000' : {
        'nevents' : 12283,
        'xsec'    : 1e-03*0.5824*0.5824*0.,#not done yet!!
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500' : {
        'nevents' : 12198,
        'xsec'    : 1e-03*0.5824*0.5824*0.,#not done yet!!
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000' : {
        'nevents' : 11719,
        'xsec'    : 1e-03*0.5824*0.5824*0.,#not done yet!!
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    #HH new x-sec according to https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYNLONNLLCrossSections13TeVHinoAll

    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HH' : {
        'nevents' : 164114,
        'xsec'    : 7.7314,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HH' : {
        'nevents' : 162306,
        'xsec'    : 7.7314,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HH' : {
        'nevents' : 135983,
        'xsec'    : 4.00783,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HH' : {
        'nevents' : 136349,
        'xsec'    : 4.00783,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HH' : {
        'nevents' : 89471,
        'xsec'    : 2.29307,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HH' : {
        'nevents' : 94055,
        'xsec'    : 2.29307,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HH' : {
        'nevents' : 67193,
        'xsec'    : 1.4065,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HH' : {
        'nevents' : 65703,
        'xsec'    : 1.4065,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HH' : {
        'nevents' : 37517,
        'xsec'    : 0.61117,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HH' : {
        'nevents' : 36228,
        'xsec'    : 0.61117,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HH' : {
        'nevents' : 24403,
        'xsec'    : 0.302815,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HH' : {
        'nevents' : 26652,
        'xsec'    : 0.302815,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HH' : {
        'nevents' : 11962,
        'xsec'    : 0.0946976,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HH' : {
        'nevents' : 14107,
        'xsec'    : 0.0946976,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HH' : {
        'nevents' : 12831,
        'xsec'    : 0.0156263,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HH' : {
        'nevents' : 13069,
        'xsec'    : 0.0156263,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HH' : {
        'nevents' : 12391,
        'xsec'    : 0.00366385,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HH' : {
        'nevents' : 12560,
        'xsec'    : 0.00366385,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HH' : {
        'nevents' : 12986,
        'xsec'    : 0.00102654,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HH' : {
        'nevents' : 13069,
        'xsec'    : 0.00102654,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HH' : {
        'nevents' : 12265,
        'xsec'    : 0.000243209,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HH' : {
        'nevents' : 11783,
        'xsec'    : 0.000243209,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HH' : {
        'nevents' : 12644,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HH' : {
        'nevents' : 12283,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HH' : {
        'nevents' : 12198,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HH' : {
        'nevents' : 11719,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.5824*0.5824,
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    #HZ new x-sec according to https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYNLONNLLCrossSections13TeVHinoAll
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HZ' : {
        'nevents' : 395533,
        'xsec'    : 7.7314,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HZ' : {
        'nevents' : 392837,
        'xsec'    : 7.7314,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HZ' : {
        'nevents' : 330303,
        'xsec'    : 4.00783,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HZ' : {
        'nevents' : 331499,
        'xsec'    : 4.00783,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HZ' : {
        'nevents' : 216431,
        'xsec'    : 2.29307,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HZ' : {
        'nevents' : 226886,
        'xsec'    : 2.29307,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HZ' : {
        'nevents' : 162691,
        'xsec'    : 1.4065,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HZ' : {
        'nevents' : 158338,
        'xsec'    : 1.4065,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HZ' : {
        'nevents' : 91576,
        'xsec'    : 0.61117,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HZ' : {
        'nevents' : 86544,
        'xsec'    : 0.61117,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HZ' : {
        'nevents' : 59567,
        'xsec'    : 0.302815,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HZ' : {
        'nevents' : 64168,
        'xsec'    : 0.302815,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HZ' : {
        'nevents' : 29037,
        'xsec'    : 0.0946976,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HZ' : {
        'nevents' : 34203,
        'xsec'    : 0.0946976,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HZ' : {
        'nevents' : 30945,
        'xsec'    : 0.0156263,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HZ' : {
        'nevents' : 32434,
        'xsec'    : 0.0156263,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HZ' : {
        'nevents' : 30203,
        'xsec'    : 0.00366385,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HZ' : {
        'nevents' : 30886,
        'xsec'    : 0.00366385,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HZ' : {
        'nevents' : 30998,
        'xsec'    : 0.00102654,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HZ' : {
        'nevents' : 31893,
        'xsec'    : 0.00102654,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HZ' : {
        'nevents' : 29745,
        'xsec'    : 0.000243209,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HZ' : {
        'nevents' : 28409,
        'xsec'    : 0.000243209,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HZ' : {
        'nevents' : 30818,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HZ' : {
        'nevents' : 29702,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HZ' : {
        'nevents' : 29299,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HZ' : {
        'nevents' : 28280,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.5824*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },

    #ZZ new x-sec according to https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYNLONNLLCrossSections13TeVHinoAll
    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_ZZ' : {
        'nevents' : 239865,
        'xsec'    : 7.7314,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_ZZ' : {
        'nevents' : 237961,
        'xsec'    : 7.7314,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_ZZ' : {
        'nevents' : 200345,
        'xsec'    : 4.00783,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_ZZ' : {
        'nevents' : 199439,
        'xsec'    : 4.00783,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_ZZ' : {
        'nevents' : 130278,
        'xsec'    : 2.29307,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_ZZ' : {
        'nevents' : 137267,
        'xsec'    : 2.29307,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_ZZ' : {
        'nevents' : 98907,
        'xsec'    : 1.4065,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_ZZ' : {
        'nevents' : 96131,
        'xsec'    : 1.4065,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_ZZ' : {
        'nevents' : 55288,
        'xsec'    : 0.61117,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_ZZ' : {
        'nevents' : 52731,
        'xsec'    : 0.61117,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_ZZ' : {
        'nevents' : 35919,
        'xsec'    : 0.302815,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_ZZ' : {
        'nevents' : 38603,
        'xsec'    : 0.302815,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_ZZ' : {
        'nevents' : 17871,
        'xsec'    : 0.0946976,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_ZZ' : {
        'nevents' : 20485,
        'xsec'    : 0.0946976,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_ZZ' : {
        'nevents' : 18894,
        'xsec'    : 0.0156263,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_ZZ' : {
        'nevents' : 19653,
        'xsec'    : 0.0156263,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_ZZ' : {
        'nevents' : 18211,
        'xsec'    : 0.00366385,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_ZZ' : {
        'nevents' : 18756,
        'xsec'    : 0.00366385,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_ZZ' : {
        'nevents' : 18788,
        'xsec'    : 0.00102654,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_ZZ' : {
        'nevents' : 19135,
        'xsec'    : 0.00102654,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_ZZ' : {
        'nevents' : 17825,
        'xsec'    : 0.000243209,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_ZZ' : {
        'nevents' : 17292,
        'xsec'    : 0.000243209,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_ZZ' : {
        'nevents' : 18742,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_ZZ' : {
        'nevents' : 18087,
        'xsec'    : 6.09395e-05,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_ZZ' : {
        'nevents' : 17580,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },


    'SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_ZZ' : {
        'nevents' : 17092,
        'xsec'    : 1.2491e-05,
        'BR'      : 0.69911*0.69911,
        'matcheff': 1.,
        'kfactor' : 1.,
    },



    'n3n2-n1-hbb-hbb_mh400_pl1000' : {
        'nevents' : 1,
        'xsec'    : 88.7325,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh300_pl1000' : {
        'nevents' : 1,
        'xsec'    : 284.855,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh250_pl1000' : {
        'nevents' : 1,
        'xsec'    : 577.314,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh200_pl1000' : {
        'nevents' : 1,
        'xsec'    : 1335.62,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh175_pl1000' : {
        'nevents' : 1,
        'xsec'    : 2583.965,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh150_pl1000' : {
        'nevents' : 1,
        'xsec'    : 3832.31,
        'matcheff': 1.,
        'kfactor' : 1.,
    },
    'n3n2-n1-hbb-hbb_mh127_pl1000' : {
        'nevents' : 1,
        'xsec'    : 10314.755,
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
    'MET' : {
        'order' : 0,
        'files' : [
            #'METRun2016B-07Aug17_ver1-v1',
            'METRun2016B-07Aug17_ver2-v1',
            'METRun2016C-07Aug17_v1',
            'METRun2016D-07Aug17_v1',
            'METRun2016E-07Aug17_v1',
            'METRun2016F-07Aug17_v1',
            'METRun2016G-07Aug17_v1',
            'METRun2016H-07Aug17_v1'
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

    'HighMET' : {
        'order' : 0,
        'files' : [
            ##'HighMETRun2016B-07Aug17_ver1-v1',
            'HighMETRun2016B-07Aug17_ver2-v1',
            'HighMETRun2016C-07Aug17_v1',
            'HighMETRun2016D-07Aug17_v1',
            'HighMETRun2016E-07Aug17_v1',
            'HighMETRun2016F-07Aug17_v1',
            #'HighMETRun2016G-07Aug17_v1',
            #'HighMETRun2016H-07Aug17_v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET 2016",
        'weight': 1.,
        'plot': True,
    },

    'HighMETCopy' : {
        'order' : 0,
        'files' : [
            ##'HighMETRun2016B-07Aug17_ver1-v1',
            'HighMETRun2016B-07Aug17_ver2-v1',
            'HighMETRun2016C-07Aug17_v1',
            'HighMETRun2016D-07Aug17_v1',
            'HighMETRun2016E-07Aug17_v1',
            'HighMETRun2016F-07Aug17_v1',
            'HighMETRun2016G-07Aug17_v1',
            'HighMETRun2016H-07Aug17_v1'
        ],
        'fillcolor' : 8,
        'fillstyle' : 1001,
        'linecolor' : 8,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET 2016, bin1",
        'weight': 1.,
        'plot': True,
    },


    'Event277096' : {
        'order' : 0,
        'files' : [
            'skim_pickevents_277096_153555117',
        ],
        'fillcolor' : 418,#632-7,
        'fillstyle' : 1001,
        'linecolor' : 418,#632-7,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Event 277096",
        'weight': 1.,
        'plot': True,
    },

    'HighMETBH' : {
        'order' : 0,
        'files' : [
            ##'HighMETRun2016B-07Aug17_ver1-v1',
            'HighMETRun2016B-07Aug17_ver2-v1BH',
            'HighMETRun2016C-07Aug17_v1BH',
            'HighMETRun2016D-07Aug17_v1BH',
            'HighMETRun2016E-07Aug17_v1BH',
            'HighMETRun2016F-07Aug17_v1BH',
            'HighMETRun2016G-07Aug17_v1BH',
            'HighMETRun2016H-07Aug17_v1BH'
        ],
        'fillcolor' : 861,#632-7,
        'fillstyle' : 1001,
        'linecolor' : 861,#632-7,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "HighMET with Beam Halo",
        'weight': 1.,
        'plot': True,
    },

    'SingleMuon' : {
        'order' : 0,
        'files' : [
            ##'SingleMuonRun2016B-07Aug17_ver1-v1',
            #'SingleMuonRun2016B-07Aug17_ver2-v1',
            #'SingleMuonRun2016C-07Aug17-v1',
            #'SingleMuonRun2016D-07Aug17-v1',
            #'SingleMuonRun2016E-07Aug17-v1',
            #'SingleMuonRun2016F-07Aug17-v1',
            'SingleMuonRun2016G-07Aug17-v1',
            'SingleMuonRun2016H-07Aug17-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon 2016",
        'weight': 1.,
        'plot': True,
    },
    'SingleElectron' : {
        'order' : 0,
        'files' : [
            ##'SingleElectronRun2016B-07Aug17_ver1-v1',
            #'SingleElectronRun2016B-07Aug17_ver2-v2',
            #'SingleElectronRun2016C-07Aug17-v1',
            #'SingleElectronRun2016D-07Aug17-v1',
            #'SingleElectronRun2016E-07Aug17-v1',
            #'SingleElectronRun2016F-07Aug17-v1',
            'SingleElectronRun2016G-07Aug17-v1',
            'SingleElectronRun2016H-07Aug17-v1'#only one good
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleEle 2016",
        'weight': 1.,
        'plot': True,
    },


    'SingleLepton' : {
        'order' : 0,
        'files' : [
            ##'SingleMuonRun2016B-07Aug17_ver1-v1',
            #'SingleMuonRun2016B-07Aug17_ver2-v1',
            #'SingleMuonRun2016C-07Aug17-v1',
            #'SingleMuonRun2016D-07Aug17-v1',
            #'SingleMuonRun2016E-07Aug17-v1',
            #'SingleMuonRun2016F-07Aug17-v1',
            'SingleMuonRun2016G-07Aug17-v1',
            'SingleMuonRun2016H-07Aug17-v1',
            'SingleElectronRun2016G-07Aug17-v1',
            'SingleElectronRun2016H-07Aug17-v1'#only one good
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SingleMuon+Ele 2016",
        'weight': 1.,
        'plot': True,
    },

    'JetHT' : {
        'order' : 0,
        'files' : [
            ##'JetHTRun2016B-07Aug17_ver1-v1',
            #'JetHTRun2016B-07Aug17_ver2-v1',
            #'JetHTRun2016C-07Aug17-v1',
            #'JetHTRun2016D-07Aug17-v1',
            #'JetHTRun2016E-07Aug17-v1',
            #'JetHTRun2016F-07Aug17-v1',
            'JetHTRun2016G-07Aug17-v1',
            'JetHTRun2016H-07Aug17-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "JetHT 2016",
        'weight': 1.,
        'plot': True,
    },
    'MuonEG' : {
        'order' : 0,
        'files' : [
            ##'MuonEGRun2016B-07Aug17_ver1-v1',
            #'MuonEGRun2016B-07Aug17_ver2-v1',
            #'MuonEGRun2016C-07Aug17-v1',
            #'MuonEGRun2016D-07Aug17-v1',
            #'MuonEGRun2016E-07Aug17-v1',
            #'MuonEGRun2016F-07Aug17-v1',
            'MuonEGRun2016G-07Aug17-v1',
            'MuonEGRun2016H-07Aug17-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "MuonEG 2016",
        'weight': 1.,
        'plot': True,
    },
    'SinglePhoton' : {
        'order' : 0,
        'files' : [
            ##'SinglePhotonRun2016B-07Aug17_ver1-v1',
            'SinglePhotonRun2016B-07Aug17_ver2-v1',
            'SinglePhotonRun2016C-07Aug17-v1',
            'SinglePhotonRun2016D-07Aug17-v1',
            'SinglePhotonRun2016E-07Aug17-v1',
            'SinglePhotonRun2016F-07Aug17-v1',
            #'SinglePhotonRun2016G-07Aug17-v1',
            #'SinglePhotonRun2016H-07Aug17-v1'
        ],
        'fillcolor' : 0,
        'fillstyle' : 1,
        'linecolor' : 1,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "SinglePhoton 2016",
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
    #QCD
    'All' : {
        'files' : [
            'QCD_HT50to100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'QCD_HT100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
            'QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
            'QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
            'QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
            'QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
            'QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'TTJets_SingleLeptFromTbar_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'TTJets_SingleLeptFromT_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'TTJets_DiLept_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'ZJetsToNuNu_HT-100To200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-200To400_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-400To600_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-600To800_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-800To1200_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-1200To2500_13TeV-madgraph-v1', 
            'ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph-v1',
            'WJetsToLNu_HT-70To100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            #'WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WW_TuneCUETP8M1_13TeV-pythia8-v1', 
            'WZ_TuneCUETP8M1_13TeV-pythia8-v1', 
            'ZZ_TuneCUETP8M1_13TeV-pythia8-v1'
        ],
        'fillcolor' : 920,
        'fillstyle' : 1001,
        'linecolor' : 920,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : 'All',
        'title' : 'All',
        'weight': 1.,
        'plot': True,
    },
    'QCD' : {
        'files' : [
             'QCD_HT50to100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
             'QCD_HT100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
             'QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
             'QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
             'QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
             'QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
             'QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
             'QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1', 
             'QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1'],
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

    #TTbarGenMET
    'TTbarGenMET' : {
        'files' : [
             'TTJets_SingleLeptFromTbar_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
             'TTJets_SingleLeptFromT_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
             'TTJets_DiLept_genMET-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
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
    'TTbar' : {
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

    #DYJetsToLL
    'DYJetsToLL' : {
        'files' : ['DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia_ext1-v1'],
        'fillcolor' : 418,
        'fillstyle' : 1001,
        'linecolor' : 418,
        'linewidth' : 2,
        'linestyle' : 1,
        'label' : "Z/#gamma #rightarrow ll + jets",
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
            'WJetsToLNu_HT-70To100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-200To400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-600To800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-800To1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-1200To2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
            'WJetsToLNu_HT-2500ToInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8-v1',
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
    #Dibosons
    'VV' : {
        'files' : [
            'WW_TuneCUETP8M1_13TeV-pythia8-v1', 
            'WZ_TuneCUETP8M1_13TeV-pythia8-v1', 
            'ZZ_TuneCUETP8M1_13TeV-pythia8-v1'
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
    #Central SUSY
    'SUSY_mh127_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500'],
        'mass' : 127,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh127_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000'],
        'mass' : 127,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #150
    'SUSY_mh150_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500'],
        'mass' : 150,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh150_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000'],
        'mass' : 150,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #175
    'SUSY_mh175_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500'],
        'mass' : 175,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh175_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000'],
        'mass' : 175,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #200
    'SUSY_mh200_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500'],
        'mass' : 200,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh200_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000'],
        'mass' : 200,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #250
    'SUSY_mh250_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500'],
        'mass' : 250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh250_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000'],
        'mass' : 250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #300
    'SUSY_mh300_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500'],
        'mass' : 300,
        'ctau' : 500,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh300_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000'],
        'mass' : 300,
        'ctau' : 3000,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #400
    'SUSY_mh400_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500'],
        'mass' : 400,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh400_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000'],
        'mass' : 400,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #600
    'SUSY_mh600_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500'],
        'mass' : 600,
        'ctau' : 500,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh600_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000'],
        'mass' : 600,
        'ctau' : 3000,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #800
    'SUSY_mh800_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500'],
        'mass' : 800,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh800_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000'],
        'mass' : 800,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1000
    'SUSY_mh1000_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500'],
        'mass' : 1000,
        'ctau' : 500,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1000_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000'],
        'mass' : 1000,
        'ctau' : 3000,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1250
    'SUSY_mh1250_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500'],
        'mass' : 1250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1250_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000'],
        'mass' : 1250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1500
    'SUSY_mh1500_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500'],
        'mass' : 1500,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1500_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000'],
        'mass' : 1500,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1800
    'SUSY_mh1800_ctau500' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500'],
        'mass' : 1800,
        'ctau' : 500,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1800_ctau3000' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000'],
        'mass' : 1800,
        'ctau' : 3000,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #HH
    #Central SUSY
    'SUSY_mh127_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HH'],
        'mass' : 127,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh127_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HH'],
        'mass' : 127,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #150
    'SUSY_mh150_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HH'],
        'mass' : 150,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh150_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HH'],
        'mass' : 150,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #175
    'SUSY_mh175_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HH'],
        'mass' : 175,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh175_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HH'],
        'mass' : 175,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #200
    'SUSY_mh200_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HH'],
        'mass' : 200,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh200_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HH'],
        'mass' : 200,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #250
    'SUSY_mh250_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HH'],
        'mass' : 250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh250_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HH'],
        'mass' : 250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #300
    'SUSY_mh300_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HH'],
        'mass' : 300,
        'ctau' : 500,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh300_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HH'],
        'mass' : 300,
        'ctau' : 3000,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #400
    'SUSY_mh400_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HH'],
        'mass' : 400,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh400_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HH'],
        'mass' : 400,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #600
    'SUSY_mh600_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HH'],
        'mass' : 600,
        'ctau' : 500,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh600_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HH'],
        'mass' : 600,
        'ctau' : 3000,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #800
    'SUSY_mh800_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HH'],
        'mass' : 800,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh800_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HH'],
        'mass' : 800,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1000
    'SUSY_mh1000_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HH'],
        'mass' : 1000,
        'ctau' : 500,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1000_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HH'],
        'mass' : 1000,
        'ctau' : 3000,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1250
    'SUSY_mh1250_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HH'],
        'mass' : 1250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1250_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HH'],
        'mass' : 1250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1500
    'SUSY_mh1500_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HH'],
        'mass' : 1500,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1500_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HH'],
        'mass' : 1500,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1800
    'SUSY_mh1800_ctau500_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HH'],
        'mass' : 1800,
        'ctau' : 500,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1800_ctau3000_HH' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HH'],
        'mass' : 1800,
        'ctau' : 3000,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },

    #HZ
    #Central SUSY
    'SUSY_mh127_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_HZ'],
        'mass' : 127,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh127_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_HZ'],
        'mass' : 127,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #150
    'SUSY_mh150_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_HZ'],
        'mass' : 150,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh150_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_HZ'],
        'mass' : 150,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #175
    'SUSY_mh175_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_HZ'],
        'mass' : 175,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh175_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_HZ'],
        'mass' : 175,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #200
    'SUSY_mh200_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_HZ'],
        'mass' : 200,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh200_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_HZ'],
        'mass' : 200,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #250
    'SUSY_mh250_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_HZ'],
        'mass' : 250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh250_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_HZ'],
        'mass' : 250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #300
    'SUSY_mh300_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_HZ'],
        'mass' : 300,
        'ctau' : 500,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh300_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_HZ'],
        'mass' : 300,
        'ctau' : 3000,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #400
    'SUSY_mh400_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_HZ'],
        'mass' : 400,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh400_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_HZ'],
        'mass' : 400,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #600
    'SUSY_mh600_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_HZ'],
        'mass' : 600,
        'ctau' : 500,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh600_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_HZ'],
        'mass' : 600,
        'ctau' : 3000,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #800
    'SUSY_mh800_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_HZ'],
        'mass' : 800,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh800_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_HZ'],
        'mass' : 800,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1000
    'SUSY_mh1000_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_HZ'],
        'mass' : 1000,
        'ctau' : 500,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1000_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_HZ'],
        'mass' : 1000,
        'ctau' : 3000,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1250
    'SUSY_mh1250_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_HZ'],
        'mass' : 1250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1250_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_HZ'],
        'mass' : 1250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1500
    'SUSY_mh1500_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_HZ'],
        'mass' : 1500,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1500_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_HZ'],
        'mass' : 1500,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1800
    'SUSY_mh1800_ctau500_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_HZ'],
        'mass' : 1800,
        'ctau' : 500,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1800_ctau3000_HZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_HZ'],
        'mass' : 1800,
        'ctau' : 3000,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #ZZ
    #Central SUSY
    'SUSY_mh127_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau500_ZZ'],
        'mass' : 127,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh127_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m127_ctau3000_ZZ'],
        'mass' : 127,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 127 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #150
    'SUSY_mh150_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau500_ZZ'],
        'mass' : 150,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh150_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m150_ctau3000_ZZ'],
        'mass' : 150,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 150 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #175
    'SUSY_mh175_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau500_ZZ'],
        'mass' : 175,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh175_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m175_ctau3000_ZZ'],
        'mass' : 175,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 175 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #200
    'SUSY_mh200_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau500_ZZ'],
        'mass' : 200,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh200_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m200_ctau3000_ZZ'],
        'mass' : 200,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 200 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #250
    'SUSY_mh250_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau500_ZZ'],
        'mass' : 250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh250_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m250_ctau3000_ZZ'],
        'mass' : 250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #300
    'SUSY_mh300_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau500_ZZ'],
        'mass' : 300,
        'ctau' : 500,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh300_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m300_ctau3000_ZZ'],
        'mass' : 300,
        'ctau' : 3000,
        'fillcolor' : 800,
        'fillstyle' : 0,
        'linecolor' : 800,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 300 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #400
    'SUSY_mh400_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau500_ZZ'],
        'mass' : 400,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh400_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m400_ctau3000_ZZ'],
        'mass' : 400,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 400 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #600
    'SUSY_mh600_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau500_ZZ'],
        'mass' : 600,
        'ctau' : 500,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh600_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m600_ctau3000_ZZ'],
        'mass' : 600,
        'ctau' : 3000,
        'fillcolor' : 418,
        'fillstyle' : 0,
        'linecolor' : 418,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 600 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #800
    'SUSY_mh800_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau500_ZZ'],
        'mass' : 800,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh800_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m800_ctau3000_ZZ'],
        'mass' : 800,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1000
    'SUSY_mh1000_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau500_ZZ'],
        'mass' : 1000,
        'ctau' : 500,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1000_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1000_ctau3000_ZZ'],
        'mass' : 1000,
        'ctau' : 3000,
        'fillcolor' : 856,
        'fillstyle' : 0,
        'linecolor' : 856,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1000 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1250
    'SUSY_mh1250_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau500_ZZ'],
        'mass' : 1250,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1250_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1250_ctau3000_ZZ'],
        'mass' : 1250,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1250 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1500
    'SUSY_mh1500_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau500_ZZ'],
        'mass' : 1500,
        'ctau' : 500,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1500_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1500_ctau3000_ZZ'],
        'mass' : 1500,
        'ctau' : 3000,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1500 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },
    #1800
    'SUSY_mh1800_ctau500_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau500_ZZ'],
        'mass' : 1800,
        'ctau' : 500,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 0.5 m",
        'weight': 1.,
        'plot': True,
    },
    'SUSY_mh1800_ctau3000_ZZ' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3_m1800_ctau3000_ZZ'],
        'mass' : 1800,
        'ctau' : 3000,
        'fillcolor' : 602,
        'fillstyle' : 0,
        'linecolor' : 602,
        'linewidth' : 3,
        'linestyle' : 7,
        'label' : "m_{#chi} = 1800 GeV, c#tau_{0} = 3 m",
        'weight': 1.,
        'plot': True,
    },

#  JiaJing's AOD
    'SUSY_central' : {
        'files' : ['SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3'],
        'mass' : 0,
        'ctau' : 0,
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
        'linewidth' : 3,
        'linestyle' : 1,
        'label' : "m_{#chi} = 0 GeV, c#tau_{0} = 0 m",
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
        'fillcolor' : 2,
        'fillstyle' : 0,
        'linecolor' : 2,
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
        'fillcolor' : 3,
        'fillstyle' : 0,
        'linecolor' : 3,
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
