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
RUN_ERA = 2016#6#7#8

doRegion = "doSR"#"doSR"#"doTtoEM"#"doZtoEE"#MN/EN
resubm_label = ""
#resubm_label = "_resubmission_3"
##

#Remember: all v5 MC don't have correct JER smearing, only v6
if resubm_label=="_resubmission_0" or resubm_label=="":
    if doRegion=="doSR" and RUN_ERA==2018:
        #print "!!!!!!"
        #print "!!!!!!"
        #print "check v5"
        #print "!!!!!!"
        #print "!!!!!!"
        #INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_30March2021_HEM/")% str(RUN_ERA)
        INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_31December2020"+"/")% str(RUN_ERA)
    else:
        INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_31December2020"+"/")% str(RUN_ERA)
else:
    #INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_31December2020"+resubm_label+"/")% str(RUN_ERA)
    INPUTDIR = ("/pnfs/desy.de/cms/tier2/store/user/lbenato/v5_calo_AOD_%s_31December2020"+resubm_label+"/")% str(RUN_ERA)

#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_%s_SR_xcheck_tf_and_skim_condor_v5_updated"+resubm_label+"/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_TtoEM"+resubm_label+"_v5_ntuples_validate_timeRecHits/")%(RUN_ERA)
OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR"+resubm_label+"_v5_ntuples/")%(RUN_ERA)

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

else:
    print "Invalid run era, aborting..."
    exit()


dicty_o = defaultdict()


#data = ["EGamma"]
#data = ["SingleMuon"]
data = ["SingleElectron"]
#data = ["SinglePhoton"]
#data = ["MuonEG"]
#data = ["MET"]
#data = ["HighMET"]
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
        'SUSY_mh400_pl1000', 'SUSY_mh300_pl1000', 'SUSY_mh250_pl1000', 'SUSY_mh200_pl1000', 'SUSY_mh175_pl1000', 'SUSY_mh150_pl1000', 'SUSY_mh127_pl1000',
        #'SUSY_central',
        #'SUSY_mh400_prompt', 'SUSY_mh300_prompt', 'SUSY_mh200_prompt',
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

sample_list = back#data#back#data#back#sign#data#sign#back#sign#data#sign#data#back#sign#data#back#sign#data#sign#back#data#back#sign#sign#data#back#sign#data#back#data#data+back

dicty = {}
#for s in sign:

for s in sample_list:
    for ss in samples[s]["files"]:
        print ss
        print requests[ss]
        s1 = requests[ss][1:].split('/')[0]
        print s1
        dicty[ss] = s1+'/crab_'+ss+'/'
        if s=="DYJetsToLL" and RUN_ERA==2018:
            new_ss = ss.replace('pythia8','pythia')
            dicty[ss] = s1+'8/crab_'+new_ss+'/'

#print dicty_o
#print "new:"
#print dicty
#exit()

if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 4000#5000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 3600*12#86400
    root_files_per_job = 100#40#20#40#
    
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
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            print "Writing results in ", OUT
            OUT += "/" + subd + '/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)

            cond_name = "skim_"+str(RUN_ERA)+"_v5_xcheck_"+doRegion#+"_BeamHalo"
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
                    #here loop over the files
                    for b in np.arange(start,stop+1):
                        #print b, root_files[b]
                        fout.write('echo "Processing '+ root_files[b]  +' . . ." \n')
                        #HEM
                        fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v5_updated ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' '  + doRegion + ' \n')
                        fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v5_updated ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' \n')
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
                os.system('condor_submit ' + 'submit_skim_'+str(j_num)+'.submit' + ' \n')            
                #os.system('sh job_skim_'+str(j_num)+'.sh \n')

                #if j_num<9:
                #    print "job n. ", j_num, " non lo faccio"
                #else:
                #    print "job n. ", j_num, " DA FARE"
                #    os.system('sh job_skim_'+str(j_num)+'.sh \n')
                j_num +=1

            os.chdir('../../')

else:
    ROOT.gROOT.LoadMacro("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/utils/skimJetsAcceptanceCaloFast.cc")
    for s in (back):
        for ss in samples[s]["files"]:
            print ss
            filename = ss + ".root"
            ROOT.skimJetsAcceptanceCaloFast(INPUTDIR+filename,OUTPUTDIR+filename,0,-1,True)
