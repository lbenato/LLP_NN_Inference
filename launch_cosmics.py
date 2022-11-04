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
RUN_ERA = 2018
#only for loading the dataset names

#Test with 2/3 stations:
INPUTDIR = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_cosmics_"+str(RUN_ERA)+"_27January2022/"
OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"/"
#OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"_noCosmicMuons/"
OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"_2stations/"
####OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"_cross_check/"
##OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"_no3stations_noCosmicMuons/"

#Study on cosmic MC
OUTPUTDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_"+str(RUN_ERA)+"/"

if not(os.path.exists(OUTPUTDIR)):
    os.mkdir(OUTPUTDIR)

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

data = ["Cosmics"]
data = ["CosmicsMC"]
sample_list = data
dicty = {}

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
        if "JER" in INPUTDIR:
            dicty[ss] = s1+'/crab_'+ss+'_HH/'

if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 512#4000#####5000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 360/2#3600*12#####86400
    root_files_per_job = 100#40#20#40#
    
    sample_to_loop = dicty.keys()
    for s in sample_to_loop:
        print s, ": . . ."

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
        num_subdirs = [x for x in os.listdir(IN) if os.path.isdir(os.path.join(IN, x))]

        #Here must implement having multiple subdirs
        for subd in num_subdirs:
            INPS = IN + "/"+subd+"/"
            root_files = [x for x in os.listdir(INPS) if os.path.isfile(os.path.join(INPS, x))]

            #create out subdir
            OUT = OUTPUTDIR + s
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)
            print "Writing results in ", OUT
            OUT += "/" + subd + '/'
            if not(os.path.exists(OUT)):
                os.mkdir(OUT)

            cond_name = "skim_"+str(RUN_ERA)+"_v6_cosmics_local"
            COND_DIR = '/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/condor_'+cond_name
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

                        #Cosmic MC
                        fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_cosmics_MC ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + ' \n')
                        fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_cosmics_MC ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + ' \n')
                        #Cosmic data
                        #fout.write('echo ../bin/slc7_amd64_gcc820/tf_and_skim_v6_cosmics ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + ' \n')
                        #fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v6_cosmics ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + ' \n')

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
                #os.system('condor_submit ' + 'submit_skim_'+str(j_num)+'.submit' + ' \n')            
                os.system('sh job_skim_'+str(j_num)+'.sh \n')

                j_num +=1

            os.chdir('../../')
else:
    os.chdir('/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src')
    os.system('../bin/slc7_amd64_gcc820/tf_and_skim_v6_cosmics /pnfs/desy.de/cms/tier2/store/user/lbenato/v6_calo_cosmics_2018_27January2022/Cosmics/crab_CosmicsRun2018D-CosmicSP-PromptReco-v1/220127_110017/0000/output_18.root   /afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/test.root   \n')
    #/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/Analyzer/LLP2018/output.root
