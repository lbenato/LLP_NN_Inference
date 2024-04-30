#!/usr/bin/env python

import os
import subprocess
#import ROOT as ROOT
import multiprocessing
from collections import defaultdict
#import numpy as np
#import uproot
#import root_numpy

run_condor = True

# # # # # #
# Run era #
# # # # # #
RUN_ERA = 2016#6#7#8

INPUTDIR = ("/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_%s_reinterpretation/")%(RUN_ERA)
#OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_%s_reinterpretation/post_processed/")%(RUN_ERA)
OUTPUTDIR = ("/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_%s_reinterpretation/post_processed_v2/")%(RUN_ERA)

HOMEDIR = '/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/'

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


sign = [
        'SUSY_mh127_ctau500_HH', 'SUSY_mh127_ctau3000_HH',
        'SUSY_mh150_ctau500_HH', 'SUSY_mh150_ctau3000_HH',
        'SUSY_mh175_ctau500_HH', 'SUSY_mh175_ctau3000_HH',
        'SUSY_mh200_ctau500_HH', 'SUSY_mh200_ctau3000_HH',
        'SUSY_mh250_ctau500_HH', 'SUSY_mh250_ctau3000_HH',
        'SUSY_mh300_ctau500_HH', 'SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH', 'SUSY_mh400_ctau3000_HH',
        'SUSY_mh600_ctau500_HH', 'SUSY_mh600_ctau3000_HH',
        'SUSY_mh800_ctau500_HH', 'SUSY_mh800_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH','SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1250_ctau500_HH','SUSY_mh1250_ctau3000_HH',
        'SUSY_mh1500_ctau500_HH','SUSY_mh1500_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH', 'SUSY_mh1800_ctau3000_HH',

        'SUSY_mh127_ctau500_HZ', 'SUSY_mh127_ctau3000_HZ',
        'SUSY_mh150_ctau500_HZ', 'SUSY_mh150_ctau3000_HZ',
        'SUSY_mh175_ctau500_HZ', 'SUSY_mh175_ctau3000_HZ',
        'SUSY_mh200_ctau500_HZ', 'SUSY_mh200_ctau3000_HZ',
        'SUSY_mh250_ctau500_HZ', 'SUSY_mh250_ctau3000_HZ',
        'SUSY_mh300_ctau500_HZ', 'SUSY_mh300_ctau3000_HZ',
        'SUSY_mh400_ctau500_HZ', 'SUSY_mh400_ctau3000_HZ',
        'SUSY_mh600_ctau500_HZ', 'SUSY_mh600_ctau3000_HZ',
        'SUSY_mh800_ctau500_HZ', 'SUSY_mh800_ctau3000_HZ',
        'SUSY_mh1000_ctau500_HZ','SUSY_mh1000_ctau3000_HZ',
        'SUSY_mh1250_ctau500_HZ','SUSY_mh1250_ctau3000_HZ',
        'SUSY_mh1500_ctau500_HZ','SUSY_mh1500_ctau3000_HZ',
        'SUSY_mh1800_ctau500_HZ', 'SUSY_mh1800_ctau3000_HZ',

        'SUSY_mh127_ctau500_ZZ', 'SUSY_mh127_ctau3000_ZZ',
        'SUSY_mh150_ctau500_ZZ', 'SUSY_mh150_ctau3000_ZZ',
        'SUSY_mh175_ctau500_ZZ', 'SUSY_mh175_ctau3000_ZZ',
        'SUSY_mh200_ctau500_ZZ', 'SUSY_mh200_ctau3000_ZZ',
        'SUSY_mh250_ctau500_ZZ', 'SUSY_mh250_ctau3000_ZZ',
        'SUSY_mh300_ctau500_ZZ', 'SUSY_mh300_ctau3000_ZZ',
        'SUSY_mh400_ctau500_ZZ', 'SUSY_mh400_ctau3000_ZZ',
        'SUSY_mh600_ctau500_ZZ', 'SUSY_mh600_ctau3000_ZZ',
        'SUSY_mh800_ctau500_ZZ', 'SUSY_mh800_ctau3000_ZZ',
        'SUSY_mh1000_ctau500_ZZ','SUSY_mh1000_ctau3000_ZZ',
        'SUSY_mh1250_ctau500_ZZ','SUSY_mh1250_ctau3000_ZZ',
        'SUSY_mh1500_ctau500_ZZ','SUSY_mh1500_ctau3000_ZZ',
        'SUSY_mh1800_ctau500_ZZ', 'SUSY_mh1800_ctau3000_ZZ',
        ]


sample_list = sign

dicty = {}
#for s in sign:

sample_to_loop = []
for s in sample_list:
    for ss in samples[s]["files"]:
        sample_to_loop.append(ss)



if run_condor:
    print "Run condor!"
    NCPUS   = 1
    MEMORY  = 512#4000#5000#2000 too small?#10000#tried 10 GB for a job killed by condor automatically
    RUNTIME = 3600#3600*12#86400
  
    for s in sample_to_loop:
        print s, ": . . ."

        #Time smearing data file
        #TODO 2016 together or split???
        smearing_file = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+str(RUN_ERA)+"_TtoEM_v5_ntuples_validate_timeRecHits/data_smear_file_CSV_0p8_all_jets.root"

        #read input files
        in_file  = INPUTDIR  + s+".root"
        out_file = OUTPUTDIR + s+".root"

        if not(os.path.isfile(in_file)):
            print in_file , " : file does not exist, go to next..."
            continue

        os.chdir('/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src')
        #os.system('../bin/slc7_amd64_gcc820/post_processing_v7_reinterpretation ' + in_file + '  ' + out_file  + ' ' + str(RUN_ERA) + ' ' + smearing_file + ' \n')
        os.system('../bin/slc7_amd64_gcc820/post_processing_v7_reinterpretation_v2 ' + in_file + '  ' + out_file  + ' ' + str(RUN_ERA) + ' ' + smearing_file + ' \n')
        os.system('echo \n')


        '''
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
        fout.write('../bin/slc7_amd64_gcc820/tf_and_skim_v7_reinterpretation ' + INPS+root_files[b] + '  ' + OUT+root_files[b] + skip_string  + ' ' + str(isSignal) + ' ' + str(isData)  + ' ' + mc_PU_file+ ' ' + mc_trigger_file + ' ' + mc_trigger_string + ' ' + doRegion + ' ' + smearing_file + ' \n')

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

        #print "Only 1 root file per job!!!"
        #exit()
        j_num +=1

        os.chdir('../../')
        '''

os.chdir(HOMEDIR)
