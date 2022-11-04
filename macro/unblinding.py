#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
#import root_numpy
import numpy as np
import yaml
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TMultiGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
#from sklearn.cluster import DBSCAN
#from sklearn.linear_model import LinearRegression
#from scipy.stats import chisquare
#import matplotlib.pyplot as plt
from collections import defaultdict
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *


ERA = "all"#"2016"#
TAG = ""#
#TAG = "GH"

LUMI = -1
if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi["HighMET"]["tot"]
if ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi["HighMET"]["tot"]
if ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples, sample
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    if "GH" in TAG:
        LUMI  = lumi["HighMET"]["G"]+lumi["HighMET"]["H"] 
    elif "BF" in TAG:
        LUMI  = lumi["HighMET"]["B"]+lumi["HighMET"]["C"]+lumi["HighMET"]["D"]+lumi["HighMET"]["E"]+lumi["HighMET"]["F"]
    else:
        LUMI  = lumi["HighMET"]["tot"]

if ERA=="all":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import samples as samples2018
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples as samples2017
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import samples as samples2016
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016
    LUMI = lumi2016["HighMET"]["tot"] + lumi2017["HighMET"]["tot"] + lumi2018["HighMET"]["tot"]

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

data = [
    "HighMET"+TAG,
]

sign = ["SUSY_mh400_ctau500_HH","SUSY_mh400_ctau3000_HH"]

#data = ["HighMET"]
MAIN = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
MAIN_S = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_time_smeared/"

#YIELDDIR_BASE      = "plots/Yields_AN_fix/v6_calo_AOD_"+ERA+"_"
YIELDDIR_BASE      = "plots/Yields_AN_fix_ARC/v6_calo_AOD_"+ERA+"_"
UNCDIR             = "plots/v6_calo_AOD_"+ERA+"_SR_signal_uncertainties_fix/"
UNCDIR_ALL         = "plots/v6_calo_AOD_all_SR_signal_uncertainties_fix/"

OUT_pho = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_ZtoLLPho_v5_ntuples_updated/smearing/"
OUT_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+ERA+"_E_v5_ntuples_reweighted/weighted/smearing/"

OUT = "plots/v6_calo_AOD_"+ERA+"_unblinding_ARC/"

PRELIMINARY = True
OUT_paper = "fig/Preliminary/" if PRELIMINARY else "fig/"

dnn_threshold = 0.996

def count_data(label):

    #Open background prediction files, performed per era
    main_pred_reg    = "SR"                                                                                      
    main_pred_sample = "HighMET"
    extr_region      = "WtoLN"
    dataset_label = ""

    PREDDIR = YIELDDIR_BASE+main_pred_reg+"/"
    pred_file_name = PREDDIR+"BkgPredResults_"+ERA+"_"+main_pred_reg+"_"+main_pred_sample
    pred_file_name+="_eta_1p0"
    if ERA!="2016":
        pred_file_name+="_phi_cut"
    else:
        dataset_label += "_"+TAG[0]+"-"+TAG[1]
    pred_file_name+= "_vs_eta"

    #tag_label: part of the yaml file name, but also part of the dictionary keys
    tag_label = ""
    if ERA=="2016":
        tag_label+=dataset_label###"_"+TAG[0]+"-"+TAG[1]
    tag_label += "_MinDPhi_0p5"
    pred_file_name+= tag_label

    #Mistag
    with open(pred_file_name+".yaml","r") as f:
        #print "Info: opening dictionary in file "+pred_file_name+".yaml"
        results = yaml.load(f, Loader=yaml.Loader)
        #print results
        f.close()
        #print "  --  --  --  --  --"

    #Mistag uncertainties
    #with open(UNCDIR+"signal_bkg_datacard_unc"+dataset_label+".yaml","r") as f:
    with open(UNCDIR+"signal_bkg_unc"+dataset_label+".yaml","r") as f:
        uncertainties = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #print uncertainties
        #print "  --  --  --  --  --"

    #Cosmics
    cosmic_file_name = PREDDIR+"BkgPredResults_"+ERA+"_SR_cosmic"+dataset_label+".yaml"
    with open(cosmic_file_name,"r") as f:
        #print "Info: opening dictionary in file "+cosmic_file_name
        cosmic = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #print cosmic
        #print "  --  --  --  --  --"

    #Beam halo
    bh_file_name = PREDDIR+"BkgPredResults_"+ERA+"_SR_beam_halo"+dataset_label+".yaml"
    with open(bh_file_name,"r") as f:
        #print "Info: opening dictionary in file "+bh_file_name
        bh = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #print bh
        #print "  --  --  --  --  --"


    y2_mistag  = results[extr_region+tag_label][main_pred_sample]['pred_2_from_1']
    y2_cosmic = cosmic["bkg_cosmic"]
    y2_bh = bh["bkg_bh"]

    y2_unc_mistag  = uncertainties['bkg_tot']/100.
    y2_unc_cosmic = cosmic["unc_cosmic"]/100.
    y2_unc_bh = bh["unc_bh"]/100.

    y1_mistag  = results[extr_region+tag_label][main_pred_sample]['pred_1']
    y1_unc_mistag = uncertainties['bkg_tot_bin1']/100.

    print "SF files: ",OUT_pho+"data_MC_SF"+dataset_label+"_1bin.root"
    sf_pho_file = TFile(OUT_pho+"data_MC_SF"+dataset_label+"_1bin.root","READ")
    sf_pho_file.cd()
    sf_pho_1ns = sf_pho_file.Get("ratio_1ns")
    sf_pho_2ns = sf_pho_file.Get("ratio_2ns")
    sf_pho_1ns.SetDirectory(0)
    sf_pho_2ns.SetDirectory(0)
    sf_pho_file.Close()

    sf_ele_file = TFile(OUT_ele+"data_MC_SF"+dataset_label+"_1bin.root","READ")
    sf_ele_file.cd()
    sf_ele_1ns = sf_ele_file.Get("ratio_1ns")
    sf_ele_2ns = sf_ele_file.Get("ratio_2ns")
    sf_ele_1ns.SetDirectory(0)
    sf_ele_2ns.SetDirectory(0)
    sf_ele_file.Close()

    sf_pho = sf_pho_1ns.GetBinContent(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinContent(1)
    sf_unc_pho = sf_pho_1ns.GetBinError(1) if ( abs(1-sf_pho_1ns.GetBinContent(1))>abs(1-sf_pho_2ns.GetBinContent(1)) ) else sf_pho_2ns.GetBinError(1)

    sf_ele = sf_ele_1ns.GetBinContent(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinContent(1)
    sf_unc_ele = sf_ele_1ns.GetBinError(1) if ( abs(1-sf_ele_1ns.GetBinContent(1))>abs(1-sf_ele_2ns.GetBinContent(1)) ) else sf_ele_2ns.GetBinError(1)

    chunk_size = 100000
    ev_count = defaultdict(dict)

    tree_weight_dict = defaultdict(dict)

    for b in sign:
        for i, ss in enumerate(samples[b]['files']):
            t_w = 0.
            if ('Run201') in ss:
                t_w = 1.
            else:
                if not os.path.isfile(MAIN_S+ss+'.root'):
                    print("!!!File ", MAIN_S+ss+'.root', " does not exist! Continuing")
                    continue
                print "opening ", MAIN_S+ss+'.root'
                filename = TFile(MAIN_S+ss+'.root', "READ")
                if filename.GetListOfKeys().Contains("n_pass"):
                    b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                    n_pass      = filename.Get("n_pass").GetBinContent(1)
                    n_odd       = filename.Get("n_odd").GetBinContent(1)
                    filename.Close()
                    if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                        nevents = sample[ss]['nevents']
                        xs = 1.
                    else:
                        nevents = filename.Get("c_nEvents").GetBinContent(1)
                        xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                    t_w = LUMI * xs / nevents
                    if(b_skipTrain>0):
                        if(n_odd>0):
                            t_w *= float(n_pass/n_odd)

            tree_weight_dict[ss] = t_w


    #calculate first signal yield
    for d in sign:
        list_of_variables = ["isSR","Jets.eta","Jets.pt","Jets.phi","Jets.sigprob","Jets.nRecHitsEB","dt_ecal_dist","MinJetMetDPhi","min_dPhi_jets_eta_1p0_"+str(dnn_threshold).replace(".","p"),"EventWeight","PUReWeight","TriggerWeight"]
        print "\n"
        list_files = samples[d]["files"]

        pre = np.array([])
        bin0 = np.array([])
        bin1 = np.array([])
        bin2 = np.array([])
        
        for i, l in enumerate(list_files):

            print "Reading ", l
            gen = uproot.iterate(MAIN_S+l+".root","tree",list_of_variables,entrysteps=chunk_size)
            n_it = 0

            #print gen
            for arrays in gen:
                ################################################################
                print "Ev n. ", chunk_size*(1+n_it)

                #SR mask
                cut_mask = arrays["isSR"]>0

                #MinDPhi
                cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                #cosmic veto
                cosmic_mask = arrays["dt_ecal_dist"]>0.5
                cut_mask = np.logical_and(cut_mask,cosmic_mask)

                #cut jets kinematics
                mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                cut_jets = mask_eta

                if ERA=="2017":
                    MINPHI = 3.5
                    MAXPHI = 2.7
                    mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_jets = np.logical_and(mask_eta,mask_phi)

                if ERA=="2018":
                    MINPHI = 0.9
                    MAXPHI = 0.4
                    mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_jets = np.logical_and(mask_eta,mask_phi)
                    #cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))


                if ERA=="2016":
                    cut_jets = mask_eta

                cut_mask = np.logical_and(cut_mask,(cut_jets.any()==True))

                #beam halo
                mask_dphi = arrays["min_dPhi_jets_eta_1p0_"+str(dnn_threshold).replace(".","p")]<0.05
                mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
                mask_low_multi_tag = np.logical_and(mask_dphi,mask_low_multi_tag)
                bh_mask = np.logical_not(mask_low_multi_tag.any()==True)

                cut_mask = np.logical_and(cut_mask,bh_mask)
                cut_jets = np.logical_and(cut_jets,cut_mask)


                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                triggerweight = arrays["TriggerWeight"][cut_mask]
                weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[l]

                sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
                pt      = arrays["Jets.pt"][cut_jets][cut_mask]

                tag_mask = (sigprob > dnn_threshold)
                bin0_m = (sigprob[tag_mask].counts == 0)
                bin1_m = (sigprob[tag_mask].counts == 1)
                bin2_m = (sigprob[tag_mask].counts > 1)

                #Corrections for tagged jets
                sigprob_bin1 = sigprob[bin1_m]
                sigprob_bin2 = sigprob[bin2_m]

                pt_bin1 = pt[bin1_m]
                pt_bin2 = pt[bin2_m]

                pt_tag_bin1 = pt_bin1[sigprob_bin1>dnn_threshold]
                pt_tag_bin2 = pt_bin2[sigprob_bin2>dnn_threshold]

                pt_ele_mask_bin1 = (pt_tag_bin1 > 70)
                pt_pho_mask_bin1 = (pt_tag_bin1 <= 70)

                pt_ele_mask_bin2 = (pt_tag_bin2 > 70)
                pt_pho_mask_bin2 = (pt_tag_bin2 <= 70)

                dnnweight_bin1 = (sf_ele*pt_ele_mask_bin1+sf_pho*pt_pho_mask_bin1).prod()
                dnnweight_bin2 = (sf_ele*pt_ele_mask_bin2+sf_pho*pt_pho_mask_bin2).prod()

                #print dnnweight_bin1
                #print dnnweight_bin2
                pre = np.concatenate( (pre, np.multiply(cut_mask[cut_mask],weight)  ) )
                bin0 = np.concatenate( (bin0, np.multiply(bin0_m,weight)  ) ) 
                #bin1 = np.concatenate( (bin1, np.multiply(bin1_m,weight)  ) ) 
                #bin2 = np.concatenate( (bin2, np.multiply(bin2_m,weight)  ) ) 
                bin1 = np.concatenate( (bin1, np.multiply(bin1_m[bin1_m], np.multiply(weight[bin1_m],dnnweight_bin1)  )  ) ) 
                bin2 = np.concatenate( (bin2, np.multiply(bin2_m[bin2_m], np.multiply(weight[bin2_m],dnnweight_bin2)  )  ) ) 

                n_it+=1
                #if n_it>0:
                #    break

        ev_count[d]['pre'] = pre.sum()
        ev_count[d]['bin0'] = bin0.sum()
        ev_count[d]['bin1'] = bin1.sum()
        ev_count[d]['bin2'] = bin2.sum()

        print "events passing in signal ", d
        #print "presel : ", pre.sum()
        print "bin 0 : ", bin0.sum()
        print "bin 1 : ", bin1.sum()
        print "bin 2 : ", bin2.sum()


    for d in data:
        list_of_variables = ["isSR","Jets.eta","Jets.phi","Jets.sigprob","Jets.nRecHitsEB","dt_ecal*_dist","MinJetMetDPhi","min_dPhi_jets_eta_1p0_"+str(dnn_threshold).replace(".","p"),"RunNumber","LumiNumber","EventNumber"]#["*"]
        if ERA=="2018":
            list_of_variables += ["nCHSJets_in_HEM*"]
        print "\n"
        list_files = samples[d]["files"]

        pre = np.array([])
        bin0 = np.array([])
        bin1 = np.array([])
        bin2 = np.array([])
        RunNumber = np.array([])
        LumiNumber = np.array([])
        EventNumber = np.array([])
        RunNumber_bin1 = np.array([])
        LumiNumber_bin1 = np.array([])
        EventNumber_bin1 = np.array([])
        
        for i, l in enumerate(list_files):

            tmp_lumi = lumi["HighMET"][(l.split("-")[0]).split("HighMETRun"+ERA)[1]] 

            pre_tmp = np.array([])
            bin0_tmp = np.array([])
            bin1_tmp = np.array([])
            bin2_tmp = np.array([])

            print "Reading ", l
            gen = uproot.iterate(MAIN+l+".root","tree",list_of_variables,entrysteps=chunk_size)
            n_it = 0

            #print gen
            for arrays in gen:
                ################################################################
                print "Ev n. ", chunk_size*(1+n_it)

                #SR mask
                cut_mask = arrays["isSR"]>0

                #MinDPhi
                cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                #cosmic veto
                cosmic_mask = arrays["dt_ecal_dist"]>0.5
                cut_mask = np.logical_and(cut_mask,cosmic_mask)

                if dnn_threshold<0.996:
                    cosmic_mask = arrays["dt_ecal_no_tag_dist"]>0.5
                    cut_mask = np.logical_and(cut_mask,cosmic_mask)

                #cut jets kinematics
                mask_eta = np.logical_and(arrays["Jets.eta"]>-1. , arrays["Jets.eta"]<1.)
                cut_jets = mask_eta

                if ERA=="2017":
                    MINPHI = 3.5
                    MAXPHI = 2.7
                    mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_jets = np.logical_and(mask_eta,mask_phi)

                if ERA=="2018":
                    MINPHI = 0.9
                    MAXPHI = 0.4
                    mask_phi = np.logical_or(arrays["Jets.phi"]>MINPHI , arrays["Jets.phi"]<MAXPHI)
                    cut_jets = np.logical_and(mask_eta,mask_phi)
                    cut_mask = np.logical_and( cut_mask, np.logical_or( arrays["RunNumber"]<319077, np.logical_and( arrays["RunNumber"]>=319077, arrays["nCHSJets_in_HEM_pt_30_all_eta"]==0)))


                if ERA=="2016":
                    cut_jets = mask_eta

                cut_mask = np.logical_and(cut_mask,(cut_jets.any()==True))

                #beam halo
                mask_dphi = arrays["min_dPhi_jets_eta_1p0_"+str(dnn_threshold).replace(".","p")]<0.05
                mask_low_multi_tag = np.logical_and(arrays["Jets.sigprob"]>0.996 , arrays["Jets.nRecHitsEB"]<=10)
                mask_low_multi_tag = np.logical_and(mask_dphi,mask_low_multi_tag)
                bh_mask = np.logical_not(mask_low_multi_tag.any()==True)

                cut_mask = np.logical_and(cut_mask,bh_mask)
                cut_jets = np.logical_and(cut_jets,cut_mask)


                sigprob = arrays["Jets.sigprob"][cut_jets][cut_mask]
                tag_mask = (sigprob > dnn_threshold)
                bin0_m = (sigprob[tag_mask].counts == 0)
                bin1_m = (sigprob[tag_mask].counts == 1)
                bin2_m = (sigprob[tag_mask].counts > 1)
                
                pre = np.concatenate( (pre, np.multiply(cut_mask,1.)  ) )
                bin0 = np.concatenate( (bin0, np.multiply(bin0_m,1.)  ) ) 
                bin1 = np.concatenate( (bin1, np.multiply(bin1_m,1.)  ) ) 
                bin2 = np.concatenate( (bin2, np.multiply(bin2_m,1.)  ) ) 

                pre_tmp = np.concatenate( (pre_tmp, np.multiply(cut_mask,1.)  ) )
                bin0_tmp = np.concatenate( (bin0_tmp, np.multiply(bin0_m,1.)  ) ) 
                bin1_tmp = np.concatenate( (bin1_tmp, np.multiply(bin1_m,1.)  ) ) 
                bin2_tmp = np.concatenate( (bin2_tmp, np.multiply(bin2_m,1.)  ) ) 


                RunNumber = np.concatenate( (RunNumber, arrays["RunNumber"][cut_mask][bin2_m] ) )
                LumiNumber = np.concatenate( (LumiNumber, arrays["LumiNumber"][cut_mask][bin2_m] ) )
                EventNumber = np.concatenate( (EventNumber, arrays["EventNumber"][cut_mask][bin2_m] ) )

                RunNumber_bin1 = np.concatenate( (RunNumber_bin1, arrays["RunNumber"][cut_mask][bin1_m] ) )
                LumiNumber_bin1 = np.concatenate( (LumiNumber_bin1, arrays["LumiNumber"][cut_mask][bin1_m] ) )
                EventNumber_bin1 = np.concatenate( (EventNumber_bin1, arrays["EventNumber"][cut_mask][bin1_m] ) )

                n_it+=1
                #if n_it>0:
                #    break

            ev_count[l]['pre'] = pre_tmp.sum()
            ev_count[l]['bin0'] = bin0_tmp.sum()
            ev_count[l]['bin1'] = bin1_tmp.sum()
            ev_count[l]['bin2'] = bin2_tmp.sum()

            ev_count[l]['bin1_pred'] = y1_mistag*tmp_lumi/LUMI
            ev_count[l]['bin2_pred'] = (y2_mistag + y2_cosmic + y2_bh)*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_mistag'] = (y2_mistag)*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_cosmic'] = (y2_cosmic)*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_bh'] = (y2_bh)*tmp_lumi/LUMI

            ev_count[l]['bin1_pred_unc'] = y1_unc_mistag*y1_mistag*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_unc'] = math.sqrt(  (y2_mistag*y2_unc_mistag)**2 + (y2_cosmic*y2_unc_cosmic)**2 + (y2_bh*y2_unc_bh)**2  )*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_unc_mistag'] = y2_mistag*y2_unc_mistag*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_unc_cosmic'] = y2_cosmic*y2_unc_cosmic*tmp_lumi/LUMI
            ev_count[l]['bin2_pred_unc_bh'] = y2_bh*y2_unc_bh*tmp_lumi/LUMI

            #Here: loop over each run era
            '''
            print "Stop at Run A"
            print "bin 0 : ", bin0_tmp.sum()
            print "bin 1 : ", bin1_tmp.sum()
            print "bin 2 : ", bin2_tmp.sum()

            print "Events in bin 2"
            for a in range(RunNumber.shape[0]):
                np.set_printoptions(suppress=True)
                print int(RunNumber[a]),":",int(LumiNumber[a]),":",int(EventNumber[a])

            print "Events in bin 1"
            for a in range(RunNumber_bin1.shape[0]):
                np.set_printoptions(suppress=True)
                print int(RunNumber_bin1[a]),":",int(LumiNumber_bin1[a]),":",int(EventNumber_bin1[a])
            break
            '''
        #exit()

        ev_count[d]['pre'] = pre.sum()
        ev_count[d]['bin0'] = bin0.sum()
        ev_count[d]['bin1'] = bin1.sum()
        ev_count[d]['bin2'] = bin2.sum()
        ev_count[d]['bin1_pred'] = y1_mistag
        ev_count[d]['bin2_pred'] = y2_mistag + y2_cosmic + y2_bh

        ev_count[d]['bin2_pred_mistag'] = (y2_mistag)
        ev_count[d]['bin2_pred_cosmic'] = (y2_cosmic)
        ev_count[d]['bin2_pred_bh'] = (y2_bh)

        ev_count[d]['bin1_pred_unc'] = y1_unc_mistag*y1_mistag
        ev_count[d]['bin2_pred_unc'] = math.sqrt(  (y2_mistag*y2_unc_mistag)**2 + (y2_cosmic*y2_unc_cosmic)**2 + (y2_bh*y2_unc_bh)**2  )

        ev_count[d]['bin2_pred_unc_mistag'] = y2_mistag*y2_unc_mistag
        ev_count[d]['bin2_pred_unc_cosmic'] = y2_cosmic*y2_unc_cosmic
        ev_count[d]['bin2_pred_unc_bh'] = y2_bh*y2_unc_bh


        print "events passing in ", d
        #print "presel : ", pre.sum()
        print "bin 0 : ", bin0.sum()
        print "bin 1 : ", bin1.sum()
        print "bin 2 : ", bin2.sum()

    print ev_count
    dump_name = OUT+"Dict_unblinding"+label
    if TAG!="":
        dump_name+="_"+TAG[0]+"-"+TAG[1]
    dump_name+=".yaml"

    with open(dump_name,"w") as f:
        yaml.dump(ev_count, f)
        f.close()
        print "Info: dictionary written in file "+dump_name

def plot_data(label=""):

    dump_name = OUT+"Dict_unblinding"+label
    if TAG!="":
        dump_name+="_"+TAG[0]+"-"+TAG[1]
    dump_name+=".yaml"

    with open(dump_name,"r") as f:
        ev_count = yaml.load(f, Loader=yaml.Loader)
        f.close()


    min_val = 0.001
    #can also read signal from yaml datacards and scale it accordingly!

    for k in ev_count.keys():
        if "HighMET" not in k:
            continue

        if "Run"+ERA in k:
            print (k.split("-")[0]).split("HighMETRun"+ERA)[1]
            tmp_lumi = lumi["HighMET"][(k.split("-")[0]).split("HighMETRun"+ERA)[1]] 
        else:
            tmp_lumi = LUMI

        #print ev_count[k]

        #nice table

        print k.split("-")[0]
        print "\n"
        print "Year & bin 1 yield & bin 1 prediction & bin 2 yield & bin 2 prediction "+"\\"+"\\"
        print "\hline"
        print k.split("-")[0], " & ", int(ev_count[k]['bin1']), " & %.2f" % round(ev_count[k]['bin1_pred'],2), " $\pm$ %.2f" % round(ev_count[k]['bin1_pred_unc'],2), ("(%.1f" % round(100*ev_count[k]['bin1_pred_unc']/ev_count[k]['bin1_pred'],1)) +"\%)" , " & ", int(ev_count[k]['bin2']), " & %.3f" % round(ev_count[k]['bin2_pred'],3), " $\pm$ %.3f" % round(ev_count[k]['bin2_pred_unc'],3), ("(%.1f" % round(100*ev_count[k]['bin2_pred_unc']/ev_count[k]['bin2_pred'],1)) +"\%)","\\"+"\\"
        print "\hline"


        bins = np.array([0.,1.,2.])
        
        fake = TH1F("","",3,-0.5,2.5)
        yp = np.array([0., ev_count[k]['bin1_pred'], ev_count[k]['bin2_pred']])
        ep = np.array([0., ev_count[k]['bin1_pred_unc'], ev_count[k]['bin2_pred_unc']])

        yo  = np.array([ev_count[k]['bin0'], ev_count[k]['bin1'], ev_count[k]['bin2'] if ev_count[k]['bin2']>0 else min_val])
        eo = np.sqrt(yo)
        eo[2] = math.sqrt(ev_count[k]['bin2']) if ev_count[k]['bin2']>0 else 1.8

        ys = np.array([ ev_count['SUSY_mh400_ctau500_HH']['bin0']*tmp_lumi/LUMI, ev_count['SUSY_mh400_ctau500_HH']['bin1']*tmp_lumi/LUMI, ev_count['SUSY_mh400_ctau500_HH']['bin2']*tmp_lumi/LUMI ])*0.01
        es = np.sqrt(ys)

        ys3m = np.array([ev_count['SUSY_mh400_ctau3000_HH']['bin0']*tmp_lumi/LUMI, ev_count['SUSY_mh400_ctau3000_HH']['bin1']*tmp_lumi/LUMI, ev_count['SUSY_mh400_ctau3000_HH']['bin2']*tmp_lumi/LUMI])*0.01
        es3m = np.sqrt(ys3m)

        #ys3m = np.array([7.87377,6.35219,1.19317])
        #es3m = np.array([0.32297,0.28982,0.12476])

        gp = TGraphAsymmErrors()
        go = TGraphAsymmErrors()
        gs = TH1F("","",3,-0.5,2.5)
        gs.Sumw2()
        gs3m = TH1F("","",3,-0.5,2.5)
        gs3m.Sumw2()
        for i in range(yp.shape[0]):
            gp.SetPointX(i,bins[i])
            gp.SetPointY(i,yp[i])
            gp.SetPointEXhigh(i, 0.5)
            gp.SetPointEXlow(i, 0.5)
            gp.SetPointEYhigh(i, 0.5*ep[i])
            gp.SetPointEYlow(i, 0.5*ep[i])

            go.SetPointX(i,bins[i])
            go.SetPointY(i,yo[i])
            go.SetPointEXhigh(i, 0.5)
            go.SetPointEXlow(i, 0.5)
            go.SetPointEYhigh(i, 0.5*eo[i])
            go.SetPointEYlow(i, 0.5*eo[i])

            gs.Fill(bins[i],ys[i])
            gs3m.Fill(bins[i],ys3m[i])

            can = TCanvas("can","can",900,800)
            can.cd()
            #can.SetGrid()
            can.SetLogy()
            can.SetRightMargin(0.05)
            can.SetLeftMargin(0.13)
            can.SetBottomMargin(0.12)
            leg = TLegend(0.45, 0.73, 0.9, 0.88)
            leg.SetTextSize(0.035)
            leg.SetBorderSize(0)

            gs.SetLineColor(2)
            gs3m.SetLineColor(416+2)
            gs.SetLineStyle(1)
            gs3m.SetLineStyle(2)
            gs.SetLineWidth(3)
            gs3m.SetLineWidth(3)

            gp.SetFillColorAlpha(861,0.3)
            gp.SetLineColor(861)
            gp.SetLineWidth(0)
            gp.SetMarkerStyle(21)
            gp.SetMarkerColor(861)

            go.SetLineColor(1)
            go.SetLineWidth(2)
            go.SetMarkerColor(1)
            go.SetMarkerStyle(20)
            go.SetMarkerSize(1.3)

            leg.AddEntry(gp,"prediction","PF")
            leg.AddEntry(go,"data obs.","PL")
            leg.AddEntry(gs,"m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m","L")
            leg.AddEntry(gs3m,"m_{#chi} = 400 GeV, c#tau_{0} = 3 m","L")

            fake.Draw("")
            gs.Draw("HISTO,sames")
            gs3m.Draw("HISTO,sames")
            gp.Draw("PE2,sames")
            go.Draw("PE,sames")
            fake.GetXaxis().SetBinLabel(1,"0 tags")
            fake.GetXaxis().SetBinLabel(2,"1 tag")
            fake.GetXaxis().SetBinLabel(3,"#geq 2 tags")
            #fake.GetYaxis().SetRangeUser(min_val,5.e7)
            fake.SetMinimum(min_val)
            fake.SetMaximum(5.e7)
            fake.GetXaxis().SetLimits(-0.5,2.5)
            fake.GetYaxis().SetTitle("Number of events")
            fake.GetYaxis().SetTitleSize(0.04)
            fake.GetXaxis().SetTitleOffset(1.2)
            fake.GetXaxis().SetLabelSize(0.06)
            fake.GetXaxis().SetTitle("Number of TDJ tagged jets")
            fake.GetXaxis().SetTitleSize(0.04)
            #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
            if PRELIMINARY:
                drawCMS_simple(tmp_lumi, "Preliminary", ERA="", onTop=True)
            else:
                drawCMS_simple(tmp_lumi, "", ERA="", onTop=True)
            leg.Draw()
            can.Print(OUT+"nTDJ_"+k+".pdf")
            can.Print(OUT+"nTDJ_"+k+".png")
            can.Close()


def plot_data_all(label=""):

    ev_count = {}
    for e in ["2017","2018"]:
        dump_name = "plots/v6_calo_AOD_"+e+"_unblinding_ARC/"
        dump_name+= "Dict_unblinding"+label
        dump_name+=".yaml"
        with open(dump_name,"r") as f:
            ev_count[e] = yaml.load(f, Loader=yaml.Loader)
            f.close()
    
    for e in ["2016"]:
        dump_name = "plots/v6_calo_AOD_"+e+"_unblinding_ARC/"
        dump_name+= "Dict_unblinding"+label

        dump_nameBF=dump_name+"_B-F"
        dump_nameBF+=".yaml"
        with open(dump_nameBF,"r") as f:
            ev_count[e+"BF"] = yaml.load(f, Loader=yaml.Loader)
            f.close()

        dump_nameGH=dump_name+"_G-H"
        dump_nameGH+=".yaml"
        with open(dump_nameGH,"r") as f:
            ev_count[e+"GH"] = yaml.load(f, Loader=yaml.Loader)
            f.close()

    with open(UNCDIR_ALL+"signal_bkg_unc"+".yaml","r") as f:
        uncertainties = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #print uncertainties
        #print "  --  --  --  --  --"

    tot_data = np.zeros(3)
    unc_data = np.zeros(3)

    tot_pred = np.zeros(3)
    unc_pred = np.zeros(3)

    #unc_pred must be split among various components
    tot_pred_bh = np.zeros(3)
    unc_pred_bh = np.zeros(3)
    tot_pred_cosmic = np.zeros(3)
    unc_pred_cosmic = np.zeros(3)
    tot_pred_mistag = np.zeros(3)
    unc_pred_mistag = np.zeros(3)

    tot_sign = {}
    unc_sign = {}
    
    for s in sign:
        tot_sign[s] = np.zeros(3)
        unc_sign[s] = np.zeros(3)

    for e in ev_count.keys():
        for k in ev_count[e].keys():
            if k=="HighMET" or k=="HighMETBF" or k=="HighMETGH":
                tot_data[0] += ev_count[e][k]['bin0']
                tot_data[1] += ev_count[e][k]['bin1']
                tot_data[2] += ev_count[e][k]['bin2']

                #unc squared
                unc_data[0] += ev_count[e][k]['bin0']
                unc_data[1] += ev_count[e][k]['bin1']
                unc_data[2] += ev_count[e][k]['bin2']

                tot_pred[0] += 0
                tot_pred[1] += ev_count[e][k]['bin1_pred']
                tot_pred[2] += ev_count[e][k]['bin2_pred']
                
                #bin 1 pred: only mistag
                tot_pred_mistag[1] += ev_count[e][k]['bin1_pred']#*uncertainties[xxxhere mistag uncxxx]

                tot_pred_mistag[2] += ev_count[e][k]['bin2_pred_mistag']#*uncertainties[xxxhere mistag uncxxx]
                tot_pred_cosmic[2] += ev_count[e][k]['bin2_pred_cosmic']
                tot_pred_bh[2] += ev_count[e][k]['bin2_pred_bh']

                #unc squared
                unc_pred_cosmic[2] += ev_count[e][k]['bin2_pred_unc_cosmic']**2
                unc_pred_bh[2] += ev_count[e][k]['bin2_pred_unc_bh']**2

            if k in sign:
                tot_sign[k][0] += ev_count[e][k]['bin0']
                tot_sign[k][1] += ev_count[e][k]['bin1']
                tot_sign[k][2] += ev_count[e][k]['bin2']
                #unc squared
                unc_sign[k][0] += ev_count[e][k]['bin0']
                unc_sign[k][1] += ev_count[e][k]['bin1']
                unc_sign[k][2] += ev_count[e][k]['bin2']

    for s in sign:
        unc_sign[s] = np.sqrt(unc_sign[s])
    unc_data = np.sqrt(unc_data)
    unc_pred_cosmic = np.sqrt(unc_pred_cosmic)
    unc_pred_bh = np.sqrt(unc_pred_bh)

    unc_pred_mistag[1] = tot_pred[1]*uncertainties['bkg_tot_bin1']/100.
    unc_pred_mistag[2] = tot_pred_mistag[2]*uncertainties['bkg_tot']/100.

    unc_pred[1] = tot_pred[1]*uncertainties['bkg_tot_bin1']/100.
    unc_pred[2] = math.sqrt(unc_pred_mistag[2]**2 + unc_pred_cosmic[2]**2 + unc_pred_bh[2]**2)

    min_val = 0.001
    #can also read signal from yaml datacards and scale it accordingly!


    #no need to loop anymore
    print "\n"
    print "Year & bin 1 yield & bin 1 prediction & bin 2 yield & bin 2 prediction "+"\\"+"\\"
    print "\hline"
    print "Tot. & ", int(tot_data[1]), " & %.2f" % round(tot_pred[1],2), " $\pm$ %.2f" % round(unc_pred[1],2), " (%.1f" % round(100*unc_pred[1]/tot_pred[1]) +"\%)" , " & ", int(tot_data[2]), " & %.3f" % round(tot_pred[2],3), " $\pm$ %.3f" % round(unc_pred[2],3), ("(%.1f" % round(100*unc_pred[2]/tot_pred[2],1)) +"\%)","\\"+"\\"
    print "\hline"

    bins = np.array([0.,1.,2.])
        
    fake = TH1F("","",3,-0.5,2.5)
    yp = np.array([0., tot_pred[1], tot_pred[2]])
    ep = np.array([0., unc_pred[1], unc_pred[2]])

    yo = np.array([tot_data[0], tot_data[1], tot_data[2] if tot_data[2]>0 else min_val])
    eo = np.array([unc_data[0], unc_data[1], unc_data[2] if tot_data[2]>0 else 1.8])

    ys = np.array([ tot_sign['SUSY_mh400_ctau500_HH'][0], tot_sign['SUSY_mh400_ctau500_HH'][1], tot_sign['SUSY_mh400_ctau500_HH'][2] ])*0.01
    es = np.sqrt(ys)

    ys3m = np.array([ tot_sign['SUSY_mh400_ctau3000_HH'][0], tot_sign['SUSY_mh400_ctau3000_HH'][1], tot_sign['SUSY_mh400_ctau3000_HH'][2] ])*0.01
    es3m = np.sqrt(ys)


    #ys3m = np.array([7.87377,6.35219,1.19317])
    #es3m = np.array([0.32297,0.28982,0.12476])

    gp = TGraphAsymmErrors()
    go = TGraphAsymmErrors()
    gs = TH1F("","",3,-0.5,2.5)
    gs.Sumw2()
    gs3m = TH1F("","",3,-0.5,2.5)
    gs3m.Sumw2()
    for i in range(yp.shape[0]):
        gp.SetPointX(i,bins[i])
        gp.SetPointY(i,yp[i])
        gp.SetPointEXhigh(i, 0.5)
        gp.SetPointEXlow(i, 0.5)
        gp.SetPointEYhigh(i, 0.5*ep[i])
        gp.SetPointEYlow(i, 0.5*ep[i])

        go.SetPointX(i,bins[i])
        go.SetPointY(i,yo[i])
        go.SetPointEXhigh(i, 0.5)
        go.SetPointEXlow(i, 0.5)
        go.SetPointEYhigh(i, 0.5*eo[i])
        go.SetPointEYlow(i, 0.5*eo[i])

        gs.Fill(bins[i],ys[i])
        gs3m.Fill(bins[i],ys3m[i])

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.12)
    leg = TLegend(0.45, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)

    gs.SetLineColor(2)
    gs3m.SetLineColor(416+2)
    gs.SetLineStyle(1)
    gs3m.SetLineStyle(2)
    gs.SetLineWidth(3)
    gs3m.SetLineWidth(3)
    
    gp.SetFillColorAlpha(861,0.3)
    gp.SetLineColor(861)
    gp.SetLineWidth(0)
    gp.SetMarkerStyle(21)
    gp.SetMarkerColor(861)

    go.SetLineColor(1)
    go.SetLineWidth(2)
    go.SetMarkerColor(1)
    go.SetMarkerStyle(20)
    go.SetMarkerSize(1.3)

    leg.AddEntry(gp,"prediction","PF")
    leg.AddEntry(go,"data obs.","PL")
    leg.AddEntry(gs,"m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m","L")
    leg.AddEntry(gs3m,"m_{#chi} = 400 GeV, c#tau_{0} = 3 m","L")
    
    fake.Draw("")
    gs.Draw("HISTO,sames")
    gs3m.Draw("HISTO,sames")
    gp.Draw("PE2,sames")
    go.Draw("PE,sames")
    fake.GetXaxis().SetBinLabel(1,"0 tags")
    fake.GetXaxis().SetBinLabel(2,"1 tag")
    fake.GetXaxis().SetBinLabel(3,"#geq 2 tags")
    #fake.GetYaxis().SetRangeUser(min_val,5.e7)
    fake.SetMinimum(min_val)
    fake.SetMaximum(5.e7)
    fake.GetXaxis().SetLimits(-0.5,2.5)
    fake.GetYaxis().SetTitle("Number of events")
    fake.GetYaxis().SetTitleSize(0.04)
    fake.GetXaxis().SetTitleOffset(1.2)
    fake.GetXaxis().SetLabelSize(0.06)
    fake.GetXaxis().SetTitle("Number of TDJ tagged jets")
    fake.GetXaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    leg.Draw()
    can.Print(OUT+"nTDJ.pdf")
    can.Print(OUT+"nTDJ.png")

    can.Print(OUT_paper+"nTDJ.pdf")
    can.Print(OUT_paper+"nTDJ.png")

    can.Close()



def plot_data_all_ratio_panel(label=""):

    ev_count = {}
    for e in ["2017","2018"]:
        dump_name = "plots/v6_calo_AOD_"+e+"_unblinding_ARC/"
        dump_name+= "Dict_unblinding"+label
        dump_name+=".yaml"
        with open(dump_name,"r") as f:
            ev_count[e] = yaml.load(f, Loader=yaml.Loader)
            f.close()
    
    for e in ["2016"]:
        dump_name = "plots/v6_calo_AOD_"+e+"_unblinding_ARC/"
        dump_name+= "Dict_unblinding"+label

        dump_nameBF=dump_name+"_B-F"
        dump_nameBF+=".yaml"
        with open(dump_nameBF,"r") as f:
            ev_count[e+"BF"] = yaml.load(f, Loader=yaml.Loader)
            f.close()

        dump_nameGH=dump_name+"_G-H"
        dump_nameGH+=".yaml"
        with open(dump_nameGH,"r") as f:
            ev_count[e+"GH"] = yaml.load(f, Loader=yaml.Loader)
            f.close()

    with open(UNCDIR_ALL+"signal_bkg_unc"+".yaml","r") as f:
        uncertainties = yaml.load(f, Loader=yaml.Loader)
        f.close()
        #print uncertainties
        #print "  --  --  --  --  --"

    tot_data = np.zeros(3)
    unc_data = np.zeros(3)

    tot_pred = np.zeros(3)
    unc_pred = np.zeros(3)

    #unc_pred must be split among various components
    tot_pred_bh = np.zeros(3)
    unc_pred_bh = np.zeros(3)
    tot_pred_cosmic = np.zeros(3)
    unc_pred_cosmic = np.zeros(3)
    tot_pred_mistag = np.zeros(3)
    unc_pred_mistag = np.zeros(3)

    tot_sign = {}
    unc_sign = {}
    
    for s in sign:
        tot_sign[s] = np.zeros(3)
        unc_sign[s] = np.zeros(3)

    for e in ev_count.keys():
        for k in ev_count[e].keys():
            if k=="HighMET" or k=="HighMETBF" or k=="HighMETGH":
                tot_data[0] += ev_count[e][k]['bin0']
                tot_data[1] += ev_count[e][k]['bin1']
                tot_data[2] += ev_count[e][k]['bin2']

                #unc squared
                unc_data[0] += ev_count[e][k]['bin0']
                unc_data[1] += ev_count[e][k]['bin1']
                unc_data[2] += ev_count[e][k]['bin2']

                tot_pred[0] += 0
                tot_pred[1] += ev_count[e][k]['bin1_pred']
                tot_pred[2] += ev_count[e][k]['bin2_pred']
                
                #bin 1 pred: only mistag
                tot_pred_mistag[1] += ev_count[e][k]['bin1_pred']#*uncertainties[xxxhere mistag uncxxx]

                tot_pred_mistag[2] += ev_count[e][k]['bin2_pred_mistag']#*uncertainties[xxxhere mistag uncxxx]
                tot_pred_cosmic[2] += ev_count[e][k]['bin2_pred_cosmic']
                tot_pred_bh[2] += ev_count[e][k]['bin2_pred_bh']

                #unc squared
                unc_pred_cosmic[2] += ev_count[e][k]['bin2_pred_unc_cosmic']**2
                unc_pred_bh[2] += ev_count[e][k]['bin2_pred_unc_bh']**2

            if k in sign:
                tot_sign[k][0] += ev_count[e][k]['bin0']
                tot_sign[k][1] += ev_count[e][k]['bin1']
                tot_sign[k][2] += ev_count[e][k]['bin2']
                #unc squared
                unc_sign[k][0] += ev_count[e][k]['bin0']
                unc_sign[k][1] += ev_count[e][k]['bin1']
                unc_sign[k][2] += ev_count[e][k]['bin2']

    for s in sign:
        unc_sign[s] = np.sqrt(unc_sign[s])
    unc_data = np.sqrt(unc_data)
    unc_pred_cosmic = np.sqrt(unc_pred_cosmic)
    unc_pred_bh = np.sqrt(unc_pred_bh)

    unc_pred_mistag[1] = tot_pred[1]*uncertainties['bkg_tot_bin1']/100.
    unc_pred_mistag[2] = tot_pred_mistag[2]*uncertainties['bkg_tot']/100.

    unc_pred[1] = tot_pred[1]*uncertainties['bkg_tot_bin1']/100.
    unc_pred[2] = math.sqrt(unc_pred_mistag[2]**2 + unc_pred_cosmic[2]**2 + unc_pred_bh[2]**2)

    min_val = 0.001
    #can also read signal from yaml datacards and scale it accordingly!


    #no need to loop anymore
    print "\n"
    print "Year & bin 1 yield & bin 1 prediction & bin 2 yield & bin 2 prediction "+"\\"+"\\"
    print "\hline"
    print "Tot. & ", int(tot_data[1]), " & %.2f" % round(tot_pred[1],2), " $\pm$ %.2f" % round(unc_pred[1],2), " (%.1f" % round(100*unc_pred[1]/tot_pred[1]) +"\%)" , " & ", int(tot_data[2]), " & %.3f" % round(tot_pred[2],3), " $\pm$ %.3f" % round(unc_pred[2],3), ("(%.1f" % round(100*unc_pred[2]/tot_pred[2],1)) +"\%)","\\"+"\\"
    print "\hline"

    bins = np.array([0.,1.,2.])
        
    fake = TH1F("","",3,-0.5,2.5)
    fake_num = TH1F("","",3,-0.5,2.5)
    fake_den = TH1F("","",3,-0.5,2.5)
    fake_num.Sumw2()
    fake_den.Sumw2()

    yp = np.array([0., tot_pred[1], tot_pred[2]])
    ep = np.array([0., unc_pred[1], unc_pred[2]])

    yo = np.array([tot_data[0], tot_data[1], tot_data[2] if tot_data[2]>0 else min_val])
    eo = np.array([unc_data[0], unc_data[1], unc_data[2] if tot_data[2]>0 else 1.8])

    ys = np.array([ tot_sign['SUSY_mh400_ctau500_HH'][0], tot_sign['SUSY_mh400_ctau500_HH'][1], tot_sign['SUSY_mh400_ctau500_HH'][2] ])*0.01
    es = np.sqrt(ys)

    ys3m = np.array([ tot_sign['SUSY_mh400_ctau3000_HH'][0], tot_sign['SUSY_mh400_ctau3000_HH'][1], tot_sign['SUSY_mh400_ctau3000_HH'][2] ])*0.01
    es3m = np.sqrt(ys)


    print "Observed"
    print yo
    print eo
    print "Predicted"
    print yp
    print ep
    print "Signal 0.5 m"
    print ys
    print es
    print "Signal 3 m"
    print ys3m
    print es3m

    #ys3m = np.array([7.87377,6.35219,1.19317])
    #es3m = np.array([0.32297,0.28982,0.12476])

    gp = TGraphAsymmErrors()
    go = TGraphAsymmErrors()
    gs = TH1F("","",3,-0.5,2.5)
    gs.Sumw2()
    gs3m = TH1F("","",3,-0.5,2.5)
    gs3m.Sumw2()
    for i in range(yp.shape[0]):
        gp.SetPointX(i,bins[i])
        gp.SetPointY(i,yp[i])
        gp.SetPointEXhigh(i, 0.5)
        gp.SetPointEXlow(i, 0.5)
        gp.SetPointEYhigh(i, 0.5*ep[i])
        gp.SetPointEYlow(i, 0.5*ep[i])

        go.SetPointX(i,bins[i])
        go.SetPointY(i,yo[i])
        go.SetPointEXhigh(i, 0.5)
        go.SetPointEXlow(i, 0.5)
        go.SetPointEYhigh(i, 0.5*eo[i])
        go.SetPointEYlow(i, 0.5*eo[i])

        gs.Fill(bins[i],ys[i])
        gs3m.Fill(bins[i],ys3m[i])

        fake_num.Fill(bins[i],yo[i])
        fake_den.Fill(bins[i],yp[i])

    for i in range(yp.shape[0]):
        fake_num.SetBinError(i+1,eo[i])
        fake_den.SetBinError(i+1,ep[i])

    #Just a check
    '''
    print "Print the things"
    print "graph go:"
    print go.Print()
    print "hist fake_num:"
    for i in range(yp.shape[0] + 1):
        print fake_num.GetBinContent(i), fake_num.GetBinError(i)


    print "graph gp:"
    print gp.Print()
    print "hist fake_den:"
    for i in range(yp.shape[0] + 1):
        print fake_den.GetBinContent(i), fake_den.GetBinError(i)
    '''

    can = TCanvas("can","can",900,800)

    can.Divide(1,2)
    r = 4
    can.GetPad(1).SetPad("TopPad", "", 0., 1./r, 1.0, 1.0, 0, -1, 0)
    can.GetPad(1).SetRightMargin(0.05)
    can.GetPad(1).SetLeftMargin(0.13)
    can.GetPad(1).SetBottomMargin(0.02)#This was 0.05, can be changed
    can.GetPad(1).SetTopMargin(0.08)

    can.GetPad(2).SetPad("BotPad", "", 0., 0., 1.0, 1./r, 0, -1, 0)
    can.GetPad(2).SetRightMargin(0.05)
    can.GetPad(2).SetLeftMargin(0.13)
    can.GetPad(2).SetBottomMargin(0.4)
    can.GetPad(2).SetTopMargin(1)#an integer here forces the pad on top, but it shrinks it

    can.cd(1)
    #can.SetGrid()
    can.GetPad(1).SetLogy()
    #can.SetBottomMargin(0.12)
    leg = TLegend(0.45, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)

    gs.SetLineColor(2)
    gs3m.SetLineColor(416+2)
    gs.SetLineStyle(1)
    gs3m.SetLineStyle(2)
    gs.SetLineWidth(3)
    gs3m.SetLineWidth(3)
    
    gp.SetFillColorAlpha(861,0.3)
    gp.SetLineColor(861)
    gp.SetLineWidth(0)
    gp.SetMarkerStyle(21)
    gp.SetMarkerColor(861)

    go.SetLineColor(1)
    go.SetLineWidth(2)
    go.SetMarkerColor(1)
    go.SetMarkerStyle(20)
    go.SetMarkerSize(1.3)

    leg.AddEntry(gp,"prediction","PF")
    leg.AddEntry(go,"data obs.","PL")
    leg.AddEntry(gs,"m_{#chi} = 400 GeV, c#tau_{0} = 0.5 m","L")
    leg.AddEntry(gs3m,"m_{#chi} = 400 GeV, c#tau_{0} = 3 m","L")
    
    fake.Draw("")
    gs.Draw("HISTO,sames")
    gs3m.Draw("HISTO,sames")
    gp.Draw("PE2,sames")
    go.Draw("PE,sames")


    '''
    #Just a technical check
    fake_num.SetLineColor(801)
    fake_num.SetLineWidth(4)
    fake_num.SetLineStyle(1)
    fake_num.SetMarkerColor(801)
    fake_num.SetMarkerStyle(21)
    fake_num.Draw("PE,sames")
    '''
    #This will be a pain to move...

    fake.GetXaxis().SetBinLabel(1,"0 tags")
    fake.GetXaxis().SetBinLabel(2,"1 tag")
    fake.GetXaxis().SetBinLabel(3,"#geq 2 tags")
    #fake.GetYaxis().SetRangeUser(min_val,5.e7)
    fake.SetMinimum(min_val)
    fake.SetMaximum(5.e7)

    fake.GetXaxis().SetLimits(-0.5,2.5)
    fake.GetYaxis().SetTitle("Number of events")
    fake.GetYaxis().SetTitleSize(0.04)
    fake.GetXaxis().SetTitleOffset(1.2)
    #This is to avoid seeing the edges of the chars in the label/title
    fake.GetXaxis().SetTitleOffset(1.5)
    fake.GetXaxis().SetLabelOffset(1.2)


    fake.GetXaxis().SetLabelSize(0.06)
    fake.GetXaxis().SetTitle("Number of TDJ tagged jets")
    fake.GetXaxis().SetTitleSize(0.04)
    #####drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True,left_marg_CMS=0.21)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.21)
    leg.Draw()

    can.Update()

    #Ratio plot from histogram, but we have TGraph
    
    can.cd(2)
    fake_ratio = fake.Clone("fake_ratio")
    ratio = fake_num.Clone("ratio")
    ratio.Divide(fake_den)

    ratio.GetYaxis().SetTitle("Data / prediction")
    fake_ratio.GetYaxis().SetTitle("Data / prediction")

    ratio.SetLineColor(1)
    ratio.SetMarkerColor(1)
    ratio.SetMarkerStyle(20)
    ratio.SetLineWidth(2)
    ratio.SetMarkerSize(1.3)

    fake_ratio.GetYaxis().SetTitleOffset(+0.35)
    fake_ratio.GetXaxis().SetTitleOffset(+1.35)
    fake_ratio.GetYaxis().SetTitleSize(0.12)
    fake_ratio.GetYaxis().SetLabelSize(0.12)

    fake_ratio.GetXaxis().SetTitleSize(0.13)
    fake_ratio.GetXaxis().SetLabelSize(0.2)
    fake_ratio.GetXaxis().SetLabelOffset(0.04)

    fake_ratio.SetMinimum(0)
    fake_ratio.SetMaximum(4)
    fake_ratio.SetLineColor(1)
    fake_ratio.SetMarkerColor(1)
    fake_ratio.SetMarkerStyle(20)
    fake_ratio.SetLineWidth(2)
    fake_ratio.SetMarkerSize(1.3)

    fake_ratio.GetYaxis().SetNdivisions(4)
    fake_ratio.Draw("")
    ratio.Draw("PE,sames")




    can.Print(OUT+"nTDJ_ratio_panel.pdf")
    can.Print(OUT+"nTDJ_ratio_panel.png")

    can.Print(OUT_paper+"nTDJ_ratio_panel.pdf")
    can.Print(OUT_paper+"nTDJ_ratio_panel.png")

    can.Close()

label = ""
#label = "_wp_0p9"
#count_data(label)
#plot_data(label)
plot_data_all(label)
plot_data_all_ratio_panel(label)
