#! /usr/bin/env python
import os, multiprocessing
import copy
import math
import yaml
import time
import uproot
import root_numpy
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
##import mplhep as hep
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)
#plt.style.use(hep.style.CMS)

PRELIMINARY = False
out_fold = "fig/Preliminary/" if PRELIMINARY else "fig/"

def tau_weight_calc(llp_ct, new_ctau, old_ctau):
    source = np.exp(-1.0*llp_ct/old_ctau)/old_ctau**2
    weight = 1.0/new_ctau**2 * np.exp(-1.0*llp_ct/new_ctau)/source
    return weight

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def plot_ROC():
    '''ROC'''
    inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    #"/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v5_calo_AOD_2018_dnn_x-check/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    store = pd.HDFStore(inp_file)
    df_test = store.select("df")
    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    fpr, tpr, thresholds = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])
    cut_fpr = 0.00040904540701505433
    idx, _ = find_nearest(fpr,cut_fpr)
    g = TGraph(len(fpr)-1,fpr,tpr)
    p = TGraph(1,np.array([fpr[idx]]),np.array([tpr[idx]]))
    print "Filled TGraph..."
    del df_test
    del store
    del fpr
    del tpr
    del thresholds
    del idx
    print "Delete transient things"
    #bin_x = array('d',np.logspace(-4,-2,20000).tolist())
    #bin_x = array('d',np.linspace(0.0001,0.01,40000).tolist())
    #bin_y = array('d',np.linspace(0.6,1.,40000).tolist())
    #h = TH2F("h","",len(bin_x)-1,bin_x,len(bin_y)-1,bin_y)
    #root_numpy.fill_hist(h,np.transpose(np.stack((fpr,tpr))))
    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    g.SetTitle("")
    g.GetXaxis().SetTitle("False positive rate")
    g.GetYaxis().SetTitle("True positive rate")
    g.GetXaxis().SetTitleSize(0.04)
    g.GetYaxis().SetTitleSize(0.04)
    g.SetMarkerColor(2)
    g.SetLineColor(2)
    g.SetLineWidth(4)
    print "Plotting TGraph..."
    g.Draw("ACPL")
    p.SetMarkerStyle(20)
    p.SetMarkerSize(1.5)
    p.SetMarkerColor(4)
    p.Draw("P,sames")
    g.SetMaximum(1.01)
    g.SetMinimum(0.5)
    g.GetXaxis().SetRangeUser(0.00001,0.01)
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True)
    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)
    leg.AddEntry(g,"ROC curve","L")
    leg.AddEntry(p,"DNN>0.996 w. p.","P")
    leg.SetBorderSize(0)
    leg.Draw()
    print "Printing canvas... "
    can.Print(out_fold+"TDJ_ROC.pdf")
    can.Print(out_fold+"TDJ_ROC.png")
    del g
    del p


def tag_eff_vs_ct(ERA):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        #'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        #'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
    ]

    CUT = "isSR"
    chain = {}
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [1,418,2,856,801]

    ctaus_500          = np.array([10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
    ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 25000, 50000, 100000])
    ctaus = np.unique(np.concatenate((ctaus_500,ctaus_3000)))
    #ctaus = np.array([500,1000,2000,5000])
    #ctaus = np.array([5000])

    h_num = {}
    h_den = {}
    eff = {}

    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetGrid()
    can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    #Perform the ctau reweighting
    for i, s in enumerate(sign):
        truth_s = np.array([])
        prob_s = np.array([])
        w_s = np.array([])
        ctau_event_weight = {}
        travel_dist_reweighted = {}

        den_val = np.array([])
        num_val = np.array([])
        den_w = np.array([])
        num_w = np.array([])


        for ct in ctaus:
            ctau_event_weight[ct] = np.array([])
            travel_dist_reweighted[ct] = np.array([])
            
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Gen*",CUT]

        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]["files"]):
            print l, ss
            chain[s].Add(NTUPLEDIR + ss + ".root")

            chain_entries=chain[s].GetEntries()
            print "Entries per sample ", ss, " : ", chain_entries

            chunk_size = 100000
            n_iter = int(float(chain_entries)/float(chunk_size))
            c = 0
            if chain_entries%chunk_size!=0:
                n_iter += 1
                
            print "chain_entries: ", chain_entries
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                key_list = arrays.keys()
                cut_mask = (arrays[CUT]>0)
                #Not done in the original rocs, skip
                #if KILL_QCD:
                #    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                #Signal: consider only gen-matched jets
                #No need to match the calorimeter volume
                cut_mask_gen = np.logical_and(cut_mask,arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1)
                cut_mask = (cut_mask_gen.any()==True)
                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                tagged = (sigprob>0.996)
                pt = arrays["Jets.pt"][cut_mask_gen][cut_mask]
                matched_radius = arrays["Jets.radiusLLP"][cut_mask_gen][cut_mask]

                genRadius = arrays["GenLLPs.travelRadius"][cut_mask]
                genX = arrays["GenLLPs.travelX"][cut_mask]
                genY = arrays["GenLLPs.travelY"][cut_mask]
                genZ = arrays["GenLLPs.travelZ"][cut_mask]
                #genTime = arrays["GenLLPs.travelTime"][cut_mask]
                genBeta = arrays["GenLLPs.beta"][cut_mask]
                genGamma = np.divide(1.,np.sqrt(1-np.multiply(genBeta,genBeta)))
                genTravelDist = np.sqrt( np.multiply(genX,genX) + np.multiply(genY,genY) + np.multiply(genZ,genZ) )
                genPosteriorTime = np.divide(genTravelDist,np.multiply(genBeta , genGamma))
                #print "genPosteriorTime"
                #print genPosteriorTime
                #associate the correct travel distance to the jet
                genLLP0_radius = np.transpose(np.stack(( genRadius )))[0:1].flatten()
                genLLP1_radius = np.transpose(np.stack(( genRadius )))[1:2].flatten()

                genLLP0_travel_dist = np.transpose(np.stack(( genTravelDist/100. )))[0:1].flatten()
                genLLP1_travel_dist = np.transpose(np.stack(( genTravelDist/100. )))[1:2].flatten()

                genLLP0_travel_time = np.transpose(np.stack(( genPosteriorTime )))[0:1].flatten()
                genLLP1_travel_time = np.transpose(np.stack(( genPosteriorTime )))[1:2].flatten()

                #n_jet_per_event = matched_radius.astype(bool).count()
                compare_radius_LLP0 = np.multiply(matched_radius.astype(bool),genLLP0_radius)
                compare_radius_LLP1 = np.multiply(matched_radius.astype(bool),genLLP1_radius)

                matched_radius_mask_0 = (matched_radius==compare_radius_LLP0)
                matched_radius_mask_1 = (matched_radius==compare_radius_LLP1)

                matched_travel_dist_LLP0 = np.multiply(matched_radius_mask_0,genLLP0_travel_dist)
                matched_travel_dist_LLP1 = np.multiply(matched_radius_mask_1,genLLP1_travel_dist)
                matched_travel_dist = matched_travel_dist_LLP0 + matched_travel_dist_LLP1

                matched_travel_time_LLP0 = np.multiply(matched_radius_mask_0,genLLP0_travel_time)
                matched_travel_time_LLP1 = np.multiply(matched_radius_mask_1,genLLP1_travel_time)
                matched_travel_time = matched_travel_time_LLP0 + matched_travel_time_LLP1

                #print "genRadius"
                #print genRadius

                #print "matched_radius"
                #print matched_radius

                #print "genTravelDist in m"
                #print genTravelDist/100.

                #print "matched_travel_dist"
                #print matched_travel_dist

                #ctau_weighteight = tau_weight_calc(genPosteriorTime.sum(), float(ct)/10., float( samples[pr]['ctau'] )/10.)
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]
                for ct in ctaus:
                    #This is per event --> now per jet
                    #unflattened version

                    #I have no other choice but concatenate, this has the right shape anyways
                    tmp_ctau_weight = tau_weight_calc(genPosteriorTime.sum(), float(ct)/10., float( samples[s]['ctau'] )/10.)

                    ctau_event_weight[ct] = np.concatenate(( ctau_event_weight[ct], np.multiply( tmp_ctau_weight, np.multiply(matched_radius.astype(bool),weight) )    ))
                    den_w = np.concatenate(( den_w, np.multiply( tmp_ctau_weight, np.multiply(matched_radius.astype(bool),weight) ) ))
                    #now apply the sigprob mask
                    num_w = np.concatenate(( num_w, np.multiply( tmp_ctau_weight, np.multiply(tagged,weight)) ))

                    #This is per jet
                    #unflattened version
                    travel_dist_reweighted[ct] = np.concatenate(( travel_dist_reweighted[ct], np.multiply( tmp_ctau_weight, matched_travel_dist )    ))
                    #den_val = np.concatenate(( den_val, np.multiply( tmp_ctau_weight, matched_travel_dist ) ))
                    #num_val = np.concatenate(( num_val, np.multiply( tmp_ctau_weight, np.multiply(tagged,matched_travel_dist) ) ))
                    den_val = np.concatenate(( den_val, np.multiply( matched_radius.astype(bool),ct/1000. ) ))
                    num_val = np.concatenate(( num_val, np.multiply( tagged,ct/1000. ) ))

                del arrays

                prob_s = np.concatenate( (prob_s, np.hstack(sigprob)) )
                w_s = np.concatenate( (w_s, np.hstack( sigprob.astype(bool)*weight ) ) )

                #We have the ctau weight
                #Now we need num/den and the corresponding ctau of the matched LLP
                
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"

        '''
        print "den_val"
        print den_val
        print den_val.shape
        print "num_val"
        print num_val
        print num_val.shape

        print "\n"
        print "den_w"
        print den_w
        print den_w.shape

        print "num_w"
        print num_w
        print num_w.shape
        #truth_s_bin = np.dstack( (np.ones(prob_s.shape[0]),np.zeros(prob_s.shape[0])) ).reshape(prob_s.shape[0],2)
        #truth_s = np.ones(prob_s.shape[0])
        '''

        #Throw these vectors to a histogram
        bins = np.array([0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000])/100.
        bins = np.array([0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,10000,100000])/100.
        bins = np.array([1,10,100,250,500,1000,2500,5000,10000,100000])/1000.
        h_num[s] = TH1F(s+"_num",s+"_num",len(bins)-1,bins)#array('f',bins))
        h_den[s] = TH1F(s+"_den",s+"_den",len(bins)-1,bins)#array('f',bins))

        arr_num = np.array([])
        arr_den = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])
        
        for row in num_val:
            arr_num = np.concatenate(( arr_num, np.stack(row)  ))

        for row in den_val:
            arr_den = np.concatenate(( arr_den, np.stack(row)  ))

        for row in num_w:
            arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))

        for row in den_w:
            arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))

        root_numpy.fill_hist(h_num[s],arr_num,arr_w_num)
        root_numpy.fill_hist(h_den[s],arr_den,arr_w_den)

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(h_num[s],h_den[s])


        eff[s].SetTitle("")
        eff[s].SetMarkerColor(colors[i])
        eff[s].SetLineColor(colors[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(20)
        eff[s].GetXaxis().SetTitle("c #tau_{0} (m)")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.01)
        eff[s].SetMinimum(-0.01)

        if i ==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")


    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,custom_spacing=0.6)

    leg = TLegend(0.6, 0.2, 0.9, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],"m_{#chi} = "+str(samples[s]['mass'])+" GeV","PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"TDJ_eff.pdf")
    can.Print(out_fold+"TDJ_eff.png")

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)


def compute_ROC_supplementary(ERA):
    '''ROC'''
    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH','SUSY_mh127_ctau3000_HH',
        'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH','SUSY_mh400_ctau3000_HH',
        'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH','SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH','SUSY_mh1800_ctau3000_HH',
    ]
    back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET","VV"]
    CUT = "isSR"
    #back = ["VV"]

    #Need to load signal with the same preselections as background
    #Can probably do uproot SR, should be the same
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2018_SR/"
    #or even
    #---> pick one with time smearing? TODO
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    NTUPLEDIR_v5 = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_v5_ntuples/"

    ROCDIR = "/nfs/dust/cms/user/lbenato/ROC_numpy_"+ERA+"/"
    chain = {}

    print "Missing tree weights!"

    tree_weight_dict_b = get_tree_weights(NTUPLEDIR_v5,back,ERA,main_pred_sample="HighMET")
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    #Background: all together
    truth_b = np.array([])
    prob_b = np.array([])
    w_b = np.array([])
    for i, s in enumerate(back):
        print "Adding bkg ", s
        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept","HT",CUT]
        chain[s] = TChain("tree")

        for l, ss in enumerate(samples[s]["files"]):
            #check if the file exists
            if not os.path.isfile(NTUPLEDIR_v5 + ss + ".root"):
                continue
            else:
                print l, ss
                chain[s].Add(NTUPLEDIR_v5 + ss + ".root")

            chain_entries=chain[s].GetEntries()
            print "Entries per sample ", ss, " : ", chain_entries

            chunk_size = 100000
            n_iter = int(float(chain_entries)/float(chunk_size))
            c = 0
            if chain_entries%chunk_size!=0:
                n_iter += 1
                
            print "chain_entries: ", chain_entries
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            gen = uproot.iterate([NTUPLEDIR_v5 + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                key_list = arrays.keys()
                cut_mask = (arrays[CUT]>0)
                #Not done in the original rocs, skip
                #if KILL_QCD:
                #    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                print "Skip events with negative weights . . ."
                cut_mask = np.logical_and( cut_mask, arrays["EventWeight"]>0)

                sigprob = arrays["Jets.sigprob"][cut_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_b[s][ss]
                del arrays

                prob_b = np.concatenate( (prob_b, np.hstack(sigprob)) )
                w_b = np.concatenate( (w_b, np.hstack( sigprob.astype(bool)*weight ) ) )

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"

                print "tmp bkg shapes: "
                print prob_b.shape
                print w_b.shape


    print "tot background : ", prob_b, w_b
    print "shapes: ",
    print prob_b.shape
    print w_b.shape
    truth_b_bin = np.dstack( (np.zeros(prob_b.shape[0]),np.ones(prob_b.shape[0])) ).reshape(prob_b.shape[0],2)
    truth_b = np.zeros(prob_b.shape[0])
    
    #Normalize background weights
    norm_b = w_b.sum(axis=0)
    w_b_norm = np.true_divide(w_b,norm_b)


    #signal tree weight should count only for the dnn output plot

    for i, s in enumerate(sign):
        truth_s = np.array([])
        prob_s = np.array([])
        w_s = np.array([])

        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatchedCaloCorrLLPAccept",CUT]


        chain[s] = TChain("tree")
        for l, ss in enumerate(samples[s]["files"]):
            print l, ss
            chain[s].Add(NTUPLEDIR + ss + ".root")

            chain_entries=chain[s].GetEntries()
            print "Entries per sample ", ss, " : ", chain_entries

            chunk_size = 100000
            n_iter = int(float(chain_entries)/float(chunk_size))
            c = 0
            if chain_entries%chunk_size!=0:
                n_iter += 1
                
            print "chain_entries: ", chain_entries
            print "chunk_size: ", chunk_size
            print "n_iter: ", n_iter

            gen = uproot.iterate([NTUPLEDIR + ss + ".root"],"tree",list_of_variables,entrysteps=chunk_size)
            for arrays in gen:
                key_list = arrays.keys()
                cut_mask = (arrays[CUT]>0)
                #Not done in the original rocs, skip
                #if KILL_QCD:
                #    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)
                print "Skip events with negative weights . . ."
                cut_mask = np.logical_and( cut_mask, arrays["EventWeight"]>0)

                #Signal: consider only gen-matched jets
                cut_mask_gen = np.logical_and(cut_mask,arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1)
                cut_mask = (cut_mask_gen.any()==True)
                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]
                del arrays

                prob_s = np.concatenate( (prob_s, np.hstack(sigprob)) )
                w_s = np.concatenate( (w_s, np.hstack( sigprob.astype(bool)*weight ) ) )

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"



        print "tot signal ",s," : ", prob_s, w_s
        print "shapes: ",
        print prob_s.shape
        print w_s.shape
        truth_s_bin = np.dstack( (np.ones(prob_s.shape[0]),np.zeros(prob_s.shape[0])) ).reshape(prob_s.shape[0],2)
        truth_s = np.ones(prob_s.shape[0])


        #Normalize weights
        norm_s = w_s.sum(axis=0)
        w_s_norm = np.true_divide(w_s,norm_s)


        y_test    = np.concatenate((truth_b,truth_s))
        w_test    = np.concatenate((w_b_norm,w_s_norm))
        prob_test = np.concatenate((prob_b,prob_s))

        tpr_L = 0.636040507008249
        fpr_L = 0.00040904540701505433
        cut_fpr = fpr_L

        AUC = roc_auc_score(y_test, prob_test, sample_weight=w_test)
        fpr, tpr, thresholds = roc_curve(y_test, prob_test, sample_weight=w_test)
        idx, _ = find_nearest(fpr,cut_fpr)

        '''
        #matplotlib
        plt.figure(figsize=(8,7))
        plt.rcParams.update({"font.size": 15})
        plt.plot(fpr, tpr, color="crimson", lw=2, label="AUC = {0:.4f}".format(AUC))
        plt.plot(fpr[idx], tpr[idx],"ro",color="crimson",label="w.p. {0:.4f}".format(thresholds[idx]))
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based")
        plt.ylim([0.6, 1.05])
        plt.xlim([0.0001, 1.05])
        plt.xscale("log")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend(loc="lower right", title="FCN")
        plt.grid(True) 
        plt.savefig(ROCDIR+"roc_"+s+".pdf")
        plt.savefig(ROCDIR+"roc_"+s+".png")
        '''

        '''
        #ROOT
        g = TGraph(len(fpr)-1,fpr,tpr)
        p = TGraph(1,np.array([fpr[idx]]),np.array([tpr[idx]]))
        #bin_x = array('d',np.logspace(-4,-2,20000).tolist())
        #bin_x = array('d',np.linspace(0.0001,0.01,40000).tolist())
        #bin_y = array('d',np.linspace(0.6,1.,40000).tolist())
        #h = TH2F("h","",len(bin_x)-1,bin_x,len(bin_y)-1,bin_y)
        #root_numpy.fill_hist(h,np.transpose(np.stack((fpr,tpr))))
        can = TCanvas("can","can",900,800)
        can.cd()
        can.SetGrid()
        can.SetLogx()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.1)
        g.SetTitle("")
        g.GetXaxis().SetTitle("False positive rate")
        g.GetYaxis().SetTitle("True positive rate")
        g.GetXaxis().SetTitleSize(0.04)
        g.GetYaxis().SetTitleSize(0.04)
        g.SetMarkerColor(2)
        g.SetLineColor(2)
        g.SetLineWidth(4)
        g.Draw("ACPL")
        p.SetMarkerStyle(20)
        p.SetMarkerSize(1.5)
        p.SetMarkerColor(4)
        p.Draw("P,sames")
        g.SetMaximum(1.01)
        g.SetMinimum(0.5)
        g.GetXaxis().SetRangeUser(0.00001,1.)#(0.00001,0.01)
        g.GetXaxis().SetNdivisions(10)#(0.00001,0.01)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True)
        leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
        leg.SetTextSize(0.035)
        leg.AddEntry(g,"ROC curve","L")
        leg.AddEntry(p,"DNN>0.996 w. p.","P")
        leg.SetBorderSize(0)
        leg.Draw()
        can.Print(ROCDIR+"roc_"+s+".pdf")
        can.Print(ROCDIR+"roc_"+s+".png")
        can.Close()

        print "Info: ROC curve file "+ROCDIR+"roc_"+s+".pdf created"
        '''
        with open(ROCDIR+"fpr_"+s+".npy", "wb") as f:
            np.save(f, fpr)
        with open(ROCDIR+"tpr_"+s+".npy", "wb") as f:
            np.save(f, tpr)
        with open(ROCDIR+"thresholds_"+s+".npy", "wb") as f:
            np.save(f, thresholds)
        with open(ROCDIR+"idx_"+s+".npy", "wb") as f:
            np.save(f, idx)
        print "Info: numpy array saved in "+ROCDIR

def plot_ROC_supplementary(ERA,extra_label=""):
    '''ROC'''
    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH','SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH','SUSY_mh400_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH','SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH','SUSY_mh1800_ctau3000_HH',
    ]
    back = ["ZJetsToNuNu","QCD","WJetsToLNu","TTbarGenMET","VV"]
    ROCDIR = "/nfs/dust/cms/user/lbenato/ROC_numpy_"+ERA+"/"
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    NTUPLEDIR_v5 = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_v5_ntuples/"

    tree_weight_dict_b = get_tree_weights(NTUPLEDIR_v5,back,ERA,main_pred_sample="HighMET")
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")
    
    #Load fpr, tpr and so on
    tpr = {}
    fpr = {}
    idx = {}
    thresholds = {}
    g = {}
    p = {}
    for i,s in enumerate(sign):
        print "Loading ", s, ROCDIR+"tpr_"+s+".npy"
        fpr[s] = (np.load(ROCDIR+"fpr_"+s+".npy"))#[0:1000000]
        tpr[s] = (np.load(ROCDIR+"tpr_"+s+".npy"))#[0:1000000]
        idx[s] = (np.load(ROCDIR+"idx_"+s+".npy"))
        print "size: ", len(fpr[s])
        g[s] = TGraph(len(fpr[s])-1,fpr[s],tpr[s])
        #p[s] = TGraph(1,np.array([fpr[s][ idx[s] ]]),np.array([tpr[s][idx[s]]]))
        del fpr[s]
        del tpr[s]
        del idx[s]

    print "Done loading, save graphs"

    g_file = TFile(out_fold+"roc_graphs_"+ERA+".root","RECREATE")
    g_file.cd()

    for s in sign:
        g[s].SetTitle("")
        g[s].GetXaxis().SetTitle("False positive rate")
        g[s].GetYaxis().SetTitle("True positive rate")
        g[s].GetXaxis().SetTitleSize(0.04)
        g[s].GetYaxis().SetTitleSize(0.04)
        g[s].SetMarkerColor(samples[s]['linecolor'])
        g[s].SetLineColor(samples[s]['linecolor'])
        g[s].SetLineWidth(4)
        #p[s].SetMarkerStyle(20)
        #p[s].SetMarkerSize(1.5)
        #p[s].SetMarkerColor(samples[s]['linecolor'])
        g[s].SetMaximum(1.01)
        g[s].SetMinimum(0.5)
        g[s].GetXaxis().SetRangeUser(0.00001,1.)
        g[s].Write("roc_"+s)
        #p[s].Write("p_"+s)
    print "Writing "+out_fold+"roc_graphs_"+ERA+".root"
    print "Closing the file... "
    g_file.Close()


    print "Setting the canvas, try to write it on root instead of printing"
    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetGrid()
    can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)

    it = 0 

    for s in sign:
        g[s].SetTitle("")
        g[s].GetXaxis().SetTitle("False positive rate")
        g[s].GetYaxis().SetTitle("True positive rate")
        g[s].GetXaxis().SetTitleSize(0.04)
        g[s].GetYaxis().SetTitleSize(0.04)
        g[s].SetMarkerColor(samples[s]['linecolor'])
        g[s].SetLineColor(samples[s]['linecolor'])
        g[s].SetLineStyle(samples[s]['linestyle'])
        g[s].SetLineWidth(4)
        if it==0:
            g[s].Draw("ACPL")
        else:
            g[s].Draw("CPL,sames")
        #p[s].SetMarkerStyle(20)
        #p[s].SetMarkerSize(1.5)
        #p[s].SetMarkerColor(samples[s]['linecolor'])
        #p[s].Draw("P,sames")
        g[s].SetMaximum(1.01)
        g[s].SetMinimum(0.5)
        g[s].GetXaxis().SetRangeUser(0.00001,1.)
        leg.AddEntry(g[s],samples[s]['label'],"L")
        #leg.AddEntry(p[s],"DNN>0.996 w. p.","P")
        it+=1

    leg.SetBorderSize(0)
    leg.Draw()
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True)

    #print "Print the canvas"
    #can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".pdf")
    #can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".png")

    print "Prepared the canvas but not printing it, try to store it in a root file"
    c_file = TFile(out_fold+"roc_canvas_"+ERA+".root","RECREATE")
    c_file.cd()
    can.Write("can")
    c_file.Close()

    print "Print the canvas"
    can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".pdf")
    can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".png")

    can.Close()
    print "Written ", out_fold+"roc_canvas_"+ERA+".root"

    del g
    del p
    exit()

    #now reopen for plotting
    print "now reopen for plotting"
    g_file = TFile(out_fold+"roc_graphs_"+ERA+".root","READ")
    g_file.cd()
    g = {}
    p = {}
    it = 0 

    print "Setting the canvas and cd to it"
    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)

    for s in sign:
        g[s] = g_file.Get("roc_"+s)
        p[s] = g_file.Get("p_"+s)
        g[s].SetTitle("")
        g[s].GetXaxis().SetTitle("False positive rate")
        g[s].GetYaxis().SetTitle("True positive rate")
        g[s].GetXaxis().SetTitleSize(0.04)
        g[s].GetYaxis().SetTitleSize(0.04)
        g[s].SetMarkerColor(samples[s]['linecolor'])
        g[s].SetLineColor(samples[s]['linecolor'])
        g[s].SetLineWidth(4)
        if it==0:
            g[s].Draw("ACPL")
        else:
            g[s].Draw("CPL,sames")
        p[s].SetMarkerStyle(20)
        p[s].SetMarkerSize(1.5)
        p[s].SetMarkerColor(samples[s]['linecolor'])
        p[s].Draw("P,sames")
        g[s].SetMaximum(1.01)
        g[s].SetMinimum(0.5)
        g[s].GetXaxis().SetRangeUser(0.00001,1.)
        leg.AddEntry(g[s],samples[s]['label'],"L")
        #leg.AddEntry(p[s],"DNN>0.996 w. p.","P")
        it+=1
    leg.SetBorderSize(0)
    leg.Draw()
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True)
    #print "Prepared the canvas but not printing it"
    print "Print the canvas"
    can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".pdf")
    #can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".png")
    g_file.Close()
    exit()

    #Matplotlib
    plt.figure(figsize=(8,7))
    plt.rcParams.update({"font.size": 15})

    #Need to define colors and styles
    colors = ['crimson','crimson','green','green','skyblue','skyblue','orange','orange','gray','gray','magenta','magenta','chocolate','chocolate','yellow','yellow','black','black','olive','olive']
    #linestyles = ['-', '--', '-.', ':','-','--','-.',':']
    #linestyles = ['-', '--', '-.', '-','--','-.',':','-', '--', '-.',]
    linestyles = ['-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--','-', '--',]
    AUC = 0.4
    for i,s in enumerate(sign):
        plt.plot(fpr[s], tpr[s], color=colors[i], lw=2, linestyle=linestyles[i], label=samples[s]['label'])#"AUC = {0:.4f}".format(AUC))
        #plt.plot(fpr[l][idx[l]], tpr[l][idx[l]],"ro",color=colors[i],label="w.p. {0:.4f}".format(thresholds[l][idx[l]]))
    #plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    #plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based, all eta(2018)")
    plt.title(str(ERA)+' MC')
    plt.ylim([0.6, 1.05])
    plt.xlim([0.0001, 1.05])
    plt.xscale("log")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", title="FCN")
    plt.grid(True)
    plt.savefig(out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".pdf")
    plt.savefig(out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".png")
    print "Info: ROC curve file "+out_fold+"plt_TDJ_ROC_supplementary"+extra_label+".pdf has been created"

    exit()
    #ROOT
    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)
    for i, s in enumerate(sign):
        g[s].SetTitle("")
        g[s].GetXaxis().SetTitle("False positive rate")
        g[s].GetYaxis().SetTitle("True positive rate")
        g[s].GetXaxis().SetTitleSize(0.04)
        g[s].GetYaxis().SetTitleSize(0.04)
        g[s].SetMarkerColor(samples[s]['linecolor'])
        g[s].SetLineColor(samples[s]['linecolor'])
        g[s].SetLineWidth(4)
        if i==0:
            g[s].Draw("ACPL")
        else:
            g[s].Draw("CPL,sames")
        p[s].SetMarkerStyle(20)
        p[s].SetMarkerSize(1.5)
        p[s].SetMarkerColor(samples[s]['linecolor'])
        p[s].Draw("P,sames")
        g[s].SetMaximum(1.01)
        g[s].SetMinimum(0.5)
        g[s].GetXaxis().SetRangeUser(0.00001,1.)
        leg.AddEntry(g[s],samples[s]['label'],"L")
        leg.AddEntry(p[s],"DNN>0.996 w. p.","P")
    leg.SetBorderSize(0)
    leg.Draw()
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True)
    can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".pdf")
    can.Print(out_fold+"root_TDJ_ROC_supplementary"+extra_label+".png")

    exit()
    '''
    plt.figure(figsize=(8,7))
    plt.rcParams.update({"font.size": 15}) #Larger font size                                                                                                                        
    AUC = 0.4
    for i,l in enumerate(list_comparison):
        tpr[l] = np.load(PLOTDIR+"tpr"+l+".npy")
        fpr[l] = np.load(PLOTDIR+"fpr"+l+".npy")
        idx[l] = np.load(PLOTDIR+"idx"+l+".npy")
        thresholds[l] = np.load(PLOTDIR+"thresholds"+l+".npy")
        plt.plot(fpr[l], tpr[l], color=colors[i], lw=2, linestyle=linestyles[i], label="ROC"+l)#"AUC = {0:.4f}".format(AUC))                                                        
        plt.plot(fpr[l][idx[l]], tpr[l][idx[l]],"ro",color=colors[i],label="w.p. {0:.4f}".format(thresholds[l][idx[l]]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot(fpr_L,tpr_L,"ro",color="blue",label="cut based, all eta(2018)")
    plt.title(str(ERA)+' MC')
    plt.ylim([0.6, 1.05])
    plt.xlim([0.0001, 1.05])
    plt.xscale("log")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", title="FCN")
    plt.grid(True)
    plt.savefig(PLOTDIR+"ROC_comparison.pdf")
    plt.savefig(PLOTDIR+"ROC_comparison.png")
    print "Info: ROC curve file "+PLOTDIR+"ROC_comparison.pdf has been created"
    fpr, tpr, thresholds = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])
    cut_fpr = 0.00040904540701505433
    idx, _ = find_nearest(fpr,cut_fpr)
    '''
    #bin_x = array('d',np.logspace(-4,-2,20000).tolist())
    #bin_x = array('d',np.linspace(0.0001,0.01,40000).tolist())
    #bin_y = array('d',np.linspace(0.6,1.,40000).tolist())
    #h = TH2F("h","",len(bin_x)-1,bin_x,len(bin_y)-1,bin_y)
    #root_numpy.fill_hist(h,np.transpose(np.stack((fpr,tpr))))

def plot_s_b(var,label,nbins,bin_min,bin_max):
    '''s and b'''
    inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    #"/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v5_calo_AOD_2018_dnn_x-check/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    store = pd.HDFStore(inp_file)
    df_test = store.select("df")
    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    #print list(df_test.columns)
    #var = "sigprob"
    weight = "EventWeightNormalized"
    back = np.array(df_test[var].loc[df_test["is_signal"]==0].values)
    sign = np.array(df_test[var].loc[df_test["Jet_isGenMatchedCaloCorrLLPAccept"]==1].values)
    back_w = np.array(df_test[weight].loc[df_test["is_signal"]==0].values)
    sign_w = np.array(df_test[weight].loc[df_test["Jet_isGenMatchedCaloCorrLLPAccept"]==1].values)
    hb = TH1F("b","",nbins,bin_min,bin_max)#50,0,1)
    hs = TH1F("s","",nbins,bin_min,bin_max)#50,0,1)
    root_numpy.fill_hist(hb,back,back_w)
    root_numpy.fill_hist(hs,sign,sign_w)
    max_val = max(hb.GetMaximum(),hs.GetMaximum())
    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    hb.SetLineColor(860)
    hs.SetLineColor(633)
    hb.SetFillColorAlpha(860,0.3)
    hs.SetFillColorAlpha(633,0.3)
    hb.SetLineWidth(2)
    hs.SetLineWidth(2)
    hb.GetXaxis().SetTitle(label)
    hb.GetYaxis().SetTitle("Events")
    hb.GetXaxis().SetTitleSize(0.04)
    hb.GetYaxis().SetTitleSize(0.04)
    hb.Draw("HISTO")
    hb.SetMaximum(1.01)#(max_val*1.2)
    hb.SetMinimum(0.0001)
    hs.Draw("HISTO,sames")
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True)
    leg = TLegend(0.45, 0.7, 0.65, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(hb,"background","F")
    leg.AddEntry(hs,"signal","F")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"TDJ_SigAndBkg_"+var+".pdf")
    can.Print(out_fold+"TDJ_SigAndBkg_"+var+".png")
    

def plot_dnn_modeling():
    '''ZtoLLPho'''
    #2016
    #2017
    #2018
    
def plot_mistag():
    '''mistag with uncertainty'''
    
    less_bins_plot = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
    label_dict = {}
    label_dict["ZtoLL"] = "Z #rightarrow ll"
    label_dict["WtoLN"] = "W #rightarrow l#nu"
    label_dict["JetHT"] = "QCD"
    label_dict["TtoEM"] = "ttbar e + #mu"

    BASEDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_%s_"
    YIELDDIR = "plots/Yields_AN_fix/v6_calo_AOD_%s_SR"
    colors = {"WtoLN":2,"ZtoLL":1,"JetHT":418,"TtoEM":4}
    markers = {"WtoLN":20,"ZtoLL":24,"JetHT":24,"TtoEM":21}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":3,"TtoEM":4}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":2,"TtoEM":2}
    maxeff = 0.0016

    def do_mistag(eras,LUMI,era_label,plot_label):
        num = {}
        den = {}
        eff = {}
        unc_up = {}
        unc_down = {}
        y = {}
        for k in label_dict.keys():
            den[k] = TH1F()#"num_"+k,"num_"+k,len(less_bins_plot)-1,less_bins_plot)
            num[k] = TH1F()#"den_"+k,"den_"+k,len(less_bins_plot)-1,less_bins_plot)
            eff[k] = TGraphAsymmErrors()
            y[k] = 0.
            unc_up[k] = []
            unc_down[k] = []
            for i, era in enumerate(eras):
                label = ""
                label_y = ""
                if "2016" in era:
                    label+="_"+era[5:8]
                label += "_MinDPhi_0p5_eta_1p0"
                label_y +="_eta_1p0"
                if "2016" not in era:
                    label+="_phi_cut"
                    label_y +="_phi_cut"
                label+="_vs_eta"
                label_y+="_vs_eta"
                if "2016" in era:
                    label_y+="_"+era[5:8]
                label_y+="_MinDPhi_0p5"
                DIR = (BASEDIR%era[0:4])+k+"/"
                yieldfile = (YIELDDIR%era[0:4])+"/BkgPredResults_"+era[0:4]+"_SR_HighMET"+label_y+".yaml"
                with open(yieldfile) as f:
                    results = yaml.load(f, Loader=yaml.Loader)
                    f.close()
                for rk in results.keys():
                    if k in rk:
                        y[k] +=results[rk]["HighMET"]["pred_2_from_1"]
                infile = TFile(DIR+"TagEff_"+k+label+".root","READ")
                tmp_den = infile.Get("den_"+k)
                tmp_num = infile.Get("num_"+k)
                tmp_den.SetDirectory(0)
                tmp_num.SetDirectory(0)
                if i==0:
                    den[k] = tmp_den.Rebin(len(less_bins_plot)-1,k+"_den2",less_bins_plot)
                    num[k] = tmp_num.Rebin(len(less_bins_plot)-1,k+"_num2",less_bins_plot)
                    den[k].SetDirectory(0)
                    num[k].SetDirectory(0)
                else:
                    den[k].Add(tmp_den.Rebin(len(less_bins_plot)-1,k+"_den2",less_bins_plot))
                    num[k].Add(tmp_num.Rebin(len(less_bins_plot)-1,k+"_num2",less_bins_plot))
                    den[k].SetDirectory(0)
                    num[k].SetDirectory(0)

        max_unc = 0
        for k in label_dict.keys():
            unc = abs(y[k] - y["WtoLN"])/y["WtoLN"]
            print k, y[k], unc
            max_unc = max(max_unc,unc)
    
        for k in ["WtoLN"]:#label_dict.keys()
            eff[k].BayesDivide(num[k],den[k])
            eff[k].SetMarkerStyle(markers[k])
            eff[k].SetMarkerColor(1)#(colors[k])
            eff[k].SetMarkerSize(1.3)
            eff[k].SetLineColor(1)#(colors[k])
            eff[k].SetFillColorAlpha(922,0.5)#(colors[k],0.5)
            #eff[k].SetFillStyle(3001)
            eff[k].SetLineWidth(2)
            eff[k].SetLineStyle(lines[k])
            eff[k].GetYaxis().SetRangeUser(-0.0001,maxeff)
            np = eff[k].GetN()
            for i in range(0,np):
                y_val = eff[k].GetPointY(i)
                print "BUGFIX!"
                eff[k].SetPointEYhigh(i, math.sqrt(eff[k].GetErrorYhigh(i)**2 + pow(max_unc*eff[k].GetPointY(i),2) ))#not 0.5! all of it!
                eff[k].SetPointEYlow(i, math.sqrt(eff[k].GetErrorYlow(i)**2 + pow(max_unc*eff[k].GetPointY(i),2) ))

            print eff[k].Print()

            eff[k].GetYaxis().SetRangeUser(-0.0001,maxeff)
            eff[k].GetXaxis().SetTitle("Jet #eta")
            eff[k].GetYaxis().SetTitle("Mis-tag efficiency")
            eff[k].GetXaxis().SetTitleSize(0.04)
            eff[k].GetYaxis().SetTitleSize(0.04)

        can = TCanvas("can","can",900,800)
        can.cd()
        #can.SetGrid()
        #can.SetLogx()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.15)
        can.SetBottomMargin(0.1)
        leg = TLegend(0.7, 0.78, 0.93, 0.88)
        leg.SetTextSize(0.035)
        leg.SetBorderSize(0)
        for i,k in enumerate(["WtoLN"]):#"ZtoLL","TtoEM","JetHT",
            if i==0:
                eff[k].Draw("APE2")
            else:
                eff[k].Draw("P,sames")
            leg.AddEntry(eff[k],label_dict[k],"PF")
        
        #drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True)
        if PRELIMINARY:
            drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True,left_marg_CMS=0.25)
        else:
            drawCMS_simple(LUMI, "", ERA=era_label, onTop=True,left_marg_CMS=0.25)
        #leg.Draw()
        can.Print(out_fold+"MistagEfficiency"+plot_label+".pdf")
        can.Print(out_fold+"MistagEfficiency"+plot_label+".png")
        can.Close()

    eras = ["2016_G-H","2017","2018"]

    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016

    #probably wrong, frozen for approval:
    #LUMI_G2018 = 111941.400399

    #new and fixed
    LUMI_G2018  = lumi2018["HighMET"]["tot"] + lumi2017["HighMET"]["tot"] + lumi2016["HighMET"]["G"] + lumi2016["HighMET"]["H"]

    era_label=""#"2016-2018"#"2016G-2018"
    plot_label="2016GTo2018"
    do_mistag(eras,LUMI_G2018,era_label=era_label,plot_label=plot_label)

    eras = ["2016_B-F"]
    era_label=""#"2016"#"2016B-F"
    plot_label="2016BF"

    #probably wrong, frozen at approval:
    #LUMI_BF = 137478.722953 - 111941.400399

    #new and fixed
    LUMI_BF  = lumi2016["HighMET"]["tot"] - lumi2016["HighMET"]["G"] - lumi2016["HighMET"]["H"]
    do_mistag(eras,LUMI_BF,era_label=era_label,plot_label=plot_label)

def plot_sf_photon():
    '''SF'''
    FOLD = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_ZtoLLPho_v5_ntuples/"
    #FOLD = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_ZtoLLPho_v5_ntuples_updated/smearing/"
    eras = ["2016_B-F","2016_G-H","2017","2018"]
    LUMI = 137478.722953

    for i,era in enumerate(eras):
        INPDIR = (FOLD%era[0:4])
        infile = TFile(INPDIR+"DataMCNumDen_ZtoLLPho_"+era+".root","READ")
        tmp_den_b = infile.Get("den_b")
        tmp_num_b = infile.Get("num_b")
        tmp_den_d = infile.Get("den_d")
        tmp_num_d = infile.Get("num_d")
        tmp_den_b_2ns = infile.Get("den_b_2ns")
        tmp_num_b_2ns = infile.Get("num_b_2ns")
        tmp_den_d_2ns = infile.Get("den_d_2ns")
        tmp_num_d_2ns = infile.Get("num_d_2ns")
        tmp_den_b.SetDirectory(0)
        tmp_num_b.SetDirectory(0)
        tmp_den_d.SetDirectory(0)
        tmp_num_d.SetDirectory(0)
        tmp_den_b_2ns.SetDirectory(0)
        tmp_num_b_2ns.SetDirectory(0)
        tmp_den_d_2ns.SetDirectory(0)
        tmp_num_d_2ns.SetDirectory(0)
        if i==0:
            num_b = tmp_num_b
            den_b = tmp_den_b
            num_d = tmp_num_d
            den_d = tmp_den_d
            num_b_2ns = tmp_num_b_2ns
            den_b_2ns = tmp_den_b_2ns
            num_d_2ns = tmp_num_d_2ns
            den_d_2ns = tmp_den_d_2ns
        else:
            num_b.Add(tmp_num_b)
            den_b.Add(tmp_den_b)
            num_d.Add(tmp_num_d)
            den_d.Add(tmp_den_d)
            num_b_2ns.Add(tmp_num_b_2ns)
            den_b_2ns.Add(tmp_den_b_2ns)
            num_d_2ns.Add(tmp_num_d_2ns)
            den_d_2ns.Add(tmp_den_d_2ns)
        den_b.SetDirectory(0)
        num_b.SetDirectory(0)
        den_d.SetDirectory(0)
        num_d.SetDirectory(0)
        den_b_2ns.SetDirectory(0)
        num_b_2ns.SetDirectory(0)
        den_d_2ns.SetDirectory(0)
        num_d_2ns.SetDirectory(0)

    print den_b.Print()
    eff_b = TGraphAsymmErrors()
    eff_d = TGraphAsymmErrors()
    eff_b.BayesDivide(num_b,den_b)
    eff_d.BayesDivide(num_d,den_d)
    ratio = num_d.Clone("r")
    ratio.Divide(den_d)
    ratio.Divide(num_b)
    ratio.Multiply(den_b)

    ratio_2ns = num_d_2ns.Clone("r_2ns")
    ratio_2ns.Divide(den_d_2ns)
    ratio_2ns.Divide(num_b_2ns)
    ratio_2ns.Multiply(den_b_2ns)



    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    eff_b.SetLineColor(861)
    eff_b.SetFillColorAlpha(861,0.3)
    #eff_b.SetFillStyle(3001)
    eff_b.SetLineWidth(0)
    eff_b.SetMarkerColor(861)
    eff_b.SetMarkerStyle(21)
    eff_b.SetMarkerSize(1.3)

    eff_d.SetLineColor(1)
    eff_d.SetLineWidth(2)
    eff_d.SetMarkerColor(1)
    eff_d.SetMarkerStyle(20)
    eff_d.SetMarkerSize(1.3)

    eff_b.GetYaxis().SetRangeUser(-0.01,1.01)
    eff_b.Draw("APE2")
    eff_d.Draw("P,sames")
    eff_b.GetXaxis().SetTitle("Jet #eta")
    eff_b.GetYaxis().SetTitle("Tag efficiency")
    eff_b.GetXaxis().SetTitleSize(0.04)
    eff_b.GetYaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    leg.AddEntry(eff_b,"MC background","PF")
    leg.AddEntry(eff_d,"data","PL")
    leg.Draw()
    can.Print(out_fold+"TDJSignalEfficiency_pho.pdf")
    can.Print(out_fold+"TDJSignalEfficiency_pho.png")
    can.Close()


    #Add discrepancy wrt 2 ns in quadrature
    g = TGraphAsymmErrors(ratio.GetNbinsX()-1)
    for b in range(1,ratio.GetNbinsX()+1):
        i = b-1
        #print "point: ", b
        diff = abs(ratio.GetBinContent(b)-ratio_2ns.GetBinContent(b))
        diff_rel = diff/ratio.GetBinContent(b)
        bin_error_rel = ratio.GetBinError(b)/ratio.GetBinContent(b)
        error_rel = math.sqrt( diff_rel**2 + bin_error_rel**2  )
        error = error_rel*ratio.GetBinContent(b)
        g.SetPointX(i,ratio.GetBinCenter(b))
        g.SetPointY(i,ratio.GetBinContent(b))
        g.SetPointEXhigh(i, 0.5)
        g.SetPointEXlow(i, 0.5)
        #Should assign 0.5 of the error to high, 0.5 to low
        print "BUGFIX!"
        g.SetPointEYhigh(i, 0.5*error)
        g.SetPointEYlow(i, 0.5*error)

    print g.Print()
    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    g.SetTitle("")
    g.SetLineColor(418)
    g.SetFillColorAlpha(418,0.5)#(920,1.)
    #g.SetFillStyle(3001)
    g.SetLineWidth(2)
    g.SetMarkerColor(418)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.3)
    g.GetYaxis().SetRangeUser(-0.01,2.01)
    ##g.GetYaxis().SetRangeUser(0.9,1.1)
    g.Draw("APE2")
    g.GetXaxis().SetTitle("Jet #eta")
    g.GetYaxis().SetTitle("Data/MC scale factor")
    g.GetXaxis().SetTitleSize(0.04)
    g.GetYaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    #leg.AddEntry(ratio,"data","PL")
    #leg.Draw()
    can.Print(out_fold+"TDJSignalEfficiencyScaleFactor_pho.pdf")
    can.Print(out_fold+"TDJSignalEfficiencyScaleFactor_pho.png")
    can.Close()


def plot_sf():
    '''SF'''
    FOLD_pho = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_ZtoLLPho_v5_ntuples_updated/smearing/"
    FOLD_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_E_v5_ntuples_reweighted/weighted/smearing/"
    eras = ["2016_B-F","2016_G-H","2017","2018"]
    eras_ele = ["2016_G-H","2017","2018"]
    #eras = ["2016_G-H"]
    #eras_ele = ["2016_G-H"]#,"2017"]#,"2018"]
    LUMI = 137478.722953

    for i,era in enumerate(eras):
        INPDIR_pho = (FOLD_pho%era[0:4])
        infile_pho = TFile(INPDIR_pho+"DataMCNumDen_ZtoLLPho_"+era+".root","READ")
        tmp_den_b_pho = infile_pho.Get("den_b_1bin")
        tmp_num_b_pho = infile_pho.Get("num_b_1bin")
        tmp_den_d_pho = infile_pho.Get("den_d_1bin")
        tmp_num_d_pho = infile_pho.Get("num_d_1bin")
        tmp_den_b_pho_2ns = infile_pho.Get("den_b_2ns_1bin")
        tmp_num_b_pho_2ns = infile_pho.Get("num_b_2ns_1bin")
        tmp_den_d_pho_2ns = infile_pho.Get("den_d_2ns_1bin")
        tmp_num_d_pho_2ns = infile_pho.Get("num_d_2ns_1bin")
        tmp_den_b_pho.SetDirectory(0)
        tmp_num_b_pho.SetDirectory(0)
        tmp_den_d_pho.SetDirectory(0)
        tmp_num_d_pho.SetDirectory(0)
        tmp_den_b_pho_2ns.SetDirectory(0)
        tmp_num_b_pho_2ns.SetDirectory(0)
        tmp_den_d_pho_2ns.SetDirectory(0)
        tmp_num_d_pho_2ns.SetDirectory(0)
        if i==0:
            num_b_pho = tmp_num_b_pho
            den_b_pho = tmp_den_b_pho
            num_d_pho = tmp_num_d_pho
            den_d_pho = tmp_den_d_pho
            num_b_pho_2ns = tmp_num_b_pho_2ns
            den_b_pho_2ns = tmp_den_b_pho_2ns
            num_d_pho_2ns = tmp_num_d_pho_2ns
            den_d_pho_2ns = tmp_den_d_pho_2ns
        else:
            num_b_pho.Add(tmp_num_b_pho)
            den_b_pho.Add(tmp_den_b_pho)
            num_d_pho.Add(tmp_num_d_pho)
            den_d_pho.Add(tmp_den_d_pho)
            num_b_pho_2ns.Add(tmp_num_b_pho_2ns)
            den_b_pho_2ns.Add(tmp_den_b_pho_2ns)
            num_d_pho_2ns.Add(tmp_num_d_pho_2ns)
            den_d_pho_2ns.Add(tmp_den_d_pho_2ns)
        den_b_pho.SetDirectory(0)
        num_b_pho.SetDirectory(0)
        den_d_pho.SetDirectory(0)
        num_d_pho.SetDirectory(0)
        den_b_pho_2ns.SetDirectory(0)
        num_b_pho_2ns.SetDirectory(0)
        den_d_pho_2ns.SetDirectory(0)
        num_d_pho_2ns.SetDirectory(0)


    eff_b_pho = TGraphAsymmErrors()
    eff_d_pho = TGraphAsymmErrors()
    eff_b_pho.BayesDivide(num_b_pho,den_b_pho)
    eff_d_pho.BayesDivide(num_d_pho,den_d_pho)

    ratio_pho = num_d_pho.Clone("r_pho")
    ratio_pho.Divide(den_d_pho)
    ratio_pho.Divide(num_b_pho)
    ratio_pho.Multiply(den_b_pho)

    ratio_pho_2ns = num_d_pho_2ns.Clone("r_pho_2ns")
    ratio_pho_2ns.Divide(den_d_pho_2ns)
    ratio_pho_2ns.Divide(num_b_pho_2ns)
    ratio_pho_2ns.Multiply(den_b_pho_2ns)



    for i,era in enumerate(eras_ele):
        INPDIR_ele = (FOLD_ele%era[0:4])
        infile_ele = TFile(INPDIR_ele+"DataMCNumDen_Electron_"+era+".root","READ")
        tmp_den_b_ele = infile_ele.Get("den_b_1bin")
        tmp_num_b_ele = infile_ele.Get("num_b_1bin")
        tmp_den_d_ele = infile_ele.Get("den_d_1bin")
        tmp_num_d_ele = infile_ele.Get("num_d_1bin")
        tmp_den_b_ele_2ns = infile_ele.Get("den_b_2ns_1bin")
        tmp_num_b_ele_2ns = infile_ele.Get("num_b_2ns_1bin")
        tmp_den_d_ele_2ns = infile_ele.Get("den_d_2ns_1bin")
        tmp_num_d_ele_2ns = infile_ele.Get("num_d_2ns_1bin")
        tmp_den_b_ele.SetDirectory(0)
        tmp_num_b_ele.SetDirectory(0)
        tmp_den_d_ele.SetDirectory(0)
        tmp_num_d_ele.SetDirectory(0)
        tmp_den_b_ele_2ns.SetDirectory(0)
        tmp_num_b_ele_2ns.SetDirectory(0)
        tmp_den_d_ele_2ns.SetDirectory(0)
        tmp_num_d_ele_2ns.SetDirectory(0)
        if i==0:
            num_b_ele = tmp_num_b_ele
            den_b_ele = tmp_den_b_ele
            num_d_ele = tmp_num_d_ele
            den_d_ele = tmp_den_d_ele
            num_b_ele_2ns = tmp_num_b_ele_2ns
            den_b_ele_2ns = tmp_den_b_ele_2ns
            num_d_ele_2ns = tmp_num_d_ele_2ns
            den_d_ele_2ns = tmp_den_d_ele_2ns
        else:
            num_b_ele.Add(tmp_num_b_ele)
            den_b_ele.Add(tmp_den_b_ele)
            num_d_ele.Add(tmp_num_d_ele)
            den_d_ele.Add(tmp_den_d_ele)
            num_b_ele_2ns.Add(tmp_num_b_ele_2ns)
            den_b_ele_2ns.Add(tmp_den_b_ele_2ns)
            num_d_ele_2ns.Add(tmp_num_d_ele_2ns)
            den_d_ele_2ns.Add(tmp_den_d_ele_2ns)
        den_b_ele.SetDirectory(0)
        num_b_ele.SetDirectory(0)
        den_d_ele.SetDirectory(0)
        num_d_ele.SetDirectory(0)
        den_b_ele_2ns.SetDirectory(0)
        num_b_ele_2ns.SetDirectory(0)
        den_d_ele_2ns.SetDirectory(0)
        num_d_ele_2ns.SetDirectory(0)


    eff_b_ele = TGraphAsymmErrors()
    eff_d_ele = TGraphAsymmErrors()
    eff_b_ele.BayesDivide(num_b_ele,den_b_ele)
    eff_d_ele.BayesDivide(num_d_ele,den_d_ele)

    ratio_ele = num_d_ele.Clone("r_ele")
    ratio_ele.Divide(den_d_ele)
    ratio_ele.Divide(num_b_ele)
    ratio_ele.Multiply(den_b_ele)

    ratio_ele_2ns = num_d_ele_2ns.Clone("r_ele_2ns")
    ratio_ele_2ns.Divide(den_d_ele_2ns)
    ratio_ele_2ns.Divide(num_b_ele_2ns)
    ratio_ele_2ns.Multiply(den_b_ele_2ns)


    #Create an artificial graph with high/low pt bins
    eff_d_fake = TGraphAsymmErrors(2)
    eff_d_fake.SetPointX(0,0)
    eff_d_fake.SetPointY(0,eff_d_pho.GetPointY(0))
    eff_d_fake.SetPointEXhigh(0, 0.5)
    eff_d_fake.SetPointEXlow(0, 0.5)
    eff_d_fake.SetPointEYhigh(0, eff_d_pho.GetErrorYhigh(0))
    eff_d_fake.SetPointEYlow(0, eff_d_pho.GetErrorYlow(0))
    eff_d_fake.SetPointX(1,1)
    eff_d_fake.SetPointY(1,eff_d_ele.GetPointY(0))
    eff_d_fake.SetPointEXhigh(1, 0.5)
    eff_d_fake.SetPointEXlow(1, 0.5)
    eff_d_fake.SetPointEYhigh(1, eff_d_ele.GetErrorYhigh(0))
    eff_d_fake.SetPointEYlow(1, eff_d_ele.GetErrorYlow(0))
    eff_d_fake.SetLineColor(1)
    eff_d_fake.SetLineWidth(2)
    eff_d_fake.SetMarkerColor(1)
    eff_d_fake.SetMarkerStyle(20)
    eff_d_fake.SetMarkerSize(1.3)
  
    eff_b_fake = TGraphAsymmErrors(2)
    eff_b_fake.SetPointX(0,0)
    eff_b_fake.SetPointY(0,eff_b_pho.GetPointY(0))
    eff_b_fake.SetPointEXhigh(0, 0.5)
    eff_b_fake.SetPointEXlow(0, 0.5)
    eff_b_fake.SetPointEYhigh(0, eff_b_pho.GetErrorYhigh(0))
    eff_b_fake.SetPointEYlow(0, eff_b_pho.GetErrorYlow(0))
    eff_b_fake.SetPointX(1,1)
    eff_b_fake.SetPointY(1,eff_b_ele.GetPointY(0))
    eff_b_fake.SetPointEXhigh(1, 0.5)
    eff_b_fake.SetPointEXlow(1, 0.5)
    eff_b_fake.SetPointEYhigh(1, eff_b_ele.GetErrorYhigh(0))
    eff_b_fake.SetPointEYlow(1, eff_b_ele.GetErrorYlow(0))
    eff_b_fake.SetLineColor(861)
    eff_b_fake.SetFillColorAlpha(861,0.3)
    #eff_b_fake.SetFillStyle(3001)
    eff_b_fake.SetLineWidth(0)
    eff_b_fake.SetMarkerColor(861)
    eff_b_fake.SetMarkerStyle(21)
    eff_b_fake.SetMarkerSize(1.3)

    fake = TH1F("","",2,-0.5,1.5)


    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    fake.Draw("")
    eff_b_fake.Draw("PE2,sames")
    eff_d_fake.Draw("PE,sames")
    fake.GetXaxis().SetBinLabel(1,"p_{T}<70 GeV")
    fake.GetXaxis().SetBinLabel(2,"p_{T}>70 GeV")
    #fake.GetYaxis().SetRangeUser(0.01,5.e7)
    fake.GetXaxis().SetLimits(-0.5,1.5)
    fake.GetXaxis().SetTitle("Jet p_{T} (GeV)")
    fake.GetYaxis().SetTitleSize(0.04)
    fake.GetXaxis().SetTitleOffset(1.2)
    fake.GetXaxis().SetLabelSize(0.06)
    fake.GetYaxis().SetTitle("Tag efficiency")
    fake.GetXaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    leg.AddEntry(eff_b_fake,"MC background","PF")
    leg.AddEntry(eff_d_fake,"data","PL")
    leg.Draw()
    can.Print(out_fold+"TDJSignalEfficiency.pdf")
    can.Print(out_fold+"TDJSignalEfficiency.png")
    can.Close()

    #Add discrepancy wrt 2 ns in quadrature
    g_pho = TGraphAsymmErrors(ratio_pho.GetNbinsX()-1)
    for b in range(1,ratio_pho.GetNbinsX()+1):
        i = b-1
        #print "point: ", b
        diff_pho = abs(ratio_pho.GetBinContent(b)-ratio_pho_2ns.GetBinContent(b))
        diff_rel_pho = diff_pho/ratio_pho.GetBinContent(b)
        bin_error_rel_pho = ratio_pho.GetBinError(b)/ratio_pho.GetBinContent(b)
        error_rel_pho = math.sqrt( diff_rel_pho**2 + bin_error_rel_pho**2  )
        error_pho = error_rel_pho*ratio_pho.GetBinContent(b)
        g_pho.SetPointX(i,ratio_pho.GetBinCenter(b))
        g_pho.SetPointY(i,ratio_pho.GetBinContent(b))
        g_pho.SetPointEXhigh(i, 0.5)
        g_pho.SetPointEXlow(i, 0.5)
        #Should assign 0.5 of the error to high, 0.5 to low --> no, we should simply use the error up and down
        #print "BUGFIX!"
        g_pho.SetPointEYhigh(i, error_pho)
        g_pho.SetPointEYlow(i, error_pho)

    g_ele = TGraphAsymmErrors(ratio_ele.GetNbinsX()-1)
    for b in range(1,ratio_ele.GetNbinsX()+1):
        i = b-1
        #print "point: ", b
        diff_ele = abs(ratio_ele.GetBinContent(b)-ratio_ele_2ns.GetBinContent(b))
        diff_rel_ele = diff_ele/ratio_ele.GetBinContent(b)
        bin_error_rel_ele = ratio_ele.GetBinError(b)/ratio_ele.GetBinContent(b)
        error_rel_ele = math.sqrt( diff_rel_ele**2 + bin_error_rel_ele**2  )
        error_ele = error_rel_ele*ratio_ele.GetBinContent(b)
        g_ele.SetPointX(i,ratio_ele.GetBinCenter(b))
        g_ele.SetPointY(i,ratio_ele.GetBinContent(b))
        g_ele.SetPointEXhigh(i, 0.5)
        g_ele.SetPointEXlow(i, 0.5)
        #Should assign 0.5 of the error to high, 0.5 to low --> no, we should simply use the error up and down
        #print "BUGFIX!"
        g_ele.SetPointEYhigh(i, error_ele)
        g_ele.SetPointEYlow(i, error_ele)


    #Create an artificial graph with high/low pt bins
    g_fake = TGraphAsymmErrors(2)
    g_fake.SetPointX(0,0)
    g_fake.SetPointY(0,g_pho.GetPointY(0))
    g_fake.SetPointEXhigh(0, 0.5)
    g_fake.SetPointEXlow(0, 0.5)
    g_fake.SetPointEYhigh(0, g_pho.GetErrorYhigh(0))
    g_fake.SetPointEYlow(0, g_pho.GetErrorYlow(0))
    g_fake.SetPointX(1,1)
    g_fake.SetPointY(1,g_ele.GetPointY(0))
    g_fake.SetPointEXhigh(1, 0.5)
    g_fake.SetPointEXlow(1, 0.5)
    g_fake.SetPointEYhigh(1, g_ele.GetErrorYhigh(0))
    g_fake.SetPointEYlow(1, g_ele.GetErrorYlow(0))

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.13)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.65, 0.73, 0.9, 0.88)
    leg.SetTextSize(0.035)
    leg.SetBorderSize(0)
    g_fake.SetTitle("")
    g_fake.SetLineColor(418)
    g_fake.SetFillColorAlpha(418,0.5)#(920,1.)
    #g_fake.SetFillStyle(3001)
    g_fake.SetLineWidth(2)
    g_fake.SetMarkerColor(418)
    g_fake.SetMarkerStyle(20)
    g_fake.SetMarkerSize(1.3)
    ##g_fake.GetYaxis().SetRangeUser(0.9,1.1)
    fake.GetXaxis().SetBinLabel(1,"p_{T}<70 GeV")
    fake.GetXaxis().SetBinLabel(2,"p_{T}>70 GeV")
    fake.GetXaxis().SetLimits(-0.5,1.5)
    fake.GetYaxis().SetRangeUser(0.5,1.5)
    fake.GetXaxis().SetTitle("Jet p_{T}")
    fake.GetYaxis().SetTitleSize(0.04)
    fake.GetXaxis().SetTitleOffset(1.2)
    fake.GetXaxis().SetLabelSize(0.06)
    fake.GetYaxis().SetTitle("Data/MC scale factor")
    fake.GetXaxis().SetTitleSize(0.04)
    fake.Draw("")
    g_fake.Draw("PE2,sames")
    g_fake.GetXaxis().SetTitle("Jet p_{T} (GeV)")
    g_fake.GetYaxis().SetTitle("Data/MC scale factor")
    g_fake.GetXaxis().SetTitleSize(0.04)
    g_fake.GetYaxis().SetTitleSize(0.04)
    #drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    #leg.AddEntry(ratio,"data","PL")
    #leg.Draw()
    can.Print(out_fold+"TDJSignalEfficiencyScaleFactor.pdf")
    can.Print(out_fold+"TDJSignalEfficiencyScaleFactor.png")
    can.Close()

def plot_ntag():
    '''nTag'''

    bins = np.array([0.,1.,2.])
    LUMI = 137478.722953


    fake = TH1F("","",3,-0.5,2.5)
    yp = np.array([0.,939.01,0.21])
    ep = np.array([0.,424.91,0.09])

    yo  = np.array([2385007,688.,0.])
    eo = np.sqrt(yo)

    ys = np.array([10.54236,9.3069,2.8792])
    es = np.array([0.38911,0.36653,0.20514])

    ys3m = np.array([7.87377,6.35219,1.19317])
    es3m = np.array([0.32297,0.28982,0.12476])

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

        if i<2:
            go.SetPointX(i,bins[i])
            go.SetPointY(i,yo[i])
            go.SetPointEXhigh(i, 0.5)
            go.SetPointEXlow(i, 0.5)
            go.SetPointEYhigh(i, 0.5*eo[i])
            go.SetPointEYlow(i, 0.5*eo[i])

        gs.Fill(bins[i],ys[i])
        gs3m.Fill(bins[i],ys3m[i])
        '''
        gs.SetPointX(i,bins[i])
        gs.SetPointY(i,ys[i])
        gs.SetPointEXhigh(i, 0.5)
        gs.SetPointEXlow(i, 0.5)
        gs.SetPointEYhigh(i, 0.5*es[i])
        gs.SetPointEYlow(i, 0.5*es[i])

        gs3m.SetPointX(i,bins[i])
        gs3m.SetPointY(i,ys3m[i])
        gs3m.SetPointEXhigh(i, 0.5)
        gs3m.SetPointEXlow(i, 0.5)
        gs3m.SetPointEYhigh(i, 0.5*es3m[i])
        gs3m.SetPointEYlow(i, 0.5*es3m[i])
        '''

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
    fake.GetYaxis().SetRangeUser(0.01,5.e7)
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
    can.Print(out_fold+"nTDJ.pdf")
    can.Print(out_fold+"nTDJ.png")
    can.Close()


def plot_dnn_score():
    FOLD_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_E_v5_ntuples_reweighted/weighted/smearing/"
    eras_ele = ["2016_G-H","2017","2018"]
    extra_era = ["_G-H","",""]
    LUMI = 137478.722953

    for i,era in enumerate(eras_ele):
        INPDIR_ele = (FOLD_ele%era[0:4])
        infile_ele = TFile(INPDIR_ele+"DataAndBkg_sigprob_zoom"+extra_era[i]+".root","READ")
        #infile_ele = TFile(INPDIR_ele+"DataAndBkg_sigprob"+extra_era[i]+".root","READ")
        tmp_back = infile_ele.Get("b")
        tmp_data = infile_ele.Get("d")
        tmp_back.SetDirectory(0)
        tmp_data.SetDirectory(0)
        if i==0:
            print era
            print "era: clone"
            back = tmp_back.Clone("back")
            data = tmp_data.Clone("data")
            back.SetDirectory(0)
            data.SetDirectory(0)
        else:
            print era
            print "era: add"
            back.Add(tmp_back)
            data.Add(tmp_data)
            back.SetDirectory(0)
            data.SetDirectory(0)
        
    back = back.Rebin(2)
    data = data.Rebin(2)

    back_unc = back.Clone("BkgUnc")
    back_unc.SetFillStyle(3002)
    back_unc.SetFillColor(860)
    back_unc.SetLineWidth(0)
    back_unc.SetMarkerStyle(0)

    can = TCanvas("can","can",900,800)
    can.cd()
    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    back.SetLineColor(860)
    data.SetLineColor(1)
    data.SetMarkerColor(1)
    data.SetMarkerStyle(20)
    back.SetFillColorAlpha(860,0.3)
    back.SetLineWidth(2)
    data.SetLineWidth(2)
    back.GetXaxis().SetTitle("Jet tagger output score")
    back.GetYaxis().SetTitle("Events")
    back.GetXaxis().SetTitleSize(0.04)
    back.GetYaxis().SetTitleSize(0.04)
    back.Draw("HIST,sames")
    back_unc.Draw("E2,sames")
    data.Draw("PE,sames")
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True,left_marg_CMS=0.225)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.225)
    leg = TLegend(0.4, 0.7, 0.65, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(back,"simulated background","F")
    leg.AddEntry(back_unc,"stat. unc.","F")
    leg.AddEntry(data,"data","PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"TDJ_DataAndBkg_sigprob_zoom.pdf")
    can.Print(out_fold+"TDJ_DataAndBkg_sigprob_zoom.png")

def plot_met_sf():
    met_file = "/nfs/dust/cms/group/cms-llp/MET_trigger_SF_Caltech/METTriggers_SF.root"
    eras = ["2016","2017","2018"]
    labels = ["trigger_efficiency_Summer16","trigger_efficiency_Fall17","trigger_efficiency_Fall18"]
    lumis = [1,1,1]
    LUMI = 137478.722953

    infile = TFile(met_file,"READ")
    print infile.ls()
    sf = {}

    for i,era in enumerate(eras):
        sf[era] = infile.Get(labels[i])

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    leg = TLegend(0.4+0.2,0.3, 0.65+0.2, 0.55)
    leg.SetTextSize(0.035)

    colors = [1,418,2]
    markers = [24,25,26]
    for i,era in enumerate(eras):
        sf[era].SetTitle("")
        sf[era].SetMarkerColor(colors[i])
        sf[era].SetLineColor(colors[i])
        sf[era].SetLineWidth(2)
        sf[era].SetMarkerStyle(markers[i])
        sf[era].SetMarkerSize(1.25)
        sf[era].GetXaxis().SetTitle("E_{T}^{miss} (GeV)")
        sf[era].GetXaxis().SetTitleSize(0.04)
        sf[era].GetXaxis().SetLabelSize(0.04)
        sf[era].GetYaxis().SetTitle("Data/MC scale factor")
        sf[era].GetYaxis().SetTitleSize(0.04)
        sf[era].GetYaxis().SetLabelSize(0.04)
        sf[era].Draw("PL,sames")
        leg.AddEntry(sf[era],era,"PL")

    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True)
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"met_sf.pdf")
    can.Print(out_fold+"met_sf.png")

    exit()
    for i,era in enumerate(eras):
        tmp_back = infile_ele.Get("b")
        tmp_data = infile_ele.Get("d")
        tmp_back.SetDirectory(0)
        tmp_data.SetDirectory(0)
        if i==0:
            print era
            print "era: clone"
            back = tmp_back.Clone("back")
            data = tmp_data.Clone("data")
            back.SetDirectory(0)
            data.SetDirectory(0)
        else:
            print era
            print "era: add"
            back.Add(tmp_back)
            data.Add(tmp_data)
            back.SetDirectory(0)
            data.SetDirectory(0)
        
    back = back.Rebin(2)
    data = data.Rebin(2)

    back_unc = back.Clone("BkgUnc")
    back_unc.SetFillStyle(3002)
    back_unc.SetFillColor(860)
    back_unc.SetLineWidth(0)
    back_unc.SetMarkerStyle(0)


    back.SetLineColor(860)
    data.SetLineColor(1)
    data.SetMarkerColor(1)
    data.SetMarkerStyle(20)
    back.SetFillColorAlpha(860,0.3)
    back.SetLineWidth(2)
    data.SetLineWidth(2)
    back.GetXaxis().SetTitle("Jet tagger output score")
    back.GetYaxis().SetTitle("Events")
    back.GetXaxis().SetTitleSize(0.04)
    back.GetYaxis().SetTitleSize(0.04)
    back.Draw("HIST,sames")
    back_unc.Draw("E2,sames")
    data.Draw("PE,sames")

def get_tree_weights(ntupledir,sample_list,ERA,main_pred_sample="HighMET"):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
        from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi

    LUMI  = lumi[ main_pred_sample ]["tot"]
    tree_w_dict = defaultdict(dict)
    for i, s in enumerate(sample_list):
        for l, ss in enumerate(samples[s]['files']):
            if ('Run201') in ss:
                t_w = 1.
            else:
                #check if the file exists
                if not os.path.isfile(ntupledir + ss + ".root"):
                    continue
                filename = TFile(ntupledir+ss+'.root', "READ")
                nevents = filename.Get("c_nEvents").GetBinContent(1)
                if ("SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3") in ss:
                    print "SUSY central, consider sample dictionary for nevents!"
                    nevents = sample[ss]['nevents']
                b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                n_pass      = filename.Get("n_pass").GetBinContent(1)
                n_odd       = filename.Get("n_odd").GetBinContent(1)
                filename.Close()
                if('GluGluH2_H2ToSSTobbbb') in ss:
                    xs = 1.
                elif('XXTo4J') in ss:
                    xs = 1.*0.001
                elif('gluinoGMSB') in ss:
                    xs = 1.*0.001
                elif('GluinoGluinoToNeutralinoNeutralinoTo2T2B2S') in ss:
                    xs = 1.*0.001
                elif ('n3n2-n1-hbb-hbb') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('TChiHH') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif ('splitSUSY') in ss:
                    print "Scaling splitSUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                elif('SMS-TChiHZ_ZToQQ_HToBB_LongLivedN2N3') in ss:
                    print "Scaling SUSY to 1. for absolute x-sec sensitivity"
                    xs = 1.
                else:
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                print "LUMI ", LUMI
                print "xs ", xs
                print "nevents ", nevents
                t_w = LUMI * xs / nevents
                if(b_skipTrain>0):
                    if(n_odd>0):
                        t_w *= float(n_pass/n_odd)
            print("%s has tree weight %f")%(ss,t_w)
            tree_w_dict[s][ss] = t_w
    return tree_w_dict

def plot_2D(tree_name="tree",v_x="Jets.sigprob",v_y="min_dPhi_jets",cut="Jets.isGenMatchedCaloCorr",do_radius=False,add_label=""):
    '''plot_2D'''

    data = ["HighMET"]
    dataBH = ["HighMETBH"]
    sign = ['SUSY_mh1000_ctau500_HH']

    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]

    list_samples = [data,sign,dataBH]
    list_cuts = ["isSR && nTagJets_0p996_JJ<2","isSR","isSR && nTagJets_0p996_JJ<2"]
    list_palette = [100,55,104,57]
    legend_xmin = [0.77,0.425,0.45]
    legend_xmax = [0.85,0.85,0.85]
    list_label = ["Data","m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m","Noncollision enriched data"]
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR/"

    nbins_x = 50
    min_x = 0
    max_x = 1.
    nbins_y = 50
    min_y = 0
    max_y = 1.
    for i, samp in enumerate(list_samples):
        chain = {}
        hist = {}
        for s in samp:
            print "Doing ", s
            chain[s] = TChain("tree")
            for p, ss in enumerate(samples[s]['files']):
                chain[s].Add(NTUPLEDIR + ss + ".root")
            hist[s] = TH2F(s, "", nbins_x, min_x, max_x, nbins_y, min_y, max_y)
            hist[s].Sumw2()
            chain[s].Project(s, v_y+":"+v_x, list_cuts[i])

            gStyle.SetPalette(list_palette[i])
            can = TCanvas("can","can",900,800)
            can.cd()
            can.SetGrid()
            can.SetLogz()
            can.SetRightMargin(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.15)            

            print "Is hist[s] empty?"
            print hist[s].GetEntries()

            hist[s].Draw("CONT4Z")#("COLZ")#cont4 or colz
            hist[s].GetYaxis().SetTitle("min #Delta #varphi (jet pairs)")
            hist[s].GetXaxis().SetTitle("Jet tagger output score")
            hist[s].GetXaxis().SetTitleSize(0.04)
            hist[s].GetXaxis().SetTitleOffset(1.3)
            hist[s].GetYaxis().SetTitleSize(0.04)

            if s in sign:
                drawCMS_simple(-1, "Simulation Supplementary", onTop=True,custom_spacing=0.6)
            else:
                drawCMS_simple(LUMI, "Supplementary", onTop=True,custom_spacing=0.35)

            leg = TLegend(legend_xmin[i], 0.8, legend_xmax[i], 0.875)
            leg.SetTextSize(0.04)
            leg.SetHeader(list_label[i])
            leg.SetBorderSize(0)
            leg.SetFillColorAlpha(18,0.8)
            leg.Draw()
            can.Print(out_fold+"TDJ_min_dPhi_correlation_"+s+"_supplementary.pdf")
            can.Print(out_fold+"TDJ_min_dPhi_correlation_"+s+"_supplementary.png")

#plot_s_b(var="sigprob",label="Jet tagger output score",nbins=50,bin_min=0,bin_max=1)
#plot_s_b(var="Jet_cHadEFrac",label="Jet charged hadron energy fraction",nbins=50,bin_min=0,bin_max=1)
#plot_s_b(var="Jet_nHadEFrac",label="Jet neutral hadron energy fraction",nbins=50,bin_min=0,bin_max=1)
#plot_s_b(var="Jet_nTrackConstituents",label="Number of jet charged constituents",nbins=50,bin_min=0,bin_max=50)
#plot_s_b(var="Jet_minDeltaRPVTracks",label="Minimum #DeltaR(jet axis, tracks associated with the PV)",nbins=50,bin_min=0,bin_max=1.2)
#plot_s_b(var="Jet_timeRecHitsEB",label="Jet time (ns)",nbins=50,bin_min=-1,bin_max=8)
#plot_dnn_score()
##plot_dnn_modeling()
#plot_mistag()
#plot_sf_photon()
#plot_sf()
#plot_ntag()
#plot_met_sf()
#compute_ROC_supplementary(ERA="2016")
#compute_ROC_supplementary(ERA="2017")
#compute_ROC_supplementary(ERA="2018")
#tag_eff_vs_ct(ERA="2018")
#plot_ROC_supplementary(ERA="2017",extra_label="_try_all")
#plot_ROC()

plot_2D(tree_name="tree",v_x="Jets.sigprob",v_y="min_dPhi_jets",cut="",do_radius=False,add_label="")
exit()

CUT = "isSR && nTagJets_0p996_JJ<2"
label ="_BH_correlation"
x="Jets.sigprob"
y="min_dPhi_jets"
#plot_2D(sgn,v_x=x,v_y=y,cut=CUT,add_label=label)
