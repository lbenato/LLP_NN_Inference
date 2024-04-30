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
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory, gPad
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
import tensorflow as tf
from tensorflow import keras  
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)
#plt.style.use(hep.style.CMS)

PRELIMINARY = False
out_fold = "fig/Preliminary/" if PRELIMINARY else "fig_updated/"

def correct_round(num,den):
    ratio = 100.*float(num)/float(den)
    digit = Decimal( str(ratio) ).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP)
    return digit

def tau_weight_calc(llp_ct, new_ctau, old_ctau):
    source = np.exp(-1.0*llp_ct/old_ctau)/old_ctau**2
    weight = 1.0/new_ctau**2 * np.exp(-1.0*llp_ct/new_ctau)/source
    return weight

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def plot_ROC(fig_name="",x_sampl=0):

    if x_sampl!=0:
        fig_name += "_sampl"

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"TDJ_ROC"
    else:
        output_string=out_fold+fig_name

    '''ROC'''
    inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    #"/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v5_calo_AOD_2018_dnn_x-check/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    store = pd.HDFStore(inp_file)
    df_test = store.select("df")
    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    fpr, tpr, thresholds = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])
    if x_sampl!=0:
        print "Using less points to plot faster!!!!!"
        fpr, tpr, thresholds = roc_curve(df_test["is_signal"][::x_sampl], df_test["sigprob"][::x_sampl], sample_weight=df_test["EventWeightNormalized"][::x_sampl])
        fig_name += "_sampl"
    cut_fpr = 0.00040904540701505433
    idx, _ = find_nearest(fpr,cut_fpr)
    #FR: go to percentages
    g = TGraph(len(fpr)-1,fpr*100,tpr)
    p = TGraph(1,np.array([fpr[idx]*100]),np.array([tpr[idx]]))
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
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    g.SetTitle("")
    #FR: go to percentages
    g.GetXaxis().SetTitle("False positive rate (%)")
    g.GetYaxis().SetTitle("True positive rate")
    g.GetXaxis().SetTitleSize(0.0425)
    g.GetYaxis().SetTitleSize(0.0425)
    g.SetMarkerColor(2)
    g.SetLineColor(2)
    g.SetLineWidth(4)
    print "Plotting TGraph..."
    g.Draw("ACPL")
    p.SetMarkerStyle(20)
    p.SetMarkerSize(2)#(1.5)
    p.SetMarkerColor(4)
    p.Draw("P,sames")
    #FR comment: CMS Simulation inside the frame --> change maximum
    g.SetMaximum(1.01)
    g.SetMaximum(1.06)
    g.SetMinimum(0.5)
    g.GetXaxis().SetRangeUser(0.00001,0.01)
    #FR: go to percentages
    g.GetXaxis().SetRangeUser(0.00001*100,0.01*100)
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation", onTop=True,draw_s_only=True)
        #Sijin
        #drawCMS_simple(-1, "Simulation", onTop=True,left_marg_CMS=0.225,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        #FR comment: CMS Simulation inside the frame --> change maximum
        drawCMS_simple(-1, "Simulation", onTop=True,left_marg_CMS=0.275,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)
    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)
    #Sijin
    leg.SetTextSize(0.04)
    #CWR comment
    #leg.AddEntry(g,"ROC curve","L")
    leg.AddEntry(g,"TD jet tagger","L")
    leg.AddEntry(p,"DNN > 0.996 w. p.","P")
    leg.SetBorderSize(0)
    leg.Draw()
    gPad.RedrawAxis()
    print "Printing canvas... "
    can.Print(output_string+".pdf")
    can.Print(output_string+".png")
    can.Print(output_string+".C")

    #Root file for hepdata
    out_file = TFile(output_string+".root","RECREATE")
    out_file.cd()
    g.Write("ROC")
    p.Write("threshold")
    print "Writing "+output_string+".root"
    out_file.Write()
    out_file.Close()
    
    print "Using less points to plot faster!!!!!"

    del g
    del p

def plot_ROC_simplified(fig_name=""):

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"TDJ_ROC"
    else:
        output_string=out_fold+fig_name

    '''ROC'''
    inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    #"/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v5_calo_AOD_2018_dnn_x-check/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    store = pd.HDFStore(inp_file)
    df_test = store.select("df")
    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    #Uniform downsampling
    #x_sampling = 100
    #print "Downsampling for HEPData: 1 point every ",x_sampling
    #fpr, tpr, thresholds = roc_curve(df_test["is_signal"][::x_sampling], df_test["sigprob"][::x_sampling], sample_weight=df_test["EventWeightNormalized"][::x_sampling])
    
    #Resample based on efficiency
    fpr, tpr, thresholds = roc_curve(df_test["is_signal"], df_test["sigprob"], sample_weight=df_test["EventWeightNormalized"])

    argsort_fpr = np.argsort(tpr)
    #argsort_fpr = np.argsort(fpr)
    #Reorder tpr according to fpr
    sort_fpr = fpr[argsort_fpr]
    sort_tpr = tpr[argsort_fpr]
    n_split = 1000
    split_fpr = np.array_split(sort_fpr,n_split)
    split_tpr = np.array_split(sort_tpr,n_split)

    s_fpr = []
    s_tpr = []

    for i in range(n_split):
        #print "Chunk ", i
        #print split_fpr[i]
        #print split_fpr[i].shape
        #print split_tpr[i].shape
        s_fpr.append(split_fpr[i][0])
        s_tpr.append(split_tpr[i][0])
    
    #s_fpr = sort_fpr
    #s_tpr = sort_tpr

    #Sample considering point density
    s_fpr = np.array(s_fpr)
    s_tpr = np.array(s_tpr)

    print "Sampled shape ", s_fpr.shape 
    print "fpr"
    print s_fpr
    print "tpr"
    print s_tpr

    #Sample considering interval density


    cut_fpr = 0.00040904540701505433
    idx, _ = find_nearest(s_fpr,cut_fpr)

    g = TGraph(len(s_fpr)-1,s_fpr,s_tpr)
    p = TGraph(1,np.array([s_fpr[idx]]),np.array([s_tpr[idx]]))
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
    p.SetMarkerSize(2)#(1.5)
    p.SetMarkerColor(4)
    p.Draw("P,sames")
    g.SetMaximum(1.01)
    g.SetMinimum(0.5)
    g.GetXaxis().SetRangeUser(0.00001,0.01)
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True,draw_s_only=True)
    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)
    #CWR comment
    #leg.AddEntry(g,"ROC curve","L")
    leg.AddEntry(g,"TD jet tagger","L")
    leg.AddEntry(p,"DNN > 0.996 w. p.","P")
    leg.SetBorderSize(0)
    leg.Draw()
    print "Printing canvas... "
    can.Print(output_string+".pdf")
    can.Print(output_string+".png")
    can.Print(output_string+".C")

    #Root file for hepdata
    out_file = TFile(output_string+".root","RECREATE")
    out_file.cd()
    g.Write("ROC")
    p.Write("threshold")
    print "Writing "+output_string+".root"
    out_file.Write()
    out_file.Close()

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
        #'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
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

    colors = [602,2,418,856,801]
    lines = [1,1,1]
    markers = [20,21,22]

    ctaus_500          = np.array([10, 20, 30, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2150])
    ctaus_3000         = np.array([2150, 2200, 2500, 3000, 4000, 5000, 7000, 10000, 25000, 50000, 100000])
    ctaus = np.unique(np.concatenate((ctaus_500,ctaus_3000)))
    #ctaus = np.array([500,1000,2000,5000])
    #ctaus = np.array([5000])

    h_num = {}
    h_den = {}
    eff = {}

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
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
        eff[s].SetLineStyle(lines[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(markers[i])
        eff[s].GetXaxis().SetTitle("c #tau (m)")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.21)
        eff[s].SetMinimum(-0.01)

        if i ==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")


    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6, 0.2, 0.9, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],"m_{#chi} = "+str(samples[s]['mass'])+" GeV","PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_ct.pdf")
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_ct.png")

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)



def additional_signal_plot(ERA):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',

        'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
    ]

    CUT = "isSR"
    chain = {}
    h_met = {}
    h_radius = {}
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418, 856,801]
    lines = [1,1,1,    2,2,2]
    markers = [20,21,22, 24,25,26]

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)


    #Perform the ctau reweighting
    for i, s in enumerate(sign):
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Gen*",CUT]

        chain[s] = TChain("tree")


        met = np.array([])
        radius = np.array([])
        distance = np.array([])
        weight = np.array([])

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
                #SR cuts, first attempt
                #cut_mask = (arrays[CUT]>0)
                #noSR cuts, first attempt
                cut_mask = (arrays["isMC"]>0)
                #Not done in the original rocs, skip
                #if KILL_QCD:
                #    cut_mask = np.logical_and( cut_mask, arrays["MinJetMetDPhi"]>0.5)

                #Gen-matched jets
                #cut_mask_gen = np.logical_and(cut_mask,arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1)
                #cut_mask = (cut_mask_gen.any()==True)

                #sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                #tagged = (sigprob>0.996)
                tmp_met = arrays["pt"][cut_mask]
                #matched_radius = arrays["Jets.radiusLLP"][cut_mask_gen][cut_mask]
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

                #ctau_weighteight = tau_weight_calc(genPosteriorTime.sum(), float(ct)/10., float( samples[pr]['ctau'] )/10.)
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                tmp_weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                del arrays
                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"

                met = np.concatenate((met,tmp_met))
                weight = np.concatenate((weight,tmp_weight))
                radius = np.concatenate((radius,genLLP0_radius/100.))
                distance = np.concatenate((distance,genLLP0_travel_dist/100.))

        met_bins = np.array([0,50,100,200,300,400,500,600,800,1000,1250,1500,2000])
        met_bins = np.array([0,1000,20000])
        radius_bins = np.array([1,10,100,250,500,1000,2500,5000,10000,100000])/1000.

        #h_met[s] = TH1F(s+"_met",s+"_met",len(met_bins),met_bins)#variable["MEt.pt"]["nbins"],variable["MEt.pt"]["min"],variable["MEt.pt"]["max"])
        h_met[s] = TH1F(s+"_met",s+"_met",10,variable["MEt.pt"]["min"],variable["MEt.pt"]["max"])#len(met_bins)-1,met_bins)#
        h_radius[s] = TH1F(s+"_radius",s+"_radius",len(radius_bins)-1,radius_bins)
        root_numpy.fill_hist(h_met[s],met,weight)
        print "check: ", s, h_met[s].Integral()
        root_numpy.fill_hist(h_radius[s],radius,weight)


    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    leg = TLegend(0.6-0.05, 0.2+0.35, 0.9-0.05, 0.45+0.35)
    leg.SetTextSize(0.03)#5)

    i=0
    for s in sign:
        h_met[s].SetTitle("")
        h_met[s].SetMarkerColor(colors[i])
        h_met[s].SetLineColor(colors[i])
        h_met[s].SetLineStyle(lines[i])
        h_met[s].SetLineWidth(2)
        h_met[s].SetMarkerSize(1.4)
        h_met[s].SetMarkerStyle(markers[i])
        h_met[s].GetXaxis().SetTitle(variable["MEt.pt"]["title"])
        #h_met[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        h_met[s].GetYaxis().SetTitle("Events / bin")
        h_met[s].GetXaxis().SetTitleSize(0.04)
        h_met[s].GetXaxis().SetTitleOffset(1.3)
        h_met[s].GetYaxis().SetTitleSize(0.04)
        h_met[s].Scale(1./h_met[s].Integral())
        h_met[s].SetMaximum(500.01)
        h_met[s].SetMinimum(1.e-6)

        #leg.AddEntry(h_met[s],"m_{#chi} = "+str(samples[s]['mass'])+" GeV","PL")
        leg.AddEntry(h_met[s],samples[s]["label"],"PL")

        if i ==0:
            h_met[s].Draw("P")
        else:
            h_met[s].Draw("P,sames")
        i+=1

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)

    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/MET.pdf")
    can.Print(out_fold+"/Supplementary/MET.png")


    can.Clear()

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetLogx()
    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    leg = TLegend(0.6-0.4, 0.2, 0.9-0.4, 0.45)
    leg.SetTextSize(0.03)#5)

    i=0
    for s in sign:
        h_radius[s].SetTitle("")
        h_radius[s].SetMarkerColor(colors[i])
        h_radius[s].SetLineColor(colors[i])
        h_radius[s].SetLineStyle(lines[i])
        h_radius[s].SetLineWidth(2)
        h_radius[s].SetMarkerSize(1.4)
        h_radius[s].SetMarkerStyle(markers[i])
        h_radius[s].GetXaxis().SetTitle(variable["radius"]["title"])
        #h_radius[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        h_radius[s].GetYaxis().SetTitle("Events / bin")
        h_radius[s].GetXaxis().SetTitleSize(0.04)
        h_radius[s].GetXaxis().SetTitleOffset(1.3)
        h_radius[s].GetYaxis().SetTitleSize(0.04)
        h_radius[s].Scale(1./h_radius[s].Integral())
        h_radius[s].SetMaximum(5.01)
        h_radius[s].SetMinimum(1.e-6)
        #h_radius[s].SetMinimum(-0.01)

        #leg.AddEntry(h_radius[s],"m_{#chi} = "+str(samples[s]['mass'])+" GeV","PL")
        leg.AddEntry(h_radius[s],samples[s]["label"],"PL")

        if i ==0:
            h_radius[s].Draw("P")
        else:
            h_radius[s].Draw("P,sames")
        i+=1

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)

    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/radius.pdf")
    can.Print(out_fold+"/Supplementary/radius.png")

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)


def tag_eff_vs_eta(ERA,ntupledir,cut="isMC",postfix=""):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
    ]

    CUT = cut
    #CUT = "isSR"
    chain = {}
    NTUPLEDIR = ntupledir%ERA#
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR_Gen = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418, 856,801]
    lines = [1,1,1,    2,2,2]
    markers = [20,21,22, 24,25,26]

    h_num = {}
    h_den = {}
    eff = {}

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    #can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    #Perform the ctau reweighting
    for i, s in enumerate(sign):

        den_val = np.array([])
        num_val = np.array([])
        den_w = np.array([])
        num_w = np.array([])

        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Gen*","Jets.eta",CUT]

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

                #Signal: consider only gen-matched jets in fiducial volume
                cut_mask_gen = np.logical_and( np.logical_and(arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1, np.logical_and(arrays["Jets.eta"]>-1.,arrays["Jets.eta"]<1.))  , cut_mask)
                cut_mask = np.logical_and((cut_mask_gen.any()==True),cut_mask)

                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                pt = arrays["Jets.pt"][cut_mask_gen][cut_mask]
                eta = arrays["Jets.eta"][cut_mask_gen][cut_mask]
                dummy = (pt>-1.)
                tagged = np.logical_and(dummy,(sigprob>0.996))

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                den_w = np.concatenate(( den_w, np.multiply(dummy, weight)))
                num_w = np.concatenate(( num_w, np.multiply(tagged, weight) ))

                den_val = np.concatenate(( den_val, np.multiply(dummy, eta ) ))
                num_val = np.concatenate(( num_val, np.multiply(tagged, eta ) ))

                del arrays

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        #Throw these vectors to a histogram
        bins = np.array([-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])

        h_num[s] = TH1F(s+"_num",s+"_num",len(bins)-1,bins)
        h_den[s] = TH1F(s+"_den",s+"_den",len(bins)-1,bins)

        arr_num = np.array([])
        arr_den = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])
        
        for row in num_val:
            if len(row)>0:
                arr_num = np.concatenate(( arr_num, np.stack(row)  ))

        for row in num_w:
            if len(row)>0:
                arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))

        for row in den_val:
            if len(row)>0:
                arr_den = np.concatenate(( arr_den, np.stack(row)  ))

        for row in den_w:
            if len(row)>0:
                arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))

        root_numpy.fill_hist(h_num[s],arr_num,arr_w_num)
        root_numpy.fill_hist(h_den[s],arr_den,arr_w_den)

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(h_num[s],h_den[s])

        eff[s].SetTitle("")
        eff[s].SetMarkerColor(colors[i])
        eff[s].SetLineColor(colors[i])
        eff[s].SetLineStyle(lines[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(markers[i])
        eff[s].GetXaxis().SetTitle("Jet #eta")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.21)
        eff[s].SetMinimum(-0.01)

        if i ==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")


    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_eta"+postfix+".pdf")
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_eta"+postfix+".png")

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)


def tag_eff_vs_pt(ERA,ntupledir,cut="isMC",postfix=""):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
    ]

    CUT = cut
    #CUT = "isSR"
    chain = {}
    NTUPLEDIR = ntupledir%ERA#
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR_Gen = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418, 856,801]
    lines = [1,1,1,    2,2,2]
    markers = [20,21,22, 24,25,26]

    h_num = {}
    h_den = {}
    eff = {}

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetLogx()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    #Perform the ctau reweighting
    for i, s in enumerate(sign):

        den_val = np.array([])
        num_val = np.array([])
        den_w = np.array([])
        num_w = np.array([])

        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Gen*","Jets.eta",CUT]

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

                #Signal: consider only gen-matched jets in fiducial volume
                cut_mask_gen = np.logical_and( np.logical_and(arrays["Jets.isGenMatchedCaloCorrLLPAccept"]==1, np.logical_and(arrays["Jets.eta"]>-1.,arrays["Jets.eta"]<1.))  , cut_mask)
                cut_mask = np.logical_and((cut_mask_gen.any()==True),cut_mask)

                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                pt = arrays["Jets.pt"][cut_mask_gen][cut_mask]
                eta = arrays["Jets.eta"][cut_mask_gen][cut_mask]
                dummy = (pt>-1.)
                tagged = np.logical_and(dummy,(sigprob>0.996))

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                den_w = np.concatenate(( den_w, np.multiply(dummy, weight)))
                num_w = np.concatenate(( num_w, np.multiply(tagged, weight) ))

                den_val = np.concatenate(( den_val, np.multiply(dummy, pt ) ))
                num_val = np.concatenate(( num_val, np.multiply(tagged, pt ) ))

                del arrays

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        #Throw these vectors to a histogram
        bins = np.array([30.,40.,50.,75.,100.,150.,250.,500.,1000.])
        #Less bins as in parametrization
        bins = np.array([30.,40.,50.,75.,100.,200.,500.,1000.])

        h_num[s] = TH1F(s+"_num",s+"_num",len(bins)-1,bins)
        h_den[s] = TH1F(s+"_den",s+"_den",len(bins)-1,bins)

        arr_num = np.array([])
        arr_den = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])
        
        for row in num_val:
            if len(row)>0:
                arr_num = np.concatenate(( arr_num, np.stack(row)  ))

        for row in num_w:
            if len(row)>0:
                arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))

        for row in den_val:
            if len(row)>0:
                arr_den = np.concatenate(( arr_den, np.stack(row)  ))

        for row in den_w:
            if len(row)>0:
                arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))

        root_numpy.fill_hist(h_num[s],arr_num,arr_w_num)
        root_numpy.fill_hist(h_den[s],arr_den,arr_w_den)

        addOverflow(h_den[s])
        addOverflow(h_num[s])

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(h_num[s],h_den[s])

        eff[s].SetTitle("")
        eff[s].SetMarkerColor(colors[i])
        eff[s].SetLineColor(colors[i])
        eff[s].SetLineStyle(lines[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(markers[i])
        eff[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.21)
        eff[s].SetMinimum(-0.01)

        if i ==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")


    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_pt"+postfix+".pdf")
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_pt"+postfix+".png")

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)


def tag_eff_vs_topology(ERA,ntupledir,cut="isMC",postfix=""):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        'SUSY_mh1800_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
    ]

    CUT = cut
    #CUT = "isSR"
    chain = {}
    NTUPLEDIR = ntupledir%ERA#
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR_Gen = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418, 856,801]
    lines = [1,1,1,    2,2,2,  3,3]
    markers = [20,21,22, 24,25,26, 27,27]

    h_num_pt = {}
    h_den_pt = {}
    eff_pt = {}

    h_num_z = {}
    h_den_z = {}
    eff_z = {}

    h_num_r = {}
    h_den_r = {}
    eff_r = {}


    c_arr_num_pt = np.array([])
    c_arr_den_pt = np.array([])
    c_arr_num_r = np.array([])
    c_arr_den_r = np.array([])
    c_arr_num_z = np.array([])
    c_arr_den_z = np.array([])
    c_arr_w_num = np.array([])
    c_arr_w_den = np.array([])


    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)


    #Perform the ctau reweighting
    for i, s in enumerate(sign):

        den_val_pt = np.array([])
        num_val_pt = np.array([])
        den_val_z = np.array([])
        num_val_z = np.array([])
        den_val_r = np.array([])
        num_val_r = np.array([])
        den_w = np.array([])
        num_w = np.array([])

        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Jets.zLLP","Gen*","Jets.eta",CUT]

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

                #Signal: consider only gen-matched jets, consider all decay volume
                cut_mask_gen = np.logical_and( np.logical_and(arrays["Jets.isGenMatchedCaloCorr"]==1, np.logical_and(arrays["Jets.eta"]>-1.,arrays["Jets.eta"]<1.))  , cut_mask)
                cut_mask = np.logical_and((cut_mask_gen.any()==True),cut_mask)

                #print " W A R N I N G!!! Looking only at merged events!!!"
                print " W A R N I N G!!! Looking only at resolved events!!!"
                #tmp_genLLP0_topo = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"] )))[0:1] < 0.4).flatten()
                #tmp_genLLP1_topo = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"] )))[1:2] < 0.4).flatten()
                tmp_genLLP0_topo = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"] )))[0:1] >= 0.4).flatten()
                tmp_genLLP1_topo = (np.transpose(np.stack(( arrays["GenHiggs.dRdaughters"] )))[1:2] >= 0.4).flatten()


                tmp_genLLP0_radius = np.multiply(np.transpose(np.stack(( arrays["GenLLPs.travelRadius"] )))[0:1].flatten(),tmp_genLLP0_topo)
                tmp_genLLP1_radius = np.multiply(np.transpose(np.stack(( arrays["GenLLPs.travelRadius"] )))[1:2].flatten(),tmp_genLLP1_topo)
                
                tmp_jet_radius_mask_LLP0 = (arrays["Jets.radiusLLP"]==tmp_genLLP0_radius)
                tmp_jet_radius_mask_LLP1 = (arrays["Jets.radiusLLP"]==tmp_genLLP1_radius)

                jet_mask = np.logical_and(np.logical_or(tmp_jet_radius_mask_LLP0,tmp_jet_radius_mask_LLP1) , cut_mask_gen)
                event_mask = np.logical_and((jet_mask.any()==True),cut_mask)

                sigprob = arrays["Jets.sigprob"][jet_mask][event_mask]
                pt = arrays["Jets.pt"][jet_mask][event_mask]
                z = abs(arrays["Jets.zLLP"][jet_mask][event_mask]/100.)
                radius = arrays["Jets.radiusLLP"][jet_mask][event_mask]/100.
                eta = arrays["Jets.eta"][jet_mask][event_mask]
                dummy = (pt>-1.)
                tagged = np.logical_and(dummy,(sigprob>0.996))

                eventweight = arrays["EventWeight"][event_mask]
                pureweight = arrays["PUReWeight"][event_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                den_w = np.concatenate(( den_w, np.multiply(dummy, weight)))
                num_w = np.concatenate(( num_w, np.multiply(tagged, weight) ))

                den_val_pt = np.concatenate(( den_val_pt, np.multiply(dummy, pt ) ))
                num_val_pt = np.concatenate(( num_val_pt, np.multiply(tagged, pt ) ))
                den_val_r = np.concatenate(( den_val_r, np.multiply(dummy, radius ) ))
                num_val_r = np.concatenate(( num_val_r, np.multiply(tagged, radius ) ))
                den_val_z = np.concatenate(( den_val_z, np.multiply(dummy, z ) ))
                num_val_z = np.concatenate(( num_val_z, np.multiply(tagged, z ) ))

                del arrays

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        #Throw these vectors to a histogram
        bins = np.array([30.,40.,50.,75.,100.,150.,250.,500.,1000.])
        #Less bins as in parametrization
        bins_pt_resolved = np.array([30.,40.,50.,75.,100.,200.,500.,1000.])
        bins_pt_merged_more = np.array([30.,100.,150.,300.,600.,800.,1000.,2500.,5000.])
        bins_pt_merged = np.array([30.,200.,1000.,5000.])
        c_bins_pt_merged = np.array([30.,550.,800.,5000.])
        bins_pt = c_bins_pt_merged#bins_pt_merged_more#np.array([30.,40.,50.,75.,100.,200.,500.,1000.,5000.])
        #be inclusive!!!
        decent_bins_pt = np.logspace(0,4,20,endpoint=True)
        #reducing them:
        decent_bins_pt = np.array([30., 40., 70., 120., 200., 300., 500.,1.00000000e+04,])
        #optimized for merged
        chosen_bins_pt_merged = np.array([30.,50.,120.,180.,300.,500.,800.,1300.,2000.,5000.])
        bins_pt = decent_bins_pt

        bins_r_resolved = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 1.7, 2., 10.])
        bins_r_merged = np.array([0.01, 0.1, 0.2, 0.5, 0.8, 1.2, 10.])
        bins_r = bins_r_merged
        #be inclusive!!!
        decent_bins_r = np.logspace(-2,1,20,endpoint=True)
        #optimized for merged
        chosen_bins_r_merged = np.array([0.001,0.02,0.05,0.1,0.2,0.5,1.,1.5,10.])
        bins_r = chosen_bins_r_merged#decent_bins_r
        
        bins_z_resolved = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1., 1.5, 2., 3., 5., 10.])
        bins_z_merged = np.array([0.01, 0.1, 0.2, 0.5, 1., 2., 10.])
        bins_z = bins_z_merged
        #be inclusive!!!
        decent_bins_z = np.logspace(-2,1,20,endpoint=True)
        #optimized for merged
        chosen_bins_z_merged = np.array([0.001,0.05,0.1,0.2,1.,2.,100.])
        bins_z = chosen_bins_z_merged#decent_bins_z

        h_num_pt[s] = TH1F(s+"_num_pt",s+"_num_pt",len(bins_pt)-1,bins_pt)
        h_den_pt[s] = TH1F(s+"_den_pt",s+"_den_pt",len(bins_pt)-1,bins_pt)
        h_num_r[s] = TH1F(s+"_num_r",s+"_num_r",len(bins_r)-1,bins_r)
        h_den_r[s] = TH1F(s+"_den_r",s+"_den_r",len(bins_r)-1,bins_r)
        h_num_z[s] = TH1F(s+"_num_z",s+"_num_z",len(bins_z)-1,bins_z)
        h_den_z[s] = TH1F(s+"_den_z",s+"_den_z",len(bins_z)-1,bins_z)

        arr_num_pt = np.array([])
        arr_den_pt = np.array([])
        arr_num_r = np.array([])
        arr_den_r = np.array([])
        arr_num_z = np.array([])
        arr_den_z = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])
        
        for row in num_w:
            if len(row)>0:
                arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))
                c_arr_w_num = np.concatenate(( c_arr_w_num, np.stack(row)  ))

        for row in den_w:
            if len(row)>0:
                arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))
                c_arr_w_den = np.concatenate(( c_arr_w_den, np.stack(row)  ))

        for row in num_val_pt:
            if len(row)>0:
                arr_num_pt = np.concatenate(( arr_num_pt, np.stack(row)  ))
                c_arr_num_pt = np.concatenate(( c_arr_num_pt, np.stack(row)  ))

        for row in den_val_pt:
            if len(row)>0:
                arr_den_pt = np.concatenate(( arr_den_pt, np.stack(row)  ))
                c_arr_den_pt = np.concatenate(( c_arr_den_pt, np.stack(row)  ))

        for row in num_val_r:
            if len(row)>0:
                arr_num_r = np.concatenate(( arr_num_r, np.stack(row)  ))
                c_arr_num_r = np.concatenate(( c_arr_num_r, np.stack(row)  ))

        for row in den_val_r:
            if len(row)>0:
                arr_den_r = np.concatenate(( arr_den_r, np.stack(row)  ))
                c_arr_den_r = np.concatenate(( c_arr_den_r, np.stack(row)  ))

        for row in num_val_z:
            if len(row)>0:
                arr_num_z = np.concatenate(( arr_num_z, np.stack(row)  ))
                c_arr_num_z = np.concatenate(( c_arr_num_z, np.stack(row)  ))

        for row in den_val_z:
            if len(row)>0:
                arr_den_z = np.concatenate(( arr_den_z, np.stack(row)  ))
                c_arr_den_z = np.concatenate(( c_arr_den_z, np.stack(row)  ))


        root_numpy.fill_hist(h_num_pt[s],arr_num_pt,arr_w_num)
        root_numpy.fill_hist(h_den_pt[s],arr_den_pt,arr_w_den)
        root_numpy.fill_hist(h_num_r[s],arr_num_r,arr_w_num)
        root_numpy.fill_hist(h_den_r[s],arr_den_r,arr_w_den)
        root_numpy.fill_hist(h_num_z[s],arr_num_z,arr_w_num)
        root_numpy.fill_hist(h_den_z[s],arr_den_z,arr_w_den)

        addOverflow(h_den_pt[s])
        addOverflow(h_num_pt[s])
        addOverflow(h_den_r[s])
        addOverflow(h_num_r[s])
        addOverflow(h_den_z[s])
        addOverflow(h_num_z[s])

    n_splits = 4
    print "Global split: ", n_splits
    print "pt den"
    sorted_c_arr_den_pt = np.sort(c_arr_den_pt)
    print np.array_split(sorted_c_arr_den_pt,n_splits)[0]
    print np.array_split(sorted_c_arr_den_pt,n_splits)[1]
    print np.array_split(sorted_c_arr_den_pt,n_splits)[2]
    print np.array_split(sorted_c_arr_den_pt,n_splits)[3]
    print "pt num"
    sorted_c_arr_num_pt = np.sort(c_arr_num_pt)
    print np.array_split(sorted_c_arr_num_pt,n_splits)[0]
    print np.array_split(sorted_c_arr_num_pt,n_splits)[1]
    print np.array_split(sorted_c_arr_num_pt,n_splits)[2]
    print np.array_split(sorted_c_arr_num_pt,n_splits)[3]


    n_splits = 4
    print "Global split: ", n_splits
    print "r den"
    sorted_c_arr_den_r = np.sort(c_arr_den_r)
    print np.array_split(sorted_c_arr_den_r,n_splits)[0]
    print np.array_split(sorted_c_arr_den_r,n_splits)[1]
    print np.array_split(sorted_c_arr_den_r,n_splits)[2]
    print np.array_split(sorted_c_arr_den_r,n_splits)[3]
    print "r num"
    sorted_c_arr_num_r = np.sort(c_arr_num_r)
    print np.array_split(sorted_c_arr_num_r,n_splits)[0]
    print np.array_split(sorted_c_arr_num_r,n_splits)[1]
    print np.array_split(sorted_c_arr_num_r,n_splits)[2]
    print np.array_split(sorted_c_arr_num_r,n_splits)[3]

    n_splits = 4
    print "Global split: ", n_splits
    print "z den"
    sorted_c_arr_den_z = np.sort(c_arr_den_z)
    print np.array_split(sorted_c_arr_den_z,n_splits)[0]
    print np.array_split(sorted_c_arr_den_z,n_splits)[1]
    print np.array_split(sorted_c_arr_den_z,n_splits)[2]
    print np.array_split(sorted_c_arr_den_z,n_splits)[3]
    print "z num"
    sorted_c_arr_num_z = np.sort(c_arr_num_z)
    print np.array_split(sorted_c_arr_num_z,n_splits)[0]
    print np.array_split(sorted_c_arr_num_z,n_splits)[1]
    print np.array_split(sorted_c_arr_num_z,n_splits)[2]
    print np.array_split(sorted_c_arr_num_z,n_splits)[3]
    

    #cumulative arrays and histos
    c_h_num_pt = TH1F("num_pt","num_pt",len(bins_pt)-1,bins_pt)
    c_h_den_pt = TH1F("den_pt","den_pt",len(bins_pt)-1,bins_pt)
    c_h_num_r = TH1F("num_r","num_r",len(bins_r)-1,bins_r)
    c_h_den_r = TH1F("den_r","den_r",len(bins_r)-1,bins_r)
    c_h_num_z = TH1F("num_z","num_z",len(bins_z)-1,bins_z)
    c_h_den_z = TH1F("den_z","den_z",len(bins_z)-1,bins_z)

    root_numpy.fill_hist(c_h_num_pt,c_arr_num_pt,c_arr_w_num)
    root_numpy.fill_hist(c_h_den_pt,c_arr_den_pt,c_arr_w_den)
    root_numpy.fill_hist(c_h_num_r,c_arr_num_r,c_arr_w_num)
    root_numpy.fill_hist(c_h_den_r,c_arr_den_r,c_arr_w_den)
    root_numpy.fill_hist(c_h_num_z,c_arr_num_z,c_arr_w_num)
    root_numpy.fill_hist(c_h_den_z,c_arr_den_z,c_arr_w_den)

    addOverflow(c_h_den_pt)
    addOverflow(c_h_num_pt)
    addOverflow(c_h_den_r)
    addOverflow(c_h_num_r)
    addOverflow(c_h_den_z)
    addOverflow(c_h_num_z)


    can_pt = TCanvas("can_pt","can_pt",900,800)
    can_pt.cd()
    #can_pt.SetGrid()
    can_pt.SetLogx()
    can_pt.SetRightMargin(0.05)
    can_pt.SetLeftMargin(0.12)
    can_pt.SetBottomMargin(0.15)

    i=0
    for s in sign:
        eff_pt[s] = TGraphAsymmErrors()
        eff_pt[s].BayesDivide(h_num_pt[s],h_den_pt[s])
        #h_num_pt[s].Scale(1./h_num_pt[s].Integral())
        #eff_pt[s] = TGraphAsymmErrors(h_num_pt[s])

        eff_pt[s].SetTitle("")
        eff_pt[s].SetMarkerColor(colors[i])
        eff_pt[s].SetLineColor(colors[i])
        eff_pt[s].SetLineStyle(lines[i])
        eff_pt[s].SetLineWidth(2)
        eff_pt[s].SetMarkerSize(1.4)
        eff_pt[s].SetMarkerStyle(markers[i])
        eff_pt[s].GetXaxis().SetTitle("Jet p_{T} (GeV)")
        #eff_pt[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff_pt[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff_pt[s].GetXaxis().SetTitleSize(0.04)
        eff_pt[s].GetXaxis().SetTitleOffset(1.3)
        eff_pt[s].GetYaxis().SetTitleSize(0.04)
        eff_pt[s].SetMaximum(1.21)
        eff_pt[s].SetMinimum(-0.01)
        if i ==0:
            eff_pt[s].Draw("AP")
        else:
            eff_pt[s].Draw("P,sames")
        i+=1

    c_eff_pt = TGraphAsymmErrors()
    c_eff_pt.BayesDivide(c_h_num_pt,c_h_den_pt)
    c_eff_pt.SetMarkerColor(1)
    c_eff_pt.SetLineColor(1)
    c_eff_pt.SetLineStyle(1)
    c_eff_pt.SetLineWidth(2)
    c_eff_pt.SetMarkerSize(1.5)
    c_eff_pt.SetMarkerStyle(20)
    c_eff_pt.Draw("P,sames")

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff_pt[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    #leg.Draw()
    can_pt.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_pt"+postfix+".pdf")
    can_pt.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_pt"+postfix+".png")
    can_pt.Close()



    can_r = TCanvas("can_r","can_r",900,800)
    can_r.cd()
    #can_r.SetGrid()
    can_r.SetLogx()
    can_r.SetRightMargin(0.05)
    can_r.SetLeftMargin(0.12)
    can_r.SetBottomMargin(0.15)
    i=0
    for s in sign:
        eff_r[s] = TGraphAsymmErrors()
        eff_r[s].BayesDivide(h_num_r[s],h_den_r[s])
        #h_num_r[s].Scale(1./h_num_r[s].Integral())
        #eff_r[s] = TGraphAsymmErrors(h_num_r[s])

        eff_r[s].SetTitle("")
        eff_r[s].SetMarkerColor(colors[i])
        eff_r[s].SetLineColor(colors[i])
        eff_r[s].SetLineStyle(lines[i])
        eff_r[s].SetLineWidth(2)
        eff_r[s].SetMarkerSize(1.4)
        eff_r[s].SetMarkerStyle(markers[i])
        eff_r[s].GetXaxis().SetTitle("LLP transverse decay length (m)")
        #eff_r[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff_r[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff_r[s].GetXaxis().SetTitleSize(0.04)
        eff_r[s].GetXaxis().SetTitleOffset(1.3)
        eff_r[s].GetYaxis().SetTitleSize(0.04)
        eff_r[s].SetMaximum(1.21)
        eff_r[s].SetMinimum(-0.01)
        if i ==0:
            eff_r[s].Draw("AP")
        else:
            eff_r[s].Draw("P,sames")
        i+=1

    c_eff_r = TGraphAsymmErrors()
    c_eff_r.BayesDivide(c_h_num_r,c_h_den_r)
    c_eff_r.SetMarkerColor(1)
    c_eff_r.SetLineColor(1)
    c_eff_r.SetLineStyle(1)
    c_eff_r.SetLineWidth(2)
    c_eff_r.SetMarkerSize(1.5)
    c_eff_r.SetMarkerStyle(20)
    c_eff_r.Draw("P,sames")

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff_r[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    #leg.Draw()
    can_r.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_r"+postfix+".pdf")
    can_r.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_r"+postfix+".png")
    can_r.Close()


    can_z = TCanvas("can_z","can_z",900,800)
    can_z.cd()
    #can_z.SetGrid()
    can_z.SetLogx()
    can_z.SetRightMargin(0.05)
    can_z.SetLeftMargin(0.12)
    can_z.SetBottomMargin(0.15)

    i=0
    for s in sign:
        eff_z[s] = TGraphAsymmErrors()
        eff_z[s].BayesDivide(h_num_z[s],h_den_z[s])
        #h_num_z[s].Scale(1./h_num_z[s].Integral())
        #eff_z[s] = TGraphAsymmErrors(h_num_z[s])

        eff_z[s].SetTitle("")
        eff_z[s].SetMarkerColor(colors[i])
        eff_z[s].SetLineColor(colors[i])
        eff_z[s].SetLineStyle(lines[i])
        eff_z[s].SetLineWidth(2)
        eff_z[s].SetMarkerSize(1.4)
        eff_z[s].SetMarkerStyle(markers[i])
        eff_z[s].GetXaxis().SetTitle("|LLP longitudinal decay length| (m)")
        #eff_z[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff_z[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff_z[s].GetXaxis().SetTitleSize(0.04)
        eff_z[s].GetXaxis().SetTitleOffset(1.3)
        eff_z[s].GetYaxis().SetTitleSize(0.04)
        eff_z[s].SetMaximum(1.21)
        eff_z[s].SetMinimum(-0.01)
        if i ==0:
            eff_z[s].Draw("AP")
        else:
            eff_z[s].Draw("P,sames")
        i+=1

    c_eff_z = TGraphAsymmErrors()
    c_eff_z.BayesDivide(c_h_num_z,c_h_den_z)
    c_eff_z.SetMarkerColor(1)
    c_eff_z.SetLineColor(1)
    c_eff_z.SetLineStyle(1)
    c_eff_z.SetLineWidth(2)
    c_eff_z.SetMarkerSize(1.5)
    c_eff_z.SetMarkerStyle(20)
    c_eff_z.Draw("P,sames")

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff_z[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    #leg.Draw()
    can_z.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_z"+postfix+".pdf")
    can_z.Print(out_fold+"/Supplementary/TDJ_topology_eff_vs_z"+postfix+".png")
    can_z.Close()

    print "---"
    print "bins_pt"
    print bins_pt
    print "bins_r"
    print bins_r
    print "bins_z"
    print bins_z


def tag_eff_vs_radius(ERA,ntupledir,cut="isMC",postfix=""):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        #'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        #'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        #'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        #'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        #'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        #'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        #'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
        #'SUSY_mh1800_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        #'SUSY_mh1800_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',

    ]

    CUT = cut
    #CUT = "isSR"
    chain = {}
    NTUPLEDIR = ntupledir%ERA#
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR_Gen = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418,    856, 801]

    colors = [2,602,418,  602,2,418,    856, 801]
    lines = [1,1,1,  2,2,2,               3,   3]
    markers = [20,21,22, 24,25,26,       23,  23]

    h_num = {}
    h_den = {}
    eff = {}
    eff_val_x = {}
    eff_val_y = {}
    eff_val_y_unc_low = {}
    eff_val_y_unc_high = {}

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetLogx()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    #Perform the ctau reweighting
    for i, s in enumerate(sign):

        den_val = np.array([])
        num_val = np.array([])
        den_w = np.array([])
        num_w = np.array([])


        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Gen*","Jets.eta",CUT]

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
                cut_mask_gen = np.logical_and(arrays["Jets.isGenMatchedCaloCorr"]==1, cut_mask)
                cut_mask = np.logical_and((cut_mask_gen.any()==True),cut_mask)

                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                pt = arrays["Jets.pt"][cut_mask_gen][cut_mask]
                radius = arrays["Jets.radiusLLP"][cut_mask_gen][cut_mask]/100
                dummy = (pt>-1.)
                tagged = np.logical_and(dummy,(sigprob>0.996))

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                den_w = np.concatenate(( den_w, np.multiply(dummy, weight)))
                num_w = np.concatenate(( num_w, np.multiply(tagged, weight) ))

                den_val = np.concatenate(( den_val, np.multiply(dummy, radius ) ))
                num_val = np.concatenate(( num_val, np.multiply(tagged, radius ) ))

                del arrays

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        #Throw these vectors to a histogram
        radius_bins = np.array([1,10,100,250,500,1000,2500,5000,10000,100000])/1000.
        radius_bins_orig = np.array([0.01,0.1,0.25,0.5,1,3.,100.])
        radius_bins = np.logspace(-2,2,200,endpoint=True)
        bins_scan_points = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1., 1.2, 1.5, 2., 3., 10.])
        bins_scan_points = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1., 1.2, 1.5, 1.7, 2., 3., 10.])

        bins_scan_points = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 1.7, 2., 10.])
        bins = radius_bins#np.array([0.01,0.1,0.25,0.5,1,2.5,5.,10.,100.])
        bins = bins_scan_points
        bins = radius_bins_orig

        h_num[s] = TH1F(s+"_num",s+"_num",len(bins)-1,bins)
        h_den[s] = TH1F(s+"_den",s+"_den",len(bins)-1,bins)

        arr_num = np.array([])
        arr_den = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])

        eff_val_x[s] = []
        eff_val_y[s] = []
        eff_val_y_unc_low[s] = []
        eff_val_y_unc_high[s] = []
        
        for row in num_val:
            if len(row)>0:
                arr_num = np.concatenate(( arr_num, np.stack(row)  ))

        for row in num_w:
            if len(row)>0:
                arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))

        for row in den_val:
            if len(row)>0:
                arr_den = np.concatenate(( arr_den, np.stack(row)  ))

        for row in den_w:
            if len(row)>0:
                arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))

        root_numpy.fill_hist(h_num[s],arr_num,arr_w_num)
        root_numpy.fill_hist(h_den[s],arr_den,arr_w_den)

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(h_num[s],h_den[s])
        #Show only den to check if it makes sense
        #h_den[s].Scale(1./h_den[s].Integral())
        #h_num[s].Scale(1./h_num[s].Integral())
        #eff[s] = TGraphAsymmErrors(h_den[s])
        #eff[s] = TGraphAsymmErrors(h_num[s])

        #arrays to search for efficiency point
        npts = eff[s].GetN()
        for ipt in range(0,npts):
            eff_val_x[s].append(eff[s].GetPointX(ipt))
            eff_val_y[s].append(eff[s].GetPointY(ipt))
            eff_val_y_unc_low[s].append(abs(eff[s].GetErrorYlow(ipt))/eff[s].GetPointY(ipt) if eff[s].GetPointY(ipt)>0 else 0 )
            eff_val_y_unc_high[s].append(abs(eff[s].GetErrorYhigh(ipt))/eff[s].GetPointY(ipt) if eff[s].GetPointY(ipt)>0 else 0)

        print "\n"
        print s
        print eff_val_x[s]
        print eff_val_y[s]
        print "unc low percentage"
        print eff_val_y_unc_low[s]
        print "unc high percentage"
        print eff_val_y_unc_high[s]
        print "\n"

        #There we have the vectors, now build the average

    y_avg = np.array([])
    for i, s in enumerate(sign):
        if i==0:
            y_avg = np.array(eff_val_y[s])
        else:
            y_avg += np.array(eff_val_y[s])

    y_avg = 0.5*y_avg
    y_unc = np.abs(y_avg-eff_val_y[sign[0]])
    y_stat_low = np.zeros(y_unc.shape[0])
    y_stat_high = np.zeros(y_unc.shape[0])

    for i in range(y_avg.shape[0]):
        max_low = 0.
        max_high = 0.
        for s in sign:
            max_low = max(max_low, y_avg[i]*abs(eff_val_y_unc_low[s][i]) )
            max_high = max(max_high, y_avg[i]*abs(eff_val_y_unc_high[s][i]) )
        y_stat_low[i] = max_low
        y_stat_high[i] = max_high


    print "low/high percentage should be the largest"
    print y_avg
    print "kinda syst"
    print y_unc
    print y_stat_low
    print y_stat_high
    
    e = eff[ sign[0] ].Clone("eff")
    for ipt in range(0,npts):
        #.SetPointX(eff_val_x[sign[0]][ipt])
        e.SetPointY(ipt,y_avg[ipt])
        e.SetPointEYhigh(ipt, math.sqrt(y_stat_high[ipt]**2 + y_unc[ipt]**2) )#)y_stat_high[ipt]+y_unc[ipt])#
        e.SetPointEYlow(ipt, math.sqrt(y_stat_low[ipt]**2 + y_unc[ipt]**2) )#)y_stat_low[ipt]+y_unc[ipt])#
    '''
    for i in range(y_avg.shape[0]):
        max_diff = 0.
        for s in sign:
            max_diff = max(max_diff, abs(eff_val_y[s][i] - y_avg[i]))
        y_unc[i] = max_diff
    '''

    for i, s in enumerate(sign):
        eff[s].SetTitle("")
        eff[s].SetMarkerColor(colors[i])
        eff[s].SetLineColor(colors[i])
        eff[s].SetLineStyle(lines[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(markers[i])
        eff[s].GetXaxis().SetTitle("LLP transverse decay length (m)")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.21)
        eff[s].SetMinimum(-0.01)
        eff[s].SetMaximum(1.2)
        eff[s].SetMinimum(0.)
        #eff[s].SetMaximum(5)
        #eff[s].SetMinimum(1.e-6)

        #if i ==0:
        #    eff[s].Draw("AP")
        #else:
        #    eff[s].Draw("P,sames")


    e.SetTitle("")
    e.SetMarkerColor(2)
    e.SetLineColor(2)
    e.SetLineStyle(1)
    e.SetLineWidth(2)
    e.SetMarkerSize(1.4)
    e.SetMarkerStyle(20)
    e.GetXaxis().SetTitle("LLP transverse decay length (m)")
    #e.GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
    e.GetYaxis().SetTitle("TDJ tagger efficiency")
    e.GetXaxis().SetTitleSize(0.04)
    e.GetXaxis().SetTitleOffset(1.3)
    e.GetYaxis().SetTitleSize(0.04)
    e.SetMaximum(1.21)
    e.SetMinimum(-0.01)
    e.SetMaximum(1.2)
    e.SetMinimum(0.)
    e.Draw("AP")
    gPad.RedrawAxis()

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)
        #from s_b plot above
        drawCMS_simple(-1, "Simulation", onTop=True,left_marg_CMS=0.275,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg = TLegend(0.15, 0.3+0.4, 0.4, 0.55+0.2)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    #leg.Draw()
    #can.Print(out_fold+"/Supplementary/TDJ_eff_vs_radius"+postfix+".pdf")
    #can.Print(out_fold+"/Supplementary/TDJ_eff_vs_radius"+postfix+".png")

    can.Print(out_fold+"/Figure_004.pdf")
    can.Print(out_fold+"/Figure_004.png")


    #Root file for hepdata
    out_file = TFile(out_fold+"/Figure_004.root","RECREATE")
    out_file.cd()
    e.Write("eff")
    print "Writing "+out_fold+"/Figure_004.root"
    out_file.Write()
    out_file.Close()

    #Normalize weights
    #norm_s = w_s.sum(axis=0)
    #w_s_norm = np.true_divide(w_s,norm_s)


def tag_eff_vs_z(ERA,ntupledir,cut="isMC",postfix=""):

    if ERA=="2018":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    elif ERA=="2017":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    elif ERA=="2016":
        from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

    sign = [
        'SUSY_mh127_ctau500_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau500_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau500_HH',#'SUSY_mh1000_ctau3000_HH',
        ##'SUSY_mh1800_ctau500_HH',#'SUSY_mh1800_ctau3000_HH',
        ##'SUSY_mh600_ctau500_HH','SUSY_mh600_ctau3000_HH',
        'SUSY_mh127_ctau3000_HH',#'SUSY_mh127_ctau3000_HH',
        ##'SUSY_mh300_ctau500_HH','SUSY_mh300_ctau3000_HH',
        'SUSY_mh400_ctau3000_HH',#'SUSY_mh400_ctau3000_HH',
        'SUSY_mh1000_ctau3000_HH',#'SUSY_mh1000_ctau3000_HH',
    ]

    CUT = cut
    #CUT = "isSR"
    chain = {}
    NTUPLEDIR = ntupledir%ERA#
    #NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR/"
    #NTUPLEDIR_Gen = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_Gen/"
    print "\n"
    print "    -- Using ntuples: ", NTUPLEDIR
    print "\n"
    tree_weight_dict_s = get_tree_weights(NTUPLEDIR,sign,ERA,main_pred_sample="HighMET")

    colors = [602,2,418,  602,2,418, 856,801]
    lines = [1,1,1,    2,2,2]
    markers = [20,21,22, 24,25,26]

    h_num = {}
    h_den = {}
    h2_num = {}
    h2_den = {}
    eff = {}

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()
    #can.SetGrid()
    can.SetLogx()
    #can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.15)

    #Perform the ctau reweighting
    for i, s in enumerate(sign):

        den_val = np.array([])
        num_val = np.array([])
        den_r_val = np.array([])
        num_r_val = np.array([])
        den_w = np.array([])
        num_w = np.array([])

        c = 0
        list_of_variables = ["isMC","Jets.pt","Jets.eta","Jets.sigprob","HLT*","MT","HT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM*","EventNumber","RunNumber","LumiNumber","EventWeight","PUWeight","PUReWeight","Jets.isGenMatched*","Jets.radiusLLP","Jets.zLLP","Gen*","Jets.eta",CUT]

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
                cut_mask_gen = np.logical_and(arrays["Jets.isGenMatchedCaloCorr"]==1, cut_mask)
                cut_mask = np.logical_and((cut_mask_gen.any()==True),cut_mask)

                sigprob = arrays["Jets.sigprob"][cut_mask_gen][cut_mask]
                pt = arrays["Jets.pt"][cut_mask_gen][cut_mask]
                z = abs(arrays["Jets.zLLP"][cut_mask_gen][cut_mask]/100.)
                radius = arrays["Jets.radiusLLP"][cut_mask_gen][cut_mask]/100.
                dummy = (pt>-1.)
                tagged = np.logical_and(dummy,(sigprob>0.996))

                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                weight = np.multiply(eventweight,pureweight)*tree_weight_dict_s[s][ss]

                den_w = np.concatenate(( den_w, np.multiply(dummy, weight)))
                num_w = np.concatenate(( num_w, np.multiply(tagged, weight) ))

                den_val = np.concatenate(( den_val, np.multiply(dummy, z ) ))
                num_val = np.concatenate(( num_val, np.multiply(tagged, z ) ))

                den_r_val = np.concatenate(( den_r_val, np.multiply(dummy, radius ) ))
                num_r_val = np.concatenate(( num_r_val, np.multiply(tagged, radius ) ))

                del arrays

                c+=1
                if c%100==0:
                    print ("uproot iteration n. %d/%d; per-it time %.4f, total time elapsed %.2f" %(c,n_iter,time.time()-st_it,time.time()-start_uproot))
                    print "***********************"


        #Throw these vectors to a histogram
        z_bins = np.array([0.01,0.1,0.25,0.5,1,3.,100.])
        z_bins = np.logspace(-2,2,100,endpoint=True)
        z_bins = np.logspace(-2,2,50,endpoint=True)
        bins_scan_points = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1., 1.2, 1.5, 3., 5. ,10.])
        bins_scan_points = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1., 1.2, 1.5, 1.7, 2., 3., 10.])
        bins_scan_points_z = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1., 1.5, 2., 3., 5., 10.])
        bins = z_bins#np.array([0.01,0.1,0.25,0.5,1,2.5,5.,10.,100.])
        bins = bins_scan_points_z

        h_num[s] = TH1F(s+"_num",s+"_num",len(bins)-1,bins)
        h_den[s] = TH1F(s+"_den",s+"_den",len(bins)-1,bins)

        h2_num[s] = TH2F(s+"_num2",s+"_num2",len(bins)-1,bins,len(bins)-1,bins)
        h2_den[s] = TH2F(s+"_den2",s+"_den2",len(bins)-1,bins,len(bins)-1,bins)

        arr_num = np.array([])
        arr_den = np.array([])
        arr_r_num = np.array([])
        arr_r_den = np.array([])
        arr_w_num = np.array([])
        arr_w_den = np.array([])
        
        for row in num_val:
            if len(row)>0:
                arr_num = np.concatenate(( arr_num, np.stack(row)  ))

        for row in num_w:
            if len(row)>0:
                arr_w_num = np.concatenate(( arr_w_num, np.stack(row)  ))

        for row in num_r_val:
            if len(row)>0:
                arr_r_num = np.concatenate(( arr_r_num, np.stack(row)  ))

        for row in den_val:
            if len(row)>0:
                arr_den = np.concatenate(( arr_den, np.stack(row)  ))

        for row in den_w:
            if len(row)>0:
                arr_w_den = np.concatenate(( arr_w_den, np.stack(row)  ))

        for row in den_r_val:
            if len(row)>0:
                arr_r_den = np.concatenate(( arr_r_den, np.stack(row)  ))

        root_numpy.fill_hist(h_num[s],arr_num,arr_w_num)
        root_numpy.fill_hist(h_den[s],arr_den,arr_w_den)

        root_numpy.fill_hist(h2_num[s],np.column_stack((arr_num,arr_r_num)),arr_w_num)
        root_numpy.fill_hist(h2_den[s],np.column_stack((arr_den,arr_r_den)),arr_w_den)

        '''
        h2_den[s].GetXaxis().SetTitle("|LLP longitudinal decay length| (m)")
        h2_den[s].GetYaxis().SetTitle("LLP transverse decay length (m)")

        h2_num[s].GetXaxis().SetTitle("|LLP longitudinal decay length| (m)")
        h2_num[s].GetYaxis().SetTitle("LLP transverse decay length (m)")

        can2d = TCanvas("can2d","can2d",900,800)
        can2d.cd()
        #can2.SetGrid()
        can2d.SetLogx()
        can2d.SetLogy()
        can2d.SetRightMargin(0.12)
        can2d.SetLeftMargin(0.12)
        can2d.SetBottomMargin(0.15)
        can2d.cd()
        h2_den[s].Draw("COLZ")
        can2d.Print(out_fold+"/Supplementary/TDJ_"+s+"_den_vs_z_r"+postfix+".pdf")
        can2d.Print(out_fold+"/Supplementary/TDJ_"+s+"_den_vs_z_r"+postfix+".png")
        can2d.Close()

        can2n = TCanvas("can2n","can2n",900,800)
        can2n.cd()
        #can2.SetGrid()
        can2n.SetLogx()
        can2n.SetLogy()
        can2n.SetRightMargin(0.12)
        can2n.SetLeftMargin(0.12)
        can2n.SetBottomMargin(0.15)
        can2n.cd()
        h2_num[s].Draw("COLZ")
        can2n.Print(out_fold+"/Supplementary/TDJ_"+s+"_num_vs_z_r"+postfix+".pdf")
        can2n.Print(out_fold+"/Supplementary/TDJ_"+s+"_num_vs_z_r"+postfix+".png")
        can2n.Close()

        can2e = TCanvas("can2e","can2e",900,800)
        can2e.cd()
        #can2.SetGrid()
        can2e.SetLogx()
        can2e.SetLogy()
        can2e.SetRightMargin(0.12)
        can2e.SetLeftMargin(0.12)
        can2e.SetBottomMargin(0.15)
        can2e.cd()
        h2_num[s].Divide(h2_den[s])
        h2_num[s].Draw("COLZ")
        can2e.Print(out_fold+"/Supplementary/TDJ_"+s+"_eff_vs_z_r"+postfix+".pdf")
        can2e.Print(out_fold+"/Supplementary/TDJ_"+s+"_eff_vs_z_r"+postfix+".png")
        can2e.Close()
        '''

        eff[s] = TGraphAsymmErrors()
        eff[s].BayesDivide(h_num[s],h_den[s])
        #Show only den to check if it makes sense
        #h_den[s].Scale(1./h_den[s].Integral())
        #h_num[s].Scale(1./h_num[s].Integral())
        #eff[s] = TGraphAsymmErrors(h_den[s])
        #eff[s] = TGraphAsymmErrors(h_num[s])

        eff[s].SetTitle("")
        eff[s].SetMarkerColor(colors[i])
        eff[s].SetLineColor(colors[i])
        eff[s].SetLineStyle(lines[i])
        eff[s].SetLineWidth(2)
        eff[s].SetMarkerSize(1.4)
        eff[s].SetMarkerStyle(markers[i])
        eff[s].GetXaxis().SetTitle("|LLP longitudinal decay length| (m)")
        #eff[s].GetXaxis().SetTitle("#tau_{physical}^{#chi} (ns)")
        eff[s].GetYaxis().SetTitle("TDJ tagger efficiency")
        eff[s].GetXaxis().SetTitleSize(0.04)
        eff[s].GetXaxis().SetTitleOffset(1.3)
        eff[s].GetYaxis().SetTitleSize(0.04)
        eff[s].SetMaximum(1.21)
        eff[s].SetMinimum(-0.01)
        #eff[s].SetMaximum(5)
        #eff[s].SetMinimum(1.e-6)

        if i ==0:
            eff[s].Draw("AP")
        else:
            eff[s].Draw("P,sames")

        

    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi = 0.975)
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.275,custom_spacing=0.6,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)


    leg = TLegend(0.6-0.3, 0.2+0.35, 0.9-0.3, 0.45+0.35)
    leg = TLegend(0.6-0.3, 0.2, 0.9-0.3, 0.45)
    leg.SetTextSize(0.035)
    for s in sign:
        leg.AddEntry(eff[s],samples[s]['label'],"PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_z"+postfix+".pdf")
    can.Print(out_fold+"/Supplementary/TDJ_eff_vs_z"+postfix+".png")

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
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        drawCMS_simple(-1, "Simulation Supplementary", onTop=True,draw_s_only=True)

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
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True,draw_s_only=True)
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
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        drawCMS_simple(-1, "Simulation", onTop=True,draw_s_only=True)
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

def plot_s_b(var,label,nbins,bin_min,bin_max,fig_name=""):
    '''s and b'''

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"TDJ_SigAndBkg_"+var
    else:
        output_string=out_fold+fig_name

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


    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can = TCanvas("can","can",900,800)
    can.cd()

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    can.SetLogy()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)
    hb.SetLineColor(860)
    hs.SetLineColor(633)
    hb.SetFillColorAlpha(860,0.3)
    hs.SetFillColorAlpha(633,0.3)
    #FR comment: thinner line
    hb.SetLineWidth(2)
    hs.SetLineWidth(2)
    hb.SetLineWidth(1)
    hs.SetLineWidth(1)
    hb.GetXaxis().SetTitle(label)
    hb.GetYaxis().SetTitle("Events / bin")
    hb.GetXaxis().SetTitleSize(0.0425)
    hb.GetYaxis().SetTitleSize(0.0425)
    hb.Draw("HISTO")
    hb.SetMaximum(1.01)#(max_val*1.2)
    hb.SetMinimum(0.0001)
    hs.Draw("HISTO,sames")
    gPad.RedrawAxis()
    if PRELIMINARY:
        drawCMS_simple(-1, "Simulation Preliminary", onTop=True,draw_s_only=True)
    else:
        #drawCMS_simple(-1, "Simulation", onTop=True,draw_s_only=True)
        #Sijin
        #drawCMS_simple(-1, "Simulation", onTop=True,left_marg_CMS=0.225,draw_s_only=True,top_marg_cms=0.96,top_marg_lumi=0.96)
        #FR comment: CMS Simulation inside the frame
        drawCMS_simple(-1, "Simulation", onTop=True,left_marg_CMS=0.275,draw_s_only=True,top_marg_cms=0.86,top_marg_lumi=0.96)

    #FR comment: CMS Simulation inside the frame --> move legend to the right
    leg = TLegend(0.45+0.2, 0.7, 0.65+0.2, 0.85)
    leg.SetTextSize(0.035)
    #Sijin
    leg.SetTextSize(0.04)
    leg.AddEntry(hb,"Background","F")
    leg.AddEntry(hs,"Signal","F")
    leg.SetBorderSize(0)
    leg.Draw()

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    can.Update()
    can.Print(output_string+".pdf")
    can.Print(output_string+".png")

    #Root file for hepdata
    out_file = TFile(output_string+".root","RECREATE")
    out_file.cd()
    hb.Write("background")
    hs.Write("signal")
    print "Writing "+output_string+".root"
    out_file.Write()
    out_file.Close()
   

def plot_dnn_modeling():
    '''ZtoLLPho'''
    #2016
    #2017
    #2018
    
def plot_mistag(fig_name=""):
    '''mistag with uncertainty'''

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"MistagEfficiency"
    else:
        output_string=out_fold+fig_name
    
    less_bins_plot = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
    label_dict = {}
    label_dict["ZtoLL"] = "Z #rightarrow ll"
    label_dict["WtoLN"] = "W #rightarrow l#nu"
    label_dict["JetHT"] = "QCD"
    label_dict["TtoEM"] = "ttbar e + #mu"

    BASEDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_%s_"
    YIELDDIR = "plots/Yields_AN_fix/v6_calo_AOD_%s_SR"
    colors = {"WtoLN":2,"ZtoLL":1,"JetHT":418,"TtoEM":4}
    markers = {"WtoLN":20,"ZtoLL":24,"JetHT":26,"TtoEM":25}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":3,"TtoEM":4}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":2,"TtoEM":2}
    colors = {"WtoLN":1,"ZtoLL":2,"JetHT":418,"TtoEM":4}
    maxeff = 0.0016
    #CWR: move to percentages
    maxeff = maxeff*100

    def do_mistag(eras,LUMI,era_label,plot_label):

        print "LUMI: ", LUMI

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
        print "Looking for largest discrepancy"
        print "Probably wrong as it should be bin-by-bin..."
        for k in label_dict.keys():
            unc = abs(y[k] - y["WtoLN"])/y["WtoLN"]
            print "region, prediction, discrepancy w.r.t. nominal"
            print k, y[k], unc
            max_unc = max(max_unc,unc)
   
        print "largest composition uncertainty"
        print max_unc*100

        #FR:
        for k in ["WtoLN"]:#label_dict.keys()
        #JHEP: show also the other lines
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT"]:
            #CWR: move to percentages
            eff[k].BayesDivide(100*num[k],den[k])
            eff[k].SetMarkerStyle(markers[k])
            eff[k].SetMarkerColor(colors[k])#(1)#
            eff[k].SetMarkerSize(1.3)
            eff[k].SetLineColor(colors[k])#(1)#
            eff[k].SetFillColorAlpha(922,0.5)#(colors[k],0.5)
            #eff[k].SetFillStyle(3001)
            eff[k].SetLineWidth(2)
            if k=="WtoLN":
                eff[k].SetLineWidth(0)
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
            eff[k].GetYaxis().SetTitle("Mis-tag efficiency (%)")
            eff[k].GetXaxis().SetTitleSize(0.04)
            eff[k].GetYaxis().SetTitleSize(0.04)

        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can = TCanvas("can","can",900,800)
        can.cd()
        #can.SetGrid()
        #can.SetLogx()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.15)
        can.SetBottomMargin(0.1)
        leg = TLegend(0.7, 0.78, 0.93, 0.88)
        leg.SetTextSize(0.035)
        #JHEP
        #leg = TLegend(0.6, 0.6, 0.9, 0.8)
        #leg.SetTextSize(0.04)
        #leg.SetBorderSize(0)

        #FR: show only WToLN
        for i,k in enumerate(["WtoLN"]):#"ZtoLL","TtoEM","JetHT",
            if i==0:
                eff[k].Draw("APE2")
            else:
                eff[k].Draw("P,sames")
            leg.AddEntry(eff[k],label_dict[k],"PF")

        #JHEP show all the points
        #i=0
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT"]:
        #    if i==0:
        #        eff[k].Draw("APE2")
        #        #eff[k].Draw("APE0")
        #        #JHEP legend
        #        leg.AddEntry(eff[k],label_dict[k],"PF")
        #    else:
        #        eff[k].Draw("PELX,sames")
        #        #JHEP legend
        #        leg.AddEntry(eff[k],label_dict[k],"PL")
        #    i+=1
        ##JHEP: redraw W on top
        #for k in ["WtoLN"]:
        #    eff[k].Draw("PEXY0,sames")
        
        #drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True)
        if PRELIMINARY:
            drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True,left_marg_CMS=0.25,top_marg_cms=0.96,top_marg_lumi=0.965)
        else:
            drawCMS_simple(LUMI, "", ERA=era_label, onTop=True,left_marg_CMS=0.3,top_marg_cms=0.87,top_marg_lumi=0.965)
        ##JHEP
        #leg.Draw()
        #FR: no legend printed
        can.Print(output_string+plot_label+".pdf")
        can.Print(output_string+plot_label+".png")
        can.Print(output_string+plot_label+".C")
        can.Close()

        #Root file for hepdata
        out_file = TFile(output_string+plot_label+".root","RECREATE")
        out_file.cd()
        eff["WtoLN"].Write("mistag_MR")
        eff["ZtoLL"].Write("mistag_ZtoLL")
        eff["TtoEM"].Write("mistag_TtoEM")
        eff["JetHT"].Write("mistag_JetHT")
        print "Writing "+output_string+plot_label+".root"
        out_file.Write()
        out_file.Close()


    eras = ["2016_G-H","2017","2018"]

    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016

    #probably wrong, frozen for approval:
    #LUMI_G2018 = 111941.400399

    #new and fixed
    LUMI_G2018  = lumi2018["HighMET"]["tot"] + lumi2017["HighMET"]["tot"] + lumi2016["HighMET"]["G"] + lumi2016["HighMET"]["H"]

    era_label="2016-2018"#"2016-2018"#"2016G-2018"
    plot_label="2016GTo2018" if fig_name=="" else "-b"
    do_mistag(eras,LUMI_G2018,era_label=era_label,plot_label=plot_label)

    eras = ["2016_B-F"]
    era_label="2016"#"2016"#"2016B-F"
    plot_label="2016BF" if fig_name=="" else "-a"

    #probably wrong, frozen at approval:
    #LUMI_BF = 137478.722953 - 111941.400399

    #new and fixed
    LUMI_BF  = lumi2016["HighMET"]["tot"] - lumi2016["HighMET"]["G"] - lumi2016["HighMET"]["H"]
    do_mistag(eras,LUMI_BF,era_label=era_label,plot_label=plot_label)


def plot_mistag_JHEP(fig_name=""):
    '''mistag with uncertainty'''

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"MistagEfficiency"
    else:
        output_string=out_fold+fig_name
    
    less_bins_plot = array('d',[-1.5, -1.25, -1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1., 1.25, 1.5])
    label_dict = {}
    label_dict["ZtoLL"] = "Z+jets MR"#"Z #rightarrow ll"
    label_dict["WtoLN"] = "Nominal MR"#"W #rightarrow l#nu"
    label_dict["JetHT"] = "QCD MR"#"QCD"
    label_dict["TtoEM"] = "t#bar{t} MR"#"ttbar e + #mu"
    label_dict["HighMET"] = "SR"#"SR"

    BASEDIR_SR = "plots/Efficiency_AN_fix/v6_calo_AOD_%s_SR_JHEP/"

    BASEDIR = "plots/Efficiency_AN_fix/v5_calo_AOD_%s_"
    YIELDDIR = "plots/Yields_AN_fix/v6_calo_AOD_%s_SR"
    colors = {"WtoLN":2,"ZtoLL":1,"JetHT":418,"TtoEM":4,"HighMET":2}
    markers = {"WtoLN":20,"ZtoLL":24,"JetHT":26,"TtoEM":25,"HighMET":24}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":3,"TtoEM":4}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":2,"TtoEM":2,"HighMET":1}

    lines = {"WtoLN":1,"ZtoLL":1,"JetHT":1,"TtoEM":1,"HighMET":1}

    colors = {"WtoLN":1,"ZtoLL":418,"JetHT":4,"TtoEM":800-4,"HighMET":2}
    colors = {"WtoLN":1,"ZtoLL":418,"JetHT":4,"TtoEM":2,"HighMET":2}
    maxeff = 0.0016
    #CWR: move to percentages
    maxeff = maxeff*100

    def do_mistag_JHEP(eras,LUMI,era_label,plot_label):

        print "LUMI: ", LUMI

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

                if k=="HighMET":
                    DIR = (BASEDIR_SR%era[0:4])

                yieldfile = (YIELDDIR%era[0:4])+"/BkgPredResults_"+era[0:4]+"_SR_HighMET"+label_y+".yaml"
                with open(yieldfile) as f:
                    results = yaml.load(f, Loader=yaml.Loader)
                    f.close()
                for rk in results.keys():
                    if k in rk:
                        y[k] +=results[rk]["HighMET"]["pred_2_from_1"]
                if k=="HighMET":
                    infile = TFile(DIR+"TagEff_HighMET"+label+".root","READ")
                else:
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


        #First: do the bayes divide
        for k in ["WtoLN","ZtoLL","TtoEM","JetHT","HighMET"]:
            #CWR: move to percentages
            eff[k].BayesDivide(100*num[k],den[k])

        #Second: bin-by-bin uncertainty
        max_per_bin_unc = []
        np = eff["WtoLN"].GetN()
        for i in range(0,np):
            tmp_max = 0
            for k in ["ZtoLL","TtoEM","JetHT"]:
                y_val = abs(eff[k].GetPointY(i) - eff["WtoLN"].GetPointY(i))/eff["WtoLN"].GetPointY(i)
                tmp_max = max(tmp_max,y_val)
            max_per_bin_unc.append(tmp_max)
   
        print "largest composition uncertainty"
        print max_per_bin_unc

        #FR:
        #for k in ["WtoLN"]:#label_dict.keys()
        #JHEP: show also the other lines
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT","HighMET"]:
        for k in ["WtoLN","ZtoLL","TtoEM","JetHT"]:
            #CWR: move to percentages
            #eff[k].BayesDivide(100*num[k],den[k])
            eff[k].SetMarkerStyle(markers[k])
            eff[k].SetMarkerColor(colors[k])#(1)#
            eff[k].SetMarkerSize(1.3)
            eff[k].SetLineColor(colors[k])#(1)#
            eff[k].SetFillColorAlpha(922,0.5)#(colors[k],0.5)
            #eff[k].SetFillStyle(3001)
            eff[k].SetLineWidth(2)
            if k=="WtoLN":
                eff[k].SetLineWidth(0)
            eff[k].SetLineStyle(lines[k])
            eff[k].GetYaxis().SetRangeUser(-0.0001,maxeff)
            if k=="WtoLN":
                np = eff[k].GetN()
                for i in range(0,np):
                    print "Set the global uncertainty!"
                    eff[k].SetPointEYhigh(i, math.sqrt(eff[k].GetErrorYhigh(i)**2 + pow(max_per_bin_unc[i]*eff[k].GetPointY(i),2) ))#not 0.5! all of it!
                    eff[k].SetPointEYlow(i, math.sqrt(eff[k].GetErrorYlow(i)**2 + pow(max_per_bin_unc[i]*eff[k].GetPointY(i),2) ))

            print eff[k].Print()

            eff[k].GetYaxis().SetRangeUser(-0.0001,maxeff)
            eff[k].GetXaxis().SetTitle("Jet #eta")
            eff[k].GetYaxis().SetTitle("Mis-tag efficiency (%)")
            eff[k].GetXaxis().SetTitleSize(0.04)
            eff[k].GetYaxis().SetTitleSize(0.04)

        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can = TCanvas("can","can",900,800)
        can.cd()
        #can.SetGrid()
        #can.SetLogx()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.15)
        can.SetBottomMargin(0.1)
        leg = TLegend(0.7, 0.78, 0.93, 0.88)
        leg.SetTextSize(0.035)
        #JHEP
        leg = TLegend(0.6, 0.6, 0.9, 0.8)
        leg.SetTextSize(0.04)
        leg.SetBorderSize(0)
        #FR: show only WToLN
        #for i,k in enumerate(["WtoLN"]):#"ZtoLL","TtoEM","JetHT",
        #    if i==0:
        #        eff[k].Draw("APE2")
        #    else:
        #        eff[k].Draw("P,sames")
        #    leg.AddEntry(eff[k],label_dict[k],"PF")

        #JHEP show all the points
        i=0
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT","HighMET"]:
        for k in ["WtoLN","ZtoLL","TtoEM","JetHT"]:
        #for k in ["WtoLN","HighMET"]:
            if i==0:
                eff[k].Draw("APE2")
                #eff[k].Draw("APE0")
                #JHEP legend
                leg.AddEntry(eff[k],label_dict[k],"PF")
            else:
                eff[k].Draw("PE,sames")
                #JHEP legend
                leg.AddEntry(eff[k],label_dict[k],"PL")
            i+=1
        #JHEP: redraw W on top
        for k in ["WtoLN"]:
            eff[k].Draw("PEXY0,sames")
        
        #drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True)
        if PRELIMINARY:
            drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True,left_marg_CMS=0.25,top_marg_cms=0.96,top_marg_lumi=0.965)
        else:
            drawCMS_simple(LUMI, "", ERA=era_label, onTop=True,left_marg_CMS=0.3,top_marg_cms=0.87,top_marg_lumi=0.965)
            #drawCMS_simple(LUMI, "Supplementary", ERA=era_label, onTop=True,left_marg_CMS=0.3,custom_spacing=0.35,draw_s_only=False,top_marg_cms=0.86,top_marg_lumi=0.96)

        #JHEP
        leg.Draw()
        #FR: no legend printed
        can.Print(output_string+plot_label+".pdf")
        can.Print(output_string+plot_label+".png")
        can.Print(output_string+plot_label+".C")
        can.Close()

        #Root file for hepdata
        out_file = TFile(output_string+plot_label+".root","RECREATE")
        out_file.cd()
        eff["WtoLN"].Write("mistag_MR")
        eff["ZtoLL"].Write("mistag_ZtoLL")
        eff["TtoEM"].Write("mistag_TtoEM")
        eff["JetHT"].Write("mistag_JetHT")
        print "Writing "+output_string+plot_label+".root"
        out_file.Write()
        out_file.Close()


    eras = ["2016_G-H","2017","2018"]

    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016

    #probably wrong, frozen for approval:
    #LUMI_G2018 = 111941.400399

    #new and fixed
    LUMI_G2018  = lumi2018["HighMET"]["tot"] + lumi2017["HighMET"]["tot"] + lumi2016["HighMET"]["G"] + lumi2016["HighMET"]["H"]

    era_label="2016-2018"#"2016-2018"#"2016G-2018"
    plot_label="2016GTo2018" if fig_name=="" else "-b"
    do_mistag_JHEP(eras,LUMI_G2018,era_label=era_label,plot_label=plot_label)

    eras = ["2016_B-F"]
    era_label="2016"#"2016"#"2016B-F"
    plot_label="2016BF" if fig_name=="" else "-a"

    #probably wrong, frozen at approval:
    #LUMI_BF = 137478.722953 - 111941.400399

    #new and fixed
    LUMI_BF  = lumi2016["HighMET"]["tot"] - lumi2016["HighMET"]["G"] - lumi2016["HighMET"]["H"]
    do_mistag_JHEP(eras,LUMI_BF,era_label=era_label,plot_label=plot_label)


def plot_bin1_pred_JHEP(fig_name=""):

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"Bin1Prediction"
    else:
        output_string=out_fold+fig_name
    
    less_bins_plot = array('d',[-1., -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.])
    label_dict = {}
    label_dict["ZtoLL"] = "Z+jets MR"#"Z #rightarrow ll"
    label_dict["WtoLN"] = "1 tag prediction"#"W #rightarrow l#nu"
    label_dict["JetHT"] = "QCD MR"#"QCD"
    label_dict["TtoEM"] = "t#bar{t} MR"#"ttbar e + #mu"
    label_dict["SR"] = "1 tag observed"#"SR"

    BASEDIR = "plots/Efficiency_JHEP/v6_calo_AOD_%s_SR/"

    colors = {"WtoLN":2,"ZtoLL":1,"JetHT":418,"TtoEM":4,"HighMET":2}
    markers = {"WtoLN":20,"ZtoLL":24,"JetHT":26,"TtoEM":25,"HighMET":24}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":3,"TtoEM":4}
    lines = {"WtoLN":1,"ZtoLL":2,"JetHT":2,"TtoEM":2,"HighMET":1}

    lines = {"WtoLN":1,"ZtoLL":1,"JetHT":1,"TtoEM":1,"HighMET":1}

    colors = {"WtoLN":1,"ZtoLL":418,"JetHT":4,"TtoEM":800-4,"HighMET":2}
    colors = {"WtoLN":1,"ZtoLL":418,"JetHT":4,"TtoEM":2,"HighMET":2}

    maxeff = 0.0016
    #CWR: move to percentages
    maxeff = 10000#maxeff*100

    def do_bin1_JHEP(eras,LUMI,era_label,plot_label):

        print "LUMI: ", LUMI
        h = {}
        g = {}
        for k in label_dict.keys():
            h[k] = TH1D()

        #First loop over files
        for i, era in enumerate(eras):
            label = ""
            label_y = ""
            label_era = ""
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
                label_era+="_"+era[5:8]
            label_y+="_MinDPhi_0p5"

            DIR = (BASEDIR%era[0:4])
            print DIR+"Bin1_Jets_eta"+label_era+".root"
            infile = TFile(DIR+"Bin1_Jets_eta"+label_era+".root","READ")
            print infile.ls()
            #tmp_den = infile.Get("den_"+k)

            for k in label_dict.keys():
                print k
                tmp_h = infile.Get(k)
                tmp_h.SetDirectory(0)
                if i==0:
                    h[k] = tmp_h.Clone(k)
                else:
                    h[k].Add(tmp_h)
                h[k].SetDirectory(0)
                #here add to the main one

        for k in label_dict.keys():
            print k
            print h[k].Print()
            h[k] = h[k].Rebin(len(less_bins_plot)-1,k+"2",less_bins_plot)
            g[k] = TGraphAsymmErrors(h[k])
            print g[k].Print()


        #Second: bin-by-bin uncertainty, out of the era loop
        max_per_bin_unc = []
        np = g["WtoLN"].GetN()
        for i in range(0,np):
            tmp_max = 0
            for k in ["ZtoLL","TtoEM","JetHT"]:
                y_val = abs(g[k].GetPointY(i) - g["WtoLN"].GetPointY(i))/g["WtoLN"].GetPointY(i) if g["WtoLN"].GetPointY(i)>0 else 0
                tmp_max = max(tmp_max,y_val)
            max_per_bin_unc.append(tmp_max)


        for i in range(0,np):
            for k in ["WtoLN"]:
                print "Set the global uncertainty!"
                g[k].SetPointEYhigh(i, math.sqrt(g[k].GetErrorYhigh(i)**2 + pow(max_per_bin_unc[i]*g[k].GetPointY(i),2) ))#not 0.5! all of it!
                g[k].SetPointEYlow(i, math.sqrt(g[k].GetErrorYlow(i)**2 + pow(max_per_bin_unc[i]*g[k].GetPointY(i),2) ))

   
        print "largest composition uncertainty"
        print max_per_bin_unc

        for k in ["WtoLN","ZtoLL","TtoEM","JetHT"]:
            g[k].SetMarkerStyle(25)#(markers[k])
            g[k].SetMarkerColor(colors[k])#(1)#
            g[k].SetMarkerSize(1.3)
            g[k].SetLineColor(colors[k])#(1)#
            g[k].SetFillColorAlpha(922,0.5)#(colors[k],0.5)
            #g[k].SetFillStyle(3001)
            g[k].SetLineWidth(2)
            if k=="WtoLN":
                g[k].SetLineWidth(0)
            g[k].SetLineStyle(lines[k])
            g[k].GetYaxis().SetRangeUser(9.,1100.)
            g[k].GetXaxis().SetTitle("Jet #eta")
            g[k].GetYaxis().SetTitle("Events / bin")
            g[k].GetXaxis().SetTitleSize(0.04)
            g[k].GetYaxis().SetTitleSize(0.04)

        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can = TCanvas("can","can",900,800)
        can.cd()
        #can.SetGrid()
        can.SetLogy()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.15)
        can.SetBottomMargin(0.1)
        leg = TLegend(0.7, 0.78, 0.93, 0.88)
        leg.SetTextSize(0.035)
        #JHEP
        leg = TLegend(0.6, 0.65, 0.9, 0.85)
        leg.SetTextSize(0.04)
        leg.SetBorderSize(0)

        #JHEP show all the points
        i=0
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT","HighMET"]:
        #for k in ["WtoLN","ZtoLL","TtoEM","JetHT","SR"]:
        for k in ["WtoLN","SR"]:
            if i==0:
                g[k].Draw("APE2")
                #eff[k].Draw("APE0")
                #JHEP legend
                leg.AddEntry(g[k],label_dict[k],"PF")
            else:
                g[k].Draw("PE,sames")
                #JHEP legend
                leg.AddEntry(g[k],label_dict[k],"PEL")
            i+=1
        #JHEP: redraw W on top
        for k in ["WtoLN"]:
            g[k].Draw("PEXY0,sames")
        
        #drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True)
        if PRELIMINARY:
            drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True,left_marg_CMS=0.25,top_marg_cms=0.96,top_marg_lumi=0.965)
        else:
            drawCMS_simple(LUMI, "", ERA=era_label, onTop=True,left_marg_CMS=0.3,top_marg_cms=0.87,top_marg_lumi=0.965)
            #drawCMS_simple(LUMI, "Supplementary", ERA=era_label, onTop=True,left_marg_CMS=0.3,custom_spacing=0.35,draw_s_only=False,top_marg_cms=0.86,top_marg_lumi=0.96)

        #JHEP
        leg.Draw()
        #FR: no legend printed
        can.Print(output_string+plot_label+".pdf")
        can.Print(output_string+plot_label+".png")
        can.Print(output_string+plot_label+".C")
        can.Close()

        #Root file for hepdata
        out_file = TFile(output_string+plot_label+".root","RECREATE")
        out_file.cd()
        g["WtoLN"].Write("pred")
        g["SR"].Write("obs")
        print "Writing "+output_string+plot_label+".root"
        out_file.Write()
        out_file.Close()

    eras = ["2016_B-F","2016_G-H","2017","2018"]

    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi as lumi2018
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi as lumi2017
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi as lumi2016

    #probably wrong, frozen for approval:
    #LUMI_G2018 = 111941.400399

    #new and fixed
    LUMI_TOT  = lumi2018["HighMET"]["tot"] + lumi2017["HighMET"]["tot"] + lumi2016["HighMET"]["tot"]

    era_label=""#"2016-2018"#"2016G-2018"
    plot_label=""#"2016GTo2018" if fig_name=="" else "-b"
    do_bin1_JHEP(eras,LUMI_TOT,era_label=era_label,plot_label=plot_label)


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



    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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


    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
    fake.GetXaxis().SetTitle("Number of TD-tagged jets")
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
        
    #back = back.Rebin(2)
    #data = data.Rebin(2)
    less_bins = np.array([0.9  , 0.908, 0.916, 0.924, 0.932, 0.94 , 0.948, 0.956, 0.964,0.972, 0.98 , 0.988, 0.996,1.])
    back.Rebin(len(less_bins)-1,"back2",less_bins)
    data.Rebin(len(less_bins)-1,"data2",less_bins)
    print "Rebin anyone?"

    back_unc = back.Clone("BkgUnc")
    back_unc.SetFillStyle(3002)
    back_unc.SetFillColor(860)
    back_unc.SetLineWidth(0)
    back_unc.SetMarkerStyle(0)

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
    #CWR: make back line thicker
    back.SetLineWidth(3)
    data.SetLineWidth(3)
    data.SetMarkerSize(1.3)
    back.GetXaxis().SetTitle("Jet tagger output score")
    back.GetYaxis().SetTitle("Events / bin")
    back.GetXaxis().SetTitleSize(0.04)
    back.GetYaxis().SetTitleSize(0.04)
    #CWR: make title closer to Y-axis
    back.GetYaxis().SetTitleOffset(1.0)

    back.Draw("HIST,sames")
    back_unc.Draw("E2,sames")
    data.Draw("PE,sames")
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True,left_marg_CMS=0.225,top_marg_cms=0.96,top_marg_lumi=0.965)
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.225,top_marg_cms=0.96,top_marg_lumi=0.965)
    leg = TLegend(0.4, 0.7-0.2, 0.65, 0.85-0.2)
    leg.SetTextSize(0.035)
    leg.AddEntry(back,"Simulated background","F")
    leg.AddEntry(back_unc,"Stat. unc.","F")
    leg.AddEntry(data,"Data","PL")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print(out_fold+"TDJ_DataAndBkg_sigprob_zoom.pdf")
    can.Print(out_fold+"TDJ_DataAndBkg_sigprob_zoom.png")



def plot_dnn_score_ratio_panel(fig_name=""):

    output_string = ""
    if fig_name=="":
        output_string=out_fold+"TDJ_DataAndBkg_sigprob_zoom_ratio_panel"
    else:
        output_string=out_fold+fig_name

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
        
    #FR: try leaving 30 bins instead of 15
    #back = back.Rebin(2)
    #data = data.Rebin(2)
    less_bins = np.array([0.9  , 0.908, 0.916, 0.924, 0.932, 0.94 , 0.948, 0.956, 0.964,0.972, 0.98 , 0.988, 0.996,1.])
    back = back.Rebin(len(less_bins)-1,"back2",less_bins)
    data = data.Rebin(len(less_bins)-1,"data2",less_bins)
    back.Scale(1./back.Integral())
    data.Scale(1./data.Integral())
    print "Rebin anyone?"
    back_unc = back.Clone("BkgUnc")
    back_unc.SetFillStyle(3002)
    back_unc.SetFillColor(860)
    back_unc.SetLineWidth(0)
    back_unc.SetMarkerStyle(0)

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    can = TCanvas("can","can",900,800)

    can.Divide(1,2)
    r = 4
    can.GetPad(1).SetPad("TopPad", "", 0., 1./r, 1.0, 1.0, 0, -1, 0)
    can.GetPad(1).SetRightMargin(0.05)
    can.GetPad(1).SetLeftMargin(0.13)
    can.GetPad(1).SetBottomMargin(0.02)
    can.GetPad(1).SetTopMargin(0.08)

    can.GetPad(2).SetPad("BotPad", "", 0., 0., 1.0, 1./r, 0, -1, 0)
    can.GetPad(2).SetRightMargin(0.05)
    can.GetPad(2).SetLeftMargin(0.13)
    can.GetPad(2).SetBottomMargin(0.4)
    can.GetPad(2).SetTopMargin(1)

    #Old settings, let's see
    can.cd(1)
    can.GetPad(1).SetLogy()
    #can.SetRightMargin(0.05)
    #can.SetLeftMargin(0.12)
    #can.SetBottomMargin(0.1)

    back.SetLineColor(860)
    data.SetLineColor(1)
    data.SetMarkerColor(1)
    data.SetMarkerStyle(20)
    back.SetFillColorAlpha(860,0.3)
    #CWR: make back line thicker
    back.SetLineWidth(3)
    data.SetLineWidth(3)
    data.SetMarkerSize(1.3)
    back.GetXaxis().SetTitle("Jet tagger output score")
    back.GetYaxis().SetTitle("Events / bin")
    back.GetXaxis().SetTitleSize(0.04*(1+1./r))
    back.GetYaxis().SetTitleSize(0.04*(1+1./r))
    back.GetYaxis().SetLabelSize(  back.GetYaxis().GetLabelSize()  *(1+1./r))

    #CWR: make title closer to Y-axis
    back.GetYaxis().SetTitleOffset(1.0)

    #Avoid seeing the upper edges of the labels
    back.GetXaxis().SetTitleOffset(1.5)
    back.GetXaxis().SetLabelOffset(1.2)

    back.Draw("HIST,sames")
    back_unc.Draw("E2,sames")
    data.Draw("PEX0,sames")
    if PRELIMINARY:
        drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True,left_marg_CMS=0.225, text_size=0.045*(1+1./r),cms_text_size=0.06*(1+1./r),lumi_text_size=0.04*(1+1./r))
    else:
        drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.275, top_marg_cms=0.87, text_size=0.045*(1+1./r),cms_text_size=0.06*(1+1./r),lumi_text_size=0.04*(1+1./r))
    leg = TLegend(0.4, 0.7-0.2, 0.65, 0.85-0.2)
    leg.SetTextSize(0.035*(1+1./r))
    leg.AddEntry(back,"Simulated background","F")
    leg.AddEntry(back_unc,"Stat. unc. of MC samples","F")
    leg.AddEntry(data,"Data","PE")
    leg.SetBorderSize(0)
    leg.Draw()

    #ratio panel
    can.cd(2)
    ratio = data.Clone("ratio")
    ratio.Divide(back)

    ratio.SetLineColor(1)
    ratio.SetMarkerColor(1)
    ratio.SetMarkerStyle(20)
    ratio.SetLineWidth(2)
    ratio.SetMarkerSize(1.3)

    ratio.GetYaxis().SetTitleOffset(+0.35)
    ratio.GetYaxis().SetTitleOffset(+0.4)
    ratio.GetXaxis().SetTitleOffset(+1.2)

    ratio.GetYaxis().SetTitleSize(0.12*(1+1./(3.2*r) ))
    ratio.GetYaxis().SetLabelSize(0.12*(1+1./r))
    #ratio.GetYaxis().SetTitle("Data / sim. back.")
    #FR
    ratio.GetYaxis().SetTitle("Data/Bkgrd")
    #ratio.GetYaxis().CenterTitle(True)

    ratio.GetXaxis().SetTitleSize(0.13*(1+1./r))
    ratio.GetXaxis().SetLabelSize(0.2*(1+1./r))
    ratio.GetXaxis().SetLabelOffset(0.04)
    ratio.GetXaxis().SetTitle("Jet tagger output score")
    ratio.GetXaxis().SetLabelSize(0.12*(1+1./r))

    ratio.SetMinimum(0)
    ratio.SetMaximum(2)
    ratio.SetLineColor(1)
    ratio.SetFillColorAlpha(1,0.2)
    ratio.SetMarkerColor(1)
    ratio.SetMarkerStyle(20)
    ratio.SetLineWidth(2)
    ratio.SetMarkerSize(1.3)

    ratio.GetYaxis().SetNdivisions(402)
    line = TLine(0.9, 1., 1., 1.)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.SetLineColor(1)
    ratio.Draw("PE2,sames")
    line.Draw("sames")
    ratio.Draw("PE2,sames")

    can.Print(output_string+".pdf")
    can.Print(output_string+".png")

    #Root file for hepdata
    out_file = TFile(output_string+".root","RECREATE")
    out_file.cd()
    back.Write("back")
    back_unc.Write("back_unc")
    data.Write("data")
    ratio.Write("ratio")
    print "Writing "+output_string+".root"
    out_file.Write()
    out_file.Close()


def plot_dnn_inputs():
    FOLD_ele = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_E_v5_ntuples_reweighted/weighted/smearing/"
    eras_ele = ["2016_G-H","2017","2018"]
    extra_era = ["_G-H","",""]

    eras_ele = ["2016","2017","2018"]
    extra_era = ["_G-H","",""]
    LUMI = 137478.722953

    dnn_input_list= [
        "nTrackConstituents",
        "nSelectedTracks",
        "timeShift1",
        "eFracRecHitsEB",
        "nRecHitsEB",
        "sig1EB",
        "sig2EB",
        "ptDEB",
        "cHadEFrac",
        "nHadEFrac",
        "eleEFrac",
        "photonEFrac",
        "ptAllTracks",
        "ptAllPVTracks",
        "alphaMax",
        "betaMax",
        "gammaMax",
        "gammaMaxEM",
        "gammaMaxHadronic",
        "gammaMaxET",
        "minDeltaRAllTracks",
        "minDeltaRPVTracks",
    ]


    bins_list= [
        np.linspace(0,30,15+1),#"nTrackConstituents",
        np.linspace(0,20,10+1),#"nSelectedTracks",
        np.linspace(-1.,5.,15+1),#np.array([]),#"timeShift1",
        np.linspace(0,1,10+1),#np.array([]),#"eFracRecHitsEB",
        np.linspace(0,100,10+1),#41+60+1),#np.array([]),#"nRecHitsEB",
        np.linspace(0,0.5,10+1),#np.array([0,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.3,0.5]),#np.array([]),#np.array([0,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5]),#"sig1EB",
        np.linspace(0,0.05,10+1),#np.array([0,0.005,0.01,0.02,0.05]),#np.array([]),#"sig2EB",
        np.linspace(0,1,10+1),#np.array([]),#np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,1]),#"ptDEB",
        #np.linspace(0,1,10+1),#np.array([]),#"cHadEFrac",
        np.array([0,0.1,0.2,0.3,0.4,1]),#np.array([]),#"cHadEFrac",
        np.linspace(0,1,10+1),#np.array([]),#"nHadEFrac",
        np.linspace(0,1,10+1),#np.array([]),#"eleEFrac",
        np.linspace(0,1,10+1),#np.array([]),#"photonEFrac",
        np.linspace(0,200,10+1),#np.array([]),#"ptAllTracks",
        np.linspace(0,500,10+1),#np.array([]),#np.linspace(0,50,10),#"ptAllPVTracks",
        np.linspace(0,1,10+1),#np.array([]),#"alphaMax",
        np.linspace(0,2,10+1),#np.array([]),#"betaMax",
        np.linspace(0,2,10+1),#np.array([]),#"gammaMax",
        np.linspace(0,2,10+1),#np.array([]),#np.linspace(0,0.5,10),#"gammaMaxEM",
        np.linspace(0,100,10+1),#np.array([]),#np.linspace(0,0.5,10),#"gammaMaxHadronic",
        np.linspace(0,2,10+1),#np.array([]),#np.linspace(0,0.2,10),#"gammaMaxET",
        np.linspace(0,0.5,10+1),#np.array([]),#"minDeltaRAllTracks",
        np.linspace(0,0.5,10+1),#np.array([]),#"minDeltaRPVTracks",
    ]

    back = {}
    data = {}

    for inp in dnn_input_list:
        for i,era in enumerate(eras_ele):
            INPDIR_ele = (FOLD_ele%era[0:4])
            infile_ele = TFile(INPDIR_ele+"DataAndBkg_"+inp+extra_era[i]+".root","READ")
            tmp_back = infile_ele.Get("b")
            tmp_data = infile_ele.Get("d")
            tmp_back.SetDirectory(0)
            tmp_data.SetDirectory(0)
            if i==0:
                print era
                print "era: clone"
                back[inp] = tmp_back.Clone("back")
                data[inp] = tmp_data.Clone("data")
                back[inp].SetDirectory(0)
                data[inp].SetDirectory(0)
            else:
                print era
                print "era: add"
                back[inp].Add(tmp_back)
                data[inp].Add(tmp_data)
                back[inp].SetDirectory(0)
                data[inp].SetDirectory(0)

    inp_it=0    
    for inp in dnn_input_list:
        #Rebin
        if len(bins_list[inp_it])>0:
            back[inp] = back[inp].Rebin(len(bins_list[inp_it])-1,"b_"+inp+"_2",bins_list[inp_it])
            data[inp] = data[inp].Rebin(len(bins_list[inp_it])-1,"b_"+inp+"_2",bins_list[inp_it])
        #Scale
        back[inp].Scale(1./back[inp].Integral())
        data[inp].Scale(1./data[inp].Integral())

        back_unc = back[inp].Clone("BkgUnc")
        back_unc.SetFillStyle(3002)
        back_unc.SetFillColor(860)
        back_unc.SetLineWidth(0)
        back_unc.SetMarkerStyle(0)

        gStyle.SetPadTickX(1)
        gStyle.SetPadTickY(1)
        can = TCanvas("can","can",900,800)
        can.cd()
        can.SetLogy()
        can.SetRightMargin(0.05)
        can.SetLeftMargin(0.12)
        can.SetBottomMargin(0.1)

        back[inp].SetLineColor(860)
        data[inp].SetLineColor(1)
        data[inp].SetMarkerColor(1)
        data[inp].SetMarkerStyle(20)
        back[inp].SetFillColorAlpha(860,0.3)
        #CWR: make back line thicker
        back[inp].SetLineWidth(3)
        data[inp].SetLineWidth(3)
        data[inp].SetMarkerSize(1.3)
        vr = "Jets."+inp
        if inp=="timeShift1":
            vr = "Jets.timeRecHitsEB"
        back[inp].GetXaxis().SetTitle(variable[vr]["title"])
        back[inp].GetYaxis().SetTitle("Events / bin")
        back[inp].GetXaxis().SetTitleSize(0.04)
        back[inp].GetYaxis().SetTitleSize(0.04)
        #CWR: make title closer to Y-axis
        back[inp].GetYaxis().SetTitleOffset(1.2)
        back[inp].SetMaximum(11)
        back[inp].Draw("HIST,sames")
        back_unc.Draw("E2,sames")
        data[inp].Draw("PE,sames")
        if PRELIMINARY:
            drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True,left_marg_CMS=0.225,top_marg_cms=0.96,top_marg_lumi=0.965)
        else:
            #drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.225,top_marg_cms=0.96,top_marg_lumi=0.965)
            #drawCMS_simple(LUMI, "Supplementary", ERA="", onTop=True,left_marg_CMS=0.26,custom_spacing=0.35,draw_s_only=False,top_marg_cms=0.86,top_marg_lumi=0.96)
            drawCMS_simple(LUMI, "", ERA="", onTop=True,left_marg_CMS=0.3,top_marg_cms=0.87,top_marg_lumi=0.965)



        leg = TLegend(0.4+0.175, 0.7-0.05, 0.65+0.175, 0.85-0.05)
        leg.SetTextSize(0.035)
        leg.AddEntry(back[inp],"Simulated background","F")
        leg.AddEntry(back_unc,"Stat. unc.","F")
        leg.AddEntry(data[inp],"Data","PL")
        leg.SetBorderSize(0)
        leg.Draw()
        can.Print(out_fold+"Supplementary/TDJ_DataAndBkg_"+inp+".pdf")
        can.Print(out_fold+"Supplementary/TDJ_DataAndBkg_"+inp+".png")

        if inp=="cHadEFrac":
            can.Print("fig_updated/Figure_011.pdf")
            can.Print("fig_updated/Figure_011.png")

            out_file = TFile("fig_updated/Figure_011.root","RECREATE")
            out_file.cd()
            back[inp].Write("back")
            back_unc.Write("back_unc")
            data[inp].Write("data")
            print "Writing fig_updated/Figure_011.root"
            out_file.Write()
            out_file.Close()

        if inp=="nHadEFrac":
            can.Print("fig_updated/Figure_012.pdf")
            can.Print("fig_updated/Figure_012.png")

            out_file = TFile("fig_updated/Figure_012.root","RECREATE")
            out_file.cd()
            back[inp].Write("back")
            back_unc.Write("back_unc")
            data[inp].Write("data")
            print "Writing fig_updated/Figure_012.root"
            out_file.Write()
            out_file.Close()

        if inp=="nTrackConstituents":
            can.Print("fig_updated/Figure_013.pdf")
            can.Print("fig_updated/Figure_013.png")
            
            out_file = TFile("fig_updated/Figure_013.root","RECREATE")
            out_file.cd()
            back[inp].Write("back")
            back_unc.Write("back_unc")
            data[inp].Write("data")
            print "Writing fig_updated/Figure_013.root"
            out_file.Write()
            out_file.Close()

        can.Close()
        del back_unc
        inp_it+=1

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

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
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
    back.GetYaxis().SetTitle("Events / bin")
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
    legend_xmin = [0.77-0.05,0.425-0.05,0.45-0.05]
    legend_xmax = [0.85-0.05,0.85-0.05,0.85-0.05]
    list_label = ["Data","m_{#chi} = 1000 GeV, c#tau_{0} = 0.5 m","Beam halo enriched data"]
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
            gStyle.SetPadTickX(1)
            gStyle.SetPadTickY(1)
            can = TCanvas("can","can",900,800)
            can.cd()
            #can.SetGrid()
            can.SetLogz()
            can.SetRightMargin(0.16)#(0.12)
            can.SetLeftMargin(0.12)
            can.SetBottomMargin(0.12)#5)            

            print "Is hist[s] empty?"
            print hist[s].GetEntries()

            hist[s].Draw("COLZ")#("CONT4Z")#cont4 or colz
            hist[s].GetYaxis().SetTitle("min #Delta #varphi (jet pairs)")
            hist[s].GetXaxis().SetTitle("Jet tagger output score")
            hist[s].GetXaxis().SetTitleSize(0.04)
            hist[s].GetXaxis().SetTitleOffset(1.3)
            hist[s].GetYaxis().SetTitleSize(0.04)
            hist[s].GetYaxis().SetTitleOffset(1.2)#(1.6)
            hist[s].GetZaxis().SetTitle("Events / bin")
            hist[s].GetZaxis().SetTitleSize(0.04)
            hist[s].GetZaxis().SetTitleOffset(1.2)

            if s in sign:
                drawCMS_simple(-1, "Simulation Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.6,draw_s_only=True,top_marg_lumi=0.975)
            else:
                drawCMS_simple(LUMI, "Supplementary", onTop=True,left_marg_CMS=0.15,custom_spacing=0.35,top_marg_lumi=0.98)

            leg = TLegend(legend_xmin[i], 0.8, legend_xmax[i], 0.875)
            leg.SetTextSize(0.04)
            leg.SetHeader(list_label[i])
            leg.SetBorderSize(0)
            leg.SetFillColorAlpha(18,0.8)
            leg.Draw()
            can.Print(out_fold+"/Supplementary/TDJ_min_dPhi_correlation_"+s+"_supplementary.pdf")
            can.Print(out_fold+"/Supplementary/TDJ_min_dPhi_correlation_"+s+"_supplementary.png")

def cutflow_dict(ERA,decay_mode,label=""):

    #Load root file from gen ntuples
    MAIN = "/nfs/dust/cms/group/cms-llp/v7_calo_AOD/v7_calo_AOD_"+str(ERA)+"_reinterpretation/"
    OUT = out_fold
    OUT+="/cutflow_"+ERA+"/"
    if not os.path.isdir(OUT): os.mkdir(OUT)

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
        LUMI  = lumi["HighMET"]["tot"]

    sign_base = [
        #'SUSY_mh127_ctau500','SUSY_mh127_ctau3000',
        #'SUSY_mh150_ctau500','SUSY_mh150_ctau3000',
        #'SUSY_mh175_ctau500','SUSY_mh175_ctau3000',
        #'SUSY_mh200_ctau500','SUSY_mh200_ctau3000',
        #'SUSY_mh250_ctau500','SUSY_mh250_ctau3000',
        #'SUSY_mh300_ctau500','SUSY_mh300_ctau3000',
        #'SUSY_mh400_ctau500','SUSY_mh400_ctau3000',
        #'SUSY_mh600_ctau500','SUSY_mh600_ctau3000',
        #'SUSY_mh800_ctau500','SUSY_mh800_ctau3000',
        #'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
        #'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
        'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
        #'SUSY_mh1800_ctau500', 'SUSY_mh1800_ctau3000',
    ]

    sign = []
    if decay_mode=="HH":
        #MAIN+="/"
        for s in sign_base:
            sign.append(s+"_HH")
    elif decay_mode=="ZZ":
        #MAIN+="_ZZ/"
        for s in sign_base:
            sign.append(s+"_ZZ")
    elif decay_mode=="HZ":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="ZH":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="all":
        for s in sign_base:
            sign.append(s+"_")
    else:
        print "Invalid decay mode, aborting . . ."
        exit()

    eff = defaultdict(dict)


    #Count gen events
    print "\n"
    print "\n"
    print "opening: ", MAIN

    for d in sign:
        if "SUSY" in d:
            m = samples[d]["mass"]
            eff[d]['m'] = m
            c = samples[d]["ctau"]
            eff[d]['c'] = c

        gen_events = 0
        list_files = []
        for l in samples[d]["files"]:
            list_files.append(MAIN+l+".root")
            print l
            if "SMS" in l:
                name = l
                gen_events += sample[ l ]['nevents']

        eff[d]['gen'] = gen_events
        print list_files
        print gen_events

        #Load data-MC SB for time smearing
        smearing_file = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_"+str(ERA)+"_TtoEM_v5_ntuples_validate_timeRecHits/data_smear_file_CSV_0p8_all_jets"+label+".root"
        timeCBFile = TFile(smearing_file,"READ")
        dataCB = timeCBFile.Get("data_CB")
        mcCB = timeCBFile.Get("back_CB")
        smearCB = dataCB.Clone("smear_cb")
        smearCB.SetParameter(0,dataCB.GetParameter(0))
        smearCB.SetParameter(1,dataCB.GetParameter(1) - mcCB.GetParameter(1))
        smearCB.SetParameter(2, math.sqrt( abs( (dataCB.GetParameter(2)**2) - (mcCB.GetParameter(2)**2) )) )
        smearCB.SetParameter(3,dataCB.GetParameter(3))
        smearCB.SetParameter(4,dataCB.GetParameter(4))
        timeCBFile.Close()

        #Generate sufficient number of smearing numbers before the loop 
        jet_smear_vec = []
        for g in range(gen_events*10):
            jet_smear_vec.append(smearCB.GetRandom())

        print "Created time smearing corrections"

        #Load tensorflow model
        model = keras.models.load_model('nn_inference/tagger_AK4_v3/model.h5')
        
        dnn_score = 0.996
        nTag = []
        nAccJets = []

        #Define cutflow sequence
        trigger_counter = 0
        met_counter = 0
        lep_veto_counter = 0
        pho_veto_counter = 0
        tau_veto_counter = 0
        min_dphi_counter = 0
        met_filters_counter = 0
        cosmic_counter = 0
        beam_halo_counter = 0
        jet_counter = 0

        for l in samples[d]["files"]:
            tmp_file = TFile(MAIN+l+".root","READ")
            tree = tmp_file.Get("tree")
            tree_weight = tree.GetWeight()

            #print "Check if entries are matching gen events:"
            #print tree.GetEntries()
            #print gen_events
            #if tree.GetEntries()!=gen_events:
            #    print "Something went wrong!!!"
            #    continue

            for e in range(0,tree.GetEntries()):#10001):#
                tree.GetEntry(e)

                neg_jets = getattr(tree,"JetsNegative")
                jets = getattr(tree,"Jets")
                met = getattr(tree,"MEt")

                if e%1000 == 0:
                    print "Event n. ", e, "/", tree.GetEntries()

                if not(tree.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or tree.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or tree.HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v):
                    continue
                trigger_counter +=1

                if met.pt<=200:
                    continue
                met_counter += 1

                if tree.nMuonsPassing!=0:
                    continue
                if tree.nElectronsPassing!=0:
                    continue
                lep_veto_counter += 1

                if tree.nPhotonsPassing!=0:
                    continue
                pho_veto_counter += 1

                if tree.nTausPassing!=0:
                    continue
                tau_veto_counter += 1

                if tree.MinJetMetDPhi<=0.5:
                    continue                
                min_dphi_counter += 1

                if not(tree.Flag2_EcalDeadCellTriggerPrimitiveFilter):
                    continue
                if not(tree.Flag2_HBHENoiseFilter):
                    continue
                if not(tree.Flag2_HBHEIsoNoiseFilter):
                    continue
                if not(tree.Flag2_ecalBadCalibFilter):
                    continue
                if not(tree.Flag2_eeBadScFilter):
                    continue
                if not(tree.Flag2_BadPFMuonFilter):
                    continue
                met_filters_counter += 1

                if tree.dt_ecal_dist<0.5:
                    continue
                cosmic_counter += 1

                #Beam halo counter must be at event level
                #Old and inconsistent but that's what we have for the analysis: unsmeared time and unsmeared sigprob and correct acceptance. Check the impact.
                beam_halo_vetoed = False
                for m in range(jets.size()):
                    if jets[m].eta<=-1:
                        continue
                    if jets[m].eta>=1:
                        continue
                    if ERA=="2017":
                        if jets[m].phi>=2.7: continue
                    if ERA=="2018":
                        if jets[m].phi>=0.4 and jets[m].phi<0.9: continue
                    if tree.min_dPhi_jets_eta_1p0_0p996<0.05:
                        if jets[m].timeRecHitsEB>-1 and jets[m].sigprob>0.996 and jets[m].nRecHitsEB<=10:
                            beam_halo_vetoed = True

                if beam_halo_vetoed==True:
                    continue
                beam_halo_counter+=1

                tmp_n_tag = 0
                tmp_acc_jet_counter = 0

                #Going to SR, can skip things not in acceptance
                for n in range(neg_jets.size()):
                    if neg_jets[n].eta<=-1:
                        continue
                    if neg_jets[n].eta>=1:
                        continue

                    if ERA=="2017":
                        if neg_jets[n].phi>=2.7: continue
                    if ERA=="2018":
                        if neg_jets[n].phi>=0.4 and neg_jets[n].phi<0.9: continue

                    jet_counter+=1

                    #variable list
                    v_tmp_n_jet_acceptance = []                                                                                
                    v_tmp_nTrackConstituents = []
                    v_tmp_nSelectedTracks = []
                    v_tmp_timeRecHitsEB = []
                    v_tmp_timeSmeared = []
                    v_tmp_eta = []
                    v_tmp_pt = []
                    v_tmp_eFracRecHitsEB = []
                    v_tmp_nRecHitsEB = []
                    v_tmp_sig1EB = []
                    v_tmp_sig2EB = []
                    v_tmp_ptDEB = []
                    v_tmp_cHadEFrac = []
                    v_tmp_nHadEFrac = []
                    v_tmp_eleEFrac = []
                    v_tmp_photonEFrac = []
                    v_tmp_ptAllTracks = []
                    v_tmp_ptAllPVTracks = []
                    v_tmp_alphaMax = []
                    v_tmp_betaMax = []
                    v_tmp_gammaMax = []
                    v_tmp_gammaMaxEM = []
                    v_tmp_gammaMaxHadronic = []
                    v_tmp_gammaMaxET = []
                    v_tmp_minDeltaRAllTracks = []
                    v_tmp_minDeltaRPVTracks = []
                    v_tmp_EventWeight = []

                    time_rnd_smear = neg_jets[n].timeRecHitsEB + jet_smear_vec[jet_counter]
                    
                    if time_rnd_smear < -1:
                        continue

                    tmp_acc_jet_counter += 1


                    v_tmp_nTrackConstituents.append(neg_jets[n].nTrackConstituents)
                    v_tmp_nSelectedTracks.append(neg_jets[n].nSelectedTracks)
                    v_tmp_timeSmeared.append(time_rnd_smear)
                    v_tmp_eta.append(neg_jets[n].eta)
                    v_tmp_pt.append(neg_jets[n].pt)
                    v_tmp_eFracRecHitsEB.append(neg_jets[n].eFracRecHitsEB)
                    v_tmp_nRecHitsEB.append(neg_jets[n].nRecHitsEB)
                    v_tmp_sig1EB.append(neg_jets[n].sig1EB)
                    v_tmp_sig2EB.append(neg_jets[n].sig2EB)
                    v_tmp_ptDEB.append(neg_jets[n].ptDEB)
                    v_tmp_cHadEFrac.append(neg_jets[n].cHadEFrac)
                    v_tmp_nHadEFrac.append(neg_jets[n].nHadEFrac)
                    v_tmp_eleEFrac.append(neg_jets[n].eleEFrac)
                    v_tmp_photonEFrac.append(neg_jets[n].photonEFrac)
                    v_tmp_ptAllTracks.append(neg_jets[n].ptAllTracks)
                    v_tmp_ptAllPVTracks.append(neg_jets[n].ptAllPVTracks)
                    v_tmp_alphaMax.append(neg_jets[n].alphaMax)
                    v_tmp_betaMax.append(neg_jets[n].betaMax)
                    v_tmp_gammaMax.append(neg_jets[n].gammaMax)
                    v_tmp_gammaMaxEM.append(neg_jets[n].gammaMaxEM)
                    v_tmp_gammaMaxHadronic.append(neg_jets[n].gammaMaxHadronic)
                    v_tmp_gammaMaxET.append(neg_jets[n].gammaMaxET)
                    v_tmp_minDeltaRAllTracks.append(neg_jets[n].minDeltaRAllTracks)
                    v_tmp_minDeltaRPVTracks.append(neg_jets[n].minDeltaRPVTracks)

                    v_tmp_dat_list_smeared = [
                        v_tmp_nTrackConstituents,
                        v_tmp_nSelectedTracks,
                        v_tmp_timeSmeared,
                        v_tmp_eFracRecHitsEB,
                        v_tmp_nRecHitsEB,
                        v_tmp_sig1EB,
                        v_tmp_sig2EB,
                        v_tmp_ptDEB,
                        v_tmp_cHadEFrac,
                        v_tmp_nHadEFrac,
                        v_tmp_eleEFrac,
                        v_tmp_photonEFrac,
                        v_tmp_ptAllTracks,
                        v_tmp_ptAllPVTracks,
                        v_tmp_alphaMax,
                        v_tmp_betaMax,
                        v_tmp_gammaMax,
                        v_tmp_gammaMaxEM,
                        v_tmp_gammaMaxHadronic,
                        v_tmp_gammaMaxET,
                        v_tmp_minDeltaRAllTracks,
                        v_tmp_minDeltaRPVTracks,
                    ]

                    v_tmp_X_smear = np.transpose(np.stack((v_tmp_dat_list_smeared)))
                    v_tmp_probs_smear = model.predict(v_tmp_X_smear)
                    if time_rnd_smear>-1 and v_tmp_probs_smear[:,1][0]>dnn_score:
                        tmp_n_tag+=1

                #Event level: nTag per event
                nTag.append(tmp_n_tag)
                nAccJets.append(tmp_acc_jet_counter)

        #Out of event loop
        print "Events going to jet loop: ", beam_halo_counter
        print len(nAccJets)
        print "n accepted jets: "
        print nAccJets
        print "nTag: "
        print nTag

        m_nj = (np.array(nAccJets)>0)

        nTag = np.array(nTag)

        m_b2 = (nTag>1)
        m_b1 = (nTag==1)
        m_b0 = (nTag==0)

        bin2 = (m_b2).sum()
        bin1 = (m_b1).sum()
        bin0 = (m_b0).sum()

        bin2_stat_unc = np.sqrt( sum(x*x for x in (m_b2) ) ).sum()
        bin1_stat_unc = np.sqrt( sum(x*x for x in (m_b1) ) ).sum()
        bin0_stat_unc = np.sqrt( sum(x*x for x in (m_b0) ) ).sum()

        eff[d]['trigger'] = trigger_counter
        eff[d]['met'] = met_counter
        eff[d]['lep_veto'] = lep_veto_counter
        eff[d]['pho_veto'] = pho_veto_counter
        eff[d]['tau_veto'] = tau_veto_counter
        eff[d]['min_dphi'] = min_dphi_counter
        eff[d]['met_filters'] = met_filters_counter
        eff[d]['cosmic'] = cosmic_counter
        eff[d]['beam_halo'] = beam_halo_counter
        eff[d]['acc_jet'] = (m_nj).sum()

        eff[d]['b0'] = bin0
        eff[d]['b1'] = bin1
        eff[d]['b2'] = bin2
        eff[d]['b0_stat_unc'] = bin0_stat_unc
        eff[d]['b1_stat_unc'] = bin1_stat_unc
        eff[d]['b2_stat_unc'] = bin2_stat_unc

        print 'gen', eff[d]['gen']
        print 'trigger', eff[d]['trigger']
        print 'met', eff[d]['met']
        print 'lep_veto', eff[d]['lep_veto']
        print 'pho_veto', eff[d]['pho_veto']
        print 'tau_veto', eff[d]['tau_veto']
        print 'min_dphi', eff[d]['min_dphi']
        print 'met_filters', eff[d]['met_filters']
        print 'cosmic', eff[d]['cosmic']
        print 'beam_halo', eff[d]['beam_halo']
        print 'acc_jet', eff[d]['acc_jet']
        print 'b0', eff[d]['b0']
        print 'b1', eff[d]['b1']
        print 'b2', eff[d]['b2']
        print "\n"

        with open(OUT+"Dict_cutflow_"+d+label+".yaml","w") as f:
            yaml.dump(eff[d], f)
            f.close()
            print "Info: dictionary written in file "+OUT+"Dict_cutflow_"+d+label+".yaml"


    #Cutflow efficiency in terms of entries
    #SR: time smearing
    #store yaml

def cutflow_table_prev(ERA,decay_mode,label=""):

    OUT = out_fold
    OUT_LIST = []
    if type(ERA) is list:
        for e in ERA:
            OUT_LIST.append(OUT+"/cutflow_"+e+"/")
    else:
        OUT+="/cutflow_"+ERA+"/"

    sign_base = [
        'SUSY_mh127_ctau500','SUSY_mh127_ctau3000',
        'SUSY_mh150_ctau500','SUSY_mh150_ctau3000',
        'SUSY_mh175_ctau500','SUSY_mh175_ctau3000',
        'SUSY_mh200_ctau500','SUSY_mh200_ctau3000',
        'SUSY_mh250_ctau500','SUSY_mh250_ctau3000',
        'SUSY_mh300_ctau500','SUSY_mh300_ctau3000',
        'SUSY_mh400_ctau500','SUSY_mh400_ctau3000',
        'SUSY_mh600_ctau500','SUSY_mh600_ctau3000',
        'SUSY_mh800_ctau500','SUSY_mh800_ctau3000',
        'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
        'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
        'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
        'SUSY_mh1800_ctau500','SUSY_mh1800_ctau3000',
    ]

    sign = []
    if decay_mode=="HH":
        #MAIN+="/"
        for s in sign_base:
            sign.append(s+"_HH")
    elif decay_mode=="ZZ":
        #MAIN+="_ZZ/"
        for s in sign_base:
            sign.append(s+"_ZZ")
    elif decay_mode=="HZ":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="ZH":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="all":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_")
    else:
        print "Invalid decay mode, aborting . . ."
        exit()


    numb = 1
    for d in sign:

        gen = 0
        trigger = 0
        met = 0
        lep_veto = 0
        pho_veto = 0
        tau_veto = 0
        min_dphi = 0
        met_filters = 0
        cosmic = 0
        beam_halo = 0
        b2 = 0

        if type(ERA) is list:
            for o in OUT_LIST:
                #print "Info: dictionary read from file "+o+"Dict_cutflow_"+d+label+".yaml"
                with open(o+"Dict_cutflow_"+d+label+".yaml") as f:
                    results = yaml.load(f, Loader=yaml.Loader)
                    f.close()

                gen += results['gen']
                trigger += results['trigger']
                met += results['met']
                lep_veto += results['lep_veto']
                pho_veto += results['pho_veto']
                tau_veto += results['tau_veto']
                min_dphi += results['min_dphi']
                met_filters += results['met_filters']
                cosmic += results['cosmic']
                beam_halo += results['beam_halo']
                b2 += results['b2']

        else:
            print "Info: dictionary read from file "+OUT+"Dict_cutflow_"+d+label+".yaml"
            with open(OUT+"Dict_cutflow_"+d+label+".yaml") as f:
                results = yaml.load(f, Loader=yaml.Loader)
                f.close()

            gen += results['gen']
            trigger += results['trigger']
            met += results['met']
            lep_veto += results['lep_veto']
            pho_veto += results['pho_veto']
            tau_veto += results['tau_veto']
            min_dphi += results['min_dphi']
            met_filters += results['met_filters']
            cosmic += results['cosmic']
            beam_halo += results['beam_halo']
            b2 += results['b2']

        ct = 0.5 if results['c']==500 else 3
        m = results['m']
        #print 'No selection & ', results['gen'] , ' \\\ '

        print '\\begin{table}[htb]'
        print '\centering'
        print '\\topcaption{Cutflow table for a \PSGczDo signal sample with a mass of '+str(m)+'\GeV and proper decay length of '+str(ct)+'\unit{m}.'
        print '\label{tab:Cutflow_m'+str(m)+'_ct'+str(ct).replace('.','p')+'}'
        print '}'
        print '\\begin{tabular}{|l|c|}'
        print '\hline'
        print 'Selection & Efficiency (\%) \\\ '
        print '\hline '
        print 'Trigger & ', correct_round(trigger,gen) , ' \\\ '
        print '$\ptmiss > 200$\GeV & ', correct_round(met,gen) , ' \\\ '
        print 'Lepton veto & ', correct_round(lep_veto,gen) , ' \\\ '
        print 'Photon veto & ', correct_round(pho_veto,gen) , ' \\\ '
        print 'Tau veto & ', correct_round(tau_veto,gen) , ' \\\ '
        print '$\dphimin > 0.5$ & ', correct_round(min_dphi,gen) , ' \\\ '
        print '\ptmiss noise filters & ', correct_round(met_filters,gen) , ' \\\ '
        #print 'met_filters', results['met_filters']
        print 'Cosmic muon veto & ', correct_round(cosmic,gen) , ' \\\ '
        print 'Beam halo veto & ', correct_round(beam_halo,gen) , ' \\\ '
        #print 'acc_jet', results['acc_jet']
        #print 'b0', results['b0']
        #print 'b1', results['b1']
        print 'Signal region & ', correct_round(b2,gen) , ' \\\ '
        print '\hline'
        print '\end{tabular}'
        print '\end{table}'
        print "\n"

        tab = 'Selection | Efficiency [$\%$] \n'
        tab+= 'Trigger | %s \n' % str(correct_round(trigger,gen))
        tab+= '$p_T^{\\text{miss}} > 200$ GeV | %s \n' % str(correct_round(met,gen))
        tab+= 'Lepton veto | %s \n' % str(correct_round(lep_veto,gen))
        tab+= 'Photon veto | %s \n' % str(correct_round(pho_veto,gen))
        tab+= 'Tau veto | %s \n' % str(correct_round(tau_veto,gen))
        tab+= '$\Delta \phi_{\\text{min}} > 0.5$ | %s \n' % str(correct_round(min_dphi,gen))
        tab+= '$p_T^{\\text{miss}}$ noise filters | %s \n' % str(correct_round(met_filters,gen))
        tab+= 'Cosmic muon veto | %s \n' % str(correct_round(cosmic,gen))
        tab+= 'Beam halo veto | %s \n' % str(correct_round(beam_halo,gen))
        tab+= 'Signal region | %s \n' % str(correct_round(b2,gen))


        print tab
        num_str = "00"+str(numb) if numb<10 else "0"+str(numb)
        outname = "/afs/desy.de/user/l/lbenato/HEPData/exo-21-014-hepdata/input/Table-aux_"+str(num_str)+'.txt'
        outfile = open(outname, 'w')
        outfile.write(tab)
        outfile.close()
        print "Info: " , outname, " written"
        numb+=1

def cutflow_table(ERA,decay_mode,label=""):

    OUT = out_fold
    OUT_LIST = []
    if type(ERA) is list:
        for e in ERA:
            OUT_LIST.append(OUT+"/cutflow_"+e+"/")
    else:
        OUT+="/cutflow_"+ERA+"/"

    sign_base = [
        'SUSY_mh127_ctau500','SUSY_mh127_ctau3000',
        'SUSY_mh150_ctau500','SUSY_mh150_ctau3000',
        'SUSY_mh175_ctau500','SUSY_mh175_ctau3000',
        'SUSY_mh200_ctau500','SUSY_mh200_ctau3000',
        'SUSY_mh250_ctau500','SUSY_mh250_ctau3000',
        'SUSY_mh300_ctau500','SUSY_mh300_ctau3000',
        'SUSY_mh400_ctau500','SUSY_mh400_ctau3000',
        'SUSY_mh600_ctau500','SUSY_mh600_ctau3000',
        'SUSY_mh800_ctau500','SUSY_mh800_ctau3000',
        'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
        'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
        'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
        'SUSY_mh1800_ctau500','SUSY_mh1800_ctau3000',
    ]
    
    mass_list = [127,150,175,200,250,300,400,600,800,1000,1250,1500,1800]
    ctau_list = [500,3000]

    sign = []
    if decay_mode=="HH":
        #MAIN+="/"
        for s in sign_base:
            sign.append(s+"_HH")
    elif decay_mode=="ZZ":
        #MAIN+="_ZZ/"
        for s in sign_base:
            sign.append(s+"_ZZ")
    elif decay_mode=="HZ":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="ZH":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_HZ")
    elif decay_mode=="all":
        #MAIN+="_HZ/"
        for s in sign_base:
            sign.append(s+"_")
    else:
        print "Invalid decay mode, aborting . . ."
        exit()


    numb = 1

    gen = defaultdict(dict)
    trigger = defaultdict(dict)
    met = defaultdict(dict)
    lep_veto = defaultdict(dict)
    pho_veto = defaultdict(dict)
    tau_veto = defaultdict(dict)
    min_dphi = defaultdict(dict)
    met_filters = defaultdict(dict)
    cosmic = defaultdict(dict)
    beam_halo = defaultdict(dict)
    b2 = defaultdict(dict)

    for m in mass_list:
        for ct in ctau_list:
            gen[m][ct] = 0
            trigger[m][ct] = 0
            met[m][ct] = 0
            lep_veto[m][ct] = 0
            pho_veto[m][ct] = 0
            tau_veto[m][ct] = 0
            min_dphi[m][ct] = 0
            met_filters[m][ct] = 0
            cosmic[m][ct] = 0
            beam_halo[m][ct] = 0
            b2[m][ct] = 0


    for d in sign:
        if type(ERA) is list:
            for o in OUT_LIST:
                #print "Info: dictionary read from file "+o+"Dict_cutflow_"+d+label+".yaml"
                with open(o+"Dict_cutflow_"+d+label+".yaml") as f:
                    results = yaml.load(f, Loader=yaml.Loader)
                    f.close()

                m = results['m']
                ct = results['c']
                gen[m][ct] += results['gen']
                trigger[m][ct] += results['trigger']
                met[m][ct] += results['met']
                lep_veto[m][ct] += results['lep_veto']
                pho_veto[m][ct] += results['pho_veto']
                tau_veto[m][ct] += results['tau_veto']
                min_dphi[m][ct] += results['min_dphi']
                met_filters[m][ct] += results['met_filters']
                cosmic[m][ct] += results['cosmic']
                beam_halo[m][ct] += results['beam_halo']
                b2[m][ct] += results['b2']

        else:
            print "Info: dictionary read from file "+OUT+"Dict_cutflow_"+d+label+".yaml"
            with open(OUT+"Dict_cutflow_"+d+label+".yaml") as f:
                results = yaml.load(f, Loader=yaml.Loader)
                f.close()

            m = results['m']
            ct = results['c']
            gen[m][ct] += results['gen']
            trigger[m][ct] += results['trigger']
            met[m][ct] += results['met']
            lep_veto[m][ct] += results['lep_veto']
            pho_veto[m][ct] += results['pho_veto']
            tau_veto[m][ct] += results['tau_veto']
            min_dphi[m][ct] += results['min_dphi']
            met_filters[m][ct] += results['met_filters']
            cosmic[m][ct] += results['cosmic']
            beam_halo[m][ct] += results['beam_halo']
            b2[m][ct] += results['b2']

    for m in mass_list:
        ct_lab = [0.5,3]

        print '\\begin{table}[htb]'
        print '\centering'
        print '\\topcaption{Cutflow table for a \PSGczDo signal sample with a mass of '+str(m)+'\GeV.'# and proper decay length of '+str(ct)+'\unit{m}.'
        print '\label{tab:Cutflow_m'+str(m)+'}'#+'_ct'+str(ct).replace('.','p')+'}'
        print '}'
        print '\\begin{tabular}{|l|c|c|}'
        print '\hline'
        print 'Selection & \multicolumn{2}{c|}{Efficiency (\%)} \\\ '
        print ' & $c\\tau=0.5$\unit{m} & $c\\tau=3$\unit{m} \\\ '
        print '\hline '
        print 'Trigger & ', correct_round(trigger[m][500],gen[m][500]) , ' & ' , correct_round(trigger[m][3000],gen[m][3000]), ' \\\ '
        print '$\ptmiss > 200$\GeV & ', correct_round(met[m][500],gen[m][500]) , ' & ' , correct_round(met[m][3000],gen[m][3000]), ' \\\ '
        print 'Lepton veto & ', correct_round(lep_veto[m][500],gen[m][500]) , ' & ' , correct_round(lep_veto[m][3000],gen[m][3000]), ' \\\ '
        print 'Photon veto & ', correct_round(pho_veto[m][500],gen[m][500]) , ' & ' , correct_round(pho_veto[m][3000],gen[m][3000]), ' \\\ '
        print 'Tau veto & ', correct_round(tau_veto[m][500],gen[m][500]) , ' & ' , correct_round(tau_veto[m][3000],gen[m][3000]), ' \\\ '
        print '$\dphimin > 0.5$ & ', correct_round(min_dphi[m][500],gen[m][500]) , ' & ' , correct_round(min_dphi[m][3000],gen[m][3000]), ' \\\ '
        print '\ptmiss noise filters & ', correct_round(met_filters[m][500],gen[m][500]) , ' & ' , correct_round(met_filters[m][3000],gen[m][3000]), ' \\\ '
        #print 'met_filters', results['met_filters']
        print 'Cosmic muon veto & ', correct_round(cosmic[m][500],gen[m][500]) , ' & ' , correct_round(cosmic[m][3000],gen[m][3000]), ' \\\ '
        print 'Beam halo veto & ', correct_round(beam_halo[m][500],gen[m][500]) , ' & ' , correct_round(beam_halo[m][3000],gen[m][3000]), ' \\\ '
        #print 'acc_jet', results['acc_jet']
        #print 'b0', results['b0']
        #print 'b1', results['b1']
        print 'Signal region & ', correct_round(b2[m][500],gen[m][500]) , ' & ' , correct_round(b2[m][3000],gen[m][3000]), ' \\\ '
        print '\hline'
        print '\end{tabular}'
        print '\end{table}'
        print "\n"

        tab = 'Selection | Efficiency $c\\tau=0.5$ m [$\%$] | Efficiency $c\\tau=3$ m [$\%$] \n'
        tab+= 'Trigger | %s | %s \n' % ( str(correct_round(trigger[m][500],gen[m][500])), str(correct_round(trigger[m][3000],gen[m][3000])) )
        tab+= '$p_T^{\\text{miss}} > 200$ GeV | %s | %s \n' % ( str(correct_round(met[m][500],gen[m][500])) , str(correct_round(met[m][3000],gen[m][3000])) )
        tab+= 'Lepton veto | %s | %s \n' % ( str(correct_round(lep_veto[m][500],gen[m][500])) , str(correct_round(lep_veto[m][3000],gen[m][3000])) )
        tab+= 'Photon veto | %s | %s \n' % ( str(correct_round(pho_veto[m][500],gen[m][500])) , str(correct_round(pho_veto[m][3000],gen[m][3000])) )
        tab+= 'Tau veto | %s | %s \n' % ( str(correct_round(tau_veto[m][500],gen[m][500])) , str(correct_round(tau_veto[m][3000],gen[m][3000])) )
        tab+= '$\Delta \phi_{\\text{min}} > 0.5$ | %s | %s \n' % ( str(correct_round(min_dphi[m][500],gen[m][500])) , str(correct_round(min_dphi[m][3000],gen[m][3000])) )
        tab+= '$p_T^{\\text{miss}}$ noise filters | %s | %s \n' % ( str(correct_round(met_filters[m][500],gen[m][500])) , str(correct_round(met_filters[m][3000],gen[m][3000])) )
        tab+= 'Cosmic muon veto | %s | %s \n' % ( str(correct_round(cosmic[m][500],gen[m][500])) , str(correct_round(cosmic[m][3000],gen[m][3000])) )
        tab+= 'Beam halo veto | %s | %s \n' % ( str(correct_round(beam_halo[m][500],gen[m][500])) , str(correct_round(beam_halo[m][3000],gen[m][3000])) )
        tab+= 'Signal region | %s | %s \n' % ( str(correct_round(b2[m][500],gen[m][500])) , str(correct_round(b2[m][3000],gen[m][3000])) )


        print tab
        num_str = "00"+str(numb) if numb<10 else "0"+str(numb)
        outname = "/afs/desy.de/user/l/lbenato/HEPData/exo-21-014-hepdata/input/Table-aux_"+str(num_str)+'.txt'
        outfile = open(outname, 'w')
        outfile.write(tab)
        outfile.close()
        print "Info: " , outname, " written"
        numb+=1

####plot_s_b(var="Jet_eleEFrac",label="Jet electron energy fraction",nbins=50,bin_min=0,bin_max=1,fig_name="")
####plot_s_b(var="Jet_photonEFrac",label="Jet photon energy fraction",nbins=50,bin_min=0,bin_max=1,fig_name="")
#plot_s_b(var="sigprob",label="Jet tagger output score",nbins=50,bin_min=0,bin_max=1,fig_name="Figure_003-a")
#plot_s_b(var="Jet_cHadEFrac",label="Jet charged hadron energy fraction",nbins=50,bin_min=0,bin_max=1,fig_name="Figure_002-a")
#plot_s_b(var="Jet_nHadEFrac",label="Jet neutral hadron energy fraction",nbins=50,bin_min=0,bin_max=1,fig_name="Figure_002-b")
#plot_s_b(var="Jet_nTrackConstituents",label="Number of jet charged constituents",nbins=50,bin_min=0,bin_max=50,fig_name="Figure_002-c")
#plot_s_b(var="Jet_minDeltaRPVTracks",label="Minimum #DeltaR(jet axis, tracks associated with the PV)",nbins=50,bin_min=0,bin_max=1.2,fig_name="Figure_002-d")
#plot_s_b(var="Jet_timeRecHitsEB",label="Jet time (ns)",nbins=50,bin_min=-1,bin_max=8,fig_name="Figure_002-e")

#plot_dnn_inputs()
#plot_dnn_score()
#plot_dnn_score_ratio_panel(fig_name="Figure_005")

##plot_dnn_modeling()
#plot_mistag_JHEP(fig_name="Figure_006")
#plot_bin1_pred_JHEP(fig_name="Figure_014")
#tag_eff_vs_radius(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="")
#exit()
#plot_mistag_JHEP(fig_name="Figure_005")#_JHEP
#plot_sf_photon()
#plot_sf()
#plot_ntag()
#plot_met_sf()
#additional_signal_plot(ERA="2018")
#compute_ROC_supplementary(ERA="2016")
#compute_ROC_supplementary(ERA="2017")
#compute_ROC_supplementary(ERA="2018")
#tag_eff_vs_ct(ERA="2018")

#tag_eff_vs_eta(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR/",cut="isSR",postfix="_SR")
#tag_eff_vs_eta(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="")

#tag_eff_vs_pt(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR/",cut="isSR",postfix="_SR")
#tag_eff_vs_pt(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="")

#tag_eff_vs_radius(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR/",cut="isSR",postfix="_SR")

#tag_eff_vs_z(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="")

#more bins studies
#tag_eff_vs_topology(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="_merged")
#tag_eff_vs_topology(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="_resolved")
#tag_eff_vs_radius(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="_more_bins")
#tag_eff_vs_z(ERA="2018",ntupledir="/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen/",cut="isMC",postfix="_more_bins")

#plot_ROC_supplementary(ERA="2017",extra_label="_try_all")
#plot_ROC(fig_name="Figure_003-b",x_sampl=0)
#plot_ROC_simplified(fig_name="Figure_003-b_HEPData")
#plot_ROC_simplified(fig_name="Figure_003-b_HEPData_downsampling")

#exit()
#plot_2D(tree_name="tree",v_x="Jets.sigprob",v_y="min_dPhi_jets",cut="",do_radius=False,add_label="")
#cutflow_dict("2016","ZZ",label = "_B-F")
#cutflow_dict("2018","HH",label = "")
#cutflow_table("2017","HZ",label = "")
#cutflow_dict("2018","all",label = "")
#cutflow_table("2017","all",label = "")
cutflow_table(["2016","2017","2018"],"all",label = "")
exit()

CUT = "isSR && nTagJets_0p996_JJ<2"
label ="_BH_correlation"
x="Jets.sigprob"
y="min_dPhi_jets"
#plot_2D(sgn,v_x=x,v_y=y,cut=CUT,add_label=label)
