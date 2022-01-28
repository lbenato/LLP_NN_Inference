#! /usr/bin/env python
import os, multiprocessing
import copy
import math
#import uproot
import yaml
import root_numpy
import numpy as np
import pandas as pd
#import mplhep as hep
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
gROOT.SetBatch(True)
gStyle.SetOptStat(0000)
#plt.style.use(hep.style.CMS)

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
    g.Draw("AL")
    p.SetMarkerStyle(20)
    p.SetMarkerSize(1.5)
    p.SetMarkerColor(4)
    p.Draw("P,sames")
    g.SetMaximum(1.01)
    g.SetMinimum(0.5)
    g.GetXaxis().SetRangeUser(0.00001,0.01)
    drawCMS_simple(-1, "Simulation", onTop=True)
    leg = TLegend(0.45, 0.7-0.2, 0.65+0.1, 0.85-0.2)
    leg.SetTextSize(0.035)
    leg.AddEntry(g,"ROC curve","L")
    leg.AddEntry(p,"DNN>0.996 w. p.","P")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print("fig/TDJ_ROC.pdf")
    can.Print("fig/TDJ_ROC.png")

def plot_s_b():
    '''s and b'''
    inp_file = "/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v4_calo_AOD_2018_dnn_5Upsampling_0p25Background_Final/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    #"/nfs/dust/cms/user/lbenato/ML_LLP/GraphNetJetTaggerCalo/model_weights/v5_calo_AOD_2018_dnn_x-check/SUSY/AK4jets/model_FCN_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1/test_score_2_NoHCAL_MinimalShapesEB_Adam_ReLU_1000epochs_patience100_batch_size_512_dropout_0p1_signal_matched_Jet_isGenMatchedCaloCorrLLPAccept.h5"
    store = pd.HDFStore(inp_file)
    df_test = store.select("df")
    print("    Remove negative weights at testing!!!!!!")
    df_test = df_test.loc[df_test['EventWeight']>=0]
    var = "sigprob"
    weight = "EventWeightNormalized"
    back = np.array(df_test[var].loc[df_test["is_signal"]==0].values)
    sign = np.array(df_test[var].loc[df_test["Jet_isGenMatchedCaloCorrLLPAccept"]==1].values)
    back_w = np.array(df_test[weight].loc[df_test["is_signal"]==0].values)
    sign_w = np.array(df_test[weight].loc[df_test["Jet_isGenMatchedCaloCorrLLPAccept"]==1].values)
    hb = TH1F("b","",50,0,1)
    hs = TH1F("s","",50,0,1)
    root_numpy.fill_hist(hb,back,back_w)
    root_numpy.fill_hist(hs,sign,sign_w)
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
    hb.GetXaxis().SetTitle("Jet tagger output score")
    hb.GetYaxis().SetTitle("Events")
    hb.GetXaxis().SetTitleSize(0.04)
    hb.GetYaxis().SetTitleSize(0.04)
    hb.Draw("HISTO")
    hs.Draw("HISTO,sames")
    drawCMS_simple(-1, "Simulation", onTop=True)
    leg = TLegend(0.45, 0.7, 0.65, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(hb,"background","F")
    leg.AddEntry(hs,"signal","F")
    leg.SetBorderSize(0)
    leg.Draw()
    can.Print("fig/TDJ_SigAndBkg.pdf")
    can.Print("fig/TDJ_SigAndBkg.png")
    

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
            eff[k].SetMarkerColor(colors[k])
            eff[k].SetMarkerSize(1.3)
            eff[k].SetLineColor(colors[k])
            eff[k].SetFillColorAlpha(colors[k],0.5)
            #eff[k].SetFillStyle(3001)
            eff[k].SetLineWidth(2)
            eff[k].SetLineStyle(lines[k])
            eff[k].GetYaxis().SetRangeUser(-0.0001,maxeff)
            np = eff[k].GetN()
            for i in range(0,np):
                y_val = eff[k].GetPointY(i)
                eff[k].SetPointEYhigh(i, math.sqrt(eff[k].GetErrorYhigh(i)**2 + pow(max_unc*eff[k].GetPointY(i),2) ))
                eff[k].SetPointEYlow(i, math.sqrt(eff[k].GetErrorYlow(i)**2 + pow(max_unc*eff[k].GetPointY(i),2) ))
                #eff[k].SetPointEXhigh(i, eff[k].GetErrorXhigh(i))#+n_g*0.01)#not needed
                #eff[k].SetPointEXlow(i, eff[k].GetErrorXlow(i))#+n_g*0.01)
                #eff[k].SetPointX(i, eff[k].GetPointX(i)+n_g*0.01)

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
        
        drawCMS_simple(LUMI, "Preliminary", ERA=era_label, onTop=True)
        #leg.Draw()
        can.Print("fig/MistagEfficiency"+plot_label+".pdf")
        can.Print("fig/MistagEfficiency"+plot_label+".png")
        can.Close()

    eras = ["2016_G-H","2017","2018"]
    LUMI_G2018 = 111941.400399
    era_label="2016G-2018"
    plot_label="2016GTo2018"
    do_mistag(eras,LUMI_G2018,era_label=era_label,plot_label=plot_label)

    eras = ["2016_B-F"]
    era_label="2016B-F"
    plot_label="2016BF"
    LUMI_BF = 137478.722953 - 111941.400399
    do_mistag(eras,LUMI_BF,era_label=era_label,plot_label=plot_label)

def plot_sf():
    '''SF'''
    FOLD = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/v6_calo_AOD_%s_ZtoLLPho_v5_ntuples/"
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
    drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    leg.AddEntry(eff_b,"MC background","PF")
    leg.AddEntry(eff_d,"data","PL")
    leg.Draw()
    can.Print("fig/TDJSignalEfficiency.pdf")
    can.Print("fig/TDJSignalEfficiency.png")
    can.Close()


    #Add discrepancy wrt 2 ns in quadrature
    g = TGraphAsymmErrors(ratio.GetNbinsX()-1)
    for b in range(1,ratio.GetNbinsX()+1):
        i = b-1
        print b, ratio.GetBinCenter(b), ratio.GetBinContent(b), ratio.GetBinError(b)
        diff = abs(ratio.GetBinContent(b)-ratio_2ns.GetBinContent(b))/ratio.GetBinContent(b)
        print "point: ", b
        g.SetPointX(i,ratio.GetBinCenter(b))
        g.SetPointY(i,ratio.GetBinContent(b))
        g.SetPointEXhigh(i, 0.5)
        g.SetPointEXlow(i, 0.5)
        g.SetPointEYhigh(i, math.sqrt(ratio.GetBinError(b)**2 + pow(diff*ratio.GetBinContent(b),2) ))
        g.SetPointEYlow(i, math.sqrt(ratio.GetBinError(b)**2 + pow(diff*ratio.GetBinContent(b),2) ))

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
    g.Draw("APE2")
    g.GetXaxis().SetTitle("Jet #eta")
    g.GetYaxis().SetTitle("Data/MC scale factor")
    g.GetXaxis().SetTitleSize(0.04)
    g.GetYaxis().SetTitleSize(0.04)
    drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    #leg.AddEntry(ratio,"data","PL")
    #leg.Draw()
    can.Print("fig/TDJSignalEfficiencyScaleFactor.pdf")
    can.Print("fig/TDJSignalEfficiencyScaleFactor.png")
    can.Close()

def plot_ntag():
    '''nTag'''

    bins = np.array([0.,1.,2.])
    LUMI = 137478.722953


    fake = TH1F("","",3,-0.5,2.5)
    yp = np.array([0.,939.01,0.149])
    ep = np.array([0.,424.91,0.077])

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
    drawCMS_simple(LUMI, "Preliminary", ERA="", onTop=True)
    leg.Draw()
    can.Print("fig/nTDJ.pdf")
    can.Print("fig/nTDJ.png")
    can.Close()


#plot_s_b()
#plot_ROC()
##plot_dnn_modeling()
plot_mistag()
plot_sf()
plot_ntag()
