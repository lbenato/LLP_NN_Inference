#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
import yaml
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TMultiGraph, TGraphAsymmErrors, TSpline, TSpline3, TGraphErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox
from collections import defaultdict, OrderedDict

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples

gROOT.SetBatch(True)
ch = "2b2nu"
#ch = "4b"
year = "2016"
#year = "2017"
year = "2018"
base_path = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_%s_SR_DS/SUSY/zPrime/datacards/"
path = "/afs/desy.de/user/l/lbenato/LLP_code_slc7/CMSSW_10_2_18/src/HiggsAnalysis/CombinedLimit/v6_calo_AOD_"+year+"_SR_DS/SUSY/zPrime/datacards/"
filename = "ctau_weights_dict_zPrimeTo"+ch+"_mZ4500_mX2200.yaml"
#filename = "ctau_weights_dict_zPrimeTo"+ch+"_mZ4500_mX450.yaml"
#filename = "ctau_weights_dict_zPrimeTo"+ch+"_mZ3000_mX1450.yaml"
#filename = "ctau_weights_dict_zPrimeTo"+ch+"_mZ3000_mX300.yaml"

with open(path+filename,"r") as f:
    pre_results = yaml.load(f, Loader=yaml.Loader)
    f.close()

sign_list = pre_results.keys()
l_ct = np.sort(np.array( pre_results[ sign_list[0] ].keys()  ))

years = ["2016","2017","2018"]
full_results = defaultdict(dict)


#list_sign = ['zPrimeTo4b_mZ4500_mX450_ct1000']
list_sign = pre_results.keys()

for k in list_sign:
    for p in range(len(l_ct)):
        full_results[k][l_ct[p]] = np.array([])

for i, y in enumerate(years):
    print "load: ",(base_path % y+filename)
    with open( (base_path % y) +filename,"r") as f:
        results = yaml.load(f, Loader=yaml.Loader)
        for k in list_sign:
            for p in range(len(l_ct)):
                full_results[k][l_ct[p]] = np.concatenate(( full_results[k][l_ct[p]], results[k][ l_ct[p] ]))
        f.close()


gr = {}
colors = [1,2,4,8,401,880,25]
c = TCanvas("c", "Exclusion Limits", 1400, 675)
leg = TLegend(0., 0.625, 0.35, 0.9)
leg.SetBorderSize(0)
leg.SetTextSize(0.04)

for i, k in enumerate(full_results.keys()):
    gr[k] = TGraphErrors(len(l_ct))
    gr[k].SetTitle("")
    for p in range(len(l_ct)):
        #print p, l_ct[p], full_results[k][ l_ct[p] ].sum()
        stat_unc = np.sqrt( sum(x*x for x in full_results[k][ l_ct[p] ]  ) ).sum()
        ev_yield = full_results[k][ l_ct[p] ].sum()
        if stat_unc/ev_yield<0.75:
            gr[k].SetPoint(p, l_ct[p], ev_yield)
            gr[k].SetPointError(p, 0., stat_unc )
    gr[k].SetLineColor(colors[i])
    gr[k].SetMarkerColor(colors[i])
    gr[k].SetMarkerStyle(20)
    gr[k].SetMaximum(10)
    if ch=="2b2nu":
        gr[k].SetMaximum(1000)
    gr[k].SetMinimum(0.001)
    axis = gr[k].GetXaxis()
    axis.SetLimits(1,100000)
    if i==0:
        gr[k].Draw("APL")
    else:
        gr[k].Draw("PL sames")
    leg.AddEntry(gr[k],k,'PL')
    leg.Draw()

c.SetLogx()
c.SetLogy()
c.Print(path+"ctau_check.pdf")
c.Print(path+"ctau_check.png")

exit()

for ch in channels:
    dicty = {}

    for ms in l_ms:
        for ct in l_ctaus:
            filename="Stealth"+ch+"_vs_stop_mass_ms_"+str(ms)+"_ctau_"+str(ct)
            if os.path.isfile(path+filename+".root"):
                g_file = TFile(path+filename+".root","READ")
                g_file.cd()
                if g_file.Get("ms_"+str(ms)+"_obs").GetN()>=2:
                    names.append(filename)
                    g[filename] = g_file.Get("ms_"+str(ms)+"_obs")
                    dicty[filename] = ct.replace("p",".")
                    #print g[filename].Print()
                g_file.Close()
    #here make a canvas
    out_file = TFile(path+"Stealth"+ch+".root","RECREATE")
    out_file.cd()
    for filename in names:
        print filename
        g[filename].Write(filename)

    
    c100 = TCanvas("c100", "Exclusion Limits", 900, 675)
    c100.GetPad(0).SetBottomMargin(0.12)
    c100.GetPad(0).SetTopMargin(0.08)
    c100.GetPad(0).SetRightMargin(0.05)
    c100.GetPad(0).SetTicks(1, 1)
    c100.GetPad(0).SetLogy()
    c.GetPad(0).SetBottomMargin(0.12)
    c.GetPad(0).SetTopMargin(0.08)
    c.GetPad(0).SetRightMargin(0.05)
    c.GetPad(0).SetTicks(1, 1)
    c.GetPad(0).SetLogy()
    count100 = 0
    count = 0
    leg100 = TLegend(0.6, 0.625, 0.93, 0.9)
    leg100.SetBorderSize(0)
    leg100.SetHeader("#splitline{95% CL observed limits}{"+ch+" ; m_{S} = 100 GeV}")
    leg100.SetTextSize(0.04)
    leg = TLegend(0.6, 0.625, 0.93, 0.9)
    leg.SetBorderSize(0)
    leg.SetHeader("#splitline{95% CL observed limits}{"+ch+" ; m_{S} = (m_{#tilde{t}} - 225) GeV}")
    leg.SetTextSize(0.04)
    for fl in names:
        if ch not in fl:
            continue
        if "ms_100" in fl:
            c100.cd()
            if g[fl].GetN()<2:
                continue
            g[fl].GetXaxis().SetTitle("m_{#tilde{t}} (GeV)")
            g[fl].GetYaxis().SetTitle("#sigma (pb)")
            g[fl].GetXaxis().SetTitleSize(0.048)
            g[fl].GetYaxis().SetTitleSize(0.048)
            g[fl].GetXaxis().SetTitleOffset(0.9)
            g[fl].GetYaxis().SetTitleOffset(0.85)
            g[fl].SetLineColor(colors[dicty[fl]])
            g[fl].SetLineWidth(2)
            g[fl].SetMaximum(1001)
            g[fl].SetMinimum(0.00009)
            axis = g[fl].GetXaxis()
            axis.SetLimits(199.,1600)

            if count100==0:
                print "Draw: ", fl
                g[fl].Draw("AL")
            else:
                print "Draw: ", fl
                g[fl].Draw("L,sames")
            leg100.AddEntry(g[fl],"c#tau = "+dicty[fl]+" mm",'L')
            count100+=1
        else:
            c.cd()
            if g[fl].GetN()<2:
                continue
            g[fl].GetYaxis().SetTitle("#sigma (pb)")
            g[fl].GetXaxis().SetTitle("m_{#tilde{t}} (GeV)")
            g[fl].GetXaxis().SetTitleSize(0.048)
            g[fl].GetYaxis().SetTitleSize(0.048)
            g[fl].GetXaxis().SetTitleOffset(0.9)
            g[fl].GetYaxis().SetTitleOffset(0.85)
            g[fl].SetLineColor(colors[dicty[fl]])
            g[fl].SetLineWidth(2)
            g[fl].SetMaximum(1001)
            g[fl].SetMinimum(0.00009)
            axis = g[fl].GetXaxis()
            axis.SetLimits(199.,1600)
            if count==0:
                print "Draw: ", fl
                g[fl].Draw("AL")
            else:
                print "Draw: ", fl
                g[fl].Draw("L,sames")
            count+=1

    out_file.cd()
    
    leg100.AddEntry(gr_th,"Theory",'L')
    leg.AddEntry(gr_th,"Theory",'L')

    c100.cd()
    gr_th.Draw("L,sames")
    c100.SetLogy()
    leg100.Draw()
    c100.Write("c100")
    c100.Print(path+"Stealth"+ch+"_ms100.pdf")
    c100.Print(path+"Stealth"+ch+"_ms100.png")

    c.cd()
    gr_th.Draw("L,sames")
    c.SetLogy()
    leg.Draw()
    c.Write("c")
    c.Print(path+"Stealth"+ch+"_ms225.pdf")
    c.Print(path+"Stealth"+ch+"_ms225.png")

    print "Writing "+path+"Stealth"+ch+".root"
    out_file.Write()
    out_file.Close()
