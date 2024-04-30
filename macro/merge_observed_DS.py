#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TMultiGraph, TGraphAsymmErrors, TSpline, TSpline3
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples

gROOT.SetBatch(True)


path = "plots/Limits_AN/v6_calo_AOD_SR_DS/zPrime/2016_BFGH_2017_2018/Exclusion_"
channels = ["To4b","To2b2nu"]
names = []
l_mZ = [3000,4500]
g = {}
for ch in channels:
    for mZ in l_mZ:
        l_mX = [mZ/2 -50,mZ/10]
        for mX in l_mX:
            filename="zPrime"+ch+"_vs_ctau_mZ_"+str(mZ)+"_mX_"+str(mX)
            names.append(filename)
            print path+filename+".root"
            g_file = TFile(path+filename+".root","READ")
            g_file.cd()
            g[filename] = g_file.Get("m"+str(mX)+"_obs")
            #g.SetDirectory(0)
            print g[filename].Print()
            g_file.Close()


out_file = TFile(path+"zPrime.root","RECREATE")
out_file.cd()
for filename in names:
    g[filename].Write(filename)
print "Writing "+path+"zPrime.root"
out_file.Write()
out_file.Close()



path = "plots/Limits_AN/v6_calo_AOD_SR_DS/HeavyHiggs/2016_BFGH_2017_2018/Exclusion_"
channels = ["To4b","To2b2nu"]
names = []
l_mZ = [400,800]
g = {}
for ch in channels:
    for mZ in l_mZ:
        l_mX = [mZ/2 -50,mZ/10]
        for mX in l_mX:
            filename="HeavyHiggs"+ch+"_vs_ctau_mH_"+str(mZ)+"_mX_"+str(mX)
            names.append(filename)
            print path+filename+".root"
            g_file = TFile(path+filename+".root","READ")
            g_file.cd()
            g[filename] = g_file.Get("m"+str(mX)+"_obs")
            #g.SetDirectory(0)
            print g[filename].Print()
            g_file.Close()


out_file = TFile(path+"HeavyHiggs.root","RECREATE")
out_file.cd()
for filename in names:
    g[filename].Write(filename)
print "Writing "+path+"HeavyHiggs.root"
out_file.Write()
out_file.Close()


'''

path = "plots/Limits_AN/v6_calo_AOD_SR_DS/Stealth/2016_BFGH_2017_2018/Exclusion_"
channels = ["SHH","SYY"]
channels = ["SHH","SYY"]
#Exclusion_StealthSHH_vs_stop_mass_ms_min_225_ctau_1000
names = []
l_ctaus = ["0p01","0p1","1","10","100","1000"]
#colors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
colors = {}
colors["1"] = 856
colors["10"] = 1
colors["100"] = 418 
colors["1000"] = 2 
l_ms = ["100","min_225"]
g = {}

x_theory = np.array([300., 500., 700., 900., 1100., 1300., 1500.])
y_theory = np.array([10.00, 0.609, 0.0783, 0.0145, 0.00335, 0.000887, 0.000257])

gr_th = TGraph(len(x_theory),x_theory,y_theory)
gr_th.SetLineColor(4)
gr_th.SetLineWidth(2)
gr_th.SetLineStyle(2)

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
    c = TCanvas("c", "Exclusion Limits", 900, 675)
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
            leg.AddEntry(g[fl],"c#tau = "+dicty[fl]+" mm",'L')
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
'''
