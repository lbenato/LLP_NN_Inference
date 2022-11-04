#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
#import matplotlib.pyplot as plt

for ev in range(EventNumber.shape[0]):
    
    DT        = np.dstack((DT_x[ev],DT_y[ev],DT_z[ev])).reshape(-1,3)
    ECAL      = np.dstack((ECAL_x[ev],ECAL_y[ev],ECAL_z[ev])).reshape(-1,3)
    CSC       = np.dstack((CSC_x[ev],CSC_y[ev],CSC_z[ev])).reshape(-1,3)

    h_xy_dt       = TH2F("h_xy_dt","",     100, -10, 10, 100, -10, 10)
    h_xy_ecal     = TH2F("h_xy_ecal","",   100, -10, 10, 100, -10, 10)
    h_xy_csc      = TH2F("h_xy_csc","",     100, -10, 10, 100, -10, 10)
            
    h_yz_dt       = TH2F("h_yz_dt","",     100, -10, 10, 100, -10, 10)
    h_yz_ecal     = TH2F("h_yz_ecal","",   100, -10, 10, 100, -10, 10)
    h_yz_csc      = TH2F("h_yz_csc","",     100, -10, 10, 100, -10, 10)
    
    h_xz_dt       = TH2F("h_xz_dt","",     100, -10, 10, 100, -10, 10)
    h_xz_ecal     = TH2F("h_xz_ecal","",   100, -10, 10, 100, -10, 10)
    h_xz_csc      = TH2F("h_xz_csc","",     100, -10, 10, 100, -10, 10)
    
    h_xyz_dt      = TH3F("h_xyz_dt","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
    h_xyz_ecal    = TH3F("h_xyz_ecal","", 100, -10, 10, 100, -10, 10, 100, -10, 10)
    h_xyz_csc     = TH3F("h_xyz_csc","",   100, -10, 10, 100, -10, 10, 100, -10, 10)
    
    root_numpy.fill_hist(h_xy_ecal,ECAL[:,[0,1]])
    root_numpy.fill_hist(h_xz_ecal,ECAL[:,[2,0]])
    root_numpy.fill_hist(h_yz_ecal,ECAL[:,[2,1]])
    root_numpy.fill_hist(h_xyz_ecal,ECAL[:,[2,0,1]])

    h_xy_ecal.SetMarkerColor(2)
    h_xy_ecal.SetMarkerStyle(20)
    h_xz_ecal.SetMarkerColor(2)
    h_xz_ecal.SetMarkerStyle(20)
    h_yz_ecal.SetMarkerColor(2)
    h_yz_ecal.SetMarkerStyle(20)
    h_xyz_ecal.SetMarkerColor(2)
    h_xyz_ecal.SetMarkerStyle(20)
    
    root_numpy.fill_hist(h_xy_csc,CSC[:,[0,1]])
    root_numpy.fill_hist(h_xz_csc,CSC[:,[2,0]])
    root_numpy.fill_hist(h_yz_csc,CSC[:,[2,1]])
    root_numpy.fill_hist(h_xyz_csc,CSC[:,[2,0,1]])
    
    h_xy_csc.SetMarkerColor(800)
    h_xy_csc.SetMarkerStyle(20)
    h_yz_csc.SetMarkerColor(800)
    h_yz_csc.SetMarkerStyle(20)
    h_xz_csc.SetMarkerColor(800)
    h_xz_csc.SetMarkerStyle(20)
    h_xyz_csc.SetMarkerColor(800)
    h_xyz_csc.SetMarkerStyle(20)
    
    root_numpy.fill_hist(h_xy_dt,DT[:,[0,1]])
    root_numpy.fill_hist(h_xz_dt,DT[:,[2,0]])
    root_numpy.fill_hist(h_yz_dt,DT[:,[2,1]])
    root_numpy.fill_hist(h_xyz_dt,DT[:,[2,0,1]])
    
    h_xy_dt.SetMarkerColor(4)
    h_xy_dt.SetMarkerStyle(20)
    h_yz_dt.SetMarkerColor(4)
    h_yz_dt.SetMarkerStyle(20)
    h_xz_dt.SetMarkerColor(4)
    h_xz_dt.SetMarkerStyle(20)
    h_xyz_dt.SetMarkerColor(4)
    h_xyz_dt.SetMarkerStyle(20)

    #xy

    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.AddEntry(h_xy_dt,"DT segments","P")
    leg.AddEntry(h_xy_csc,"CSC segments","P")
    leg.AddEntry(h_xy_ecal,"EB rec hits","P")
    
    can_xy = TCanvas("can_xy","can_xy",900,800)
    can_xy.cd()
    can_xy.SetGrid()
    can_xy.SetRightMargin(0.05)
    h_xy_dt.GetXaxis().SetTitle("x (m)")
    h_xy_dt.GetYaxis().SetTitle("y (m)")
    h_xy_dt.GetYaxis().SetTitleOffset(1.4)
    
    h_xy_dt.Draw("")
    h_xy_csc.Draw("sames")
    h_xy_ecal.Draw("sames")
    
    OUTSTRING = OUT
    OUTSTRING+="data_"
    OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  
    
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xy.Update()
    can_xy.Print(OUTSTRING+'_xy.png')
    can_xy.Print(OUTSTRING+'_xy.pdf')
    can_xy.Close()
    h_xy_dt.Delete()
    h_xy_csc.Delete()
    h_xy_ecal.Delete()
    
    
    #xz
    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.AddEntry(h_xz_dt,"DT segments","P")
    leg.AddEntry(h_xz_csc,"CSC segments","P")
    leg.AddEntry(h_xz_ecal,"EB rec hits","P")
    
    can_xz = TCanvas("can_xz","can_xz",900,800)
    can_xz.cd()
    can_xz.SetGrid()
    can_xz.SetRightMargin(0.05)
    h_xz_dt.GetXaxis().SetTitle("z (m)")
    h_xz_dt.GetYaxis().SetTitle("x (m)")
    h_xz_dt.GetYaxis().SetTitleOffset(1.4)
    
    h_xz_dt.Draw("")
    h_xz_csc.Draw("sames")
    h_xz_ecal.Draw("sames")
    
    OUTSTRING = OUT
    OUTSTRING+="data_"
    OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  
    
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xz.Update()
    can_xz.Print(OUTSTRING+'_xz.png')
    can_xz.Print(OUTSTRING+'_xz.pdf')
    can_xz.Close()
    h_xz_dt.Delete()
    h_xz_csc.Delete()
    h_xz_ecal.Delete()
    
    #yz
    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.AddEntry(h_yz_dt,"DT segments","P")
    leg.AddEntry(h_yz_csc,"CSC segments","P")
    leg.AddEntry(h_yz_ecal,"EB rec hits","P")
    
    can_yz = TCanvas("can_yz","can_yz",900,800)
    can_yz.cd()
    can_yz.SetGrid()
    can_yz.SetRightMargin(0.05)
    h_yz_dt.GetXaxis().SetTitle("z (m)")
    h_yz_dt.GetYaxis().SetTitle("y (m)")
    h_yz_dt.GetYaxis().SetTitleOffset(1.4)
    
    h_yz_dt.Draw("")
    h_yz_csc.Draw("sames")
    h_yz_ecal.Draw("sames")
    
    OUTSTRING = OUT
    OUTSTRING+="data_"
    OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  
    
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_yz.Update()
    can_yz.Print(OUTSTRING+'_yz.png')
    can_yz.Print(OUTSTRING+'_yz.pdf')
    can_yz.Close()
    h_yz_dt.Delete()
    h_yz_csc.Delete()
    h_yz_ecal.Delete()
    
    #xyz
    leg = TLegend(0.75, 0.7, 1., 1.)
    leg.AddEntry(h_xyz_dt,"DT segments","P")
    leg.AddEntry(h_xyz_csc,"CSC segments","P")
    leg.AddEntry(h_xyz_ecal,"EB rec hits","P")
    
    can_xyz = TCanvas("can_xyz","can_xyz",900,800)
    can_xyz.cd()
    can_xyz.SetGrid()
    can_xyz.SetRightMargin(0.05)
    h_xyz_dt.GetXaxis().SetTitle("z (m)")
    h_xyz_dt.GetYaxis().SetTitle("x (m)")
    h_xyz_dt.GetZaxis().SetTitle("y (m)")
    h_xyz_dt.GetYaxis().SetTitleOffset(1.4)
    h_xyz_dt.GetZaxis().SetTitleOffset(1.4)
    
    h_xyz_dt.Draw("")
    h_xyz_csc.Draw("sames")
    h_xyz_ecal.Draw("sames")
    
    OUTSTRING = OUT
    OUTSTRING+="data_"
    OUTSTRING += "run_"+str(int(RunNumber[ev]))+"_lumi_"+str(int(LumiNumber[ev]))+"_ev_"+str(int(EventNumber[ev]))  
    
    leg.Draw()
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextAlign(33)
    latex.SetTextSize(0.04)
    latex.SetTextFont(62)
    latex.DrawLatex(0.20, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.36, 0.96, "Simulation")
    can_xyz.Update()
    can_xyz.Print(OUTSTRING+'_xyz.png')
    can_xyz.Print(OUTSTRING+'_xyz.pdf')
    can_xyz.Close()
    h_xyz_dt.Delete()
    h_xyz_csc.Delete()
    h_xyz_ecal.Delete()
    
