#! /usr/bin/env python                                                                                                                  
import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
gROOT.SetBatch(True)
gStyle.SetOptStat(0)

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.variables import *


NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2017_tf_and_skim_unmerged_no_cuts_compare_JJ_tagger_v3/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_August_2021/v5_calo_AOD_2017_SR/"#_InvertBeamHalo/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_SR/"#_InvertBeamHalo/"
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v5_calo_AOD_2017_WtoMN_noMT/"#JetHT/"#WtoEN_noMT/"#_InvertBeamHalo/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR/"#_InvertBeamHalo/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_cosmic_veto_all_tagged/"
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_cosmic_veto_eta_1p0_tagged/"

#Data
NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR/"#_BeamHalo/"
#MC
#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_SR_v5_ntuples/"#_BeamHalo/"

#NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_2017_Gen/"
LUMI = 59.74*1000

#MAIN_OUT = "plots/v6_calo_AOD_2017_BeamHalo_studies_with_time_RMS/"#_InvertBeamHalo/"
#MAIN_OUT = "plots/v6_calo_AOD_2017_Gen/"#_InvertBeamHalo/"
#eta phi spikes
MAIN_OUT = "plots/v6_calo_AOD_2017_SR_eta_phi/"#_InvertBeamHalo/"
#beam halo
MAIN_OUT = "plots/v6_calo_AOD_2017_BeamHalo_AN/"#_InvertBeamHalo/"

#MAIN_OUT = "plots/v6_calo_AOD_2017_veto_debug/"#_InvertBeamHalo/"
#MAIN_OUT = "plots/v6_calo_AOD_2017_cosmic_veto/"#_InvertBeamHalo/"
#MAIN_OUT = "plots/v5_calo_AOD_2017_WtoMN/"#JetHT/"#WtoEN/"#_InvertBeamHalo/"

#back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
#sgn = ["ggH_MH1000_MS150_ctau1000_XL"]
sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000','SUSY_mh150_pl1000']
sgn = [
    'SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000','SUSY_mh150_pl1000',
    'SUSY_mh127_ctau500', 'SUSY_mh127_ctau3000',
    'SUSY_mh150_ctau500', 'SUSY_mh150_ctau3000',
    'SUSY_mh175_ctau500', 'SUSY_mh175_ctau3000',
    'SUSY_mh200_ctau500', 'SUSY_mh200_ctau3000',
    'SUSY_mh250_ctau500', 'SUSY_mh250_ctau3000',
    'SUSY_mh300_ctau500', 'SUSY_mh300_ctau3000',
    'SUSY_mh400_ctau500', 'SUSY_mh400_ctau3000',
    'SUSY_mh600_ctau500', 'SUSY_mh600_ctau3000',
    'SUSY_mh800_ctau500', 'SUSY_mh800_ctau3000',
    'SUSY_mh1000_ctau500','SUSY_mh1000_ctau3000',
    'SUSY_mh1250_ctau500','SUSY_mh1250_ctau3000',
    'SUSY_mh1500_ctau500','SUSY_mh1500_ctau3000',
    'SUSY_mh1800_ctau500','SUSY_mh1800_ctau3000',
]
#sgn = ["VV","WJetsToLNu","ZJetsToNuNu","TTbarGenMET","QCD"]
#sgn = ["All"]

#sgn = ['SUSY_mh400_pl1000','SUSY_mh300_pl1000','SUSY_mh200_pl1000','SUSY_mh150_pl1000']
#sgn = ["SingleMuon"]
#sgn = ["SingleElectron"]
#sgn = ["EGamma"]
#sgn = ["JetHT"]
sgn = ["HighMET","HighMETBH","SUSY_mh400_ctau500","SUSY_mh1800_ctau500","SUSY_mh1800_ctau3000"]
#sgn = ["HighMET"]
#sgn = ["SUSY_mh400_pl1000"]
#sgn = ["Event277096"]

gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetPadTopMargin(0.06)
gStyle.SetPadRightMargin(0.05)
gStyle.SetPadRightMargin(0.15)

h = {}
num = {}
den = {}
graph = {}
tree = {}
c2 = {}
profX = {}
bins=np.array([0.9,2.5,5,7.5,10,12.5,15,20,30,40,50,60,70,80,90,100,120,140,160,200,250,500,1000,1500])
WEIGHT = "EventWeight"
def pl(sampl,tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",do_radius=False,add_label=""):
    print "cut: ", cut
    r_ecal = 129
    r_hcal = 179
    r_magnet = 295
    r_mb1 = 402
    r_mb4 = 738
    r_TOB3 = 110
    r_TOB2 = 95
    r_TOB1 = 70
    r_TIB3 = 55
    r_TIB2 = 40
    r_TIB1 = 25
    minimum = 0
    maximum = 1
    v_TOB3 = TLine(r_TOB3,minimum,r_TOB3,maximum)
    v_TOB2 = TLine(r_TOB2,minimum,r_TOB2,maximum)
    v_TOB1 = TLine(r_TOB1,minimum,r_TOB1,maximum)
    v_TIB3 = TLine(r_TIB3,minimum,r_TIB3,maximum)
    v_TIB2 = TLine(r_TIB2,minimum,r_TIB2,maximum)
    v_TIB1 = TLine(r_TIB1,minimum,r_TIB1,maximum)
    v_ecal = TLine(r_ecal,minimum,r_ecal,maximum)
    v_hcal = TLine(r_hcal,minimum,r_hcal,maximum)
    v_magnet = TLine(r_magnet,minimum,r_magnet,maximum)
    v_mb1 = TLine(r_mb1,minimum,r_mb1,maximum)
    v_mb4 = TLine(r_mb4,minimum,r_mb4,maximum)
    v_ecal.SetLineColor(2)
    v_hcal.SetLineColor(881)
    v_magnet.SetLineColor(1)
    v_mb1.SetLineColor(801)
    v_mb4.SetLineColor(4)
    v_ecal.SetLineWidth(4)
    v_hcal.SetLineWidth(4)
    v_magnet.SetLineWidth(4)
    v_mb1.SetLineWidth(4)
    v_mb4.SetLineWidth(3)
    v_TIB1.SetLineWidth(3)
    v_TIB2.SetLineWidth(3)
    v_TIB3.SetLineWidth(3)
    v_TOB1.SetLineWidth(3)
    v_TOB2.SetLineWidth(3)
    v_TOB3.SetLineWidth(3)

    v_TIB1.SetLineStyle(4)
    v_TIB2.SetLineStyle(4)
    v_TIB3.SetLineStyle(4)
    v_TOB1.SetLineStyle(4)
    v_TOB2.SetLineStyle(4)
    v_TOB3.SetLineStyle(4)
    v_ecal.SetLineStyle(9)
    v_hcal.SetLineStyle(2)
    v_magnet.SetLineStyle(3)
    v_mb1.SetLineStyle(8)
    v_mb4.SetLineStyle(8)

    for b in sampl:#back+sgn:
        tree[b] = TChain(tree_name)
        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(NTUPLEDIR + ss + '.root'):
                print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            tree[b].Add(NTUPLEDIR + ss + '.root')
        if do_radius:
            h[b] = TH2F(str(b),str(b),len(bins)-1,bins,50,0,1)
        else:
            h[b] = TH2F(str(b),str(b),variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"],variable[v_y]["nbins"],variable[v_y]["min"],variable[v_y]["max"])                
        h[b].Sumw2()
        tree[b].Project(str(b),v_y+":"+v_x,cut+"*"+WEIGHT)
        print "hist content: ", h[b].GetEntries()
        profX[b] = TProfile(h[b].ProfileX("prof"+str(b)))
        h[b].GetXaxis().SetTitle(variable[v_x]["title"])
        h[b].GetYaxis().SetTitle(variable[v_y]["title"])

        if do_radius:
            leg = TLegend(1-0.3, 0.65-0.2, 0.95, 0.9-0.1)
        else:
            leg = TLegend(1-0.3, 0.9-0.2, 0.95, 0.9-0.1)
        leg.SetHeader(samples[b]["label"])
        if do_radius:
            leg.AddEntry(v_TOB3,"Strips", "L")
            leg.AddEntry(v_ecal,"ECAL","L")
            leg.AddEntry(v_hcal,"HCAL","L")
            leg.AddEntry(v_magnet,"solenoid","L")
            leg.AddEntry(v_mb1,"MB1","L")
            leg.AddEntry(v_mb4,"MB4","L")
            
        c2[b] = TCanvas("can","can",1000,800)
        c2[b].cd()
        c2[b].SetGrid()
        if do_radius:
            c2[b].SetLogx()
        #c2[b].SetLogy()
        c2[b].SetLogz()
        h[b].SetMarkerColor(861)
        h[b].SetMarkerSize(.3)
        h[b].SetMarkerStyle(24)
        h[b].Draw("COLZ")#()#
        profX[b].SetLineColor(632+2)
        profX[b].SetMarkerColor(632+2)
        profX[b].SetMarkerStyle(20)
        profX[b].SetMarkerSize(1.2)
        profX[b].SetLineWidth(3)
        profX[b].Draw("PL,sames")

        if do_radius:
            v_ecal.Draw("sames")
            v_hcal.Draw("sames")
            v_magnet.Draw("sames")
            v_mb1.Draw("sames")
            v_mb4.Draw("sames")
            v_TOB3.Draw("sames")
            v_TOB2.Draw("sames")
            v_TOB1.Draw("sames")
            v_TIB3.Draw("sames")
            v_TIB2.Draw("sames")
            v_TIB1.Draw("sames")
        leg.Draw()
        drawCMS_simple(LUMI, "Preliminary", ERA="2017", onTop=True)
        c2[b].Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+'_'+b+add_label+'.png')
        c2[b].Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+'_'+b+add_label+'.pdf')
        c2[b].Close()
        h[b].Delete()

def pl_eff(tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",do_radius=True,add_label=""):
    print "cut: ", cut
    r_ecal = 129
    r_hcal = 179
    r_magnet = 295
    r_mb1 = 402
    r_mb4 = 738
    r_TOB3 = 110
    r_TOB2 = 95
    r_TOB1 = 70
    r_TIB3 = 55
    r_TIB2 = 40
    r_TIB1 = 25

    for b in sgn:#back+sgn:
        tree[b] = TChain(tree_name)
        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(NTUPLEDIR + ss + '.root'):
                print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            tree[b].Add(NTUPLEDIR + ss + '.root')
        num[b] = TH1F(str(b)+"_num",str(b)+"_num",len(bins)-1,bins)
        den[b] = TH1F(str(b)+"_den",str(b)+"_den",len(bins)-1,bins)
        num[b].Sumw2()
        den[b].Sumw2()
        tree[b].Project(str(b)+"_den",v_x,cut+"*"+WEIGHT)
        tree[b].Project(str(b)+"_num",v_x,cut+"*(Jets.sigprob>0.996)*"+WEIGHT)
        #profX[b] = TProfile(h[b].ProfileX("prof"+str(b)))
        #h[b].GetXaxis().SetTitle(variable[v_x]["title"])
        #h[b].GetYaxis().SetTitle(variable[v_y]["title"])

        leg = TLegend(0.15, 0.55-0.1, 0.4, 0.9)
        leg.SetHeader(b)
            
        c2[b] = TCanvas("can","can",1000,800)
        c2[b].cd()
        #c2[b].SetGrid()
        if do_radius:
            c2[b].SetLogx()
        #c2[b].SetLogy()
        #c2[b].SetLogz()
        graph[b] = TGraphAsymmErrors()
        graph[b].BayesDivide(num[b],den[b])
        graph[b].SetMarkerColor(861)
        graph[b].SetLineColor(861)
        graph[b].SetLineWidth(2)
        graph[b].SetMarkerSize(1.4)
        graph[b].SetMarkerStyle(20)
        graph[b].Draw("AP")#()#
        graph[b].GetXaxis().SetTitle(variable[v_x]['title'])
        graph[b].GetYaxis().SetTitle("DNN jet tagging efficiency")
        graph[b].GetXaxis().SetTitleOffset(1.2)
        graph[b].SetMaximum(1.01)
        graph[b].SetMinimum(-0.01)

        if do_radius:
            minimum = 0
            maximum = 1#den[b].GetMaximum()
            v_TOB3 = TLine(r_TOB3,minimum,r_TOB3,maximum)
            v_TOB2 = TLine(r_TOB2,minimum,r_TOB2,maximum)
            v_TOB1 = TLine(r_TOB1,minimum,r_TOB1,maximum)
            v_TIB3 = TLine(r_TIB3,minimum,r_TIB3,maximum)
            v_TIB2 = TLine(r_TIB2,minimum,r_TIB2,maximum)
            v_TIB1 = TLine(r_TIB1,minimum,r_TIB1,maximum)
            v_ecal = TLine(r_ecal,minimum,r_ecal,maximum)
            v_hcal = TLine(r_hcal,minimum,r_hcal,maximum)
            v_magnet = TLine(r_magnet,minimum,r_magnet,maximum)
            v_mb1 = TLine(r_mb1,minimum,r_mb1,maximum)
            v_mb4 = TLine(r_mb4,minimum,r_mb4,maximum)
            v_ecal.SetLineColor(2)
            v_hcal.SetLineColor(881)
            v_magnet.SetLineColor(921)
            v_mb1.SetLineColor(801)
            v_mb4.SetLineColor(4)
            v_ecal.SetLineWidth(4)
            v_hcal.SetLineWidth(4)
            v_magnet.SetLineWidth(4)
            v_mb1.SetLineWidth(4)
            v_mb4.SetLineWidth(3)
            v_TIB1.SetLineWidth(2)
            v_TIB2.SetLineWidth(2)
            v_TIB3.SetLineWidth(2)
            v_TOB1.SetLineWidth(2)
            v_TOB2.SetLineWidth(2)
            v_TOB3.SetLineWidth(2)
            
            v_TIB1.SetLineStyle(4)
            v_TIB2.SetLineStyle(4)
            v_TIB3.SetLineStyle(4)
            v_TOB1.SetLineStyle(4)
            v_TOB2.SetLineStyle(4)
            v_TOB3.SetLineStyle(4)
            v_ecal.SetLineStyle(9)
            v_hcal.SetLineStyle(2)
            v_magnet.SetLineStyle(3)
            v_mb1.SetLineStyle(8)
            v_mb4.SetLineStyle(8)
            
            leg.AddEntry(v_TOB3,"Strips", "L")
            leg.AddEntry(v_ecal,"ECAL","L")
            leg.AddEntry(v_hcal,"HCAL","L")
            leg.AddEntry(v_magnet,"solenoid","L")
            leg.AddEntry(v_mb1,"MB1","L")
            leg.AddEntry(v_mb4,"MB4","L")
            v_ecal.Draw("sames")
            v_hcal.Draw("sames")
            v_magnet.Draw("sames")
            v_mb1.Draw("sames")
            v_mb4.Draw("sames")
            v_TOB3.Draw("sames")
            v_TOB2.Draw("sames")
            v_TOB1.Draw("sames")
            v_TIB3.Draw("sames")
            v_TIB2.Draw("sames")
            v_TIB1.Draw("sames")
            graph[b].Draw("P,sames")#()#
        
        leg.Draw()
        c2[b].Print(MAIN_OUT+'eff_vs_'+v_x.replace(".","_")+'_'+b+add_label+'.png')
        c2[b].Print(MAIN_OUT+'eff_vs_'+v_x.replace(".","_")+'_'+b+add_label+'.pdf')
            
#Beam halo final
lab = "_DNN_0p9"
cutb = selection["SR_veto_bin2_cosmics_tag_0p9"]
pl(sgn,v_x="Jets.nRecHitsEB",v_y="Jets.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets.timeRMSRecHitsEB",v_y="Jets.nRecHitsEB",cut=cutb,add_label=""+lab)
exit()
lab = "_DNN_0p6_j0_j1"
cutb = selection["SR_veto_bin2_cosmics_j"] + " && Jets[0].sigprob>0.6 && Jets[1].sigprob>0.6"
pl(sgn,v_x="Jets[0].phi",v_y="Jets[1].phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets[1].phi",v_y="Jets[0].phi",cut=cutb,add_label=""+lab)

lab = "_DNN_0p8_j0_j1"
cutb = selection["SR_veto_bin2_cosmics_j"] + " && Jets[0].sigprob>0.8 && Jets[1].sigprob>0.8"
pl(sgn,v_x="Jets[0].phi",v_y="Jets[1].phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets[1].phi",v_y="Jets[0].phi",cut=cutb,add_label=""+lab)

lab = "_DNN_0p9_j0_j1"
cutb = selection["SR_veto_bin2_cosmics_j"] + " && Jets[0].sigprob>0.9 && Jets[1].sigprob>0.9"
pl(sgn,v_x="Jets[0].phi",v_y="Jets[1].phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets[1].phi",v_y="Jets[0].phi",cut=cutb,add_label=""+lab)

lab = "_DNN_0p996_j0_j1"
cutb = selection["SR_veto_bin2_cosmics_j"] + " && Jets[0].sigprob>0.996 && Jets[1].sigprob>0.996"
pl(sgn,v_x="Jets[0].phi",v_y="Jets[1].phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets[1].phi",v_y="Jets[0].phi",cut=cutb,add_label=""+lab)
#pl(sgn,v_x="Jets.timeRecHitsEB",v_y="Jets.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
#pl(sgn,v_x="Jets.timeRMSRecHitsEB",v_y="Jets.timeRecHitsEB",cut=cutb,add_label=""+lab)
#pl(sgn,v_x="Jets.phi",v_y="Jets.sigprob",cut=cutb,add_label=""+lab)
#cutb = selection["SR_veto_bin2_cosmics_nj"]
#pl(sgn,v_x="JetsNegative.phi",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)

exit()
lab = "_DNN_0p9"
cutb = selection["SR_veto_bin2_cosmics_tag_nj_0p9"]
pl(sgn,v_x="JetsNegative.timeRecHitsEB",v_y="JetsNegative.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.timeRMSRecHitsEB",v_y="JetsNegative.timeRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.phi",v_y="JetsNegative.nRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.phi",v_y="JetsNegative.timeRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.phi",v_y="JetsNegative.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_y="JetsNegative.phi",v_x="JetsNegative.nRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_y="JetsNegative.phi",v_x="JetsNegative.timeRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_y="JetsNegative.phi",v_x="JetsNegative.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.timeRMSRecHitsEB",v_y="JetsNegative.nRecHitsEB",cut=cutb,add_label=""+lab)

cutb = selection["SR_veto_bin2_cosmics_tag_0p9"]
pl(sgn,v_x="min_dPhi_jets_0p9",v_y="Jets.timeRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_x="min_dPhi_jets_0p9",v_y="Jets.timeRMSRecHitsEB",cut=cutb,add_label=""+lab)
pl(sgn,v_y="Jets.timeRecHitsEB",v_x="min_dPhi_jets_0p9",cut=cutb,add_label=""+lab)
pl(sgn,v_y="Jets.timeRMSRecHitsEB",v_x="min_dPhi_jets_0p9",cut=cutb,add_label=""+lab)
pl(sgn,v_y="Jets.timeRMSRecHitsEB",v_x="Jets.phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets.timeRMSRecHitsEB",v_y="Jets.phi",cut=cutb,add_label=""+lab)
pl(sgn,v_y="Jets.timeRecHitsEB",v_x="Jets.phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets.timeRecHitsEB",v_y="Jets.phi",cut=cutb,add_label=""+lab)
pl(sgn,v_y="Jets.phi",v_x="min_dPhi_jets_0p9",cut=cutb,add_label=""+lab)
pl(sgn,v_x="Jets.phi",v_y="min_dPhi_jets_0p9",cut=cutb,add_label=""+lab)
exit()
pl(sgn,v_x="MEt.phi",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.phi",v_y="MEt.phi",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.phi",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.eFracRecHitsEB",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.nRecHitsEB",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.timeRecHitsEB",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
pl(sgn,v_x="JetsNegative.timeRMSRecHitsEB",v_y="JetsNegative.sigprob",cut=cutb,add_label=""+lab)
exit()

#Plots of eta/phi required by Slava
pl(sgn,v_x="Jets.phi",v_y="Jets.eta",cut="nTagJets_0p996_JJ==1",add_label="_bin1")
pl(sgn,v_x="Jets.phi",v_y="Jets.eta",cut="nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5",add_label="_bin1_mindphi")
pl(sgn,v_x="Jets.phi",v_y="Jets.eta",cut="nTagJets_0p996_JJ==1 && abs(Jets.eta)<1",add_label="_bin1_eta_1p0")
pl(sgn,v_x="Jets.phi",v_y="Jets.eta",cut="nTagJets_0p996_JJ==1 && abs(Jets.eta)<0.5",add_label="_bin1_eta_0p5")
pl(sgn,v_x="Jets.phi",v_y="Jets.nRecHitsEB",cut="nTagJets_0p996_JJ==1",add_label="_bin1")
pl(sgn,v_x="Jets.phi",v_y="Jets.nRecHitsEB",cut="nTagJets_0p996_JJ==1 && MinJetMetDPhi>0.5",add_label="_bin1_mindphi")
pl(sgn,v_x="Jets.phi",v_y="Jets.nRecHitsEB",cut="nTagJets_0p996_JJ==1 && abs(Jets.eta)<1",add_label="_bin1_eta_1p0")
pl(sgn,v_x="Jets.phi",v_y="Jets.nRecHitsEB",cut="nTagJets_0p996_JJ==1 && abs(Jets.eta)<0.5",add_label="_bin1_eta_0p5")

exit()



##2D eff
pl(sgn,v_x="Jets.radiusLLP",v_y="Jets.sigprob",cut="Jets.isGenMatched",)
pl_eff(v_x="Jets.radiusLLP",v_y="Jets.sigprob",cut="Jets.isGenMatched",)

pl(sgn,v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",)
pl_eff(v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",)
exit()


##cosmic
'''
CUT = "isSR && isDT_fit"# && nTagJets_0p996_JJ>=2"
label = "_SR_cosmic_unblinded"
label = "_SR_cosmic_all_eta"#_unblinded"
label = "_SR_cosmic_eta_1p0"#_unblinded"
y = "dt_ecal_dist"
x = "dt_fit_chi2_reduced"
super_list = [ ["HighMETBH"],["HighMET"],["SUSY_mh400_pl1000"],["SUSY_mh300_pl1000"],["SUSY_mh200_pl1000"],["Event277096"] ]
super_list = [ ["HighMET"],]
#super_list = [ ["SUSY_mh400_pl1000"],]
for s in super_list:
    pl(s,tree_name="tree",v_y=y,v_x=x,cut=CUT,do_radius=False,add_label=label)
exit()
'''

#Beam Halo, v6
#CUT = "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && min_dPhi_jets<1 && Jets.nRecHitsEB>0"
#CUT = "isSR && Jets.sigprob>0.1 && Jets.phi<2.5 && Jets.sigprob<0.996 && Jets.nRecHitsEB>0 && JetsCaloAdd.eFracEB2CSC==1"
#CUT = "isSR && Jets.sigprob>0.7 && Jets.phi<2.5 && Jets.sigprob<0.996 && Jets.nRecHitsEB>0 && (JetsCaloAdd.spreadXEB<2 || JetsCaloAdd.spreadYEB<2 || JetsCaloAdd.spreadZEB<2)"

#
CUT = "isSR && Jets.nRecHitsEB>0 && min_dPhi_jets<0.2"
CUT = "isSR && Jets.nRecHitsEB>0 && min_dPhi_jets<0.1"
CUT = "isSR && min_dPhi_jets<0.2 && Jets.nRecHitsEB>1"
label ="_BH"
x="JetsCaloAdd.meanXEB"
y="JetsCaloAdd.nRecHitsEB"


CUT = "isSR && min_dPhi_jets<0.05"
x="JetsCaloAdd.meanXEB"
y="min_dPhi_jets"
#x="Jets.sigprob"
#y = "JetsCaloAdd.spreadZEB"


##new
CUT = "isSR && min_dPhi_jets<1"
CUT = "isSR && JetsCaloAdd.nRecHitsEB>1 && min_dPhi_jets<0.5 && JetsCaloAdd.eFracEB2CSC0p1>0.8"
x="JetsCaloAdd.eFracEB2CSC"
list_x = ["JetsCaloAdd.eFracEB2CSC","JetsCaloAdd.eFracEB2CSC0p1","JetsCaloAdd.eFracEB2CSC0p04"]
y="min_dPhi_jets"
list_x = ["JetsCaloAdd.meanXEB"]
y = "JetsCaloAdd.spreadZEB"


CUT = "isSR && min_dPhi_jets<0.05 && JetsCaloAdd.nRecHitsEB<10 && JetsCaloAdd.eFracEB2CSC0p1>0.8"
CUT = "isSR && min_dPhi_jets<0.05"
list_x = ["Jets.sigprob"]
y = "JetsCaloAdd.spreadXEB"

list_x = ["JetsCaloAdd.meanXEB"]
y = "JetsCaloAdd.meanYEB"
CUT = "min_dPhi_jets_0p9<0.05 && nTagJets_0p996_JJ<2"
CUT = "isSR && nCosmicMuonsOneLeg>0 && nTagJets_0p996_JJ>0"
y = "Jets.sigprob"
list_x = ["JetsCaloAdd.eFracEB2CSC0p04"]


y = "Jets.nRecHitsEB"
list_x = ["Jets.timeRecHitsEB"]#,"Jets.sigprob"]

CUT = "isSR && Jets.sigprob>0.99"
CUT = "isSR && Jets.timeRMSRecHitsEB>5"
CUT = "isSR"
super_list = [ ["HighMETBH"],["HighMET"],["SUSY_mh400_pl1000"],["Event277096"] ]
super_list = [ ["HighMETBH"],["HighMET"],["SUSY_mh400_pl1000"] ]
#super_list = [ ["HighMET"]]
for s in super_list:
    #label = ""
    label = "_timeRMS_cut"
    label = "_no_cut"
    #label = "_inspect_noise"
    #label = "_check_BH_dEta"
    #label = "_spatial_pattern"
    #label = "_BH_check_CSC"
    #label = "_correlation_DNN_dPhi"
    #label = "_min_dPhi_0p05_rechits_less_10_CSC_0p1"
    #label = "_min_dPhi_0p05_rechits_less_10_bin1"
    for x in list_x:
        pl(s,tree_name="tree",v_y=y,v_x=x,cut=CUT,do_radius=False,add_label=label)
#
exit()

##
#this plot with these selections really shows beam halo!
CUT = "isSR && min_dPhi_jets<0.1"
label ="_BH"
x="JetsCaloAdd.meanXEB"
y="JetsCaloAdd.meanYEB"
##
pl(tree_name="tree",v_y=y,v_x=x,cut=CUT,do_radius=False,add_label=label)
#
exit()



#These plots show there's correlation bw d phi and tagging score in BH, not in data or signal
CUT = "isSR"
label ="_BH"
x="Jets.sigprob"#"meanXEB"
y="min_dPhi_jets"#"meanYEB"
pl(tree_name="tree",v_y=y,v_x=x,cut=CUT,do_radius=False,add_label=label)
#
exit()


##studies of eta/phi and efficiency, v5

#pl(tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="Jets.isGenMatchedCaloCorr",do_radius=True)
#pl(tree_name="tree",v_x="Jets.eta",v_y="Jets.sigprob",cut="(HT>0)",do_radius=False)
CUT = "(HT>0)"
#CUT = "(( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1))"
#CUT = "( ( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && MinJetMetDPhi>0.5)"
#CUT = "(Jets[0].sigprob>0.996 && MinJetMetDPhi>0.5)"
#CUT = "(Jets[0].sigprob>0.996)"
CUT = "(nTagJets_0p996_JJ==1)"
#CUT = "( ( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1) && MinJetMetDPhi>0.5 && Jets[0].sigprob>0.996)"
#CUT = "(abs(Jets[0].eta)<1)"
#pl_eff()
label = ""
#label = "__MinDPhi_0p5_tagged_jets"
#label = "_tagged_jets_B-F"
#label = "_tagged_jets"
label = "_1_tagged_jet"
#label = "_eta_smaller_1p0"

#CUT = "(nCHSJets==0)"
#label = "_0_jets"

#CUT = "(nCHSJets>0)"
#label = "_at_lest_1_jet"

#CUT = "(nCHSJets>6)"
#label = "_at_lest_7_jets"

#CUT = "(MEt.pt<250)"
#label = "_MET_less_250"

#CUT = "(nCHSJetsAcceptanceCalo==0)"
#label = "_0_jets_AcceptanceCalo"

#CUT = "(nCHSJetsAcceptanceCalo>0)"
#label = "_at_lest_1_jet_AcceptanceCalo"

#pl(tree_name="tree",v_y="EcalRecHitsAK4.eta",v_x="EcalRecHitsAK4.phi",cut=CUT,do_radius=False,add_label=label)#,add_label="_MinDPhi_0p5")#"_tagged_jets_MinDPhi_0p5")#

v = "eta"
pl(tree_name="tree",v_y="Jets."+v,v_x="Jets.phi",cut=CUT,do_radius=False,add_label=label)#,add_label="_MinDPhi_0p5")#"_ta#pl(tree_name="tree",v_y="Jets.sigprob",v_x="EcalRecHitsAK4.phi",cut=CUT,do_radius=False,add_label=label)#,add_label="_MinDPhi_0p5")#"_tagged_jets_MinDPhi_0p5")#
exit()


y_list = ["nTrackConstituents","nRecHitsEB","eFracRecHitsEB","eta","timeRecHitsEB"]
for y in y_list:
    pl(tree_name="tree",v_x="Jets[0].eta",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    pl(tree_name="tree",v_x="Jets[0].phi",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)

exit()



y_list_tagged = ["eta"]
y_list_tagged = ['eta','nTrackConstituents','nSelectedTracks', 'timeRecHitsEB', 'eFracRecHitsEB', 'nRecHitsEB', 'sig1EB', 'sig2EB', 'ptDEB', 'cHadEFrac', 'nHadEFrac', 'eleEFrac', 'photonEFrac', 'ptAllTracks', 'ptAllPVTracks', 'alphaMax', 'betaMax', 'gammaMax', 'gammaMaxEM', 'gammaMaxHadronic', 'gammaMaxET', 'minDeltaRAllTracks', 'minDeltaRPVTracks']
#y_list_tagged = ['nRecHitsEB']

#CUT = "( ( isMC ? 1 : (RunNumber>=319077 ? nCHSJets_in_HEM_pt_30_all_eta==0 : 1)) && Jets[0].sigprob>0.996)"
CUT = "(Jets[0].sigprob>0.996)"
#CUT = "(HT>0)"
label = ""
label = "_tagged_jets"
#CUT = "(Jets[0].sigprob>0.0)"
#label = "_tagged_jets_0p0"


for y in y_list_tagged:
    pl(tree_name="tree",v_x="Jets[0].eta",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    pl(tree_name="tree",v_x="Jets[0].phi",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)

exit()



y_list_log = ["sigprob"]
CUT = "(( (RunNumber>=319077 && !isMC ) ? (nCHSJets_in_HEM_pt_30_all_eta==0) : 1))"
CUT = "( isMC ? 1 : ( RunNumber>=319077 ? nCHSJets_in_HEM_pt_30_all_eta==0 : 1) )"
CUT = "(HT>0)"
for y in y_list_log:
    pl(tree_name="tree",v_x="Jets[0].eta",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    pl(tree_name="tree",v_x="Jets[0].phi",v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)
    #for x in y_list_tagged:
    #    pl(tree_name="tree",v_x="Jets[0]."+x,v_y="Jets[0]."+y,cut=CUT,do_radius=False,add_label=label)

exit()


#This with and without HEM
'''
pl(tree_name="tree",v_x="MEt.phi",v_y="Jets[0].phi",cut=CUT,do_radius=False,add_label=label)]
'''













'''
chain = TChain("tree")
##chain.Add("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/prova_"+sampl+".root")
chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2017_tf_and_skim_unmerged_no_cuts_compare_JJ/QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8-v1.root")
#chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2017_tf_and_skim_unmerged_no_cuts_compare_JJ/TChiHH_mass400_pl1000.root")
#chain.Add("/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2017_tf_and_skim_unmerged_no_cuts_compare_JJ/GluGluH2_H2ToSSTobbbb_MH-1000_MS-150_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC_Tranche3_v2.root")

jet = TH1F("jet","", 50, 0, 3.2)
jet_b = TH1F("jet_b","", 50, 0, 3.2)
f_jet = TH1F("f_jet","", 50, 0, 3.2)
f_jet_b = TH1F("f_jet_b","", 50, 0, 3.2)
jet.Sumw2()
jet_b.Sumw2()
f_jet.Sumw2()
f_jet_b.Sumw2()

chain.Project("jet", "MinJetMetDPhi", "nCHSJets>0")
chain.Project("jet_b", "MinJetMetDPhiBarrel", "nCHSJetsAcceptanceCalo>0")
chain.Project("f_jet", "MinFatJetMetDPhi", "nCHSFatJets>0")
chain.Project("f_jet_b", "MinFatJetMetDPhiBarrel", "nCHSFatJetsAcceptanceCalo>0")

jet.SetLineColor(1)
jet_b.SetLineColor(8)
f_jet.SetLineColor(4)
f_jet_b.SetLineColor(2)

jet.SetLineWidth(3)
jet_b.SetLineWidth(3)
f_jet.SetLineWidth(3)
f_jet_b.SetLineWidth(3)

jet.SetLineStyle(2)
f_jet.SetLineStyle(2)
jet.SetTitle("")
jet_b.SetTitle("")
f_jet.SetTitle("")
f_jet_b.SetTitle("")

jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")
f_jet_b.GetXaxis().SetTitle("min #Delta #varphi (jet,MET)")

leg = TLegend(0.3, 0.8, .6, 1.)
leg.SetHeader(sampl)
leg.AddEntry(jet,"AK4","L")
leg.AddEntry(jet_b,"AK4 barrel","L")
leg.AddEntry(f_jet,"AK8","L")
leg.AddEntry(f_jet_b,"AK8 barrel","L")

if sampl=="QCD":
    jet.Draw("HISTO")
f_jet.Draw("HISTO,same")
f_jet_b.Draw("HISTO,same")
f_jet.Draw("HISTO,same")
jet_b.Draw("HISTO,same")
jet.Draw("HISTO,same")
leg.Draw()
can.Print('MinDPhi_'+sampl+'.png')
can.Print('MinDPhi_'+sampl+'.pdf')
'''
