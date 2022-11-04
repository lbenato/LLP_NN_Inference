#! /usr/bin/env python
import os, multiprocessing
import copy
import math
import uproot
import yaml
import numpy as np
import root_numpy
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1F, TH2F, THStack, TGraph, TH3F, TF1, TRatioPlot, TGraph, TGraphAsymmErrors
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TProfile

gROOT.SetBatch(True)
gStyle.SetOptStat(0000)

INDIR = '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018/'
#INDIR_c = '/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018_cross_check/'
#INDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018_noCosmicMuons/"
#INDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2018_2stations/"
cosmic_files = [
    #'/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_cosmics_2017/CosmicsRun2017F-CosmicSP-PromptReco-v1.root',

    INDIR+'TKCosmic_38T_p20-3000-RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v8.root',
    INDIR+'LooseMuCosmic_38T_p5-3000-RunIIAutumn18DR-NoPU_102X_upgrade2018_realistic_v15-v9.root',
]


OUT = "/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/plots/cosmic_studies_MC/"
if not os.path.isdir(OUT): os.mkdir(OUT)

list_of_variables = ["*"]


hd = TH1F("","hd",1,-0.5,0.5)
hn = TH1F("","hn",1,-0.5,0.5)
hn_pass = TH1F("","hn_pass",1,-0.5,0.5)
ho = TH1F("","ho",1,-0.5,0.5)
ho_2016BF = TH1F("","ho_2016BF",1,-0.5,0.5)
ho_2016GH = TH1F("","ho_2016GH",1,-0.5,0.5)
ho_2017 = TH1F("","ho_2017",1,-0.5,0.5)
ho_2018 = TH1F("","ho_2018",1,-0.5,0.5)
hd.Sumw2()
hn.Sumw2()
hn_pass.Sumw2()
ho.Sumw2()
ho_2016BF.Sumw2()
ho_2016GH.Sumw2()
ho_2017.Sumw2()
ho_2018.Sumw2()

den = 0.
num = 0.
num2 = 0.
num3 = 0.
num4 = 0.
obs_2016BF = 2
obs_2016GH = 2
obs_2017 = 1
obs_2018 = 1 
obs = 6


EventNumberPassingDen = np.array([])
EventNumberAll = np.array([])

EventNumberAll_c = np.array([])


EventNumber = np.array([])
LumiNumber = np.array([])
RunNumber = np.array([])
GenMuon_pt = np.array([])
RecoMuon_pt = np.array([])
DT_x = np.array([])
DT_y = np.array([])
DT_z = np.array([])
CSC_x = np.array([])
CSC_y = np.array([])
CSC_z = np.array([])
ECAL_x = np.array([])
ECAL_y = np.array([])
ECAL_z = np.array([])
dt_ecal_dist = np.array([])
tot_size = 0

chunk_size = 100000
gen = uproot.iterate(cosmic_files,"tree",list_of_variables,chunk_size)
for arrays in gen:
    #####n_clusters_3_stations>1 && nCosmicMuonsOneLeg==1 && CosmicMuonsOneLeg.pt>50 && CaloJets.nRecHitsEB>2

    EventNumberAll = np.concatenate(( EventNumberAll, arrays["EventNumber"]))

    tot_size+= arrays["nCaloJets"].shape[0]
    '''veto efficiency
    cut_mask = arrays["n_clusters_3_stations"]>1
    #cut_mask = arrays["n_clusters_2_stations"]>1
    cut_mask = np.logical_and(cut_mask, arrays["nCosmicMuonsOneLeg"]==1)
    #skip cosmic muon!
    cut_mask = np.logical_and(cut_mask, arrays["nCosmicMuons"]>1)
    cut_mask = np.logical_and(cut_mask, arrays["nCaloJets"]<5)

    cut_cosmic = arrays["CosmicMuonsOneLeg.pt"]>50
    cut_cosmic = np.logical_and(cut_mask,cut_cosmic)
    cut_mask = (cut_cosmic.any()==True)
    cut_jet = arrays["CaloJets.nRecHitsEB"]>2
    cut_jet = np.logical_and(cut_mask,cut_jet)
    cut_mask = (cut_jet.any()==True)
    no_cut_mask = np.logical_and(cut_mask,arrays["isCosmic"]<1)
    '''

    cut_mask = arrays["nGenParticles"]>0
    cut_gen = arrays["GenParticles.pt"]>50
    cut_gen = np.logical_and(cut_mask,cut_gen)
    cut_mask = (cut_gen.any()==True)
    den += cut_mask[cut_mask].sum()


    #print "n calo jets ", arrays["nCaloJets"][cut_mask]
    #print "CaloJets.nRecHitsEB", arrays["CaloJets.nRecHitsEB"][cut_mask]
    #print "CaloJets.pt", arrays["CaloJets.pt"][cut_mask]
    #print "dt_ecal_dist", arrays["dt_ecal_dist"][cut_mask]
    #print "n_stations", arrays["n_clusters_3_stations"][cut_mask]
    
    RunNumber = np.concatenate(( RunNumber, arrays["RunNumber"][cut_mask]))
    LumiNumber = np.concatenate(( LumiNumber, arrays["LumiNumber"][cut_mask]))
    EventNumber = np.concatenate(( EventNumber, arrays["EventNumber"][cut_mask]))
    dt_ecal_dist = np.concatenate(( dt_ecal_dist, arrays["dt_ecal_dist"][cut_mask]))
    DT_x = np.concatenate(( DT_x,np.divide(arrays["DTSegments.x"][cut_mask],100.)))
    DT_y = np.concatenate(( DT_y,np.divide(arrays["DTSegments.y"][cut_mask],100.)))
    DT_z = np.concatenate(( DT_z,np.divide(arrays["DTSegments.z"][cut_mask],100.)))
    CSC_x = np.concatenate(( CSC_x,np.divide(arrays["CSCSegments.x"][cut_mask],100.) ))
    CSC_y = np.concatenate(( CSC_y,np.divide(arrays["CSCSegments.y"][cut_mask],100.) ))
    CSC_z = np.concatenate(( CSC_z,np.divide(arrays["CSCSegments.z"][cut_mask],100.) ))
    ECAL_x = np.concatenate(( ECAL_y,np.divide(arrays["skimmedEcalRecHitsAK4.x"][cut_mask],100.) ))
    ECAL_y = np.concatenate(( ECAL_y,np.divide(arrays["skimmedEcalRecHitsAK4.y"][cut_mask],100.) ))
    ECAL_z = np.concatenate(( ECAL_z,np.divide(arrays["skimmedEcalRecHitsAK4.z"][cut_mask],100.) ))
    GenMuon_pt = np.concatenate(( GenMuon_pt , arrays["GenParticles.pt"][cut_mask] ))
    RecoMuon_pt = np.concatenate(( RecoMuon_pt , arrays["CosmicMuonsOneLeg.pt"][cut_mask] ))


    EventNumberPassingDen = np.concatenate(( EventNumberPassingDen, arrays["EventNumber"][cut_mask]))

    #cut_mask = np.logical_and(cut_mask,arrays["isCosmic"]>0)
    cut_reco = arrays["CosmicMuonsOneLeg.pt"]>5
    cut_reco = np.logical_and(cut_mask,cut_reco)
    cut_mask = (cut_reco.any()==True)
    num += cut_mask[cut_mask].sum()

    cut_reco = arrays["CosmicMuonsOneLeg.pt"]>10
    cut_reco = np.logical_and(cut_mask,cut_reco)
    cut_mask = (cut_reco.any()==True)
    num2 += cut_mask[cut_mask].sum()

    cut_reco = arrays["CosmicMuonsOneLeg.pt"]>20
    cut_reco = np.logical_and(cut_mask,cut_reco)
    cut_mask = (cut_reco.any()==True)
    num3 += cut_mask[cut_mask].sum()

    cut_reco = arrays["CosmicMuonsOneLeg.pt"]>50
    cut_reco = np.logical_and(cut_mask,cut_reco)
    cut_mask = (cut_reco.any()==True)
    num4 += cut_mask[cut_mask].sum()




#print RunNumber
print "total n. events:",tot_size
print "n. events passing gen selections:",den
print "n. events passing reco selections:",num
print "ratio ", 100.*num/den
print "n. events passing reco selections > 10 pt:",num2
print "ratio ", 100.*num2/den
print "n. events passing reco selections > 20 pt:",num3
print "ratio ", 100.*num3/den
print "n. events passing reco selections > 50 pt:",num4
print "ratio ", 100.*num4/den
print GenMuon_pt - RecoMuon_pt
#print EventNumberAll.shape[0]#tolist()

exit()

for ev in range(EventNumberAll.shape[0]):
    if EventNumberAll[ev] in difference:
        print "missing event number:",EventNumberAll[ev]
        print arrays["pt"][ev]
        print arrays["nCaloJets"][ev]
        print arrays["CaloJets.nRecHitsEB"][ev]

#print DT_x
#print ECAL_x
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
    leg.SetHeader("dt ecal dist = "+str(dt_ecal_dist[ev])+" (m)")
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
    leg.SetHeader("dt ecal dist = "+str(dt_ecal_dist[ev])+" (m)")
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
    leg.SetHeader("dt ecal dist = "+str(dt_ecal_dist[ev])+" (m)")
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
    leg.SetHeader("dt ecal dist = "+str(dt_ecal_dist[ev])+" (m)")
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


for i in range(int(den)):
    hd.Fill(0)

for i in range(int(den-num)):
    hn.Fill(0)

for i in range(int(num)):
    hn_pass.Fill(0)

e = TGraphAsymmErrors()
e.BayesDivide(hn,hd)
e_unc = e.GetErrorY(0)
print "e_unc: ", e_unc

e_2DT = 1.-0.99180

e_pass = TGraphAsymmErrors()
e_pass.BayesDivide(hn_pass,hd)
e_pass_unc = e_pass.GetErrorY(0)

for i in range(int(obs)):
    ho.Fill(0)
poiss_o = ho.GetBinError(1)/ho.GetBinContent(1)
tot_unc = math.sqrt(e_unc**2 + poiss_o**2)

#2016BF
for i in range(int(obs_2016BF)):
    ho_2016BF.Fill(0)
poiss_o_2016BF = ho_2016BF.GetBinError(1)/ho_2016BF.GetBinContent(1)
y_2016BF = obs_2016BF*(1.-num/den)
y_2016BF_2DT = obs_2016BF*e_2DT
diff_2016BF_2DT = abs(y_2016BF_2DT - y_2016BF)
tot_unc_2016BF = math.sqrt(e_unc**2 + poiss_o_2016BF**2 + (diff_2016BF_2DT/y_2016BF)**2)

#2016GH
for i in range(int(obs_2016GH)):
    ho_2016GH.Fill(0)
poiss_o_2016GH = ho_2016GH.GetBinError(1)/ho_2016GH.GetBinContent(1)
y_2016GH = obs_2016GH*(1.-num/den)
y_2016GH_2DT = obs_2016GH*e_2DT
diff_2016GH_2DT = abs(y_2016GH_2DT - y_2016GH)
tot_unc_2016GH = math.sqrt(e_unc**2 + poiss_o_2016GH**2 + (diff_2016GH_2DT/y_2016GH)**2)

#2017
for i in range(int(obs_2017)):
    ho_2017.Fill(0)
poiss_o_2017 = ho_2017.GetBinError(1)/ho_2017.GetBinContent(1)
y_2017 = obs_2017*(1.-num/den)
y_2017_2DT = obs_2017*e_2DT
diff_2017_2DT = abs(y_2017_2DT - y_2017)
tot_unc_2017 = math.sqrt(e_unc**2 + poiss_o_2017**2 + (diff_2017_2DT/y_2017)**2)

#2018
for i in range(int(obs_2018)):
    ho_2018.Fill(0)
poiss_o_2018 = ho_2018.GetBinError(1)/ho_2018.GetBinContent(1)
print "poiss_o_2018: ", poiss_o_2018
y_2018 = obs_2018*(1.-num/den)
y_2018_2DT = obs_2018*e_2DT
diff_2018_2DT = abs(y_2018_2DT - y_2018)
tot_unc_2018 = math.sqrt(e_unc**2 + poiss_o_2018**2 + (diff_2018_2DT/y_2018)**2)

print "Num: ", num
print "Den: ", den
print "Cosmic veto efficiency: %.5f +- %.5f"%(num/den,e_pass_unc)
print "Cosmic veto miss efficiency: %.5f +- %.5f"%((den-num)/den,e_unc)

print "Cosmic prediction: %.5f +- %.5f"%(obs*(1.-num/den) , tot_unc*obs*(1.-num/den))
print "\n"
print "2016BF"
print "Cosmic events vetoed in SR: ",obs_2016BF
print "Cosmic prediction: %.5f +- %.5f"%(obs_2016BF*(1.-num/den) , tot_unc_2016BF*obs_2016BF*(1.-num/den))
print "2016GH"
print "Cosmic events vetoed in SR: ",obs_2016GH
print "Cosmic prediction: %.5f +- %.5f"%(obs_2016GH*(1.-num/den) , tot_unc_2016GH*obs_2016GH*(1.-num/den))
print "2017"
print "Cosmic events vetoed in SR: ",obs_2017
print "Cosmic prediction: %.5f +- %.5f"%(obs_2017*(1.-num/den) , tot_unc_2017*obs_2017*(1.-num/den))
print "2018"
print "Cosmic events vetoed in SR: ",obs_2018
print "Cosmic prediction: %.5f +- %.5f"%(obs_2018*(1.-num/den) , tot_unc_2018*obs_2018*(1.-num/den))

#prepare yaml
BASEDIR = "plots/Yields_AN_fix/v6_calo_AOD_%s_SR/"
filename = "BkgPredResults_%s_SR_cosmic%s"
#2016BF
results = {}
OUTDIR = BASEDIR%("2016")
name = filename%("2016","_B-F")
results["bkg_cosmic"] = y_2016BF
results["unc_cosmic"] = tot_unc_2016BF*100
print results
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2016GH
results = {}
OUTDIR = BASEDIR%("2016")
name = filename%("2016","_G-H")
results["bkg_cosmic"] = y_2016GH
results["unc_cosmic"] = tot_unc_2016GH*100
print results
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2017
results = {}
OUTDIR = BASEDIR%("2017")
name = filename%("2017","")
results["bkg_cosmic"] = y_2017
results["unc_cosmic"] = tot_unc_2017*100
print results
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

#2018
results = {}
OUTDIR = BASEDIR%("2018")
name = filename%("2018","")
results["bkg_cosmic"] = y_2018
results["unc_cosmic"] = tot_unc_2018*100
print results
with open(OUTDIR+name+".yaml","w") as f:
    yaml.dump(results, f)
    f.close()
print "Written ",OUTDIR+name+".yaml"

