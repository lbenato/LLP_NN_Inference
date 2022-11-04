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
import uproot
import root_numpy

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
(options, args) = parser.parse_args()
##if options.bash: 
gROOT.SetBatch(True)
gStyle.SetOptStat(0)

from NNInferenceCMSSW.LLP_NN_Inference.drawUtils import *
from NNInferenceCMSSW.LLP_NN_Inference.selections import *
from NNInferenceCMSSW.LLP_NN_Inference.variables import *

ERA = "2017"

NTUPLEDIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+ERA+"_SR_chris_hill/"
MC_NTUPLEDIR = NTUPLEDIR+"weighted/"#"/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_"+str(ERA)+"_SR_v5_ntuples/"

print "PLEASE FIX MET WEIGHTS!!!!"


MAIN_OUT = "plots/v6_calo_AOD_"+ERA+"_SR/"
LUMI = -1

back = ["All"]
data = ["HighMET"]
if ERA=="2016":
    data = ["HighMETGH"]

if ERA=="2018":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2018 import lumi
    LUMI  = lumi[ data[0] ]["tot"]
elif ERA=="2017":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2017 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2017 import lumi
    LUMI  = lumi[ data[0] ]["tot"]
elif ERA=="2016":
    from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2016 import sample, samples
    from NNInferenceCMSSW.LLP_NN_Inference.lumi_v5_2016 import lumi
    LUMI  = lumi[ data[0] ]["tot"]



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
#bins=np.array([0.9,2.5,5,7.5,10,12.5,15,20,30,40,50,60,70,80,90,100,120,140,160,200,250,500,1000,1500])
shift_bins = np.arange(0,100+1,1)
less_bins = np.arange(-0.5,99.5+2,2)
WEIGHT = "EventWeight*TriggerWeight*PUReWeight"

def pl(data, back,tree_name="tree",v_x="Jets.radiusLLPCaloCorr",v_y="Jets.sigprob",cut="",cut_d="Jets.isGenMatchedCaloCorr",do_radius=False,add_label="",max_y=1,min_y=0,max_x=50):
    print "cut: ", cut
    print "cut_d: ", cut_d

    bins = np.arange(0,101)#5,10,15,20,25,30,35,40,45,50,100,200])
    nj = []#{}
    nt = []#{}

    nj_data = np.empty((2,0))
    w_data = np.array([])
    nj_back = np.empty((2,0))
    w_back = np.array([])

    h_data = TH2F("d","d",100,0,100,20,-0.5,19.5)
    h_data.Sumw2()

    h_back = TH2F("b","b",100,0,100,20,-0.5,19.5)
    h_back.Sumw2()

    #for i in range(len(bins)):
    #    nj[i] = []
    #    nt[i] = []

    list_of_variables = ["nTagJets_0p996_JJ","isMC","Jets.pt","Jets.phi","Jets.eta","Jets.sigprob","Jets.timeRMSRecHitsEB","Jets.nRecHitsEB","JetsNegative.*","JetsNegative.pt","JetsNegative.phi","JetsNegative.eta","JetsNegative.sigprob","JetsNegative.timeRMSRecHitsEB","JetsNegative.nRecHitsEB","JetsNegative.timeRecHitsEB","HLT*","MT","pt","MinJet*DPhi*","nCHSJetsAcceptanceCalo","nCHSJets_in_HEM","EventNumber","RunNumber","LumiNumber","EventWeight","TriggerWeight","PUWeight","PUReWeight","GenLLPs.travelRadius","GenLLPs.travelX","GenLLPs.travelY","GenLLPs.travelZ","GenLLPs.travelTime","GenLLPs.beta","GenLLPs.*","isSR","MeanNumInteractions","HT","Jets.nTrackConstituents"]

    list_of_variables_data = list_of_variables  + ["dt_ecal_dist","min_dPhi_jets*"]

    tree_weight_dict = {}
    #first round, get tree weight from mc
    for b in back+data:
        for i, ss in enumerate(samples[b]['files']):
            if ('Run201') in ss:
                t_w = 1.
            else:
                if not os.path.isfile(MC_NTUPLEDIR+ss+'.root'):
                    print("!!!File ", MC_NTUPLEDIR+ss+'.root', " does not exist! Continuing")
                    continue
                filename = TFile(MC_NTUPLEDIR+ss+'.root', "READ")
                if filename.GetListOfKeys().Contains("c_nEvents"):
                    nevents = filename.Get("c_nEvents").GetBinContent(1)
                    b_skipTrain = filename.Get("b_skipTrain").GetBinContent(1)
                    n_pass      = filename.Get("n_pass").GetBinContent(1)
                    n_odd       = filename.Get("n_odd").GetBinContent(1)
                    filename.Close()
                    xs = sample[ss]['xsec'] * sample[ss]['kfactor']
                    t_w = LUMI * xs / nevents
                    if(b_skipTrain>0):
                        if(n_odd>0):
                            t_w *= float(n_pass/n_odd)
                else:
                    t_w = 0

            tree_weight_dict[ss] = t_w

    chunk_size = 100000

    #uproot: count how many jets in that specific track multiplicity range
    DIR = ""
    for d in data+back:
        for i, ss in enumerate(samples[d]['files']):
            if d in data:
                DIR=NTUPLEDIR
            if d in back:
                DIR=MC_NTUPLEDIR
            if not os.path.isfile(DIR + ss + '.root'):
                print("!!!File ", DIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            chosen_list = list_of_variables
            if d in data:
                chosen_list = list_of_variables_data
            gen = uproot.iterate(DIR + ss + ".root","tree",chosen_list,entrysteps=chunk_size)
            n_it = 0
            for arrays in gen:
                print "Ev n. ", chunk_size*(1+n_it)
                cut_mask =np.logical_and( np.logical_and(arrays["HT"]>250, arrays["pt"]>250) , arrays["nTagJets_0p996_JJ"]<2 )
                cut_mask = np.logical_and(cut_mask,arrays["nCHSJetsAcceptanceCalo"]>0)
                #pre-weight
                eventweight = arrays["EventWeight"]
                pureweight = arrays["PUReWeight"]
                triggerweight = arrays["TriggerWeight"]
                weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[ss]
                #Remove very large weights
                cut_mask = np.logical_and(cut_mask,weight<1000)#Try 1000 instead of 10k
                j_tracks = arrays["Jets.nTrackConstituents"][cut_mask]
                nj = arrays["nCHSJetsAcceptanceCalo"][cut_mask]
                eventweight = arrays["EventWeight"][cut_mask]
                pureweight = arrays["PUReWeight"][cut_mask]
                triggerweight = arrays["TriggerWeight"][cut_mask]
                weight = np.multiply(eventweight,np.multiply(pureweight,triggerweight))*tree_weight_dict[ss]
                del arrays
                print "passed events: ", j_tracks.shape
                print "passed jets: ", nj.sum()
                for i in range(len(bins)):
                    if i<len(bins)-1:
                        tmp_count = j_tracks[np.logical_and(j_tracks>=bins[i],j_tracks<bins[i+1])].counts
                        #per_jet_weight = tmp_count.astype(bool)*weight
                        #tmp_count_mask = (tmp_count>0)
                        #tmp_count = tmp_count[tmp_count_mask]
                        #tmp_weight = per_jet_weight[tmp_count_mask]
                        #if tmp_count.any()>0:
                        #    tmp_nj = np.array([np.ones(tmp_count.shape[0])*bins[i], tmp_count])
                        #    if d in data:
                        #        nj_data = np.concatenate((nj_data, tmp_nj), axis=1)
                        #        w_data = np.concatenate((w_data,  tmp_weight))
                    else:
                        tmp_count = j_tracks[(j_tracks>=bins[i])].counts

                    per_jet_weight = tmp_count.astype(bool)*weight
                    tmp_count_mask = (tmp_count>0)
                    tmp_count = tmp_count[tmp_count_mask]
                    tmp_weight = per_jet_weight[tmp_count_mask]

                    if tmp_count.any()>0:
                        if bins[i]>3:
                            #blind n tracks up to 3
                            tmp_nj = np.array([np.ones(tmp_count.shape[0])*bins[i], tmp_count])
                            if d in data:
                                nj_data = np.concatenate((nj_data, tmp_nj), axis=1)
                                w_data = np.concatenate((w_data,  tmp_weight))
                            if d in back:
                                nj_back = np.concatenate((nj_back, tmp_nj), axis=1)
                                w_back = np.concatenate((w_back,  tmp_weight))

                n_it+=1
                #if n_it>0:
                #    break

    root_numpy.fill_hist(h_data,np.swapaxes(nj_data,0,1), w_data)
    h_data_2 = h_data.RebinX(2,"h_data_2")
    profX_data = TProfile(h_data_2.ProfileX("profX_data"))
    profX_data.GetXaxis().SetTitle(variable[v_x]["title"])
    profX_data.GetYaxis().SetTitle(variable[v_y]["title"])
    profX_data.SetLineColor(1)
    profX_data.SetMarkerColor(1)
    profX_data.SetMarkerStyle(20)
    profX_data.SetMarkerSize(1.2)
    profX_data.SetLineWidth(3)


    root_numpy.fill_hist(h_back,np.swapaxes(nj_back,0,1), w_back)
    h_back_2 = h_back.RebinX(2,"h_back_2")
    profX_back = TProfile(h_back_2.ProfileX("profX_back"))
    profX_back.GetXaxis().SetTitle(variable[v_x]["title"])
    profX_back.GetYaxis().SetTitle(variable[v_y]["title"])
    profX_back.SetLineColor(8)
    profX_back.SetFillColorAlpha(8,0.5)
    profX_back.SetMarkerColor(8)
    profX_back.SetMarkerStyle(21)
    profX_back.SetMarkerSize(1.2)
    profX_back.SetLineWidth(0)

    #print nt

    can = TCanvas("can","can",1000,800)
    can.cd()
    can.SetGrid()
    #can.SetLogy()
    can.SetLogz()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    #h_data.Draw("COLZ")

    profX_back.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_back.SetMinimum(min_y)#(variable[v_y]['max'])
    profX_back.Draw("PE")
    profX_back.GetXaxis().SetRangeUser(0,max_x)
    profX_back.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_back.SetMinimum(min_y)#(variable[v_y]['max'])

    back_unc = profX_back.Clone("")
    back_unc.SetFillColorAlpha(8,0.5)
    #back_unc.SetFillStyle(3001)
    back_unc.Draw("E2,sames")
    profX_data.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_data.SetMinimum(min_y)#(variable[v_y]['max'])
    profX_data.Draw("PE,sames")
    profX_data.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_data.SetMinimum(min_y)#(variable[v_y]['max'])

    leg = TLegend(0.4+0.3, 0.7, 0.9, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(profX_back,"simulation","F")
    #leg.AddEntry(back_unc,"stat. unc.","F")
    leg.AddEntry(profX_data,"data","PL")
    leg.SetBorderSize(0)
    leg.Draw()

    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.png')
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.pdf')
    can.Close()

    outfile = TFile(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+".root","RECREATE")
    outfile.cd()
    h_data.Write("h_data")
    h_back.Write("h_back")
    profX_data.Write("profX_data")
    profX_back.Write("profX_back")
    outfile.Close()
    print "Written ",MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+".root"


def draw(v_x="",v_y="",add_label="",max_y=1,min_y=0,max_x=50):

    infile = TFile(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+".root","READ")
    infile.cd()
    h_data = infile.Get("h_data")
    h_back = infile.Get("h_back")
    h_data.SetDirectory(0)
    h_back.SetDirectory(0)
    infile.Close()

    h_data_2 = h_data.RebinX(1,"h_data_2")
    profX_data = TProfile(h_data_2.ProfileX("profX_data"))
    #profX_data = profX_data.Rebin(len(less_bins)-1,"profX_data_2",less_bins)
    profX_data.GetXaxis().SetTitle(variable[v_x]["title"])
    profX_data.GetYaxis().SetTitle("number of jets")
    profX_data.SetLineColor(1)
    profX_data.SetMarkerColor(1)
    profX_data.SetMarkerStyle(20)
    profX_data.SetMarkerSize(1.2)
    profX_data.SetLineWidth(3)

    h_back_2 = h_back.RebinX(1,"h_back_2")
    profX_back = TProfile(h_back_2.ProfileX("profX_back"))
    #profX_back = profX_back.Rebin(len(less_bins)-1,"profX_back_2",less_bins)
    profX_back.GetXaxis().SetTitle(variable[v_x]["title"])
    profX_back.GetYaxis().SetTitle("number of jets")
    profX_back.SetLineColor(8)
    profX_back.SetFillColorAlpha(8,0.5)
    profX_back.SetMarkerColor(8)
    profX_back.SetMarkerStyle(21)
    profX_back.SetMarkerSize(1.2)
    profX_back.SetLineWidth(0)

    #print nt

    can = TCanvas("can","can",1000,800)
    can.cd()
    can.SetGrid()
    #can.SetLogy()
    can.SetLogz()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    #h_data.Draw("COLZ")

    profX_back.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_back.SetMinimum(min_y)#(variable[v_y]['max'])
    profX_back.Draw("P")
    profX_back.GetXaxis().SetRangeUser(0,max_x)
    profX_back.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_back.SetMinimum(min_y)#(variable[v_y]['max'])

    back_unc = profX_back.Clone("")
    back_unc.SetFillColorAlpha(8,0.5)
    #back_unc.SetFillStyle(3001)
    back_unc.Draw("E2,sames")
    profX_data.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_data.SetMinimum(min_y)#(variable[v_y]['max'])
    profX_data.Draw("PE,sames")
    profX_data.SetMaximum(max_y)#(variable[v_y]['max'])
    profX_data.SetMinimum(min_y)#(variable[v_y]['max'])

    leg = TLegend(0.4+0.3, 0.7, 0.9, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(profX_back,"simulation","PF")
    #leg.AddEntry(back_unc,"stat. unc.","F")
    leg.AddEntry(profX_data,"data","PL")
    leg.SetBorderSize(0)
    leg.Draw()

    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.png')
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.pdf')
    can.Close()



def old_stuff():
    for b in data:
        tree[b] = TChain(tree_name)
        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(NTUPLEDIR + ss + '.root'):
                print("!!!File ", NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            tree[b].Add(NTUPLEDIR + ss + '.root')

        h[b] = TH2F(str(b),str(b),variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"],variable[v_y]["nbins"],variable[v_y]["min"],variable[v_y]["max"])
        h[b].Sumw2()

        tree[b].Project(str(b),v_y+":"+v_x,cut_d+"*"+WEIGHT)
        print "hist content: ", h[b].GetEntries()
        profX[b] = TProfile(h[b].ProfileX("prof"+str(b)))
        h[b].GetXaxis().SetTitle(variable[v_x]["title"])
        h[b].GetYaxis().SetTitle(variable[v_y]["title"])

        profX[b].SetLineColor(1)
        profX[b].SetMarkerColor(1)
        profX[b].SetMarkerStyle(20)
        profX[b].SetMarkerSize(1.2)
        profX[b].SetLineWidth(3)

        #h[b].SetMarkerColor(861)
        #h[b].SetMarkerSize(.3)
        #h[b].SetMarkerStyle(24)

    for b in back:
        tree[b] = TChain(tree_name)
        for i, ss in enumerate(samples[b]['files']):
            if not os.path.isfile(MC_NTUPLEDIR + ss + '.root'):
                print("!!!File ", MC_NTUPLEDIR + ss + '.root', " does not exist! Continuing")
                continue
            print "adding ", ss
            tree[b].Add(MC_NTUPLEDIR + ss + '.root')

        h[b] = TH2F(str(b),str(b),variable[v_x]["nbins"],variable[v_x]["min"],variable[v_x]["max"],variable[v_y]["nbins"],variable[v_y]["min"],variable[v_y]["max"])
        h[b].Sumw2()
        tree[b].Project(str(b),v_y+":"+v_x,cut+"*"+WEIGHT)
        print "hist content: ", h[b].GetEntries()        

        profX[b] = TProfile(h[b].ProfileX("prof"+str(b)))
        profX[b].GetXaxis().SetTitle(variable[v_x]["title"])
        profX[b].GetYaxis().SetTitle(variable[v_y]["title"])
        profX[b].SetLineColor(8)
        profX[b].SetFillColorAlpha(8,0.5)
        profX[b].SetMarkerColor(8)
        profX[b].SetMarkerStyle(0)
        profX[b].SetMarkerSize(1.2)
        profX[b].SetLineWidth(3)


    can = TCanvas("can","can",1000,800)
    can.cd()
    can.SetGrid()
    #can.SetLogy()
    can.SetLogz()
    can.SetRightMargin(0.05)
    can.SetLeftMargin(0.12)
    can.SetBottomMargin(0.1)

    #h[b].Draw("COLZ")#()#
    profX[back[0]].SetMaximum(max_y)#(variable[v_y]['max'])
    profX[back[0]].SetMinimum(min_y)#(variable[v_y]['max'])
    profX[back[0]].Draw("HISTO")
    profX[back[0]].SetMaximum(max_y)#(variable[v_y]['max'])
    profX[back[0]].SetMinimum(min_y)#(variable[v_y]['max'])
    back_unc = profX[back[0]].Clone("")
    back_unc.SetFillColor(8)
    back_unc.SetFillStyle(3001)
    back_unc.Draw("E2,sames")
    profX[data[0]].Draw("PE,sames")


    leg = TLegend(0.4+0.3, 0.7, 0.65+0.3, 0.85)
    leg.SetTextSize(0.035)
    leg.AddEntry(profX[back[0]],"simulation","F")
    leg.AddEntry(back_unc,"stat. unc.","F")
    leg.AddEntry(profX[data[0]],"data","PL")
    leg.SetBorderSize(0)
    leg.Draw()

    drawCMS_simple(LUMI, "Preliminary", ERA=ERA, onTop=True)
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.png')
    can.Print(MAIN_OUT+v_y.replace('.', '_').replace('/','_div_')+'_vs_'+v_x.replace('.', '_').replace('/','_div_')+add_label+'.pdf')
    can.Close()

#
#pl(data,back,v_x="Jets.nTrackConstituents",v_y="nCHSJetsAcceptanceCalo",cut="(isSR && MEt.pt>250 && HT>250 && nTagJets_0p996_JJ<2)",cut_d="(isSR && MEt.pt>250 && HT>250 && nTagJets_0p996_JJ<2)",max_y=1.1,min_y=0.9,max_x=50,add_label="")# && isMC?1:dt_ecal_dist>0.5)",max_y=4)

draw(v_x="Jets.nTrackConstituents",v_y="nCHSJetsAcceptanceCalo",add_label="",max_y=1.06,min_y=0.98,max_x=50)# && isMC?1:dt_ecal_dist>0.5)",max_y=4)
print less_bins

##pl(data,back,v_y="Jets.nTrackConstituents",v_x="nCHSJets",cut="isSR && MEt.pt>250 && HT>250",max_y=20)
##pl(data,back,v_x="Jets.nTrackConstituents",v_y="nCHSJets",cut="isSR && MEt.pt>250 && HT>250",max_y=10)
#####pl(data,back,v_y="Jets.nTrackConstituents",v_x="nCHSJetsAcceptanceCalo",cut="(isSR && MEt.pt>250 && HT>250 && nTagJets_0p996_JJ<2)",cut_d="(isSR && MEt.pt>250 && HT>250 && nTagJets_0p996_JJ<2 && (isMC?1:dt_ecal_dist>0.5))",max_y=20)
