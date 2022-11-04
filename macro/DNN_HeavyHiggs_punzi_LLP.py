#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import numpy as np
from array import array
from ROOT import ROOT, gROOT, gStyle, gRandom, TSystemDirectory
from ROOT import TFile, TChain, TTree, TCut, TH1, TH1F, TH2F, THStack, TGraph, TMultiGraph, TGraphAsymmErrors, TSpline, TSpline3
from ROOT import TStyle, TCanvas, TPad
from ROOT import TLegend, TLatex, TText, TLine, TBox, TAxis


#from NNInferenceCMSSW.LLP_NN_Inference.samples_v3 import sample, samples
from NNInferenceCMSSW.LLP_NN_Inference.selections import selection
from NNInferenceCMSSW.LLP_NN_Inference.variables import *
#from NNInferenceCMSSW.LLP_NN_Inference.skimmed_variables import *

### comment:
### updated Punzi macro according to STATCOM recommendation: https://twiki.cern.ch/twiki/bin/view/CMS/PunziFom


########## SETTINGS ##########

import optparse
usage = "usage: %prog [options]"
parser = optparse.OptionParser(usage)
parser.add_option("-v", "--variable", action="store", type="string", dest="variable", default="")
parser.add_option("-c", "--cut", action="store", type="string", dest="cut", default="")
parser.add_option("-r", "--region", action="store", type="string", dest="region", default="")
parser.add_option("-a", "--all", action="store_true", default=False, dest="all")
parser.add_option("-b", "--bash", action="store_true", default=False, dest="bash")
parser.add_option("-B", "--blind", action="store_true", default=False, dest="blind")
parser.add_option("-f", "--final", action="store_true", default=False, dest="final")
(options, args) = parser.parse_args()
if options.bash:
    gROOT.SetBatch(True)

########## SETTINGS ##########

gStyle.SetOptStat(0)

#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_all_events/"
#OUTPUTDIR   = "$CMSSW_BASE/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v4_calo_AOD_2018_tf_and_skim_unmerged_all_events/"
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_HT_cut/"
#OUTPUTDIR   = "$CMSSW_BASE/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v4_calo_AOD_2018_tf_and_skim_unmerged_no_HT_cut/"
#NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/"
#OUTPUTDIR   = "$CMSSW_BASE/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v4_calo_AOD_2018_tf_and_skim_unmerged_no_cuts_compare_JJ/"

#AK4 v3 AK8 v2
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"
OUTPUTDIR   = "$CMSSW_BASE/src/NNInferenceCMSSW/LLP_NN_Inference/plots/Efficiency/v4_calo_AOD_2018_tf_and_skim_AK4_v3__AK8_v2/"

from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples



LUMI        = 35867 # in pb-1
SIGNAL      = 1.
RATIO       = 4 # 0: No ratio plot; !=0: ratio between the top and bottom pads
BLIND       = False
POISSON     = False
verbose     = False
verbose_add = False
jobs        = []

#CHANNEL     = "VBFH"
#CHANNEL     = "SUSY"
CHANNEL     = "ggH"

########## SAMPLES ##########
sign = [
        #'ggH_MH400_MS50_ctau500',  'ggH_MH400_MS50_ctau1000',  'ggH_MH400_MS50_ctau2000',  'ggH_MH400_MS50_ctau5000',  'ggH_MH400_MS50_ctau10000',
        #'ggH_MH400_MS100_ctau500', 'ggH_MH400_MS100_ctau1000', 'ggH_MH400_MS100_ctau2000', 'ggH_MH400_MS100_ctau5000', 'ggH_MH400_MS100_ctau10000',
        'ggH_MH600_MS50_ctau500',  'ggH_MH600_MS50_ctau1000',  'ggH_MH600_MS50_ctau2000',  'ggH_MH600_MS50_ctau5000',  'ggH_MH600_MS50_ctau10000',
        'ggH_MH600_MS150_ctau500', 'ggH_MH600_MS150_ctau1000', 'ggH_MH600_MS150_ctau2000', 'ggH_MH600_MS150_ctau5000', 'ggH_MH600_MS150_ctau10000',
	'ggH_MH1000_MS150_ctau500','ggH_MH1000_MS150_ctau1000','ggH_MH1000_MS150_ctau2000','ggH_MH1000_MS150_ctau5000','ggH_MH1000_MS150_ctau10000',
	'ggH_MH1000_MS400_ctau500','ggH_MH1000_MS400_ctau1000','ggH_MH1000_MS400_ctau2000','ggH_MH1000_MS400_ctau5000','ggH_MH1000_MS400_ctau10000',
	'ggH_MH1500_MS200_ctau500','ggH_MH1500_MS200_ctau1000','ggH_MH1500_MS200_ctau2000','ggH_MH1500_MS200_ctau5000','ggH_MH1500_MS200_ctau10000',
	'ggH_MH1500_MS500_ctau500','ggH_MH1500_MS500_ctau1000','ggH_MH1500_MS500_ctau2000','ggH_MH1500_MS500_ctau5000','ggH_MH1500_MS500_ctau10000',
	'ggH_MH2000_MS250_ctau500','ggH_MH2000_MS250_ctau1000','ggH_MH2000_MS250_ctau2000','ggH_MH2000_MS250_ctau5000','ggH_MH2000_MS250_ctau10000',
	'ggH_MH2000_MS600_ctau500','ggH_MH2000_MS600_ctau1000','ggH_MH2000_MS600_ctau2000','ggH_MH2000_MS600_ctau5000','ggH_MH2000_MS600_ctau10000',
	]
back = ["VV","WJetsToLNu","ZJetsToNuNu","TTbar","QCD"]
back = []
colors = [4, 410, 856, 2, 634, 1, 881, 798, 602, 921, 870, 906, 838, 420, 398]
markers = [24,26,25,32,28,30,27,24,26,25,32,28,30,27,24,26]
########## ######## ##########

#gROOT.SetBatch(True)


def calc_punzi_FOM_vs_ctau(cutlist, labellist=[],linelist=[],mH=1000,mS=400,additional_string="",alpha=2,CL=5,FOM='punzi',header=""):
    file = {}
    nevt = {}
    tree = {}
    effs = {}
    chain = {}
    hist = {}
    eff_dict = { k:{} for k in cutlist}
    back_int = { k:{} for k in cutlist}
    back_int_weight = { k:{} for k in cutlist}
    back_eff = { k:{} for k in cutlist}
    punzi_dict = { k:{} for k in cutlist}
    graph = {}
    back_graph = {}
    ncuts = len(cutlist)
    if labellist == []:
        labellist=cutlist
    print NTUPLEDIR
    print "............."
    #prepare ctau ordered array for 1D plot                                                                
    mass_array = []
    ctau_array = []


    #for signal we have the normal efficiency                                                               
    for i, s in enumerate(sign):
        print "Doing: ", s
        print "\n"
        file[s] = TFile(NTUPLEDIR + samples[s]['files'][0] + ".root", "READ") # Read TFile                  
        tree[s] = file[s].Get("tree")#("ntuple/tree") # Read TTree       
        #nevt[s] = (file[s].Get('counter/c_nEvents')).GetBinContent(1)# all gen events before cuts!
        nevt[s] = (file[s].Get('c_nEvents')).GetBinContent(1)# all gen events before cuts!
        ##nevt[s] = tree[s].GetEntries("")#if no selections applied, this is the right denominator
        filename = TFile(NTUPLEDIR + samples[s]['files'][0] + ".root", "READ")
        if verbose_add: print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        if verbose_add: print filename
        #if verbose_add: print "x-check: n gen events in counter, first bin:"
        #if verbose_add: print (filename.Get('counter/c_nEvents')).GetBinContent(1)
        #if verbose_add: print "x-check: n entries in counter:"
        #if verbose_add: print (filename.Get('counter/c_nEvents')).GetEntries()
        effs[s] = [0]*(ncuts+1)
        effs[s] = [0]*(ncuts+1)
        weight = "1"#"EventWeight"
        var = "isMC"

        if samples[s]['MS'] not in mass_array:
            mass_array.append(samples[s]['MS'])
        if samples[s]['ctau'] not in ctau_array:
            ctau_array.append(samples[s]['ctau'])
        for j, c in enumerate(cutlist):
            tot_gen = nevt[s]
            n = tree[s].GetEntries("(" + cutlist[j] + ")")

            ###BUGFIX: efficiency should be computed w.r.t. histo integral
            #hist[s+"_cut"+str(j)] = TH1F(s+"_cut"+str(j), ";"+variable[var]['title'], variable[var]['nbins'], variable[var]['min'], variable[var]['max'])
            #hist[s+"_cut"+str(j)].Sumw2()
            #cutstring = "("+weight+")" + ("*("+cutlist[j]+")" if len(cutlist[j])>0 else "")
            #tree[s].Project(s+"_cut"+str(j), var, cutstring)
            #hist[s+"_cut"+str(j)].SetOption("%s" % tree[s].GetTree().GetEntriesFast())


            if verbose_add: print '\n'
            if verbose_add: print '**********************************************'
            if verbose_add: print "cut: ", c
            if verbose_add: print 'over signal ', s
            if verbose_add: print '\n'
            if verbose_add: print "signal num: ", n
            if verbose_add: print "signal den: ", tot_gen
            #if verbose_add: print "BUGFIX!!!!!!!!!!!"
            #if verbose: print "BUGFIX!!!!!!!!!!!"
            #if verbose_add: print "signal num from integral: ", hist[s+"_cut"+str(j)].Integral()
            #if verbose_add: print "signal den from generator: ", tot_gen
            #if verbose: print "BUGFIX!!!!!!!!!!!"
            if verbose_add: print ("signal eff %.2f") % (float(n)/(tot_gen)*100)
            if tot_gen==0:
                effs[s][j] = float(0.)
            else:
                effs[s][j] = (float(n)/(tot_gen))
            eff_dict[c][s] = {'MH' : samples[s]['MH'], 'MS' : samples[s]['MS'], 'ctau' : samples[s]['ctau'], 'eff' :effs[s][j], 'nevents' : n}


    #sort mass array
    masses = np.array(mass_array)
    masses.sort()

    ctaus = np.array(ctau_array)
    ctaus.sort()


    #define multigraph
    mg = TMultiGraph()
    #leg2 = TLegend(0.3, 0.11, 0.65, 0.45)#DCMS,gen matching
    leg2 = TLegend(0.4-0.3, 0.11+0.43, 0.85+0.05-0.3, 0.45+0.43)#EXO,summary plot
    leg2 = TLegend(0.4, 0.11, 0.85+0.05, 0.45)#EXO,summary plot
    leg2 = TLegend(0.15, 0.11, 0.45, 0.45)#DCMS,summary plot

    leg3 = TLegend(0., 0.5, 0.5, 1.)#2 plots

    leg = TLegend(0., 0.4, 0.98, 0.98)
    leg.SetTextSize(0.03)
    leg2.SetTextSize(0.03)
    leg2.SetTextSize(0.025)
    leg.SetBorderSize(0)
    leg2.SetBorderSize(0)
    leg.SetHeader("Signal: m_{H}="+str(mH)+" m_{S}=" +str(mS)+" GeV")
    leg2.SetHeader("Signal: m_{H}="+str(mH)+" m_{S}=" +str(mS)+" GeV")

    leg3.SetTextSize(0.03)
    leg3.SetTextSize(0.025)
    leg3.SetBorderSize(0)
    leg3.SetHeader("Signal: m_{H}="+str(mH)+" m_{S}=" +str(mS)+" GeV")


    #for background let's first consider the cut
    for j, c in enumerate(cutlist):
        print '\n'
        print "cut: ", c
        print 'over background'
        print '\n'
        #then loop over background
        integral = 0
        weighted_integral = 0
        back_tot_events = 0
        for i, s in enumerate(back):
            #chain[s] = TChain("ntuple/tree")
            chain[s] = TChain("tree")
            #print "back: ", s
            back_file = {}
            for p, ss in enumerate(samples[s]['files']):
                back_file[ss] = TFile(NTUPLEDIR + ss + ".root", "READ") # Read TFile                  
                #?#if verbose: print "file: ", ss
                #?#if verbose: print "gen events: ", (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                #?#if verbose: print "tree events: ", (back_file[ss].Get('ntuple/tree')).GetEntries()
                #back_tot_events += (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                back_tot_events += (back_file[ss].Get('c_nEvents')).GetBinContent(1)
                ##back_tot_events += (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                chain[s].Add(NTUPLEDIR + ss + ".root")
            #print "MODIFIED WEIGHT!!!!!!"
            #weight = ("EventWeight*%s/5000." % str(back_tot_events))
            weight = "EventWeight"
            #var = "nCHSJets"
            var = "isMC"
            hist[s] = TH1F(s, ";"+variable[var]['title'], variable[var]['nbins'], variable[var]['min'], variable[var]['max'])
            hist[s].Sumw2()
            cutstring = "("+weight+")" + ("*("+cutlist[j]+")" if len(cutlist[j])>0 else "")
            chain[s].Project(s, var, "")#"1*"+"("+weight+")")
            hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
            #if verbose: print "Hist content, no cut:"
            #if verbose: print hist[s].Print()
            #?#if verbose: print "events in the histo with get entries with empty project: ", hist[s].GetEntries()
            #?#if verbose: print "area under histo with empty project: ", hist[s].Integral()
            chain[s].Project(s, var, cutstring)#"1*"+"("+weight+")")
            hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
            hist[s].Scale(samples[s]['weight'] if hist[s].Integral() >= 0 else 0)
            #?#if verbose: print "events in the histo with get entries after project: ", hist[s].GetEntries()
            #?#if verbose: print "area under histo after project: ", hist[s].Integral()
            if verbose: print "Hist content, with cut:"
            if verbose: print hist[s].Print()
            integral += hist[s].GetEntries()
            weighted_integral += hist[s].Integral()
        back_int[c] = integral
        back_int_weight[c] = weighted_integral
        if back_tot_events==0:
            back_eff[c] = float(0.)
        else:
            back_eff[c] = float(integral)/float(back_tot_events)
        if verbose: print "cut: ", c
        if verbose: print "back tot events (unweighted):", back_tot_events
        if verbose: print "back integral (unweighted): ", back_int[c]
        if verbose: print "back integral (weighted): ", back_int_weight[c]
        if verbose: print "back eff (unweighted): ", back_eff[c]*100
        if FOM=="signaleff":
            punzi_dict[c]['back'] = {'back' : back_eff[c]*100}
        for i, s in enumerate(sign):
            if verbose: print "signal efficiency: ", eff_dict[c][s]['eff']*100
            if FOM=="punzi":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['eff']/(CL**2/2. + alpha*math.sqrt(back_int_weight[c]) + (CL/2.)*math.sqrt(CL**2 + 4*alpha*math.sqrt(back_int_weight[c]) + 4*back_int_weight[c]))}
            elif FOM=="signaleff":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['eff']*100}
            elif FOM=="entries":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['nevents']}
            else:
                print "not punzi FOM, aborting!"
                exit()

    if FOM=="signaleff":
        dummy = TGraph()#len(ct),ct, np.array(ct))
        dummy.SetMarkerStyle(0)
        dummy.SetLineWidth(2)
        dummy.SetMarkerSize(1.)
        dummy.SetLineColor(15)
        dummy.SetLineStyle(2)
        axis = TAxis(dummy.GetXaxis()) 
        axis.SetLimits(100,10100)
        if header!="":
            leg2.AddEntry(dummy, header,'')
            leg3.AddEntry(dummy, header,'')


    #for each cut, we need a graph                                                                          
    for j, c in enumerate(cutlist):
    #first let's build the ordered punzi vector w.r.t. masses, for a chosen ctau                            
        punzi_array = []
        back_array = []
        for la in ctaus:
            #la = str(a)
            if la== 0.001:
                st = CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_ctau0"
            elif la==0.05 or la==0.1:
                st = CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_ctau"+str(str(la).replace("0.","0p"))
            else:
                st = CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_ctau"+str(int(la))                                                   
            punzi_array.append(punzi_dict[c][st]['sign'])
        mass = array('d', masses)
        ct = array('d', ctaus)
        p_array = array('d',punzi_array)
        #graph[c] = TGraph(len(mass),mass, np.array(p_array))                                                   
        graph[c] = TGraph(len(ct),ct, np.array(p_array))
        graph[c].SetMarkerStyle(markers[j])#21
        graph[c].SetLineWidth(3)
        graph[c].SetMarkerSize(1.2)
        graph[c].SetMarkerColor(colors[j])
        if(len(linelist)!=0):
            graph[c].SetLineStyle(linelist[j])
        graph[c].SetLineColor(colors[j])
        graph[c].SetFillColor(colors[j])

        axis = TAxis(graph[c].GetXaxis()) 
        axis.SetLimits(100,10100)
        #graph[c].GetXaxis().SetRangeUser(100,10100)#ignored?
        #graph[c].Draw("APL")#ignored?
        #graph[c].GetXaxis().SetRangeUser(100,10100)#ignored?
        #graph[c].Draw("APL")#ignored?
        ##graph[c].SetLogx()                                                                                 

        leg.AddEntry(graph[c],labellist[j],'PL')
        leg2.AddEntry(graph[c],labellist[j],'PL')
        leg3.AddEntry(graph[c],labellist[j],'PL')
        mg.Add(graph[c])

        if FOM=="signaleff":
        #add plot for background                                                                            
            for a in ctaus:
                back_array.append(punzi_dict[c]['back']['back'])
            mass = array('d', masses)
            ct = array('d', ctaus)
            e_array = array('d',back_array)
            #back_graph[c] = TGraph(len(mass),mass, np.array(e_array))
            back_graph[c] = TGraph(len(ct),ct, np.array(e_array))
            back_graph[c].SetMarkerStyle(0)
            back_graph[c].SetLineWidth(2)
            back_graph[c].SetMarkerSize(1.)
            back_graph[c].SetMarkerColor(colors[j])
            back_graph[c].SetLineColor(colors[j])
            back_graph[c].SetLineStyle(2)
            back_graph[c].SetFillColor(colors[j])
            axis = TAxis(back_graph[c].GetXaxis()) 
            axis.SetLimits(100,10100)


    if FOM=="signaleff":
        dummy = TGraph(len(ct),ct, np.array(e_array))
        dummy.SetMarkerStyle(0)
        dummy.SetLineWidth(2)
        dummy.SetMarkerSize(1.)
        dummy.SetLineColor(15)
        dummy.SetLineStyle(2)
        axis = TAxis(dummy.GetXaxis()) 
        axis.SetLimits(100,10100)


    #cmg = TCanvas("cmg", "cmg", 2000, 800)#best
    cmg = TCanvas("cmg", "cmg", 1300, 800)#DCMS
    cmg.cd()
    cmg.SetGrid()
    cmg.SetLogx()

    if FOM=="signaleff":
        print "LOL"

    mg.SetMinimum(0.)#!!
    mg.Draw("APL")
    #axis = TAxis(mg.GetXaxis()) 
    #axis.SetLimits(100,10100)
    #mg.Draw("APL")

    ###mg.GetXaxis().SetRangeUser(100,10100)#ignored?

    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.GetXaxis().SetTitle('c#tau_{S} (mm)')
    mg.GetYaxis().SetTitleOffset(0.9);
    if FOM=="punzi":
        mg.GetYaxis().SetTitle('Punzi significance @ '+str(alpha)+' #sigma, '+CHANNEL+' cuts')
        #mg.GetYaxis().SetTitleOffset(1.5)
    elif FOM=="signaleff":
        mg.GetYaxis().SetTitle('Signal efficiency, '+CHANNEL+' cuts (%)')
        #mg.GetYaxis().SetTitle('Signal gen-matching efficiency, '+CHANNEL+' (%)')
    elif FOM=="entries":
        mg.GetYaxis().SetTitle('Signal entries surviving cuts')
    else:
        print "not punzi FOM, aborting"

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(33)
    latex.SetTextFont(62)
    latex.DrawLatex(0.25, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.66, 0.96, "Simulation Preliminary")
    cmg.Update()

    cmg.cd()
    leg2.SetTextSize(0.04)
    leg2.Draw()#in first canv? TBC
    cmgL = TCanvas("cmgL", "cmgL", 2000, 800)#DCMS
    cmgL.cd()

    leg3.SetTextSize(0.04)
    #leg3.Draw()
    cmgL.Update()


    if FOM=="punzi":
        cmg.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_"+str(alpha)+"sigma"+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_"+str(alpha)+"sigma"+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_"+str(alpha)+"sigma"+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+"_"+str(alpha)+"sigma"+additional_string+"_L.png")
    elif FOM=="signaleff":
        cmg.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+"_L.png")
    elif FOM=="entries":
        cmg.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_MH"+str(mH)+"_MS"+str(mS)+additional_string+"_L.png")
    else:
        print "not punzi FOM, aborting"

    if not options.bash: raw_input("Press Enter to continue...")
    cmg.Close()



#HERE!
def calc_punzi_FOM_vs_mass(cutlist, labellist=[],ctau_point=1000,additional_string="",alpha=2,CL=5,FOM='punzi',header=""):
    file = {}
    nevt = {}
    tree = {}
    effs = {}
    chain = {}
    hist = {}
    eff_dict = { k:{} for k in cutlist}
    back_int = { k:{} for k in cutlist}
    back_int_weight = { k:{} for k in cutlist}
    back_eff = { k:{} for k in cutlist}
    punzi_dict = { k:{} for k in cutlist}
    graph = {}
    back_graph = {}
    ncuts = len(cutlist)
    if labellist == []:
        labellist=cutlist
    print NTUPLEDIR
    print "............."
    #prepare ctau ordered array for 1D plot                                                                
    mass_array = []
    ctau_array = []


    #for signal we have the normal efficiency                                                               
    for i, s in enumerate(sign):
        file[s] = TFile(NTUPLEDIR + samples[s]['files'][0] + ".root", "READ") # Read TFile                  
        tree[s] = file[s].Get("ntuple/tree") # Read TTree       
        nevt[s] = (file[s].Get('counter/c_nEvents')).GetBinContent(1)# all gen events before cuts!
        #tree[s] = file[s].Get("skim") # Read TTree       
        #nevt[s] = tree[s].GetEntries("")#if the tree is skimmed, this becomes a relative denominator
        #nevt[s] = (file[s].Get('counter/c_nEvents')).GetBinContent(1)# all gen events before cuts!
        filename = TFile(NTUPLEDIR + samples[s]['files'][0] + ".root", "READ")
        if verbose_add: print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        if verbose_add: print filename
        if verbose_add: print "x-check: n gen events in counter, first bin:"
        if verbose_add: print (filename.Get('counter/c_nEvents')).GetBinContent(1)
        if verbose_add: print "x-check: n entries in counter:"
        if verbose_add: print (filename.Get('counter/c_nEvents')).GetEntries()
        effs[s] = [0]*(ncuts+1)
        effs[s] = [0]*(ncuts+1)
        weight = "1"#"EventWeight"
        var = "isMC"

        if samples[s]['mass'] not in mass_array:
            mass_array.append(samples[s]['mass'])
        if samples[s]['ctau'] not in ctau_array:
            ctau_array.append(samples[s]['ctau'])
        for j, c in enumerate(cutlist):
            tot_gen = nevt[s]
            n = tree[s].GetEntries("(" + cutlist[j] + ")")

            #wat?#test_op = cutlist[j] + " && number_of_matched_Jets>=1"
            #wat?#n = tree[s].GetEntries("(" + test_op + ")")

            ###BUGFIX: efficiency should be computed w.r.t. histo integral
            #hist[s+"_cut"+str(j)] = TH1F(s+"_cut"+str(j), ";"+variable[var]['title'], variable[var]['nbins'], variable[var]['min'], variable[var]['max'])
            #hist[s+"_cut"+str(j)].Sumw2()
            #cutstring = "("+weight+")" + ("*("+cutlist[j]+")" if len(cutlist[j])>0 else "")
            #tree[s].Project(s+"_cut"+str(j), var, cutstring)
            #hist[s+"_cut"+str(j)].SetOption("%s" % tree[s].GetTree().GetEntriesFast())


            if verbose_add: print '\n'
            if verbose_add: print '**********************************************'
            if verbose_add: print "cut: ", c
            if verbose_add: print 'over signal ', s
            if verbose_add: print '\n'
            if verbose_add: print "signal num: ", n
            if verbose_add: print "signal den: ", tot_gen
            #if verbose_add: print "BUGFIX!!!!!!!!!!!"
            #if verbose: print "BUGFIX!!!!!!!!!!!"
            #if verbose_add: print "signal num from integral: ", hist[s+"_cut"+str(j)].Integral()
            #if verbose_add: print "signal den from generator: ", tot_gen
            #if verbose: print "BUGFIX!!!!!!!!!!!"
            if verbose_add: print ("signal eff %.2f") % (float(n)/(tot_gen)*100)
            if tot_gen==0:
                effs[s][j] = float(0.)
            else:
                effs[s][j] = (float(n)/(tot_gen))
            eff_dict[c][s] = {'mass' : samples[s]['mass'], 'ctau' : samples[s]['ctau'], 'eff' :effs[s][j], 'nevents' : n}


    #sort mass array
    masses = np.array(mass_array)
    masses.sort()

    ctaus = np.array(ctau_array)
    ctaus.sort()


    #define multigraph
    mg = TMultiGraph()
    #leg = TLegend(0.78, 0.7, 0.98, 0.98)
    #leg2 = TLegend(0., 0.4, 0.98, 0.98)
    #leg2 = TLegend(0.3, 0.11, 0.65, 0.45)#DCMS,gen matching
    leg2 = TLegend(0.4, 0.11, 0.85, 0.45)#DCMS,summary plot
    leg2 = TLegend(0.4-0.3, 0.11+0.43, 0.85+0.05-0.3, 0.45+0.43)#EXO,summary plot
    leg2 = TLegend(0.4, 0.11, 0.85+0.05, 0.45)#EXO,summary plot

    leg3 = TLegend(0., 0.5, 0.5, 1.)#2 plots

    leg = TLegend(0., 0.4, 0.98, 0.98)
    leg.SetTextSize(0.03)
    leg2.SetTextSize(0.03)
    leg2.SetTextSize(0.025)
    leg.SetBorderSize(0)
    leg2.SetBorderSize(0)
    leg.SetHeader("Signal: c#tau_{#chi}=" +str(ctau_point)+" mm")
    leg2.SetHeader("Signal: c#tau_{#chi}=" +str(ctau_point)+" mm")

    leg3.SetTextSize(0.03)
    leg3.SetTextSize(0.025)
    leg3.SetBorderSize(0)
    leg3.SetHeader("Signal: c#tau_{#chi}=" +str(ctau_point)+" mm")


    #for background let's first consider the cut
    for j, c in enumerate(cutlist):
        print '\n'
        print "cut: ", c
        print 'over background'
        print '\n'
        #then loop over background
        integral = 0
        weighted_integral = 0
        back_tot_events = 0
        for i, s in enumerate(back):
            chain[s] = TChain("ntuple/tree")
            #chain[s] = TChain("skim")
            #print "back: ", s
            back_file = {}
            for p, ss in enumerate(samples[s]['files']):
                back_file[ss] = TFile(NTUPLEDIR + ss + ".root", "READ") # Read TFile                  
                #?#if verbose: print "file: ", ss
                #?#if verbose: print "gen events: ", (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                #?#if verbose: print "tree events: ", (back_file[ss].Get('ntuple/tree')).GetEntries()
                back_tot_events += (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                #back_tot_events += (back_file[ss].Get('counter/c_nEvents')).GetBinContent(1)
                chain[s].Add(NTUPLEDIR + ss + ".root")
            #print "MODIFIED WEIGHT!!!!!!"
            #weight = ("EventWeight*%s/5000." % str(back_tot_events))
            weight = "EventWeight"
            #var = "nCHSJets"
            var = "isMC"
            hist[s] = TH1F(s, ";"+variable[var]['title'], variable[var]['nbins'], variable[var]['min'], variable[var]['max'])
            hist[s].Sumw2()
            cutstring = "("+weight+")" + ("*("+cutlist[j]+")" if len(cutlist[j])>0 else "")
            chain[s].Project(s, var, "")#"1*"+"("+weight+")")
            hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
            #if verbose: print "Hist content, no cut:"
            #if verbose: print hist[s].Print()
            #?#if verbose: print "events in the histo with get entries with empty project: ", hist[s].GetEntries()
            #?#if verbose: print "area under histo with empty project: ", hist[s].Integral()
            chain[s].Project(s, var, cutstring)#"1*"+"("+weight+")")
            hist[s].SetOption("%s" % chain[s].GetTree().GetEntriesFast())
            hist[s].Scale(samples[s]['weight'] if hist[s].Integral() >= 0 else 0)
            #?#if verbose: print "events in the histo with get entries after project: ", hist[s].GetEntries()
            #?#if verbose: print "area under histo after project: ", hist[s].Integral()
            if verbose: print "Hist content, with cut:"
            if verbose: print hist[s].Print()
            integral += hist[s].GetEntries()
            weighted_integral += hist[s].Integral()
        back_int[c] = integral
        back_int_weight[c] = weighted_integral
        if back_tot_events==0:
            back_eff[c] = float(0.)
        else:
            back_eff[c] = float(integral)/float(back_tot_events)
        if verbose: print "cut: ", c
        if verbose: print "back tot events (unweighted):", back_tot_events
        if verbose: print "back integral (unweighted): ", back_int[c]
        if verbose: print "back integral (weighted): ", back_int_weight[c]
        if verbose: print "back eff (unweighted): ", back_eff[c]*100
        if FOM=="signaleff":
            punzi_dict[c]['back'] = {'back' : back_eff[c]*100}
        for i, s in enumerate(sign):
            if verbose: print "signal efficiency: ", eff_dict[c][s]['eff']*100
            if FOM=="punzi":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['eff']/(CL**2/2. + alpha*math.sqrt(back_int_weight[c]) + (CL/2.)*math.sqrt(CL**2 + 4*alpha*math.sqrt(back_int_weight[c]) + 4*back_int_weight[c]))}
            elif FOM=="signaleff":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['eff']*100}
            elif FOM=="entries":
                punzi_dict[c][s] = {'sign': eff_dict[c][s]['nevents']}
            else:
                print "not punzi FOM, aborting!"
                exit()

    if FOM=="signaleff":
        dummy = TGraph()#len(ct),ct, np.array(ct))
        dummy.SetMarkerStyle(0)
        dummy.SetLineWidth(2)
        dummy.SetMarkerSize(1.)
        dummy.SetLineColor(15)
        dummy.SetLineStyle(2)
        if header!="":
            leg2.AddEntry(dummy, header,'')
            leg3.AddEntry(dummy, header,'')

    print "If it runs up to here, abort"
    print eff_dict
    mS = 40
    print masses, ctaus
    #exit()

    #for each cut, we need a graph                                                                          
    for j, c in enumerate(cutlist):
    #first let's build the ordered punzi vector w.r.t. masses, for a chosen ctau                            
        punzi_array = []
        back_array = []
        for la in masses:
            #la = str(a)
            #if la== 0.001:
            #    st = CHANNEL+"_mh"+str(mS)+"_ctau0"
            #elif la==0.05 or la==0.1:
            #    st = CHANNEL+"_mh"+str(mS)+"_ctau"+str(str(la).replace("0.","0p"))
            #else:
            #    st = CHANNEL+"_mh"+str(mS)+"_ctau"+str(int(la))
            #SUSY_mh200_pl1000
            st = CHANNEL+"_mh"+str(la)+"_pl"+str(ctau_point)                                                        
            punzi_array.append(punzi_dict[c][st]['sign'])
        mass = array('d', masses)
        ct = array('d', ctaus)
        p_array = array('d',punzi_array)
        graph[c] = TGraph(len(mass),mass, np.array(p_array))                                                   
        #graph[c] = TGraph(len(ct),ct, np.array(p_array))
        graph[c].SetMarkerStyle(markers[j])#21
        graph[c].SetLineWidth(3)
        graph[c].SetMarkerSize(1.2)
        graph[c].SetMarkerColor(colors[j])
        graph[c].SetLineColor(colors[j])
        graph[c].SetFillColor(colors[j])
        #graph[c].SetLogx()                                                                                 

        leg.AddEntry(graph[c],labellist[j],'PL')
        leg2.AddEntry(graph[c],labellist[j],'PL')
        leg3.AddEntry(graph[c],labellist[j],'PL')
        mg.Add(graph[c])

        if FOM=="signaleff":
        #add plot for background                                                                            
            for a in ctaus:
                back_array.append(punzi_dict[c]['back']['back'])
            mass = array('d', masses)
            ct = array('d', ctaus)
            e_array = array('d',back_array)
            back_graph[c] = TGraph(len(mass),mass, np.array(e_array))
            #back_graph[c] = TGraph(len(ct),ct, np.array(e_array))
            back_graph[c].SetMarkerStyle(0)
            back_graph[c].SetLineWidth(2)
            back_graph[c].SetMarkerSize(1.)
            back_graph[c].SetMarkerColor(colors[j])
            back_graph[c].SetLineColor(colors[j])
            back_graph[c].SetLineStyle(2)
            back_graph[c].SetFillColor(colors[j])
            #back_graph[c].SetLogx()                                                                        
            #leg.AddEntry(back_graph[c],labellist[j]+" bkg.",'PL')
            #w#leg2.AddEntry(back_graph[c],labellist[j]+" bkg.",'PL')                                         
            #w#mg.Add(back_graph[c])

    if FOM=="signaleff":
        #dummy = TGraph(len(ct),ct, np.array(e_array))
        dummy = TGraph(len(mass),mass, np.array(e_array))
        dummy.SetMarkerStyle(0)
        dummy.SetLineWidth(2)
        dummy.SetMarkerSize(1.)
        dummy.SetLineColor(15)
        dummy.SetLineStyle(2)
        #w#leg2.AddEntry(dummy, 'cuts on bkg.','PL')


    #cmg = TCanvas("cmg", "cmg", 2000, 1400)
    #cmg = TCanvas("cmg", "cmg", 2000, 800)#best
    #cmg = TCanvas("cmg", "cmg", 1200, 1000)
    cmg = TCanvas("cmg", "cmg", 1300, 800)#DCMS
    cmg.cd()
    cmg.SetGrid()

    #cmg.SetLogx()

    #if FOM=="signaleff":
    #    cmg.SetLogx()
    #pad1 = TPad("pad1", "pad1", 0, 0., 0.85, 1.0)
    #pad1 = TPad("pad1", "pad1", 0, 0., 0.7, 1.0)
    #pad1.SetGrid()
    #pad1.SetLogx()
    if FOM=="signaleff":
        print "LOL"
        #pad1.SetLogy()
    #pad1.SetLogy()
    #pad1.Draw()
    #pad1.cd()

    #W#if FOM=="signaleff":
        #w#mg.SetMaximum(101)
        #mg.SetMinimum(1.e-50)
    mg.SetMinimum(0.)#!!
    mg.Draw("APL")
    mg.GetXaxis().SetTitleSize(0.05)
    mg.GetYaxis().SetTitleSize(0.05)
    mg.GetXaxis().SetTitle('m_{#chi} (GeV)')#('c#tau_{#pi} (mm)')
    mg.GetYaxis().SetTitleOffset(0.9);
    if FOM=="punzi":
        mg.GetYaxis().SetTitle('Punzi significance @ '+str(alpha)+' #sigma, '+CHANNEL+' cuts')
        #mg.GetYaxis().SetTitleOffset(1.5)
    elif FOM=="signaleff":
        mg.GetYaxis().SetTitle('Signal efficiency, '+CHANNEL+' cuts (%)')
        #mg.GetYaxis().SetTitle('Signal gen-matching efficiency, '+CHANNEL+' (%)')
    elif FOM=="entries":
        mg.GetYaxis().SetTitle('Signal entries surviving cuts')
    else:
        print "not punzi FOM, aborting"

    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(33)
    latex.SetTextFont(62)
    latex.DrawLatex(0.25, 0.96, "CMS")
    latex.SetTextFont(52)
    latex.DrawLatex(0.66, 0.96, "Simulation Preliminary")
    cmg.Update()

    cmg.cd()
    leg2.SetTextSize(0.04)
    #leg.Clear()#?????????
    #w#leg2.Draw()

    cmgL = TCanvas("cmgL", "cmgL", 2000, 800)#DCMS
    cmgL.cd()

    #pad2 = TPad("pad2", "pad2", 0.85, 0., 1, 1.0)
    #pad2 = TPad("pad2", "pad2", 0.7, 0., 1, 1.0)
    #pad2.SetGrid()
    #pad2.SetLogx()macro/VBF_punzi_LLP_AOD.py
    #pad2.Draw()
    #pad2.cd()
    leg3.SetTextSize(0.04)
    #leg.Clear()#?????????
    leg3.Draw()
    cmgL.Update()

    
    if FOM=="punzi":
        cmg.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_ctau"+str(ctau_point)+"_"+str(alpha)+"sigma"+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_ctau"+str(ctau_point)+"_"+str(alpha)+"sigma"+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_ctau"+str(ctau_point)+"_"+str(alpha)+"sigma"+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "Punzi_correct_"+CHANNEL+"_ctau"+str(ctau_point)+"_"+str(alpha)+"sigma"+additional_string+"_L.png")
    elif FOM=="signaleff":
        cmg.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "SignalEff_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+"_L.png")
    elif FOM=="entries":
        cmg.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+".pdf")
        cmg.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+".png")
        #cmgL.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+"_L.pdf")
        #cmgL.Print(OUTPUTDIR + "SignalEntries_"+CHANNEL+"_ctau"+str(ctau_point)+additional_string+"_L.png")
    else:
        print "not punzi FOM, aborting"
    

    if not options.bash: raw_input("Press Enter to continue...")
    cmg.Close()
    cmgL.Close()





######
#fom = ["entries","signaleff", "punzi"]
#fom = ["entries","punzi"]
#fom = ["entries","signaleff","punzi"]
fom = ["punzi"]
#fom = ["punzi","signaleff"]
fom = ["signaleff"]

colors = [856, 2, 881, 798, 602, 921, 870, 906, 838, 420,]
colors = [856, 2, 881, 798, 602, 921, 870, 906, 801, 410, 4, 634, 838, 1, 398, 6,7,8,9,10,11]

n_matches = 0


combis = {
    2000: [250,600],
    1500: [200,500],
    1000: [150,400],
    600: [50,150],
    #400: [50,100],
    }

for a in fom:

    for MH in combis.keys():
        for m in combis[MH]:
            print MH,m

            calc_punzi_FOM_vs_ctau(
                [
                    #"nTagJets_0p994_JJ>1",
                    "nTagJets_0p996_JJ>1",
                    #"nTagFatJets_0p999>=1",
                    #"FatJets[0].sigprob>0.999",
                    #"FatJets[0].sigprob>0.9999",
                    #"FatJets[0].sigprob>0.99995",
                    #"FatJets[0].sigprob>0.99999",
                    #"nTagFatJets_0p9995>=1",
                    #"nTagFatJets_0p9999>=1",
                    #"nTagFatJets_0p99995>=1",
                    #"nTagFatJets_0p99999>=1",
                    #"nTagFatJets_0p999999>=1",
                    #"isTagAK8_0p9999_170",
                    #"isTagAK8_0p9999_200",
                    #"isTagAK8_0p9999_250",
                    #"isTagAK8_0p99999_250",
                    #"isTagAK8_0p9999_300",
                    #"isTagAK8_0p99999_350",
                    #"isTagAK8_0p9999_350",
                    #"MinFatJetMetDPhiBarrel>0.5",
                    #"MinFatJetMetDPhiBarrel>0.5 && isTagAK8_0p9999_250",
                ],
                [
                ],
                linelist = [4,2,2,1,1,1,3,3,3,5,5,    2,2,1,1,3,3,4,5,5,5,5,5,5],
                header = "",#str(n_matches)+" VBF jets gen matched",
                mH = MH,
                mS = m,
                additional_string = "_HH_acceptance",#"_nTagFatJets_nTagJets", 
                alpha = 2,
                CL = 5,
                FOM = a
            )


exit()


#Combination
for a in fom:

    for MH in combis.keys():
        for m in combis[MH]:
            print MH,m

            calc_punzi_FOM_vs_ctau(
                [
                    #"isMC",
                    #"isAK8_JJ_170",
                    #"isAK8_JJ_200",
                    #"isAK8_JJ_250",
                    "nTagFatJets_0p999>=1",
                    #"FatJets[0].sigprob>0.999",
                    #"FatJets[0].sigprob>0.9995",
                    #"FatJets[0].sigprob>0.9999",
                    "nTagFatJets_0p9999>=1",
                    "nTagFatJets_0p9999>=1 && isAK8_JJ_200",
                    "nTagFatJets_0p9999>=1 && isAK8_JJ_250",
                    "nTagFatJets_0p99999>=1",
                    "nTagFatJets_0p9999>=1 || nTagJets_0p95>=2",
                    "(nTagFatJets_0p9999>=1 && isAK8_JJ_250) || nTagJets_0p95>=2",
                    #"nTagJets_cutbased>=2",
                    "nTagJets_0p95>=2",
                    "nTagJets_0p996>=2",
                    "(nTagFatJets_0p9999>=1 || nTagJets_0p95>=2) && !(nTagFatJets_0p9999>=1 && nTagJets_0p95>=2)",
                    "nTagFatJets_0p9999>=1 && nTagJets_0p95>=2",
                ],
                [
                    #"isMC",
                    #"isAK8_JJ_170",
                    #"isAK8_JJ_200",
                    #"isAK8_JJ_250",
                    "nTagFatJets_0p999>=1",
                    #"FatJets[0].sigprob>0.999",
                    #"FatJets[0].sigprob>0.9995",
                    #"FatJets[0].sigprob>0.9999",
                    "nTagFatJets_0p9999>=1",
                    "nTagFatJets_0p9999>=1 && isAK8_JJ_200",
                    "nTagFatJets_0p9999>=1 && isAK8_JJ_250",
                    "nTagFatJets_0p99999>=1",
                    "OR",
                    "OR + pT>250",
                    #"nTagJets_cutbased>=2",
                    "nTagJets_0p95>=2",
                    "nTagJets_0p996>=2",
                    "XOR",
                    "overlap boosted+resolved",
                ],
                linelist = [2,2,2,2,2,1,1,3,3,4,5],
                header = "",#str(n_matches)+" VBF jets gen matched",
                mH = MH,
                mS = m,
                additional_string = "_nTagFatJets_nTagJets", 
                alpha = 2,
                CL = 5,
                FOM = a
            )


exit()

#Event-wise
for a in fom:

    for MH in combis.keys():
        for m in combis[MH]:
            print MH,m

            calc_punzi_FOM_vs_ctau(
                [
                    "isMC && HT>100",
                    #"HT>100",
                    #"HT>200",
                    "HT>300",
                    #"HT>400",
                    "MEt.pt>300",
                    "MEt.pt>400",
                    #"MinJetMetDPhi>0.5",
                    "MinJetMetDPhi>1 && nCHSJets>0",
                    "MinFatJetMetDPhi>1 && nCHSFatJets>0",
                    #"MinFatJetMetDPhi>2 && nCHSFatJets>0",
                    #"MinJetMetDPhi>3",
                    #"MinJetMetDPhiBarrel>0.5",
                    "MinJetMetDPhiBarrel>1 && nCHSJetsAcceptanceCalo>0",
                    "MinFatJetMetDPhiBarrel>1 && nCHSFatJetsAcceptanceCalo>0",
                    #"MinFatJetMetDPhiBarrel>2 && nCHSFatJetsAcceptanceCalo>0",
                    #"MinJetMetDPhiBarrel>3",

                ],
                [
                    "Pre-sel & HT>100",
                    #"HT>100",
                    #"HT>200",
                    "HT>300",
                    #"HT>400",
                    "MEt.pt>300",
                    "MEt.pt>400",
                    "MinJetMetDPhi>1",
                    "MinFatJetMetDPhi>1",
                    #"MinFatJetMetDPhi>2",
                    "MinJetMetDPhiBarrel>1",
                    "MinFatJetMetDPhiBarrel>1",
                    #"MinFatJetMetDPhiBarrel>2",
                ],
                linelist = [1,1,2,2,3,3,3,4,4,4],
                header = "",#str(n_matches)+" VBF jets gen matched",
                mH=MH,
                mS=m,
                additional_string = "_event_wise", 
                alpha = 2,
                CL = 5,
                FOM = a
            )


exit()

#Resolved
for a in fom:

    for MH in combis.keys():
        for m in combis[MH]:
            print MH,m

            calc_punzi_FOM_vs_ctau(
                [
                    #"isMC",
                    "nTagJets_cutbased>=2",
                    #"nTagJets_0p9>=2",
                    "nTagJets_0p95>=2",
                    "nTagJets_0p95>=2 && HT>100",
                    "nTagJets_0p96>=2",
                    "nTagJets_0p96>=2 && HT>100",
                    "nTagJets_0p97>=2",
                    "nTagJets_0p98>=2",
                    "nTagJets_0p99>=2",
                    "nTagJets_0p995>=2",
                    "nTagJets_0p996>=2",
                    "nTagJets_0p999>=2",

                ],
                [
                    #"Pre-selections",
                    "nTagJets_cutbased>=2",
                    #"nTagJets_0p9>=2",
                    "nTagJets_0p95>=2",
                    "nTagJets_0p95>=2 && HT>100",
                    "nTagJets_0p96>=2",
                    "nTagJets_0p96>=2 && HT>100",
                    "nTagJets_0p97>=2",
                    "nTagJets_0p98>=2",
                    "nTagJets_0p99>=2",
                    "nTagJets_0p995>=2",
                    "nTagJets_0p996>=2",
                    "nTagJets_0p999>=2",
                ],
                header = "",#str(n_matches)+" VBF jets gen matched",
                mH=MH,
                mS=m,
                additional_string = "_nTagJets", 
                alpha = 2,
                CL = 5,
                FOM = a
            )


exit()




#Boosted
for a in fom:

    for MH in combis.keys():
        for m in combis[MH]:
            print MH,m

            calc_punzi_FOM_vs_ctau(
                [
                    #"HT>100",
                    "nTagFatJets_0p8>=1",
                    "nTagFatJets_0p9>=1",
                    "nTagFatJets_0p99>=1",
                    "nTagFatJets_0p999>=1",
                    "nTagFatJets_0p9999>=1",
                    "nTagFatJets_0p9999>=1 && HT>100",
                    "FatJets[0].sigprob>0.9999",
                    "FatJets[0].sigprob>0.9999 && HT>100",
                    "nTagFatJets_0p99995>=1",
                    "nTagFatJets_0p99999>=1",
                ],
                [
                    #"HT>100",
                    "nTagFatJets_0p8>=1",
                    "nTagFatJets_0p9>=1",
                    "nTagFatJets_0p99>=1",
                    "nTagFatJets_0p999>=1",
                    "nTagFatJets_0p9999>=1",
                    "nTagFatJets_0p9999>=1 && HT>100",
                    "FatJets[0].sigprob>0.9999",
                    "FatJets[0].sigprob>0.9999 && HT>100",
                    "nTagFatJets_0p99995>=1",
                    "nTagFatJets_0p99999>=1",
                ],
                header = "",#str(n_matches)+" VBF jets gen matched",
                mH=MH,
                mS=m,
                additional_string = "_nTagFatJets", 
                alpha = 2,
                CL = 5,
                FOM = a
            )


exit()




















for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
            "isMC",
            "Jets[0].isGenMatchedCaloCorrLLPAccept",
            "Jets[0].isGenMatchedCaloCorrLLPAccept & Jets[1].isGenMatchedCaloCorrLLPAccept",
            "FatJets[0].isGenMatchedCaloCorrLLPAccept",
            ],
        [
            "Pre-selections",
            "j_{0} gen matched",
            "j_{0} & j_{1} gen matched",
            "J_{0} gen matched",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        #mH=600,
        #mS=50,#150,#50,#
        #mH=400,
        #mS=100,#50,#
        additional_string = "_gen_matching", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()






###v3 calo AOD 2018
fom = ["punzi"]
fom = ["signaleff"]

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3 && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.3",
        selection["METPreSelSUSYAOD"] + " && fabs(Jets[0].timeRecHitsEB)>0.296 && fabs(Jets[0].gammaMaxET)<0.126 && Jets[0].minDeltaRPVTracks>0.04 && Jets[0].cHadEFrac<0.086 && (Jets[0].timeRecHitsEB>-100 && Jets[0].gammaMaxET>-100 && Jets[0].minDeltaRPVTracks<999 && Jets[0].cHadEFrac>-1)  && fabs(Jets[1].timeRecHitsEB)>0.296 && fabs(Jets[1].gammaMaxET)<0.126 && Jets[1].minDeltaRPVTracks>0.04 && Jets[1].cHadEFrac<0.086 && (Jets[1].timeRecHitsEB>-100 && Jets[1].gammaMaxET>-100 && Jets[1].minDeltaRPVTracks<999 && Jets[1].cHadEFrac>-1)",
            ],
        [
        "Pre-selections",
        #"AK8 jet[0] cHadEFrac",
        #"+ gammaMaxET",
        #"+ minDeltaRPVTracks",
        #"+ alphaMax",
        "Cut JJ on AK4 jet[0] and [1]",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        #mH=1000,
        #mS=400,#150,#
        #mH=600,
        #mS=150,#50,#
        mH=400,
        mS=100,#50,#
        additional_string = "_JJ_cut_based", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''





for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.06",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.05",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.03",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.02",

        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.03 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.05 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.05 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.2",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.15",
        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.08 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.1",

        #selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03 && FatJets[0].gammaMaxET>-1. && FatJets[0].gammaMaxET<0.02 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].minDeltaRPVTracks>0.3 && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.3",
        selection["METPreSelSUSYAOD"] + " && fabs(Jets[0].timeRecHitsEB)>0.296 && fabs(Jets[0].gammaMaxET)<0.126 && Jets[0].minDeltaRPVTracks>0.04 && Jets[0].cHadEFrac<0.086 && (Jets[0].timeRecHitsEB>-100 && Jets[0].gammaMaxET>-100 && Jets[0].minDeltaRPVTracks<999 && Jets[0].cHadEFrac>-1)  && fabs(Jets[1].timeRecHitsEB)>0.296 && fabs(Jets[1].gammaMaxET)<0.126 && Jets[1].minDeltaRPVTracks>0.04 && Jets[1].cHadEFrac<0.086 && (Jets[1].timeRecHitsEB>-100 && Jets[1].gammaMaxET>-100 && Jets[1].minDeltaRPVTracks<999 && Jets[1].cHadEFrac>-1)",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] cHadEFrac",
        "+ gammaMaxET",
        #"+ e HB/e>0.06",
        #"+ e HB/e>0.05",
        #"+ e HB/e>0.04",
        "+ e HB/e>0.03",
        #"+ e HB/e>0.02",
        #"+ minDeltaRPVTracks comb: 0.03,0.02,0.3",
        "; + minDeltaRPVTracks comb: 0.03,0.03,0.3",
        #"+ minDeltaRPVTracks comb: 0.05,0.05,0.3",
        "+ minDeltaRPVTracks comb: 0.08,0.08,0.2",
        #"+ minDeltaRPVTracks comb: 0.08,0.08,0.15",
        #"+ minDeltaRPVTracks comb: 0.08,0.08,0.1",
        #"+ alphaMax",
        "Cut JJ on AK4 jet[0] and [1]",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        #mH=2000,
        #mS=250,
        #mH=1500,
        #mS=200,
        #mH=1000,
        #mS=150,#400,
        #mH=600,
        #mS=50,#150,#50,#
        mH=400,
        mS=50,#100,#50,#
        additional_string = "_combination", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()


'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.06",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.07",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.02",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.04",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.06",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] rel energyRecHitsHB>0.02",
        "AK8 jet[0] rel energyRecHitsHB>0.04",
        "AK8 jet[0] rel energyRecHitsHB>0.06",
        "AK8 jet[0] rel energyRecHitsHB>0.07",
        #"AK8 jet[1] rel energyRecHitsHB>0.02",
        #"AK8 jet[1] rel energyRecHitsHB>0.04",
        #"AK8 jet[1] rel energyRecHitsHB>0.06",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_relative_energyRecHitsHB_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsEB>-1. && FatJets[0].energyRecHitsEB/FatJets[0].energy>0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsEB>-1. && FatJets[0].energyRecHitsEB/FatJets[0].energy>0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsEB>-1. && FatJets[0].energyRecHitsEB/FatJets[0].energy>0.06",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsEB>-1. && FatJets[0].energyRecHitsEB/FatJets[0].energy>0.07",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsEB>-1. && FatJets[1].energyRecHitsEB/FatJets[1].energy>0.02",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsEB>-1. && FatJets[1].energyRecHitsEB/FatJets[1].energy>0.04",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsEB>-1. && FatJets[1].energyRecHitsEB/FatJets[1].energy>0.06",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] rel energyRecHitsEB>0.02",
        "AK8 jet[0] rel energyRecHitsEB>0.04",
        "AK8 jet[0] rel energyRecHitsEB>0.06",
        "AK8 jet[0] rel energyRecHitsEB>0.07",
        #"AK8 jet[1] rel energyRecHitsEB>0.02",
        #"AK8 jet[1] rel energyRecHitsEB>0.04",
        #"AK8 jet[1] rel energyRecHitsEB>0.06",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_relative_energyRecHitsEB_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.6",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.7",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.8",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.85",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.9",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].nHadEFrac>-1. && FatJets[0].nHadEFrac>0.95",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] nHadEFrac>0.6",
        "AK8 jet[0] nHadEFrac>0.7",
        "AK8 jet[0] nHadEFrac>0.8",
        "AK8 jet[0] nHadEFrac>0.85",
        "AK8 jet[0] nHadEFrac>0.9",
        "AK8 jet[0] nHadEFrac>0.95",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_nHadEFrac_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''




'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.06",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].energyRecHitsHB>-1. && FatJets[0].energyRecHitsHB/FatJets[0].energy>0.07",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].energyRecHitsHB>-1. && FatJets[1].energyRecHitsHB/FatJets[1].energy>0.06",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] rel energyRecHitsHB>0.02",
        "AK8 jet[0] rel energyRecHitsHB>0.04",
        "AK8 jet[0] rel energyRecHitsHB>0.06",
        "AK8 jet[0] rel energyRecHitsHB>0.07",
        "AK8 jet[1] rel energyRecHitsHB>0.02",
        "AK8 jet[1] rel energyRecHitsHB>0.04",
        "AK8 jet[1] rel energyRecHitsHB>0.06",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_relative_energyRecHitsHB", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''


'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].puppiTau21>-1. && FatJets[0].puppiTau21<0.7",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].puppiTau21>-1. && FatJets[0].puppiTau21<0.6",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].puppiTau21>-1. && FatJets[0].puppiTau21<0.5",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].puppiTau21>-1. && FatJets[1].puppiTau21<0.7",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].puppiTau21>-1. && FatJets[1].puppiTau21<0.6",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].puppiTau21>-1. && FatJets[1].puppiTau21<0.5",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] puppiTau21<0.7",
        "AK8 jet[0] puppiTau21<0.6",
        "AK8 jet[0] puppiTau21<0.5",
        "AK8 jet[1] puppiTau21<0.7",
        "AK8 jet[1] puppiTau21<0.6",
        "AK8 jet[1] puppiTau21<0.5",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_puppiTau21", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''


'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && fabs(FatJets[0].timeRecHitsEB)>0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && fabs(FatJets[0].timeRecHitsEB)>0.5",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && fabs(FatJets[0].timeRecHitsEB)>1.",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && fabs(FatJets[0].timeRecHitsEB)>2.",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && FatJets[0].timeRecHitsEB>0.",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && FatJets[0].timeRecHitsEB>0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && FatJets[0].timeRecHitsEB>0.5",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && FatJets[0].timeRecHitsEB>1.",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].timeRecHitsEB>-100. && FatJets[0].timeRecHitsEB>2.",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].timeRecHitsEB>-100. && fabs(FatJets[1].timeRecHitsEB)>0.2",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].timeRecHitsEB>-100. && fabs(FatJets[1].timeRecHitsEB)>0.5",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].timeRecHitsEB>-100. && fabs(FatJets[1].timeRecHitsEB)>1.",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].timeRecHitsEB>-100. && fabs(FatJets[1].timeRecHitsEB)>2.",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] |timeRecHitsEB|>0.2",
        "AK8 jet[0] |timeRecHitsEB|>0.5",
        "AK8 jet[0] |timeRecHitsEB|>1",
        "AK8 jet[0] |timeRecHitsEB|>2",
        "AK8 jet[0] timeRecHitsEB>0.",
        "AK8 jet[0] timeRecHitsEB>0.2",
        "AK8 jet[0] timeRecHitsEB>0.5",
        "AK8 jet[0] timeRecHitsEB>1",
        "AK8 jet[0] timeRecHitsEB>2",
        #"AK8 jet[1] |timeRecHitsEB|>0.2",
        #"AK8 jet[1] |timeRecHitsEB|>0.5",
        #"AK8 jet[1] |timeRecHitsEB|>1",
        #"AK8 jet[1] |timeRecHitsEB|>2",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_timeRecHitsEB_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''


'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].minDeltaRPVTracks<999. && FatJets[0].minDeltaRPVTracks>0.1",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].minDeltaRPVTracks<999. && FatJets[0].minDeltaRPVTracks>0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].minDeltaRPVTracks<999. && FatJets[0].minDeltaRPVTracks>0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].minDeltaRPVTracks<999. && FatJets[0].minDeltaRPVTracks>0.4",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].minDeltaRPVTracks<999. && FatJets[0].minDeltaRPVTracks>0.5",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].minDeltaRPVTracks<999. && FatJets[1].minDeltaRPVTracks>0.1",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].minDeltaRPVTracks<999. && FatJets[1].minDeltaRPVTracks>0.2",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].minDeltaRPVTracks<999. && FatJets[1].minDeltaRPVTracks>0.3",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] minDeltaRPVTracks>0.1",
        "AK8 jet[0] minDeltaRPVTracks>0.2",
        "AK8 jet[0] minDeltaRPVTracks>0.3",
        "AK8 jet[0] minDeltaRPVTracks>0.4",
        "AK8 jet[0] minDeltaRPVTracks>0.5",
        #"AK8 jet[1] minDeltaRPVTracks>0.1",
        #"AK8 jet[1] minDeltaRPVTracks>0.2",
        #"AK8 jet[1] minDeltaRPVTracks>0.3",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_minDeltaRPVTracks_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.1",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.08",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.06",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.03",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].gammaMaxET>-100. && FatJets[0].gammaMaxET<0.01",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].gammaMaxET>-100. && FatJets[1].gammaMaxET<0.3",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].gammaMaxET>-100. && FatJets[1].gammaMaxET<0.2",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].gammaMaxET>-100. && FatJets[1].gammaMaxET<0.1",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] gammaMaxET<0.3",
        "AK8 jet[0] gammaMaxET<0.2",
        "AK8 jet[0] gammaMaxET<0.1",
        "AK8 jet[0] gammaMaxET<0.08",
        "AK8 jet[0] gammaMaxET<0.06",
        "AK8 jet[0] gammaMaxET<0.04",
        "AK8 jet[0] gammaMaxET<0.03",
        "AK8 jet[0] gammaMaxET<0.02",
        "AK8 jet[0] gammaMaxET<0.01",
        #"AK8 jet[1] gammaMaxET<0.3",
        #"AK8 jet[1] gammaMaxET<0.2",
        #"AK8 jet[1] gammaMaxET<0.1",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_gammaMaxET_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.1",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.08",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.06",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.04",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.03",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.02",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].cHadEFrac>-1. && FatJets[0].cHadEFrac<0.01",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].cHadEFrac>-1. && FatJets[1].cHadEFrac<0.2",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].cHadEFrac>-1. && FatJets[1].cHadEFrac<0.1",
        #selection["METPreSelSUSYAOD"] + " && FatJets[1].cHadEFrac>-1. && FatJets[1].cHadEFrac<0.08",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] cHadEFrac<0.2",
        "AK8 jet[0] cHadEFrac<0.1",
        "AK8 jet[0] cHadEFrac<0.08",
        "AK8 jet[0] cHadEFrac<0.06",
        "AK8 jet[0] cHadEFrac<0.04",
        "AK8 jet[0] cHadEFrac<0.03",
        "AK8 jet[0] cHadEFrac<0.02",
        "AK8 jet[0] cHadEFrac<0.01",
        #"AK8 jet[1] cHadEFrac<0.2",
        #"AK8 jet[1] cHadEFrac<0.1",
        #"AK8 jet[1] cHadEFrac<0.08",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_cHadEFrac_tune", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

'''
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.5",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[0].alphaMax>-100 && FatJets[0].alphaMax<0.1",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].alphaMax>-100 && FatJets[1].alphaMax<0.5",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].alphaMax>-100 && FatJets[1].alphaMax<0.3",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].alphaMax>-100 && FatJets[1].alphaMax<0.2",
        selection["METPreSelSUSYAOD"] + " && FatJets[1].alphaMax>-100 && FatJets[1].alphaMax<0.1",
            ],
        [
        "Pre-selections",
        "AK8 jet[0] alphaMax<0.5",
        "AK8 jet[0] alphaMax<0.3",
        "AK8 jet[0] alphaMax<0.2",
        "AK8 jet[0] alphaMax<0.1",
        "AK8 jet[1] alphaMax<0.5",
        "AK8 jet[1] alphaMax<0.3",
        "AK8 jet[1] alphaMax<0.2",
        "AK8 jet[1] alphaMax<0.1",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#400,#150,#
        additional_string = "_AK8alphaMax", 
        alpha = 2,
        CL = 5,
        FOM = a
        )
exit()
'''


'''
JiaJing
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        selection["METPreSelSUSYAOD"] + " && fabs(Jets[0].timeRecHitsEB)>0.296 && fabs(Jets[0].gammaMaxET)<0.126 && Jets[0].minDeltaRPVTracks>0.04 && Jets[0].cHadEFrac<0.086 && (Jets[0].timeRecHitsEB>-100 && Jets[0].gammaMaxET>-100 && Jets[0].minDeltaRPVTracks<999 && Jets[0].cHadEFrac>-1)",
        selection["METPreSelSUSYAOD"] + " && fabs(Jets[1].timeRecHitsEB)>0.296 && fabs(Jets[1].gammaMaxET)<0.126 && Jets[1].minDeltaRPVTracks>0.04 && Jets[1].cHadEFrac<0.086 && (Jets[1].timeRecHitsEB>-100 && Jets[1].gammaMaxET>-100 && Jets[1].minDeltaRPVTracks<999 && Jets[1].cHadEFrac>-1)",
        selection["METPreSelSUSYAOD"] + " && fabs(Jets[0].timeRecHitsEB)>0.296 && fabs(Jets[0].gammaMaxET)<0.126 && Jets[0].minDeltaRPVTracks>0.04 && Jets[0].cHadEFrac<0.086 && (Jets[0].timeRecHitsEB>-100 && Jets[0].gammaMaxET>-100 && Jets[0].minDeltaRPVTracks<999 && Jets[0].cHadEFrac>-1)  && fabs(Jets[1].timeRecHitsEB)>0.296 && fabs(Jets[1].gammaMaxET)<0.126 && Jets[1].minDeltaRPVTracks>0.04 && Jets[1].cHadEFrac<0.086 && (Jets[1].timeRecHitsEB>-100 && Jets[1].gammaMaxET>-100 && Jets[1].minDeltaRPVTracks<999 && Jets[1].cHadEFrac>-1)",


        selection["METPreSelSUSYAOD"] + " && fabs(FatJets[0].timeRecHitsEB)>0.296 && fabs(FatJets[0].gammaMaxET)<0.126 && FatJets[0].minDeltaRPVTracks>0.04 && FatJets[0].cHadEFrac<0.086 && (FatJets[0].timeRecHitsEB>-100 && FatJets[0].gammaMaxET>-100 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].cHadEFrac>-1)",
        selection["METPreSelSUSYAOD"] + " && fabs(FatJets[1].timeRecHitsEB)>0.296 && fabs(FatJets[1].gammaMaxET)<0.126 && FatJets[1].minDeltaRPVTracks>0.04 && FatJets[1].cHadEFrac<0.086 && (FatJets[1].timeRecHitsEB>-100 && FatJets[1].gammaMaxET>-100 && FatJets[1].minDeltaRPVTracks<999 && FatJets[1].cHadEFrac>-1)",
        selection["METPreSelSUSYAOD"] + " && fabs(FatJets[0].timeRecHitsEB)>0.296 && fabs(FatJets[0].gammaMaxET)<0.126 && FatJets[0].minDeltaRPVTracks>0.04 && FatJets[0].cHadEFrac<0.086 && (FatJets[0].timeRecHitsEB>-100 && FatJets[0].gammaMaxET>-100 && FatJets[0].minDeltaRPVTracks<999 && FatJets[0].cHadEFrac>-1)  && fabs(FatJets[1].timeRecHitsEB)>0.296 && fabs(FatJets[1].gammaMaxET)<0.126 && FatJets[1].minDeltaRPVTracks>0.04 && FatJets[1].cHadEFrac<0.086 && (FatJets[1].timeRecHitsEB>-100 && FatJets[1].gammaMaxET>-100 && FatJets[1].minDeltaRPVTracks<999 && FatJets[1].cHadEFrac>-1)",
            ],
        [
        "Pre-selections",
        "Cut JJ on AK4 jet[0]",
        "Cut JJ on AK4 jet[1]",
        "Cut JJ on AK4 jet[0] and [1]",
        "Cut JJ on AK8 jet[0]",
        "Cut JJ on AK8 jet[1]",
        "Cut JJ on AK8 jet[0] and [1]",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=400,#150,#
        additional_string = "_test", 
        alpha = 2,
        CL = 5,
        FOM = a
        )


exit()
'''

##test
fom = ["punzi"]
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>10",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>20",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>30",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>40",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>50",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>100",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].cHadEFrac<0.1",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].cHadEFrac<0.05",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.1",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.8",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.95",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.8",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.95",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<6",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<5",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<4",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<3",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2",
        #combis
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<5 && Jets[1].nSelectedTracks<5",
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2",
        
        selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.9 && Jets[0].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.95 && Jets[0].nHadEFrac>0.95",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.1 && Jets[0].cHadEFrac<0.1",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05 && Jets[2].cHadEFrac<0.2",
        
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2 && MEt.sign>20",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05  && MEt.sign>20",
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2 && MEt.sign>30",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05  && MEt.sign>30",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05 && Jets[2].cHadEFrac<0.05",
        #"HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v",
            ],
        [
        "Pre-selections",
        #"+ MEt.sign>50",
        #"+ MEt.sign>100",
        #"Jets[0].cHadEFrac<0.1",
        #"Jets[0].cHadEFrac<0.05",
        #"Jets[1].cHadEFrac<0.1",
        #"Jets[1].cHadEFrac<0.05",
        #"j0 nHadEFrac>0.8",
        #"j0 nHadEFrac>0.9",
        #"j0 nHadEFrac>0.95",
        #"j1 nHadEFrac>0.8",
        #"j1 nHadEFrac>0.9",
        #"j1 nHadEFrac>0.95",
        #"j0 sel tracks<6",
        #"j0 sel tracks<5",
        #"j0 sel tracks<4",
        #"j0 sel tracks<3",
        #"j0 sel tracks<2",
        #combis
        #"j0,j1 n sel tracks<5",
        "j0,j1 n sel tracks<2",
        "j0,j1 nHadEFrac>0.9",
        #"j0,j1 nHadEFrac>0.95",
        "j0,j1 cHadEFrac<0.1",
        "j0,j1 cHadEFrac<0.05",
        "j0,j1<0.05, j2 cHadEFrac<0.2",
        #"j0,j1,j2 cHadEFrac<0.05",
        "j0,j1 n sel tracks<2 && MEt.sign>20",
        "j0,j1 cHadEFrac<0.05 && MEt.sign>20",
        "j0,j1 n sel tracks<2 && MEt.sign>30",
        "j0,j1 cHadEFrac<0.05 && MEt.sign>30",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=400,#150,#
        additional_string = "_test", 
        alpha = 2,
        CL = 5,
        FOM = a
        )

    calc_punzi_FOM_vs_ctau(
        [
        selection["METPreSelSUSYAOD"],
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>10",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>20",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>30",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>40",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>50",
        #selection["METPreSelSUSYAOD"] + " && MEt.sign>100",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].cHadEFrac<0.1",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].cHadEFrac<0.05",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.1",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.8",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nHadEFrac>0.95",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.8",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.95",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<6",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<5",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<4",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<3",
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2",
        #combis
        #selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<5 && Jets[1].nSelectedTracks<5",
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2",
        
        selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.9 && Jets[0].nHadEFrac>0.9",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].nHadEFrac>0.95 && Jets[0].nHadEFrac>0.95",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.1 && Jets[0].cHadEFrac<0.1",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05 && Jets[2].cHadEFrac<0.2",
        
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2 && MEt.sign>20",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05  && MEt.sign>20",
        selection["METPreSelSUSYAOD"] + " && Jets[0].nSelectedTracks<2 && Jets[1].nSelectedTracks<2 && MEt.sign>30",
        selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05  && MEt.sign>30",
        #selection["METPreSelSUSYAOD"] + " && Jets[1].cHadEFrac<0.05 && Jets[0].cHadEFrac<0.05 && Jets[2].cHadEFrac<0.05",
        #"HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v",
            ],
        [
        "Pre-selections",
        #"+ MEt.sign>50",
        #"+ MEt.sign>100",
        #"Jets[0].cHadEFrac<0.1",
        #"Jets[0].cHadEFrac<0.05",
        #"Jets[1].cHadEFrac<0.1",
        #"Jets[1].cHadEFrac<0.05",
        #"j0 nHadEFrac>0.8",
        #"j0 nHadEFrac>0.9",
        #"j0 nHadEFrac>0.95",
        #"j1 nHadEFrac>0.8",
        #"j1 nHadEFrac>0.9",
        #"j1 nHadEFrac>0.95",
        #"j0 sel tracks<6",
        #"j0 sel tracks<5",
        #"j0 sel tracks<4",
        #"j0 sel tracks<3",
        #"j0 sel tracks<2",
        #combis
        #"j0,j1 n sel tracks<5",
        "j0,j1 n sel tracks<2",
        "j0,j1 nHadEFrac>0.9",
        #"j0,j1 nHadEFrac>0.95",
        "j0,j1 cHadEFrac<0.1",
        "j0,j1 cHadEFrac<0.05",
        "j0,j1<0.05, j2 cHadEFrac<0.2",
        #"j0,j1,j2 cHadEFrac<0.05",
        "j0,j1 n sel tracks<2 && MEt.sign>20",
        "j0,j1 cHadEFrac<0.05 && MEt.sign>20",
        "j0,j1 n sel tracks<2 && MEt.sign>30",
        "j0,j1 cHadEFrac<0.05 && MEt.sign>30",
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=150,#150,#
        additional_string = "_combinations", 
        alpha = 2,
        CL = 5,
        FOM = a
        )
exit()



###Trigger efficiency
'''
NTUPLEDIR   = "/nfs/dust/cms/group/cms-llp/v0_gen_studies_calo_AOD_HeavyHiggsSUSY/"
OUTPUTDIR   = "$CMSSW_BASE/src/Analyzer/LLP2018/macro/Efficiency/v0_gen_studies_calo_AOD_HeavyHiggsSUSY/"
from NNInferenceCMSSW.LLP_NN_Inference.samplesAOD2018 import sample, samples
fom = ["signaleff"]
back = []
for a in fom:

    calc_punzi_FOM_vs_ctau(
        [
        #'HLT_HT430_DisplacedDijet40_DisplacedTrack_v',
	#'HLT_HT430_DisplacedDijet60_DisplacedTrack_v',
	#'HLT_HT500_DisplacedDijet40_DisplacedTrack_v',
	'HLT_HT650_DisplacedDijet60_Inclusive_v',
	'HLT_AK8PFHT800_TrimMass50_v',
	#'HLT_AK8PFHT850_TrimMass50_v',
	#'HLT_AK8PFHT900_TrimMass50_v',
	'HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v',
	#'HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v',
	'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v',
	#'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v',
	'HLT_AK8PFJet360_TrimMass30_v',
	#'HLT_AK8PFJet380_TrimMass30_v',
	#'HLT_AK8PFJet400_TrimMass30_v',
	#'HLT_AK8PFJet420_TrimMass30_v',
	'HLT_AK8PFJet500_v',
	#'HLT_AK8PFJet550_v',
	#'HLT_AK8PFJetFwd500_v',
	'HLT_CaloJet500_NoJetID_v',
	#'HLT_CaloJet550_NoJetID_v',
	#'HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v',
	#'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v',
	#'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v',
	#'HLT_PFHT400_FivePFJet_100_100_60_30_30_DoublePFBTagDeepCSV_4p5_v',
	'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v',
	'HLT_PFJet500_v',
	'HLT_Rsq0p35_v',
	'HLT_DiJet110_35_Mjj650_PFMET110_v',
	'HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1_v',
	#'HLT_TripleJet110_35_35_Mjj650_PFMET110_v',
	
        #"HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v",
        "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v",
            ],
        [
            
            ],
        header = "",#str(n_matches)+" VBF jets gen matched",
        mH=1000,
        mS=400,#150,#
        additional_string = "_HLT", 
        alpha = 2,
        CL = 5,
        FOM = a
        )

exit()
'''
