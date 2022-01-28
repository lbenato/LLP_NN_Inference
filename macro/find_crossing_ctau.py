#! /usr/bin/env python
import math
from ROOT import *
import numpy as np

gStyle.SetOptStat(0)
gROOT.SetBatch(True)

#N(m) = N0 e - L(m)/ctau(m)
f3 = TF1("f3","exp(-x/3.)/(3.*3.)",0,10)#2.1,2.2)
f05 = TF1("f05","exp(-x/0.5)/(0.5*0.5)",0,10)#2.1,2.2)

can = TCanvas("can","can",900,800)
can.SetRightMargin(0.05)
can.cd()
can.SetGrid()
can.SetLogy()
#can.SetLogx()
f3.SetLineWidth(2)
f3.SetLineColor(1)
f05.SetLineWidth(2)
f05.SetLineColor(2)
f3.SetTitle("")
f3.GetYaxis().SetTitle("a.u.")
f3.GetXaxis().SetTitle("decay length (m)")
f3.Draw("L")
f05.Draw("L,sames")
leg = TLegend(0.6, 0.7, 0.9, 0.9)
leg.SetTextSize(0.035)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetFillColor(0)
leg.AddEntry(f05,"0.5 m","L")
leg.AddEntry(f3,"3 m","L")
leg.Draw()
can.Print("exp_decay.pdf")
can.Print("exp_decay.png")
