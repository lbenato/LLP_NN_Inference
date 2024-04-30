#!/usr/bin/env python
import os
import ROOT as ROOT

ERA = "2016"
DIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_time_smeared/"#Gen
DIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_Gen_time_smeared_no_cuts_debug/"#Gen
DIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_SR_time_smeared_uncorrelated/"#Gen
DIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_no_cuts_syst_unc_central_values/"
DIR = "/nfs/dust/cms/group/cms-llp/v6_calo_AOD/v6_calo_AOD_%s_PDF_QCD_syst_unc/"

def rename(era,inpdir):
    DATADIR = inpdir % (era)
    print DATADIR
    #change files
    root_files = [x for x in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, x))]
    for r in root_files:
        if "_HH.root" not in r and "_HZ.root" not in r and "_ZZ.root" not in r:
            os.system("echo mv "+DATADIR+r+" "+DATADIR+r.replace(".root","_HH.root"))
            os.system("mv "+DATADIR+r+" "+DATADIR+r.replace(".root","_HH.root"))

    #dir_files = [x for x in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, x))]
    #for r in dir_files:
    #    if "_ctau500" in r:
    #        if "00_HH" not in r and "00_HZ" not in r and "00_ZZ" not in r:
    #            os.system("echo rename "+DATADIR+r+" "+DATADIR+r.replace("_ctau500","_ctau500_HH"))
    #            os.system("mv "+DATADIR+r+" "+DATADIR+r.replace("_ctau500","_ctau500_HH"))
    #    if "_ctau3000" in r:
    #        if "00_HH" not in r and "00_HZ" not in r and "00_ZZ" not in r:
    #            os.system("echo rename "+DATADIR+r+" "+DATADIR+r.replace("_ctau3000","_ctau3000_HH"))
    #            os.system("mv "+DATADIR+r+" "+DATADIR+r.replace("_ctau3000","_ctau3000_HH"))

    print "\n"

eras = ["2016","2017","2018"]
#eras = ["2017"]
for e in eras:
    rename(e,DIR)

