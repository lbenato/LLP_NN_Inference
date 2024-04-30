#!/usr/bin/env python
import os

channels = ["SHH","SYY"]
l_mstop = [300,500,700,900,1100,1300,1500,]
for ch in channels:
    for mstop in l_mstop:
        l_ms = [100,mstop-225]
        for ms in l_ms:
            l_tmp_stop = []
            if mstop==300 and ms==75:
                continue
            print "python macro/tag_efficiency_stealth.py -r 2016 -e _G-H -t "+str(mstop)+" -s "+str(ms)+" -c "+ch+" \n"
            os.system("python macro/tag_efficiency_stealth.py -r 2016 -e _G-H -t "+str(mstop)+" -s "+str(ms)+" -c "+ch+" \n")

#python macro/tag_efficiency_stealth.py -r 2018 -t 1100 -s 100 -c SHH
