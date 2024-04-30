#!/usr/bin/env python
import os

channels = ["SHH","SYY"]
channels = ["SYY"]

for ch in channels:
    print "python macro/tag_efficiency_stealth.py  -c "+ch+" \n"
    #os.system("python macro/tag_efficiency_stealth.py -r 2018 -t "+str(mstop)+" -s "+str(ms)+" -c "+ch+" \n")

#python macro/tag_efficiency_stealth.py -r 2018 -t 1100 -s 100 -c SHH
