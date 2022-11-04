root [11] old_ctau = 50
root [12] new_ctau = 100
root [13] llp_0 = 106.59450
root [14] llp_1 = 117.95681
root [15] llp_tot = llp_0 + llp_1
root [16] source0 = exp(-1.0*llp_0/old_ctau)/pow(old_ctau,2)
root [17] source1 = exp(-1.0*llp_1/old_ctau)/pow(old_ctau,2)
root [18] source_tot = exp(-1.0*llp_tot/old_ctau)/pow(old_ctau,2)
root [19] weight0 = 1.0/pow(new_ctau,2) * exp(-1.0*llp_0/new_ctau)/source0
root [20] weight1 = 1.0/pow(new_ctau,2) * exp(-1.0*llp_1/new_ctau)/source1
root [21] weight_tot = 1.0/pow(new_ctau,2) * exp(-1.0*llp_tot/new_ctau)/source_tot
root [22] weight0*weight1
