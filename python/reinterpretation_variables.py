variable = {}

var_template = {

    #Jets variables
    "Jets.isGenMatchedCaloCorrLLPAccept": {
      "title" : "AK4 jets gen matched",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : True,
    },

    "Jets[[N]].sigprob": {
      "title" : "AK4 jet [[N]] DNN score",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.sigprob": {
      "title" : "AK4 jets DNN score",
      "nbins" : 50,#20,#50,#40,
      "min" : 0.,#0.001,#170,
      "max" : 1.,#2170,
      "log" : True,
    },
    #Jets kinematics
    "Jets[[N]].pt": {
      "title" : "jet [[N]] p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "Jets.pt": {
      "title" : "jets p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 500,#1000,
      "log" : True,
    },

    "Jets.ptGen": {
      "title" : "jets gen parton p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 1000,#1000,
      "log" : True,
    },

    "Jets[[N]].eta": {
      "title" : "jet [[N]] #eta",
      "nbins" : 100,#50,
      "min" : -1.5,#-5.2,#-3,
      "max" : 1.5,#5.2,#3,
      "log" : True,
    },
    "Jets.eta": {
      "title" : "jets #eta",
      "nbins" : 50,#25,
      "min" : -1.5,#-1.4,#-5.2,#-3,
      "max" : 1.5,#1.4,#5.2,#3,
      "log" : True,
    },
    "Jets.phi": {
      "title" : "jets #varphi",
      "nbins" : 64,#40,
      "min" : -3.2,#0.,#2.4,#-3.2,#2.4,#-3.2,
      "max" : 3.2,#3.2,
      "log" : True,
    },
    "Jets[[N]].phi": {
      "title" : "jet [[N]] #varphi",
      "nbins" : 100,#60,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },
    #######################
    #EB rec hits
    "Jets.nRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits multiplicity",
        "nbins" : 41+60,#101-50,#21,#
        "min" : 0-0.5,
        "max" : 40.5+60,#102-0.5,#20.5,#
        "log" : True,
    },
    "Jets.energyRecHitsEB": {
        "title" : "energyRecHitsEB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 250,
        "log" : True,
    },
    "Jets.eFracRecHitsEB": {
        "title" : "eFracRecHitsEB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 1,
        "log" : True,
    },

    "Jets[[N]].eFracRecHitsEB": {
        "title" : "eFracRecHitsEB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 1,
        "log" : True,
    },
    "Jets.timeRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time (ns)",
        "nbins" : 100,
        "min" : -5,
        "max" : 5,
        "log" : True,
    },
    "Jets[[N]].timeRecHitsEB": {
        "title" : "AK4 jet[[N]]: ECAL barr. rec hits time (ns)",
        "nbins" : 100,#100
        "min" : -5,#-5
        "max" : 5,#5
        "log" : True,
    },
    "Jets.timeRMSRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time RMS (ns)",
        "nbins" : 50,
        "min" : 0,
        "max" : 15,
        "log" : True,
    },
    "Jets[[N]].timeRMSRecHitsEB": {
        "title" : "AK4 jet [[N]]: ECAL barr. rec hits time RMS (ns)",
        "nbins" : 50,
        "min" : 0,
        "max" : 15,
        "log" : True,
    },
    # MET
    "MEt.pt": {
      "title" : "E_{T}^{miss} (GeV)",
      "nbins" : 25,#50
      "min" : 0,#200,#0,#200,
      "max" : 1000,#1000,#1200,#2000,#0,#2200,
      "log" : True,
    },

    "MEt.phi": {
      "title" : "E_{T}^{miss} #varphi",
      "nbins" : 50,#50
      "min" : -5,#200,#0,#200,
      "max" : 5,#1000,#1200,#2000,#0,#2200,
      "log" : True,
    },
    # GenBquarks
    "GenBquarks[[N]].eta": {
      "title" : "gen b-quark[[N]] #eta",
      "nbins" : 50,#SUSY: 50,
      "min" : -1.,
      "max" : 1.,#SUSY: 250,
      "log" : True,
    },

    "GenBquarks[[N]].pt": {
      "title" : "gen b-quark[[N]] p_{T} (GeV)",
      "nbins" : 50,#SUSY: 50,
      "min" : 0,
      "max" : 1000,#SUSY: 250,
      "log" : True,
    },
    "GenBquarks[[N]].radius": {
      "title" : "gen b-quark 3D radius (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
      "log" : True,
      "logx" : True,
    },
    "GenBquarks[[N]].travelRadius": {
      "title" : "gen b-quark transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
      "log" : True,
      "logx" : True,
    },
    "GenLLPs[[N]].travelRadius": {
      "title" : "gen LLP transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
        "bins" : [0.1,1,10,25,50,100,200,400,800],#SUSY
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 0,#20,#100,
      "min" : 0.000001,
      "max" : 200,
      "log" : True,
      "logx" : True,
    },



    "GenHiggs[[N]].travelRadiusLLP": {
      "title" : "gen LLP transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
        "bins" : [0.1,1,10,25,50,100,200,400,800],#SUSY
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 0,#20,#100,
      "min" : 0.000001,
      "max" : 200,
      "log" : True,
      "logx" : True,
    },

    "GenHiggs[[N]].travelZLLP": {
      "title" : "gen LLP z decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
        "bins" : [0.1,1,10,25,50,100,200,400,800],#SUSY
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 0,#20,#100,
      "min" : 0.000001,
      "max" : 200,
      "log" : True,
      "logx" : True,
    },


    "GenLLPs.travelRadius": {
      "title" : "gen LLP transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      "bins" : [0.1,1,10,25,50,100,200,400,800],#SUSY
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 0,#20,#100,
      "min" : 0.000001,
      "max" : 200,
      "log" : True,
      "logx" : True,
    },

    "GenLLPs.travelRadius/10": {
      "title" : "gen LLP transverse decay length (m)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000]      
      #SUSY ctau 500"
      #"bins" : [0,10,50,100,150,200,250,300,350,400,450,500,550,600,700,800,900,1000,2000],
      #SUSY ctau 3000"
      "bins" : [0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,1900.,2000,2125,2250,2500,3000,5000],
      #"bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      #"bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,400,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,1000000],#HeavyHiggs
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 100,
      "min" : 0,#.000001/10,
      "max" : 10,#00/10,
      "log" : True,
      "logx" : True,
    },

    "GenLLPs[[N]].travelRadius/10": {
      "title" : "gen LLP transverse decay length (m)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000]      
      #SUSY ctau 500"
      #"bins" : [0,10,50,100,150,200,250,300,350,400,450,500,550,600,700,800,900,1000,2000],
      #SUSY ctau 3000"
      "bins" : [0,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,1900.,2000,2125,2250,2500,3000,5000],
      #"bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      #"bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,400,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,1000000],#HeavyHiggs
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 50,
      "min" : 0,#.000001/10,
      "max" : 20,#00/10,
      "log" : True,
      "logx" : True,
    },


    "GenBquarks[[N]].radius2D": {
      "title" : "gen b-quark transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
      "log" : True,
      "logx" : True,
    },
    "GenLLPs[[N]].pt": {
      "title" : "gen long lived particle [[N]] p_{T} (GeV)",
      "nbins" : 10,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },

    "GenLLPs.pt": {
      "title" : "gen long lived particle p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },

    "GenLLPs.eta": {
      "title" : "gen long lived particle #eta",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },

    "GenLLPs[[N]].eta": {
      "title" : "gen long lived particle[[N]] #eta",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },

    "GenLLPs.rapidity": {
      "title" : "gen long lived particle rapidity",
      "nbins" : 50,
      "min" : -4,
      "max" : 4,
      "log" : True,
    },

    "GenLLPs[[N]].dRdaughters": {
      "title" : "gen #DeltaR daughters of long lived particle [N] (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 3.,
      "log" : True,
    },
    "GenHiggs[[N]].dRdaughters": {
      "title" : "gen #DeltaR daughters of Higgs [N] (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 3.,
      "log" : True,
    },
    "GenLLPs.dRdaughters": {
      "title" : "gen #DeltaR daughters of long lived particles (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },
    "GenHiggs.dRdaughters": {
      "title" : "gen #DeltaR daughters of Higgs-like particles (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },
    "GenHiggs.pt": {
      "title" : "gen Higgs-like p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },
    "GenHiggs[[N]].pt": {
      "title" : "gen Higgs[[N]] p_{T} (GeV)",
      "nbins" : 10,#20,#50,
      "min" : 0,
      "max" : 1000,#1000,
      "log" : True,
    },

    "GenHiggs[[N]].eta": {
      "title" : "gen Higgs[[N]] #eta",
      "nbins" : 50,
      "min" : -5,
      "max" : 5,
      "log" : True,
    },

    "GenHiggs[[N]].mass": {
      "title" : "gen Higgs mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 3000,
      "log" : True,
    },
    "GenHiggs[[N]].radius": {
      "title" : "gen Higgs radius (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
      "log" : True,
      "logx" : True,
    },    
    "GenHiggs[[N]].radius2D": {
      "title" : "gen Higgs transverse decay length (cm)",
      #"bins" : [0,0.0001,0.001,0.01,0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000],
      "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000],#HeavyHiggs
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
      "log" : True,
      "logx" : True,
    }, 
    
    "nTagJets_0p996": {
      "title" : "n. of jets with DNN output > 0.996, jet cleaning, |#eta|<1",
      "nbins" : 3,#7+4,#3,#
      "min" : -0.5,
      "max" : 2.5,#6.5+4,#2.5,#
      "log" : True,
    },

    #Gen-level
    "GenGravitinos[[N]].pt": {
      "title" : "gen gravitino[[N]] p_{T} (GeV)",
      "nbins" : 50,#SUSY: 50,
      "min" : 0,
      "max" : 5000,#SUSY: 250,
      "log" : True,
    },
    "GenGravitinos[[N]].eta": {
      "title" : "gen gravitino[[N]] #eta",
      "nbins" : 50,#SUSY: 50,
      "min" : -5.,
      "max" : 5.,#SUSY: 250,
      "log" : True,
    },

    "GenGravitinos[[N]].phi": {
      "title" : "gen gravitino[[N]] #varphi",
      "nbins" : 50,#SUSY: 50,
      "min" : -5.,
      "max" : 5.,#SUSY: 250,
      "log" : True,
    },


    "GenLLPs[[N]].phi": {
      "title" : "gen LLP[[N]] #varphi",
      "nbins" : 50,#SUSY: 50,
      "min" : -5.,
      "max" : 5.,#SUSY: 250,
      "log" : True,
    },


    "GenHiggs[[N]].phi": {
      "title" : "gen Higgs[[N]] #varphi",
      "nbins" : 50,#SUSY: 50,
      "min" : -5.,
      "max" : 5.,#SUSY: 250,
      "log" : True,
    },

    "GenBquarks[[N]].phi": {
      "title" : "gen b-quark[[N]] #varphi",
      "nbins" : 50,#SUSY: 50,
      "min" : -5.,
      "max" : 5.,#SUSY: 250,
      "log" : True,
    },

    "dR_Higgs": {
      "title" : "dR_Higgs",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },
    "dR_Gravitinos": {
      "title" : "dR_Gravitinos",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },


    "dR_LLPs": {
      "title" : "dR_LLPs",
      "nbins" : 0,#10,#25,
      "bins" : [2., 2.25, 2.5, 2.75, 3., 3.125, 3.25, 3.5, 3.75, 4., 4.5, 5.],
      "min" : 0,
      "max" : 5,
      "log" : True,
    },


    "dPhi_LLPs": {
      "title" : "dPhi_LLPs",
      "nbins" : 20,#0,#10,#25,
      "bins" : [2., 2.25, 2.5, 2.75, 3., 3.125, 3.25, 3.5, 3.75, 4., 4.5, 5.],
      "min" : -5,
      "max" : 5,
      "log" : True,
    },

    "dR_Gravitino_Higgs_0": {
      "title" : "dR_Gravitino_Higgs_0",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dR_Gravitino_Higgs_1": {
      "title" : "dR_Gravitino_Higgs_1",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dR_Gravitino_GenMet_0": {
      "title" : "dR_Gravitino_GenMet_0",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dR_Gravitino_GenMet_1": {
      "title" : "dR_Gravitino_GenMet_1",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dPhi_Gravitino_Met_0": {
      "title" : "dPhi_Gravitino_Met_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Gravitino_Met_1": {
      "title" : "dPhi_Gravitino_Met_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Gravitino_GenMet_0": {
      "title" : "dPhi_Gravitino_GenMet_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Gravitino_GenMet_1": {
      "title" : "dPhi_Gravitino_GenMet_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dR_LLP_GenMet_0": {
      "title" : "dR_LLP_GenMet_0",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dR_LLP_GenMet_1": {
      "title" : "dR_LLP_GenMet_1",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dPhi_LLP_Met_0": {
      "title" : "dPhi_LLP_Met_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_LLP_Met_1": {
      "title" : "dPhi_LLP_Met_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_LLP_GenMet_0": {
      "title" : "dPhi_LLP_GenMet_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_LLP_GenMet_1": {
      "title" : "dPhi_LLP_GenMet_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dR_Higgs_GenMet_0": {
      "title" : "dR_Higgs_GenMet_0",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dR_Higgs_GenMet_1": {
      "title" : "dR_Higgs_GenMet_1",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dPhi_Higgs_Met_0": {
      "title" : "dPhi_Higgs_Met_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Higgs_Met_1": {
      "title" : "dPhi_Higgs_Met_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Higgs_GenMet_0": {
      "title" : "dPhi_Higgs_GenMet_0",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_Higgs_GenMet_1": {
      "title" : "dPhi_Higgs_GenMet_1",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "DiGravitino_pt": {
      "title" : "DiGravitino p_{T}",
      "nbins" : 50,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },

    "DiGravitino_mass": {
      "title" : "DiGravitino_mass",
      "nbins" : 50,
      "min" : 0,
      "max" : 20,
      "log" : True,
    },

    "DiGravitino_eta": {
      "title" : "DiGravitino_eta",
      "nbins" : 50,
      "min" : -5.,
      "max" : 5.,
      "log" : True,
    },

    "DiGravitino_phi": {
      "title" : "DiGravitino_phi",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dR_DiGravitino_GenMet": {
      "title" : "dR_DiGravitino_GenMet",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "dPhi_DiGravitino_Met": {
      "title" : "dPhi_DiGravitino_Met",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "dPhi_DiGravitino_GenMet": {
      "title" : "dPhi_DiGravitino_GenMet",
      "nbins" : 50,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },

    "inv_mass_high_DNN": {
      "title" : "inv_mass_high_DNN",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },

    "inv_mass_tag_DNN": {
      "title" : "inv mass tagged jets with highest DNN",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },

    "perc_met_held_by_gravitinos": {
      "title" : "DiGravitino p_{T}/reco MET",
      "nbins" : 50,
      "min" : 0,
      "max" : 5.,
      "log" : True,
    },


    "nTagJets_to_LLP0": {
      "title" : "nTagJets_to_LLP0",
      "nbins" : 5,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "nTagJets_to_LLP1": {
      "title" : "nTagJets_to_LLP1",
      "nbins" : 5,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "nJets_to_LLP0": {
      "title" : "nJets_to_LLP0",
      "nbins" : 5,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "nJets_to_LLP1": {
      "title" : "nJets_to_LLP1",
      "nbins" : 5,
      "min" : 0,
      "max" : 5,
      "log" : True,
    },

    "MinJetMetDPhi": {
       "title" : "min #Delta #varphi (AK4 jet, MET)",
      "nbins" : 32,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },


}





for n, v in var_template.iteritems():
    if '[N]' in n:
        for i in range(0, 9):
            ni = n.replace('[N]', "%d" % i)
            variable[ni] = v.copy()
            variable[ni]['title'] = variable[ni]['title'].replace('[N]', "%d" % i)
    else:
        variable[n] = v

# Custom settings
#variable['CHSJets.CHSJets[2].pt']['max'] = 500
#variable['CHSJets.CHSJets[3].pt']['max'] = 400
#variable['CHSJet2.pt']['max'] = 400
#variable['CHSJet3.pt']['max'] = 200
#variable['CHSJet4.pt']['max'] = 200
