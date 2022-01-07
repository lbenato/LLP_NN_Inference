variable = {}

var_template = {

    "sigprob": {
      "title" : "DNN output score",
      "nbins" : 50,
      "min" : 0,
      "max" : 1.,
      "log" : True,
    },
    "EventWeight": {
      "title" : "event weight",
      "nbins" : 200,
      "min" : -10,
      "max" : 10,
      "log" : False,
    },
    "PUReWeight": {
      "title" : "PU weight",
      "nbins" : 200,
      "min" : -10,
      "max" : 10,
      "log" : False,
    },

    "EventNumber": {
      "title" : "event number",
      "nbins" : 10000000,
      "min" : 0,
      "max" : 1.e7,
      "log" : False,
    },
    "LumiNumber": {
      "title" : "lumisection number",
      "nbins" : 2000,
      "min" : 0,
      "max" : 2000,
      "log" : False,
    },
    "RunNumber": {
      "title" : "run number",
      "nbins" : 7000,
      "min" : 254000,
      "max" : 261000,
      "log" : False,
    },
    "nPV": {
      "title" : "number of reconstructed Primary Vertices",
      "nbins" : 50+40,
      "min" : -0.5,
      "max" : 49.5+40,
      "log" : False,
    },
    "MeanNumInteractions": {
      "title" : "mean number of interactions",
      "nbins" : 50+40,
      "min" : -0.5,
      "max" : 49.5+40,
      "log" : False,
    },

    "isVBF": {
      "title" : "isVBF",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : False,
    },
    "isMC": {
      "title" : "isMC",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : False,
    },
    # Overall PF Candidates
    "nPFCandidates": {
      "title" : "number of particle flow candidates",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 3999.5,
      "log" : True,
    },
    "nPFCandidatesTrack": {
      "title" : "number of charged particle flow candidates",
      "nbins" : 100-50,
      "min" : -0.5,
      "max" : 2999.5,
      "log" : True,
    },
    "nPFCandidatesHighPurityTrack": {
      "title" : "number of charged high purity particle flow candidates",
      "nbins" : 100,
      "min" : -0.5,
      "max" : 1999.5,
      "log" : True,
    },

    "nPFCandidatesFullTrackInfo": {
      "title" : "number of particle flow candidates with full track info",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 499.5,
      "log" : True,
    },
    # Number of jets
    "nJets": {
      "title" : "number of CHS jets",
      "nbins" : 50-20,
      "min" : -0.5,
      "max" : 49.5-20,
      "log" : True,
    },
    "nCHSJets": {
      "title" : "number of CHS jets",
      "nbins" : 50-20,
      "min" : -0.5,
      "max" : 49.5-20,
      "log" : True,
    },
    "nCHSJetsAcceptanceCalo": {
      "title" : "number of CHS jets in acceptance",
      "nbins" : 50-20-20,
      "min" : -0.5,
      "max" : 49.5-20-20,
      "log" : True,
    },
    "nMatchedCHSJets": {
      "title" : "number of CHS jets matched to at least 1 b quark",
      "nbins" : 5,
      "min" : -0.5,
      "max" : 4.5,
      "log" : True,
    },
    "number_of_b_matched_to_CHSJets": {
      "title" : "number of b quarks matched to at least 1 CHS jet",
      "nbins" : 5,
      "min" : -0.5,
      "max" : 4.5,
      "log" : True,
    },
    "nCaloJets": {
      "title" : "number of calo jets",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets70": {
      "title" : "number of calo jets 0.70 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets75": {
      "title" : "number of calo jets 0.75 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets80": {
      "title" : "number of calo jets 0.80 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets85": {
      "title" : "number of calo jets 0.85 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets90": {
      "title" : "number of calo jets 0.90 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "nCaloTagJets95": {
      "title" : "number of calo jets 0.95 pfXWP1000",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },

    "nCHSFatJets": {
      "title" : "number of AK8 jets",
      "nbins" : 5+5,
      "min" : 0.5,
      "max" : 5.5+5,
      "log" : True,
    },
    "nCHSFatJetsAcceptanceCalo": {
      "title" : "number of AK8 jets in acceptance",
      "nbins" : 5+5,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    # HT and MinDPhi
    "HT": {
      "title" : "H_{T} (GeV)",
      "nbins" : 25,#45,
      "min" : 0,#100,
      "max" : 1000,#2500,
      "log" : True,
    },
    "ggHJetMetDPhi": {
      "title" : "#Delta #varphi (ggH jet, MET)",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinJetMetDPhi_ntuple": {
       "title" : "min #Delta #varphi (AK4 jet, MET) at ntuple level",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinJetMetDPhi": {
       "title" : "min #Delta #varphi (AK4 jet, MET)",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinLeadingJetMetDPhi": {
      "title" : "min #Delta #varphi (AK4 leading jet, MET)",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinSubLeadingJetMetDPhi": {
      "title" : "min #Delta #varphi (AK4 sub-leading jet, MET)",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinJetMetDPhiBarrel": {
      "title" : "min #Delta #varphi (AK4 jet, MET) barrel",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },

    "MinFatJetMetDPhi": {
      "title" : "min #Delta #varphi (AK8 jet, MET)",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    "MinFatJetMetDPhiBarrel": {
      "title" : "min #Delta #varphi (AK8 jet, MET) barrel",
      "nbins" : 32,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },

    "MinJetMetDPhiAllJets": {
      "title" : "MinJetMetDPhi all jets up to |#eta|=5.2",
      "nbins" : 64,#10,
      "min" : 0,
      "max" : 3.14,
      "log" : True,
    },
    #Beam Halo
    "min_dR_jets": {
      "title" : "min_dR_jets",
      "nbins" : 50,#45,
      "min" : 0.4,#100,
      "max" : 4.2,#2500,
      "log" : True,
    },

    "min_dPhi_jets": {
      "title" : "min #Delta #varphi jets",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 10.,#,3.1415,#3.2,#2500,
      "log" : True,
    },

    "min_dEta_jets": {
      "title" : "min #Delta #eta jets",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1.,#,3.1415,#3.2,#2500,
      "log" : True,
    },
    "min_dPhi_jets_0p7": {
      "title" : "min #Delta #varphi jets with DNN>0.7",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1.,#,3.1415,#3.2,#2500,
      "log" : True,
    },
    "min_dPhi_jets_0p9": {
      "title" : "min #Delta #varphi jets with DNN>0.9",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 3.1415,#3.2,#2500,
      "log" : True,
    },
    "min_dPhi_jets_0p996": {
      "title" : "min_dPhi_jets_0p996",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1.,#,3.1415,#3.2,#2500,
      "log" : True,
    },
    "min_dPhi_jets_eta_1p0_0p9": {
      "title" : "min #Delta #varphi (jets with DNN > 0.9, |#eta|<1)",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1.,#,3.1415,#3.2,#2500,
      "log" : True,
    },
    "min_dPhi_jets_eta_1p0_0p996": {
      "title" : "min #Delta #varphi (jets with DNN > 0.996, |#eta|<1)",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1.,#,3.1415,#3.2,#2500,
      "log" : True,
    },


    #Jets with additional variables
    "JetsCaloAdd.nRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits multiplicity",
        "nbins" : 21,
        "min" : 0-0.5,
        "max" : 21-0.5,
        "log" : True,
    },
    "JetsCaloAdd.meanREB": {
        "title" : "mean radius of EB rec hits in jets",
        "nbins" : 50,
        "min" : 128,#-150,
        "max" : 131,#150,
        "log" : True,
    },
    "JetsCaloAdd.spreadREB": {
        "title" : "spread radius of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.5,#25,
        "log" : True,
    },
    "JetsCaloAdd.meanXEB": {
        "title" : "mean x of EB rec hits in jets",
        "nbins" : 50,
        "min" : -150,#120
        "max" : 150,#130
        "log" : True,
    },
    "JetsCaloAdd.spreadXEB": {
        "title" : "spread x of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 10,#25,
        "log" : True,
    },
    "JetsCaloAdd.meanYEB": {
        "title" : "mean y of EB rec hits in jets",
        "nbins" : 50,
        "min" : -150,
        "max" : 150,
        "log" : True,
    },
    "JetsCaloAdd.spreadYEB": {
        "title" : "spread y of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 25,#25,
        "log" : True,
    },
    "JetsCaloAdd.meanZEB": {
        "title" : "mean z of EB rec hits in jets",
        "nbins" : 50,
        "min" : -300,
        "max" : 300,
        "log" : True,
    },
    "JetsCaloAdd.spreadZEB": {
        "title" : "spread z of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 50.,
        "log" : True,
    },
    "JetsCaloAdd.spreadXEB/JetsCaloAdd.spreadZEB": {
        "title" : "spread x/spread z of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1,
        "log" : True,
    },

    "JetsCaloAdd.meanPhiEB": {
        "title" : "mean #varphi of EB rec hits in jets",
        "nbins" : 50,
        "min" : -3.2,
        "max" : 3.2,
        "log" : True,
    },
    "JetsCaloAdd.spreadPhiEB": {
        "title" : "spread #varphi of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.3,
        "log" : True,
    },
    "JetsCaloAdd.meanEtaEB": {
        "title" : "mean #eta of EB rec hits in jets",
        "nbins" : 50,
        "min" : -1.5,
        "max" : 1.5,
        "log" : True,
    },
    "JetsCaloAdd.spreadEtaEB": {
        "title" : "spread #eta of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.5,
        "log" : True,
    },
    "JetsCaloAdd.meanATLASEB": {
        "title" : "mean ATLAS var. of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.5,
        "log" : True,
    },
    "JetsCaloAdd.spreadATLASEB": {
        "title" : "spread ATLAs var. of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.2,
        "log" : True,
    },
    "JetsCaloAdd.eFracEB2CSC": {
        "title" : "EB energy fraction associated to CSC (#Delta #varphi < 0.4)",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1.,
        "log" : True,
    },
    "JetsCaloAdd.eFracEB2CSC0p1": {
        "title" : "EB energy fraction associated to CSC (#Delta #varphi < 0.1)",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1.,
        "log" : True,
    },
    "JetsCaloAdd.eFracEB2CSC0p04": {
        "title" : "EB energy fraction associated to CSC (#Delta #varphi < 0.04)",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1.,
        "log" : True,
    },

    "JetsCaloAdd[[N]].eFracEB2CSC0p1": {
        "title" : "EB energy fraction associated to CSC (#Delta #varphi < 0.1)",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1.,
        "log" : True,
    },
    "JetsCaloAdd[[N]].eFracEB2CSC0p04": {
        "title" : "EB energy fraction associated to CSC (#Delta #varphi < 0.04)",
        "nbins" : 50,
        "min" : 0.,
        "max" : 1.,
        "log" : True,
    },

    #Jets variables
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
    "Jets[[N]].ptJER": {
      "title" : "jet [[N]] p_{T} JER smear (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },
    "Jets.ptJER": {
      "title" : "jets p_{T} JER smear (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 3000,
      "log" : True,
    },
    "Jets.energy": {
      "title" : "jets energy (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 500,#3000,
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
    "Jets[[N]].mass": {
      "title" : "jet [[N]] mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 100,
      "log" : True,
    },
    "Jets.CSV": {
      "title" : "jets CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "JetsNegative.sigprob": {
      "title" : "AK4 jets DNN score",
      "nbins" : 50,#20,#50,#40,
      "min" : 0.,#0.001,#170,
      "max" : 1.,#2170,
      "log" : True,
    },
    "JetsNegative.CSV": {
      "title" : "jets CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "JetsNegative[[N]].CSV": {
      "title" : "jet [N] CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].CSV": {
      "title" : "jet [[N]] CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].alphaMax": {
      "title" : "jet [[N]] #alpha_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "Jets.alphaMax": {
      "title" : "jets #alpha_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "Jets.betaMax": {
      "title" : "jet  #beta_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.5,#3.,
      "log" : True,
    },
    "Jets[[N]].betaMax": {
      "title" : "jet[[N]]  #beta_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1,#0.5,#3.,
      "log" : True,
    },
    "Jets.gammaMax": {
      "title" : "jet  #gamma_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.2,#3.,
      "log" : True,
    },
    "Jets[[N]].gammaMax": {
      "title" : "jet[[N]]  #gamma_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1,#0.2,#3.,
      "log" : True,
    },
    "Jets.gammaMaxEM": {
      "title" : "jet  #gamma_{max} EM",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.5,#3.,
      "log" : True,
    },
    "Jets[[N]].gammaMaxEM": {
      "title" : "jet[[N]]  #gamma_{max} EM",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1,#0.5,#3.,
      "log" : True,
    },
    "Jets.gammaMaxHadronic": {
      "title" : "jet  #gamma_{max} hadronic",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.5,#3.,
      "log" : True,
    },
    "Jets[[N]].gammaMaxHadronic": {
      "title" : "jet[[N]]  #gamma_{max} hadronic",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1,#0.5,#3.,
      "log" : True,
    },
    "Jets.gammaMaxET": {
      "title" : "jet  gammaMaxET",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.2,#2.,
      "log" : True,
    },       
    "Jets[[N]].gammaMaxET": {
      "title" : "jet[[N]]  gammaMaxET",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1,#0.2,#2.,
      "log" : True,
    },       
    "Jets.minDeltaRPVTracks": {
      "title" : "jet  minDeltaRPVTracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "Jets[[N]].minDeltaRPVTracks": {
      "title" : "jet[[N]]  minDeltaRPVTracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "Jets.minDeltaRAllTracks": {
      "title" : "jet  minDeltaRAllTracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.5,#1.,
      "log" : True,
    },
    "Jets[[N]].minDeltaRAllTracks": {
      "title" : "jet[[N]]  minDeltaRAllTracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 0.5,#1.,
      "log" : True,
    },
    "Jets.ptAllPVTracks": {
      "title" : "tracks p_{T} in PVs in AK4 jet ",
      "nbins" : 50,
      "min" : 0.,
      "max" : 50.,#500.,
      "log" : True,
    },
    "Jets[[N]].ptAllPVTracks": {
      "title" : "tracks p_{T} in PVs in AK4 jet[[N]] ",
      "nbins" : 50,
      "min" : 0.,
      "max" : 50.,#500.,
      "log" : True,
    },
    "Jets.ptAllTracks": {
      "title" : "all tracks p_{T} in AK4 jet ",
      "nbins" : 50,
      "min" : 0.,
      "max" : 200.,#500.,
      "log" : True,
    },
    "Jets[[N]].ptAllTracks": {
      "title" : "all tracks p_{T} in AK4 jet[[N]] ",
      "nbins" : 50,
      "min" : 0.,
      "max" : 200.,#500.,
      "log" : True,
    },
    
    "Jets.nPixelHitsMedian": {
      "title" : "jets nPixelHitsMedian",
      "nbins" : 15,
      "min" : 0.-0.5,
      "max" : 15.-0.5,
      "log" : True,
    },
    
    "Jets.nHitsMedian": {
      "title" : "jets nHitsMedian",
      "nbins" : 50,
      "min" : 0.-0.5,
      "max" : 50.-0.5,
      "log" : True,
    },
    
    "Jets.dxyMedian": {
      "title" : "jets dxyMedian",
      "nbins" : 50,
      "min" : -20.,
      "max" : 20.,
      "log" : True,
    },
    
    "Jets.dzMedian": {
      "title" : "jets dzMedian",
      "nbins" : 50,
      "min" : -40.,
      "max" : 40.,
      "log" : True,
    },
      
    "Jets.medianIP2D": {
      "title" : "jets medianIP2D",
      "nbins" : 50,
      "min" : -10000,
      "max" : 10000,
      "log" : True,
    },

    "Jets[[N]].sigIP2DMedian": {
      "title" : "jet [[N]] sigIP2DMedian",
      "nbins" : 50,
      "min" : -10000,#-50,
      "max" : 10000,#50,
      "log" : True,
    },
    "-log(abs(Jets[[N]].sigIP2DMedian))": {
      "title" : "jet [[N]] sigIP2DMedian",
      "nbins" : 100,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },
    "-log10(abs(Jets[[N]].sigIP2DMedian))": {
      "title" : "jet [[N]] sigIP2DMedian",
      "nbins" : 100,
      "min" : -5,
      "max" : 5,
      "log" : True,
    },
    "Jets[[N]].theta2DMedian": {
      "title" : "jet [[N]] theta2DMedian",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].flavour": {
      "title" : "jet [[N]] flavour",
      "nbins" : 25,
      "min" : -0.5,
      "max" : 24.5,
      "log" : False,
    },
    "Jets[[N]].FracCal": {
      "title" : "jet [[N]] ECAL energy/HCAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 10,
      "log" : True,
    },
    "Jets.FracCal": {
      "title" : "jets ECAL energy/HCAL energy",
      "nbins" : 50,#50
      "min" : 0,
      "max" : 2,#10
      "log" : True,
    },
    "Jets.hcalE": {
      "title" : "jets HCAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,#500,#2500
      "log" : True,
    },
    "Jets.ecalE": {
      "title" : "jets ECAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 100,#200,#2500
      "log" : True,
    },
    "(-Jets.ecalE + Jets.hcalE)/(Jets.hcalE + Jets.ecalE)": {
      "title" : "jets HCAL/ECAL imbalance",
      "nbins" : 50,
      "min" : -2,
      "max" : 2,#200,#2500
      "log" : True,
    },

    "Jets[[N]].hcalE": {
      "title" : "jet [[N]] HCAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 2500,
      "log" : True,
    },
    "Jets[[N]].ecalE": {
      "title" : "jet [[N]] ECAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 2500,
      "log" : True,
    },
    "Jets[[N]].ecalE/Jets[[N]].energyRaw": {
      "title" : "jet [[N]] ECAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].hcalE/Jets[[N]].energyRaw": {
      "title" : "jet [[N]] HCAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.ecalE/Jets.energyRaw": {
      "title" : "jets ECAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.hcalE/Jets.energyRaw": {
      "title" : "jets HCAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },

    "Jets.ecalE/Jets.energy": {
      "title" : "jets ECAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.hcalE/Jets.energy": {
      "title" : "jets HCAL energy fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #Jet constituents
    "Jets[[N]].nConstituents": {
      "title" : "jet [[N]] number of jet constituents",
      "nbins" : 100,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },
    "Jets.nConstituents": {
      "title" : "number of constituents per jet",
      "nbins" : 100,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },
    "Jets[[N]].nTrackConstituents": {
      "title" : "jet[[N]]  number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-20+20-20,
      "min" : -0.5,
      "max" : 49.5-20+20-20,
      "log" : True,
    },
    "Jets.nSelectedTracks": {
      "title" : "jet  number of selected tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-30-5,
      "min" : -0.5,
      "max" : 49.5-30-5,
      "log" : True,
    },
    "Jets[[N]].nSelectedTracks": {
      "title" : "jet[[N]]  number of selected tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-30-5,
      "min" : -0.5,
      "max" : 49.5-30-5,
      "log" : True,
    },
    "Jets.nTrackConstituents": {
      "title" : "number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-20+20,
      "min" : -0.5,
      "max" : 49.5-20+20,
      "log" : True,
    },
    "(Jets[[N]].nTrackConstituents)/(Jets[[N]].nConstituents)": {
      "title" : "percentage of jet [[N]] constituents with tracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    #Jet energies
    "Jets[[N]].cHadE": {
      "title" : "jet [[N]] charged hadron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    "Jets[[N]].nHadE": {
      "title" : "jet [[N]] neutral hadron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    "Jets[[N]].muE": {
      "title" : "jet [[N]] #mu energy",
      "nbins" : 20,
      "min" : 0,
      "max" : 20,
      "log" : True,
    },
    "Jets[[N]].eleE": {
      "title" : "jet [[N]] electron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "Jets[[N]].photonE": {
      "title" : "jet [[N]] photon energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    #Jet energy fractions
    "Jets[[N]].cHadEFrac": {
      "title" : "jet [[N]] charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.cHadEFrac": {
      "title" : "jets charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].nHadEFrac": {
      "title" : "jet [[N]] neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.nHadEFrac": {
      "title" : "jets neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.muEFrac": {
      "title" : "jets #mu energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets[[N]].muEFrac": {
      "title" : "jet [[N]] #mu energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.eleEFrac": {
      "title" : "jets electron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,#0.1
      "log" : True,
    },
    "Jets[[N]].eleEFrac": {
      "title" : "jet [[N]] electron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,#0.1
      "log" : True,
    },
    "Jets[[N]].photonEFrac": {
      "title" : "jet [[N]] photon energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets.photonEFrac": {
      "title" : "jets photon energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #Jet multiplicities
    "Jets[[N]].eleMulti": {
      "title" : "jet [[N]] electron multiplicity",
      "nbins" : 20,
      "min" : 0-0.5,
      "max" : 20-0.5,
      "log" : True,
    },
    "Jets[[N]].muMulti": {
      "title" : "jet [[N]] #mu multiplicity",
      "nbins" : 20,
      "min" : 0-0.5,
      "max" : 20-0.5,
      "log" : True,
    },
    "Jets[[N]].photonMulti": {
      "title" : "jet [[N]] photon multiplicity",
      "nbins" : 50,
      "min" : 0-0.5,
      "max" : 50-0.5,
      "log" : True,
    },
    "Jets[[N]].cHadMulti": {
      "title" : "jet [[N]] charged hadron multiplicity",
      "nbins" : 50,
      "min" : 0-0.5,
      "max" : 50-0.5,
      "log" : True,
    },
    "Jets[[N]].nHadMulti": {
      "title" : "jet [[N]] neutral hadron multiplicity",
      "nbins" : 20,
      "min" : 0-0.5,
      "max" : 20-0.5,
      "log" : True,
    },
    "Jets[[N]].cMulti": {
      "title" : "jet [[N]] charged multiplicity",
      "nbins" : 50,
      "min" : 0-0.5,
      "max" : 50-0.5,
      "log" : True,
    },
    "Jets[[N]].nMulti": {
      "title" : "jet [[N]] neutral multiplicity",
      "nbins" : 50,
      "min" : 0-0.5,
      "max" : 50-0.5,
      "log" : True,
    },
    "Jets[[N]].npr": {
      "title" : "jet [[N]] number of components",
      "nbins" : 100,
      "min" : 0-0.5,
      "max" : 100-0.5,
      "log" : True,
    },
    #Jet multiplicity fractions
    "Jets[[N]].eleMultiFrac": {
      "title" : "jet [[N]] electron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].muMultiFrac": {
      "title" : "jet [[N]] #mu multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].photonMultiFrac": {
      "title" : "jet [[N]] photon multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].cHadMultiFrac": {
      "title" : "jet [[N]] charged hadron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].nHadMultiFrac": {
      "title" : "jet [[N]] neutral hadron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].cMultiFrac": {
      "title" : "jet [[N]] charged multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].nMultiFrac": {
      "title" : "jet [[N]] neutral multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    #Imperial College tagger
    "Jets[[N]].pfXWP0p01": {
      "title" : "jet [[N]] pfXWP0p01",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].pfXWP1": {
      "title" : "jet [[N]] pfXWP1",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].pfXWP100": {
      "title" : "jet [[N]] pfXWP100",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "Jets[[N]].pfXWP1000": {
      "title" : "jet [[N]] pfXWP1000",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    #Jet substructure
    "Jets[[N]].tau1": {
      "title" : "jet [[N]] #tau_{1}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : False,
    },
    "Jets[[N]].tau2": {
      "title" : "jet [[N]] #tau_{2}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : False,
    },
    "Jets[[N]].tau21": {
      "title" : "jet [[N]] #tau_{2}/#tau_{1}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : False,
    },
    #Jets tracks hits
    "Jets[[N]].nPixelHitsMedian": {
      "title" : "jet [[N]] median number of pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "Jets[[N]].nHitsMedian": {
      "title" : "jet [[N]] median number of tracker hits",
      "nbins" : 20+5,
      "min" : -0.5,
      "max" : 19.5+5,
      "log" : True,
    },
    "Jets[[N]].nTracks0PixelHits": {
      "title" : "jet [[N]] number of tracks with 0 pixel hits",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "Jets[[N]].nTracks1PixelHit": {
      "title" : "jet [[N]] number of tracks with 1 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracks2PixelHits": {
      "title" : "jet [[N]] number of tracks with 2 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracks3PixelHits": {
      "title" : "jet [[N]] number of tracks with 3 pixel hits",
      #"title" : "Leading jet: n. of tracks with 3 pixel hits",
      "nbins" : 40-10,
      "min" : -0.5,
      "max" : 39.5-10,
      "log" : True,
    },
    "Jets[[N]].nTracks4PixelHits": {
      "title" : "jet [[N]] number of tracks with 4 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracks5PixelHits": {
      "title" : "jet [[N]] number of tracks with 5 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "Jets[[N]].nTracks6PixelHits": {
      "title" : "jet [[N]] number of tracks with 6 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "Jets[[N]].nTracksLarger5PixelHits": {
      "title" : "jet [[N]] number of tracks with more than 5 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "Jets[[N]].nTracksLarger6PixelHits": {
      "title" : "jet [[N]] number of tracks with more than 6 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "Jets[[N]].nTracks0LostInnerHits": {
      "title" : "jet [[N]] number of tracks with 0 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracks1LostInnerHit": {
      "title" : "jet [[N]] number of tracks with 1 lost inner hit",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracks2LostInnerHits": {
      "title" : "jet [[N]] number of tracks with 2 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "Jets[[N]].nTracksLarger2LostInnerHits": {
      "title" : "jet [[N]] number of tracks with more than 2 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },

    #Calo jets
    "CaloJets[[N]].pt": {
      "title" : "calo jet [[N]] p_{T} (GeV)",
      "nbins" : 40,#40
      "min" : 0,
      "max" : 400,
      "log" : True,
    },
    "CaloJets[[N]].emEnergyFraction": {
      "title" : "calo jet [[N]] ECAL energy fraction",
      "nbins" : 40,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "CaloJets[[N]].energyFractionHadronic": {
      "title" : "calo jet [[N]] HCAL energy fraction",
      "nbins" : 40,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },

    #ggH jet
    "ggHJet[[N]].pt": {
      "title" : "ggH jet p_{T} (GeV)",
      "nbins" : 40,#40
      "min" : 0,
      "max" : 400,
      "log" : True,
    },
    "ggHJet[[N]].nTrackConstituents": {
      "title" : "ggH jet number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-20,
      "min" : -0.5,
      "max" : 49.5-20,#+50,
      "log" : True,
    },
    "ggHJet[[N]].cMulti": {
      "title" : "ggH jet number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50-20,
      "min" : -0.5,
      "max" : 49.5-20,#+50,
      "log" : True,
    },
    "ggHJet[[N]].nHadEFrac": {
      "title" : "ggH jet neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #######################
    #Higgs masses
    "HDiCHS": {
      "title" : "dijet mass CHS (GeV)",
      "nbins" : 20,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HTriCHS": {
      "title" : "trijet mass CHS (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HQuadCHS": {
      "title" : "quadjet mass CHS (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HDiCHSMatched": {
      "title" : "Matched dijet mass CHS (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HTriCHSMatched": {
      "title" : "Matched trijet mass CHS (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HQuadCHSMatched": {
      "title" : "Matched quadjet mass CHS (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },

    "HDiCalo": {
      "title" : "dijet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HTriCalo": {
      "title" : "trijet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HQuadCalo": {
      "title" : "quadjet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HDiCaloMatched": {
      "title" : "Matched dijet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HTriCaloMatched": {
      "title" : "Matched trijet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },
    "HQuadCaloMatched": {
      "title" : "Matched quadjet mass Calo (GeV)",
      "nbins" : 100,
      "min" : 0.5,
      "max" : 500.5,
      "log" : True,
    },

    #######################
    # Trigger variables
    "VBFPairJets.VBFPairJets[[N]].pt": {
      "title" : "VBF pair jet [[N]] p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 800,
      "log" : True,
    },
    "VBFPairJets.VBFPairJets[[N]].eta": {
      "title" : "VBF pair jet [[N]] #eta",
      "nbins" : 40,
      "min" : -5.2,
      "max" : 5.2,
      "log" : True,
    },
    "DisplacedJets.DisplacedJets[[N]].nHadEFrac": {
      "title" : "displaced jet [[N]] neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "DisplacedJets.DisplacedJets[[N]].pt": {
      "title" : "displaced jet [[N]] p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 800,
      "log" : True,
    },
    "DisplacedJets.DisplacedJets[[N]].nTrackConstituents": {
      "title" : "Number of constituents with tracks per displaced jet [[N]]",
      "nbins" : 50,
      "min" : 0,
      "max" : 49,
      "log" : False,
    },

    # b-tagging variables
    "Jets.nSV": {
        "title" : "Number of SV per jet",
        "nbins" : 10,
        "min" : -0.5,
        "max" : 9.5,
        "log" : True,
    },
    "Jets.nSVCand": {
        "title" : "Number of SV candidates per jet [[N]]",
        "nbins" : 10,
        "min" : 0,
        "max" : 9,
        "log" : True,
    },
    "Jets.nVertexTracks": {
        "title" : "Number of tracks per vertex",
        "nbins" : 21,
        "min" : 0,"max" : 20,
        "log" : True,
    },
    "Jets.nSelectedTracks": {
        "title" : "Number of selected tracks per jet [[N]]",
        "nbins" : 20,
        "min" : 0,
        "max" : 20,
        "log" : True,
    },
    "Jets.nSelectedTracks": {
        "title" : "Number of selected tracks per jets",
        "nbins" : 20,
        "min" : 0-0.5,
        "max" : 20-0.5,
        "log" : True,
    },
    "Jets.dRSVJet": {
        "title" : "dR between SV and jet",
        "nbins" : 51,
        "min" : 0,
        "max" : 0.4,
        "log" : True,
    },
    "Jets.flightDist2d": {
        "title" : "Flight distance vertex [[N]] 2D (cm)",
        "nbins" : 50,
        "min" : 0,
        "max" : 10,
        "log" : True,
    },
    "Jets.flightDist3d": {
        "title" : "Flight distance vertex [[N]] 3D (cm)",
        "nbins" : 50,
        "min" : 0,
        "max" : 10,
        "log" : True,
    },
    "Jets.nTracksSV": {
        "title" : "Number of selected tracks per vertex [[N]]",
        "nbins" : 21,
        "min" : 0,
        "max" : 20,
        "log" : True,
    },
    "Jets.SV_mass": {
        "title" : "Mass of vertex [[N]] (GeV)",
        "nbins" : 31,
        "min" : 0,
        "max" : 30,
        "log" : True,
    },

    "Jets.dPhi_met": {
        "title" : "jet-met #Delta #varphi (per jet)",
        "nbins" : 50,
        "min" : -3.14,
        "max" : 3.14,
        "log" : True,
    },
    #EB rec hits
    "Jets.nRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits multiplicity",
        "nbins" : 41,#101-50,#21,#
        "min" : 0-0.5,
        "max" : 40.5,#102-0.5,#20.5,#
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

    "JetsNegative.pt": {
      "title" : "jets p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 500,#1000,
      "log" : True,
    },
    "JetsNegative.eta": {
      "title" : "jets #eta",
      "nbins" : 50,#25,
      "min" : -1.5,#-1.4,#-5.2,#-3,
      "max" : 1.5,#1.4,#5.2,#3,
      "log" : True,
    },
    "JetsNegative.phi": {
      "title" : "jets #varphi",
      "nbins" : 64,#40,
      "min" : -3.2,#0.,#2.4,#-3.2,#2.4,#-3.2,
      "max" : 3.2,#3.2,
      "log" : True,
    },

    "JetsNegative[[N]].pt": {
      "title" : "jet [N] p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 500,#1000,
      "log" : True,
    },
    "JetsNegative[[N]].eta": {
      "title" : "jet [N] #eta",
      "nbins" : 50,#25,
      "min" : -1.5,#-1.4,#-5.2,#-3,
      "max" : 1.5,#1.4,#5.2,#3,
      "log" : True,
    },
    "JetsNegative[[N]].phi": {
      "title" : "jet [N] #varphi",
      "nbins" : 64,#40,
      "min" : -3.2,#0.,#2.4,#-3.2,#2.4,#-3.2,
      "max" : 3.2,#3.2,
      "log" : True,
    },

    "JetsNegative.nRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits multiplicity",
        "nbins" : 101-50,
        "min" : 0-0.5,
        "max" : 102-0.5,
        "log" : True,
    },
    "JetsNegative[[N]].nRecHitsEB": {
        "title" : "AK4 jet [[N]]: ECAL barr. rec hits multiplicity",
        "nbins" : 101-50,
        "min" : 0-0.5,
        "max" : 102-0.5,
        "log" : True,
    },
    "JetsNegative.energyRecHitsEB": {
        "title" : "energyRecHitsEB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 250,
        "log" : True,
    },
    "JetsNegative.eFracRecHitsEB": {
        "title" : "eFracRecHitsEB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 1,
        "log" : True,
    },

    "Jets[[N]].nRecHitsEB": {
        "title" : "AK4 jet [N]: ECAL barr. rec hits multiplicity",
        "nbins" : 51,
        "min" : 0-0.5,
        "max" : 102-0.5,
        "log" : True,
    },
    "Jets[[N]].energyRecHitsEB": {
        "title" : "energyRecHitsEB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "Jets[[N]].eFracRecHitsEB": {
        "title" : "eFracRecHitsEB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 1,
        "log" : True,
    },

    "Jets.energyRecHitsEB/Jets.energy": {
        "title" : "energyRecHitsEB/energy AK4 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,
        "log" : True,
    },

    "JetsNegative.timeRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time (ns)",
        "nbins" : 100,
        "min" : -5,
        "max" : 5,
        "log" : True,
    },
    "JetsNegative.timeRMSRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time RMS (ns)",
        "nbins" : 50,
        "min" : 0,
        "max" : 15,
        "log" : True,
    },
    "JetsNegative[[N]].timeRecHitsEB": {
        "title" : "AK4 jet[[N]]: ECAL barr. rec hits time (ns)",
        "nbins" : 100,#100
        "min" : -5,#-5
        "max" : 5,#5
        "log" : True,
    },
    "JetsNegative[[N]].timeRMSRecHitsEB": {
        "title" : "AK4 jet [[N]]: ECAL barr. rec hits time RMS (ns)",
        "nbins" : 50,
        "min" : 0,
        "max" : 15,
        "log" : True,
    },

    "Jets.timeRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time (ns)",
        "nbins" : 50,
        "min" : -1,
        "max" : 10,
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
    "Jets.timeRMSRecHitsEB/Jets.nRecHitsEB": {
        "title" : "AK4 jets: ECAL barr. rec hits time RMS corrected. (ns)",
        "nbins" : 50,
        "min" : 0,
        "max" : 4,
        "log" : True,
    },
    "Jets[[N]].timeRecHitsEB": {
        "title" : "AK4 jet [[N]]: ECAL barr. rec hits time (ns)",
        "nbins" : 50,
        "min" : -1,
        "max" : 5,
        "log" : True,
    },
    #shapes
    "Jets.sig1EB": {
        "title" : "sig1EB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "Jets.sig2EB": {
        "title" : "sig2EB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,#0.5,
        "log" : True,
    },
    "Jets[[N]].sig1EB": {
        "title" : "sig1EB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "Jets[[N]].sig2EB": {
        "title" : "sig2EB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "Jets.sigAvEB": {
        "title" : "sigAvEB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "Jets.ptDEB": {
        "title" : "fragmentation function with EB hits in AK4 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 2.,
        "log" : True,
    },
    "Jets[[N]].ptDEB": {
        "title" : "ptDEB jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 2.,
        "log" : True,
    },

    "Jets.sig1PF": {
        "title" : "sig1PF jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "Jets.sig2PF": {
        "title" : "sig2PF jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "Jets[[N]].sig1PF": {
        "title" : "sig1PF jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "Jets[[N]].sig2PF": {
        "title" : "sig2PF jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "Jets.sigAvPF": {
        "title" : "sigAvPF jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "Jets.ptDPF": {
        "title" : "p_{T}D PF jets",
        "nbins" : 100,
        "min" : 0,
        "max" : 1.5,
        "log" : True,
    },
    "Jets[[N]].ptDPF": {
        "title" : "ptDPF jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 2,
        "log" : True,
    },

    #Beam Halo
    "Jets.sig1EE": {
        "title" : "average #eta position of EB rec hits in jets",
        "nbins" : 50,
        "min" : -1.5,
        "max" : 1.5,
        "log" : True,
    },
    "Jets.sig2EE": {
        "title" : "average #varphi position of EB rec hits in jets",
        "nbins" : 50,
        "min" : -3.5,
        "max" : 3.5,#0.5,
        "log" : True,
    },
    "Jets.sigAvEE": {
        "title" : "mean ATLAS variable",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.5,#0.5,
        "log" : True,
    },

    "Jets.sig1HB": {
        "title" : "spread #eta position of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.3,
        "log" : True,
    },
    "Jets.sig2HB": {
        "title" : "spread #varphi position of EB rec hits in jets",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.3,#0.5,
        "log" : True,
    },
    "Jets.sigAvHB": {
        "title" : "spread ATLAS variable",
        "nbins" : 50,
        "min" : 0.,
        "max" : 0.3,#0.5,
        "log" : True,
    },



    "Jets.xRecHitsEB": {
        "title" : "xRecHitsEB jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.yRecHitsEB": {
        "title" : "yRecHitsEB jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.zRecHitsEB": {
        "title" : "zRecHitsEB jets",
        "nbins" : 50,
        "min" : -300,
        "max" : 300,
        "log" : True,
    },
    "Jets.radiusRecHitsEB": {
        "title" : "radiusRecHitsEB jets",
        "nbins" : 50,
        "min" : 125,
        "max" : 180,
        "log" : True,
    }, 
    #EE rec hits
    "Jets.nRecHitsEE": {
        "title" : "nRecHitsEE jets",
        "nbins" : 31,
        "min" : 0-0.5,
        "max" : 31-0.5,
        "log" : True,
    },
    "Jets.energyRecHitsEE": {
        "title" : "energyRecHitsEE jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "Jets.timeRecHitsEE": {
        "title" : "timeRecHitsEE jets",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },
    "Jets.xRecHitsEE": {
        "title" : "xRecHitsEE jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.yRecHitsEE": {
        "title" : "yRecHitsEE jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.zRecHitsEE": {
        "title" : "zRecHitsEE jets",
        "nbins" : 50,
        "min" : -400,
        "max" : 400,
        "log" : True,
    },
    "Jets.radiusRecHitsEE": {
        "title" : "radiusRecHitsEE jets",
        "nbins" : 50,
        "min" : 125,
        "max" : 180,
        "log" : True,
    },         
    "Jets.xLLP": {
        "title" : "xLLP jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.xGenb": {
        "title" : "xGenb jets",
        "nbins" : 50,
        "min" : -140,
        "max" : 140,
        "log" : True,
    },
    "Jets.yLLP": {
        "title" : "yLLP jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "Jets.zLLP": {
        "title" : "zLLP jets",
        "nbins" : 50,
        "min" : -450,
        "max" : 450,
        "log" : True,
    },
    "Jets.radiusLLP": {
        "title" : "radiusLLP jets",
        "nbins" : 50,
        "min" : 30,
        "max" : 300,
        "log" : True,
    },
    "Jets.radiusLLPCaloCorr": {
        "title" : "radius of LLP matched to corresponding jets (cm)",
        "nbins" : 50,
        "min" : 30,
        "max" : 300,
        "log" : True,
    },

    #HB rec hits
    "Jets.nRecHitsHB": {
        "title" : "nRecHitsHB jets",
        "nbins" : 51,
        "min" : 0-0.5,
        "max" : 51-0.5,
        "log" : True,
    },
    "Jets.energyRecHitsHB": {
        "title" : "energyRecHitsHB jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "Jets.energyRecHitsHB/Jets.energy": {
        "title" : "energyRecHitsHB/energy AK4 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,
        "log" : True,
    },
    "Jets.timeRecHitsHB": {
        "title" : "timeRecHitsHB jets",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },
    "Jets[[N]].timeRecHitsHB": {
        "title" : "timeRecHitsHB jet [[N]]",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },
    "Jets.xRecHitsHB": {
        "title" : "xRecHitsHB jets",
        "nbins" : 50,
        "min" : -290,
        "max" : 290,
        "log" : True,
    },
    "Jets.yRecHitsHB": {
        "title" : "yRecHitsHB jets",
        "nbins" : 50,
        "min" : -290,
        "max" : 290,
        "log" : True,
    },
    "Jets.zRecHitsHB": {
        "title" : "zRecHitsHB jets",
        "nbins" : 50,
        "min" : -450,
        "max" : 450,
        "log" : True,
    },
    "Jets.radiusRecHitsHB": {
        "title" : "radiusRecHitsHB jets",
        "nbins" : 50,
        "min" : 175,
        "max" : 290,
        "log" : True,
    }, 
#nRecHitsEB(-1), timeRecHitsEB(-100.), timeRMSRecHitsEB(-1.), energyRecHitsEB(-1.), energyErrorRecHitsEB(-1.), xRecHitsEB(-1000.), yRecHitsEB(-1000.), zRecHitsEB(-1000.), radiusRecHitsEB
    #VBF Pair jets
    "VBFPairJets.VBFPairJets[[N]].pt": {
      "title" : "VBF pair jet [[N]] p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 800,
      "log" : True,
    },


    # JetConstits vector
    "JetConstits.JetConstits.pt": {
      "title" : "jet constituents p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    "JetConstits.JetConstits[[N]].pt": {
      "title" : "jet constituent [[N]] p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },


    # MatchedCHSJets
    "MatchedCHSJet[N].pt": {
      "title" : "jet [N] p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 800,
      "log" : True,
    },
    "MatchedCHSJet[N].eta": {
      "title" : "jet [N] #eta",
      "nbins" : 30,
      "min" : -3,
      "max" : 3,
      "log" : False,
    },
    "MatchedCHSJet[N].phi": {
      "title" : "jet [N] #varphi",
      "nbins" : 60,
      "min" : -3.15,
      "max" : 3.15,
      "log" : False,
    },
    "MatchedCHSJet[N].mass": {
      "title" : "jet [N] mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 150,
      "log" : False,
    },
    "MatchedCHSJet[N].CSV": {
      "title" : "jet [N] CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].flavour": {
      "title" : "jet [N] flavour",
      "nbins" : 25,
      "min" : -0.5,
      "max" : 24.5,
      "log" : False,
    },
    "MatchedCHSJet[N].chf": {
      "title" : "jet [N] charged hadron fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].nhf": {
      "title" : "jet [N] neutral hadron fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].phf": {
      "title" : "jet [N] photon fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].elf": {
      "title" : "jet [N] electron fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].muf": {
      "title" : "jet [N] muon fraction",
      "nbins" : 20,
      "min" : 0,
      "max" : 1,
      "log" : False,
    },
    "MatchedCHSJet[N].chm": {
      "title" : "jet [N] charged multiplicity",
      "nbins" : 20,
      "min" : 0,
      "max" : 50,
      "log" : False,
    },
    
    # Z Control Region
    "MT": {
      "title" : "MET + l transverse mass (GeV)",
      "nbins" : 51,
      "min" : 0,
      "max" : 102,
      "log" : False,
    },
    "Z_mass": {
      "title" : "Z->ll mass (GeV)",
      "nbins" : 40,
      "min" : 69.5,
      "max" : 109.5,
      "log" : False,
    },
    "Z_pt": {
      "title" : "Z p_{T} (GeV)",
      "nbins" : 50,
      "min" : 30,
      "max" : 100,#1000,
      "log" : True,
    },
    "Z_eta": {
      "title" : "Z #eta (GeV)",
      "nbins" : 50,
      "min" : -5.2,
      "max" : 5.2,
      "log" : True,
    },
    "Z_lep0_pt": {
      "title" : "p_{T} l_{0} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "Z_lep1_pt": {
      "title" : "p_{T} l_{1} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "Z_lep0_eta": {
      "title" : "#eta l_{0} (GeV)",
      "nbins" : 50,
      "min" : -5.2,
      "max" : 5.2,
      "log" : True,
    },
    "Z_lep1_eta": {
      "title" : "#eta l_{1} (GeV)",
      "nbins" : 50,
      "min" : -5.2,
      "max" : 5.2,
      "log" : True,
    },

    # VBFPair
    "VBFPair.mass": {
      "title" : "VBF pair mass (GeV)",
      "nbins" : 50,
      "min" : 400,
      "max" : 2400,
      "log" : True,
    },
    "VBFPair.dEta": {
      "title" : "VBF pair #Delta #eta",
      "nbins" : 50,
      "min" : 2.5,
      "max" : 10.5,
      "log" : True,
    },

    #EcalRecHits
    "EcalRecHitsAK4.eta": {
        "title" : "EB rec hits #eta (associated to AK4 jets)",
      "nbins" : 25,
      "min" : 2.,#-5.2,#-3,
      "max" : 2.,#5.2,#3,
      "log" : True,
    },
    "EcalRecHitsAK4.phi": {
      "title" : "EB rec hits #varphi (associated to AK4 jets)",
      "nbins" : 40,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },
    "skimmedEcalRecHitsAK4.eta": {
        "title" : "EB rec hits #eta (associated to AK4 barrel jets)",
      "nbins" : 25,
      "min" : -1.5,#-1.4,#-5.2,#-3,
      "max" : 1.5,#1.4,#5.2,#3,
      "log" : True,
    },
    "skimmedEcalRecHitsAK4.phi": {
      "title" : "EB rec hits #varphi (associated to AK4 barrel jets)",
      "nbins" : 40,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },


    #PFCandidates pt
    "PFCandidates.pt": {
      "title" : "jets PF candidates p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },
    "PFCandidatesAK4.pt": {
      "title" : "jets PF candidates p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "PFCandidatesAK8.pt": {
      "title" : "jets PF candidates p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 300,
      "log" : True,
    },
    "@PFCandidatesAK8.size()": {
      "title" : "number of PF candidates per AK8 jets",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },

    "PFCandidatesAK8[[N]].pt": {
      "title" : "jets PF candidates [[N]] p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 300,
      "log" : True,
    },

    # FatJets variables
    "FatJets[[N]].isGenMatched": {
      "title" : "AK8 jet [[N]] is gen matched",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : True,
    },
    "FatJets[[N]].sigprob": {
      "title" : "AK8 jet [[N]] DNN score",
      "nbins" : 50,#40
      "min" : 0.,
      "max" : 1,
      "log" : True,
    },
    "FatJets.sigprob": {
      "title" : "AK8 jets DNN score",
      "nbins" : 50,#40,
      "min" : 0,#170,
      "max" : 1,#2170,
      "log" : True,
    },
    #FatJets kinematics
    "FatJets[[N]].pt": {
      "title" : "AK8 jet [[N]] p_{T} (GeV)",
      "nbins" : 40,#40
      "min" : 170,
      "max" : 1000,
      "log" : True,
    },
    "FatJets.pt": {
      "title" : "AK8 jets p_{T} (GeV)",
      "nbins" : 50,#40,
      "min" : 0,#170,
      "max" : 3000,#2170,
      "log" : True,
    },
    "FatJets.energy": {
      "title" : "AK8 jets energy (GeV)",
      "nbins" : 50,#40
      "min" : 0,#170,
      "max" : 3000,
      "log" : True,
    },
    "FatJets[[N]].eta": {
      "title" : "AK8 jet [[N]] #eta",
      "nbins" : 50,
      "min" : -3,
      "max" : 3,
      "log" : True,
    },
    "FatJets[[N]].phi": {
      "title" : "AK8 jet [[N]] #varphi",
      "nbins" : 60,
      "min" : -3.2,
      "max" : 3.2,
      "log" : True,
    },
    "FatJets[[N]].mass": {
      "title" : "AK8 jet [[N]] mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 100,
      "log" : True,
    },
    "FatJets[[N]].CSV": {
      "title" : "AK8 jet [[N]] CSV",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].pfBoostedDoubleSVAK8": {
      "title" : "AK8 jet [[N]] pfBoostedDoubleSVAK8",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.nSelectedTracks": {
        "title" : "Number of selected tracks per AK8 jets",
        "nbins" : 20,
        "min" : 0-0.5,
        "max" : 20-0.5,
        "log" : True,
    },
    "FatJets.nConstituents": {
        "title" : "Number of constituents per AK8 jets",
        "nbins" : 20,
        "min" : 0-0.5,
        "max" : 20-0.5,
        "log" : True,
    },

    "FatJets.FracCal": {
      "title" : "AK8 jets ECAL energy/HCAL energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 10,
      "log" : True,
    },

    "FatJets[[N]].alphaMax": {
      "title" : "AK8 jet [[N]] #alpha_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJets.alphaMax": {
      "title" : "AK8 jets #alpha_{max}",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJets[[N]].sigIP2DMedian": {
      "title" : "AK8 jet [[N]] sigIP2DMedian",
      "nbins" : 50,
      "min" : -5,#-50,
      "max" : 7,#50,
      "log" : True,
    },
    "-log(abs(FatJets[[N]].sigIP2DMedian))": {
      "title" : "AK8 jet [[N]] sigIP2DMedian",
      "nbins" : 100,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].theta2DMedian": {
      "title" : "AK8 jet [[N]] theta2DMedian",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].flavour": {
      "title" : "AK8 jet [[N]] flavour",
      "nbins" : 25,
      "min" : -0.5,
      "max" : 24.5,
      "log" : False,
    },
#    "FatJets[[N]].FracCal": {
#      "title" : "AK8 jet [[N]] ECAL energy/HCAL energy",
#      "nbins" : 50,
#      "min" : 0,
#      "max" : 10,
#      "log" : True,
#    },
#    "FatJets[[N]].hcalE": {
#      "title" : "AK8 jet [[N]] HCAL energy",
#      "nbins" : 50,
#      "min" : 0,
#      "max" : 500,
#      "log" : True,
#    },
#    "FatJets[[N]].ecalE": {
#      "title" : "AK8 jet [[N]] ECAL energy",
#      "nbins" : 50,
#      "min" : 0,
#      "max" : 200,
#      "log" : True,
#    },

    #FatJets tracker
    "FatJets.minDeltaRPVTracks": {
      "title" : "AK8 jets minDeltaRPVTracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJets.medianIP2D": {
      "title" : "AK8 jets medianIP2D",
      "nbins" : 50,
      "min" : -10000,
      "max" : 10000,
      "log" : True,
    },
    "FatJets.gammaMaxET": {
      "title" : "AK8 jets gammaMaxET",
      "nbins" : 50,
      "min" : 0.,
      "max" : 2.,
      "log" : True,
    },       

    "FatJets.nPixelHitsMedian": {
      "title" : "AK8 jets nPixelHitsMedian",
      "nbins" : 15,
      "min" : 0.-0.5,
      "max" : 15.-0.5,
      "log" : True,
    },
    
    "FatJets.nHitsMedian": {
      "title" : "AK8 jets nHitsMedian",
      "nbins" : 50,
      "min" : 0.-0.5,
      "max" : 50.-0.5,
      "log" : True,
    },
    
    "FatJets.dxyMedian": {
      "title" : "AK8 jets dxyMedian",
      "nbins" : 50,
      "min" : -5.,#20 is too much?
      "max" : 5.,
      "log" : True,
    },
    
    "FatJets.dzMedian": {
      "title" : "AK8 jets dzMedian",
      "nbins" : 50,
      "min" : -40.,
      "max" : 40.,
      "log" : True,
    },


    #FatJets constituents
    "FatJets[[N]].nConstituents": {
      "title" : "AK8 jet [[N]] number of jet constituents",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 99.5+50,
      "log" : True,
    },
    "FatJets[[N]].nTrackConstituents": {
      "title" : "AK8 jet [[N]] number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5+50,
      "log" : True,
    },
    "FatJets.nTrackConstituents": {
      "title" : "AK8 jets number of jet constituents with tracks",
      #"title" : "Leading jet: n. of constituents with tracks",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5+50,
      "log" : True,
    },
    "(FatJets[[N]].nTrackConstituents)/(FatJets[[N]].nConstituents)": {
      "title" : "percentage of jet [[N]] constituents with tracks",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    #FatJets energies
    "FatJets[[N]].cHadE": {
      "title" : "AK8 jet [[N]] charged hadron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    "FatJets[[N]].nHadE": {
      "title" : "AK8 jet [[N]] neutral hadron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    "FatJets[[N]].muE": {
      "title" : "AK8 jet [[N]] #mu energy",
      "nbins" : 20,
      "min" : 0,
      "max" : 20,
      "log" : True,
    },
    "FatJets[[N]].eleE": {
      "title" : "AK8 jet [[N]] electron energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].photonE": {
      "title" : "AK8 jet [[N]] photon energy",
      "nbins" : 50,
      "min" : 0,
      "max" : 200,
      "log" : True,
    },
    #FatJets energy fractions
    "FatJets[[N]].cHadEFrac": {
      "title" : "AK8 jet [[N]] charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.cHadEFrac": {
      "title" : "AK8 jets charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].nHadEFrac": {
      "title" : "AK8 jet [[N]] neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.nHadEFrac": {
      "title" : "AK8 jet neutral hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].muEFrac": {
      "title" : "AK8 jet [[N]] #mu energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.muEFrac": {
      "title" : "AK8 jets #mu energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].eleEFrac": {
      "title" : "AK8 jet [[N]] electron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.eleEFrac": {
      "title" : "AK8 jets electron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets[[N]].photonEFrac": {
      "title" : "AK8 jet [[N]] photon energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "FatJets.photonEFrac": {
      "title" : "AK8 jets photon energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #FatJets multiplicities
    "FatJets[[N]].eleMulti": {
      "title" : "AK8 jet [[N]] electron multiplicity",
      "nbins" : 20,
      "min" : 0,
      "max" : 20,
      "log" : True,
    },
    "FatJets[[N]].muMulti": {
      "title" : "AK8 jet [[N]] #mu multiplicity",
      "nbins" : 20,
      "min" : 0,
      "max" : 20,
      "log" : True,
    },
    "FatJets[[N]].photonMulti": {
      "title" : "AK8 jet [[N]] photon multiplicity",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].cHadMulti": {
      "title" : "AK8 jet [[N]] charged hadron multiplicity",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].nHadMulti": {
      "title" : "AK8 jet [[N]] neutral hadron multiplicity",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].cMulti": {
      "title" : "AK8 jet [[N]] charged multiplicity",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].nMulti": {
      "title" : "AK8 jet [[N]] neutral multiplicity",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "FatJets[[N]].npr": {
      "title" : "AK8 jet [[N]] number of components",
      "nbins" : 100,
      "min" : 0,
      "max" : 100,
      "log" : True,
    },
    #FatJets multiplicity fractions
    "FatJets[[N]].eleMultiFrac": {
      "title" : "AK8 jet [[N]] electron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].muMultiFrac": {
      "title" : "AK8 jet [[N]] #mu multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].photonMultiFrac": {
      "title" : "AK8 jet [[N]] photon multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].cHadMultiFrac": {
      "title" : "AK8 jet [[N]] charged hadron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].nHadMultiFrac": {
      "title" : "AK8 jet [[N]] neutral hadron multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].cMultiFrac": {
      "title" : "AK8 jet [[N]] charged multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    "FatJets[[N]].nMultiFrac": {
      "title" : "AK8 jet [[N]] neutral multiplicity fraction",
      "nbins" : 51,
      "min" : -0.01,
      "max" : 1.01,
      "log" : True,
    },
    #FatJets substructure
    "FatJets[[N]].chsTau21": {
      "title" : "AK8 jet [[N]] CHS #tau_{2}/#tau_{1}",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJets[[N]].puppiTau21": {
      "title" : "AK8 jet [[N]] puppi #tau_{2}/#tau_{1}",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJets.puppiTau21": {
      "title" : "AK8 jets puppi #tau_{2}/#tau_{1}",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },    
    "FatJets.chsTau21": {
      "title" : "AK8 jets CHS #tau_{2}/#tau_{1}",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },    
    "FatJets[[N]].nSoftDropSubJets": {
      "title" : "AK8 jet [[N]] : n. CHS softdrop sub-jets",
      "nbins" : 4,
      "min" : -0.5,
      "max" : 3.5,
      "log" : True,
    },
    #FatJets tracks hits
    "FatJets[[N]].nPixelHitsMedian": {
      "title" : "AK8 jet [[N]] median number of pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "FatJets[[N]].nHitsMedian": {
      "title" : "AK8 jet [[N]] median number of tracker hits",
      "nbins" : 20+5,
      "min" : -0.5,
      "max" : 19.5+5,
      "log" : True,
    },
    "FatJets[[N]].nTracks0PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 0 pixel hits",
      "nbins" : 20,
      "min" : -0.5,
      "max" : 19.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks1PixelHit": {
      "title" : "AK8 jet [[N]] number of tracks with 1 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks2PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 2 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks3PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 3 pixel hits",
      #"title" : "Leading jet: n. of tracks with 3 pixel hits",
      "nbins" : 40-10,
      "min" : -0.5,
      "max" : 39.5-10,
      "log" : True,
    },
    "FatJets[[N]].nTracks4PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 4 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks5PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 5 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks6PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with 6 pixel hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracksLarger5PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with more than 5 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "FatJets[[N]].nTracksLarger6PixelHits": {
      "title" : "AK8 jet [[N]] number of tracks with more than 6 pixel hits",
      "nbins" : 10,
      "min" : -0.5,
      "max" : 9.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks0LostInnerHits": {
      "title" : "AK8 jet [[N]] number of tracks with 0 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks1LostInnerHit": {
      "title" : "AK8 jet [[N]] number of tracks with 1 lost inner hit",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracks2LostInnerHits": {
      "title" : "AK8 jet [[N]] number of tracks with 2 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "FatJets[[N]].nTracksLarger2LostInnerHits": {
      "title" : "AK8 jet [[N]] number of tracks with more than 2 lost inner hits",
      "nbins" : 40,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },

    "FatJets[[N]].CHSsoftdropMass": {
      "title" : "AK8 jet [[N]] CHS softdrop mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 100,
      "log" : True,
    },
    "FatJets.softdropPuppiMass": {
      "title" : "AK8 jets softdrop puppi mass (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 400,
      "log" : True,
    },
    "FatJets[[N]].CHSsoftdropMassCorr": {
      "title" : "AK8 jet [[N]] CHS softdrop mass corr  (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 100,
      "log" : True,
    },
    #EB rec hits
    "FatJets.nRecHitsEB": {
        "title" : "nRecHitsEB AK8 jets",
        "nbins" : 51,#101,
        "min" : 0-0.5,
        "max" : 102-0.5,
        "log" : True,
    },
    "FatJets.energyRecHitsEB": {
        "title" : "energyRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "FatJets.eFracRecHitsEB": {
        "title" : "eFracRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },

    "FatJets.energyRecHitsEB/FatJets.energy": {
        "title" : "energyRecHitsEB/energy AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,
        "log" : True,
    },
    "FatJets.energyRecHitsHB/FatJets.energy": {
        "title" : "energyRecHitsHB/energy AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,
        "log" : True,
    },
    "FatJets.energyRecHitsHB/FatJets.energyRaw": {
        "title" : "energyRecHitsHB/raw energy AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.2,
        "log" : True,
    },
    #"FatJets.ecalE/FatJets.energy": {
    #    "title" : "ECAL energy/energy AK8 jets",
    #    "nbins" : 50,
    #    "min" : 0,
    #    "max" : 1,
    #    "log" : True,
    #},
    #"FatJets.ecalE/FatJets.energyRaw": {
    #    "title" : "ECAL energy/raw energy AK8 jets",
    #    "nbins" : 50,
    #    "min" : 0,
    #    "max" : 1,
    #    "log" : True,
    #},
    #"FatJets.hcalE/FatJets.energy": {
    #    "title" : "HCAL energy/energy AK8 jets",
    #    "nbins" : 50,
    #    "min" : 0,
    #    "max" : 1,
    #    "log" : True,
    #},
    #"FatJets.hcalE/FatJets.energyRaw": {
    #    "title" : "HCAL energy/raw energy AK8 jets",
    #    "nbins" : 50,
    #    "min" : 0,
    #    "max" : 1,
    #    "log" : True,
    #},
    "FatJets.timeRecHitsEB": {
        "title" : "timeRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : -4,#-10,
        "max" : 4,#10,
        "log" : True,
    },
    "FatJets[[N]].timeRecHitsEB": {
        "title" : "timeRecHitsEB AK8 jet [[N]]",
        "nbins" : 50,
        "min" : -4,#-10,
        "max" : 4,#10,
        "log" : True,
    },



    #shapes
    "FatJets.sig1EB": {
        "title" : "sig1EB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "FatJets.sig2EB": {
        "title" : "sig2EB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "FatJets.sigAvEB": {
        "title" : "sigAvEB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "FatJets.ptDEB": {
        "title" : "ptDEB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 2.,
        "log" : True,
    },

    "FatJets.sig1PF": {
        "title" : "sig1PF AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "FatJets.sig2PF": {
        "title" : "sig2PF AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.5,
        "log" : True,
    },
    "FatJets.sigAvPF": {
        "title" : "sigAvPF AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 3,
        "log" : True,
    },
    "FatJets.ptDPF": {
        "title" : "p_{T}D PF AK8 jets",
        "nbins" : 100,
        "min" : 0,
        "max" : 1.5,
        "log" : True,
    },
    "FatJets[[N]].ptDPF": {
        "title" : "ptDPF AK8 jet [N]",
        "nbins" : 50,
        "min" : 0,
        "max" : 2,
        "log" : True,
    },


    "FatJets.xRecHitsEB": {
        "title" : "xRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.yRecHitsEB": {
        "title" : "yRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.zRecHitsEB": {
        "title" : "zRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : -300,
        "max" : 300,
        "log" : True,
    },
    "FatJets.radiusRecHitsEB": {
        "title" : "radiusRecHitsEB AK8 jets",
        "nbins" : 50,
        "min" : 125,
        "max" : 180,
        "log" : True,
    },    
    #EE rec hits
    "FatJets.nRecHitsEE": {
        "title" : "nRecHitsEE AK8 jets",
        "nbins" : 31,
        "min" : 0-0.5,
        "max" : 31-0.5,
        "log" : True,
    },
    "FatJets.energyRecHitsEE": {
        "title" : "energyRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "FatJets.timeRecHitsEE": {
        "title" : "timeRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },
    "FatJets.xRecHitsEE": {
        "title" : "xRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.yRecHitsEE": {
        "title" : "yRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.zRecHitsEE": {
        "title" : "zRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : -300,
        "max" : 300,
        "log" : True,
    },
    "FatJets.radiusRecHitsEE": {
        "title" : "radiusRecHitsEE AK8 jets",
        "nbins" : 50,
        "min" : 125,
        "max" : 180,
        "log" : True,
    },       
    "FatJets.xLLP": {
        "title" : "xLLP AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.xGenb": {
        "title" : "xGenb AK8 jets",
        "nbins" : 50,
        "min" : -140,
        "max" : 140,
        "log" : True,
    },
    "FatJets.yLLP": {
        "title" : "yLLP AK8 jets",
        "nbins" : 50,
        "min" : -180,
        "max" : 180,
        "log" : True,
    },
    "FatJets.zLLP": {
        "title" : "zLLP AK8 jets",
        "nbins" : 50,
        "min" : -450,
        "max" : 450,
        "log" : True,
    },
    "FatJets.radiusLLP": {
        "title" : "radiusLLP AK8 jets",
        "nbins" : 50,
        "min" : 30,
        "max" : 300,
        "log" : True,
    },
    "FatJets.radiusLLPCaloCorr": {
        "title" : "radiusLLP AK8 jets (calo corrections)",
        "nbins" : 50,
        "min" : 30,
        "max" : 300,
        "log" : True,
    },
    #HB rec hits
    "FatJets.nRecHitsHB": {
        "title" : "nRecHitsHB AK8 jets",
        "nbins" : 51,
        "min" : 0-0.5,
        "max" : 51-0.5,
        "log" : True,
    },
    "FatJets.energyRecHitsHB": {
        "title" : "energyRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "FatJets.timeRecHitsHB": {
        "title" : "timeRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : -2,#-10,
        "max" : 3,#10,
        "log" : True,
    },
    "FatJets[[N]].timeRecHitsHB": {
        "title" : "timeRecHitsHB AK8 jet [[N]]",
        "nbins" : 50,
        "min" : -2,#-10,
        "max" : 3,#10,
        "log" : True,
    },
    "FatJets.xRecHitsHB": {
        "title" : "xRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : -290,
        "max" : 290,
        "log" : True,
    },
    "FatJets.yRecHitsHB": {
        "title" : "yRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : -290,
        "max" : 290,
        "log" : True,
    },
    "FatJets.zRecHitsHB": {
        "title" : "zRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : -450,
        "max" : 450,
        "log" : True,
    },
    "FatJets.radiusRecHitsHB": {
        "title" : "radiusRecHitsHB AK8 jets",
        "nbins" : 50,
        "min" : 175,
        "max" : 290,
        "log" : True,
    }, 
#    "FatJets[N].pt": {
#      "title" : "AK8 jet [[N]] p_{T} (GeV)",
#      "nbins" : 40,
#      "min" : 200,
#      "max" : 1000,
#      "log" : True,
#    },
    # Vertices
    # .....

    # Tracks
    # .....

    # CSCSegments
    "@CSCSegments.size()": {
      "title" : "number of CSC segments",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },
    "nCSCSegments": {
      "title" : "number of CSC segments",
      "nbins" : 25,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },

    # DTSegments
    "nDTSegments": {
      "title" : "number of DT segments",
      "nbins" : 10,#55,
      "min" : -0.5,
      "max" : 99.5,
      "log" : True,
    },
    "nDTSegmentsStation1": {
      "title" : "number of DT segments in station 1",
      "nbins" : 20,#55,
      "min" : -0.5,
      "max" : 79.5,
      "log" : True,
    },
    "nDTSegmentsStation2": {
      "title" : "number of DT segments in station 2",
      "nbins" : 20,#55,
      "min" : -0.5,
      "max" : 79.5,#54.5,
      "log" : True,
    },
    "nDTSegmentsStation3": {
      "title" : "number of DT segments in station 3",
      "nbins" : 20,#55,
      "min" : -0.5,
      "max" : 39.5,
      "log" : True,
    },
    "nDTSegmentsStation4": {
      "title" : "number of DT segments in station 4",
      "nbins" : 20,#55,
      "min" : -0.5,
      "max" : 19.5,#54.5,
      "log" : True,
    },

    # Standalone muons
    "nCosmicMuons": {
      "title" : "number of cosmic muons",
      "nbins" : 15,
      "min" : -0.5,
      "max" : 14.5,
      "log" : True,
    },
    "n_clusters": {
      "title" : "n. clusters DT (DBSCAN)",
      "nbins" : 15,
      "min" : -0.5,
      "max" : 14.5,
      "log" : True,
    },
    "n_noise": {
      "title" : "n. noise DT (DBSCAN)",
      "nbins" : 15,
      "min" : -0.5,
      "max" : 14.5,
      "log" : True,
    },
    "isCosmic": {
      "title" : "Cosmic ray grazing ECAL",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : False,
    },
    "isDT_fit": {
      "title" : "DT segments in opposite emispheres",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : False,
    },

    "dt_fit_chi2": {
      "title" : "dt_fit_chi2",
      "nbins" : 25,
      "min" : 0.,
      "max" : 10.,
      "log" : True,
    },

    "dt_fit_chi2_reduced": {
      "title" : "dt_fit_chi2_reduced",
      "nbins" : 25,
      "min" : 0.,
      "max" : 0.5,
      "log" : True,
    },

    "dt_ecal_dist": {
        "title" : "Distance(cosmic 3D line fit,ECAL rec hits #in tagged jets) (m)",
      "nbins" : 25,#25
      "min" : 0.,
      "max" : 5,#2.5
      "log" : True,
    },

    "dt_ecal_acc_dist": {
      "title" : "dt_ecal_acc_dist (m)",
      "nbins" : 25,
      "min" : 0.,
      "max" : 5,#2.5,
      "log" : True,
    },

    "DT_fit_res": {
      "title" : "DT_fit_res",
      "nbins" : 50,
      "min" : 0.,
      "max" : 10.,
      "log" : True,
    },

    "nCosmicMuonsOneLeg": {
      "title" : "number of cosmic muons one leg",
      "nbins" : 15,
      "min" : -0.5,
      "max" : 14.5,
      "log" : True,
    },

    "nStandAloneMuons": {
      "title" : "number of standalone muons",
      "nbins" : 15,
      "min" : -0.5,
      "max" : 14.5,
      "log" : True,
    },
    "StandAloneMuons[[N]].pt": {
      "title" : "standalone muon [[N]] p_{T} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "StandAloneMuons[[N]].dxy": {
      "title" : "standalone muon [[N]] dxy (cm)",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },
    "StandAloneMuons[[N]].dxyError": {
      "title" : "standalone muon [[N]] dxy error (cm)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "StandAloneMuons[[N]].dz": {
      "title" : "standalone muon [[N]] dz (cm)",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },
    "StandAloneMuons[[N]].dzError": {
      "title" : "standalone muon [[N]] dz error (cm)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "StandAloneMuons.dxy/StandAloneMuons.dxyError": {
      "title" : "standalone muons dxy sign.",
      "nbins" : 50,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },
    "StandAloneMuons.dz/StandAloneMuons.dzError": {
      "title" : "standalone muons dz sign.",
      "nbins" : 50,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },

    "StandAloneMuons[[N]].dz/StandAloneMuons[[N]].dzError": {
      "title" : "standalone muon [[N]] dz sign.",
      "nbins" : 50,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },
    "StandAloneMuons[[N]].dxy/StandAloneMuons[[N]].dxyError": {
      "title" : "standalone muon [[N]] dxy sign.",
      "nbins" : 50,
      "min" : -50,
      "max" : 50,
      "log" : True,
    },
    "StandAloneMuons[[N]].d0/StandAloneMuons[[N]].d0Error": {
      "title" : "standalone muon [[N]] d0 sign.",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "StandAloneMuons[[N]].d0": {
      "title" : "standalone muon [[N]] d0 (cm)",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "StandAloneMuons[[N]].dsz": {
      "title" : "standalone muon [[N]] dsz (cm)",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "StandAloneMuons[[N]].numberOfValidHits": {
      "title" : "standalone muon [[N]] n. valid hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "StandAloneMuons[[N]].numberOfLostHits": {
      "title" : "standalone muon [[N]] n. lost hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "StandAloneMuons[[N]].recHitsSize": {
      "title" : "standalone muon [[N]] n. hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "StandAloneMuons[[N]].ndof": {
      "title" : "standalone muon [[N]] n. d.o.f.",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "StandAloneMuons[[N]].validFraction": {
      "title" : "standalone muon [[N]] validFraction",
      "nbins" : 10,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    # displaced standalone muons
    "nDisplacedStandAloneMuons": {
      "title" : "number of displaced standalone muons",
      "nbins" : 25,
      "min" : -0.5,
      "max" : 24.5,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].pt": {
      "title" : "standalone muon [[N]] p_{T} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dxy": {
      "title" : "standalone muon [[N]] dxy (cm)",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dxyError": {
      "title" : "standalone muon [[N]] dxy error (cm)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dz": {
      "title" : "standalone muon [[N]] dz (cm)",
      "nbins" : 50,
      "min" : -10,
      "max" : 10,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dzError": {
      "title" : "standalone muon [[N]] dz error (cm)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dz/DisplacedStandAloneMuons[[N]].dzError": {
      "title" : "standalone muon [[N]] dz sign.",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].d0": {
      "title" : "standalone muon [[N]] d0 (cm)",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].dsz": {
      "title" : "standalone muon [[N]] dsz (cm)",
      "nbins" : 50,
      "min" : -100,
      "max" : 100,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].numberOfValidHits": {
      "title" : "standalone muon [[N]] n. valid hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].numberOfLostHits": {
      "title" : "standalone muon [[N]] n. lost hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].recHitsSize": {
      "title" : "standalone muon [[N]] n. hits",
      "nbins" : 50,
      "min" : -0.5,
      "max" : 49.5,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].ndof": {
      "title" : "standalone muon [[N]] n. d.o.f.",
      "nbins" : 50,
      "min" : 0,
      "max" : 50,
      "log" : True,
    },
    "DisplacedStandAloneMuons[[N]].validFraction": {
      "title" : "standalone muon [[N]] validFraction",
      "nbins" : 10,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },


    # Leptons
    "nMuons": {
      "title" : "number of loose muons",
      "nbins" : 6,
      "min" : -0.5,
      "max" : 5.5,
      "log" : True,
    },
    "nTightMuons": {
      "title" : "number of tight muons",
      "nbins" : 6,
      "min" : -0.5,
      "max" : 5.5,
      "log" : True,
    },
    "nElectrons": {
      "title" : "number of veto electrons",
      "nbins" : 6,
      "min" : -0.5,
      "max" : 5.5,
      "log" : True,
    },
    "nTaus": {
      "title" : "number of loose taus",
      "nbins" : 6,
      "min" : -0.5,
      "max" : 5.5,
      "log" : True,
    },
    "nPhotons": {
      "title" : "number of loose photons",
      "nbins" : 6,
      "min" : -0.5,
      "max" : 5.5,
      "log" : True,
    },

    #Leptons (vectors)
    "Lepton[N].pt": {
      "title" : "lepton[N] p_{T} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },

    "Muon1_pt": {
      "title" : "#mu_{1} p_{T} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },
    "Muon1_phi": {
      "title" : "#mu_{1} #varphi",
      "nbins" : 60,
      "min" : -3.2,#15,
      "max" : 3.2,#15,
      "log" : True,
    },
    "Muon1_eta": {
      "title" : "#mu_{1} #eta",
      "nbins" : 50,
      "min" : -2.5,#-3,
      "max" : 2.5,#3,
      "log" : True,
    },

    "isZtoEE": {
      "title" : "is Z->ee",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : True,
    },

    "isZtoMM": {
      "title" : "is Z->#mu #mu",
      "nbins" : 2,
      "min" : -0.5,
      "max" : 1.5,
      "log" : True,
    },
    # MET
    "MEt.pt": {
      "title" : "E_{T}^{miss} (GeV)",
      "nbins" : 50,
      "min" : 0,#200,#0,#200,
      "max" : 2000,#1200,#2000,#0,#2200,
      "log" : True,
    },
    "MEt.ptGen": {
      "title" : "GEN E_{T}^{miss} (GeV)",
      "nbins" : 50,
      "min" : 0,#200,
      "max" : 1000,#0,#2200,
      "log" : True,
    },
    "MEt.sign": {
      "title" : "#slash{E}_{T} significance (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "MEt.phi": {
      "title" : "#slash{E}_{T} #varphi",
      "nbins" : 50,
      "min" : -3.15,
      "max" : 3.15,
      "log" : True,
    },
    "MEt_pt": {
      "title" : "E^{T}_{miss} (GeV)",
      "nbins" : 50,#45,
      "min" : 200,#100,
      "max" : 2200,
      "log" : True,
    },
    "met_pt_nomu": {
      "title" : "E_{T}^{miss} no #mu (GeV)",
      "nbins" : 50,#45,
      "min" : 0,#100,
      "max" : 1000,
      "log" : True,
    },


    # GenBquarks
    "GenBquarks[[N]].pt": {
      "title" : "gen b-quark p_{T} (GeV)",
      "nbins" : 50,#SUSY: 50,
      "min" : 0,
      "max" : 500,#SUSY: 250,
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
        "bins" : [0.1,0.5,1,2,5,10,25,50,100,200,300,400,500,700,1000,2000,3000,5000,7000,10000,20000,50000,100000,1000000],#HeavyHiggs
      #"bins" : [0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,225,250,300,325,350,375,400,450,500,600,700,800,900,1000,1250,1500,2000,3000,5000,7000,10000],#HeavyHiggs
      "nbins" : 0,#100,
      "min" : 0.000001,
      "max" : 1000,
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
      "title" : "gen long lived particle p_{T} (GeV)",
      "nbins" : 40,
      "min" : 0,
      "max" : 1000,
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
      "title" : "gen Higgs p_{T} (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
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
    
    #LEADER
    "MEt_sign": {
      "title" : "#slash{E}_{T} significance (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "Jets0_sigprob": {
      "title" : "jet_{0} DNN output",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets1_sigprob": {
      "title" : "jet_{1} DNN output",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets2_sigprob": {
      "title" : "jet_{2} DNN output",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets3_sigprob": {
      "title" : "jet_{3} DNN output",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets0_cHadEFrac": {
      "title" : "jet_{0} charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets1_cHadEFrac": {
      "title" : "jet_{1} charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets2_cHadEFrac": {
      "title" : "jet_{2} charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jets3_cHadEFrac": {
      "title" : "jet_{3} charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "nTagJets_0p5": {
      "title" : "n. of jets with DNN output > 0.5",
      "nbins" : 3,#7+4,
      "min" : -0.5,
      "max" : 2.5,#6.5+4,
      "log" : True,
    },
    "nTagJets_0p6": {
      "title" : "n. of jets with DNN output > 0.6",
      "nbins" : 3,#7+4,
      "min" : -0.5,
      "max" : 2.5,#6.5+4,
      "log" : True,
    },
    "nTagJets_0p7": {
      "title" : "n. of jets with DNN output > 0.7",
      "nbins" : 3,#7+4,
      "min" : -0.5,
      "max" : 2.5,#6.5+4,
      "log" : True,
    },
    "nTagJets_0p8": {
      "title" : "n. of jets with DNN output > 0.8",
      "nbins" : 3,#7+4,
      "min" : -0.5,
      "max" : 2.5,#6.5+4,
      "log" : True,
    },
    "nTagJets_0p9": {
      "title" : "n. of jets with DNN output > 0.9",
      "nbins" : 3,#7+4,#3,#
      "min" : -0.5,
      "max" : 2.5,#6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p95": {
      "title" : "n. of jets with DNN output > 0.95",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p96": {
      "title" : "n. of jets with DNN output > 0.96",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p97": {
      "title" : "n. of jets with DNN output > 0.97",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p975": {
      "title" : "n. of jets with DNN output > 0.95",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p98": {
      "title" : "n. of jets with DNN output > 0.98",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p99": {
      "title" : "n. of jets with DNN output > 0.99",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p992": {
      "title" : "n. of jets with DNN output > 0.992",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p994": {
      "title" : "n. of jets with DNN output > 0.994",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },

    "nTagJets_0p994_JJ": {
      "title" : "n. of jets with DNN output > 0.994, jet cleaning",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },

    "nTagJets_0p996_JJ": {
      "title" : "n. of jets with DNN output > 0.996, jet cleaning",
      "nbins" : 3,#7+4,#3,#
      "min" : -0.5,
      "max" : 2.5,#6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p996_JJ_eta_1p0": {
      "title" : "n. of jets with DNN output > 0.996, jet cleaning, |#eta|<1",
      "nbins" : 3,#7+4,#3,#
      "min" : -0.5,
      "max" : 2.5,#6.5+4,#2.5,#
      "log" : True,
    },

    "nTagJets_0p99_JJ": {
      "title" : "n. of jets with DNN output > 0.99, jet cleaning",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },


    "nTagJets_0p995": {
      "title" : "n. of jets with DNN output > 0.995",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p996": {
      "title" : "n. of jets with DNN output > 0.996",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagJets_0p9975": {
      "title" : "n. of jets with DNN output > 0.9975",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },

    "nTagJets_cutbased": {
      "title" : "n. of jets passing cut based selections",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTags_cHadEFrac_wp0p2": {
      "title" : "n. of jets with cHadEFrac < 0.2",
      "nbins" : 7,
      "min" : -0.5,
      "max" : 6.5,
      "log" : True,
    },
    "nTags_cHadEFrac_wp0p1": {
      "title" : "n. of jets with cHadEFrac < 0.1",
      "nbins" : 7,
      "min" : -0.5,
      "max" : 6.5,
      "log" : True,
    },
    "nTags_cHadEFrac_wp0p05": {
      "title" : "n. of jets with cHadEFrac < 0.05",
      "nbins" : 7,
      "min" : -0.5,
      "max" : 6.5,
      "log" : True,
    },
    "nTags_cHadEFrac_wp0p02": {
      "title" : "n. of jets with cHadEFrac < 0.02",
      "nbins" : 7,
      "min" : -0.5,
      "max" : 6.5,
      "log" : True,
    },

    ##skim tree
    "Jet_[N]_sigprob": {
      "title" : "AK4 jet [[N]] DNN output",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    #gammaMaxET
    "Jet_[N]_gammaMaxET": {
      "title" : "AK4 jet [[N]] gammaMaxET",
      "nbins" : 40,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },       
    #cHadEFrac
    "Jet_[N]_cHadEFrac": {
      "title" : "AK4 jet [[N]] charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    "Jet_[N]_eleEFrac": {
      "title" : "AK4 jet [[N]] electron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #timeRecHitsEB
    "Jet_[N]_timeRecHitsEB": {
        "title" : "AK4 jet [[N]] ECAL barr. rec hits time (ns)",
        "nbins" : 50,
        "min" : -2,
        "max" : 10,
        "log" : True,
    },
    #timeRecHitsEB
    "Jet_[N]_timeRecHitsHB": {
        "title" : "AK4 jet [[N]] HCAL barr. rec hits time (ns)",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },


    "Jet_[N]_nRecHitsEB": {
        "title" : "nRecHitsEB jet [[N]]",
        "nbins" : 101,
        "min" : 0-0.5,
        "max" : 101-0.5,
        "log" : True,
    },

    "Jet_[N]_nRecHitsHB": {
        "title" : "nRecHitsHB jet [[N]]",
        "nbins" : 51,
        "min" : 0-0.5,
        "max" : 51-0.5,
        "log" : True,
    },

    #energy rec hits
    "Jet_[N]_energyRecHitsEB": {
        "title" : "energyRecHitsEB jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },
    "Jet_[N]_energyRecHitsHB": {
        "title" : "energyRecHitsHB jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 50,
        "log" : True,
    },

    "Jet_[N]_energyRecHitsEB/Jet_[N]_energy": {
        "title" : "energyRecHitsEB/energy jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.3,
        "log" : True,
    },
    "Jet_[N]_energyRecHitsHB/Jet_[N]_energy": {
        "title" : "energyRecHitsHB/energy jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 0.3,
        "log" : True,
    },
    "Jet_[N]_energy": {
        "title" : "energy jet [[N]]",
        "nbins" : 50,
        "min" : 0,
        "max" : 1000,
        "log" : True,
    },



    #minDeltaRPVTracks
    "Jet_[N]_minDeltaRPVTracks": {
      "title" : "AK4 jet [[N]] min #Delta R with tracks from PV",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "Jet_[N]_pt": {
      "title" : "AK4 jet [[N]] p_{T} (GeV)",
      "nbins" : 50,#40
      "min" : 0,
      "max" : 1000,
      "log" : True,
    },

    #AK8
    "nTagFatJets_0p8": {
      "title" : "n. of AK8 jets with DNN output > 0.8",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p9": {
      "title" : "n. of AK8 jets with DNN output > 0.9",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p95": {
      "title" : "n. of AK8 jets with DNN output > 0.95",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p96": {
      "title" : "n. of AK8 jets with DNN output > 0.96",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p97": {
      "title" : "n. of AK8 jets with DNN output > 0.97",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p98": {
      "title" : "n. of AK8 jets with DNN output > 0.98",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p99": {
      "title" : "n. of AK8 jets with DNN output > 0.99",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p999": {
      "title" : "n. of AK8 jets with DNN output > 0.999",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p9999": {
      "title" : "n. of AK8 jets with DNN output > 0.9999",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p99999": {
      "title" : "n. of AK8 jets with DNN output > 0.99999",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_0p999999": {
      "title" : "n. of AK8 jets with DNN output > 0.999999",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },
    "nTagFatJets_cutbased": {
      "title" : "n. of AK8 jets passing cut based selections",
      "nbins" : 7+4,#3,#
      "min" : -0.5,
      "max" : 6.5+4,#2.5,#
      "log" : True,
    },

    "FatJet_[N]_sigprob": {
      "title" : "AK8 jet [[N]] DNN output",
      "nbins" : 25,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    #gammaMaxET
    "FatJet_[N]_gammaMaxET": {
      "title" : "AK8 jet [[N]] gammaMaxET",
      "nbins" : 50,
      "min" : 0.,
      "max" : 2.,
      "log" : True,
    },       
    #cHadEFrac
    "FatJet_[N]_cHadEFrac": {
      "title" : "AK8 jet [[N]] charged hadron energy fraction",
      "nbins" : 50,
      "min" : 0,
      "max" : 1,
      "log" : True,
    },
    #timeRecHitsEB
    "FatJet_[N]_timeRecHitsEB": {
        "title" : "AK8 jet [[N]]  ECAL barr. rec hits time (ns)",
        "nbins" : 50,
        "min" : -10,
        "max" : 10,
        "log" : True,
    },
    #minDeltaRPVTracks
    "FatJet_[N]_minDeltaRPVTracks": {
      "title" : "AK8 jet [[N]] min #Delta R with tracks from PV",
      "nbins" : 50,
      "min" : 0.,
      "max" : 1.,
      "log" : True,
    },
    "FatJet_[N]_pt": {
      "title" : "AK8 jet [[N]] p_{T} (GeV)",
      "nbins" : 40,#40
      "min" : 170,
      "max" : 2000,
      "log" : True,
    },

    #Gen-level
    "GenGravitinos[[N]].pt": {
      "title" : "gen gravitino[[N]] p_{T} (GeV)",
      "nbins" : 50,#SUSY: 50,
      "min" : 0,
      "max" : 500,#SUSY: 250,
      "log" : True,
    },

    "inv_mass_reco_higgs_0": {
      "title" : "inv_mass_reco_higgs_0 (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "inv_mass_reco_higgs_1": {
      "title" : "inv_mass_reco_higgs_1 (GeV)",
      "nbins" : 50,
      "min" : 0,
      "max" : 500,
      "log" : True,
    },
    "dR_LLPs": {
      "title" : "dR_LLPs",
      "nbins" : 50,
      "min" : 0,
      "max" : 5,
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
