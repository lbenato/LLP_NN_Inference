/*
 * Minimal example showing how to evaluate data contained in a flat tree structure using TensorFlow.
 * By default, the inference code uses single threading and no batching. The thread model is
 * configurable, however, note that this is done differently depending on the version of TensorFlow,
 * which changed significantly as of version 2.
 *
 * Author: Marcel Rieger
 */

// Adapted to LLP displaced jets in calorimeter by Lisa Benato

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "TFile.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TMath.h"
#include "TGraph2D.h"
#include "TRandom2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TF2.h"
#include "TF3.h"
#include "TF1.h"
#include "Math/Functor.h"
#include "TPolyLine3D.h"
#include "Math/Vector3D.h"
#include "Fit/Fitter.h"
//#include "TLinearFitter.h"

#include <cassert>
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

using namespace ROOT::Math;
using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

    if(argc<3)
    //if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    //needed argv: input, output, a sting to tell the year for phi selections, a string for time smearing

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "ERA string: " << argv[3] << std::endl;

    auto start = std::chrono::system_clock::now();//time!     
    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    std::string eraString = argv[3];

    float minphi(-9999.);
    float maxphi( 9999.);
    std::string tag_2017 = "2017";
    std::string tag_2018 = "2018";
    if(eraString.find(tag_2017) != std::string::npos)
      {
	minphi = 3.5;
	maxphi = 2.7;
      }
    if(eraString.find(tag_2018) != std::string::npos)
      {
	minphi = 0.9;
	maxphi = 0.4;
      }

    std::string timeCBFilename = argv[10];

    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;

    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_AK4_v3/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_AK4_v3/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;

    // threading setup
    // to enable tensorflow-native multi-threading, change to "tensorflow" and increase nThreads
    std::string threadPool = "no_threads";
    int nThreads = 1;

    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());
    TH1F   *counter = (TH1F*)inputFile->Get("counter");
    TH1F   *n_pass = (TH1F*)inputFile->Get("n_pass");
    TH1F   *n_odd = (TH1F*)inputFile->Get("n_odd");
    TH1F   *n_even = (TH1F*)inputFile->Get("n_even");

    float  tree_weight = inputTree->GetWeight();

    TFile *timeCBFile = TFile::Open(timeCBFilename.data(),"READ"); if (!timeCBFile) return 0;
    TF1  *dataCB = (TF1*)timeCBFile->Get("data_CB");
    TF1  *mcCB = (TF1*)timeCBFile->Get("back_CB");

    TF1 *smearCB = (TF1*)dataCB->Clone("smear_cb");
    smearCB->SetParameter(0,dataCB->GetParameter(0));
    smearCB->SetParameter(1,dataCB->GetParameter(1) - mcCB->GetParameter(1));
    smearCB->SetParameter(2, sqrt( abs( pow(dataCB->GetParameter(2),2) - pow(mcCB->GetParameter(2),2) )) );
    smearCB->SetParameter(3,dataCB->GetParameter(3));
    smearCB->SetParameter(4,dataCB->GetParameter(4));


    // Input variables
    /*


HERE the variables

    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    float EventWeight;
    float PUWeight;
    float PUReWeight;
    float PUReWeightUp;
    float PUReWeightDown;
    float TriggerWeight;
    bool isMC, isCosmic, isDT_fit, isCosmicVetoWithTags, isPreselected;
    int MeanNumInteractions;
    bool HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    bool Flag2_globalSuperTightHalo2016Filter, Flag2_goodVertices, Flag2_EcalDeadCellTriggerPrimitiveFilter, Flag2_HBHENoiseFilter, Flag2_HBHEIsoNoiseFilter, Flag2_ecalBadCalibFilter, Flag2_eeBadScFilter, Flag2_BadPFMuonFilter;
    float HT, MT, dPhi, dR_LLPs, dR_Higgs, dR_Gravitinos, dPhi_LLPs, dPhi_Higgs, dPhi_Gravitinos, dR_Gravitino_Higgs_0, dR_Gravitino_Higgs_1, dR_Gravitino_GenMet_0, dR_Gravitino_GenMet_1, dPhi_Gravitino_Met_0, dPhi_Gravitino_Met_1, dPhi_Gravitino_GenMet_0, dPhi_Gravitino_GenMet_1, dR_LLP_GenMet_0, dR_LLP_GenMet_1, dPhi_LLP_Met_0, dPhi_LLP_Met_1, dPhi_LLP_GenMet_0, dPhi_LLP_GenMet_1, dR_Higgs_GenMet_0, dR_Higgs_GenMet_1, dPhi_Higgs_Met_0, dPhi_Higgs_Met_1, dPhi_Higgs_GenMet_0, dPhi_Higgs_GenMet_1, DiGravitino_pt, DiGravitino_mass, DiGravitino_eta, DiGravitino_phi, dR_DiGravitino_GenMet, dPhi_DiGravitino_GenMet, dPhi_DiGravitino_Met, dPhi_DiGravitino_Higgs_0, dPhi_DiGravitino_Higgs_1, dPhi_Gravitino_0_Higgs_0, dPhi_Gravitino_1_Higgs_1, perc_met_held_by_gravitinos;
    int nLeptons;
    float MinJetMetDPhi;
    int nPV;
    int nCHSJets, nCHSJetsNegative, nCHSJetsFullEta, nCHSJetsMatch, nCHSJetsAcceptanceMatch, nElectrons, nMuons, nPhotons, nTausPreVeto, nTaus, nElectronsPassing, nMuonsPassing, nPhotonsPassing, nPhotonsTight, nTausPassing, nDTSegments, nCSCSegments, nCosmicMuons, nCosmicMuonsOneLeg;
    float dt_ecal_no_tag_dist, dt_ecal_acc_no_tag_dist, dt_ecal_dist, dt_ecal_acc_dist;
    float min_dPhi_jets_0p996, min_dPhi_jets_eta_1p0_0p996;
    int m_chi, ctau;
    inputTree->SetBranchAddress("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    inputTree->SetBranchAddress("nLLP_b_acc", &nLLP_b_acc, "nLLP_b_acc/I");
    inputTree->SetBranchAddress("nLLPInCalo_b_acc", &nLLPInCalo_b_acc, "nLLPInCalo_b_acc/I");
    inputTree->SetBranchAddress("isLLP0_b_acc",     &isLLP0_b_acc,      "isLLP0_b_acc/B");
    inputTree->SetBranchAddress("isLLP1_b_acc",     &isLLP1_b_acc,      "isLLP1_b_acc/B");
    inputTree->SetBranchAddress("isLLP0InCalo_b_acc",     &isLLP0InCalo_b_acc,      "isLLP0InCalo_b_acc/B");
    inputTree->SetBranchAddress("isLLP1InCalo_b_acc",     &isLLP1InCalo_b_acc,      "isLLP1InCalo_b_acc/B");
    inputTree->SetBranchAddress("isLLP0_boost",     &isLLP0_boost,      "isLLP0_boost/B");
    inputTree->SetBranchAddress("isLLP1_boost",     &isLLP1_boost,      "isLLP1_boost/B");
    inputTree->SetBranchAddress("isLLP0_res",     &isLLP0_res,      "isLLP0_res/B");
    inputTree->SetBranchAddress("isLLP1_res",     &isLLP1_res,      "isLLP1_res/B");
    inputTree->SetBranchAddress("nJets_to_LLP0",     &nJets_to_LLP0,      "nJets_to_LLP0/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0",  &nTagJets_to_LLP0,   "nTagJets_to_LLP0/I");
    inputTree->SetBranchAddress("nJets_to_LLP1",     &nJets_to_LLP1,      "nJets_to_LLP1/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1",  &nTagJets_to_LLP1,   "nTagJets_to_LLP1/I");
    inputTree->SetBranchAddress("nJets_to_LLP0_b_acc",     &nJets_to_LLP0_b_acc,      "nJets_to_LLP0_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0_b_acc",  &nTagJets_to_LLP0_b_acc,   "nTagJets_to_LLP0_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP1_b_acc",     &nJets_to_LLP1_b_acc,      "nJets_to_LLP1_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1_b_acc",  &nTagJets_to_LLP1_b_acc,   "nTagJets_to_LLP1_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP0InCalo",     &nJets_to_LLP0InCalo,      "nJets_to_LLP0InCalo/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0InCalo",  &nTagJets_to_LLP0InCalo,   "nTagJets_to_LLP0InCalo/I");
    inputTree->SetBranchAddress("nJets_to_LLP1InCalo",     &nJets_to_LLP1InCalo,      "nJets_to_LLP1InCalo/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1InCalo",  &nTagJets_to_LLP1InCalo,   "nTagJets_to_LLP1InCalo/I");
    inputTree->SetBranchAddress("nJets_to_LLP0InCalo_b_acc",     &nJets_to_LLP0InCalo_b_acc,      "nJets_to_LLP0InCalo_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0InCalo_b_acc",  &nTagJets_to_LLP0InCalo_b_acc,   "nTagJets_to_LLP0InCalo_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP1InCalo_b_acc",     &nJets_to_LLP1InCalo_b_acc,      "nJets_to_LLP1InCalo_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1InCalo_b_acc",  &nTagJets_to_LLP1InCalo_b_acc,   "nTagJets_to_LLP1InCalo_b_acc/I");

    */

    ////
    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    float    EventWeight;
    float    PUWeight;
    bool   isMC;
    int    MeanNumInteractions;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v(false);
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v(false);
    bool   HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v(false);
    bool   HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v(false);

    bool   Flag2_globalSuperTightHalo2016Filter;
    bool   Flag2_goodVertices;
    bool   Flag2_EcalDeadCellTriggerPrimitiveFilter;
    bool   Flag2_HBHENoiseFilter;
    bool   Flag2_HBHEIsoNoiseFilter;
    bool   Flag2_ecalBadCalibFilter;
    bool   Flag2_eeBadScFilter;
    bool   Flag2_BadPFMuonFilter;
    float  HT;
    float  MinJetMetDPhi_ntuple;

    Long64_t nPV;
    Long64_t nDTSegments;
    Long64_t nCSCSegments;
    Long64_t nCosmicMuons, nCosmicMuonsOneLeg;
    int    nElectrons;
    int    nMuons;
    int    nPhotons;
    //int    nTaus;
    int    nPFCandidates;
    int    nPFCandidatesTrack;
    int    nLLPInCalo;
    int    m_chi;
    int    ctau;
    bool   is_central;
    std::vector<TauType>         *Taus = 0;
    std::vector<PhotonType>      *Photons = 0;
    std::vector<LeptonType>      *Muons = 0;
    std::vector<LeptonType>      *Electrons = 0;
    std::vector<JetType>         *Jets = 0;
    std::vector<ecalRecHitType>  *EcalRecHitsAK4 = 0;
    MEtType                      *MEt = 0;
    std::vector<GenPType>        *GenHiggs = 0;
    std::vector<GenPType>        *GenLLPs = 0;
    std::vector<GenPType>        *GenBquarks = 0;
    std::vector<GenPType>        *GenGravitinos = 0;
    std::vector<DT4DSegmentType> *DTSegments = 0;
    std::vector<CSCSegmentType>  *CSCSegments = 0;

    // Input branches
    /*

HERE THE BRANCHES

     */

    TBranch        *b_Taus = 0;
    TBranch        *b_Photons = 0;
    TBranch        *b_Muons = 0;
    TBranch        *b_Electrons = 0;
    TBranch        *b_Jets = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_GenGravitinos = 0;
    TBranch        *b_EcalRecHitsAK4 = 0;
    TBranch        *b_DTSegments = 0;
    TBranch        *b_CSCSegments = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_PUWeight;
    TBranch        *b_isMC;
    TBranch        *b_MeanNumInteractions;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;


    TBranch        *b_Flag2_globalSuperTightHalo2016Filter;
    TBranch        *b_Flag2_goodVertices;
    TBranch        *b_Flag2_EcalDeadCellTriggerPrimitiveFilter;
    TBranch        *b_Flag2_HBHENoiseFilter;
    TBranch        *b_Flag2_HBHEIsoNoiseFilter;
    TBranch        *b_Flag2_ecalBadCalibFilter;
    TBranch        *b_Flag2_eeBadScFilter;
    TBranch        *b_Flag2_BadPFMuonFilter;
    TBranch        *b_HT;
    TBranch        *b_MinJetMetDPhi;
    //TBranch        *b_nCHSJets;
    TBranch        *b_nPV;
    TBranch        *b_nDTSegments;
    TBranch        *b_nCSCSegments;
    TBranch        *b_nCosmicMuons, *b_nCosmicMuonsOneLeg;
    TBranch        *b_nElectrons;
    TBranch        *b_nMuons;
    TBranch        *b_nPhotons;
    //TBranch        *b_nTaus;
    TBranch        *b_nPFCandidates;
    TBranch        *b_nPFCandidatesTrack;
    TBranch        *b_nLLPInCalo;
    TBranch        *b_m_chi;
    TBranch        *b_ctau;
    TBranch        *b_is_central;




    /*
      
      All the inputs I want

    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       "EventNumber/L");
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         "RunNumber/L");
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       "EventWeight/F");
    inputTree->SetBranchAddress("PUWeight",          &PUWeight,          "PUWeight/F");
    inputTree->SetBranchAddress("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    inputTree->SetBranchAddress("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    inputTree->SetBranchAddress("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    inputTree->SetBranchAddress("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    inputTree->SetBranchAddress("isMC",              &isMC,              "isMC/O");
    inputTree->SetBranchAddress("isCosmic",          &isCosmic,          "isCosmic/O");
    inputTree->SetBranchAddress("isDT_fit",          &isDT_fit,          "isDT_fit/O");
    inputTree->SetBranchAddress("isCosmicVetoWithTags", &isCosmicVetoWithTags, "isCosmicVetoWithTags/O");
    inputTree->SetBranchAddress("isPreselected",     &isPreselected,     "isPreselected/O");
    inputTree->SetBranchAddress("MeanNumInteractions",             &MeanNumInteractions,             "MeanNumInteractions/I");
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");
    inputTree->SetBranchAddress("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, "Flag2_globalSuperTightHalo2016Filter/O");
    inputTree->SetBranchAddress("Flag2_goodVertices", &Flag2_goodVertices, "Flag2_goodVertices/O");
    inputTree->SetBranchAddress("Flag2_EcalDeadCellTriggerPrimitiveFilter", &Flag2_EcalDeadCellTriggerPrimitiveFilter, "Flag2_EcalDeadCellTriggerPrimitiveFilter/O");
    inputTree->SetBranchAddress("Flag2_HBHENoiseFilter", &Flag2_HBHENoiseFilter, "Flag2_HBHENoiseFilter/O");
    inputTree->SetBranchAddress("Flag2_HBHEIsoNoiseFilter", &Flag2_HBHEIsoNoiseFilter, "Flag2_HBHEIsoNoiseFilter/O");
    inputTree->SetBranchAddress("Flag2_ecalBadCalibFilter", &Flag2_ecalBadCalibFilter, "Flag2_ecalBadCalibFilter/O");
    inputTree->SetBranchAddress("Flag2_eeBadScFilter", &Flag2_eeBadScFilter, "Flag2_eeBadScFilter/O");
    inputTree->SetBranchAddress("Flag2_BadPFMuonFilter", &Flag2_BadPFMuonFilter, "Flag2_BadPFMuonFilter/O");
    inputTree->SetBranchAddress("HT",                &HT,                "HT/F");
    inputTree->SetBranchAddress("MT",                &MT,                "MT/F");
    inputTree->SetBranchAddress("dPhi",              &dPhi,              "dPhi/F");
    inputTree->SetBranchAddress("dR_LLPs", &dR_LLPs, "dR_LLPs/F");
    inputTree->SetBranchAddress("dR_Higgs", &dR_Higgs, "dR_Higgs/F");
    inputTree->SetBranchAddress("dR_Gravitinos", &dR_Gravitinos, "dR_Gravitinos/F");
    inputTree->SetBranchAddress("dPhi_LLPs", &dPhi_LLPs, "dPhi_LLPs/F");
    inputTree->SetBranchAddress("dPhi_Higgs", &dPhi_Higgs, "dPhi_Higgs/F");
    inputTree->SetBranchAddress("dPhi_Gravitinos", &dPhi_Gravitinos, "dPhi_Gravitinos/F");
    inputTree->SetBranchAddress("dR_Gravitino_Higgs_0", &dR_Gravitino_Higgs_0, "dR_Gravitino_Higgs_0/F");
    inputTree->SetBranchAddress("dR_Gravitino_Higgs_1", &dR_Gravitino_Higgs_1, "dR_Gravitino_Higgs_1/F");
    inputTree->SetBranchAddress("dR_Gravitino_GenMet_0", &dR_Gravitino_GenMet_0, "dR_Gravitino_GenMet_0/F");
    inputTree->SetBranchAddress("dR_Gravitino_GenMet_1", &dR_Gravitino_GenMet_1, "dR_Gravitino_GenMet_1/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_Met_0", &dPhi_Gravitino_Met_0, "dPhi_Gravitino_Met_0/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_Met_1", &dPhi_Gravitino_Met_1, "dPhi_Gravitino_Met_1/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_GenMet_0", &dPhi_Gravitino_GenMet_0, "dPhi_Gravitino_GenMet_0/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_GenMet_1", &dPhi_Gravitino_GenMet_1, "dPhi_Gravitino_GenMet_1/F");
    inputTree->SetBranchAddress("dR_LLP_GenMet_0", &dR_LLP_GenMet_0, "dR_LLP_GenMet_0/F");
    inputTree->SetBranchAddress("dR_LLP_GenMet_1", &dR_LLP_GenMet_1, "dR_LLP_GenMet_1/F");
    inputTree->SetBranchAddress("dPhi_LLP_Met_0", &dPhi_LLP_Met_0, "dPhi_LLP_Met_0/F");
    inputTree->SetBranchAddress("dPhi_LLP_Met_1", &dPhi_LLP_Met_1, "dPhi_LLP_Met_1/F");
    inputTree->SetBranchAddress("dPhi_LLP_GenMet_0", &dPhi_LLP_GenMet_0, "dPhi_LLP_GenMet_0/F");
    inputTree->SetBranchAddress("dPhi_LLP_GenMet_1", &dPhi_LLP_GenMet_1, "dPhi_LLP_GenMet_1/F");
    inputTree->SetBranchAddress("dR_Higgs_GenMet_0", &dR_Higgs_GenMet_0, "dR_Higgs_GenMet_0/F");
    inputTree->SetBranchAddress("dR_Higgs_GenMet_1", &dR_Higgs_GenMet_1, "dR_Higgs_GenMet_1/F");
    inputTree->SetBranchAddress("dPhi_Higgs_Met_0", &dPhi_Higgs_Met_0, "dPhi_Higgs_Met_0/F");
    inputTree->SetBranchAddress("dPhi_Higgs_Met_1", &dPhi_Higgs_Met_1, "dPhi_Higgs_Met_1/F");
    inputTree->SetBranchAddress("dPhi_Higgs_GenMet_0", &dPhi_Higgs_GenMet_0, "dPhi_Higgs_GenMet_0/F");
    inputTree->SetBranchAddress("dPhi_Higgs_GenMet_1", &dPhi_Higgs_GenMet_1, "dPhi_Higgs_GenMet_1/F");
    inputTree->SetBranchAddress("DiGravitino_pt", &DiGravitino_pt, "DiGravitino_pt/F");
    inputTree->SetBranchAddress("DiGravitino_mass", &DiGravitino_mass, "DiGravitino_mass/F");
    inputTree->SetBranchAddress("DiGravitino_eta", &DiGravitino_eta, "DiGravitino_eta/F");
    inputTree->SetBranchAddress("DiGravitino_phi", &DiGravitino_phi, "DiGravitino_phi/F");
    inputTree->SetBranchAddress("dR_DiGravitino_GenMet", &dR_DiGravitino_GenMet, "dR_DiGravitino_GenMet/F");
    inputTree->SetBranchAddress("dPhi_DiGravitino_GenMet", &dPhi_DiGravitino_GenMet, "dPhi_DiGravitino_GenMet/F");
    inputTree->SetBranchAddress("dPhi_DiGravitino_Met", &dPhi_DiGravitino_Met, "dPhi_DiGravitino_Met/F");
    inputTree->SetBranchAddress("dPhi_DiGravitino_Higgs_0", &dPhi_DiGravitino_Higgs_0, "dPhi_DiGravitino_Higgs_0/F");
    inputTree->SetBranchAddress("dPhi_DiGravitino_Higgs_1", &dPhi_DiGravitino_Higgs_1, "dPhi_DiGravitino_Higgs_1/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_0_Higgs_0", &dPhi_Gravitino_0_Higgs_0, "dPhi_Gravitino_0_Higgs_0/F");
    inputTree->SetBranchAddress("dPhi_Gravitino_1_Higgs_1", &dPhi_Gravitino_1_Higgs_1, "dPhi_Gravitino_1_Higgs_1/F");
    inputTree->SetBranchAddress("perc_met_held_by_gravitinos", &perc_met_held_by_gravitinos, "perc_met_held_by_gravitinos/F");
    inputTree->SetBranchAddress("nLeptons", &nLeptons, "nLeptons/I");
    inputTree->SetBranchAddress("MinJetMetDPhi_ntuple",     &MinJetMetDPhi_ntuple,     "MinJetMetDPhi_ntuple/F");
    inputTree->SetBranchAddress("MinJetMetDPhi",  &MinJetMetDPhi,  "MinJetMetDPhi/F");
    inputTree->SetBranchAddress("nPV",          &nPV,          "nPV/I");
    inputTree->SetBranchAddress("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    inputTree->SetBranchAddress("nCHSJetsNegative",          &nCHSJetsNegative,          "nCHSJetsNegative/I");
    inputTree->SetBranchAddress("nCHSJetsFullEta",          &nCHSJetsFullEta,          "nCHSJetsFullEta/I");
    inputTree->SetBranchAddress("nCHSJetsMatch",          &nCHSJetsMatch,          "nCHSJetsMatch/I");
    inputTree->SetBranchAddress("nCHSJetsAcceptanceMatch",          &nCHSJetsAcceptanceMatch,          "nCHSJetsAcceptanceMatch/I");
    inputTree->SetBranchAddress("nElectrons",        &nElectrons,        "nElectrons/I");
    inputTree->SetBranchAddress("nMuons",            &nMuons,            "nMuons/I");
    inputTree->SetBranchAddress("nPhotons",          &nPhotons,          "nPhotons/I");
    inputTree->SetBranchAddress("nTausPreVeto",      &nTausPreVeto,      "nTausPreVeto/I");
    inputTree->SetBranchAddress("nTaus",             &nTaus,             "nTaus/I");
    inputTree->SetBranchAddress("nElectronsPassing",        &nElectronsPassing,        "nElectronsPassing/I");
    inputTree->SetBranchAddress("nMuonsPassing",            &nMuonsPassing,            "nMuonsPassing/I");
    inputTree->SetBranchAddress("nPhotonsPassing",          &nPhotonsPassing,          "nPhotonsPassing/I");
    inputTree->SetBranchAddress("nPhotonsTight",          &nPhotonsTight,          "nPhotonsTight/I");
    inputTree->SetBranchAddress("nTausPassing",             &nTausPassing,             "nTausPassing/I");
    inputTree->SetBranchAddress("nDTSegments",       &nDTSegments,       "nDTSegments/I");
    inputTree->SetBranchAddress("nCSCSegments",      &nCSCSegments,      "nCSCSegments/I");
    inputTree->SetBranchAddress("nCosmicMuons",      &nCosmicMuons,      "nCosmicMuons/I");
    inputTree->SetBranchAddress("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      "nCosmicMuonsOneLeg/I");
    inputTree->SetBranchAddress("dt_ecal_no_tag_dist", &dt_ecal_no_tag_dist, "dt_ecal_no_tag_dist/F");
    inputTree->SetBranchAddress("dt_ecal_acc_no_tag_dist", &dt_ecal_acc_no_tag_dist, "dt_ecal_acc_no_tag_dist/F");
    inputTree->SetBranchAddress("dt_ecal_dist", &dt_ecal_dist, "dt_ecal_dist/F");
    inputTree->SetBranchAddress("dt_ecal_acc_dist", &dt_ecal_acc_dist, "dt_ecal_acc_dist/F");
    inputTree->SetBranchAddress("min_dPhi_jets_0p996", &min_dPhi_jets_0p996, "min_dPhi_jets_0p996/F");
    inputTree->SetBranchAddress("min_dPhi_jets_eta_1p0_0p996", &min_dPhi_jets_eta_1p0_0p996, "min_dPhi_jets_eta_1p0_0p996/F");
    inputTree->SetBranchAddress("m_chi", &m_chi, "m_chi/I");
    inputTree->SetBranchAddress("ctau", &ctau, "ctau/I");
    inputTree->SetBranchAddress("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    inputTree->SetBranchAddress("nLLP_b_acc", &nLLP_b_acc, "nLLP_b_acc/I");
    inputTree->SetBranchAddress("nLLPInCalo_b_acc", &nLLPInCalo_b_acc, "nLLPInCalo_b_acc/I");
    inputTree->SetBranchAddress("isLLP0_b_acc",     &isLLP0_b_acc,      "isLLP0_b_acc/B");
    inputTree->SetBranchAddress("isLLP1_b_acc",     &isLLP1_b_acc,      "isLLP1_b_acc/B");
    inputTree->SetBranchAddress("isLLP0InCalo_b_acc",     &isLLP0InCalo_b_acc,      "isLLP0InCalo_b_acc/B");
    inputTree->SetBranchAddress("isLLP1InCalo_b_acc",     &isLLP1InCalo_b_acc,      "isLLP1InCalo_b_acc/B");
    inputTree->SetBranchAddress("isLLP0_boost",     &isLLP0_boost,      "isLLP0_boost/B");
    inputTree->SetBranchAddress("isLLP1_boost",     &isLLP1_boost,      "isLLP1_boost/B");
    inputTree->SetBranchAddress("isLLP0_res",     &isLLP0_res,      "isLLP0_res/B");
    inputTree->SetBranchAddress("isLLP1_res",     &isLLP1_res,      "isLLP1_res/B");
    inputTree->SetBranchAddress("nJets_to_LLP0",     &nJets_to_LLP0,      "nJets_to_LLP0/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0",  &nTagJets_to_LLP0,   "nTagJets_to_LLP0/I");
    inputTree->SetBranchAddress("nJets_to_LLP1",     &nJets_to_LLP1,      "nJets_to_LLP1/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1",  &nTagJets_to_LLP1,   "nTagJets_to_LLP1/I");
    inputTree->SetBranchAddress("nJets_to_LLP0_b_acc",     &nJets_to_LLP0_b_acc,      "nJets_to_LLP0_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0_b_acc",  &nTagJets_to_LLP0_b_acc,   "nTagJets_to_LLP0_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP1_b_acc",     &nJets_to_LLP1_b_acc,      "nJets_to_LLP1_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1_b_acc",  &nTagJets_to_LLP1_b_acc,   "nTagJets_to_LLP1_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP0InCalo",     &nJets_to_LLP0InCalo,      "nJets_to_LLP0InCalo/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0InCalo",  &nTagJets_to_LLP0InCalo,   "nTagJets_to_LLP0InCalo/I");
    inputTree->SetBranchAddress("nJets_to_LLP1InCalo",     &nJets_to_LLP1InCalo,      "nJets_to_LLP1InCalo/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1InCalo",  &nTagJets_to_LLP1InCalo,   "nTagJets_to_LLP1InCalo/I");
    inputTree->SetBranchAddress("nJets_to_LLP0InCalo_b_acc",     &nJets_to_LLP0InCalo_b_acc,      "nJets_to_LLP0InCalo_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP0InCalo_b_acc",  &nTagJets_to_LLP0InCalo_b_acc,   "nTagJets_to_LLP0InCalo_b_acc/I");
    inputTree->SetBranchAddress("nJets_to_LLP1InCalo_b_acc",     &nJets_to_LLP1InCalo_b_acc,      "nJets_to_LLP1InCalo_b_acc/I");
    inputTree->SetBranchAddress("nTagJets_to_LLP1InCalo_b_acc",  &nTagJets_to_LLP1InCalo_b_acc,   "nTagJets_to_LLP1InCalo_b_acc/I");
    //
    inputTree->SetBranchAddress("JetsFullEta", &skimmedJetsFullEta);
    inputTree->SetBranchAddress("Jets", &skimmedJets);
    inputTree->SetBranchAddress("JetsNegative", &skimmedJetsNegative);
    inputTree->SetBranchAddress("JetsMatch", &skimmedJetsMatch);
    inputTree->SetBranchAddress("JetsAcceptanceMatch", &skimmedJetsAcceptanceMatch);
    inputTree->SetBranchAddress("MEt", &MEt);
    inputTree->SetBranchAddress("GenHiggs", &GenHiggs);
    inputTree->SetBranchAddress("GenLLPs", &GenLLPs);
    inputTree->SetBranchAddress("GenBquarks", &GenBquarks);
    inputTree->SetBranchAddress("GenGravitinos", &GenGravitinos);

     */

    inputTree->SetBranchAddress("Taus",              &Taus,              &b_Taus);
    inputTree->SetBranchAddress("Photons",           &Photons,           &b_Photons);
    inputTree->SetBranchAddress("Muons",             &Muons,             &b_Muons);
    inputTree->SetBranchAddress("Electrons",         &Electrons,         &b_Electrons);
    inputTree->SetBranchAddress("Jets",              &Jets,              &b_Jets);
    inputTree->SetBranchAddress("EcalRecHitsAK4",    &EcalRecHitsAK4,    &b_EcalRecHitsAK4);
    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt); 
    inputTree->SetBranchAddress("GenHiggs",          &GenHiggs,          &b_GenHiggs); 
    inputTree->SetBranchAddress("GenLLPs",           &GenLLPs,           &b_GenLLPs); 
    inputTree->SetBranchAddress("GenBquarks",        &GenBquarks,        &b_GenBquarks); 
    inputTree->SetBranchAddress("GenGravitinos",     &GenGravitinos,     &b_GenGravitinos);
    inputTree->SetBranchAddress("DTSegments",        &DTSegments,        &b_DTSegments); 
    inputTree->SetBranchAddress("CSCSegments",       &CSCSegments,       &b_CSCSegments); 
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("PUWeight",          &PUWeight,          &b_PUWeight);
    inputTree->SetBranchAddress("isMC",              &isMC,              &b_isMC);
    inputTree->SetBranchAddress("MeanNumInteractions",  &MeanNumInteractions,  &b_MeanNumInteractions);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v);


    inputTree->SetBranchAddress("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, &b_Flag2_globalSuperTightHalo2016Filter);
    inputTree->SetBranchAddress("Flag2_goodVertices", &Flag2_goodVertices, &b_Flag2_goodVertices);
    inputTree->SetBranchAddress("Flag2_EcalDeadCellTriggerPrimitiveFilter", &Flag2_EcalDeadCellTriggerPrimitiveFilter, &b_Flag2_EcalDeadCellTriggerPrimitiveFilter);
    inputTree->SetBranchAddress("Flag2_HBHENoiseFilter", &Flag2_HBHENoiseFilter, &b_Flag2_HBHENoiseFilter);
    inputTree->SetBranchAddress("Flag2_HBHEIsoNoiseFilter", &Flag2_HBHEIsoNoiseFilter, &b_Flag2_HBHEIsoNoiseFilter);
    inputTree->SetBranchAddress("Flag2_ecalBadCalibFilter", &Flag2_ecalBadCalibFilter, &b_Flag2_ecalBadCalibFilter);
    inputTree->SetBranchAddress("Flag2_eeBadScFilter", &Flag2_eeBadScFilter, &b_Flag2_eeBadScFilter);
    inputTree->SetBranchAddress("Flag2_BadPFMuonFilter", &Flag2_BadPFMuonFilter, &b_Flag2_BadPFMuonFilter);

    inputTree->SetBranchAddress("HT",                &HT,                &b_HT);
    inputTree->SetBranchAddress("MinJetMetDPhi",     &MinJetMetDPhi_ntuple,     &b_MinJetMetDPhi);
    inputTree->SetBranchAddress("nPV",          &nPV,          &b_nPV);
    //inputTree->SetBranchAddress("nCHSJets",          &nCHSJets,          &b_nCHSJets);
    inputTree->SetBranchAddress("nElectrons",        &nElectrons,        &b_nElectrons);
    inputTree->SetBranchAddress("nMuons",            &nMuons,            &b_nMuons);
    inputTree->SetBranchAddress("nPhotons",          &nPhotons,          &b_nPhotons);
    inputTree->SetBranchAddress("nDTSegments",       &nDTSegments,       &b_nDTSegments);
    inputTree->SetBranchAddress("nCSCSegments",      &nCSCSegments,      &b_nCSCSegments);
    inputTree->SetBranchAddress("nCosmicMuons",      &nCosmicMuons,      &b_nCosmicMuons);
    inputTree->SetBranchAddress("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      &b_nCosmicMuonsOneLeg);
    //inputTree->SetBranchAddress("nTaus",             &nTaus,             &b_nTaus);
    inputTree->SetBranchAddress("nPFCandidates",     &nPFCandidates,     &b_nPFCandidates);
    inputTree->SetBranchAddress("nPFCandidatesTrack", &nPFCandidatesTrack, &b_nPFCandidatesTrack);
    inputTree->SetBranchAddress("nLLPInCalo", &nLLPInCalo, &b_nLLPInCalo);
    inputTree->SetBranchAddress("m_chi", &m_chi, &b_m_chi);
    inputTree->SetBranchAddress("ctau", &ctau, &b_ctau);
    inputTree->SetBranchAddress("is_central", &is_central, &b_is_central);
    //inputTree->SetBranchStatus("*",0);

    // Read jet input features from metadata.dat file
    //AK4
    std::ifstream finAK4;
    std::string featAK4;
    finAK4.open(MetaDataFileAK4);
    std::vector<std::string> featuresAK4;
    std::string toEraseAK4 = "Jet_";
    //std::cout << "   -- > Features AK4: " << std::endl;
    while (finAK4 >> featAK4)
      {
	size_t pos = featAK4.find(toEraseAK4);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    featAK4.erase(pos, toEraseAK4.length());
	  }
	//std::string new_feat = featAK4.substr(position);
	//std::cout << featAK4 << std::endl;
	featuresAK4.push_back(featAK4);
      }
    finAK4.close();


    // ================= 
    // Output
    // ================= 

    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");


    //Flags for SR/CR
    bool isPreselected(false);

    bool isCosmic(false);
    bool isDT_fit(false);
    bool isCosmicVetoWithTags(false);

    std::vector<TauType>    skimmedTaus;

    std::vector<JetType>    skimmedJetsFullEta;
    std::vector<JetType>    skimmedJets;
    std::vector<JetType>    skimmedJetsNegative;
    std::vector<JetType>    skimmedJetsMatch;
    std::vector<JetType>    skimmedJetsAcceptanceMatch;

    //Additional output variables
    //DT fit
    float dt_ecal_dist(9999.);
    float dt_ecal_no_tag_dist(9999.);
    float dt_ecal_acc_no_tag_dist(9999.);
    float dt_ecal_acc_dist(9999.);

    //Beam Halo
    float min_dPhi_jets_0p996(9999.);
    float min_dPhi_jets_eta_1p0_0p996(9999.);


    float PUReWeight(1.);
    float PUReWeightUp(1.);
    float PUReWeightDown(1.);
    float TriggerWeight(1.);
    float MinJetMetDPhi(10.);

    float dPhi(-9.);
    float MT(-1.);

    //Gen level studies
    float dR_LLPs(-9.);
    float dR_Higgs(-9.);
    float dR_Gravitinos(-9.);

    float dPhi_LLPs(-9.);
    float dPhi_Higgs(-9.);
    float dPhi_Gravitinos(-9.);

    float dR_Gravitino_Higgs_0(-9.);
    float dR_Gravitino_Higgs_1(-9.);
    float dR_Gravitino_GenMet_0(-9.);
    float dR_Gravitino_GenMet_1(-9.);
    float dPhi_Gravitino_GenMet_0(-9.);
    float dPhi_Gravitino_GenMet_1(-9.);
    float dPhi_Gravitino_Met_0(-9.);
    float dPhi_Gravitino_Met_1(-9.);
    float dR_LLP_GenMet_0(-9.);
    float dR_LLP_GenMet_1(-9.);
    float dPhi_LLP_Met_0(-9.);
    float dPhi_LLP_Met_1(-9.);
    float dPhi_LLP_GenMet_0(-9.);
    float dPhi_LLP_GenMet_1(-9.);
    float dR_Higgs_GenMet_0(-9.);
    float dR_Higgs_GenMet_1(-9.);
    float dPhi_Higgs_Met_0(-9.);
    float dPhi_Higgs_Met_1(-9.);
    float dPhi_Higgs_GenMet_0(-9.);
    float dPhi_Higgs_GenMet_1(-9.);
    float DiGravitino_pt(-1.);
    float DiGravitino_mass(-1.);
    float DiGravitino_eta(-1.);
    float DiGravitino_phi(-1.);
    float dR_DiGravitino_GenMet(-9.);
    float dPhi_DiGravitino_Met(-9.);
    float dPhi_DiGravitino_GenMet(-9.);
    float dPhi_DiGravitino_Higgs_0(-9.);
    float dPhi_DiGravitino_Higgs_1(-9.);
    float dPhi_Gravitino_0_Higgs_0(-9.);
    float dPhi_Gravitino_1_Higgs_1(-9.);
    float perc_met_held_by_gravitinos(-1.);


    int nLeptons(0);

    int nTaus(0);
    int nTausPreVeto(0);
    //int nPhotons(0);
    //int nMuons(0);
    int nMuonsPassing(0);
    int nElectronsPassing(0);
    int nPhotonsPassing(0);
    int nPhotonsTight(0);
    int nTausPassing(0);

    int nCHSJets;
    int nCHSJetsNegative;
    int nCHSJetsFullEta;
    int nCHSJetsMatch;
    int nCHSJetsAcceptanceMatch;
    //HEM thing can be discarded, as we are not looking at data

    int nTagJets_0p996(0);
    //int nTagJets_0p996_unsmeared(0);


    //TBD
    int  nLLP_b_acc(0);
    int  nLLPInCalo_b_acc(0);

    bool isLLP0_b_acc (false);
    bool isLLP1_b_acc (false);
    bool isLLP0InCalo_b_acc (false);
    bool isLLP1InCalo_b_acc (false);

    bool isLLP0_boost (false);
    bool isLLP1_boost (false);
    bool isLLP0_res (false);
    bool isLLP1_res (false);

    int nJets_to_LLP0 (0);
    int nTagJets_to_LLP0 (0);
    int nJets_to_LLP1 (0);
    int nTagJets_to_LLP1 (0);

    int nJets_to_LLP0_b_acc (0);
    int nTagJets_to_LLP0_b_acc (0);
    int nJets_to_LLP1_b_acc (0);
    int nTagJets_to_LLP1_b_acc (0);

    int nJets_to_LLP0InCalo (0);
    int nTagJets_to_LLP0InCalo (0);
    int nJets_to_LLP1InCalo (0);
    int nTagJets_to_LLP1InCalo (0);

    int nJets_to_LLP0InCalo_b_acc (0);
    int nTagJets_to_LLP0InCalo_b_acc (0);
    int nJets_to_LLP1InCalo_b_acc (0);
    int nTagJets_to_LLP1InCalo_b_acc (0);



    // Output branches 
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("RunNumber",         &RunNumber,         "RunNumber/L");
    outputTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    outputTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
    outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    outputTree->Branch("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    outputTree->Branch("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    outputTree->Branch("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isCosmic",          &isCosmic,          "isCosmic/O");
    outputTree->Branch("isDT_fit",          &isDT_fit,          "isDT_fit/O");
    outputTree->Branch("isCosmicVetoWithTags", &isCosmicVetoWithTags, "isCosmicVetoWithTags/O");
    outputTree->Branch("isPreselected",     &isPreselected,     "isPreselected/O");
    outputTree->Branch("MeanNumInteractions",             &MeanNumInteractions,             "MeanNumInteractions/I");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");
    outputTree->Branch("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, "Flag2_globalSuperTightHalo2016Filter/O");
    outputTree->Branch("Flag2_goodVertices", &Flag2_goodVertices, "Flag2_goodVertices/O");
    outputTree->Branch("Flag2_EcalDeadCellTriggerPrimitiveFilter", &Flag2_EcalDeadCellTriggerPrimitiveFilter, "Flag2_EcalDeadCellTriggerPrimitiveFilter/O");
    outputTree->Branch("Flag2_HBHENoiseFilter", &Flag2_HBHENoiseFilter, "Flag2_HBHENoiseFilter/O");
    outputTree->Branch("Flag2_HBHEIsoNoiseFilter", &Flag2_HBHEIsoNoiseFilter, "Flag2_HBHEIsoNoiseFilter/O");
    outputTree->Branch("Flag2_ecalBadCalibFilter", &Flag2_ecalBadCalibFilter, "Flag2_ecalBadCalibFilter/O");
    outputTree->Branch("Flag2_eeBadScFilter", &Flag2_eeBadScFilter, "Flag2_eeBadScFilter/O");
    outputTree->Branch("Flag2_BadPFMuonFilter", &Flag2_BadPFMuonFilter, "Flag2_BadPFMuonFilter/O");
    outputTree->Branch("HT",                &HT,                "HT/F");
    outputTree->Branch("MT",                &MT,                "MT/F");
    outputTree->Branch("dPhi",              &dPhi,              "dPhi/F");
    outputTree->Branch("dR_LLPs", &dR_LLPs, "dR_LLPs/F");
    outputTree->Branch("dR_Higgs", &dR_Higgs, "dR_Higgs/F");
    outputTree->Branch("dR_Gravitinos", &dR_Gravitinos, "dR_Gravitinos/F");
    outputTree->Branch("dPhi_LLPs", &dPhi_LLPs, "dPhi_LLPs/F");
    outputTree->Branch("dPhi_Higgs", &dPhi_Higgs, "dPhi_Higgs/F");
    outputTree->Branch("dPhi_Gravitinos", &dPhi_Gravitinos, "dPhi_Gravitinos/F");
    outputTree->Branch("dR_Gravitino_Higgs_0", &dR_Gravitino_Higgs_0, "dR_Gravitino_Higgs_0/F");
    outputTree->Branch("dR_Gravitino_Higgs_1", &dR_Gravitino_Higgs_1, "dR_Gravitino_Higgs_1/F");
    outputTree->Branch("dR_Gravitino_GenMet_0", &dR_Gravitino_GenMet_0, "dR_Gravitino_GenMet_0/F");
    outputTree->Branch("dR_Gravitino_GenMet_1", &dR_Gravitino_GenMet_1, "dR_Gravitino_GenMet_1/F");
    outputTree->Branch("dPhi_Gravitino_Met_0", &dPhi_Gravitino_Met_0, "dPhi_Gravitino_Met_0/F");
    outputTree->Branch("dPhi_Gravitino_Met_1", &dPhi_Gravitino_Met_1, "dPhi_Gravitino_Met_1/F");
    outputTree->Branch("dPhi_Gravitino_GenMet_0", &dPhi_Gravitino_GenMet_0, "dPhi_Gravitino_GenMet_0/F");
    outputTree->Branch("dPhi_Gravitino_GenMet_1", &dPhi_Gravitino_GenMet_1, "dPhi_Gravitino_GenMet_1/F");
    outputTree->Branch("dR_LLP_GenMet_0", &dR_LLP_GenMet_0, "dR_LLP_GenMet_0/F");
    outputTree->Branch("dR_LLP_GenMet_1", &dR_LLP_GenMet_1, "dR_LLP_GenMet_1/F");
    outputTree->Branch("dPhi_LLP_Met_0", &dPhi_LLP_Met_0, "dPhi_LLP_Met_0/F");
    outputTree->Branch("dPhi_LLP_Met_1", &dPhi_LLP_Met_1, "dPhi_LLP_Met_1/F");
    outputTree->Branch("dPhi_LLP_GenMet_0", &dPhi_LLP_GenMet_0, "dPhi_LLP_GenMet_0/F");
    outputTree->Branch("dPhi_LLP_GenMet_1", &dPhi_LLP_GenMet_1, "dPhi_LLP_GenMet_1/F");
    outputTree->Branch("dR_Higgs_GenMet_0", &dR_Higgs_GenMet_0, "dR_Higgs_GenMet_0/F");
    outputTree->Branch("dR_Higgs_GenMet_1", &dR_Higgs_GenMet_1, "dR_Higgs_GenMet_1/F");
    outputTree->Branch("dPhi_Higgs_Met_0", &dPhi_Higgs_Met_0, "dPhi_Higgs_Met_0/F");
    outputTree->Branch("dPhi_Higgs_Met_1", &dPhi_Higgs_Met_1, "dPhi_Higgs_Met_1/F");
    outputTree->Branch("dPhi_Higgs_GenMet_0", &dPhi_Higgs_GenMet_0, "dPhi_Higgs_GenMet_0/F");
    outputTree->Branch("dPhi_Higgs_GenMet_1", &dPhi_Higgs_GenMet_1, "dPhi_Higgs_GenMet_1/F");
    outputTree->Branch("DiGravitino_pt", &DiGravitino_pt, "DiGravitino_pt/F");
    outputTree->Branch("DiGravitino_mass", &DiGravitino_mass, "DiGravitino_mass/F");
    outputTree->Branch("DiGravitino_eta", &DiGravitino_eta, "DiGravitino_eta/F");
    outputTree->Branch("DiGravitino_phi", &DiGravitino_phi, "DiGravitino_phi/F");
    outputTree->Branch("dR_DiGravitino_GenMet", &dR_DiGravitino_GenMet, "dR_DiGravitino_GenMet/F");
    outputTree->Branch("dPhi_DiGravitino_GenMet", &dPhi_DiGravitino_GenMet, "dPhi_DiGravitino_GenMet/F");
    outputTree->Branch("dPhi_DiGravitino_Met", &dPhi_DiGravitino_Met, "dPhi_DiGravitino_Met/F");
    outputTree->Branch("dPhi_DiGravitino_Higgs_0", &dPhi_DiGravitino_Higgs_0, "dPhi_DiGravitino_Higgs_0/F");
    outputTree->Branch("dPhi_DiGravitino_Higgs_1", &dPhi_DiGravitino_Higgs_1, "dPhi_DiGravitino_Higgs_1/F");
    outputTree->Branch("dPhi_Gravitino_0_Higgs_0", &dPhi_Gravitino_0_Higgs_0, "dPhi_Gravitino_0_Higgs_0/F");
    outputTree->Branch("dPhi_Gravitino_1_Higgs_1", &dPhi_Gravitino_1_Higgs_1, "dPhi_Gravitino_1_Higgs_1/F");
    outputTree->Branch("perc_met_held_by_gravitinos", &perc_met_held_by_gravitinos, "perc_met_held_by_gravitinos/F");
    outputTree->Branch("nLeptons", &nLeptons, "nLeptons/I");
    outputTree->Branch("MinJetMetDPhi_ntuple",     &MinJetMetDPhi_ntuple,     "MinJetMetDPhi_ntuple/F");
    outputTree->Branch("MinJetMetDPhi",  &MinJetMetDPhi,  "MinJetMetDPhi/F");
    outputTree->Branch("nPV",          &nPV,          "nPV/I");
    outputTree->Branch("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    outputTree->Branch("nCHSJetsNegative",          &nCHSJetsNegative,          "nCHSJetsNegative/I");
    outputTree->Branch("nCHSJetsFullEta",          &nCHSJetsFullEta,          "nCHSJetsFullEta/I");
    outputTree->Branch("nCHSJetsMatch",          &nCHSJetsMatch,          "nCHSJetsMatch/I");
    outputTree->Branch("nCHSJetsAcceptanceMatch",          &nCHSJetsAcceptanceMatch,          "nCHSJetsAcceptanceMatch/I");
    outputTree->Branch("nElectrons",        &nElectrons,        "nElectrons/I");
    outputTree->Branch("nMuons",            &nMuons,            "nMuons/I");
    outputTree->Branch("nPhotons",          &nPhotons,          "nPhotons/I");
    outputTree->Branch("nTausPreVeto",      &nTausPreVeto,      "nTausPreVeto/I");
    outputTree->Branch("nTaus",             &nTaus,             "nTaus/I");
    outputTree->Branch("nElectronsPassing",        &nElectronsPassing,        "nElectronsPassing/I");
    outputTree->Branch("nMuonsPassing",            &nMuonsPassing,            "nMuonsPassing/I");
    outputTree->Branch("nPhotonsPassing",          &nPhotonsPassing,          "nPhotonsPassing/I");
    outputTree->Branch("nPhotonsTight",          &nPhotonsTight,          "nPhotonsTight/I");
    outputTree->Branch("nTausPassing",             &nTausPassing,             "nTausPassing/I");
    outputTree->Branch("nDTSegments",       &nDTSegments,       "nDTSegments/I");
    outputTree->Branch("nCSCSegments",      &nCSCSegments,      "nCSCSegments/I");
    outputTree->Branch("nCosmicMuons",      &nCosmicMuons,      "nCosmicMuons/I");
    outputTree->Branch("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      "nCosmicMuonsOneLeg/I");
    outputTree->Branch("dt_ecal_no_tag_dist", &dt_ecal_no_tag_dist, "dt_ecal_no_tag_dist/F");
    outputTree->Branch("dt_ecal_acc_no_tag_dist", &dt_ecal_acc_no_tag_dist, "dt_ecal_acc_no_tag_dist/F");
    outputTree->Branch("dt_ecal_dist", &dt_ecal_dist, "dt_ecal_dist/F");
    outputTree->Branch("dt_ecal_acc_dist", &dt_ecal_acc_dist, "dt_ecal_acc_dist/F");
    outputTree->Branch("min_dPhi_jets_0p996", &min_dPhi_jets_0p996, "min_dPhi_jets_0p996/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p996", &min_dPhi_jets_eta_1p0_0p996, "min_dPhi_jets_eta_1p0_0p996/F");
    outputTree->Branch("m_chi", &m_chi, "m_chi/I");
    outputTree->Branch("ctau", &ctau, "ctau/I");
    outputTree->Branch("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    outputTree->Branch("nLLP_b_acc", &nLLP_b_acc, "nLLP_b_acc/I");
    outputTree->Branch("nLLPInCalo_b_acc", &nLLPInCalo_b_acc, "nLLPInCalo_b_acc/I");
    outputTree->Branch("isLLP0_b_acc",     &isLLP0_b_acc,      "isLLP0_b_acc/B");
    outputTree->Branch("isLLP1_b_acc",     &isLLP1_b_acc,      "isLLP1_b_acc/B");
    outputTree->Branch("isLLP0InCalo_b_acc",     &isLLP0InCalo_b_acc,      "isLLP0InCalo_b_acc/B");
    outputTree->Branch("isLLP1InCalo_b_acc",     &isLLP1InCalo_b_acc,      "isLLP1InCalo_b_acc/B");
    outputTree->Branch("isLLP0_boost",     &isLLP0_boost,      "isLLP0_boost/B");
    outputTree->Branch("isLLP1_boost",     &isLLP1_boost,      "isLLP1_boost/B");
    outputTree->Branch("isLLP0_res",     &isLLP0_res,      "isLLP0_res/B");
    outputTree->Branch("isLLP1_res",     &isLLP1_res,      "isLLP1_res/B");
    outputTree->Branch("nJets_to_LLP0",     &nJets_to_LLP0,      "nJets_to_LLP0/I");
    outputTree->Branch("nTagJets_to_LLP0",  &nTagJets_to_LLP0,   "nTagJets_to_LLP0/I");
    outputTree->Branch("nJets_to_LLP1",     &nJets_to_LLP1,      "nJets_to_LLP1/I");
    outputTree->Branch("nTagJets_to_LLP1",  &nTagJets_to_LLP1,   "nTagJets_to_LLP1/I");
    outputTree->Branch("nJets_to_LLP0_b_acc",     &nJets_to_LLP0_b_acc,      "nJets_to_LLP0_b_acc/I");
    outputTree->Branch("nTagJets_to_LLP0_b_acc",  &nTagJets_to_LLP0_b_acc,   "nTagJets_to_LLP0_b_acc/I");
    outputTree->Branch("nJets_to_LLP1_b_acc",     &nJets_to_LLP1_b_acc,      "nJets_to_LLP1_b_acc/I");
    outputTree->Branch("nTagJets_to_LLP1_b_acc",  &nTagJets_to_LLP1_b_acc,   "nTagJets_to_LLP1_b_acc/I");
    outputTree->Branch("nJets_to_LLP0InCalo",     &nJets_to_LLP0InCalo,      "nJets_to_LLP0InCalo/I");
    outputTree->Branch("nTagJets_to_LLP0InCalo",  &nTagJets_to_LLP0InCalo,   "nTagJets_to_LLP0InCalo/I");
    outputTree->Branch("nJets_to_LLP1InCalo",     &nJets_to_LLP1InCalo,      "nJets_to_LLP1InCalo/I");
    outputTree->Branch("nTagJets_to_LLP1InCalo",  &nTagJets_to_LLP1InCalo,   "nTagJets_to_LLP1InCalo/I");
    outputTree->Branch("nJets_to_LLP0InCalo_b_acc",     &nJets_to_LLP0InCalo_b_acc,      "nJets_to_LLP0InCalo_b_acc/I");
    outputTree->Branch("nTagJets_to_LLP0InCalo_b_acc",  &nTagJets_to_LLP0InCalo_b_acc,   "nTagJets_to_LLP0InCalo_b_acc/I");
    outputTree->Branch("nJets_to_LLP1InCalo_b_acc",     &nJets_to_LLP1InCalo_b_acc,      "nJets_to_LLP1InCalo_b_acc/I");
    outputTree->Branch("nTagJets_to_LLP1InCalo_b_acc",  &nTagJets_to_LLP1InCalo_b_acc,   "nTagJets_to_LLP1InCalo_b_acc/I");
    //
    outputTree->Branch("JetsFullEta", &skimmedJetsFullEta);
    outputTree->Branch("Jets", &skimmedJets);
    outputTree->Branch("JetsNegative", &skimmedJetsNegative);
    outputTree->Branch("JetsMatch", &skimmedJetsMatch);
    outputTree->Branch("JetsAcceptanceMatch", &skimmedJetsAcceptanceMatch);
    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);
    outputTree->Branch("GenGravitinos", &GenGravitinos);


    // setup TensorFlow objects
    tensorflow::setLogging();
    tensorflow::GraphDef* graphDefAK4 = tensorflow::loadGraphDef(graphPathAK4);
    // TF < 2
    //tensorflow::SessionOptions sessionOptions;
    //tensorflow::setThreading(sessionOptions, nThreads, threadPool);
    //tensorflow::Session* session = tensorflow::createSession(graphDef, sessionOptions);
    // TF >= 2
    tensorflow::Session* sessionAK4 = tensorflow::createSession(graphDefAK4, nThreads);
    tensorflow::Tensor inputTensorAK4(tensorflow::DT_FLOAT, {1, int(featuresAK4.size()) });
    float outputValueAK4;


    // Event loop

    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        TriggerWeight = 1.;
        PUReWeight = 1.;
        PUReWeightUp = 1.;
        PUReWeightDown = 1.;
	//Initialize nTagJets at every event
        nCHSJets = 0;
        nCHSJetsFullEta = 0;
        nCHSJetsMatch = 0;
        nCHSJetsAcceptanceMatch = 0;

	MinJetMetDPhi = 10.;
	//Initialize veto objects counter
	nTausPreVeto = 0;
	nTaus = 0;
	//nPhotons = 0;
	//nMuons = 0;
	//nElectrons = 0;
	nPhotonsPassing = 0;
	nPhotonsTight = 0;
	nTausPassing = 0;
	nMuonsPassing = 0;
	nElectronsPassing = 0;

        dR_LLPs = -9.;
        dR_Higgs = -9.;
        dR_Gravitinos = -9.;
        dPhi_LLPs = -9.;
        dPhi_Higgs = -9.;
        dPhi_Gravitinos = -9.;
        dR_Gravitino_Higgs_0 = -9.;
        dR_Gravitino_Higgs_1 = -9.;
	dR_Gravitino_GenMet_0 = -9.;
        dR_Gravitino_GenMet_1 = -9.;
        dPhi_Gravitino_Met_0 = -9.;
        dPhi_Gravitino_Met_1 = -9.;
        dPhi_Gravitino_GenMet_0 = -9.;
	dPhi_Gravitino_GenMet_1 = -9.;
        dR_LLP_GenMet_0 = -9.;
        dR_LLP_GenMet_1 = -9.;
        dPhi_LLP_Met_0 = -9.;
        dPhi_LLP_Met_1 = -9.;
        dPhi_LLP_GenMet_0 = -9.;
        dPhi_LLP_GenMet_1 = -9.;
        dR_Higgs_GenMet_0 = -9.;
        dR_Higgs_GenMet_1 = -9.;
        dPhi_Higgs_Met_0 = -9.;
        dPhi_Higgs_Met_1 = -9.;
        dPhi_Higgs_GenMet_0 = -9.;
	dPhi_Higgs_GenMet_1 = -9.;
        DiGravitino_pt = -1.;
        DiGravitino_mass = -1.;
        DiGravitino_eta = -1.;
        DiGravitino_phi = -1.;
        dR_DiGravitino_GenMet = -9.;
        dPhi_DiGravitino_Met = -9.;
        dPhi_DiGravitino_GenMet = -9.;
	dPhi_DiGravitino_Higgs_0 = -9.;
	dPhi_DiGravitino_Higgs_1 = -9.;
	dPhi_Gravitino_0_Higgs_0 = -9.;
	dPhi_Gravitino_1_Higgs_1 = -9.;
        perc_met_held_by_gravitinos = -1.;
 

	dt_ecal_no_tag_dist = 9999.;
	dt_ecal_acc_no_tag_dist = 9999.;
	dt_ecal_dist = 9999.;
	dt_ecal_acc_dist = 9999.;
	min_dPhi_jets_0p996 = 9999.;
	min_dPhi_jets_eta_1p0_0p996 = 9999.;
	nTagJets_0p996 = 0;

	//TBD
	nLLPInCalo_b_acc = 0;
	nLLP_b_acc = 0;

	isLLP0_b_acc = false;
	isLLP1_b_acc = false;
	isLLP0InCalo_b_acc = false;
	isLLP1InCalo_b_acc = false;

	isLLP0_boost = false;
	isLLP1_boost = false;
	isLLP0_res = false;
	isLLP1_res = false;

	nJets_to_LLP0  = 0;
	nTagJets_to_LLP0 = 0;
	nJets_to_LLP1  = 0;
	nTagJets_to_LLP1 = 0;

	nJets_to_LLP0_b_acc  = 0;
	nTagJets_to_LLP0_b_acc = 0;
	nJets_to_LLP1_b_acc  = 0;
	nTagJets_to_LLP1_b_acc = 0;

	nJets_to_LLP0InCalo  = 0;
	nTagJets_to_LLP0InCalo = 0;
	nJets_to_LLP1InCalo  = 0;
	nTagJets_to_LLP1InCalo = 0;

	nJets_to_LLP0InCalo_b_acc  = 0;
	nTagJets_to_LLP0InCalo_b_acc = 0;
	nJets_to_LLP1InCalo_b_acc  = 0;
	nTagJets_to_LLP1InCalo_b_acc = 0;

	//Clear all the vectors
	//very dangerous with continue statement!
	skimmedTaus.clear();
        skimmedJetsFullEta.clear();
        skimmedJets.clear();
        skimmedJetsNegative.clear();
        skimmedJetsMatch.clear();
        skimmedJetsAcceptanceMatch.clear();

        //if (i % 1000 == 0) {
        //    std::cout << "evaluating entry " << i << std::endl;
        //}
        inputTree->GetEntry(i);

        //std::cout << "======== " << std::endl;
        //std::cout << "Event " << i << std::endl;
	//std::cout << "======== " << std::endl;

	for (unsigned int j=0; j<Jets->size(); j++)
	  {

	    //Time smearing!
	    //We need:
	    //1. data file --> can pass from python
	    //2. signal file fit --> can pass from python
	    //3. name of the CB --> can pass from python

	    //Don't remove jets with eta>1 now, it messes up with the cosmic veto!
	    if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8)//cleaned jets!

	      {

		//Time smearing here
		// ** commented as suboptimal **
		//float pre_time = Jets->at(j).timeRecHitsEB;
		//float smearer = smearCB->GetRandom();
		////Keep also the original time if needed
		//Jets->at(j).timeRecHitsHB = pre_time;
		//Jets->at(j).timeRecHitsEB = pre_time + smearer;

		//here build the inputVector for each jet
		std::vector<float> inputValues(featuresAK4.size());
		//std::vector<float> inputValuesUnsmeared(featuresAK4.size());

		//tagger_AK4_v3
		inputValues.at(0) = Jets->at(j).nTrackConstituents;
		inputValues.at(1) = Jets->at(j).nSelectedTracks;
		inputValues.at(2) = Jets->at(j).timeRecHitsEB;
		inputValues.at(3) = Jets->at(j).eFracRecHitsEB;
		inputValues.at(4) = Jets->at(j).nRecHitsEB;
		inputValues.at(5) = Jets->at(j).sig1EB;
		inputValues.at(6) = Jets->at(j).sig2EB;
		inputValues.at(7) = Jets->at(j).ptDEB;
		inputValues.at(8) = Jets->at(j).cHadEFrac;
		inputValues.at(9) = Jets->at(j).nHadEFrac;
		inputValues.at(10) = Jets->at(j).eleEFrac;
		inputValues.at(11) = Jets->at(j).photonEFrac;
		inputValues.at(12) = Jets->at(j).ptAllTracks;
		inputValues.at(13) = Jets->at(j).ptAllPVTracks;
		inputValues.at(14) = Jets->at(j).alphaMax;
	        inputValues.at(15) = Jets->at(j).betaMax;
		inputValues.at(16) = Jets->at(j).gammaMax;
		inputValues.at(17) = Jets->at(j).gammaMaxEM;
		inputValues.at(18) = Jets->at(j).gammaMaxHadronic;
		inputValues.at(19) = Jets->at(j).gammaMaxET;
		inputValues.at(20) = Jets->at(j).minDeltaRAllTracks;
		inputValues.at(21) = Jets->at(j).minDeltaRPVTracks;


		float* d = inputTensorAK4.flat<float>().data();
		for (float v : inputValues) {
		  *d = v;
		  d++;
		}

		// run the inference
		std::vector<tensorflow::Tensor> outputsAK4;
		tensorflow::run(sessionAK4, {{inputTensorNameAK4, inputTensorAK4}}, {outputTensorNameAK4}, &outputsAK4, threadPool);

		// store the result
		outputValueAK4 = outputsAK4[0].matrix<float>()(0, 1);
		// keras cannot predict the output for invalid jets
		// fix it manually
		if(Jets->at(j).pt<0) outputValueAK4 = -1;
		Jets->at(j).sigprob = outputValueAK4;


		//28.11.22: unsmeared time
		if(outputValueAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1 and abs(Jets->at(j).eta)<1.) nTagJets_0p996++;

		//do not cut on time rec hits eb now since the smearing is not completely random
		if( abs(Jets->at(j).eta) < 1.)
		  {
		    //reject phi pathological regions
		    if( Jets->at(j).phi>minphi or Jets->at(j).phi<maxphi )
		      {
			skimmedJetsNegative.push_back(Jets->at(j));
		      }
		  }


		//apply time cut, affects acceptance
		//old approach: do the time cut before the time smearing, suboptimal but consistent
		if(Jets->at(j).timeRecHitsEB>-1.)
		  {
		    skimmedJetsFullEta.push_back(Jets->at(j));
		    
		    //apply all angular acceptance cuts
		    if( abs(Jets->at(j).eta) < 1.)
		      {
			//reject phi pathological regions
			if( Jets->at(j).phi>minphi or Jets->at(j).phi<maxphi )
			  {
			    skimmedJets.push_back(Jets->at(j));
			  }
		      }

		  }

		//save also jets including negative times

	      }//acceptance

	  }//jet loop
	

	//Three jets collections:
	//skimmedJets: unsmeared time, cut at -1 ns, all angular cuts
	//skimmedJetsNegative: unsmeared time, all angular cuts
	//skimmedJetsFullEta: unsmeared time, cut at -1 ns, no angular cuts --> used for cosmic and BH veto

        nCHSJetsFullEta = skimmedJetsFullEta.size();
        nCHSJets = skimmedJets.size();
        nCHSJetsNegative = skimmedJetsNegative.size();

	outputTree->Fill();


    }


    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
    n_pass->Write();
    n_odd->Write();
    n_even->Write();

    outputFile->Write();
    outputFile->Close();
    inputFile->Close();
    

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
