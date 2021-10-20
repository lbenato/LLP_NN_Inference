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
#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TH1.h"
#include "TMath.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v5.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

bool pt_sorter(const PFCandidateType& x, const PFCandidateType& y) { return x.pt > y.pt; }
bool energy_sorter(const ecalRecHitType& x, const ecalRecHitType& y) { return x.energy > y.energy; }
//bool h_energy_sorter(const hcalRecHitType& x, const hcalRecHitType& y) { return x.energy > y.energy; }

double avg ( std::vector<double> & v )
{
  double return_value = 0.0;
  int n = v.size();

  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i);
    }

  return ( return_value / n);
}


int main(int argc, char **argv) {

    if(argc<3)
    //if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    bool skipTrain(false);
    if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1) skipTrain=true;

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Skip even EventNumber: " << skipTrain << std::endl;

    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    // input and output file settings
    //unskimmed crab output
    //std::string inputPath = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v4_calo_AOD_2018_18October2020/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/201017_234633/0000/output_1.root";
    std::string inputPath = argv[1];

    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";
    //std::string inputTreeName = "skim";
    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;

    //bool doPFCand=false;

    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_AK4_v3/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_AK4_v3/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?
    //int nInputs = 10;
    std::string graphPathAK8 = basePath + "/tagger_AK8_v2_double_match/graph.pb";
    std::string MetaDataFileAK8 = basePath + "/tagger_AK8_v2_double_match/metadata.dat";
    std::string inputTensorNameAK8 = "input_input";
    std::string outputTensorNameAK8 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?

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
    TH1F   *counter = (TH1F*)inputFile->Get("counter/c_nEvents");
    TH1F   *n_pass = new TH1F("n_pass", "n_pass", 1, 0., 1.);
    TH1F   *n_odd = new TH1F("n_odd", "n_odd", 1, 0., 1.);
    TH1F   *n_even = new TH1F("n_even", "n_even", 1, 0., 1.);
    TH1F   *b_skipTrain = new TH1F("b_skipTrain", "b_skipTrain", 1, 0, 1);
    TH1F   *dR_bb_jet_0_den = new TH1F("dR_bb_jet_0_den", "dR_bb_jet_0_den", 100, 0., 10.);
    TH1F   *dR_bb_jet_0_num = new TH1F("dR_bb_jet_0_num", "dR_bb_jet_0_num", 100, 0., 10.);
    TH1F   *dR_bb_fatjet_0_den = new TH1F("dR_bb_fatjet_0_den", "dR_bb_fatjet_0_den", 100, 0., 10.);
    TH1F   *dR_bb_fatjet_0_num = new TH1F("dR_bb_fatjet_0_num", "dR_bb_fatjet_0_num", 100, 0., 10.);
    n_odd->Sumw2();
    n_even->Sumw2();
    n_pass->Sumw2();
    b_skipTrain->Sumw2();
    dR_bb_jet_0_den->Sumw2();
    dR_bb_jet_0_num->Sumw2();
    dR_bb_fatjet_0_den->Sumw2();
    dR_bb_fatjet_0_num->Sumw2();
    float  tree_weight = inputTree->GetWeight();
    std::cout << "Tree weight: " << tree_weight << std::endl;

    if(skipTrain) b_skipTrain->Fill(0);

    // Input variables
    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    float    EventWeight;
    bool   isMC;
    bool   isVBF;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    bool   Flag2_globalSuperTightHalo2016Filter;
    bool   Flag2_goodVertices;
    bool   Flag2_EcalDeadCellTriggerPrimitiveFilter;
    bool   Flag2_HBHENoiseFilter;
    bool   Flag2_HBHEIsoNoiseFilter;
    bool   Flag2_ecalBadCalibFilter;
    bool   Flag2_eeBadScFilter;
    bool   Flag2_BadPFMuonFilter;
    bool   HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    bool   HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    float  HT;
    float  MinJetMetDPhi;
    Long64_t nCHSJets;
    Long64_t nCHSFatJets;
    int    nElectrons;
    int    nMuons;
    int    nPhotons;
    int    nTaus;
    int    nPFCandidates;
    int    nPFCandidatesTrack;
    //std::vector<TauType>         *Taus = 0;
    //std::vector<PhotonType>      *Photons = 0;
    //std::vector<LeptonType>      *Muons = 0;
    //std::vector<LeptonType>      *Electrons = 0;
    std::vector<JetType>         *Jets = 0;
    std::vector<FatJetType>      *FatJets = 0;
    MEtType                      *MEt = 0;
    std::vector<GenPType>        *GenHiggs = 0;
    std::vector<GenPType>        *GenLLPs = 0;
    std::vector<GenPType>        *GenBquarks = 0;

    // Input branches
    //TBranch        *b_Taus = 0;
    //TBranch        *b_Photons = 0;
    //TBranch        *b_Muons = 0;
    //TBranch        *b_Electrons = 0;
    TBranch        *b_Jets = 0;
    TBranch        *b_FatJets = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_isMC;
    TBranch        *b_isVBF;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;
    TBranch        *b_Flag2_globalSuperTightHalo2016Filter;
    TBranch        *b_Flag2_goodVertices;
    TBranch        *b_Flag2_EcalDeadCellTriggerPrimitiveFilter;
    TBranch        *b_Flag2_HBHENoiseFilter;
    TBranch        *b_Flag2_HBHEIsoNoiseFilter;
    TBranch        *b_Flag2_ecalBadCalibFilter;
    TBranch        *b_Flag2_eeBadScFilter;
    TBranch        *b_Flag2_BadPFMuonFilter;
    TBranch        *b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v;
    TBranch        *b_HT;
    TBranch        *b_MinJetMetDPhi;
    TBranch        *b_nCHSJets;
    TBranch        *b_nCHSFatJets;
    TBranch        *b_nElectrons;
    TBranch        *b_nMuons;
    TBranch        *b_nPhotons;
    TBranch        *b_nTaus;
    TBranch        *b_nPFCandidates;
    TBranch        *b_nPFCandidatesTrack;

    //inputTree->SetBranchAddress("Taus",              &Taus,              &b_Taus);
    //inputTree->SetBranchAddress("Photons",           &Photons,           &b_Photons);
    //inputTree->SetBranchAddress("Muons",             &Muons,             &b_Muons);
    //inputTree->SetBranchAddress("Electrons",         &Electrons,         &b_Electrons);
    inputTree->SetBranchAddress("Jets",              &Jets,              &b_Jets);
    inputTree->SetBranchAddress("FatJets",           &FatJets,           &b_FatJets);
    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt); 
    inputTree->SetBranchAddress("GenHiggs",          &GenHiggs,          &b_GenHiggs); 
    inputTree->SetBranchAddress("GenLLPs",           &GenLLPs,           &b_GenLLPs); 
    inputTree->SetBranchAddress("GenBquarks",        &GenBquarks,        &b_GenBquarks); 
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("isMC",              &isMC,              &b_isMC);
    inputTree->SetBranchAddress("isVBF",             &isVBF,             &b_isVBF);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);
    inputTree->SetBranchAddress("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, &b_Flag2_globalSuperTightHalo2016Filter);
    inputTree->SetBranchAddress("Flag2_goodVertices", &Flag2_goodVertices, &b_Flag2_goodVertices);
    inputTree->SetBranchAddress("Flag2_EcalDeadCellTriggerPrimitiveFilter", &Flag2_EcalDeadCellTriggerPrimitiveFilter, &b_Flag2_EcalDeadCellTriggerPrimitiveFilter);
    inputTree->SetBranchAddress("Flag2_HBHENoiseFilter", &Flag2_HBHENoiseFilter, &b_Flag2_HBHENoiseFilter);
    inputTree->SetBranchAddress("Flag2_HBHEIsoNoiseFilter", &Flag2_HBHEIsoNoiseFilter, &b_Flag2_HBHEIsoNoiseFilter);
    inputTree->SetBranchAddress("Flag2_ecalBadCalibFilter", &Flag2_ecalBadCalibFilter, &b_Flag2_ecalBadCalibFilter);
    inputTree->SetBranchAddress("Flag2_eeBadScFilter", &Flag2_eeBadScFilter, &b_Flag2_eeBadScFilter);
    inputTree->SetBranchAddress("Flag2_BadPFMuonFilter", &Flag2_BadPFMuonFilter, &b_Flag2_BadPFMuonFilter);
    inputTree->SetBranchAddress("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, &b_HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, &b_HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v);
    inputTree->SetBranchAddress("HT",                &HT,                &b_HT);
    inputTree->SetBranchAddress("MinJetMetDPhi",     &MinJetMetDPhi,     &b_MinJetMetDPhi);
    inputTree->SetBranchAddress("nCHSJets",          &nCHSJets,          &b_nCHSJets);
    inputTree->SetBranchAddress("nCHSFatJets",       &nCHSFatJets,       &b_nCHSFatJets);
    inputTree->SetBranchAddress("nElectrons",        &nElectrons,        &b_nElectrons);
    inputTree->SetBranchAddress("nMuons",            &nMuons,            &b_nMuons);
    inputTree->SetBranchAddress("nPhotons",          &nPhotons,          &b_nPhotons);
    inputTree->SetBranchAddress("nTaus",             &nTaus,             &b_nTaus);
    inputTree->SetBranchAddress("nPFCandidates",     &nPFCandidates,     &b_nPFCandidates);
    inputTree->SetBranchAddress("nPFCandidatesTrack", &nPFCandidatesTrack, &b_nPFCandidatesTrack);

    //inputTree->SetBranchStatus("*",0);

    // Read jet input features from metadata.dat file
    //AK4
    std::ifstream finAK4;
    std::string featAK4;
    finAK4.open(MetaDataFileAK4);
    std::vector<std::string> featuresAK4;
    std::string toEraseAK4 = "Jet_";
    std::cout << "   -- > Features AK4: " << std::endl;
    while (finAK4 >> featAK4)
      {
	size_t pos = featAK4.find(toEraseAK4);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    featAK4.erase(pos, toEraseAK4.length());
	  }
	//std::string new_feat = featAK4.substr(position);
	std::cout << featAK4 << std::endl;
	featuresAK4.push_back(featAK4);
      }
    finAK4.close();

    //AK8
    std::ifstream finAK8;
    std::string featAK8;
    finAK8.open(MetaDataFileAK8);
    std::vector<std::string> featuresAK8;
    std::string toEraseAK8 = "FatJet_";
    std::cout << "   -- > Features AK8: " << std::endl;
    while (finAK8 >> featAK8)
      {
	size_t pos = featAK8.find(toEraseAK8);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    featAK8.erase(pos, toEraseAK8.length());
	  }
	//std::string new_feat = featAK8.substr(position);
	std::cout << featAK8 << std::endl;
	featuresAK8.push_back(featAK8);
      }
    finAK8.close();


    //inputTree->SetBranchStatus("Jets_pt",1);//needed?
    

    //do per branch
    //float pt;
    //inputTree->SetBranchAddress("Jets.pt", &pt );


    // This allows to activate only the needed branches
    //for(unsigned int f; f<features.size(); f++)
      //{
	//std::cout<<features.at(f)<<std::endl;
	//std::string tmp_feat = "Jets.";
	//tmp_feat.append(features.at(f)); 
	//std::cout<<tmp_feat<<std::endl;
	//char * cstr = new char [tmp_feat.length()+1];
	//std::strcpy (cstr, tmp_feat.c_str());
	//inputTree->SetBranchStatus(cstr,1);//needed?
      //}


    //const char* L = "Jets.ptAllTracks";
    //inputTree->SetBranchStatus(L,1);//needed?



    // ================= 
    // Output
    // ================= 

    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");


    std::vector<JetType>    skimmedJets;
    std::vector<FatJetType> skimmedFatJets;

    float MinFatJetMetDPhi(10.);
    float MinFatJetMetDPhiBarrel(10.);
    float MinFatJetMetDPhiBarrelMatched(10.);
    float MinJetMetDPhiBarrel(10.);

    float dR_bb_jet_0(-1.);
    float dR_bb_fatjet_0(-1.);

    //int nTaus(0);
    //int nTausPreVeto(0);
    //int nPhotonsPassing(0);
    //int nMuonsPassing(0);
    //int nElectronsPassing(0);

    int nCHSJetsAcceptanceCalo;
    int nCHSFatJetsAcceptanceCalo;

    //float AK4_jet_width_ECAL(0.);
    //float AK8_jet_width_ECAL(0.);
    //float AK4_jet_width_HCAL(0.);
    //float AK8_jet_width_HCAL(0.);

    int nTagJets_cutbased(0);
    int nTagJets_0p9(0);
    int nTagJets_0p95(0);
    int nTagJets_0p96(0);
    int nTagJets_0p97(0);
    int nTagJets_0p98(0);
    int nTagJets_0p99(0);
    int nTagJets_0p994(0);
    int nTagJets_0p995(0);
    int nTagJets_0p996(0);
    int nTagJets_0p997(0);
    int nTagJets_0p999(0);

    int nTagJets_cutbased_JJ(0);
    int nTagJets_0p99_JJ(0);
    int nTagJets_0p994_JJ(0);
    int nTagJets_0p996_JJ(0);
    int nTagJets_0p997_JJ(0);

    int nTagFatJets_cutbased(0);
    int nTagFatJets_0p8(0);
    int nTagFatJets_0p9(0);
    int nTagFatJets_0p92(0);
    int nTagFatJets_0p95(0);
    int nTagFatJets_0p96(0);
    int nTagFatJets_0p97(0);
    int nTagFatJets_0p98(0);
    int nTagFatJets_0p99(0);
    int nTagFatJets_0p995(0);
    int nTagFatJets_0p997(0);
    int nTagFatJets_0p999(0);
    int nTagFatJets_0p9995(0);
    int nTagFatJets_0p9999(0);
    int nTagFatJets_0p99995(0);
    int nTagFatJets_0p99999(0);
    int nTagFatJets_0p999995(0);
    int nTagFatJets_0p999999(0);

    bool isTagAK8_0p9999_170;
    bool isTagAK8_0p9999_200;
    bool isTagAK8_0p9999_250;
    bool isTagAK8_0p9999_300;
    bool isTagAK8_0p9999_350;

    bool isTagAK8_0p99999_170;
    bool isTagAK8_0p99999_200;
    bool isTagAK8_0p99999_250;
    bool isTagAK8_0p99999_300;
    bool isTagAK8_0p99999_350;

    bool isTagAK8_0p999995_170;
    bool isTagAK8_0p999995_200;
    bool isTagAK8_0p999995_250;
    bool isTagAK8_0p999995_300;
    bool isTagAK8_0p999995_350;

    bool isTagAK8_0p999999_170;
    bool isTagAK8_0p999999_200;
    bool isTagAK8_0p999999_250;
    bool isTagAK8_0p999999_300;
    bool isTagAK8_0p999999_350;

    bool isTagAK4_0p99;
    bool isTagAK4_0p994;
    bool isTagAK4_0p996;
    bool isTagAK4_0p997;
    bool isTagAK4_0p99_JJ;
    bool isTagAK4_0p994_JJ;
    bool isTagAK4_0p996_JJ;
    bool isTagAK4_0p997_JJ;

    // Output branches 
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("RunNumber",         &RunNumber,         "RunNumber/L");
    outputTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    outputTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isVBF",             &isVBF,             "isVBF/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");
    outputTree->Branch("HT",                &HT,                "HT/F");
    outputTree->Branch("MinJetMetDPhi",     &MinJetMetDPhi,     "MinJetMetDPhi/F");
    outputTree->Branch("MinJetMetDPhiBarrel",  &MinJetMetDPhiBarrel,  "MinJetMetDPhiBarrel/F");
    outputTree->Branch("MinFatJetMetDPhi",  &MinFatJetMetDPhi,  "MinFatJetMetDPhi/F");
    outputTree->Branch("MinFatJetMetDPhiBarrel",  &MinFatJetMetDPhiBarrel,  "MinFatJetMetDPhiBarrel/F");
    outputTree->Branch("MinFatJetMetDPhiBarrelMatched",  &MinFatJetMetDPhiBarrelMatched,  "MinFatJetMetDPhiBarrelMatched/F");
    outputTree->Branch("dR_bb_jet_0",       &dR_bb_jet_0,          "dR_bb_jet_0/F");
    outputTree->Branch("dR_bb_fatjet_0",    &dR_bb_fatjet_0,       "dR_bb_fatjet_0/F");
    outputTree->Branch("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    outputTree->Branch("nCHSFatJets",       &nCHSFatJets,       "nCHSFatJets/I");
    outputTree->Branch("nCHSJetsAcceptanceCalo",          &nCHSJetsAcceptanceCalo,          "nCHSJetsAcceptanceCalo/I");
    outputTree->Branch("nCHSFatJetsAcceptanceCalo",       &nCHSFatJetsAcceptanceCalo,       "nCHSFatJetsAcceptanceCalo/I");
    outputTree->Branch("nElectrons",        &nElectrons,        "nElectrons/I");
    outputTree->Branch("nMuons",            &nMuons,            "nMuons/I");
    outputTree->Branch("nPhotons",          &nPhotons,          "nPhotons/I");
    //outputTree->Branch("nTausPreVeto",      &nTausPreVeto,      "nTausPreVeto/I");
    outputTree->Branch("nTaus",             &nTaus,             "nTaus/I");

    //outputTree->Branch("nElectronsPassing",        &nElectronsPassing,        "nElectronsPassing/I");
    //outputTree->Branch("nMuonsPassing",            &nMuonsPassing,            "nMuonsPassing/I");
    //outputTree->Branch("nPhotonsPassing",          &nPhotonsPassing,          "nPhotonsPassing/I");
    //outputTree->Branch("nTausPassing",             &nTausPassing,             "nTausPassing/I");

    outputTree->Branch("nPFCandidates",     &nPFCandidates,     "nPFCandidates/I");
    outputTree->Branch("nPFCandidatesTrack", &nPFCandidatesTrack, "nPFCandidatesTrack/I");
    outputTree->Branch("Jets", &skimmedJets);
    outputTree->Branch("FatJets", &skimmedFatJets);

    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);

    //outputTree->Branch("AK4_jet_width_ECAL", &AK4_jet_width_ECAL,  "AK4_jet_width_ECAL/F");
    //outputTree->Branch("AK8_jet_width_ECAL", &AK8_jet_width_ECAL,  "AK8_jet_width_ECAL/F");
    //outputTree->Branch("AK4_jet_width_HCAL", &AK4_jet_width_HCAL,  "AK4_jet_width_HCAL/F");
    //outputTree->Branch("AK8_jet_width_HCAL", &AK8_jet_width_HCAL,  "AK8_jet_width_HCAL/F");

    outputTree->Branch("nTagJets_cutbased", &nTagJets_cutbased,  "nTagJets_cutbased/I");
    outputTree->Branch("nTagJets_0p9",      &nTagJets_0p9,       "nTagJets_0p9/I");
    outputTree->Branch("nTagJets_0p95",     &nTagJets_0p95,      "nTagJets_0p95/I");
    outputTree->Branch("nTagJets_0p96",     &nTagJets_0p96,      "nTagJets_0p96/I");
    outputTree->Branch("nTagJets_0p97",     &nTagJets_0p97,      "nTagJets_0p97/I");
    outputTree->Branch("nTagJets_0p98",     &nTagJets_0p98,      "nTagJets_0p98/I");
    outputTree->Branch("nTagJets_0p99",     &nTagJets_0p99,      "nTagJets_0p99/I");
    outputTree->Branch("nTagJets_0p994",     &nTagJets_0p994,      "nTagJets_0p994/I");
    outputTree->Branch("nTagJets_0p995",     &nTagJets_0p995,      "nTagJets_0p995/I");
    outputTree->Branch("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    outputTree->Branch("nTagJets_0p997",     &nTagJets_0p997,      "nTagJets_0p997/I");
    outputTree->Branch("nTagJets_0p999",     &nTagJets_0p999,      "nTagJets_0p999/I");

    outputTree->Branch("nTagJets_cutbased_JJ", &nTagJets_cutbased_JJ,  "nTagJets_cutbased_JJ/I");
    outputTree->Branch("nTagJets_0p99_JJ",     &nTagJets_0p99_JJ,      "nTagJets_0p99_JJ/I");
    outputTree->Branch("nTagJets_0p994_JJ",     &nTagJets_0p994_JJ,      "nTagJets_0p994_JJ/I");
    outputTree->Branch("nTagJets_0p996_JJ",     &nTagJets_0p996_JJ,      "nTagJets_0p996_JJ/I");
    outputTree->Branch("nTagJets_0p997_JJ",     &nTagJets_0p997_JJ,      "nTagJets_0p997_JJ/I");

    outputTree->Branch("nTagFatJets_cutbased", &nTagFatJets_cutbased,  "nTagFatJets_cutbased/I");
    outputTree->Branch("nTagFatJets_0p8",      &nTagFatJets_0p8,       "nTagFatJets_0p8/I");
    outputTree->Branch("nTagFatJets_0p9",      &nTagFatJets_0p9,       "nTagFatJets_0p9/I");
    outputTree->Branch("nTagFatJets_0p92",      &nTagFatJets_0p92,       "nTagFatJets_0p92/I");
    outputTree->Branch("nTagFatJets_0p95",     &nTagFatJets_0p95,      "nTagFatJets_0p95/I");
    outputTree->Branch("nTagFatJets_0p96",     &nTagFatJets_0p96,      "nTagFatJets_0p96/I");
    outputTree->Branch("nTagFatJets_0p97",     &nTagFatJets_0p97,      "nTagFatJets_0p97/I");
    outputTree->Branch("nTagFatJets_0p98",     &nTagFatJets_0p98,      "nTagFatJets_0p98/I");
    outputTree->Branch("nTagFatJets_0p99",     &nTagFatJets_0p99,      "nTagFatJets_0p99/I");
    outputTree->Branch("nTagFatJets_0p995",     &nTagFatJets_0p995,      "nTagFatJets_0p995/I");
    outputTree->Branch("nTagFatJets_0p996",     &nTagFatJets_0p997,      "nTagFatJets_0p997/I");
    outputTree->Branch("nTagFatJets_0p997",     &nTagFatJets_0p999,      "nTagFatJets_0p999/I");
    outputTree->Branch("nTagFatJets_0p9995",     &nTagFatJets_0p9995,      "nTagFatJets_0p9995/I");
    outputTree->Branch("nTagFatJets_0p9999",     &nTagFatJets_0p9999,      "nTagFatJets_0p9999/I");
    outputTree->Branch("nTagFatJets_0p99995",     &nTagFatJets_0p99995,      "nTagFatJets_0p99995/I");
    outputTree->Branch("nTagFatJets_0p99999",     &nTagFatJets_0p99999,      "nTagFatJets_0p99999/I");
    outputTree->Branch("nTagFatJets_0p999995",     &nTagFatJets_0p999995,      "nTagFatJets_0p999995/I");
    outputTree->Branch("nTagFatJets_0p999999",     &nTagFatJets_0p999999,      "nTagFatJets_0p999999/I");

    outputTree->Branch("isTagAK4_0p99", &isTagAK4_0p99, "isTagAK4_0p99/O");
    outputTree->Branch("isTagAK4_0p994", &isTagAK4_0p994, "isTagAK4_0p994/O");
    outputTree->Branch("isTagAK4_0p996", &isTagAK4_0p996, "isTagAK4_0p996/O");
    outputTree->Branch("isTagAK4_0p997", &isTagAK4_0p997, "isTagAK4_0p997/O");
    outputTree->Branch("isTagAK4_0p99_JJ", &isTagAK4_0p99_JJ, "isTagAK4_0p99_JJ/O");
    outputTree->Branch("isTagAK4_0p994_JJ", &isTagAK4_0p994_JJ, "isTagAK4_0p994_JJ/O");
    outputTree->Branch("isTagAK4_0p996_JJ", &isTagAK4_0p996_JJ, "isTagAK4_0p996_JJ/O");
    outputTree->Branch("isTagAK4_0p997_JJ", &isTagAK4_0p997_JJ, "isTagAK4_0p997_JJ/O");

    outputTree->Branch("isTagAK8_0p9999_170",  &isTagAK8_0p9999_170,   "isTagAK8_0p9999_170/O");
    outputTree->Branch("isTagAK8_0p9999_200",  &isTagAK8_0p9999_200,   "isTagAK8_0p9999_200/O");
    outputTree->Branch("isTagAK8_0p9999_250",  &isTagAK8_0p9999_250,   "isTagAK8_0p9999_250/O");
    outputTree->Branch("isTagAK8_0p9999_300",  &isTagAK8_0p9999_300,   "isTagAK8_0p9999_300/O");
    outputTree->Branch("isTagAK8_0p9999_350",  &isTagAK8_0p9999_350,   "isTagAK8_0p9999_350/O");

    outputTree->Branch("isTagAK8_0p99999_170",  &isTagAK8_0p99999_170,   "isTagAK8_0p99999_170/O");
    outputTree->Branch("isTagAK8_0p99999_200",  &isTagAK8_0p99999_200,   "isTagAK8_0p99999_200/O");
    outputTree->Branch("isTagAK8_0p99999_250",  &isTagAK8_0p99999_250,   "isTagAK8_0p99999_250/O");
    outputTree->Branch("isTagAK8_0p99999_300",  &isTagAK8_0p99999_300,   "isTagAK8_0p99999_300/O");
    outputTree->Branch("isTagAK8_0p99999_350",  &isTagAK8_0p99999_350,   "isTagAK8_0p99999_350/O");

    outputTree->Branch("isTagAK8_0p999995_170",  &isTagAK8_0p999995_170,   "isTagAK8_0p999995_170/O");
    outputTree->Branch("isTagAK8_0p999995_200",  &isTagAK8_0p999995_200,   "isTagAK8_0p999995_200/O");
    outputTree->Branch("isTagAK8_0p999995_250",  &isTagAK8_0p999995_250,   "isTagAK8_0p999995_250/O");
    outputTree->Branch("isTagAK8_0p999995_300",  &isTagAK8_0p999995_300,   "isTagAK8_0p999995_300/O");
    outputTree->Branch("isTagAK8_0p999995_350",  &isTagAK8_0p999995_350,   "isTagAK8_0p999995_350/O");

    outputTree->Branch("isTagAK8_0p999999_170",  &isTagAK8_0p999999_170,   "isTagAK8_0p999999_170/O");
    outputTree->Branch("isTagAK8_0p999999_200",  &isTagAK8_0p999999_200,   "isTagAK8_0p999999_200/O");
    outputTree->Branch("isTagAK8_0p999999_250",  &isTagAK8_0p999999_250,   "isTagAK8_0p999999_250/O");
    outputTree->Branch("isTagAK8_0p999999_300",  &isTagAK8_0p999999_300,   "isTagAK8_0p999999_300/O");
    outputTree->Branch("isTagAK8_0p999999_350",  &isTagAK8_0p999999_350,   "isTagAK8_0p999999_350/O");


    //do it as a loop
    //std::vector<float> Jet_0_inputValues(features.size());
    //Here loop and to the branch thing



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

    tensorflow::GraphDef* graphDefAK8 = tensorflow::loadGraphDef(graphPathAK8);
    tensorflow::Session* sessionAK8 = tensorflow::createSession(graphDefAK8, nThreads);
    tensorflow::Tensor inputTensorAK8(tensorflow::DT_FLOAT, {1, int(featuresAK8.size()) });
    float outputValueAK8;


    // Event loop

    //for(int i = 0; i < 1; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

	//Initialize nTagJets at every event
        nCHSJetsAcceptanceCalo = 0;
        nCHSFatJetsAcceptanceCalo = 0;
	MinFatJetMetDPhi = 10.;
	MinJetMetDPhiBarrel = 10.;
	MinFatJetMetDPhiBarrel = 10.;
	MinFatJetMetDPhiBarrelMatched = 10.;
	//Initialize veto objects counter
	//nTausPreVeto = 0;
	//nTaus = 0;
	//nPhotonsPassing = 0;
	//nMuonsPassing = 0;
	//nElectronsPassing = 0;

	//AK4_jet_width_ECAL = 0.;
	//AK8_jet_width_ECAL = 0.;
	//AK4_jet_width_HCAL = 0.;
	//AK8_jet_width_HCAL = 0.;

	nTagJets_cutbased = 0;
	nTagJets_0p9 = 0;
	nTagJets_0p95 = 0;
	nTagJets_0p96 = 0;
	nTagJets_0p97 = 0;
	nTagJets_0p98 = 0;
	nTagJets_0p99 = 0;
	nTagJets_0p994 = 0;
	nTagJets_0p995 = 0;
	nTagJets_0p996 = 0;
	nTagJets_0p997 = 0;
	nTagJets_0p999 = 0;
	nTagJets_cutbased_JJ = 0;
	nTagJets_0p99_JJ = 0;
	nTagJets_0p994_JJ = 0;
	nTagJets_0p996_JJ = 0;
	nTagJets_0p997_JJ = 0;
	nTagFatJets_cutbased = 0;
	nTagFatJets_0p8 = 0;
	nTagFatJets_0p9 = 0;
	nTagFatJets_0p92 = 0;
	nTagFatJets_0p95 = 0;
	nTagFatJets_0p96 = 0;
	nTagFatJets_0p97 = 0;
	nTagFatJets_0p98 = 0;
	nTagFatJets_0p99 = 0;
	nTagFatJets_0p995 = 0;
	nTagFatJets_0p997 = 0;
	nTagFatJets_0p999 = 0;
	nTagFatJets_0p9995 = 0;
	nTagFatJets_0p9999 = 0;
	nTagFatJets_0p99995 = 0;
	nTagFatJets_0p99999 = 0;
	nTagFatJets_0p999995 = 0;
	nTagFatJets_0p999999 = 0;

        isTagAK8_0p9999_170 = false;
        isTagAK8_0p9999_200 = false;
        isTagAK8_0p9999_250 = false;
        isTagAK8_0p9999_300 = false;
        isTagAK8_0p99999_170 = false;
        isTagAK8_0p99999_200 = false;
        isTagAK8_0p99999_250 = false;
        isTagAK8_0p99999_300 = false;
        isTagAK8_0p99999_350 = false;

        isTagAK8_0p999995_170 = false;
        isTagAK8_0p999995_200 = false;
        isTagAK8_0p999995_250 = false;
        isTagAK8_0p999995_300 = false;
        isTagAK8_0p999995_350 = false;

        isTagAK8_0p999999_170 = false;
        isTagAK8_0p999999_200 = false;
        isTagAK8_0p999999_250 = false;
        isTagAK8_0p999999_300 = false;
        isTagAK8_0p999999_350 = false;

        isTagAK8_0p9999_350 = false;
	isTagAK4_0p99 = false;
	isTagAK4_0p994 = false;
	isTagAK4_0p996 = false;
	isTagAK4_0p997 = false;
	isTagAK4_0p99_JJ = false;
	isTagAK4_0p994_JJ = false;
	isTagAK4_0p996_JJ = false;
	isTagAK4_0p997_JJ = false;

	//Clear all the vectors
	//very dangerous with continue statement!
        skimmedJets.clear();
        skimmedFatJets.clear();



        if (i % 1000 == 0) {
            std::cout << "evaluating entry " << i << std::endl;
        }
        inputTree->GetEntry(i);

	//if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1)
	//{
	//if (EventNumber % 2 == 0)
	//{
	//std::cout << "Skip even EventNumber! " << std::endl;
	//continue;
	//}
	//}



        std::cout << "======== " << std::endl;
        std::cout << "Event " << i << std::endl;
	std::cout << "======== " << std::endl;

	//Loop on veto objects
	//Taus
	//nTausPreVeto = int(Taus->size());
	//for (unsigned int t=0; t<Taus->size(); t++)
	//  {
	//    if(Taus->at(t).decayModeFinding == true and Taus->at(t).byLooseCombinedIsolationDeltaBetaCorr3Hits == true) nTaus++;
	//  }
	//std::cout << "nTaus: " << nTaus << std::endl;
	//std::cout << "nTausPassing: " << nTausPassing << std::endl;
	//Photons
	//Muons
	//Electrons


        if(!HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v) continue;
        if(nCHSJets<1 and nCHSFatJets<1) continue;
        if(nTaus>0) continue;
        if(nPhotons>0) continue;
        if(nMuons>0) continue;
        if(nElectrons>0) continue;
        if(!Flag2_globalSuperTightHalo2016Filter) continue;
        if(!Flag2_EcalDeadCellTriggerPrimitiveFilter) continue;
        if(!Flag2_HBHENoiseFilter) continue;
        if(!Flag2_HBHEIsoNoiseFilter) continue;
        if(!Flag2_ecalBadCalibFilter) continue;
        if(!Flag2_eeBadScFilter) continue;
        if(!Flag2_BadPFMuonFilter) continue;
        //if(HT<100) continue;
        if(MEt->pt<200) continue;


	//Apply acceptance cuts to jets and fat jets 
	std::vector<int> validJetIndex;
	std::vector<int> validFatJetIndex;

	for (unsigned int j=0; j<Jets->size(); j++)
	  {

	    /*
	      Additional pre-selections
		"muEFrac" : {"min" : -1., "max" : 0.6},
		"eleEFrac" : {"min" : -1., "max" : 0.6},
		"photonEFrac" : {"min" : -1., "max" : 0.8},
	    */

	    //if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1)

	    if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100.)
	      {

		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhiBarrel) MinJetMetDPhiBarrel = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));

		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		//std::cout<< "Jet n. " << j << " eFracRecHitsEB: " << Jets->at(j).eFracRecHitsEB  << std::endl;
		Jets->at(j).eFracRecHitsEB = (Jets->at(j).energy>0 and Jets->at(j).energyRecHitsEB>0) ? Jets->at(j).energyRecHitsEB/Jets->at(j).energy : -1.;
		//std::cout<< "FIXED: Jet n. " << j << " eFracRecHitsEB: " << Jets->at(j).eFracRecHitsEB  << std::endl;

		//std::cout<< "Jet n. " << j << " pt: " << Jets->at(j).pt << " ; sigprob: " << Jets->at(j).sigprob  << std::endl;
		//here build the inputVector for each jet
		std::vector<float> inputValues(featuresAK4.size());

		//tagger_AK4_v3
		inputValues.at(0) = Jets->at(j).nTrackConstituents;
		inputValues.at(1) = Jets->at(j).nSelectedTracks;
		inputValues.at(2) = Jets->at(j).timeRecHitsEB;
		inputValues.at(3) = Jets->at(j).eFracRecHitsEB;
		inputValues.at(4) = Jets->at(j).nRecHitsEB;
		inputValues.at(5) = Jets->at(j).sig1EB;
		inputValues.at(6) = Jets->at(j).sig2EB;
		inputValues.at(7) = Jets->at(j).ptDEB;
		//v3 does not have those:
		//inputValues.at(8) = Jets->at(j).sig1PF;
		//inputValues.at(9) = Jets->at(j).sig2PF;
		//inputValues.at(10) = Jets->at(j).ptDPF;
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
		  //std::cout<< " input value: " << v <<std::endl;
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

		/*
		 Cut based- definition:
		"timeRecHitsEB" : {"min" : 0.09, "max" : 999.e+10},
		"gammaMaxET" : {"min" : -100.-10., "max" : 0.16},
		"minDeltaRPVTracks" : {"min" : 0.06, "max" : 999.+10.},
		"cHadEFrac" : {"min" : -1., "max" : 0.06},
		*/
		if(Jets->at(j).timeRecHitsEB>0.09 and Jets->at(j).gammaMaxET<0.16 and Jets->at(j).minDeltaRPVTracks>0.06 and Jets->at(j).cHadEFrac<0.06) nTagJets_cutbased++;
		if(outputValueAK4>0.9) nTagJets_0p9++;
		if(outputValueAK4>0.95) nTagJets_0p95++;
		if(outputValueAK4>0.96) nTagJets_0p96++;
		if(outputValueAK4>0.97) nTagJets_0p97++;
		if(outputValueAK4>0.98) nTagJets_0p98++;
		if(outputValueAK4>0.99) nTagJets_0p99++;
		if(outputValueAK4>0.994) nTagJets_0p994++;
		if(outputValueAK4>0.995) nTagJets_0p995++;
		if(outputValueAK4>0.996) nTagJets_0p996++;
		if(outputValueAK4>0.997) nTagJets_0p997++;
		if(outputValueAK4>0.999) nTagJets_0p999++;

		if(Jets->at(j).timeRecHitsEB>0.09 and Jets->at(j).gammaMaxET<0.16 and Jets->at(j).minDeltaRPVTracks>0.06 and Jets->at(j).cHadEFrac<0.06 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8) nTagJets_cutbased_JJ++;
		if(outputValueAK4>0.99 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p99_JJ++;
		if(outputValueAK4>0.994 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p994_JJ++;
		if(outputValueAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p996_JJ++;
		if(outputValueAK4>0.997 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p997_JJ++;



		//Redo gen-matching to calculate dR of the b quarks
		for (unsigned int l=0; l<GenLLPs->size(); l++)
		  {
		    //std::cout << "GenLLP travelRadius " << GenLLPs->at(l).travelRadius << std::endl;
		    //std::cout << "Jets->at(j).radiusLLP " << Jets->at(j).radiusLLP << std::endl;
		    if(GenLLPs->at(l).travelRadius==Jets->at(j).radiusLLP and Jets->at(j).isGenMatchedCaloCorrLLPAccept)
		      {
			//std::cout << "LLP travel radius: " << GenLLPs->at(l).travelRadius << std::endl;
			//std::cout << "LLP dRdaughters: " << GenLLPs->at(l).dRdaughters << std::endl;
			if(j==0)
			  {
			    dR_bb_jet_0_den->Fill(GenLLPs->at(l).dRdaughters);
			    dR_bb_jet_0 = GenLLPs->at(l).dRdaughters;
			  }
			if(j==0 && Jets->at(j).sigprob>0.996) dR_bb_jet_0_num->Fill(GenLLPs->at(l).dRdaughters);
			//Jets->at(j).dR_bb = GenLLPs->at(l).dRdaughters;
		      }
		  }





		//store jets passing acceptance and with inference
		nCHSJetsAcceptanceCalo++;
		skimmedJets.push_back(Jets->at(j));
		validJetIndex.push_back(j);

	      }//acceptance

	  }//jet loop



        for (unsigned int j=0; j<FatJets->size(); j++)
          {
	    if(fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi)) < MinFatJetMetDPhi) MinFatJetMetDPhi = fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi));

            if( FatJets->at(j).pt>170 && fabs(FatJets->at(j).eta)<1.48 and FatJets->at(j).timeRecHitsEB>-100.)
              {
		if(fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi)) < MinFatJetMetDPhiBarrel) MinFatJetMetDPhiBarrel = fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi));


		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		FatJets->at(j).eFracRecHitsEB = (FatJets->at(j).energy>0 and FatJets->at(j).energyRecHitsEB>0) ? FatJets->at(j).energyRecHitsEB/FatJets->at(j).energy : -1.;


		std::vector<float> inputValues(featuresAK8.size());

		inputValues.at(0) = FatJets->at(j).nConstituents;
		inputValues.at(1) = FatJets->at(j).nTrackConstituents;
		inputValues.at(2) = FatJets->at(j).timeRecHitsEB;
		inputValues.at(3) = FatJets->at(j).eFracRecHitsEB;
		inputValues.at(4) = FatJets->at(j).nRecHitsEB;
		inputValues.at(5) = FatJets->at(j).cHadEFrac;
		inputValues.at(6) = FatJets->at(j).nHadEFrac;
		inputValues.at(7) = FatJets->at(j).eleEFrac;
		inputValues.at(8) = FatJets->at(j).photonEFrac;
		inputValues.at(9) = FatJets->at(j).ptPVTracksMax;
		inputValues.at(10) = FatJets->at(j).gammaMaxET;
		inputValues.at(11) = FatJets->at(j).minDeltaRAllTracks;
		inputValues.at(12) = FatJets->at(j).minDeltaRPVTracks;
		inputValues.at(13) = FatJets->at(j).chsTau21;
		inputValues.at(14) = FatJets->at(j).sig1EB;
		inputValues.at(15) = FatJets->at(j).sig2EB;
		inputValues.at(16) = FatJets->at(j).ptDEB;

		float* d = inputTensorAK8.flat<float>().data();
		for (float v : inputValues) {
		  //std::cout<< " input value: " << v <<std::endl;
		  *d = v;
		  d++;
		}

		// run the inference
		std::vector<tensorflow::Tensor> outputsAK8;
		tensorflow::run(sessionAK8, {{inputTensorNameAK8, inputTensorAK8}}, {outputTensorNameAK8}, &outputsAK8, threadPool);

		// store the result
		outputValueAK8 = outputsAK8[0].matrix<float>()(0, 1);
		// keras cannot predict the output for invalid jets
		// fix it manually
		if(FatJets->at(j).pt<0) outputValueAK8 = -1;
		FatJets->at(j).sigprob = outputValueAK8;

		if(outputValueAK8>0.8) nTagFatJets_0p8++;
		if(outputValueAK8>0.9) nTagFatJets_0p9++;
		if(outputValueAK8>0.92) nTagFatJets_0p92++;
		if(outputValueAK8>0.95) nTagFatJets_0p95++;
		if(outputValueAK8>0.96) nTagFatJets_0p96++;
		if(outputValueAK8>0.97) nTagFatJets_0p97++;
		if(outputValueAK8>0.98) nTagFatJets_0p98++;
		if(outputValueAK8>0.99) nTagFatJets_0p99++;
		if(outputValueAK8>0.995) nTagFatJets_0p995++;
		if(outputValueAK8>0.997) nTagFatJets_0p997++;
		if(outputValueAK8>0.999) nTagFatJets_0p999++;
		if(outputValueAK8>0.9995) nTagFatJets_0p9995++;
		if(outputValueAK8>0.9999) nTagFatJets_0p9999++;
		if(outputValueAK8>0.99995) nTagFatJets_0p99995++;
		if(outputValueAK8>0.99999) nTagFatJets_0p99999++;
		if(outputValueAK8>0.999995) nTagFatJets_0p999995++;
		if(outputValueAK8>0.999999) nTagFatJets_0p999999++;

		//Classify boosted analysis
		//based on having a fat jet
		//with a certain pT
		if(FatJets->at(j).pt>170 and outputValueAK8>0.9999) isTagAK8_0p9999_170 = true;
		if(FatJets->at(j).pt>200 and outputValueAK8>0.9999) isTagAK8_0p9999_200 = true;
		if(FatJets->at(j).pt>250 and outputValueAK8>0.9999) isTagAK8_0p9999_250 = true;
		if(FatJets->at(j).pt>300 and outputValueAK8>0.9999) isTagAK8_0p9999_300 = true;
		if(FatJets->at(j).pt>350 and outputValueAK8>0.9999) isTagAK8_0p9999_350 = true;

		if(FatJets->at(j).pt>170 and outputValueAK8>0.99999) isTagAK8_0p99999_170 = true;
		if(FatJets->at(j).pt>200 and outputValueAK8>0.99999) isTagAK8_0p99999_200 = true;
		if(FatJets->at(j).pt>250 and outputValueAK8>0.99999) isTagAK8_0p99999_250 = true;
		if(FatJets->at(j).pt>300 and outputValueAK8>0.99999) isTagAK8_0p99999_300 = true;
		if(FatJets->at(j).pt>350 and outputValueAK8>0.99999) isTagAK8_0p99999_350 = true;

		if(FatJets->at(j).pt>170 and outputValueAK8>0.999995) isTagAK8_0p999995_170 = true;
		if(FatJets->at(j).pt>200 and outputValueAK8>0.999995) isTagAK8_0p999995_200 = true;
		if(FatJets->at(j).pt>250 and outputValueAK8>0.999995) isTagAK8_0p999995_250 = true;
		if(FatJets->at(j).pt>300 and outputValueAK8>0.999995) isTagAK8_0p999995_300 = true;
		if(FatJets->at(j).pt>350 and outputValueAK8>0.999995) isTagAK8_0p999995_350 = true;

		if(FatJets->at(j).pt>170 and outputValueAK8>0.999999) isTagAK8_0p999999_170 = true;
		if(FatJets->at(j).pt>200 and outputValueAK8>0.999999) isTagAK8_0p999999_200 = true;
		if(FatJets->at(j).pt>250 and outputValueAK8>0.999999) isTagAK8_0p999999_250 = true;
		if(FatJets->at(j).pt>300 and outputValueAK8>0.999999) isTagAK8_0p999999_300 = true;
		if(FatJets->at(j).pt>350 and outputValueAK8>0.999999) isTagAK8_0p999999_350 = true;


		//Redo gen-matchign to compute double matched jets
		int n_g = 0;
		for (unsigned int g=0; g<GenBquarks->size(); g++)
		  {
		    if(GenBquarks->at(g).travelRadiusLLP==FatJets->at(j).radiusLLP and FatJets->at(j).isGenMatchedCaloCorrLLPAccept)
		      {
			float dr = fabs(reco::deltaR(FatJets->at(j).eta, FatJets->at(j).phi, GenBquarks->at(g).eta, GenBquarks->at(g).phi)) ;
			if( dr<0.8 )
			  {
			    n_g++;
			  }
		      }
		  }

		FatJets->at(j).nMatchedGenBquarksCaloCorr = n_g;


		for (unsigned int l=0; l<GenLLPs->size(); l++)
		  {
		    //std::cout << "GenLLP travelRadius " << GenLLPs->at(l).travelRadius << std::endl;
		    //std::cout << "Jets->at(j).radiusLLP " << Jets->at(j).radiusLLP << std::endl;
		    if(GenLLPs->at(l).travelRadius==FatJets->at(j).radiusLLP and FatJets->at(j).isGenMatchedCaloCorrLLPAccept)
		      {
			//std::cout << "LLP travel radius: " << GenLLPs->at(l).travelRadius << std::endl;
			//std::cout << "LLP dRdaughters: " << GenLLPs->at(l).dRdaughters << std::endl;
			if(j==0)
			  {
			    dR_bb_fatjet_0_den->Fill(GenLLPs->at(l).dRdaughters);
			    dR_bb_fatjet_0 = GenLLPs->at(l).dRdaughters;
			  }
			if(j==0 && FatJets->at(j).sigprob>0.99999 && FatJets->at(j).pt>250) dR_bb_fatjet_0_num->Fill(GenLLPs->at(l).dRdaughters);
			//Jets->at(j).dR_bb = GenLLPs->at(l).dRdaughters;
		      }
		  }
		

		if(fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi)) < MinFatJetMetDPhiBarrelMatched && FatJets->at(j).nMatchedGenBquarksCaloCorr==2) MinFatJetMetDPhiBarrelMatched = fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi));
                nCHSFatJetsAcceptanceCalo++;
                skimmedFatJets.push_back(FatJets->at(j));
                validFatJetIndex.push_back(j);
              }
          }

	//Define categories
        if(nTagJets_0p99>1) isTagAK4_0p99 = true;
        if(nTagJets_0p994>1) isTagAK4_0p994 = true;
        if(nTagJets_0p996>1) isTagAK4_0p996 = true;
        if(nTagJets_0p997>1) isTagAK4_0p997 = true;
        if(nTagJets_0p99_JJ>1) isTagAK4_0p99_JJ = true;
        if(nTagJets_0p994_JJ>1) isTagAK4_0p994_JJ = true;
        if(nTagJets_0p996_JJ>1) isTagAK4_0p996_JJ = true;
        if(nTagJets_0p997_JJ>1) isTagAK4_0p997_JJ = true;


	//No jets in acceptance, go to next event
	if(nCHSJetsAcceptanceCalo==0 and nCHSFatJetsAcceptanceCalo==0) continue;
	n_pass->Fill(0.);
	if(EventNumber % 2 == 0) n_even->Fill(0.);
	if(EventNumber % 2 != 0) n_odd->Fill(0.);


	//Observed worse agreement, skip this --> redo
	if(skipTrain==true and EventNumber % 2 == 0) continue;
	outputTree->Fill();

    }


    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
    n_pass->Write();
    n_odd->Write();
    n_even->Write();
    b_skipTrain->Write();
    dR_bb_jet_0_den->Write();
    dR_bb_jet_0_num->Write();
    dR_bb_fatjet_0_den->Write();
    dR_bb_fatjet_0_num->Write();
    outputFile->Write();
    outputFile->Close();
    inputFile->Close();

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "finished  computations at " << std::ctime(&end_time)
	      << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "**************************************************" << std::endl;

    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
