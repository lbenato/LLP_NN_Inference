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

//JJ:
struct Particle {
  TLorentzVector vec;
  int charge;
  int pdgId;
};

bool pt_sorter(const PFCandidateType& x, const PFCandidateType& y) { return x.pt > y.pt; }
bool energy_sorter(const ecalRecHitType& x, const ecalRecHitType& y) { return x.energy > y.energy; }
//bool h_energy_sorter(const hcalRecHitType& x, const hcalRecHitType& y) { return x.energy > y.energy; }

void NormalizeHist(TH1F *hist)
{
  Double_t norm = 0;
  for (UInt_t b=0; int(b)<hist->GetXaxis()->GetNbins()+2; ++b) {
    norm += hist->GetBinContent(b);
  }
  for (UInt_t b=0; int(b)<hist->GetXaxis()->GetNbins()+2; ++b) {
    hist->SetBinContent(b,hist->GetBinContent(b) / norm);
    hist->SetBinError(b,hist->GetBinError(b) / norm);
  }
}

void DivideHist(TH1 *ratio, TH1 *num, TH1 *den) {
  for (UInt_t b=0; int(b)<num->GetXaxis()->GetNbins()+2; ++b) {
    if ( den->GetBinContent(b) > 1.0e-4 ) {
      //debug: //std::cout << "Bin: " << b << " " << ratio->GetXaxis()->GetBinCenter(b) << " : " << num->GetBinContent(b) << " / " << den->GetBinContent(b) << " = " << num->GetBinContent(b) / den->GetBinContent(b) << "\n";
      ratio->SetBinContent(b,num->GetBinContent(b) / den->GetBinContent(b));
      ratio->SetBinError(b, (num->GetBinContent(b) / den->GetBinContent(b))*sqrt( pow(num->GetBinError(b)/num->GetBinContent(b),2) + pow(den->GetBinError(b)/den->GetBinContent(b),2)));
    } else {
      ratio->SetBinContent(b,0);
      ratio->SetBinError(b,0);
    }
  }
}

void MultiplyHist(TH1 *mult, TH1 *fact1, TH1 *fact2) {
  for (UInt_t b=0; int(b)<fact1->GetXaxis()->GetNbins()+2; ++b) {
    if ( fact2->GetBinContent(b) > 1.0e-4 ) {
      mult->SetBinContent(b,fact1->GetBinContent(b) * fact2->GetBinContent(b));
      mult->SetBinError(b, (fact1->GetBinContent(b) * fact2->GetBinContent(b))*sqrt( pow(fact1->GetBinError(b)*fact1->GetBinContent(b),2) + pow(fact2->GetBinError(b)*fact2->GetBinContent(b),2)));
    } else {
      mult->SetBinContent(b,0);
      mult->SetBinError(b,0);
    }
  }
}

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

    const float ELE_MASS = 0.000511;
    const float MU_MASS  = 0.105658;
    const float TAU_MASS  = 1.77686;
    const float Z_MASS   = 91.2;

    if(argc<10)
    //if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    bool skipTrain(false);
    if(strcmp(argv[3], "y")==0 || strcmp(argv[3], "yes")==0) skipTrain=true;
    bool isSignal(false);
    if(strcmp(argv[4], "True")==0) isSignal=true;
    if(strcmp(argv[4], "true")==0) isSignal=true;
    bool isData(false);
    if(strcmp(argv[5], "True")==0) isData=true;
    if(strcmp(argv[5], "true")==0) isData=true;

    //Flags for SR/CR
    bool doSR(false);
    if(strcmp(argv[9], "doSR")==0) doSR=true;

    bool isVerbose(false);

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Skip even EventNumber: " << skipTrain << std::endl;
    std::cout << "isSignal: " << isSignal << std::endl;
    std::cout << "isData: " << isData << std::endl;
    std::cout << "MC PU file: " << argv[6] << std::endl;
    std::cout << "MC trigger file: " << argv[7] << std::endl;
    std::cout << "MC trigger string: " << argv[8] << std::endl;
    if(doSR) std::cout << "SR selections" << std::endl;

    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    // input and output file settings
    //unskimmed crab output
    //std::string inputPath = "/pnfs/desy.de/cms/tier2/store/user/lbenato/v4_calo_AOD_2018_18October2020/GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/crab_GluGluH2_H2ToSSTobbbb_MH-2000_MS-250_ctauS-1000_TuneCP5_13TeV-pythia8_PRIVATE-MC/201017_234633/0000/output_1.root";
    std::string inputPath = argv[1];

    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";

    std::string mcPUFilename = argv[6];
    std::string mcTriggerFilename = argv[7];
    std::string mcTriggerString = argv[8];
    //std::string dataFilename = argv[5];
    //std::string dataFilenameUp = argv[6];
    //std::string dataFilenameDown = argv[7];

    //std::string inputTreeName = "skim";
    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;

    bool doPFCand=false;

    // model and inference settings
    std::string graphPathAK4 = basePath + "/tagger_AK4_v3/graph.pb";
    std::string MetaDataFileAK4 = basePath + "/tagger_AK4_v3/metadata.dat";
    std::string inputTensorNameAK4 = "input_input";
    std::string outputTensorNameAK4 = "FCN/output/Softmax";//"FCN/dense_4/Softmax";//or Softmax?

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
    n_odd->Sumw2();
    n_even->Sumw2();
    n_pass->Sumw2();
    b_skipTrain->Sumw2();
    float  tree_weight = inputTree->GetWeight();
    if(isVerbose) std::cout << "Tree weight: " << tree_weight << std::endl;

    if(skipTrain) b_skipTrain->Fill(0);

    TFile *mcPUFile = TFile::Open(mcPUFilename.data(),"READ"); if (!mcPUFile) return 0;
    TH1F  *pu = (TH1F*)mcPUFile->Get("PileupReweight");
    TH1F  *pu_up = (TH1F*)mcPUFile->Get("PileupReweightSysUp");
    TH1F  *pu_down = (TH1F*)mcPUFile->Get("PileupReweightSysDown");
    if(isVerbose) std::cout<< "PU histo loaded" << std::endl;

    TFile *mcTriggerFile = TFile::Open(mcTriggerFilename.data(),"READ"); if (!mcTriggerFile) return 0;
    TH1F  *tr = (TH1F*)mcTriggerFile->Get(mcTriggerString.c_str());
    if(isVerbose) std::cout<< "Trigger histo loaded" << std::endl;


    // Input variables
    Long64_t EventNumber;
    Long64_t RunNumber;
    Long64_t LumiNumber;
    float    EventWeight;
    float    PUWeight;
    bool   isMC;
    bool   isVBF;
    int    MeanNumInteractions;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v(false);
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v(false);
    //Mu CR
    bool   HLT_IsoMu24_v(false);
    bool   HLT_IsoMu27_v(false);
    bool   HLT_IsoMu24_eta2p1_v(false);//partially prescaled in 2018
    //Ele CR
    bool   HLT_Ele32_WPTight_Gsf_v(false);
    bool   HLT_Ele32_eta2p1_WPLoose_Gsf_v(false);//not available in 2018
    bool   HLT_Ele35_WPTight_Gsf_v(false);
    //E-MU CR
    bool   HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v(false);//not available in 2018
    bool   HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v(false);//not available in 2018
    bool   HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v(false);
    bool   HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v(false);
    bool   HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v(false);//not available in 2018
    bool   HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v(false);//not available in 2018
    bool   HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v(false);//not available in 2018
    bool   HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v(false);//not available in 2018
    bool   HLT_Mu27_Ele37_CaloIdL_MW_v(false);
    bool   HLT_Mu37_Ele27_CaloIdL_MW_v(false);
    //Photon CR
    bool   HLT_Photon22_v(false);//not available in 2018
    bool   HLT_Photon30_v(false);//not available in 2018
    bool   HLT_Photon33_v(false);
    bool   HLT_Photon36_v(false);//not available in 2018
    bool   HLT_Photon50_v(false);
    bool   HLT_Photon75_v(false);
    bool   HLT_Photon90_v(false);
    bool   HLT_Photon120_v(false);
    bool   HLT_Photon125_v(false);//not available in 2018
    bool   HLT_Photon150_v(false);
    bool   HLT_Photon175_v(false);
    bool   HLT_Photon200_v(false);//unprescaled
    bool   HLT_Photon250_NoHE_v(false);//not available in 2018
    bool   HLT_Photon300_NoHE_v(false);
    bool   HLT_Photon500_v(false);//not available in 2018
    bool   HLT_Photon600_v(false);//not available in 2018
    //JetHT
    bool   HLT_DiPFJetAve40_v(false);
    bool   HLT_DiPFJetAve60_v(false);
    bool   HLT_DiPFJetAve80_v(false);
    bool   HLT_DiPFJetAve200_v(false);
    bool   HLT_DiPFJetAve500_v(false);
    bool   HLT_PFJet40_v(false);
    bool   HLT_PFJet60_v(false);
    bool   HLT_PFJet80_v(false);
    bool   HLT_PFJet140_v(false);
    bool   HLT_PFJet200_v(false);
    bool   HLT_PFJet260_v(false);
    bool   HLT_PFJet320_v(false);
    bool   HLT_PFJet400_v(false);
    bool   HLT_PFJet450_v(false);
    bool   HLT_PFJet500_v(false);//unprescaled
    bool   HLT_PFJet550_v(false);//unprescaled
    bool   HLT_AK8PFJet40_v(false);
    bool   HLT_AK8PFJet60_v(false);
    bool   HLT_AK8PFJet80_v(false);
    bool   HLT_AK8PFJet200_v(false);
    bool   HLT_AK8PFJet500_v(false);//unprescaled
    bool   HLT_AK8PFJet550_v(false);//unprescaled  

    bool   Flag2_globalSuperTightHalo2016Filter;
    bool   Flag2_goodVertices;
    bool   Flag2_EcalDeadCellTriggerPrimitiveFilter;
    bool   Flag2_HBHENoiseFilter;
    bool   Flag2_HBHEIsoNoiseFilter;
    bool   Flag2_ecalBadCalibFilter;
    bool   Flag2_eeBadScFilter;
    bool   Flag2_BadPFMuonFilter;
    bool   HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v(false);
    bool   HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v(false);
    float  HT;
    float  MinJetMetDPhi_ntuple;
    Long64_t nCHSJets;
    Long64_t nCHSFatJets;
    Long64_t nPV;
    int    nElectrons;
    int    nMuons;
    int    nPhotons;
    //int    nTaus;
    int    nPFCandidates;
    int    nPFCandidatesTrack;
    std::vector<TauType>         *Taus = 0;
    std::vector<PhotonType>      *Photons = 0;
    std::vector<LeptonType>      *Muons = 0;
    std::vector<LeptonType>      *Electrons = 0;
    std::vector<JetType>         *Jets = 0;
    MEtType                      *MEt = 0;
    std::vector<GenPType>        *GenHiggs = 0;
    std::vector<GenPType>        *GenLLPs = 0;
    std::vector<GenPType>        *GenBquarks = 0;

    // Input branches
    TBranch        *b_Taus = 0;
    TBranch        *b_Photons = 0;
    TBranch        *b_Muons = 0;
    TBranch        *b_Electrons = 0;
    TBranch        *b_Jets = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_PUWeight;
    TBranch        *b_isMC;
    TBranch        *b_isVBF;
    TBranch        *b_MeanNumInteractions;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v;
    TBranch        *b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v;

    //Mu CR
    TBranch        *b_HLT_IsoMu24_v;
    TBranch        *b_HLT_IsoMu27_v;
    TBranch        *b_HLT_IsoMu24_eta2p1_v;//partially prescaled in 2018
    //Ele CR
    TBranch        *b_HLT_Ele32_WPTight_Gsf_v;
    TBranch        *b_HLT_Ele32_eta2p1_WPLoose_Gsf_v;//not available in 2018
    TBranch        *b_HLT_Ele35_WPTight_Gsf_v;
    //E-MU CR
    TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v;//not available in 2018
    TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v;//not available in 2018
    TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v;
    TBranch        *b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v;
    TBranch        *b_HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v;//not available in 2018
    TBranch        *b_HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v;//not available in 2018
    TBranch        *b_HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v;//not available in 2018
    TBranch        *b_HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v;//not available in 2018
    TBranch        *b_HLT_Mu27_Ele37_CaloIdL_MW_v;
    TBranch        *b_HLT_Mu37_Ele27_CaloIdL_MW_v;
    //Photon CR
    TBranch        *b_HLT_Photon22_v;//not available in 2018
    TBranch        *b_HLT_Photon30_v;//not available in 2018
    TBranch        *b_HLT_Photon33_v;
    TBranch        *b_HLT_Photon36_v;//not available in 2018
    TBranch        *b_HLT_Photon50_v;
    TBranch        *b_HLT_Photon75_v;
    TBranch        *b_HLT_Photon90_v;
    TBranch        *b_HLT_Photon120_v;
    TBranch        *b_HLT_Photon125_v;//not available in 2018
    TBranch        *b_HLT_Photon150_v;
    TBranch        *b_HLT_Photon175_v;
    TBranch        *b_HLT_Photon200_v;//unprescaled
    TBranch        *b_HLT_Photon250_NoHE_v;//not available in 2018
    TBranch        *b_HLT_Photon300_NoHE_v;
    TBranch        *b_HLT_Photon500_v;//not available in 2018
    TBranch        *b_HLT_Photon600_v;//not available in 2018
    //JetHT
    TBranch        *b_HLT_DiPFJetAve40_v;
    TBranch        *b_HLT_DiPFJetAve60_v;
    TBranch        *b_HLT_DiPFJetAve80_v;
    TBranch        *b_HLT_DiPFJetAve200_v;
    TBranch        *b_HLT_DiPFJetAve500_v;
    TBranch        *b_HLT_PFJet40_v;
    TBranch        *b_HLT_PFJet60_v;
    TBranch        *b_HLT_PFJet80_v;
    TBranch        *b_HLT_PFJet140_v;
    TBranch        *b_HLT_PFJet200_v;
    TBranch        *b_HLT_PFJet260_v;
    TBranch        *b_HLT_PFJet320_v;
    TBranch        *b_HLT_PFJet400_v;
    TBranch        *b_HLT_PFJet450_v;
    TBranch        *b_HLT_PFJet500_v;//unprescaled
    TBranch        *b_HLT_PFJet550_v;//unprescaled
    TBranch        *b_HLT_AK8PFJet40_v;
    TBranch        *b_HLT_AK8PFJet60_v;
    TBranch        *b_HLT_AK8PFJet80_v;
    TBranch        *b_HLT_AK8PFJet200_v;
    TBranch        *b_HLT_AK8PFJet500_v;//unprescaled
    TBranch        *b_HLT_AK8PFJet550_v;//unprescaled  


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
    TBranch        *b_nPV;
    TBranch        *b_nElectrons;
    TBranch        *b_nMuons;
    TBranch        *b_nPhotons;
    //TBranch        *b_nTaus;
    TBranch        *b_nPFCandidates;
    TBranch        *b_nPFCandidatesTrack;

    inputTree->SetBranchAddress("Taus",              &Taus,              &b_Taus);
    inputTree->SetBranchAddress("Photons",           &Photons,           &b_Photons);
    inputTree->SetBranchAddress("Muons",             &Muons,             &b_Muons);
    inputTree->SetBranchAddress("Electrons",         &Electrons,         &b_Electrons);
    inputTree->SetBranchAddress("Jets",              &Jets,              &b_Jets);
    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt); 
    inputTree->SetBranchAddress("GenHiggs",          &GenHiggs,          &b_GenHiggs); 
    inputTree->SetBranchAddress("GenLLPs",           &GenLLPs,           &b_GenLLPs); 
    inputTree->SetBranchAddress("GenBquarks",        &GenBquarks,        &b_GenBquarks); 
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("PUWeight",          &PUWeight,          &b_PUWeight);
    inputTree->SetBranchAddress("isMC",              &isMC,              &b_isMC);
    inputTree->SetBranchAddress("isVBF",             &isVBF,             &b_isVBF);
    inputTree->SetBranchAddress("MeanNumInteractions",  &MeanNumInteractions,  &b_MeanNumInteractions);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v);
    inputTree->SetBranchAddress("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, &b_HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v);

    inputTree->SetBranchAddress("HLT_IsoMu24_v", &HLT_IsoMu24_v, &b_HLT_IsoMu24_v);
    inputTree->SetBranchAddress("HLT_IsoMu27_v", &HLT_IsoMu27_v, &b_HLT_IsoMu27_v);
    inputTree->SetBranchAddress("HLT_IsoMu24_eta2p1_v", &HLT_IsoMu24_eta2p1_v, &b_HLT_IsoMu24_eta2p1_v);
    inputTree->SetBranchAddress("HLT_Ele32_WPTight_Gsf_v", &HLT_Ele32_WPTight_Gsf_v, &b_HLT_Ele32_WPTight_Gsf_v);
    inputTree->SetBranchAddress("HLT_Ele32_eta2p1_WPLoose_Gsf_v", &HLT_Ele32_eta2p1_WPLoose_Gsf_v, &b_HLT_Ele32_eta2p1_WPLoose_Gsf_v);
    inputTree->SetBranchAddress("HLT_Ele35_WPTight_Gsf_v", &HLT_Ele35_WPTight_Gsf_v , &b_HLT_Ele35_WPTight_Gsf_v);
    inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v", &HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v, &b_HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v);
    inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v", &HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v, &b_HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v);
    inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v, &b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v);
    inputTree->SetBranchAddress("HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v", &HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v, &b_HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v);
    inputTree->SetBranchAddress("HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v", &HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v, &b_HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v);
    inputTree->SetBranchAddress("HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v", &HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v, &b_HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v);
    inputTree->SetBranchAddress("HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v", &HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v, &b_HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v);
    inputTree->SetBranchAddress("HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v", &HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v, &b_HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v);
    inputTree->SetBranchAddress("HLT_Mu27_Ele37_CaloIdL_MW_v", &HLT_Mu27_Ele37_CaloIdL_MW_v, &b_HLT_Mu27_Ele37_CaloIdL_MW_v);
    inputTree->SetBranchAddress("HLT_Mu37_Ele27_CaloIdL_MW_v", &HLT_Mu37_Ele27_CaloIdL_MW_v, &b_HLT_Mu37_Ele27_CaloIdL_MW_v);
    //prescaled triggers
    if(isData or isSignal)
      {
	std::cout << " prescaled triggers? " << std::endl;
	inputTree->SetBranchAddress("HLT_Photon22_v", &HLT_Photon22_v, &b_HLT_Photon22_v);
	inputTree->SetBranchAddress("HLT_Photon30_v", &HLT_Photon30_v, &b_HLT_Photon30_v);
	inputTree->SetBranchAddress("HLT_Photon33_v", &HLT_Photon33_v, &b_HLT_Photon33_v);
	inputTree->SetBranchAddress("HLT_Photon36_v", &HLT_Photon36_v, &b_HLT_Photon36_v);
	inputTree->SetBranchAddress("HLT_Photon50_v", &HLT_Photon50_v, &b_HLT_Photon50_v);
	inputTree->SetBranchAddress("HLT_Photon75_v", &HLT_Photon75_v, &b_HLT_Photon75_v);
	inputTree->SetBranchAddress("HLT_Photon90_v", &HLT_Photon90_v, &b_HLT_Photon90_v);
	inputTree->SetBranchAddress("HLT_Photon120_v", &HLT_Photon120_v, &b_HLT_Photon120_v);
	inputTree->SetBranchAddress("HLT_Photon125_v", &HLT_Photon125_v, &b_HLT_Photon125_v);
	inputTree->SetBranchAddress("HLT_Photon150_v", &HLT_Photon150_v, &b_HLT_Photon150_v);
	inputTree->SetBranchAddress("HLT_Photon175_v", &HLT_Photon175_v, &b_HLT_Photon175_v);//unprescaled??
      }
    inputTree->SetBranchAddress("HLT_Photon200_v", &HLT_Photon200_v, &b_HLT_Photon200_v);
    inputTree->SetBranchAddress("HLT_Photon250_NoHE_v", &HLT_Photon250_NoHE_v, &b_HLT_Photon250_NoHE_v);
    inputTree->SetBranchAddress("HLT_Photon300_NoHE_v", &HLT_Photon300_NoHE_v, &b_HLT_Photon300_NoHE_v);
    inputTree->SetBranchAddress("HLT_Photon500_v", &HLT_Photon500_v, &b_HLT_Photon500_v);
    inputTree->SetBranchAddress("HLT_Photon600_v", &HLT_Photon600_v, &b_HLT_Photon600_v);
    if(isData or isSignal)
      {
	inputTree->SetBranchAddress("HLT_DiPFJetAve40_v", &HLT_DiPFJetAve40_v, &b_HLT_DiPFJetAve40_v);
	inputTree->SetBranchAddress("HLT_DiPFJetAve60_v", &HLT_DiPFJetAve60_v, &b_HLT_DiPFJetAve60_v);
	inputTree->SetBranchAddress("HLT_DiPFJetAve80_v", &HLT_DiPFJetAve80_v, &b_HLT_DiPFJetAve80_v);
	inputTree->SetBranchAddress("HLT_DiPFJetAve200_v", &HLT_DiPFJetAve200_v, &b_HLT_DiPFJetAve200_v);
	inputTree->SetBranchAddress("HLT_DiPFJetAve500_v", &HLT_DiPFJetAve500_v, &b_HLT_DiPFJetAve500_v);
	inputTree->SetBranchAddress("HLT_PFJet40_v", &HLT_PFJet40_v, &b_HLT_PFJet40_v);
	inputTree->SetBranchAddress("HLT_PFJet60_v", &HLT_PFJet60_v, &b_HLT_PFJet60_v);
	inputTree->SetBranchAddress("HLT_PFJet80_v", &HLT_PFJet80_v, &b_HLT_PFJet80_v);
	inputTree->SetBranchAddress("HLT_PFJet140_v", &HLT_PFJet140_v, &b_HLT_PFJet140_v);
	inputTree->SetBranchAddress("HLT_PFJet200_v", &HLT_PFJet200_v, &b_HLT_PFJet200_v);
	inputTree->SetBranchAddress("HLT_PFJet260_v", &HLT_PFJet260_v, &b_HLT_PFJet260_v);
	inputTree->SetBranchAddress("HLT_PFJet320_v", &HLT_PFJet320_v, &b_HLT_PFJet320_v);
	inputTree->SetBranchAddress("HLT_PFJet400_v", &HLT_PFJet400_v, &b_HLT_PFJet400_v);
	inputTree->SetBranchAddress("HLT_PFJet450_v", &HLT_PFJet450_v, &b_HLT_PFJet450_v);
	inputTree->SetBranchAddress("HLT_AK8PFJet40_v", &HLT_AK8PFJet40_v, &b_HLT_AK8PFJet40_v);
	inputTree->SetBranchAddress("HLT_AK8PFJet60_v", &HLT_AK8PFJet60_v, &b_HLT_AK8PFJet60_v);
	inputTree->SetBranchAddress("HLT_AK8PFJet80_v", &HLT_AK8PFJet80_v, &b_HLT_AK8PFJet80_v);
	inputTree->SetBranchAddress("HLT_AK8PFJet200_v", &HLT_AK8PFJet200_v, &b_HLT_AK8PFJet200_v);
      }
    inputTree->SetBranchAddress("HLT_PFJet500_v", &HLT_PFJet500_v, &b_HLT_PFJet500_v);
    inputTree->SetBranchAddress("HLT_PFJet550_v", &HLT_PFJet550_v, &b_HLT_PFJet550_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet500_v", &HLT_AK8PFJet500_v, &b_HLT_AK8PFJet500_v);
    inputTree->SetBranchAddress("HLT_AK8PFJet550_v", &HLT_AK8PFJet550_v, &b_HLT_AK8PFJet550_v);

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
    inputTree->SetBranchAddress("MinJetMetDPhi",     &MinJetMetDPhi_ntuple,     &b_MinJetMetDPhi);
    inputTree->SetBranchAddress("nPV",          &nPV,          &b_nPV);
    inputTree->SetBranchAddress("nCHSJets",          &nCHSJets,          &b_nCHSJets);
    inputTree->SetBranchAddress("nCHSFatJets",       &nCHSFatJets,       &b_nCHSFatJets);
    inputTree->SetBranchAddress("nElectrons",        &nElectrons,        &b_nElectrons);
    inputTree->SetBranchAddress("nMuons",            &nMuons,            &b_nMuons);
    inputTree->SetBranchAddress("nPhotons",          &nPhotons,          &b_nPhotons);
    //inputTree->SetBranchAddress("nTaus",             &nTaus,             &b_nTaus);
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
    bool isSR(false);

    std::vector<TauType>    skimmedTaus;
    std::vector<JetType>    skimmedJets;

    float PUReWeight(1.);
    float PUReWeightUp(1.);
    float PUReWeightDown(1.);
    float TriggerWeight(1.);
    float MinLeadingJetMetDPhi(-1.);
    float MinSubLeadingJetMetDPhi(-1.);
    float MinSubSubLeadingJetMetDPhi(-1.);
    float MinJetMetDPhi(10.);
    float MinJetMetDPhiStar(10.);
    float MinJetMetDPhiBarrel(10.);
    float MinJetMetDPhiBarrelStar(10.);

    float dPhi(-9.);
    float MT(-1.);
    float Z_mass(-1.);
    float Z_pt(-1.);
    float Z_phi(-1.);
    float Z_eta(-1.);
    float Z_lep0_pt(-1.);
    float Z_lep0_phi(-9.);
    float Z_lep0_eta(-9.);
    float Z_lep1_pt(-1.);
    float Z_lep1_phi(-9.);
    float Z_lep1_eta(-9.);

    int nLeptons(0);
    std::vector<int> LepPdgId;
    std::vector<int> LepCharge;
    std::vector<float> LepPt;
    std::vector<float> LepEta;
    std::vector<float> LepPhi;
    std::vector<float> LepMass;

    int nTaus(0);
    int nTausPreVeto(0);
    //int nPhotons(0);
    //int nMuons(0);
    int nMuonsPassing(0);
    int nElectronsPassing(0);
    int nPhotonsPassing(0);
    int nPhotonsTight(0);
    int nTausPassing(0);

    int nCHSJetsAcceptanceCalo;
    int nCHSJets_in_HEM(0);

    int nCHSJets_in_HEM_pt_20_all_eta(0);
    int nCHSJets_in_HEM_pt_30_all_eta(0);

    int nCHSJets_in_HEM_pt_20_eta_2p4(0);
    int nCHSJets_in_HEM_pt_30_eta_2p4(0);
    int nCHSJets_in_HEM_eta_2p5(0);
    int nPhotons_in_HEM(0);
    int nElectrons_in_HEM(0);
    bool RunNumber_in_HEM(false);

    //float AK4_jet_width_ECAL(0.);
    //float AK8_jet_width_ECAL(0.);
    //float AK4_jet_width_HCAL(0.);
    //float AK8_jet_width_HCAL(0.);

    int nTagJets_cutbased(0);
    int nTagJets_0p996(0);

    int nTagJets_cutbased_JJ(0);
    int nTagJets_0p996_JJ(0);

    //Systematic uncertainties
    int nTagJets_0p996_JER(0);
    int nTagJets_0p996_JESUp(0);
    int nTagJets_0p996_JESDown(0);
    int nTagJets_0p996_JERUp(0);
    int nTagJets_0p996_JERDown(0);
    int nTagJets_0p996_UnclusteredEnUp(0);
    int nTagJets_0p996_UnclusteredEnDown(0);

    float MinJetMetDPhi_JER(10.);
    float MinJetMetDPhi_JESUp(10.);
    float MinJetMetDPhi_JESDown(10.);
    float MinJetMetDPhi_JERUp(10.);
    float MinJetMetDPhi_JERDown(10.);
    float MinJetMetDPhi_UnclusteredEnUp(10.);
    float MinJetMetDPhi_UnclusteredEnDown(10.);

    float TriggerWeight_JER(1.);
    float TriggerWeight_JESUp(1.);
    float TriggerWeight_JESDown(1.);
    float TriggerWeight_JERUp(1.);
    float TriggerWeight_JERDown(1.);
    float TriggerWeight_UnclusteredEnUp(1.);
    float TriggerWeight_UnclusteredEnDown(1.);


    // Output branches 
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("RunNumber",         &RunNumber,         "RunNumber/L");
    outputTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    outputTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
    outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    outputTree->Branch("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    outputTree->Branch("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isSR",              &isSR,              "isSR/O");
    outputTree->Branch("isVBF",             &isVBF,             "isVBF/O");
    outputTree->Branch("MeanNumInteractions",             &MeanNumInteractions,             "MeanNumInteractions/I");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");
    outputTree->Branch("Flag2_globalSuperTightHalo2016Filter", &Flag2_globalSuperTightHalo2016Filter, "Flag2_globalSuperTightHalo2016Filter/O");

    if(isData or isSignal)
      {
	outputTree->Branch("HLT_DiPFJetAve40_v", &HLT_DiPFJetAve40_v, "HLT_DiPFJetAve40_v/O");
	outputTree->Branch("HLT_DiPFJetAve60_v", &HLT_DiPFJetAve60_v, "HLT_DiPFJetAve60_v/O");
	outputTree->Branch("HLT_DiPFJetAve80_v", &HLT_DiPFJetAve80_v, "HLT_DiPFJetAve80_v/O");
	outputTree->Branch("HLT_DiPFJetAve200_v", &HLT_DiPFJetAve200_v, "HLT_DiPFJetAve200_v/O");
	outputTree->Branch("HLT_DiPFJetAve500_v", &HLT_DiPFJetAve500_v, "HLT_DiPFJetAve500_v/O");
	outputTree->Branch("HLT_PFJet40_v", &HLT_PFJet40_v, "HLT_PFJet40_v/O");
	outputTree->Branch("HLT_PFJet60_v", &HLT_PFJet60_v, "HLT_PFJet60_v/O");
	outputTree->Branch("HLT_PFJet80_v", &HLT_PFJet80_v, "HLT_PFJet80_v/O");
	outputTree->Branch("HLT_PFJet140_v", &HLT_PFJet140_v, "HLT_PFJet140_v/O");
	outputTree->Branch("HLT_PFJet200_v", &HLT_PFJet200_v, "HLT_PFJet200_v/O");
	outputTree->Branch("HLT_PFJet260_v", &HLT_PFJet260_v, "HLT_PFJet260_v/O");
	outputTree->Branch("HLT_PFJet320_v", &HLT_PFJet320_v, "HLT_PFJet320_v/O");
	outputTree->Branch("HLT_PFJet400_v", &HLT_PFJet400_v, "HLT_PFJet400_v/O");
	outputTree->Branch("HLT_PFJet450_v", &HLT_PFJet450_v, "HLT_PFJet450_v/O");
	outputTree->Branch("HLT_AK8PFJet40_v", &HLT_AK8PFJet40_v, "HLT_AK8PFJet40_v/O");
	outputTree->Branch("HLT_AK8PFJet60_v", &HLT_AK8PFJet60_v, "HLT_AK8PFJet60_v/O");
	outputTree->Branch("HLT_AK8PFJet80_v", &HLT_AK8PFJet80_v, "HLT_AK8PFJet80_v/O");
	outputTree->Branch("HLT_AK8PFJet200_v", &HLT_AK8PFJet200_v, "HLT_AK8PFJet200_v/O");
      }
    outputTree->Branch("HLT_PFJet500_v", &HLT_PFJet500_v, "HLT_PFJet500_v/O");
    outputTree->Branch("HLT_PFJet550_v", &HLT_PFJet550_v, "HLT_PFJet550_v/O");
    outputTree->Branch("HLT_AK8PFJet500_v", &HLT_AK8PFJet500_v, "HLT_AK8PFJet500_v/O");
    outputTree->Branch("HLT_AK8PFJet550_v", &HLT_AK8PFJet550_v, "HLT_AK8PFJet550_v/O");

    outputTree->Branch("HT",                &HT,                "HT/F");
    outputTree->Branch("MT",                &MT,                "MT/F");
    outputTree->Branch("dPhi",              &dPhi,              "dPhi/F");
    outputTree->Branch("Z_mass",            &Z_mass,            "Z_mass/F");
    outputTree->Branch("Z_pt",              &Z_pt,              "Z_pt/F");
    outputTree->Branch("Z_phi",             &Z_phi,             "Z_phi/F");
    outputTree->Branch("Z_eta",             &Z_eta,             "Z_eta/F");
    outputTree->Branch("Z_lep0_pt",         &Z_lep0_pt,         "Z_lep0_pt/F");
    outputTree->Branch("Z_lep0_phi",        &Z_lep0_phi,        "Z_lep0_phi/F");
    outputTree->Branch("Z_lep0_eta",        &Z_lep0_eta,        "Z_lep0_eta/F");
    outputTree->Branch("Z_lep1_pt",         &Z_lep1_pt,         "Z_lep1_pt/F");
    outputTree->Branch("Z_lep1_phi",        &Z_lep1_phi,        "Z_lep1_phi/F");
    outputTree->Branch("Z_lep1_eta",        &Z_lep1_eta,        "Z_lep1_eta/F");

    outputTree->Branch("nLeptons", &nLeptons, "nLeptons/I");
    outputTree->Branch("LepPdgId", &LepPdgId);
    outputTree->Branch("LepCharge", &LepCharge);
    outputTree->Branch("LepPt", &LepPt);
    outputTree->Branch("LepEta", &LepEta);
    outputTree->Branch("LepPhi", &LepPhi);
    outputTree->Branch("LepMass", &LepMass);


    outputTree->Branch("TriggerWeight_JER",     &TriggerWeight_JER,     "TriggerWeight_JER/F");
    outputTree->Branch("TriggerWeight_JESUp",     &TriggerWeight_JESUp,     "TriggerWeight_JESUp/F");
    outputTree->Branch("TriggerWeight_JESDown",     &TriggerWeight_JESDown,     "TriggerWeight_JESDown/F");
    outputTree->Branch("TriggerWeight_JERUp",     &TriggerWeight_JERUp,     "TriggerWeight_JERUp/F");
    outputTree->Branch("TriggerWeight_JERDown",     &TriggerWeight_JERDown,     "TriggerWeight_JERDown/F");
    outputTree->Branch("TriggerWeight_UnclusteredEnUp",     &TriggerWeight_UnclusteredEnUp,     "TriggerWeight_UnclusteredEnUp/F");
    outputTree->Branch("TriggerWeight_UnclusteredEnDown",     &TriggerWeight_UnclusteredEnDown,     "TriggerWeight_UnclusteredEnDown/F");

    outputTree->Branch("MinJetMetDPhi_ntuple",     &MinJetMetDPhi_ntuple,     "MinJetMetDPhi_ntuple/F");
    outputTree->Branch("MinJetMetDPhi",  &MinJetMetDPhi,  "MinJetMetDPhi/F");
    outputTree->Branch("MinJetMetDPhi_JER",  &MinJetMetDPhi_JER,  "MinJetMetDPhi_JER/F");
    outputTree->Branch("MinJetMetDPhi_JESUp",  &MinJetMetDPhi_JESUp,  "MinJetMetDPhi_JESUp/F");
    outputTree->Branch("MinJetMetDPhi_JESDown",  &MinJetMetDPhi_JESDown,  "MinJetMetDPhi_JESDown/F");
    outputTree->Branch("MinJetMetDPhi_JERUp",  &MinJetMetDPhi_JERUp,  "MinJetMetDPhi_JERUp/F");
    outputTree->Branch("MinJetMetDPhi_JERDown",  &MinJetMetDPhi_JERDown,  "MinJetMetDPhi_JERDown/F");
    outputTree->Branch("MinJetMetDPhi_UnclusteredEnUp",  &MinJetMetDPhi_UnclusteredEnUp,  "MinJetMetDPhi_UnclusteredEnUp/F");
    outputTree->Branch("MinJetMetDPhi_UnclusteredEnDown",  &MinJetMetDPhi_UnclusteredEnDown,  "MinJetMetDPhi_UnclusteredEnDown/F");
    outputTree->Branch("MinJetMetDPhiBarrel",  &MinJetMetDPhiBarrel,  "MinJetMetDPhiBarrel/F");
    outputTree->Branch("MinJetMetDPhiStar",  &MinJetMetDPhiStar,  "MinJetMetDPhiStar/F");
    outputTree->Branch("MinJetMetDPhiBarrelStar",  &MinJetMetDPhiBarrelStar,  "MinJetMetDPhiBarrelStar/F");
    outputTree->Branch("MinLeadingJetMetDPhi", &MinLeadingJetMetDPhi, "MinLeadingJetMetDPhi/F");
    outputTree->Branch("MinSubLeadingJetMetDPhi", &MinSubLeadingJetMetDPhi, "MinSubLeadingJetMetDPhi/F");
    outputTree->Branch("MinSubSubLeadingJetMetDPhi", &MinSubSubLeadingJetMetDPhi, "MinSubSubLeadingJetMetDPhi/F");
    outputTree->Branch("nPV",          &nPV,          "nPV/I");
    outputTree->Branch("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    outputTree->Branch("nCHSFatJets",       &nCHSFatJets,       "nCHSFatJets/I");
    outputTree->Branch("nCHSJetsAcceptanceCalo",          &nCHSJetsAcceptanceCalo,          "nCHSJetsAcceptanceCalo/I");
    outputTree->Branch("nCHSJets_in_HEM" , &nCHSJets_in_HEM, "nCHSJets_in_HEM/I");
    outputTree->Branch("nCHSJets_in_HEM_pt_20_all_eta" , &nCHSJets_in_HEM_pt_20_all_eta, "nCHSJets_in_HEM_pt_20_all_eta/I");
    outputTree->Branch("nCHSJets_in_HEM_pt_30_all_eta" , &nCHSJets_in_HEM_pt_30_all_eta, "nCHSJets_in_HEM_pt_30_all_eta/I");
    outputTree->Branch("nCHSJets_in_HEM_eta_2p5" , &nCHSJets_in_HEM_eta_2p5, "nCHSJets_in_HEM_eta_2p5/I");
    outputTree->Branch("nCHSJets_in_HEM_pt_20_eta_2p4" , &nCHSJets_in_HEM_pt_20_eta_2p4, "nCHSJets_in_HEM_pt_20_eta_2p4/I");
    outputTree->Branch("nCHSJets_in_HEM_pt_30_eta_2p4" , &nCHSJets_in_HEM_pt_30_eta_2p4, "nCHSJets_in_HEM_pt_30_eta_2p4/I");
    outputTree->Branch("nPhotons_in_HEM" , &nPhotons_in_HEM, "nPhotons_in_HEM/I");
    outputTree->Branch("nElectrons_in_HEM" , &nElectrons_in_HEM, "nElectrons_in_HEM/I");
    outputTree->Branch("RunNumber_in_HEM" , &RunNumber_in_HEM, "RunNumber_in_HEM/O");

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

    outputTree->Branch("nPFCandidates",     &nPFCandidates,     "nPFCandidates/I");
    outputTree->Branch("nPFCandidatesTrack", &nPFCandidatesTrack, "nPFCandidatesTrack/I");
    outputTree->Branch("Muons", &Muons);
    outputTree->Branch("Electrons", &Electrons);
    outputTree->Branch("Photons", &Photons);
    outputTree->Branch("Taus", &skimmedTaus);
    outputTree->Branch("Jets", &skimmedJets);

    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);


    outputTree->Branch("nTagJets_cutbased", &nTagJets_cutbased,  "nTagJets_cutbased/I");
    outputTree->Branch("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    outputTree->Branch("nTagJets_cutbased_JJ", &nTagJets_cutbased_JJ,  "nTagJets_cutbased_JJ/I");
    outputTree->Branch("nTagJets_0p996_JJ",     &nTagJets_0p996_JJ,      "nTagJets_0p996_JJ/I");
    outputTree->Branch("nTagJets_0p996_JER",     &nTagJets_0p996_JER,      "nTagJets_0p996_JER/I");
    outputTree->Branch("nTagJets_0p996_JESUp",     &nTagJets_0p996_JESUp,      "nTagJets_0p996_JESUp/I");
    outputTree->Branch("nTagJets_0p996_JESDown",     &nTagJets_0p996_JESDown,      "nTagJets_0p996_JESDown/I");
    outputTree->Branch("nTagJets_0p996_JERUp",     &nTagJets_0p996_JERUp,      "nTagJets_0p996_JERUp/I");
    outputTree->Branch("nTagJets_0p996_JERDown",     &nTagJets_0p996_JERDown,      "nTagJets_0p996_JERDown/I");
    outputTree->Branch("nTagJets_0p996_UnclusteredEnUp",     &nTagJets_0p996_UnclusteredEnUp,      "nTagJets_0p996_UnclusteredEnUp/I");
    outputTree->Branch("nTagJets_0p996_UnclusteredEnDown",     &nTagJets_0p996_UnclusteredEnDown,      "nTagJets_0p996_UnclusteredEnDown/I");



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
        nCHSJetsAcceptanceCalo = 0;
	nCHSJets_in_HEM = 0;
	nCHSJets_in_HEM_pt_20_all_eta = 0;
	nCHSJets_in_HEM_pt_30_all_eta = 0;
	nCHSJets_in_HEM_pt_20_eta_2p4 = 0;
	nCHSJets_in_HEM_pt_30_eta_2p4 = 0;
	nCHSJets_in_HEM_eta_2p5 = 0;
	nPhotons_in_HEM = 0;
	nElectrons_in_HEM = 0;
	RunNumber_in_HEM = false;

	MinLeadingJetMetDPhi = -1.;
	MinSubLeadingJetMetDPhi = -1.;
	MinSubSubLeadingJetMetDPhi = -1.;
	MinJetMetDPhi = 10.;
	MinJetMetDPhiBarrel = 10.;
	MinJetMetDPhiStar = 10.;
	MinJetMetDPhiBarrelStar = 10.;
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



	nTagJets_cutbased = 0;
	nTagJets_0p996 = 0;
	nTagJets_cutbased_JJ = 0;
	nTagJets_0p996_JJ = 0;


        TriggerWeight_JER = 1.;
        TriggerWeight_JESUp = 1.;
        TriggerWeight_JESDown = 1.;
        TriggerWeight_JERUp = 1.;
        TriggerWeight_JERDown = 1.;
        TriggerWeight_UnclusteredEnUp = 1.;
        TriggerWeight_UnclusteredEnDown = 1.;

	nTagJets_0p996_JER = 0;
	nTagJets_0p996_JESUp = 0;
	nTagJets_0p996_JESDown = 0;
	nTagJets_0p996_JERUp = 0;
	nTagJets_0p996_JERDown = 0;
	nTagJets_0p996_UnclusteredEnUp = 0;
	nTagJets_0p996_UnclusteredEnDown = 0;

	MinJetMetDPhi_JER = 10.;
	MinJetMetDPhi_JESUp = 10.;
	MinJetMetDPhi_JESDown = 10.;
	MinJetMetDPhi_JERUp = 10.;
	MinJetMetDPhi_JERDown = 10.;
	MinJetMetDPhi_UnclusteredEnUp = 10.;
	MinJetMetDPhi_UnclusteredEnDown = 10.;

	//Clear all the vectors
	//very dangerous with continue statement!
	skimmedTaus.clear();
        skimmedJets.clear();


	LepPdgId.clear();
	LepCharge.clear();
	LepPt.clear();
	LepEta.clear();
	LepPhi.clear();
	LepMass.clear();

        //if (i % 1000 == 0) {
        //    std::cout << "evaluating entry " << i << std::endl;
        //}
        inputTree->GetEntry(i);

	if(RunNumber>=319077)
	  {
	    RunNumber_in_HEM = true;
	  }
	//if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1)
	//{
	//if (EventNumber % 2 == 0)
	//{
	//std::cout << "Skip even EventNumber! " << std::endl;
	//continue;
	//}
	//}



	//Consider PU weight


	if(isMC)
	  {
	    TriggerWeight = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->pt));
	    PUReWeight = pu->GetBinContent(pu->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightUp = pu_up->GetBinContent(pu_up->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightDown = pu_down->GetBinContent(pu_down->GetXaxis()->FindBin(MeanNumInteractions));

	    TriggerWeight_JER = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->pt));
	    TriggerWeight_JESUp = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftJetEnUp));
	    TriggerWeight_JESDown = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftJetEnDown));
	    TriggerWeight_JERUp = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftJetResUp));
	    TriggerWeight_JERDown = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftJetResDown));
	    TriggerWeight_UnclusteredEnUp = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftUnclusteredEnUp));
	    TriggerWeight_UnclusteredEnDown = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->ptShiftUnclusteredEnDown));
	  }

	//MET filters always fulfilled
        if(!Flag2_globalSuperTightHalo2016Filter) continue;
        if(!Flag2_EcalDeadCellTriggerPrimitiveFilter) continue;
	if(!Flag2_HBHENoiseFilter) continue;
        if(!Flag2_HBHEIsoNoiseFilter) continue;
        if(!Flag2_ecalBadCalibFilter) continue;
        if(!Flag2_eeBadScFilter) continue;
        if(!Flag2_BadPFMuonFilter) continue;

	//Trigger selections
	//MET trigger always fulfilled, also for uncertainty
	if(doSR and not(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) ) continue;

	//Loop on veto objects
	//JJ
	std::vector<Particle> LeptonsStruct;
	std::vector<Particle> MuonsStruct;
	std::vector<Particle> ElectronsStruct;
	std::vector<Particle> PhotonsStruct;
	std::vector<Particle> TausStruct;
	//Muons
	float mu_iso = 0.4;
	//if(Muons->size()>1) std::cout << "Muons size: " << Muons->size() << std::endl;
	for (unsigned int m=0; m<Muons->size(); m++)
	  {
	    
	    //JJ:
	    //remove overlaps
	    ////////////////////////
	    bool overlap = false;
	    for(auto& lep : LeptonsStruct)
	      {
		if (reco::deltaR(Muons->at(m).eta,Muons->at(m).phi,lep.vec.Eta(),lep.vec.Phi()) < mu_iso) overlap = true;
	      }
	    if(overlap) continue;//wait!
	    
	    Particle tmpMuon;
	    tmpMuon.vec.SetPtEtaPhiM(Muons->at(m).pt,Muons->at(m).eta, Muons->at(m).phi, MU_MASS);
	    tmpMuon.pdgId = Muons->at(m).pdgId;
	    tmpMuon.charge = Muons->at(m).charge;

	    LeptonsStruct.push_back(tmpMuon);
	    MuonsStruct.push_back(tmpMuon);
	    nMuonsPassing++;
	  }

	//Electrons
	float ele_iso = 0.4;
	//if(Electrons->size()>0) std::cout << "Electrons size: " << Electrons->size() << std::endl;
	for (unsigned int e=0; e<Electrons->size(); e++)
	  {

	    if(Electrons->at(e).eta>-3. and Electrons->at(e).eta<-1.3 and Electrons->at(e).phi>-1.57 and Electrons->at(e).phi<-0.87)
	      {
		nElectrons_in_HEM++;
	      }

	    //remove overlaps
	    bool overlap = false;
	    for(auto& lep : LeptonsStruct)
	      {
		if (reco::deltaR(Electrons->at(e).eta, Electrons->at(e).phi,lep.vec.Eta(),lep.vec.Phi()) < ele_iso) overlap = true;
	      }
	    if(overlap) continue;//wait!

	    Particle tmpElectron;
	    tmpElectron.vec.SetPtEtaPhiM(Electrons->at(e).pt, Electrons->at(e).eta, Electrons->at(e).phi, ELE_MASS);
	    tmpElectron.pdgId = Electrons->at(e).pdgId;
	    tmpElectron.charge = Electrons->at(e).charge;
	    LeptonsStruct.push_back(tmpElectron);
	    ElectronsStruct.push_back(tmpElectron);
	    nElectronsPassing++;

	  }


	//Taus
	nTausPreVeto = int(Taus->size());
	float tau_iso = 0.5;
	for (unsigned int t=0; t<Taus->size(); t++)
	  {
	    //JJ uses "decayModeFindingNewDMs" and  byLoose, which is incorrect
	    //correct would be: "decayModeFinding"
	    if(Taus->at(t).decayModeFinding == true and Taus->at(t).byLooseCombinedIsolationDeltaBetaCorr3Hits == true)
	      {
		skimmedTaus.push_back(Taus->at(t));
		nTaus++;


		//remove overlaps
		bool overlap = false;
		for(auto& lep : LeptonsStruct)
		  {
		  if (reco::deltaR(Taus->at(t).eta,Taus->at(t).phi,lep.vec.Eta(),lep.vec.Phi()) < tau_iso) overlap = true;
		  }
		if(overlap) continue;

		bool overlap_tau = false;
		for(auto& tau : TausStruct)
		  {
		    if (reco::deltaR(Taus->at(t).eta,Taus->at(t).phi,tau.vec.Eta(),tau.vec.Phi()) < tau_iso) overlap_tau = true;
		  }
		if(overlap_tau) continue;

		Particle tmpTau;
		tmpTau.vec.SetPtEtaPhiM(Taus->at(t).pt,Taus->at(t).eta,Taus->at(t).phi, TAU_MASS);
		tmpTau.pdgId = Taus->at(t).pdgId;
		tmpTau.charge = Taus->at(t).charge;
		TausStruct.push_back(tmpTau);
		nTausPassing++;
		
	      }

	  }
	//std::cout << "nTaus: " << nTaus << std::endl;
	//std::cout << "nTausPassing: " << nTausPassing << std::endl;


	//Photons
	float pho_iso = 0.4;
	for (unsigned int p=0; p<Photons->size(); p++)
          {
	    
	    if(Photons->at(p).eta>-3. and Photons->at(p).eta<-1.3 and Photons->at(p).phi>-1.57 and Photons->at(p).phi<-0.87)
	      {
		nPhotons_in_HEM++;
	      }

	    //remove overlaps
	    bool overlap = false;
	    for(auto& lep : LeptonsStruct)
	      {
		if (reco::deltaR(Photons->at(p).eta,Photons->at(p).phi,lep.vec.Eta(),lep.vec.Phi()) < pho_iso) overlap = true;
	      }
	    if(overlap) continue;

	    bool overlap_tau = false;
	    for(auto& tau : TausStruct)
	      {
		if (reco::deltaR(Photons->at(p).eta,Photons->at(p).phi,tau.vec.Eta(),tau.vec.Phi()) < pho_iso) overlap_tau = true;
	      }
	    if(overlap_tau) continue;

	    bool overlap_pho = false;
	    for(auto& pho : PhotonsStruct)
	      {
		if (reco::deltaR(Photons->at(p).eta,Photons->at(p).phi,pho.vec.Eta(),pho.vec.Phi()) < pho_iso) overlap_pho = true;
	      }
	    if(overlap_pho) continue;

	    Particle tmpPhoton;
	    tmpPhoton.vec.SetPtEtaPhiM(Photons->at(p).pt,Photons->at(p).eta,Photons->at(p).phi,0.);
	    tmpPhoton.pdgId = Photons->at(p).pdgId;
	    tmpPhoton.charge = Photons->at(p).charge;
	    PhotonsStruct.push_back(tmpPhoton);
	    if(Photons->at(p).isTight) nPhotonsTight++;
	    nPhotonsPassing++;
	  }


	//Lepton veto always fulfilled, also for uncertainty
	//Veto objects
	if(doSR and nMuonsPassing!=0) continue;
	if(doSR and nElectronsPassing!=0) continue;
	if(doSR and nTausPassing!=0) continue;
	if(doSR and nPhotonsPassing!=0) continue;

	//Transverse mass met + Leptons (e and mu)
	TLorentzVector lepp4;
	for ( auto &tmp : LeptonsStruct )
	  {
	    lepp4 += tmp.vec;
	  }
	dPhi = reco::deltaPhi(MEt->phi, lepp4.Phi());
	MT = sqrt(2*(MEt->pt)*lepp4.Pt()*(1-cos(dPhi)));
	nLeptons = LeptonsStruct.size();

        if(isVerbose) std::cout << "======================================== " << std::endl;
        if(isVerbose) std::cout << "EventNumber " << EventNumber << "\tLumiNumber " << LumiNumber << std::endl;


	//Evaluate DNN for all jets
	//Then count based on cuts and uncertainties
	//And smear met in the same way
	//Also min jet met delta phi depends on JES/JER (changes the accounted jets)


	//Apply acceptance cuts to jets and fat jets 
	std::vector<int> validJetIndex;

	for (unsigned int j=0; j<Jets->size(); j++)
	  {

	    //HEM: reject events with jets in problematic region
	    /* can be ignored in signal
	    if(Jets->at(j).eta>-3. and Jets->at(j).eta<-1.3 and Jets->at(j).phi>-1.57 and Jets->at(j).phi<-0.87)
	      {
		nCHSJets_in_HEM++;
		
		if(Jets->at(j).pt>20) nCHSJets_in_HEM_pt_20_all_eta++;
		if(Jets->at(j).pt>30) nCHSJets_in_HEM_pt_30_all_eta++;

		if(Jets->at(j).pt>20 and fabs(Jets->at(j).eta)<2.4)
		  {
		    nCHSJets_in_HEM_pt_20_eta_2p4++;
		    if(Jets->at(j).pt>30)
		      {
			nCHSJets_in_HEM_pt_30_eta_2p4++;
		      }
		  }

	      }
	    if(Jets->at(j).eta>-3. and Jets->at(j).eta<-1.3 and Jets->at(j).phi>-1.57 and Jets->at(j).phi<-0.87 and fabs(Jets->at(j).eta)<2.5)
	      {
		nCHSJets_in_HEM_eta_2p5++;
	      }

	    */

	    //Correct acceptance for MinJetMetDPhi:
	    //Jet pt>30, Jet eta<2.4, MET>200
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30 and MEt->pt>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi) MinJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		TLorentzVector jet0;
		jet0.SetPtEtaPhiM(Jets->at(j).pt, 0, Jets->at(j).phi, 0 );
		TLorentzVector met;
		met.SetPtEtaPhiM(MEt->pt, 0, MEt->phi, 0 );
		if(fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi())) < MinJetMetDPhiStar) MinJetMetDPhiStar = fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi() ));
	      }

	    //Here effects of uncertainties on MinJetMetDPhi
	    //Remember objects affected: jets and met in a correlated way
	    //1. JER --> we don't have smeared MET but okay
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).ptJER>30 and MEt->pt>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_JER) MinJetMetDPhi_JER = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //2. JES up
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).ptJESUp>30 and MEt->ptShiftJetEnUp>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_JESUp) MinJetMetDPhi_JESUp = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //3. JES down
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).ptJESDown>30 and MEt->ptShiftJetEnDown>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_JESDown) MinJetMetDPhi_JESDown = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //4. JER up
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).ptJERUp>30 and MEt->ptShiftJetResUp>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_JERUp) MinJetMetDPhi_JERUp = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //5. JER down
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).ptJERDown>30 and MEt->ptShiftJetResDown>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_JERDown) MinJetMetDPhi_JERDown = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //6. unclustered up
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30 and MEt->ptShiftUnclusteredEnUp>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_UnclusteredEnUp) MinJetMetDPhi_UnclusteredEnUp = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    //7. unclustered down
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30 and MEt->ptShiftUnclusteredEnDown>200)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi_UnclusteredEnDown) MinJetMetDPhi_UnclusteredEnDown = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }


	    //Now loop on jets, remove any pt constraint, perform tensorflow for all at once
	    //if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1)//cleaned jets!
	    if( fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1)//cleaned jets!
	      {

		//Ignore jets overlapped to leptons, photons and taus
		//This also depends on jet pt but let's ignore it at the moment
		float jet_iso = 0.4;
		//Leptons
		float dR_lep = -1;
		for(auto& lep : LeptonsStruct){
		  float thisDR = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,lep.vec.Eta(),lep.vec.Phi());
		  if(dR_lep < 0 || thisDR < dR_lep) dR_lep = thisDR;
		}
		if(dR_lep > 0 && dR_lep < jet_iso) continue;

		//Taus
		float dR_tau = -1;
		for(auto& tau : TausStruct){
		  float thisDR_tau = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,tau.vec.Eta(),tau.vec.Phi());
		  if(dR_tau < 0 || thisDR_tau < dR_tau) dR_tau = thisDR_tau;
		}
		if(dR_tau > 0 && dR_tau < jet_iso) continue;

		//Photons
		float dR_pho = -1;
		for(auto& pho : PhotonsStruct){
		  float thisDR_pho = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,pho.vec.Eta(),pho.vec.Phi());
		  if(dR_pho < 0 || thisDR_pho < dR_pho) dR_pho = thisDR_pho;
		}
		if(dR_pho > 0 && dR_pho < jet_iso) continue;
		
		//Here: passed acceptance --> remember pt>30!
		if(Jets->at(j).pt>30) nCHSJetsAcceptanceCalo++;

		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		Jets->at(j).eFracRecHitsEB = (Jets->at(j).energy>0 and Jets->at(j).energyRecHitsEB>0) ? Jets->at(j).energyRecHitsEB/Jets->at(j).energy : -1.;
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

		// Here: apply all the uncertainty-dependent cuts!
		if(Jets->at(j).pt>30. and MEt->pt>200 and outputValueAK4>0.996) nTagJets_0p996_JJ++;

		//1. JER --> we don't have smeared MET but okay
		if(Jets->at(j).ptJER>30. and MEt->pt>200 and outputValueAK4>0.996) nTagJets_0p996_JER++;
		//2. JES up
		if(Jets->at(j).ptJESUp>30. and MEt->ptShiftJetEnUp>200 and outputValueAK4>0.996) nTagJets_0p996_JESUp++;
		//3. JES down
		if(Jets->at(j).ptJESDown>30. and MEt->ptShiftJetEnDown>200 and outputValueAK4>0.996) nTagJets_0p996_JESDown++;
		//4. JER up
		if(Jets->at(j).ptJERUp>30. and MEt->ptShiftJetResUp>200 and outputValueAK4>0.996) nTagJets_0p996_JERUp++;
		//5. JER down
		if(Jets->at(j).ptJERDown>30. and MEt->ptShiftJetResDown>200 and outputValueAK4>0.996) nTagJets_0p996_JERDown++;
		//6. unclustered up
		if(Jets->at(j).pt>30. and MEt->ptShiftUnclusteredEnUp>200 and outputValueAK4>0.996) nTagJets_0p996_UnclusteredEnUp++;
		//6. unclustered down
		if(Jets->at(j).pt>30. and MEt->ptShiftUnclusteredEnDown>200 and outputValueAK4>0.996) nTagJets_0p996_UnclusteredEnDown++;

		//store jets passing acceptance and with inference
		skimmedJets.push_back(Jets->at(j));
		validJetIndex.push_back(j);

	      }//acceptance

	  }//jet loop

	if(isVerbose) std::cout << "n. tagged jets " << nTagJets_0p996_JJ << std::endl;
        if(isVerbose) std::cout << "======================================== " << std::endl;


	//Selection on MET
        //if(doSR and MEt->pt<200) continue;

	//Fill lepton vector
	for ( auto &tmp : LeptonsStruct )
	  {
	    LepPdgId.push_back(tmp.pdgId);
	    LepCharge.push_back(tmp.charge);
	    LepPt.push_back(tmp.vec.Pt());
	    LepEta.push_back(tmp.vec.Eta());
	    LepPhi.push_back(tmp.vec.Phi());
	    LepMass.push_back(tmp.vec.M());
	  }

	//Prepare boolean flags
	//At this point, doSR and doZtoMM should be all fulfilled, cross check
	if(doSR) isSR = true;

	//Observed worse agreement, skip this --> redo
	n_pass->Fill(0.);
	if(EventNumber % 2 == 0) n_even->Fill(0.);
	if(EventNumber % 2 != 0) n_odd->Fill(0.);
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


    outputFile->Write();
    outputFile->Close();
    mcPUFile->Close();
    mcTriggerFile->Close();
    inputFile->Close();

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}
