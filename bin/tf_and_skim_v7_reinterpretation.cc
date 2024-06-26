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

//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6_smear.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/Objects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/CaloObjects_v6.h"
#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
//#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/MyAnalysis.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

using namespace ROOT::Math;
using namespace std;
using namespace Eigen;

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

float avg ( std::vector<float> & v )
{
  float return_value = 0.0;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i);
    }
  return ( return_value / n);
}

float weighted_avg ( std::vector<float> & v, std::vector<float> & w )
{
  float return_value = 0.0;
  float w_sum = 0.;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += v.at(i)*w.at(i);
      w_sum += w.at(i);
    }
  return ( return_value / w_sum);
}

float stdev ( std::vector<float> & v )
{
  float return_value = 0.0;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += pow(v.at(i) - avg(v),2);
    }
  return sqrt( return_value / n);
}

float biased_weighted_stdev ( std::vector<float> & v , std::vector<float> & w )
//https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
{
  float return_value = 0.0;
  float w_sum = 0.;
  int n = v.size();
  for ( int i=0; i < n; i++)
    {
      return_value += w.at(i)*pow(v.at(i) - weighted_avg(v,w),2);
      w_sum += w.at(i);
    }
  return sqrt( return_value / w_sum);
}


//DBSCAN
#define MINIMUM_POINTS 3     // minimum number of cluster
#define EPSILON (1.4*1.4)//  // distance for clustering, metre^2

//#define MINIMUM_POINTS 4     // minimum number of cluster
//#define EPSILON (1.*1.)//(0.75*0.75)  // distance for clustering, metre^2

void readBenchmarkData(vector<Point>& points)
{
  // load point cloud
  FILE *stream;
  stream = fopen ("/afs/desy.de/user/l/lbenato/LLP_inference/CMSSW_11_1_3/src/NNInferenceCMSSW/LLP_NN_Inference/dbscan_example/benchmark_hepta.dat","ra");

  unsigned int minpts, num_points, cluster, i = 0;
  double epsilon;
  fscanf(stream, "%u\n", &num_points);

  Point *p = (Point *)calloc(num_points, sizeof(Point));

  while (i < num_points)
    {
      fscanf(stream, "%f,%f,%f,%d\n", &(p[i].x), &(p[i].y), &(p[i].z), &cluster);
      p[i].clusterID = UNCLASSIFIED;
      points.push_back(p[i]);
      ++i;
    }

  free(p);
  fclose(stream);
}

void printResults(vector<Point>& points, int num_points)
{
  int i = 0;
  /*
  printf("Number of points: %u\n"
        " x     y     z     cluster_id\n"
        "-----------------------------\n"
  	 , num_points);
  while (i < num_points)
    {
      printf("%5.2lf %5.2lf %5.2lf: %d\n",
  	     points[i].x,
  	     points[i].y, points[i].z,
  	     points[i].clusterID);
      ++i;
    }
  */
  
  printf("Number of points: %u\n"
        " x     y     z     time     eta  phi  station  nRecHits  cluster_id\n"
        "------------------------------------------------------------------------------\n"
	 , num_points);
  while (i < num_points)
    {
      printf("%5.2lf %5.2lf %5.2lf  %5.2lf   %5.2lf \t%5.2lf \t%d \t%d \t%d\n",
	     points[i].x,
	     points[i].y, points[i].z,
	     points[i].time, 
	     points[i].eta, points[i].phi, points[i].station, points[i].nRecHits, 
	     points[i].clusterID);
      ++i;
    }
  
}


//3D line
float get_coord_line(float z, VectorXf Sol) {
  float coord(-999999.);
  if(Sol.size()==2 and Sol[0]!=0)
    {
      coord = (z - Sol[1])/Sol[0];
    }
  return coord;
}


//Assigns x, y, z based on t and p (size 4)
void line(float t, float &x, float &y, float &z, VectorXf SolXZ, VectorXf SolYZ) {
  // a parametric line is define from 6 parameters but 4 are independent
  // x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
  // can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
  x = get_coord_line(t,SolXZ);
  y = get_coord_line(t,SolYZ);
  z = t;
}

//calculate distance between a point and a parametric line
//it looks at two points with coordinates z=0 and z=1
float distance2(float x,float y,float z, VectorXf SolXZ, VectorXf SolYZ) {
    // distance line point is D= | (xp-x0) cross  ux |
    // where ux is direction of line and x0 is a point in the line (like t = 0)
    XYZVector p(x,y,z);
    float x0, y0, z0 = -9999.; 
    float x1, y1, z1 = -9999.;
    line(-1.,x0,y0,z0,SolXZ,SolYZ);
    line(1.,x1,y1,z1,SolXZ,SolYZ);
    //std::cout<< "x0, y0, z0 " << x0 << " " << y0 << " " << z0 << endl;
    //std::cout<< "x1, y1, z1 " << x1 << " " << y1 << " " << z1 << endl;
    XYZVector p0(x0,y0,z0);
    XYZVector p1(x1,y1,z1);
    XYZVector u = (p1-p0).Unit();
    double d2 = ((p-p0).Cross(u)).Mag2();
    return d2;
  }


////////

int main(int argc, char **argv) {

    const float ELE_MASS = 0.000511;
    const float MU_MASS  = 0.105658;
    const float TAU_MASS  = 1.77686;
    const float Z_MASS   = 91.2;

    if(argc<11)
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
    bool doGen(false);
    if(strcmp(argv[9], "doGen")==0) doGen=true;
    bool doSR(false);
    if(strcmp(argv[9], "doSR")==0) doSR=true;

    bool isVerbose(false);
    bool printFit(false);

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Skip even EventNumber: " << skipTrain << std::endl;
    std::cout << "isSignal: " << isSignal << std::endl;
    std::cout << "isData: " << isData << std::endl;
    std::cout << "MC PU file: " << argv[6] << std::endl;
    std::cout << "MC trigger file: " << argv[7] << std::endl;
    std::cout << "MC trigger string: " << argv[8] << std::endl;
    //std::cout << "Data PU file: " << argv[5] << std::endl;
    //std::cout << "Data PU up file: " << argv[6] << std::endl;
    //std::cout << "Data PU down file: " << argv[7] << std::endl;
    if(doGen) std::cout << "Gen studies, no selections" << std::endl;
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

    float minphi(-9999.);
    float maxphi( 9999.);
    std::string tag_2017 = "trigger_efficiency_Fall17";
    std::string tag_2018 = "trigger_efficiency_Fall18";
    if(mcTriggerString.find(tag_2017) != std::string::npos)
      {
	minphi = 3.5;
	maxphi = 2.7;
      }
    if(mcTriggerString.find(tag_2018) != std::string::npos)
      {
	minphi = 0.9;
	maxphi = 0.4;
      }

    std::string timeCBFilename = argv[10];

    //This is not really needed. It changes the event yield but not the acceptance. Do it later.
    //std::string phoSFFilename = argv[13];
    //std::string eleSFFilename = argv[14];

    //std::string dataFilename = argv[5];
    //std::string dataFilenameUp = argv[6];
    //std::string dataFilenameDown = argv[7];

    //std::string inputTreeName = "skim";
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

    TFile *timeCBFile = TFile::Open(timeCBFilename.data(),"READ"); if (!timeCBFile) return 0;
    TF1  *dataCB = (TF1*)timeCBFile->Get("data_CB");
    TF1  *mcCB = (TF1*)timeCBFile->Get("back_CB");

    TF1 *smearCB = (TF1*)dataCB->Clone("smear_cb");
    smearCB->SetParameter(0,dataCB->GetParameter(0));
    smearCB->SetParameter(1,dataCB->GetParameter(1) - mcCB->GetParameter(1));
    smearCB->SetParameter(2, sqrt( abs( pow(dataCB->GetParameter(2),2) - pow(mcCB->GetParameter(2),2) )) );
    smearCB->SetParameter(3,dataCB->GetParameter(3));
    smearCB->SetParameter(4,dataCB->GetParameter(4));

    //TFile *phoSFFile = TFile::Open(phoSFFilename.data(),"READ"); if (!phoSFFile) return 0;
    //TH1F  *phoSF_1ns = (TH1F*)phoSFFile->Get("ratio_1ns");
    //TH1F  *phoSF_2ns = (TH1F*)phoSFFile->Get("ratio_2ns");
    //float sf_pho_1ns = phoSF_1ns->GetBinContent(1);
    //float sf_pho_2ns = phoSF_2ns->GetBinContent(1);
    //float sf_pho;
    //if(abs(1-sf_pho_1ns) > abs(1-sf_pho_2ns))
    //{
    //sf_pho = sf_pho_1ns;
    //}
    //else
    //{
    //sf_pho = sf_pho_2ns;
    //}
    
    //TFile *eleSFFile = TFile::Open(eleSFFilename.data(),"READ"); if (!eleSFFile) return 0;
    //TH1F  *eleSF_1ns = (TH1F*)eleSFFile->Get("ratio_1ns");
    //TH1F  *eleSF_2ns = (TH1F*)eleSFFile->Get("ratio_2ns");
    //float sf_ele_1ns = eleSF_1ns->GetBinContent(1);
    //float sf_ele_2ns = eleSF_2ns->GetBinContent(1);
    //float sf_ele;
    //if(abs(1-sf_ele_1ns) > abs(1-sf_ele_2ns))
    //{
    //sf_ele = sf_ele_1ns;
    //}
    //else
    //{
    //sf_ele = sf_ele_2ns;
    //}


    // Input variables
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
    //Long64_t nCHSJets;
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


    //Flags for SR/CR
    bool isPreselected(false);
    bool isSR(false);

    bool isCosmic(false);
    bool isDT_fit(false);
    bool isCosmicVetoWithTags(false);
    //TH1F *PUWeightHist = (TH1F*)pileup_mc->Clone("PUWeight");
    //DivideHist( PUWeightHist , pileup_data, pileup_mc);
    //PUWeightHist->GetYaxis()->SetTitle("PU data/PU mc");
    //TH1F *PUWeightHistUp = (TH1F*)pileup_mc->Clone("PUWeightUp");
    //DivideHist( PUWeightHistUp , pileup_data_up, pileup_mc);
    //PUWeightHistUp->GetYaxis()->SetTitle("PU data/PU mc");
    //TH1F *PUWeightHistDown = (TH1F*)pileup_mc->Clone("PUWeightDown");
    //DivideHist( PUWeightHistDown , pileup_data_down, pileup_mc);
    //PUWeightHistDown->GetYaxis()->SetTitle("PU data/PU mc");

    std::vector<TauType>    skimmedTaus;

    std::vector<JetType>    skimmedJetsFullEta;
    std::vector<JetType>    skimmedJets;
    std::vector<JetType>    skimmedJetsNegative;
    std::vector<JetType>    skimmedJetsMatch;
    std::vector<JetType>    skimmedJetsAcceptanceMatch;

    std::vector<JetType>    JetsNum;
    std::vector<JetType>    JetsDen;

    std::vector<ecalRecHitType> skimmedEcalRecHitsAK4;
    std::vector<ecalRecHitType> skimmedAcceptanceEcalRecHitsAK4;
    std::vector<ecalRecHitType> taggedEcalRecHitsAK4;
    std::vector<ecalRecHitType> taggedAcceptanceEcalRecHitsAK4;
    
    //DBSCAN
    std::vector<Point> points;
    std::vector<Point> points_valid_time;
    int n_clusters;
    int n_noise;
    int n_clusters_valid_time;
    int n_noise_valid_time;

    //Additional output variables
    //DT fit
    float dt_fit_chi2(9999.);
    float dt_fit_chi2_reduced(9999.);
    float dt_ecal_dist(9999.);
    float dt_ecal_no_tag_dist(9999.);
    float dt_ecal_acc_no_tag_dist(9999.);
    float dt_ecal_acc_dist(9999.);
    float m_xz(-9999.);
    float c_xz(-9999.);
    float m_yz(-9999.);
    float c_yz(-9999.);
    std::vector<float> DT_fit_xx;
    std::vector<float> DT_fit_yy;
    std::vector<float> DT_fit_zz;
    std::vector<float> DT_fit_res;
    //Beam Halo
    //float min_dPhi_jets(9999.);
    //float min_dPhi_jets_0p7(9999.);
    //float min_dPhi_jets_0p9(9999.);
    //float min_dPhi_jets_0p9_no_tags(9999.);
    float min_dPhi_jets_0p996(9999.);

    //float min_dPhi_jets_eta_1p0(9999.);
    //float min_dPhi_jets_eta_1p0_0p7(9999.);
    //float min_dPhi_jets_eta_1p0_0p9(9999.);
    //float min_dPhi_jets_eta_1p0_0p9_no_tags(9999.);
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
    outputTree->Branch("isSR",              &isSR,              "isSR/O");
    outputTree->Branch("MeanNumInteractions",             &MeanNumInteractions,             "MeanNumInteractions/I");

    //MET triggers
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");

    //MET filters
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

    outputTree->Branch("n_clusters", &n_clusters, "n_clusters/I");
    outputTree->Branch("n_noise", &n_noise, "n_noise/I");
    outputTree->Branch("n_clusters_valid_time", &n_clusters_valid_time, "n_clusters_valid_time/I");
    outputTree->Branch("n_noise_valid_time", &n_noise_valid_time, "n_noise_valid_time/I");
    outputTree->Branch("dt_fit_chi2", &dt_fit_chi2, "dt_fit_chi2/F");
    outputTree->Branch("dt_fit_chi2_reduced", &dt_fit_chi2_reduced, "dt_fit_chi2_reduced/F");
    outputTree->Branch("dt_ecal_no_tag_dist", &dt_ecal_no_tag_dist, "dt_ecal_no_tag_dist/F");
    outputTree->Branch("dt_ecal_acc_no_tag_dist", &dt_ecal_acc_no_tag_dist, "dt_ecal_acc_no_tag_dist/F");
    outputTree->Branch("dt_ecal_dist", &dt_ecal_dist, "dt_ecal_dist/F");
    outputTree->Branch("dt_ecal_acc_dist", &dt_ecal_acc_dist, "dt_ecal_acc_dist/F");
    outputTree->Branch("m_xz", &m_xz, "m_xz/F");
    outputTree->Branch("c_xz", &c_xz, "c_xz/F");
    outputTree->Branch("m_yz", &m_yz, "m_yz/F");
    outputTree->Branch("c_yz", &c_yz, "c_yz/F");
    //outputTree->Branch("min_dPhi_jets", &min_dPhi_jets, "min_dPhi_jets/F");
    //outputTree->Branch("min_dPhi_jets_0p7", &min_dPhi_jets_0p7, "min_dPhi_jets_0p7/F");
    //outputTree->Branch("min_dPhi_jets_0p9", &min_dPhi_jets_0p9, "min_dPhi_jets_0p9/F");
    //outputTree->Branch("min_dPhi_jets_0p9_no_tags", &min_dPhi_jets_0p9_no_tags, "min_dPhi_jets_0p9_no_tags/F");
    outputTree->Branch("min_dPhi_jets_0p996", &min_dPhi_jets_0p996, "min_dPhi_jets_0p996/F");

    //outputTree->Branch("min_dPhi_jets_eta_1p0", &min_dPhi_jets_eta_1p0, "min_dPhi_jets_eta_1p0/F");
    //outputTree->Branch("min_dPhi_jets_eta_1p0_0p7", &min_dPhi_jets_eta_1p0_0p7, "min_dPhi_jets_eta_1p0_0p7/F");
    //outputTree->Branch("min_dPhi_jets_eta_1p0_0p9", &min_dPhi_jets_eta_1p0_0p9, "min_dPhi_jets_eta_1p0_0p9/F");
    //outputTree->Branch("min_dPhi_jets_eta_1p0_0p9_no_tags", &min_dPhi_jets_eta_1p0_0p9_no_tags, "min_dPhi_jets_eta_1p0_0p9_no_tags/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p996", &min_dPhi_jets_eta_1p0_0p996, "min_dPhi_jets_eta_1p0_0p996/F");

    outputTree->Branch("nPFCandidates",     &nPFCandidates,     "nPFCandidates/I");
    outputTree->Branch("nPFCandidatesTrack", &nPFCandidatesTrack, "nPFCandidatesTrack/I");
    outputTree->Branch("nLLPInCalo", &nLLPInCalo, "nLLPInCalo/I");

    outputTree->Branch("m_chi", &m_chi, "m_chi/I");
    outputTree->Branch("ctau", &ctau, "ctau/I");
    outputTree->Branch("is_central", &is_central, "is_central/O");
    outputTree->Branch("Muons", &Muons);
    outputTree->Branch("Electrons", &Electrons);
    outputTree->Branch("Photons", &Photons);
    outputTree->Branch("Taus", &skimmedTaus);
    outputTree->Branch("JetsFullEta", &skimmedJetsFullEta);
    outputTree->Branch("Jets", &skimmedJets);
    outputTree->Branch("JetsNegative", &skimmedJetsNegative);
    outputTree->Branch("JetsMatch", &skimmedJetsMatch);
    outputTree->Branch("JetsAcceptanceMatch", &skimmedJetsAcceptanceMatch);
    outputTree->Branch("JetsNum", &JetsNum);
    outputTree->Branch("JetsDen", &JetsDen);
    outputTree->Branch("EcalRecHitsAK4", &EcalRecHitsAK4);
    outputTree->Branch("skimmedEcalRecHitsAK4", &skimmedEcalRecHitsAK4);
    outputTree->Branch("skimmedAcceptanceEcalRecHitsAK4", &skimmedAcceptanceEcalRecHitsAK4);
    outputTree->Branch("taggedEcalRecHitsAK4", &taggedEcalRecHitsAK4);
    outputTree->Branch("taggedAcceptanceEcalRecHitsAK4", &taggedAcceptanceEcalRecHitsAK4);
    outputTree->Branch("DT_fit_xx", &DT_fit_xx);
    outputTree->Branch("DT_fit_yy", &DT_fit_yy);
    outputTree->Branch("DT_fit_zz", &DT_fit_zz);
    outputTree->Branch("DT_fit_res", &DT_fit_res);

    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);
    outputTree->Branch("GenGravitinos", &GenGravitinos);
    outputTree->Branch("DTSegments", &DTSegments);
    outputTree->Branch("CSCSegments", &CSCSegments);

    outputTree->Branch("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");
    //outputTree->Branch("nTagJets_0p996_unsmeared",     &nTagJets_0p996_unsmeared,      "nTagJets_0p996_unsmeared/I");

    //TBD
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

    //Unsmeared
    //tensorflow::GraphDef* graphDefUnsmearedAK4 = tensorflow::loadGraphDef(graphPathAK4);
    //tensorflow::Session* sessionUnsmearedAK4 = tensorflow::createSession(graphDefUnsmearedAK4, nThreads);
    //tensorflow::Tensor inputTensorUnsmearedAK4(tensorflow::DT_FLOAT, {1, int(featuresAK4.size()) });
    //float outputValueUnsmearedAK4;



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
 

	n_clusters = -1;
	n_noise = -1;
	n_clusters_valid_time = -1;
	n_noise_valid_time = -1;

	dt_fit_chi2 = 9999.;
	dt_fit_chi2_reduced = 9999.;
	dt_ecal_no_tag_dist = 9999.;
	dt_ecal_acc_no_tag_dist = 9999.;
	dt_ecal_dist = 9999.;
	dt_ecal_acc_dist = 9999.;
	isCosmic = false;
	isCosmicVetoWithTags = false;
	isDT_fit = false;

	m_xz = -9999.;
	c_xz = -9999.;
	m_yz = -9999.;
	c_yz = -9999.;

	//min_dPhi_jets = 9999.;
	//min_dPhi_jets_0p7 = 9999.;
	//min_dPhi_jets_0p9 = 9999.;
	//min_dPhi_jets_0p9_no_tags = 9999.;
	min_dPhi_jets_0p996 = 9999.;

	//min_dPhi_jets_eta_1p0 = 9999.;
	//min_dPhi_jets_eta_1p0_0p7 = 9999.;
	//min_dPhi_jets_eta_1p0_0p9 = 9999.;
	//min_dPhi_jets_eta_1p0_0p9_no_tags = 9999.;
	min_dPhi_jets_eta_1p0_0p996 = 9999.;


	nTagJets_0p996 = 0;
	//nTagJets_0p996_unsmeared = 0;

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
        JetsNum.clear();
        JetsDen.clear();
	skimmedEcalRecHitsAK4.clear();
	skimmedAcceptanceEcalRecHitsAK4.clear();
	taggedEcalRecHitsAK4.clear();
	taggedAcceptanceEcalRecHitsAK4.clear();
	points.clear();
	points_valid_time.clear();
	DT_fit_xx.clear();
	DT_fit_yy.clear();
	DT_fit_zz.clear();
	DT_fit_res.clear();


        //if (i % 1000 == 0) {
        //    std::cout << "evaluating entry " << i << std::endl;
        //}
        inputTree->GetEntry(i);

	Long64_t TagNumber;
	if(isMC) TagNumber=EventNumber;
	else TagNumber=RunNumber;


        //std::cout << "======== " << std::endl;
        //std::cout << "Event " << i << std::endl;
	//std::cout << "======== " << std::endl;

	//Consider PU weight

	//PUReWeight = PUWeightHist->GetBinContent(PUWeightHist->GetXaxis()->FindBin(MeanNumInteractions));
	//PUReWeightUp = PUWeightHistUp->GetBinContent(PUWeightHistUp->GetXaxis()->FindBin(MeanNumInteractions));
	//PUReWeightDown = PUWeightHistDown->GetBinContent(PUWeightHistDown->GetXaxis()->FindBin(MeanNumInteractions));

	if(isMC)
	  {
	    if(doSR or doGen) TriggerWeight = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->pt));//only for SR MC!!
	    PUReWeight = pu->GetBinContent(pu->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightUp = pu_up->GetBinContent(pu_up->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightDown = pu_down->GetBinContent(pu_down->GetXaxis()->FindBin(MeanNumInteractions));
	  }

	if(isMC and doGen)
	  {
            dR_LLPs = reco::deltaR(GenLLPs->at(0).eta,GenLLPs->at(0).phi,GenLLPs->at(1).eta,GenLLPs->at(1).phi);
            dR_Higgs = GenHiggs->size()==2 ? reco::deltaR(GenHiggs->at(0).eta,GenHiggs->at(0).phi,GenHiggs->at(1).eta,GenHiggs->at(1).phi) : -9.;
            dR_Gravitinos = reco::deltaR(GenGravitinos->at(0).eta,GenGravitinos->at(0).phi,GenGravitinos->at(1).eta,GenGravitinos->at(1).phi);


            dPhi_LLPs = reco::deltaPhi(GenLLPs->at(0).phi,GenLLPs->at(1).phi);
            dPhi_Higgs = GenHiggs->size()==2 ? reco::deltaPhi(GenHiggs->at(0).phi,GenHiggs->at(1).phi) : -9.;
            dPhi_Gravitinos = reco::deltaPhi(GenGravitinos->at(0).phi,GenGravitinos->at(1).phi);

            dR_Gravitino_Higgs_0 = (GenGravitinos->at(0).travelRadiusLLP == GenHiggs->at(0).travelRadiusLLP) ? reco::deltaR(GenGravitinos->at(0).eta,GenGravitinos->at(0).phi,GenHiggs->at(0).eta,GenHiggs->at(0).phi) : -9.;
            dR_Gravitino_Higgs_1 = GenHiggs->size()==2 ? ((GenGravitinos->at(1).travelRadiusLLP == GenHiggs->at(1).travelRadiusLLP) ? reco::deltaR(GenGravitinos->at(1).eta,GenGravitinos->at(1).phi,GenHiggs->at(1).eta,GenHiggs->at(1).phi) : -9.) : -9.;
            dR_Gravitino_GenMet_0 = reco::deltaR(GenGravitinos->at(0).eta,GenGravitinos->at(0).phi,MEt->etaGen,MEt->phiGen);
            dR_Gravitino_GenMet_1 = reco::deltaR(GenGravitinos->at(1).eta,GenGravitinos->at(1).phi,MEt->etaGen,MEt->phiGen);
            dPhi_Gravitino_Met_0 = reco::deltaPhi(GenGravitinos->at(0).phi,MEt->phi);
            dPhi_Gravitino_Met_1 = reco::deltaPhi(GenGravitinos->at(1).phi,MEt->phi);
            dPhi_Gravitino_GenMet_0 = reco::deltaPhi(GenGravitinos->at(0).phi,MEt->phiGen);
            dPhi_Gravitino_GenMet_1 = reco::deltaPhi(GenGravitinos->at(1).phi,MEt->phiGen);
            dR_LLP_GenMet_0 = reco::deltaR(GenLLPs->at(0).eta,GenLLPs->at(0).phi,MEt->etaGen,MEt->phiGen);
            dR_LLP_GenMet_1 = reco::deltaR(GenLLPs->at(1).eta,GenLLPs->at(1).phi,MEt->etaGen,MEt->phiGen);
            dPhi_LLP_Met_0 = reco::deltaPhi(GenLLPs->at(0).phi,MEt->phi);
            dPhi_LLP_Met_1 = reco::deltaPhi(GenLLPs->at(1).phi,MEt->phi);
            dPhi_LLP_GenMet_0 = reco::deltaPhi(GenLLPs->at(0).phi,MEt->phiGen);
            dPhi_LLP_GenMet_1 = reco::deltaPhi(GenLLPs->at(1).phi,MEt->phiGen);
            dR_Higgs_GenMet_0 = reco::deltaR(GenHiggs->at(0).eta,GenHiggs->at(0).phi,MEt->etaGen,MEt->phiGen);
            dR_Higgs_GenMet_1 = GenHiggs->size()==2 ? reco::deltaR(GenHiggs->at(1).eta,GenHiggs->at(1).phi,MEt->etaGen,MEt->phiGen) : -9.;
            dPhi_Higgs_Met_0 = reco::deltaPhi(GenHiggs->at(0).phi,MEt->phi);
            dPhi_Higgs_Met_1 = GenHiggs->size()==2 ? reco::deltaPhi(GenHiggs->at(1).phi,MEt->phi) : -9.;
            dPhi_Higgs_GenMet_0 = reco::deltaPhi(GenHiggs->at(0).phi,MEt->phiGen);
            dPhi_Higgs_GenMet_1 = GenHiggs->size()==2 ? reco::deltaPhi(GenHiggs->at(1).phi,MEt->phiGen) : -9.;
            TLorentzVector DiGravitino;
            TLorentzVector Grav0;
            TLorentzVector Grav1;
            Grav0.SetPtEtaPhiM(GenGravitinos->at(0).pt,GenGravitinos->at(0).eta,GenGravitinos->at(0).phi,GenGravitinos->at(0).mass);
            Grav0.SetPtEtaPhiM(GenGravitinos->at(1).pt,GenGravitinos->at(1).eta,GenGravitinos->at(1).phi,GenGravitinos->at(1).mass);
            DiGravitino = Grav0 + Grav1;
            DiGravitino_pt = DiGravitino.Pt();
            DiGravitino_mass = DiGravitino.M();
            DiGravitino_eta = DiGravitino.Eta();
            DiGravitino_phi = DiGravitino.Phi();
            dR_DiGravitino_GenMet = reco::deltaR(DiGravitino.Eta(),DiGravitino.Phi(),MEt->etaGen,MEt->phiGen);
            dPhi_DiGravitino_Met = reco::deltaPhi(DiGravitino.Phi(),MEt->phi);
            dPhi_DiGravitino_GenMet = reco::deltaPhi(DiGravitino.Phi(),MEt->phiGen);
	    dPhi_DiGravitino_Higgs_0 = reco::deltaPhi(DiGravitino.Phi(),GenHiggs->at(0).phi);
	    dPhi_DiGravitino_Higgs_1 = reco::deltaPhi(DiGravitino.Phi(),GenHiggs->at(1).phi);
	    dPhi_Gravitino_0_Higgs_0 = reco::deltaPhi(GenHiggs->at(0).phi,GenGravitinos->at(0).phi);
	    dPhi_Gravitino_1_Higgs_1 = reco::deltaPhi(GenHiggs->at(1).phi,GenGravitinos->at(1).phi);
            perc_met_held_by_gravitinos = MEt->pt>0 ? DiGravitino_pt/MEt->pt : -1.;
          }

	//Trigger selections

	//MET filters always fulfilled
	//Invert Beam Halo
        //if(Flag2_globalSuperTightHalo2016Filter) continue;
	//InvertHBHE
	//if(Flag2_HBHENoiseFilter and Flag2_HBHEIsoNoiseFilter) continue;
	if(not doGen)
	  {
	    if(!Flag2_globalSuperTightHalo2016Filter) continue;
	    if(!Flag2_EcalDeadCellTriggerPrimitiveFilter) continue;
	    if(!Flag2_HBHENoiseFilter) continue;
	    if(!Flag2_HBHEIsoNoiseFilter) continue;
	    if(!Flag2_ecalBadCalibFilter) continue;
	    if(!Flag2_eeBadScFilter) continue;
	    if(!Flag2_BadPFMuonFilter) continue;
	  }

	if(doSR and not(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) ) continue;

	//Selection on MET
        if(doSR and MEt->pt<200) continue;

	//Loop on veto objects
	//JJ
	std::vector<Particle> LeptonsStruct;
	std::vector<Particle> MuonsStruct;
	std::vector<Particle> ElectronsStruct;
	std::vector<Particle> PhotonsStruct;
	std::vector<Particle> TausStruct;
	//Muons
	float mu_iso = 0.4;
	for (unsigned int m=0; m<Muons->size(); m++)
	  {

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


	//Transverse mass met + Leptons (e and mu)
	TLorentzVector lepp4;
	for ( auto &tmp : LeptonsStruct )
	  {
	    lepp4 += tmp.vec;
	  }
	dPhi = reco::deltaPhi(MEt->phi, lepp4.Phi());
	MT = sqrt(2*(MEt->pt)*lepp4.Pt()*(1-cos(dPhi)));
	nLeptons = LeptonsStruct.size();

        //if(nTaus>0) continue;
        //if(nPhotons>0) continue;
        //if(nMuons>0) continue;
        //if(nElectrons>0) continue;

        if(isVerbose) std::cout << "======================================== " << std::endl;
        if(isVerbose) std::cout << "EventNumber " << EventNumber << "\tLumiNumber " << LumiNumber << std::endl;


	//Apply acceptance cuts to jets and fat jets 
	std::vector<int> validJetIndex;

	for (unsigned int j=0; j<Jets->size(); j++)
	  {

	    //Correct acceptance for MinJetMetDPhi:
	    //Jet pt>30, Jet eta<2.4
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi) MinJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    

	    //Time smearing!
	    //We need:
	    //1. data file --> can pass from python
	    //2. signal file fit --> can pass from python
	    //3. name of the CB --> can pass from python

	    //Don't remove jets with eta>1 now, it messes up with the cosmic veto!
	    if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8)//cleaned jets!

	      {

		//Ignore jets overlapped to leptons, photons and taus
		float jet_iso = 0.4;
		//Leptons
		float dR_lep = -1;
		for(auto& lep : LeptonsStruct)
		  {
		    float thisDR = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,lep.vec.Eta(),lep.vec.Phi());
		    if(dR_lep < 0 || thisDR < dR_lep) dR_lep = thisDR;
		  }
		if(dR_lep > 0 && dR_lep < jet_iso) continue;

		//Taus
		float dR_tau = -1;
		for(auto& tau : TausStruct)
		  {
		    float thisDR_tau = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,tau.vec.Eta(),tau.vec.Phi());
		    if(dR_tau < 0 || thisDR_tau < dR_tau) dR_tau = thisDR_tau;
		  }
		if(dR_tau > 0 && dR_tau < jet_iso) continue;

		//Photons
		float dR_pho = -1;
		for(auto& pho : PhotonsStruct)
		  {
		    float thisDR_pho = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,pho.vec.Eta(),pho.vec.Phi());
		    if(dR_pho < 0 || thisDR_pho < dR_pho) dR_pho = thisDR_pho;
		  }
		if(dR_pho > 0 && dR_pho < jet_iso) continue;
		
		//Here: passed acceptance
		//Redone at the end!!!
		//if(Jets->at(j).timeRecHitsEB>-1) nCHSJetsAcceptanceCalo++;


		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		//std::cout<< "Jet n. " << j << " eFracRecHitsEB: " << Jets->at(j).eFracRecHitsEB  << std::endl;
		Jets->at(j).eFracRecHitsEB = (Jets->at(j).energy>0 and Jets->at(j).energyRecHitsEB>0) ? Jets->at(j).energyRecHitsEB/Jets->at(j).energy : -1.;
		
		//Fix also timeRMS dividing by sqrt nRecHitsEB
		Jets->at(j).timeRMSRecHitsEB = (Jets->at(j).nRecHitsEB>0) ? Jets->at(j).timeRMSRecHitsEB/sqrt(Jets->at(j).nRecHitsEB) : -1.;

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

		/*
		inputValuesUnsmeared.at(0) = Jets->at(j).nTrackConstituents;
		inputValuesUnsmeared.at(1) = Jets->at(j).nSelectedTracks;
		inputValuesUnsmeared.at(2) = Jets->at(j).timeRecHitsHB;
		inputValuesUnsmeared.at(3) = Jets->at(j).eFracRecHitsEB;
		inputValuesUnsmeared.at(4) = Jets->at(j).nRecHitsEB;
		inputValuesUnsmeared.at(5) = Jets->at(j).sig1EB;
		inputValuesUnsmeared.at(6) = Jets->at(j).sig2EB;
		inputValuesUnsmeared.at(7) = Jets->at(j).ptDEB;
		inputValuesUnsmeared.at(8) = Jets->at(j).cHadEFrac;
		inputValuesUnsmeared.at(9) = Jets->at(j).nHadEFrac;
		inputValuesUnsmeared.at(10) = Jets->at(j).eleEFrac;
		inputValuesUnsmeared.at(11) = Jets->at(j).photonEFrac;
		inputValuesUnsmeared.at(12) = Jets->at(j).ptAllTracks;
		inputValuesUnsmeared.at(13) = Jets->at(j).ptAllPVTracks;
		inputValuesUnsmeared.at(14) = Jets->at(j).alphaMax;
	        inputValuesUnsmeared.at(15) = Jets->at(j).betaMax;
		inputValuesUnsmeared.at(16) = Jets->at(j).gammaMax;
		inputValuesUnsmeared.at(17) = Jets->at(j).gammaMaxEM;
		inputValuesUnsmeared.at(18) = Jets->at(j).gammaMaxHadronic;
		inputValuesUnsmeared.at(19) = Jets->at(j).gammaMaxET;
		inputValuesUnsmeared.at(20) = Jets->at(j).minDeltaRAllTracks;
		inputValuesUnsmeared.at(21) = Jets->at(j).minDeltaRPVTracks;
		*/

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


		//Unsmeared
		/*
		float* u = inputTensorUnsmearedAK4.flat<float>().data();
		for (float s : inputValuesUnsmeared) {
		  *u = s;
		  u++;
		}

		// run the inference
		std::vector<tensorflow::Tensor> outputsUnsmearedAK4;
		tensorflow::run(sessionUnsmearedAK4, {{inputTensorNameAK4, inputTensorUnsmearedAK4}}, {outputTensorNameAK4}, &outputsUnsmearedAK4, threadPool);
		// store the result
		outputValueUnsmearedAK4 = outputsUnsmearedAK4[0].matrix<float>()(0, 1);
		// keras cannot predict the output for invalid jets
		// fix it manually
		if(Jets->at(j).pt<0) outputValueUnsmearedAK4 = -1;
		Jets->at(j).pfXWP1000 = outputValueUnsmearedAK4;
		*/


		//28.11.22: unsmeared time
		if(outputValueAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1 and abs(Jets->at(j).eta)<1.) nTagJets_0p996++;

		//Unsmeared: time is timeRecHitsHB
		//if(outputValueUnsmearedAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsHB>-1 and abs(Jets->at(j).eta)<1.) nTagJets_0p996_unsmeared++;


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
		    //store jets passing acceptance and with inference
		    //check if we use only >-1 jets for cosmic and beam halo veto! --> yes!
		    //validJetIndex: no angular cuts, >-1 ns cut, same size as skimmedJetsFullEta
		    skimmedJetsFullEta.push_back(Jets->at(j));
		    validJetIndex.push_back(j);
		    
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

	if(isVerbose) std::cout << "n. tagged jets " << nTagJets_0p996 << std::endl;
        if(isVerbose) std::cout << "======================================== " << std::endl;


	//if(doGen and nLLPInCalo<1) continue;

	//Here
	//must loop over LLPs and check if it made a jet
	//this will fill jet num and jet den
	//if no jet was made, put an empty jet den object and nothing at num

	//TBD

	//Acceptance: defined as at least one b quark from a GenLLPs in eta and pT
	if( (abs(GenBquarks->at(0).eta)<1. and GenBquarks->at(0).pt>30) or (abs(GenBquarks->at(1).eta)<1. and GenBquarks->at(1).pt>30) ) isLLP0_b_acc=true;
	if( (abs(GenBquarks->at(2).eta)<1. and GenBquarks->at(2).pt>30) or (abs(GenBquarks->at(3).eta)<1. and GenBquarks->at(3).pt>30) ) isLLP1_b_acc=true;

	if( GenLLPs->at(0).isLLPInCaloAcceptance and (abs(GenBquarks->at(0).eta)<1. and GenBquarks->at(0).pt>30) or (abs(GenBquarks->at(1).eta)<1. and GenBquarks->at(1).pt>30) ) isLLP0_b_acc=true;
	if( GenLLPs->at(1).isLLPInCaloAcceptance and (abs(GenBquarks->at(2).eta)<1. and GenBquarks->at(2).pt>30) or (abs(GenBquarks->at(3).eta)<1. and GenBquarks->at(3).pt>30) ) isLLP1_b_acc=true;

	if(GenLLPs->at(0).isLLPInCaloAcceptance and isLLP0_b_acc ) nLLPInCalo_b_acc++;
	if(GenLLPs->at(1).isLLPInCaloAcceptance and isLLP1_b_acc ) nLLPInCalo_b_acc++;

	if(isLLP0_b_acc) nLLP_b_acc++;
	if(isLLP1_b_acc) nLLP_b_acc++;

	if(isVerbose) std::cout << "isLLP0_b_acc " << isLLP0_b_acc << std::endl;
	if(isVerbose) std::cout << "isLLP0 in calo " << GenLLPs->at(0).isLLPInCaloAcceptance << std::endl;
	if(isVerbose) std::cout << "isLLP1_b_acc " << isLLP1_b_acc << std::endl;
	if(isVerbose) std::cout << "isLLP1 in calo " << GenLLPs->at(1).isLLPInCaloAcceptance << std::endl;
	if(isVerbose) std::cout << "nTagJets_0p996 " << nTagJets_0p996 << std::endl;
	if(isVerbose) std::cout << "---" << std::endl;
	//std::cout << "nLLPInCalo " << nLLPInCalo << std::endl;
	//std::cout << "nLLP_b_acc " << nLLP_b_acc << std::endl;
	//std::cout << "nLLPInCalo_b_acc " << nLLPInCalo_b_acc << std::endl;
	

	for (unsigned int j=0; j<skimmedJets.size(); j++)
	  {
	    if(isVerbose) std::cout << "jet["<<j<<"] gen match " << skimmedJets.at(j).isGenMatchedCaloCorr << ", LLP radius " << skimmedJets.at(j).radiusLLP << ", DNN " << skimmedJets.at(j).sigprob << std::endl;

	    //Here: stuff matched but without asking LLP in calo acceptance
	    if(skimmedJets.at(j).isGenMatchedCaloCorr)
	      {
		skimmedJetsMatch.push_back(skimmedJets.at(j));
		//LLP0
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(0).travelRadius)
		  {
		    nJets_to_LLP0++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP0++;
		  }
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(1).travelRadius)
		  {
		    nJets_to_LLP1++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP1++;
		  }
	      }

	    //Here: stuff matched with b-quarks in acceptance
	    if(skimmedJets.at(j).isGenMatchedCaloCorr)
	      {
		skimmedJetsMatch.push_back(skimmedJets.at(j));
		//LLP0
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(0).travelRadius and isLLP0_b_acc)
		  {
		    nJets_to_LLP0_b_acc++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP0_b_acc++;
		  }
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(1).travelRadius and isLLP1_b_acc)
		  {
		    nJets_to_LLP1_b_acc++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP1_b_acc++;
		  }
	      }

	    if(skimmedJets.at(j).isGenMatchedCaloCorrLLPAccept)
	      {
		skimmedJetsAcceptanceMatch.push_back(skimmedJets.at(j));
		//LLP0
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(0).travelRadius)
		  {
		    nJets_to_LLP0InCalo++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP0InCalo++;
		  }
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(1).travelRadius)
		  {
		    nJets_to_LLP1InCalo++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP1InCalo++;
		  }
	      }


	    if(skimmedJets.at(j).isGenMatchedCaloCorrLLPAccept)
	      {
		skimmedJetsAcceptanceMatch.push_back(skimmedJets.at(j));
		//LLP0
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(0).travelRadius and isLLP0_b_acc)
		  {
		    nJets_to_LLP0InCalo_b_acc++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP0InCalo_b_acc++;
		  }
		if(skimmedJets.at(j).radiusLLP==GenLLPs->at(1).travelRadius and isLLP1_b_acc)
		  {
		    nJets_to_LLP1InCalo_b_acc++;
		    if(skimmedJets.at(j).sigprob>0.996) nTagJets_to_LLP1InCalo_b_acc++;
		  }
	      }

	    
	  }


	//determine boost-resolved at gen level
	if(isVerbose) std::cout << "gen higgs 0: dR dau " << GenHiggs->at(0).dRdaughters << std::endl;
	//std::cout << "nJets_to_LLP0 " << nJets_to_LLP0 << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP0 " << nTagJets_to_LLP0 << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP0_b_acc " << nTagJets_to_LLP0_b_acc << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP0InCalo_b_acc " << nTagJets_to_LLP0InCalo_b_acc << std::endl;
	//std::cout << "nJets_to_LLP0InCalo " << nJets_to_LLP0InCalo << std::endl;
	//std::cout << "nTagJets_to_LLP0InCalo " << nTagJets_to_LLP0InCalo << std::endl;

	if(isVerbose) std::cout<< "" << std::endl;

	if(isVerbose) std::cout << "gen higgs 1: dR dau " << GenHiggs->at(1).dRdaughters << std::endl;
	//std::cout << "nJets_to_LLP1 " << nJets_to_LLP1 << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP1 " << nTagJets_to_LLP1 << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP1_b_acc " << nTagJets_to_LLP1_b_acc << std::endl;
	if(isVerbose) std::cout << "nTagJets_to_LLP1InCalo_b_acc " << nTagJets_to_LLP1InCalo_b_acc << std::endl;
	//std::cout << "nJets_to_LLP1InCalo " << nJets_to_LLP1InCalo << std::endl;
	//std::cout << "nTagJets_to_LLP1InCalo " << nTagJets_to_LLP1InCalo << std::endl;


	//No jets in acceptance, go to next event
        //Sort PF candidates by their pt 

	//Sort EcalRecHitsAK4
	//Do we really need this? Probably not
	//std::sort(EcalRecHitsAK4->begin(), EcalRecHitsAK4->end(), energy_sorter);

	//Loop on EcalRecHitsAK4
	//Debug: look at ECAL rec hits that belong to jets in acceptance!
	//Remember: at ntuple level, EcalRecHits are stored in a cone 0.5, hence there are overlaps
	//Redo the matching
	for (unsigned int j=0; j<validJetIndex.size(); j++)
	  {
	    //Defined at each jet

	    for(unsigned int p=0; p<EcalRecHitsAK4->size(); p++)
	      {
		//j corresponds to the skimmed jet, validJetIndex.at(j) corresponds to the original jets
		//for each valid jet skimmedJetsFullEta[j] I want the Rec hits features
		//Beam Halo rejection variables
		//min_dR_jets
		//Calculate sparsity of associated ecal rec hits

		if(int(EcalRecHitsAK4->at(p).jetIndex) == int(validJetIndex.at(j)) )//only this is complaining...
		  {
		    //0.4 matching
		    if (reco::deltaR(Jets->at( int(validJetIndex.at(j)) ).eta, Jets->at( int(validJetIndex.at(j)) ).phi, EcalRecHitsAK4->at(p).eta, EcalRecHitsAK4->at(p).phi) < 0.4)
		      {

			skimmedEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));
			if(abs(Jets->at( int(validJetIndex.at(j)) ).eta)<1) skimmedAcceptanceEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));

			if(Jets->at(int(validJetIndex.at(j))).sigprob > 0.996)
			  {
			    taggedEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p)); 
			    if( abs(Jets->at(int(validJetIndex.at(j))).eta) < 1.)
			      {
				taggedAcceptanceEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p)); 
			      }
			  }//fill taggedEcalRecHitsAK4
		      }//fill skimmedEcalRecHitsAK4

		  }//check if considered EB associated to jet indices

	      }//loop on EcalRecHitsAK4



	  }//loop on jet indices



	///////
	for (unsigned int j=0; j<skimmedJetsFullEta.size(); j++)
	  {
	    //second loop on jets to calculate delta phi/R
	    for (unsigned int k=j+1; k<skimmedJetsFullEta.size() && k!=j; k++)
	      {
		//0p996
		if(skimmedJetsFullEta.at(j).sigprob>0.996 and skimmedJetsFullEta.at(k).sigprob>0.996)
		  {
		    min_dPhi_jets_0p996 = std::min(fabs(min_dPhi_jets_0p996),fabs(reco::deltaPhi(skimmedJetsFullEta.at(j).phi,skimmedJetsFullEta.at(k).phi)));

		    if( abs(skimmedJetsFullEta.at(j).eta)<1.0 and abs(skimmedJetsFullEta.at(k).eta)<1.0)
		      {
			min_dPhi_jets_eta_1p0_0p996 = std::min(fabs(min_dPhi_jets_eta_1p0_0p996),fabs(reco::deltaPhi(skimmedJetsFullEta.at(j).phi,skimmedJetsFullEta.at(k).phi)));

		      }
		  }

	      }

	  }
	//////


	//Calculate center of gravity ECAL rec hits of tagged jets;
	//Used for cosmic veto
	//Tagged jets |eta|<1.4
	std::vector<float> vec_ECAL_tag_x;
	std::vector<float> vec_ECAL_tag_y;
	std::vector<float> vec_ECAL_tag_z;
	float mean_ECAL_tag_x(-9999999.);
	float mean_ECAL_tag_y(-9999999.);
	float mean_ECAL_tag_z(-9999999.);
	std::transform(taggedEcalRecHitsAK4.begin(), taggedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_tag_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(taggedEcalRecHitsAK4.size()>0) mean_ECAL_tag_x=avg(vec_ECAL_tag_x);
	std::transform(taggedEcalRecHitsAK4.begin(), taggedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_tag_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(taggedEcalRecHitsAK4.size()>0) mean_ECAL_tag_y=avg(vec_ECAL_tag_y);
	std::transform(taggedEcalRecHitsAK4.begin(), taggedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_tag_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(taggedEcalRecHitsAK4.size()>0) mean_ECAL_tag_z=avg(vec_ECAL_tag_z);

	//Tagged jets |eta|<1.
	std::vector<float> vec_acc_ECAL_tag_x;
	std::vector<float> vec_acc_ECAL_tag_y;
	std::vector<float> vec_acc_ECAL_tag_z;
	float mean_acc_ECAL_tag_x(-9999999.);
	float mean_acc_ECAL_tag_y(-9999999.);
	float mean_acc_ECAL_tag_z(-9999999.);
	std::transform(taggedAcceptanceEcalRecHitsAK4.begin(), taggedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_tag_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(taggedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_tag_x=avg(vec_acc_ECAL_tag_x);
	std::transform(taggedAcceptanceEcalRecHitsAK4.begin(), taggedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_tag_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(taggedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_tag_y=avg(vec_acc_ECAL_tag_y);
	std::transform(taggedAcceptanceEcalRecHitsAK4.begin(), taggedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_tag_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(taggedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_tag_z=avg(vec_acc_ECAL_tag_z);

	//All jets |eta|<1.4
	std::vector<float> vec_ECAL_x;
	std::vector<float> vec_ECAL_y;
	std::vector<float> vec_ECAL_z;
	float mean_ECAL_x(-9999999.);
	float mean_ECAL_y(-9999999.);
	float mean_ECAL_z(-9999999.);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_x=avg(vec_ECAL_x);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_y=avg(vec_ECAL_y);
	std::transform(skimmedEcalRecHitsAK4.begin(), skimmedEcalRecHitsAK4.end(), std::back_inserter(vec_ECAL_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(skimmedEcalRecHitsAK4.size()>0) mean_ECAL_z=avg(vec_ECAL_z);

	//All jets |eta|<1.
	std::vector<float> vec_acc_ECAL_x;
	std::vector<float> vec_acc_ECAL_y;
	std::vector<float> vec_acc_ECAL_z;
	float mean_acc_ECAL_x(-9999999.);
	float mean_acc_ECAL_y(-9999999.);
	float mean_acc_ECAL_z(-9999999.);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_x),[](ecalRecHitType const& er) { return er.x/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_x=avg(vec_acc_ECAL_x);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_y),[](ecalRecHitType const& er) { return er.y/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_y=avg(vec_acc_ECAL_y);
	std::transform(skimmedAcceptanceEcalRecHitsAK4.begin(), skimmedAcceptanceEcalRecHitsAK4.end(), std::back_inserter(vec_acc_ECAL_z),[](ecalRecHitType const& er) { return er.z/100.; });
	if(skimmedAcceptanceEcalRecHitsAK4.size()>0) mean_acc_ECAL_z=avg(vec_acc_ECAL_z);



	//Cosmic veto:
	//DBSCAN on DTSegments
	for(unsigned int d=0; d<DTSegments->size(); d++)
	  {
	    Point p;
	    //Currently not removing points with invalid time;
	    //TODO: check if the result changes
	    p.x = DTSegments->at(d).x/100.;
	    p.y = DTSegments->at(d).y/100.;
	    p.z = DTSegments->at(d).z/100.;
	    p.eta = DTSegments->at(d).eta;
	    p.phi = DTSegments->at(d).phi;
	    p.time = DTSegments->at(d).time;
	    p.wheel = DTSegments->at(d).wheel;
	    p.sector = DTSegments->at(d).sector;
	    p.station = DTSegments->at(d).station;
	    p.nRecHits = DTSegments->at(d).nRecHits;
	    p.clusterID = UNCLASSIFIED;
	    points.push_back(p);

	  }

	DBSCAN ds(MINIMUM_POINTS, EPSILON, points);
	ds.run();

	std::vector<int> labels;
	std::vector<float> xx;
	std::vector<float> yy;
	std::vector<float> zz;
	std::vector<float> tt;
	std::vector<float> ss;
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(labels),[](Point const& p) { return p.clusterID; });
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(xx),[](Point const& p) { return p.x; });
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(yy),[](Point const& p) { return p.y; });
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(zz),[](Point const& p) { return p.z; });
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(tt),[](Point const& p) { return p.time; });
	std::transform(ds.m_points.begin(), ds.m_points.end(), std::back_inserter(ss),[](Point const& p) { return p.station; });

	if(labels.size()>0)
	  {
	    n_noise = std::count (labels.begin(), labels.end(), -1);
	    int max = *max_element(labels.begin(), labels.end());
	    if(max == -1) n_clusters = 0;
	    else n_clusters = max+1;// - 1*int( bool(n_noise_) );
	  }
	//Fit of the cosmic trajectory if present
	//Choose the right pair of cosmic clouds
	if(n_clusters>=2 and nCosmicMuonsOneLeg>0 and nCosmicMuons>1)
	  {
	    std::vector<std::vector<float>> vec_xx(n_clusters,std::vector<float>());
	    std::vector<std::vector<float>> vec_yy(n_clusters,std::vector<float>());
	    std::vector<std::vector<float>> vec_zz(n_clusters,std::vector<float>());
	    std::vector<std::vector<int>>   vec_label(n_clusters,std::vector<int>());
	    std::vector<std::vector<float>> vec_tt(n_clusters,std::vector<float>());
	    std::vector<std::vector<int>>   vec_ss(n_clusters,std::vector<int>());

	    for(unsigned int l=0;l<labels.size();l++)
	      {
		if(labels.at(l)>-1)
		{
		  vec_label.at( labels.at(l) ).push_back(labels.at(l));
		  vec_xx.at( labels.at(l) ).push_back(xx.at(l));
		  vec_yy.at( labels.at(l) ).push_back(yy.at(l));
		  vec_zz.at( labels.at(l) ).push_back(zz.at(l));
		  if(tt.at(l)>-9999.)
		    {
		      //Default at -9999. messes up the average
		      //TODO: include the median!!
		      vec_tt.at( labels.at(l) ).push_back(tt.at(l));
		    }
		  else
		    {
		      vec_tt.at( labels.at(l) ).push_back(0.);
		    }
		  vec_ss.at( labels.at(l) ).push_back(ss.at(l));
		}
	      }

	    //I now have n_clusters vectors
	    //I can loop over the clusters
	    int ch_k1 = -1;
	    int ch_k2 = -1;
	    float mean_time_ch_k1 = -9999.;
	    float mean_time_ch_k2 = -9999.;
	    float std_time_ch_k1 = -9999.;
	    float std_time_ch_k2 = -9999.;
	    int n_s_ch_k1 = -1;
	    int n_s_ch_k2 = -1;
	    float dz_DT = 1000.;
	    float dz_ECAL = 1000.;
	    //float dz_acc_ECAL = 1000.;

	    for(int k1 = 0; k1<n_clusters; k1++)
	      {

		//for(int k2 = 1; k2<n_clusters && k2>k1; k2++)
		for(int k2 = k1+1; k2<n_clusters && k2!=k1; k2++)//new loop giving problems! misses some events!!
		//for example it misses 297411:830:1385167469 Run2017B that is a clear cosmic!
		  {
		    float mean_k1_x=avg(vec_xx.at(k1));
		    float mean_k1_y=avg(vec_yy.at(k1));
		    float mean_k1_z=avg(vec_zz.at(k1));
		    float mean_k1_t=avg(vec_tt.at(k1));
		    float std_k1_t=stdev(vec_tt.at(k1));
 		    std::vector<int> stations_k1 = vec_ss.at(k1); 
		    stations_k1.resize(std::distance(stations_k1.begin(), std::unique(stations_k1.begin(), stations_k1.end())  ));
		    int n_k1_s = stations_k1.size();

		    float mean_k2_x=avg(vec_xx.at(k2));
		    float mean_k2_y=avg(vec_yy.at(k2));
		    float mean_k2_z=avg(vec_zz.at(k2));
		    float mean_k2_t=avg(vec_tt.at(k2));
		    float std_k2_t=stdev(vec_tt.at(k2));
		    std::vector<int> stations_k2 = vec_ss.at(k2); 
		    stations_k2.resize(std::distance(stations_k2.begin(), std::unique(stations_k2.begin(), stations_k2.end())  ));
		    int n_k2_s = stations_k2.size();


		    if(  (mean_k1_x*mean_k2_x<0 or mean_k1_y*mean_k2_y<0 or mean_k1_z*mean_k2_z<0)  )
		      {
			float tmp_z = abs(mean_k1_z - mean_k2_z);
			dz_DT = std::min(dz_DT,tmp_z);
			//TODO: can probably compute the mean instead of doing the average
			//this choice depends on what ecal rec hits we consider...
			//THIS: choice based on 1p0 jets (less restrictive veto)

			//Here: choice of calo hits associated to the cosmic
			//If no tagged jet, look at non-tagged rec hits
			////Here: all eta
			////float tmp_ECAL = abs((mean_k1_z+mean_k2_z)/2. - mean_ECAL_tag_z);
			//Here: eta<1
			float tmp_ECAL = 99999999.;
			//Make the decision based on all rec hits up to 1.4
			//More conservative but probably better
			//earlier: based on taggedAcceptanceEcalRecHitsAK4, too "loose"
			if(taggedEcalRecHitsAK4.size()>0)
			  {
			    //tmp_ECAL = abs((mean_k1_z+mean_k2_z)/2. - mean_acc_ECAL_tag_z);
			    tmp_ECAL = abs((mean_k1_z+mean_k2_z)/2. - mean_ECAL_tag_z);
			    isCosmicVetoWithTags = true;
			  }
			else
			  {
			    if(skimmedEcalRecHitsAK4.size()>0)
			      {
				tmp_ECAL = abs((mean_k1_z+mean_k2_z)/2. - mean_ECAL_z);
				//dz_ECAL = std::min(dz_ECAL,tmp_ECAL);
			      }
			    else
			      {
				tmp_ECAL = 99999999.;//very large number so that this is always false
				//dz_ECAL = std::min(dz_ECAL,tmp_ECAL);
			      }
			  }
			dz_ECAL = std::min(dz_ECAL,tmp_ECAL);
			////Here: all eta
			////if(dz_DT==tmp_z and dz_ECAL==tmp_ECAL and taggedEcalRecHitsAK4.size()>0)
			//Here: eta<1
			if(dz_DT==tmp_z and dz_ECAL==tmp_ECAL)// and taggedAcceptanceEcalRecHitsAK4.size()>0)
			  {
                            ch_k1 = k1;
                            ch_k2 = k2;
			    mean_time_ch_k1 = mean_k1_t;
			    mean_time_ch_k2 = mean_k2_t;
			    std_time_ch_k1 = std_k1_t;
			    std_time_ch_k2 = std_k2_t;
			    n_s_ch_k1 = n_k1_s;
			    n_s_ch_k2 = n_k2_s;
			  }
		      }//opposite condition
		  }//loop k2

	      }//loop k1

	    //For printing purposes
	    if(ch_k1>-1 and ch_k2>-1)
	      {
		//cout << "Chosen pair: " << ch_k1 << " " << ch_k2 << endl;
		bool smallness1(false);
		bool smallness2(false);
		bool smallness(false);
		if(abs(avg(vec_yy.at(ch_k1)))<0.2) smallness1=true;
		if(abs(avg(vec_yy.at(ch_k2)))<0.2) smallness2=true;
		smallness = (smallness1 || smallness2);

		if(printFit) std::cout << TagNumber << " & " << avg(vec_xx.at(ch_k1)) << " & " << avg(vec_xx.at(ch_k2)) << " & " << (avg(vec_xx.at(ch_k1))*avg(vec_xx.at(ch_k2))<0)  << " & " << avg(vec_yy.at(ch_k1)) << " & " << avg(vec_yy.at(ch_k2)) << " & " << (avg(vec_yy.at(ch_k1))*avg(vec_yy.at(ch_k2))<0)  << " & " << smallness << "\\\\ " << std::endl;

		if(printFit) std::cout << TagNumber << " & " << mean_time_ch_k1 << " & " << mean_time_ch_k2 << " & " << (mean_time_ch_k1*mean_time_ch_k2<0) << " & " << std_time_ch_k1 << " & " << std_time_ch_k2 << " & " << n_s_ch_k1 << " & " << n_s_ch_k2 << "\\\\ " << std::endl;
		    
		//printResults(ds.m_points, ds.getTotalPointSize()); 

		DT_fit_xx.reserve(vec_xx.at(ch_k1).size() + vec_xx.at(ch_k2).size() );
		DT_fit_xx.insert( DT_fit_xx.end(), vec_xx.at(ch_k1).begin(), vec_xx.at(ch_k1).end());
		DT_fit_xx.insert( DT_fit_xx.end(), vec_xx.at(ch_k2).begin(), vec_xx.at(ch_k2).end());
		DT_fit_yy.reserve(vec_yy.at(ch_k1).size() + vec_yy.at(ch_k2).size() );
		DT_fit_yy.insert( DT_fit_yy.end(), vec_yy.at(ch_k1).begin(), vec_yy.at(ch_k1).end());
		DT_fit_yy.insert( DT_fit_yy.end(), vec_yy.at(ch_k2).begin(), vec_yy.at(ch_k2).end());
		DT_fit_zz.reserve(vec_zz.at(ch_k1).size() + vec_zz.at(ch_k2).size() );
		DT_fit_zz.insert( DT_fit_zz.end(), vec_zz.at(ch_k1).begin(), vec_zz.at(ch_k1).end());
		DT_fit_zz.insert( DT_fit_zz.end(), vec_zz.at(ch_k2).begin(), vec_zz.at(ch_k2).end());


		//TODO these vector insertion are probably not needed and can fit the VectorXf directly
		Map<VectorXf> VX(DT_fit_xx.data(),DT_fit_xx.size());
		Map<VectorXf> VY(DT_fit_yy.data(),DT_fit_yy.size());
		Map<VectorXf> VZ(DT_fit_zz.data(),DT_fit_zz.size());
		VectorXf One(DT_fit_xx.size());
		VectorXf SolXZ(2);
		VectorXf SolYZ(2);
		One.setOnes();
		//here Axz, Ayz
		MatrixXf Axz(DT_fit_xx.size(),2);
		Axz << VX, One;
		MatrixXf Ayz(DT_fit_xx.size(),2);
		Ayz << VY , One;
		SolXZ = (Axz.transpose() * Axz).ldlt().solve(Axz.transpose() * VZ);
		SolYZ = (Ayz.transpose() * Ayz).ldlt().solve(Ayz.transpose() * VZ);
		m_xz = SolXZ[0];
		c_xz = SolXZ[1];
		m_yz = SolYZ[0];
		c_yz = SolYZ[1];
		
		//Leave for debugging purposes:
		//cout << "The Axz solution using normal equations is:\n"
		//     << " m: " << SolXZ[0] << " ; c: " << SolXZ[1] << endl;
		//cout << "The Ayz solution using normal equations is:\n"
		//     << " m: " << SolYZ[0] << " ; c: " << SolYZ[1] << endl;

		//Debugging
		//cout << "dist from origin " << sqrt(distance2(0.,0.,0.,SolXZ,SolYZ)) << endl;
		//cout << "dist from ECAL " << sqrt(distance2(mean_ECAL_tag_x,mean_ECAL_tag_y,mean_ECAL_tag_z,SolXZ,SolYZ)) << endl;
		//cout << "dt_fit_chi2 " << dt_fit_chi2 << endl;

		if(DT_fit_xx.size()>0)
		  {
		    //Only if we have valid fit points: reset chi squared
		    dt_fit_chi2 = 0.;
		    for(unsigned int c=0; c<DT_fit_xx.size(); c++)
		      {
			float tmp_dist2 = distance2(DT_fit_xx.at(c),DT_fit_yy.at(c),DT_fit_zz.at(c),SolXZ,SolYZ);
			dt_fit_chi2    += tmp_dist2;
			DT_fit_res.push_back(tmp_dist2);
		      }
		    
		    if(printFit) cout << "dt_fit_chi2: " << dt_fit_chi2 <<endl;
		    isDT_fit = true;
		    dt_fit_chi2_reduced = dt_fit_chi2/DT_fit_xx.size();
		    //If fit performed and rec hits not empty, calculate distance
		    if(taggedEcalRecHitsAK4.size()>0) dt_ecal_dist = sqrt(distance2(mean_ECAL_tag_x,mean_ECAL_tag_y,mean_ECAL_tag_z,SolXZ,SolYZ));
		    if(taggedAcceptanceEcalRecHitsAK4.size()>0) dt_ecal_acc_dist = sqrt(distance2(mean_acc_ECAL_tag_x,mean_acc_ECAL_tag_y,mean_acc_ECAL_tag_z,SolXZ,SolYZ));
		    if(skimmedEcalRecHitsAK4.size()>0) dt_ecal_no_tag_dist = sqrt(distance2(mean_ECAL_x,mean_ECAL_y,mean_ECAL_z,SolXZ,SolYZ));
		    if(skimmedAcceptanceEcalRecHitsAK4.size()>0) dt_ecal_acc_no_tag_dist = sqrt(distance2(mean_acc_ECAL_x,mean_acc_ECAL_y,mean_acc_ECAL_z,SolXZ,SolYZ));
		  }

		if(printFit) cout << "   Final result   " << endl;
		if(printFit) std::cout << TagNumber<< " & " << nTagJets_0p996 << " & " << DT_fit_xx.size()  << " &" << dt_fit_chi2 << " & " << dt_fit_chi2_reduced << " & " << sqrt(distance2(mean_ECAL_tag_x,mean_ECAL_tag_y,mean_ECAL_tag_z,SolXZ,SolYZ)) << "\\\\ " << std::endl;
		if(printFit) cout << "      " << endl;


		if(dt_ecal_dist<0.5)
		  {
		    isCosmic = true;
		  }

	      }

	  }//if 2 clusters


	//Veto objects
	if(doSR and nMuonsPassing!=0) continue;
	if(doSR and nElectronsPassing!=0) continue;
	if(doSR and nTausPassing!=0) continue;
	if(doSR and nPhotonsPassing!=0) continue;


	//Here select bin 1/2 LISA
	//if(doSR and nTagJets_0p996<1) continue;


	//BH veto and eta/phi cuts

	if( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) and MEt->pt>200 and nMuonsPassing==0 and nElectronsPassing==0 and nTausPassing==0 and nPhotonsPassing==0 and dt_ecal_dist>0.5 and MinJetMetDPhi>0.5) isPreselected = true;

	int n_jets_tagged_close_phi_low_multi (0);
	if(min_dPhi_jets_eta_1p0_0p996<0.05)
	  {
	    
	    for (unsigned int j=0; j<skimmedJets.size(); j++)
	      {
		
		if(skimmedJets.at(j).sigprob>0.996 and skimmedJets.at(j).nRecHitsEB<=10) n_jets_tagged_close_phi_low_multi++;
	      }
	    //loop over tagged jets and count their multiplicity
	  }
	//reject beam halo-like jets
	if (n_jets_tagged_close_phi_low_multi>0) isPreselected = false;

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

    //std::cout << "**************************************************" << std::endl;
    //std::cout << "finished  computations at " << std::ctime(&end_time)
    //      << "elapsed time: " << elapsed_seconds.count() << "s\n";
    //std::cout << "**************************************************" << std::endl;
    //std::cout << " " << std::endl;

    std::cout << "**************************************************" << std::endl;
    std::cout << "Output written: " << outputPath << std::endl;
    std::cout << "\n" << std::endl;

    return 0;
}


//DBSCAN functions
int DBSCAN::run()
{
  int clusterID = 0;//Original was 1!
  vector<Point>::iterator iter;
  for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
      if ( iter->clusterID == UNCLASSIFIED )
        {
	  if ( expandCluster(*iter, clusterID) != FAILURE )
            {
	      clusterID += 1;
            }
        }
    }

  return 0;
}

int DBSCAN::expandCluster(Point point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y && m_points.at(*iterSeeds).z == point.z )
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(Point point)
{
    int index = 0;
    vector<Point>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(const Point& pointCore, const Point& pointTarget )
{
    return pow(pointCore.x - pointTarget.x,2)+pow(pointCore.y - pointTarget.y,2)+pow(pointCore.z - pointTarget.z,2);
}


