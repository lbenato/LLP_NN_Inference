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
    bool doGen(false);
    if(strcmp(argv[9], "doGen")==0) doGen=true;

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
    if(doGen) std::cout << "Gen studies, no selections" << std::endl;


    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    std::string inputPath = argv[1];

    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";

    std::string mcPUFilename = argv[6];
    std::string mcTriggerFilename = argv[7];
    std::string mcTriggerString = argv[8];

    std::string inputTreeName = "ntuple/tree";
    std::string outputTreeName = "tree";//inputTreeName;


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
    int    MeanNumInteractions;
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v(false);
    bool   HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v(false);

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

    std::vector<TauType>         *Taus = 0;
    std::vector<PhotonType>      *Photons = 0;
    std::vector<LeptonType>      *Muons = 0;
    std::vector<LeptonType>      *Electrons = 0;
    std::vector<JetType>         *Jets = 0;
    MEtType                      *MEt = 0;

    std::vector<float>           *QCDWeights = 0;
    std::vector<float>           *PDFWeights = 0;

    // Input branches
    TBranch        *b_Taus = 0;
    TBranch        *b_Photons = 0;
    TBranch        *b_Muons = 0;
    TBranch        *b_Electrons = 0;
    TBranch        *b_Jets = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_QCDWeights = 0;
    TBranch        *b_PDFWeights = 0;
    TBranch        *b_EventNumber;
    TBranch        *b_RunNumber;
    TBranch        *b_LumiNumber;
    TBranch        *b_EventWeight;
    TBranch        *b_PUWeight;
    TBranch        *b_isMC;
    TBranch        *b_MeanNumInteractions;
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
    TBranch        *b_MinJetMetDPhi;

    inputTree->SetBranchAddress("Taus",              &Taus,              &b_Taus);
    inputTree->SetBranchAddress("Photons",           &Photons,           &b_Photons);
    inputTree->SetBranchAddress("Muons",             &Muons,             &b_Muons);
    inputTree->SetBranchAddress("Electrons",         &Electrons,         &b_Electrons);
    inputTree->SetBranchAddress("Jets",              &Jets,              &b_Jets);
    inputTree->SetBranchAddress("MEt",               &MEt,               &b_MEt); 
    inputTree->SetBranchAddress("QCDWeights",       &QCDWeights,       &b_QCDWeights); 
    inputTree->SetBranchAddress("PDFWeights",       &PDFWeights,       &b_PDFWeights); 
    inputTree->SetBranchAddress("EventNumber",       &EventNumber,       &b_EventNumber);
    inputTree->SetBranchAddress("RunNumber",         &RunNumber,         &b_RunNumber);
    inputTree->SetBranchAddress("LumiNumber",        &LumiNumber,        &b_LumiNumber);
    inputTree->SetBranchAddress("EventWeight",       &EventWeight,       &b_EventWeight);
    inputTree->SetBranchAddress("PUWeight",          &PUWeight,          &b_PUWeight);
    inputTree->SetBranchAddress("isMC",              &isMC,              &b_isMC);
    inputTree->SetBranchAddress("MeanNumInteractions",  &MeanNumInteractions,  &b_MeanNumInteractions);
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
    bool isSR_bin2(false);

    std::vector<TauType>    skimmedTaus;
    std::vector<JetType>    skimmedJets;
   
    std::vector<float> PDFWeightsSR;
    std::vector<float> QCDWeightsSR;

    float PUReWeight(1.);
    //float PUReWeightUp(1.);
    //float PUReWeightDown(1.);
    float TriggerWeight(1.);
    float MinJetMetDPhi(10.);

    int nMuonsPassing(0);
    int nElectronsPassing(0);
    int nPhotonsPassing(0);
    int nPhotonsTight(0);
    int nTausPassing(0);

    int nTagJets_0p996(0);

    //Systematic uncertainties 

    // Output branches 
    outputTree->Branch("EventNumber",       &EventNumber,       "EventNumber/L");
    outputTree->Branch("RunNumber",         &RunNumber,         "RunNumber/L");
    outputTree->Branch("LumiNumber",        &LumiNumber,        "LumiNumber/L");
    outputTree->Branch("EventWeight",       &EventWeight,       "EventWeight/F");
    //outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    //outputTree->Branch("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    //outputTree->Branch("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    outputTree->Branch("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    outputTree->Branch("isSR_bin2",              &isSR_bin2,              "isSR_bin2/O");
    //outputTree->Branch("MeanNumInteractions",             &MeanNumInteractions,             "MeanNumInteractions/I");
    //outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v/O");
    //outputTree->Branch("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v", &HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v, "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v/O");
    //outputTree->Branch("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v", &HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v, "HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v/O");
    //outputTree->Branch("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v", &HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v, "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v/O");

    outputTree->Branch("MinJetMetDPhi",  &MinJetMetDPhi,  "MinJetMetDPhi/F");
    outputTree->Branch("nTagJets_0p996",     &nTagJets_0p996,      "nTagJets_0p996/I");

    //outputTree->Branch("Jets", &skimmedJets);
    //outputTree->Branch("MEt", &MEt);
    outputTree->Branch("QCDWeights", &QCDWeights);
    outputTree->Branch("PDFWeights", &PDFWeights);
    outputTree->Branch("QCDWeightsSR", &QCDWeightsSR);
    outputTree->Branch("PDFWeightsSR", &PDFWeightsSR);


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


    // Event loop

    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        TriggerWeight = 1.;
        PUReWeight = 1.;
        //PUReWeightUp = 1.;
        //PUReWeightDown = 1.;
	//Initialize nTagJets at every event

	MinJetMetDPhi = 10.;
	//Initialize veto objects counter
	nPhotonsPassing = 0;
	nPhotonsTight = 0;
	nTausPassing = 0;
	nMuonsPassing = 0;
	nElectronsPassing = 0;

 

	nTagJets_0p996 = 0;

	//Clear all the vectors
	//very dangerous with continue statement!
	skimmedTaus.clear();
        skimmedJets.clear();
	PDFWeightsSR.clear();
	QCDWeightsSR.clear();

        //if (i % 1000 == 0) {
        //    std::cout << "evaluating entry " << i << std::endl;
        //}
        inputTree->GetEntry(i);



	if(isMC)
	  {
	    TriggerWeight = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->pt));//only for SR MC!!
	    PUReWeight = pu->GetBinContent(pu->GetXaxis()->FindBin(MeanNumInteractions));
	  }


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
	float tau_iso = 0.5;
	for (unsigned int t=0; t<Taus->size(); t++)
	  {
	    //JJ uses "decayModeFindingNewDMs" and  byLoose, which is incorrect
	    //correct would be: "decayModeFinding"
	    if(Taus->at(t).decayModeFinding == true and Taus->at(t).byLooseCombinedIsolationDeltaBetaCorr3Hits == true)
	      {
		skimmedTaus.push_back(Taus->at(t));


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


	for (unsigned int j=0; j<Jets->size(); j++)
	  {
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi) MinJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
	      }
	    
	    //Apply all jet selections, no time smearing
	    if( fabs(Jets->at(j).eta)<1. and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1.)
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
		
		    
		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		Jets->at(j).eFracRecHitsEB = (Jets->at(j).energy>0 and Jets->at(j).energyRecHitsEB>0) ? Jets->at(j).energyRecHitsEB/Jets->at(j).energy : -1.;
		
		//Fix also timeRMS dividing by sqrt nRecHitsEB
		Jets->at(j).timeRMSRecHitsEB = (Jets->at(j).nRecHitsEB>0) ? Jets->at(j).timeRMSRecHitsEB/sqrt(Jets->at(j).nRecHitsEB) : -1.;

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

		if(outputValueAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p996++;

		//store jets passing acceptance and with inference
		skimmedJets.push_back(Jets->at(j));

	      }//acceptance

	  }//jet loop


	//Minimal set of cuts: let's ignore cosmic and beam halo veto
	if( (HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) and MEt->pt>200 and (Flag2_globalSuperTightHalo2016Filter) and (Flag2_EcalDeadCellTriggerPrimitiveFilter) and (Flag2_HBHENoiseFilter) and (Flag2_HBHEIsoNoiseFilter) and (Flag2_ecalBadCalibFilter) and (Flag2_eeBadScFilter) and (Flag2_BadPFMuonFilter) and nMuonsPassing==0 and nElectronsPassing==0 and nTausPassing==0 and nPhotonsPassing==0 and MinJetMetDPhi>0.5 and nTagJets_0p996>1) isSR_bin2=true;

	//Here: fill weights if SR passed
	if(isSR_bin2)
	  {
	    std::copy(PDFWeights->begin(), PDFWeights->end(), back_inserter(PDFWeightsSR)); 
	    std::copy(QCDWeights->begin(), QCDWeights->end(), back_inserter(QCDWeightsSR)); 
	  }

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


