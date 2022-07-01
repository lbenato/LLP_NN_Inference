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
  bool isLoose;
  bool isTight;
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


///
void FillJetCaloType(JetCaloType& I, JetType& R, bool isMC) {
    I.pt          = R.pt;
    I.eta         = R.eta;
    I.phi         = R.phi;
    I.mass        = R.mass;
    I.energy      = R.energy;
    I.ptGenJ      = R.ptGenJ;
    I.etaGenJ     = R.etaGenJ;
    I.phiGenJ     = R.phiGenJ;
    I.massGenJ    = R.massGenJ;
    I.ptGen       = R.ptGen;
    I.etaGen      = R.etaGen;
    I.phiGen      = R.phiGen;
    I.massGen     = R.massGen;
    I.pdgIdGen    = R.pdgIdGen;
    I.energyRaw   = R.energyRaw;
    I.ptRaw       = R.ptRaw;
    I.ptUnc       = R.ptUnc;
    I.dPhi_met    = R.dPhi_met;
    I.CSV         = R.CSV;
    I.CMVA        = R.CMVA;
    I.cHadE       = R.cHadE;
    I.nHadE       = R.nHadE;
    I.eleE        = R.eleE;
    I.photonE     = R.photonE;
    I.muE         = R.muE;
    I.nEmE        = R.nEmE;
    I.cEmE        = R.cEmE;
    I.cmuE        = R.cmuE;
    I.cHadEFrac   = R.cHadEFrac;
    I.nHadEFrac   = R.nHadEFrac;
    I.eleEFrac    = R.eleEFrac;
    I.photonEFrac = R.photonEFrac;
    I.muEFrac     = R.muEFrac;
    I.nEmEFrac    = R.nEmEFrac;
    I.cEmEFrac    = R.cEmEFrac;
    I.cmuEFrac    = R.cmuEFrac;
    I.cHadMulti   = R.cHadMulti;
    I.nHadMulti   = R.nHadMulti;
    I.eleMulti    = R.eleMulti;
    I.photonMulti = R.photonMulti;
    I.muMulti     = R.muMulti;
    I.cMulti      = R.cMulti;
    I.nMulti      = R.nMulti;
    I.npr         = R.npr;
    I.cHadMultiFrac   = R.cHadMultiFrac;
    I.nHadMultiFrac   = R.nHadMultiFrac;
    I.eleMultiFrac    = R.eleMultiFrac;
    I.photonMultiFrac = R.photonMultiFrac;
    I.muMultiFrac     = R.muMultiFrac;
    I.cMultiFrac      = R.cMultiFrac;
    I.nMultiFrac      = R.nMultiFrac;
    I.partonFlavour   = R.partonFlavour;
    I.hadronFlavour   = R.hadronFlavour;
    I.mother = R.mother;
    I.isLoose     = R.isLoose;
    I.isTight     = R.isTight;
    I.isTightLepVeto     = R.isTightLepVeto;

    I.matchBquark = R.matchBquark;
    I.matchLL     = R.matchLL;
    I.isGenMatched = R.isGenMatched;
    I.isGenMatchedCaloCorr = R.isGenMatchedCaloCorr;
    I.isGenMatchedLLPAccept = R.isGenMatchedLLPAccept;
    I.isGenMatchedCaloCorrLLPAccept = R.isGenMatchedCaloCorrLLPAccept;
    I.radiusLLP = R.radiusLLP;
    I.xLLP = R.xLLP;
    I.yLLP = R.yLLP;
    I.zLLP = R.zLLP;
    I.radiusLLPCaloCorr = R.radiusLLPCaloCorr;
    I.xLLPCaloCorr = R.xLLPCaloCorr;
    I.yLLPCaloCorr = R.yLLPCaloCorr;
    I.zLLPCaloCorr = R.zLLPCaloCorr;
    I.xGenb = R.xGenb;
    I.yGenb = R.yGenb;
    I.zGenb = R.zGenb;
    I.xGenbCaloCorr = R.xGenbCaloCorr;
    I.yGenbCaloCorr = R.yGenbCaloCorr;
    I.zGenbCaloCorr = R.zGenbCaloCorr;
    I.isVBFGenMatched = R.isVBFGenMatched;
    //track, new implementation
    I.ptAllTracks    = R.ptAllTracks;
    I.ptAllPVTracks  = R.ptAllPVTracks;
    I.ptPVTracksMax  = R.ptPVTracksMax;
    I.nTracksAll     = R.nTracksAll;
    I.nTracksPVMax   = R.nTracksPVMax;
    I.medianIP2D     = R.medianIP2D;
    I.medianTheta2D  = R.medianTheta2D;
    I.alphaMax       = R.alphaMax;
    I.betaMax        = R.betaMax;
    I.gammaMax       = R.gammaMax;
    I.gammaMaxEM     = R.gammaMaxEM;
    I.gammaMaxHadronic  = R.gammaMaxHadronic;
    I.gammaMaxET     = R.gammaMaxET;
    I.minDeltaRAllTracks = R.minDeltaRAllTracks;
    I.minDeltaRPVTracks  = R.minDeltaRPVTracks;
    I.nHitsMedian    = R.nHitsMedian;
    I.nPixelHitsMedian    = R.nPixelHitsMedian;
    I.dzMedian       = R.dzMedian;
    I.dxyMedian      = R.dxyMedian;  
    I.hcalE       = R.hcalE;
    I.ecalE       = R.ecalE;
    I.FracCal     = R.FracCal;
    I.isCaloTag   = R.isCaloTag;
    I.ptJESUp     = R.ptJESUp;
    I.ptJESDown   = R.ptJESDown;

    I.ptJER       = R.ptJER;
    I.ptJERUp     = R.ptJERUp;
    I.ptJERDown   = R.ptJERDown;
    I.energyJER   = R.energyJER;
    I.energyJERUp = R.energyJERUp;
    I.energyJERDown = R.energyJERDown;
    I.etaJER      = R.etaJER;
    I.etaJERUp    = R.etaJERUp;
    I.etaJERDown  = R.etaJERDown;
    //scale factors
    I.JERresolution    = R.JERresolution;
    I.JERsf            = R.JERsf;
    I.JERsfUp          = R.JERsfUp;
    I.JERsfDown        = R.JERsfDown;
    I.JERsmearFactor   = R.JERsmearFactor;
    I.JERsmearFactorUp = R.JERsmearFactorUp;
    I.JERsmearFactorDown = R.JERsmearFactorDown;

    I.nConstituents      = R.nConstituents;
    I.nTrackConstituents = R.nTrackConstituents;
    I.nTrackConstituentsWithPtLarger0p95 = R.nTrackConstituentsWithPtLarger0p95;
    I.nTrackConstituentsWithTrackDetails = R.nTrackConstituentsWithTrackDetails;
    I.nTrackConstituentsWithTrackDetailsPtLarger0p95 = R.nTrackConstituentsWithTrackDetailsPtLarger0p95;
    I.nMatchedGenBquarks = R.nMatchedGenBquarks;
    I.nMatchedGenBquarksCaloCorr = R.nMatchedGenBquarksCaloCorr;

    I.nRecHitsEB       = R.nRecHitsEB;
    I.timeRecHitsEB    = R.timeRecHitsEB;
    I.timeRMSRecHitsEB = R.timeRMSRecHitsEB;
    I.energyRecHitsEB  = R.energyRecHitsEB;
    I.energyErrorRecHitsEB = R.energyErrorRecHitsEB;
    I.xRecHitsEB       = R.xRecHitsEB;
    I.yRecHitsEB       = R.yRecHitsEB;
    I.zRecHitsEB       = R.zRecHitsEB;
    I.radiusRecHitsEB  = R.radiusRecHitsEB;

    I.nRecHitsEE       = R.nRecHitsEE;
    I.timeRecHitsEE    = R.timeRecHitsEE;
    I.timeRMSRecHitsEE = R.timeRMSRecHitsEE;
    I.energyRecHitsEE  = R.energyRecHitsEE;
    I.energyErrorRecHitsEE = R.energyErrorRecHitsEE;
    I.xRecHitsEE       = R.xRecHitsEE;
    I.yRecHitsEE       = R.yRecHitsEE;
    I.zRecHitsEE       = R.zRecHitsEE;
    I.radiusRecHitsEE  = R.radiusRecHitsEE;

    I.nRecHitsHB = R.nRecHitsHB;
    I.timeRecHitsHB = R.timeRecHitsHB;
    I.timeRMSRecHitsHB = R.timeRMSRecHitsHB;
    I.energyRecHitsHB = R.energyRecHitsHB;
    I.energyErrorRecHitsHB = R.energyErrorRecHitsHB;
    I.xRecHitsHB = R.xRecHitsHB;
    I.yRecHitsHB = R.yRecHitsHB;
    I.zRecHitsHB = R.zRecHitsHB;
    I.radiusRecHitsHB = R.radiusRecHitsHB;
    I.nRecHitsHE = R.nRecHitsHE;
    I.timeRecHitsHE = R.timeRecHitsHE;
    I.timeRMSRecHitsHE = R.timeRMSRecHitsHE;
    I.energyRecHitsHE = R.energyRecHitsHE;
    I.energyErrorRecHitsHE = R.energyErrorRecHitsHE;
    I.xRecHitsHE = R.xRecHitsHE;
    I.yRecHitsHE = R.yRecHitsHE;
    I.zRecHitsHE = R.zRecHitsHE;
    I.radiusRecHitsHE = R.radiusRecHitsHE;

    I.eFracRecHitsEB = R.eFracRecHitsEB;
    I.eFracRecHitsEE = R.eFracRecHitsEE;
    I.eFracRecHitsHB = R.eFracRecHitsHB;
    I.eFracRecHitsHE = R.eFracRecHitsHE;

    I.sig1EB  = R.sig1EB;
    I.sig2EB  = R.sig2EB;
    I.sigAvEB = R.sigAvEB;
    I.tan2thetaEB  = R.tan2thetaEB;
    I.ptDEB  = R.ptDEB;
    I.sig1EE  = R.sig1EE;
    I.sig2EE  = R.sig2EE;
    I.sigAvEE = R.sigAvEE;
    I.tan2thetaEE  = R.tan2thetaEE;
    I.ptDEE  = R.ptDEE;
    I.sig1HB  = R.sig1HB;
    I.sig2HB  = R.sig2HB;
    I.sigAvHB = R.sigAvHB;
    I.tan2thetaHB  = R.tan2thetaHB;
    I.ptDHB  = R.ptDHB;

    I.sig1PF  = R.sig1PF;
    I.sig2PF  = R.sig2PF;
    I.sigAvPF = R.sigAvPF;
    I.tan2thetaPF  = R.tan2thetaPF;
    I.ptDPF  = R.ptDPF;
    I.sigprob     = R.sigprob;
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
    bool doZtoMM(false);
    if(strcmp(argv[9], "doZtoMM")==0) doZtoMM=true;
    bool doZtoEE(false);
    if(strcmp(argv[9], "doZtoEE")==0) doZtoEE=true;

    bool doZtoMMPho(false);
    if(strcmp(argv[9], "doZtoMMPho")==0) doZtoMMPho=true;
    bool doZtoEEPho(false);
    if(strcmp(argv[9], "doZtoEEPho")==0) doZtoEEPho=true;

    bool doPho(false);
    if(strcmp(argv[9], "doPho")==0) doPho=true;

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
    if(doZtoMM) std::cout << "ZtoMM selections" << std::endl;
    if(doZtoEE) std::cout << "ZtoEE selections" << std::endl;
    if(doZtoMMPho) std::cout << "ZtoMMPho selections" << std::endl;
    if(doZtoEEPho) std::cout << "ZtoEEPho selections" << std::endl;


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
    //Long64_t nDTSegments;
    //Long64_t nCSCSegments;
    //Long64_t nCosmicMuons, nCosmicMuonsOneLeg;
    int    nElectrons;
    int    nMuons;
    int    nPhotons;
    //int    nTaus;
    int    nPFCandidates;
    int    nPFCandidatesTrack;
    int    nLLPInCalo;
    //int    m_chi;
    //int    ctau;
    //bool   is_central;
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
    //std::vector<DT4DSegmentType> *DTSegments = 0;
    //std::vector<CSCSegmentType>  *CSCSegments = 0;

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
    TBranch        *b_EcalRecHitsAK4 = 0;
    //TBranch        *b_DTSegments = 0;
    //TBranch        *b_CSCSegments = 0;
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
    //TBranch        *b_nDTSegments;
    //TBranch        *b_nCSCSegments;
    //TBranch        *b_nCosmicMuons, *b_nCosmicMuonsOneLeg;
    TBranch        *b_nElectrons;
    TBranch        *b_nMuons;
    TBranch        *b_nPhotons;
    //TBranch        *b_nTaus;
    TBranch        *b_nPFCandidates;
    TBranch        *b_nPFCandidatesTrack;
    TBranch        *b_nLLPInCalo;
    //TBranch        *b_m_chi;
    //TBranch        *b_ctau;
    //TBranch        *b_is_central;

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
    //inputTree->SetBranchAddress("nDTSegments",       &nDTSegments,       &b_nDTSegments);
    //inputTree->SetBranchAddress("nCSCSegments",      &nCSCSegments,      &b_nCSCSegments);
    //inputTree->SetBranchAddress("nCosmicMuons",      &nCosmicMuons,      &b_nCosmicMuons);
    //inputTree->SetBranchAddress("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      &b_nCosmicMuonsOneLeg);
    //inputTree->SetBranchAddress("nTaus",             &nTaus,             &b_nTaus);
    inputTree->SetBranchAddress("nPFCandidates",     &nPFCandidates,     &b_nPFCandidates);
    inputTree->SetBranchAddress("nPFCandidatesTrack", &nPFCandidatesTrack, &b_nPFCandidatesTrack);
    inputTree->SetBranchAddress("nLLPInCalo", &nLLPInCalo, &b_nLLPInCalo);
    //inputTree->SetBranchAddress("m_chi", &m_chi, &b_m_chi);
    //inputTree->SetBranchAddress("ctau", &ctau, &b_ctau);
    //inputTree->SetBranchAddress("is_central", &is_central, &b_is_central);
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

    TTree *outputTree = new TTree(outputTreeName.c_str(), "");


    //Flags for SR/CR
    bool isZtoMM(false);
    bool isZtoEE(false);

    bool isZtoMMPho(false);
    bool isZtoEEPho(false);

    bool isPho(false);

    bool isCosmic(false);
    bool isDT_fit(false);
    bool isCosmicVetoWithTags(false);

    std::vector<TauType>    skimmedTaus;
    std::vector<JetType>    skimmedJets;
    std::vector<JetType>    skimmedJetsNegative;
    std::vector<ecalRecHitType> skimmedEcalRecHitsAK4;
    std::vector<ecalRecHitType> skimmedAcceptanceEcalRecHitsAK4;
    std::vector<float>          skimmedEBEnergyCSC;
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
    float min_dPhi_jets(9999.);
    float min_dEta_jets(9999.);
    float min_dR_jets(9999.);
    float min_dPhi_jets_0p7(9999.);
    float min_dEta_jets_0p7(9999.);
    float min_dR_jets_0p7(9999.);
    float min_dPhi_jets_0p9(9999.);
    float min_dEta_jets_0p9(9999.);
    float min_dR_jets_0p9(9999.);
    float min_dPhi_jets_0p9_no_tags(9999.);
    float min_dEta_jets_0p9_no_tags(9999.);
    float min_dR_jets_0p9_no_tags(9999.);
    float min_dPhi_jets_0p996(9999.);
    float min_dEta_jets_0p996(9999.);
    float min_dR_jets_0p996(9999.);

    float min_dPhi_jets_eta_1p0(9999.);
    float min_dEta_jets_eta_1p0(9999.);
    float min_dR_jets_eta_1p0(9999.);
    float min_dPhi_jets_eta_1p0_0p7(9999.);
    float min_dEta_jets_eta_1p0_0p7(9999.);
    float min_dR_jets_eta_1p0_0p7(9999.);
    float min_dPhi_jets_eta_1p0_0p9(9999.);
    float min_dEta_jets_eta_1p0_0p9(9999.);
    float min_dR_jets_eta_1p0_0p9(9999.);
    float min_dPhi_jets_eta_1p0_0p9_no_tags(9999.);
    float min_dEta_jets_eta_1p0_0p9_no_tags(9999.);
    float min_dR_jets_eta_1p0_0p9_no_tags(9999.);
    float min_dPhi_jets_eta_1p0_0p996(9999.);
    float min_dEta_jets_eta_1p0_0p996(9999.);
    float min_dR_jets_eta_1p0_0p996(9999.);

    float eta_spread_tagged_EB(-9999.);
    float phi_spread_tagged_EB(-9999.);
    float x_spread_tagged_EB(-9999.);
    float y_spread_tagged_EB(-9999.);
    float z_spread_tagged_EB(-9999.);

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
    float Z_pho_pt(-1.);
    float Z_pho_phi(-9.);
    float Z_pho_eta(-9.);
    bool Z_pho_loose(false);
    bool Z_pho_tight(false);

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
    int nCHSJetsNegativeAcceptanceCalo;
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
    int nTagJets_0p996_JJ_eta_1p0(0);
    int nTagJets_0p997_JJ(0);

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
    outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    outputTree->Branch("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    outputTree->Branch("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    outputTree->Branch("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isCosmic",          &isCosmic,          "isCosmic/O");
    outputTree->Branch("isDT_fit",          &isDT_fit,          "isDT_fit/O");
    outputTree->Branch("isCosmicVetoWithTags", &isCosmicVetoWithTags, "isCosmicVetoWithTags/O");
    outputTree->Branch("isZtoMM",           &isZtoMM,           "isZtoMM/O");
    outputTree->Branch("isZtoEE",           &isZtoEE,           "isZtoEE/O");

    outputTree->Branch("isZtoMMPho",           &isZtoMMPho,           "isZtoMMPho/O");
    outputTree->Branch("isZtoEEPho",           &isZtoEEPho,           "isZtoEEPho/O");

    outputTree->Branch("isPho",             &isPho,             "isPho/O");

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
    outputTree->Branch("Z_pho_pt",         &Z_pho_pt,         "Z_pho_pt/F");
    outputTree->Branch("Z_pho_phi",        &Z_pho_phi,        "Z_pho_phi/F");
    outputTree->Branch("Z_pho_eta",        &Z_pho_eta,        "Z_pho_eta/F");
    outputTree->Branch("Z_pho_loose",         &Z_pho_loose,         "Z_pho_loose/O");
    outputTree->Branch("Z_pho_tight",         &Z_pho_tight,         "Z_pho_tight/O");

    outputTree->Branch("nLeptons", &nLeptons, "nLeptons/I");
    outputTree->Branch("LepPdgId", &LepPdgId);
    outputTree->Branch("LepCharge", &LepCharge);
    outputTree->Branch("LepPt", &LepPt);
    outputTree->Branch("LepEta", &LepEta);
    outputTree->Branch("LepPhi", &LepPhi);
    outputTree->Branch("LepMass", &LepMass);

    outputTree->Branch("MinJetMetDPhi_ntuple",     &MinJetMetDPhi_ntuple,     "MinJetMetDPhi_ntuple/F");
    outputTree->Branch("MinJetMetDPhi",  &MinJetMetDPhi,  "MinJetMetDPhi/F");
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
    outputTree->Branch("nCHSJetsNegativeAcceptanceCalo",          &nCHSJetsNegativeAcceptanceCalo,          "nCHSJetsNegativeAcceptanceCalo/I");
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

    //outputTree->Branch("nDTSegments",       &nDTSegments,       "nDTSegments/I");
    //outputTree->Branch("nCSCSegments",      &nCSCSegments,      "nCSCSegments/I");
    //outputTree->Branch("nCosmicMuons",      &nCosmicMuons,      "nCosmicMuons/I");
    //outputTree->Branch("nCosmicMuonsOneLeg",      &nCosmicMuonsOneLeg,      "nCosmicMuonsOneLeg/I");

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
    outputTree->Branch("min_dR_jets", &min_dR_jets, "min_dR_jets/F");
    outputTree->Branch("min_dPhi_jets", &min_dPhi_jets, "min_dPhi_jets/F");
    outputTree->Branch("min_dEta_jets", &min_dEta_jets, "min_dEta_jets/F");
    outputTree->Branch("min_dR_jets_0p7", &min_dR_jets_0p7, "min_dR_jets_0p7/F");
    outputTree->Branch("min_dPhi_jets_0p7", &min_dPhi_jets_0p7, "min_dPhi_jets_0p7/F");
    outputTree->Branch("min_dEta_jets_0p7", &min_dEta_jets_0p7, "min_dEta_jets_0p7/F");
    outputTree->Branch("min_dR_jets_0p9", &min_dR_jets_0p9, "min_dR_jets_0p9/F");
    outputTree->Branch("min_dPhi_jets_0p9", &min_dPhi_jets_0p9, "min_dPhi_jets_0p9/F");
    outputTree->Branch("min_dEta_jets_0p9", &min_dEta_jets_0p9, "min_dEta_jets_0p9/F");
    outputTree->Branch("min_dR_jets_0p9_no_tags", &min_dR_jets_0p9_no_tags, "min_dR_jets_0p9_no_tags/F");
    outputTree->Branch("min_dPhi_jets_0p9_no_tags", &min_dPhi_jets_0p9_no_tags, "min_dPhi_jets_0p9_no_tags/F");
    outputTree->Branch("min_dEta_jets_0p9_no_tags", &min_dEta_jets_0p9_no_tags, "min_dEta_jets_0p9_no_tags/F");
    outputTree->Branch("min_dR_jets_0p996", &min_dR_jets_0p996, "min_dR_jets_0p996/F");
    outputTree->Branch("min_dPhi_jets_0p996", &min_dPhi_jets_0p996, "min_dPhi_jets_0p996/F");
    outputTree->Branch("min_dEta_jets_0p996", &min_dEta_jets_0p996, "min_dEta_jets_0p996/F");

    outputTree->Branch("min_dR_jets_eta_1p0", &min_dR_jets_eta_1p0, "min_dR_jets_eta_1p0/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0", &min_dPhi_jets_eta_1p0, "min_dPhi_jets_eta_1p0/F");
    outputTree->Branch("min_dEta_jets_eta_1p0", &min_dEta_jets_eta_1p0, "min_dEta_jets_eta_1p0/F");
    outputTree->Branch("min_dR_jets_eta_1p0_0p7", &min_dR_jets_eta_1p0_0p7, "min_dR_jets_eta_1p0_0p7/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p7", &min_dPhi_jets_eta_1p0_0p7, "min_dPhi_jets_eta_1p0_0p7/F");
    outputTree->Branch("min_dEta_jets_eta_1p0_0p7", &min_dEta_jets_eta_1p0_0p7, "min_dEta_jets_eta_1p0_0p7/F");
    outputTree->Branch("min_dR_jets_eta_1p0_0p9", &min_dR_jets_eta_1p0_0p9, "min_dR_jets_eta_1p0_0p9/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p9", &min_dPhi_jets_eta_1p0_0p9, "min_dPhi_jets_eta_1p0_0p9/F");
    outputTree->Branch("min_dEta_jets_eta_1p0_0p9", &min_dEta_jets_eta_1p0_0p9, "min_dEta_jets_eta_1p0_0p9/F");
    outputTree->Branch("min_dR_jets_eta_1p0_0p9_no_tags", &min_dR_jets_eta_1p0_0p9_no_tags, "min_dR_jets_eta_1p0_0p9_no_tags/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p9_no_tags", &min_dPhi_jets_eta_1p0_0p9_no_tags, "min_dPhi_jets_eta_1p0_0p9_no_tags/F");
    outputTree->Branch("min_dEta_jets_eta_1p0_0p9_no_tags", &min_dEta_jets_eta_1p0_0p9_no_tags, "min_dEta_jets_eta_1p0_0p9_no_tags/F");
    outputTree->Branch("min_dR_jets_eta_1p0_0p996", &min_dR_jets_eta_1p0_0p996, "min_dR_jets_eta_1p0_0p996/F");
    outputTree->Branch("min_dPhi_jets_eta_1p0_0p996", &min_dPhi_jets_eta_1p0_0p996, "min_dPhi_jets_eta_1p0_0p996/F");
    outputTree->Branch("min_dEta_jets_eta_1p0_0p996", &min_dEta_jets_eta_1p0_0p996, "min_dEta_jets_eta_1p0_0p996/F");

    outputTree->Branch("eta_spread_tagged_EB", &eta_spread_tagged_EB, "eta_spread_tagged_EB/F");
    outputTree->Branch("phi_spread_tagged_EB", &phi_spread_tagged_EB, "phi_spread_tagged_EB/F");
    outputTree->Branch("x_spread_tagged_EB", &x_spread_tagged_EB, "x_spread_tagged_EB/F");
    outputTree->Branch("y_spread_tagged_EB", &y_spread_tagged_EB, "y_spread_tagged_EB/F");
    outputTree->Branch("z_spread_tagged_EB", &z_spread_tagged_EB, "z_spread_tagged_EB/F");

    outputTree->Branch("nPFCandidates",     &nPFCandidates,     "nPFCandidates/I");
    outputTree->Branch("nPFCandidatesTrack", &nPFCandidatesTrack, "nPFCandidatesTrack/I");
    outputTree->Branch("nLLPInCalo", &nLLPInCalo, "nLLPInCalo/I");
    //outputTree->Branch("m_chi", &m_chi, "m_chi/I");
    //outputTree->Branch("ctau", &ctau, "ctau/I");
    //outputTree->Branch("is_central", &is_central, "is_central/O");
    outputTree->Branch("Muons", &Muons);
    outputTree->Branch("Electrons", &Electrons);
    outputTree->Branch("Photons", &Photons);
    outputTree->Branch("Taus", &skimmedTaus);
    outputTree->Branch("Jets", &skimmedJets);
    outputTree->Branch("JetsNegative", &skimmedJetsNegative);
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
    outputTree->Branch("nTagJets_0p996_JJ_eta_1p0",     &nTagJets_0p996_JJ_eta_1p0,      "nTagJets_0p996_JJ_eta_1p0/I");
    outputTree->Branch("nTagJets_0p997_JJ",     &nTagJets_0p997_JJ,      "nTagJets_0p997_JJ/I");


    outputTree->Branch("isTagAK4_0p99", &isTagAK4_0p99, "isTagAK4_0p99/O");
    outputTree->Branch("isTagAK4_0p994", &isTagAK4_0p994, "isTagAK4_0p994/O");
    outputTree->Branch("isTagAK4_0p996", &isTagAK4_0p996, "isTagAK4_0p996/O");
    outputTree->Branch("isTagAK4_0p997", &isTagAK4_0p997, "isTagAK4_0p997/O");
    outputTree->Branch("isTagAK4_0p99_JJ", &isTagAK4_0p99_JJ, "isTagAK4_0p99_JJ/O");
    outputTree->Branch("isTagAK4_0p994_JJ", &isTagAK4_0p994_JJ, "isTagAK4_0p994_JJ/O");
    outputTree->Branch("isTagAK4_0p996_JJ", &isTagAK4_0p996_JJ, "isTagAK4_0p996_JJ/O");
    outputTree->Branch("isTagAK4_0p997_JJ", &isTagAK4_0p997_JJ, "isTagAK4_0p997_JJ/O");


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
    int events_passing(0);

    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        TriggerWeight = 1.;
        PUReWeight = 1.;
        PUReWeightUp = 1.;
        PUReWeightDown = 1.;
	//Initialize nTagJets at every event
        nCHSJetsAcceptanceCalo = 0;
        nCHSJetsNegativeAcceptanceCalo = 0;
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

	min_dR_jets = 9999.;
	min_dPhi_jets = 9999.;
	min_dEta_jets = 9999.;
	min_dR_jets_0p7 = 9999.;
	min_dPhi_jets_0p7 = 9999.;
	min_dEta_jets_0p7 = 9999.;
	min_dR_jets_0p9 = 9999.;
	min_dPhi_jets_0p9 = 9999.;
	min_dEta_jets_0p9 = 9999.;
	min_dR_jets_0p9_no_tags = 9999.;
	min_dPhi_jets_0p9_no_tags = 9999.;
	min_dEta_jets_0p9_no_tags = 9999.;
	min_dR_jets_0p996 = 9999.;
	min_dPhi_jets_0p996 = 9999.;
	min_dEta_jets_0p996 = 9999.;

	min_dR_jets_eta_1p0 = 9999.;
	min_dPhi_jets_eta_1p0 = 9999.;
	min_dEta_jets_eta_1p0 = 9999.;
	min_dR_jets_eta_1p0_0p7 = 9999.;
	min_dPhi_jets_eta_1p0_0p7 = 9999.;
	min_dEta_jets_eta_1p0_0p7 = 9999.;
	min_dR_jets_eta_1p0_0p9 = 9999.;
	min_dPhi_jets_eta_1p0_0p9 = 9999.;
	min_dEta_jets_eta_1p0_0p9 = 9999.;
	min_dR_jets_eta_1p0_0p9_no_tags = 9999.;
	min_dPhi_jets_eta_1p0_0p9_no_tags = 9999.;
	min_dEta_jets_eta_1p0_0p9_no_tags = 9999.;
	min_dR_jets_eta_1p0_0p996 = 9999.;
	min_dPhi_jets_eta_1p0_0p996 = 9999.;
	min_dEta_jets_eta_1p0_0p996 = 9999.;

	eta_spread_tagged_EB = -9999.;
	phi_spread_tagged_EB = -9999.;
	x_spread_tagged_EB = -9999.;
	y_spread_tagged_EB = -9999.;
	z_spread_tagged_EB = -9999.;


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
	nTagJets_0p996_JJ_eta_1p0 = 0;
	nTagJets_0p997_JJ = 0;
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
	skimmedTaus.clear();
        skimmedJets.clear();
        skimmedJetsNegative.clear();
	skimmedEBEnergyCSC.clear();
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


	LepPdgId.clear();
	LepCharge.clear();
	LepPt.clear();
	LepEta.clear();
	LepPhi.clear();
	LepMass.clear();

        inputTree->GetEntry(i);


	if(RunNumber>=319077)
	  {
	    RunNumber_in_HEM = true;
	  }

	if(isMC)
	  {
	    PUReWeight = pu->GetBinContent(pu->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightUp = pu_up->GetBinContent(pu_up->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightDown = pu_down->GetBinContent(pu_down->GetXaxis()->FindBin(MeanNumInteractions));
	  }

	//Trigger selections

	//MET filters always fulfilled
	//Invert Beam Halo
        //if(Flag2_globalSuperTightHalo2016Filter) continue;
  
	if(!Flag2_globalSuperTightHalo2016Filter) continue;
	if(!Flag2_EcalDeadCellTriggerPrimitiveFilter) continue;
	if(!Flag2_HBHENoiseFilter) continue;
	if(!Flag2_HBHEIsoNoiseFilter) continue;
	if(!Flag2_ecalBadCalibFilter) continue;
	if(!Flag2_eeBadScFilter) continue;
	if(!Flag2_BadPFMuonFilter) continue;

	if(doZtoMM and not(HLT_IsoMu24_v or HLT_IsoMu27_v) ) continue;
	if(doZtoEE and not(HLT_Ele32_WPTight_Gsf_v or HLT_Ele35_WPTight_Gsf_v or HLT_Ele32_eta2p1_WPLoose_Gsf_v) ) continue;

	if(doZtoMMPho and not(HLT_IsoMu24_v or HLT_IsoMu27_v) ) continue;
	if(doZtoEEPho and not(HLT_Ele32_WPTight_Gsf_v or HLT_Ele35_WPTight_Gsf_v or HLT_Ele32_eta2p1_WPLoose_Gsf_v) ) continue;

	if(doPho and not(HLT_Photon22_v or HLT_Photon30_v or HLT_Photon33_v or HLT_Photon36_v or HLT_Photon50_v or HLT_Photon75_v or HLT_Photon90_v or HLT_Photon120_v or HLT_Photon125_v or HLT_Photon150_v or HLT_Photon200_v or HLT_Photon175_v or HLT_Photon250_NoHE_v or HLT_Photon300_NoHE_v or HLT_Photon500_v or HLT_Photon600_v) ) continue;

	//Selection on MET
	if(doZtoMM and MEt->pt>=30) continue;
	if(doZtoEE and MEt->pt>=30) continue;
	if(doPho and MEt->pt>=30) continue;

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

	    //WtoMN and ZToMM CR
	    if( (doZtoMM) and (Muons->at(m).pt<30 or !Muons->at(m).isTight or Muons->at(m).pfIso04>=0.15) ) continue;
	    if( (doZtoMMPho) and (Muons->at(m).pt<30 or !Muons->at(m).isTight or Muons->at(m).pfIso04>=0.15) ) continue;
	    
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
	//if(Muons->size()>0) std::cout << "Muons size final: " << Muons->size() << std::endl;
	//nMuons = Muons->size();
	
	//ZtoMM
	if(doZtoMM and MuonsStruct.size()!=2) continue;
	if(doZtoMMPho and MuonsStruct.size()!=2) continue;


	//Electrons
	float ele_iso = 0.4;
	//if(Electrons->size()>0) std::cout << "Electrons size: " << Electrons->size() << std::endl;
	for (unsigned int e=0; e<Electrons->size(); e++)
	  {

	    if(Electrons->at(e).eta>-3. and Electrons->at(e).eta<-1.3 and Electrons->at(e).phi>-1.57 and Electrons->at(e).phi<-0.87)
	      {
		nElectrons_in_HEM++;
	      }

	    //ZtoEE CR
	    if( (doZtoEE) and (Electrons->at(e).pt<37 or !Electrons->at(e).isTight) ) continue;
	    if( (doZtoEEPho) and (Electrons->at(e).pt<37 or !Electrons->at(e).isTight) ) continue;

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

	//ZtoEE
	if(doZtoEE and ElectronsStruct.size()!=2) continue;
	if(doZtoEEPho and ElectronsStruct.size()!=2) continue;

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



	//Photons
	float pho_iso = 0.4;
	for (unsigned int p=0; p<Photons->size(); p++)
          {
	    
	    if(Photons->at(p).eta>-3. and Photons->at(p).eta<-1.3 and Photons->at(p).phi>-1.57 and Photons->at(p).phi<-0.87)
	      {
		nPhotons_in_HEM++;
	      }


	    //Improvement: store only tight photons
	    if(!Photons->at(p).isTight) continue;
	    //Improvement: only photons with eta<1
	    if(abs(Photons->at(p).eta)>=1.) continue;

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
	    tmpPhoton.isLoose = Photons->at(p).isLoose;
	    tmpPhoton.isTight = Photons->at(p).isTight;
	    PhotonsStruct.push_back(tmpPhoton);
	    if(Photons->at(p).isTight) nPhotonsTight++;
	    nPhotonsPassing++;
	  }

	//Pho CR
	if( doPho and PhotonsStruct.size() != 1 ) continue;

	//Z + gamma: one and only one photon
	//Improvement: only tight photons
	if( doZtoMMPho and PhotonsStruct.size() != 1 ) continue;
	if( doZtoEEPho and PhotonsStruct.size() != 1 ) continue;

	//Transverse mass met + Leptons (e and mu)
	TLorentzVector lepp4;
	for ( auto &tmp : LeptonsStruct )
	  {
	    lepp4 += tmp.vec;
	  }
	dPhi = reco::deltaPhi(MEt->phi, lepp4.Phi());
	MT = sqrt(2*(MEt->pt)*lepp4.Pt()*(1-cos(dPhi)));
	nLeptons = LeptonsStruct.size();

	//Z reconstruction
	TLorentzVector Z;
	if(doZtoMM)
	  {
	    if(MuonsStruct.at(0).charge==MuonsStruct.at(1).charge) continue;//we want opposite sign
	    Z = MuonsStruct.at(0).vec + MuonsStruct.at(1).vec;
	    if( fabs(Z.M() - Z_MASS)>30. ) continue;
	    Z_mass = Z.M();
	    Z_pt = Z.Pt();
	    Z_phi = Z.Phi();
	    Z_eta = Z.Eta();
	    Z_lep0_pt = MuonsStruct.at(0).vec.Pt();
	    Z_lep0_phi = MuonsStruct.at(0).vec.Phi();
	    Z_lep0_eta = MuonsStruct.at(0).vec.Eta();
	    Z_lep1_pt = MuonsStruct.at(1).vec.Pt();
	    Z_lep1_phi = MuonsStruct.at(1).vec.Phi();
	    Z_lep1_eta = MuonsStruct.at(1).vec.Eta();
	  }

	if(doZtoMMPho)
	  {
	    if(MuonsStruct.at(0).charge==MuonsStruct.at(1).charge) continue;//we want opposite sign
	    Z = MuonsStruct.at(0).vec + MuonsStruct.at(1).vec + PhotonsStruct.at(0).vec;//include photon in the Z candidate
	    if( fabs(Z.M() - Z_MASS)>10. ) continue;
	    Z_mass = Z.M();
	    Z_pt = Z.Pt();
	    Z_phi = Z.Phi();
	    Z_eta = Z.Eta();
	    Z_lep0_pt = MuonsStruct.at(0).vec.Pt();
	    Z_lep0_phi = MuonsStruct.at(0).vec.Phi();
	    Z_lep0_eta = MuonsStruct.at(0).vec.Eta();
	    Z_lep1_pt = MuonsStruct.at(1).vec.Pt();
	    Z_lep1_phi = MuonsStruct.at(1).vec.Phi();
	    Z_lep1_eta = MuonsStruct.at(1).vec.Eta();
	    Z_pho_pt  = PhotonsStruct.at(0).vec.Pt();
	    Z_pho_phi = PhotonsStruct.at(0).vec.Phi();
	    Z_pho_eta = PhotonsStruct.at(0).vec.Eta();
	    Z_pho_loose  = PhotonsStruct.at(0).isLoose;
	    Z_pho_tight  = PhotonsStruct.at(0).isTight;
	    std::cout << "Z 3 bodies: " << Z_mass << std::endl;
	    std::cout << "photon tight: " << PhotonsStruct.at(0).isTight << std::endl;
	  }

	if(doZtoEE)
	  {
	    if(ElectronsStruct.at(0).charge==ElectronsStruct.at(1).charge) continue;//we want opposite sign
	    Z = ElectronsStruct.at(0).vec + ElectronsStruct.at(1).vec;
	    if( fabs(Z.M() - Z_MASS)>30. ) continue;
	    Z_mass = Z.M();
	    Z_pt = Z.Pt();
	    Z_phi = Z.Phi();
	    Z_eta = Z.Eta();
	    Z_lep0_pt = ElectronsStruct.at(0).vec.Pt();
	    Z_lep0_phi = ElectronsStruct.at(0).vec.Phi();
	    Z_lep0_eta = ElectronsStruct.at(0).vec.Eta();
	    Z_lep1_pt = ElectronsStruct.at(1).vec.Pt();
	    Z_lep1_phi = ElectronsStruct.at(1).vec.Phi();
	    Z_lep1_eta = ElectronsStruct.at(1).vec.Eta();
	  }

	if(doZtoEEPho)
	  {
	    if(ElectronsStruct.at(0).charge==ElectronsStruct.at(1).charge) continue;//we want opposite sign
	    Z = ElectronsStruct.at(0).vec + ElectronsStruct.at(1).vec + PhotonsStruct.at(0).vec;//include photon in the Z candidate
	    if( fabs(Z.M() - Z_MASS)>10. ) continue;
	    std::cout << "--> debug: good Z mass " << std::endl;
	    Z_mass = Z.M();
	    Z_pt = Z.Pt();
	    Z_phi = Z.Phi();
	    Z_eta = Z.Eta();
	    Z_lep0_pt = ElectronsStruct.at(0).vec.Pt();
	    Z_lep0_phi = ElectronsStruct.at(0).vec.Phi();
	    Z_lep0_eta = ElectronsStruct.at(0).vec.Eta();
	    Z_lep1_pt = ElectronsStruct.at(1).vec.Pt();
	    Z_lep1_phi = ElectronsStruct.at(1).vec.Phi();
	    Z_lep1_eta = ElectronsStruct.at(1).vec.Eta();
	    Z_pho_pt  = PhotonsStruct.at(0).vec.Pt();
	    Z_pho_phi = PhotonsStruct.at(0).vec.Phi();
	    Z_pho_eta = PhotonsStruct.at(0).vec.Eta();
	    Z_pho_loose  = PhotonsStruct.at(0).isLoose;
	    Z_pho_tight  = PhotonsStruct.at(0).isTight;
	  }


        if(isVerbose) std::cout << "======================================== " << std::endl;
        if(isVerbose) std::cout << "EventNumber " << EventNumber << "\tLumiNumber " << LumiNumber << std::endl;


	//Apply acceptance cuts to jets and fat jets 
	std::vector<int> validJetIndex;

	for (unsigned int j=0; j<Jets->size(); j++)
	  {

	    //HEM: reject events with jets in problematic region
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


	    //Correct acceptance for MinJetMetDPhi:
	    //Jet pt>30, Jet eta<2.4
	    if(fabs(Jets->at(j).eta)<2.4 and Jets->at(j).pt>30)
	      {
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhi) MinJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		TLorentzVector jet0;
		jet0.SetPtEtaPhiM(Jets->at(j).pt, 0, Jets->at(j).phi, 0 );
		TLorentzVector met;
		met.SetPtEtaPhiM(MEt->pt, 0, MEt->phi, 0 );
		if(fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi())) < MinJetMetDPhiStar) MinJetMetDPhiStar = fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi() ));
	      }
	    

	    
	    //  Additional pre-selections
	    //"muEFrac" : {"min" : -1., "max" : 0.6},
	    //	"eleEFrac" : {"min" : -1., "max" : 0.6},
	    //	"photonEFrac" : {"min" : -1., "max" : 0.8},

	    //if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1)

	    //I want to save also jets with negative time<-1 to check beam halo
	    //if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1)//cleaned jets!

	    //Improvement: consider only jets with eta<1
	    if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1. and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6)//and Jets->at(j).photonEFrac<0.8)//cleaned jets!
	      {

		//This limits to jets with positive times. should redo it including negative time... otherwise biase
		//if(Jets->at(j).timeRecHitsEB>-1)
		//{

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
		//Do NOT perform photon cleaning!!! Instead, we should ask overlap!!!
		//REVERT pho cleaning!!!
		float dR_pho = -1;
		for(auto& pho : PhotonsStruct)
		  {
		    //this object will have size one, hence only one loop performed!
		    float thisDR_pho = reco::deltaR(Jets->at(j).eta,Jets->at(j).phi,pho.vec.Eta(),pho.vec.Phi());
		    if(dR_pho < 0 || thisDR_pho < dR_pho) dR_pho = thisDR_pho;
		  }
		if(doZtoEEPho or doZtoMMPho)
		  {
		    //If the photon is not found or the photon does not overlap a jet, skip
		    //This procedure should basically save only jets overlapping the only photon in the event
		    //Not sure if it's correct...
		    if(dR_pho<0 or dR_pho > jet_iso) continue;
		    std::cout << "--> debug: photon overlaps jets " << std::endl;
		    std::cout << "Jet eta/phi: " << Jets->at(j).eta << " " << Jets->at(j).phi << std::endl; 
		    std::cout << "Photon eta/phi: " << PhotonsStruct.at(0).vec.Eta() << " " << PhotonsStruct.at(0).vec.Phi() << std::endl; 
		  }
		else
		  {
		    if(dR_pho > 0 && dR_pho < jet_iso) continue;
		  }


		//JetMET CR: MinLeadingJetMetDPhi bw leading jet and met should be large (back to back)
		if(MinLeadingJetMetDPhi<0 and Jets->at(j).timeRecHitsEB>-1)
		  {
		    MinLeadingJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		    if(isVerbose) std::cout << "MET: " << MEt->pt << " ; MinLeadingJetMetDPhi " << MinLeadingJetMetDPhi << std::endl;
		    if(isVerbose) std::cout << "MinLeadingJetMetDPhi calculated with jet " << j << " ; pt: " << Jets->at(j).pt << std::endl;
		  }

		//JetMET CR: MinLeadingJetMetDPhi bw leading jet and met should be large (back to back)
		if(nCHSJetsAcceptanceCalo==2 && MinSubLeadingJetMetDPhi<0 and Jets->at(j).timeRecHitsEB>-1)
		  {
		    MinSubLeadingJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		    if(isVerbose) std::cout << "MET: " << MEt->pt << " ; MinSubLeadingJetMetDPhi " << MinSubLeadingJetMetDPhi << std::endl;
		    if(isVerbose) std::cout << "MinSubLeadingJetMetDPhi calculated with jet " << j << " ; pt: " << Jets->at(j).pt << std::endl;
		  }

		if(nCHSJetsAcceptanceCalo==3 && MinSubSubLeadingJetMetDPhi<0 and Jets->at(j).timeRecHitsEB>-1)
		  {
		    MinSubSubLeadingJetMetDPhi = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		  }
		    
		if(fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi)) < MinJetMetDPhiBarrel and Jets->at(j).timeRecHitsEB>-1) MinJetMetDPhiBarrel = fabs(reco::deltaPhi(Jets->at(j).phi, MEt->phi));
		TLorentzVector jet0;
		jet0.SetPtEtaPhiM(Jets->at(j).pt, 0, Jets->at(j).phi, 0 );
		TLorentzVector met;
		met.SetPtEtaPhiM(MEt->pt, 0, MEt->phi, 0 );
		if(fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi())) < MinJetMetDPhiBarrelStar and Jets->at(j).timeRecHitsEB>-1) MinJetMetDPhiBarrelStar = fabs(reco::deltaPhi(Jets->at(j).phi, (jet0+met).Phi() ));
		//}


		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		//std::cout<< "Jet n. " << j << " eFracRecHitsEB: " << Jets->at(j).eFracRecHitsEB  << std::endl;
		Jets->at(j).eFracRecHitsEB = (Jets->at(j).energy>0 and Jets->at(j).energyRecHitsEB>0) ? Jets->at(j).energyRecHitsEB/Jets->at(j).energy : -1.;
		
		//Fix also timeRMS dividing by sqrt nRecHitsEB
		Jets->at(j).timeRMSRecHitsEB = (Jets->at(j).nRecHitsEB>0) ? Jets->at(j).timeRMSRecHitsEB/sqrt(Jets->at(j).nRecHitsEB) : -1.;

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

		//
		// Cut based- definition:
		//"timeRecHitsEB" : {"min" : 0.09, "max" : 999.e+10},
		//"gammaMaxET" : {"min" : -100.-10., "max" : 0.16},
		//"minDeltaRPVTracks" : {"min" : 0.06, "max" : 999.+10.},
		//"cHadEFrac" : {"min" : -1., "max" : 0.06},
		//
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
		if(outputValueAK4>0.996 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1 and abs(Jets->at(j).eta)<1.) nTagJets_0p996_JJ_eta_1p0++;
		if(outputValueAK4>0.997 and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8 and Jets->at(j).timeRecHitsEB>-1) nTagJets_0p997_JJ++;



		if(Jets->at(j).timeRecHitsEB>-1.)
		  {
		    //store jets passing acceptance and with inference
		    skimmedJets.push_back(Jets->at(j));
		    validJetIndex.push_back(j);
		  }

		//save also jets including negative times
		skimmedJetsNegative.push_back(Jets->at(j));

	      }//acceptance

	  }//jet loop

	//Here: passed acceptance
	nCHSJetsAcceptanceCalo = skimmedJets.size();
	nCHSJetsNegativeAcceptanceCalo = skimmedJetsNegative.size();
	//Improvement
	if(doZtoMMPho and (skimmedJets.size()!=1 and skimmedJetsNegative.size()!=1)) continue;
	if(doZtoEEPho and (skimmedJets.size()!=1 and skimmedJetsNegative.size()!=1)) continue;
	std::cout << "skimmedJetsNegative.size() " << skimmedJetsNegative.size() << std::endl;

	if(isVerbose) std::cout << "n. tagged jets " << nTagJets_0p996_JJ << std::endl;
        if(isVerbose) std::cout << "======================================== " << std::endl;

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
	//if(nCHSJetsAcceptanceCalo==0 and nCHSFatJetsAcceptanceCalo==0) continue;
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
	    std::vector<float> EBx_j;
	    std::vector<float> EBy_j;
	    std::vector<float> EBz_j;
	    std::vector<float> EBr_j;
	    std::vector<float> EBeta_j;
	    std::vector<float> EBphi_j;
	    std::vector<float> EB_Dphi_j;
	    std::vector<float> EBenergy_j;
	    float csc_energy(0.);
	    float csc_energy_0p1(0.);
	    float csc_energy_0p04(0.);
	    //float check_ecal_energy(0.);
	    for(unsigned int p=0; p<EcalRecHitsAK4->size(); p++)
	      {
		//j corresponds to the skimmed jet, validJetIndex.at(j) corresponds to the original jets
		//for each valid jet skimmedJets[j] I want the Rec hits features
		//Beam Halo rejection variables
		//min_dR_jets
		//Calculate sparsity of associated ecal rec hits

		if(int(EcalRecHitsAK4->at(p).jetIndex) == int(validJetIndex.at(j)) )//only this is complaining...
		  {
		    //0.4 matching
		    if (reco::deltaR(Jets->at( int(validJetIndex.at(j)) ).eta, Jets->at( int(validJetIndex.at(j)) ).phi, EcalRecHitsAK4->at(p).eta, EcalRecHitsAK4->at(p).phi) < 0.4)
		      {

			//std::cout << "~~~~~~~" << endl;
			//std::cout<<"Jet n. : " << j << " has nRecHits: " << Jets->at( int(validJetIndex.at(j)) ).nRecHitsEB << endl;
			//std::cout<<"ECAL hit n. : " << p << endl;

			skimmedEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));
			if(abs(Jets->at( int(validJetIndex.at(j)) ).eta)<1) skimmedAcceptanceEcalRecHitsAK4.push_back(EcalRecHitsAK4->at(p));
			EBx_j.push_back(EcalRecHitsAK4->at(p).x);
			EBy_j.push_back(EcalRecHitsAK4->at(p).y);
			EBz_j.push_back(EcalRecHitsAK4->at(p).z);
			EBr_j.push_back(sqrt( pow(EcalRecHitsAK4->at(p).x,2) + pow(EcalRecHitsAK4->at(p).y,2)));
			EBeta_j.push_back(EcalRecHitsAK4->at(p).eta);
			EBphi_j.push_back(EcalRecHitsAK4->at(p).phi);
			EB_Dphi_j.push_back( abs(reco::deltaPhi(EcalRecHitsAK4->at(p).phi,Jets->at( int(validJetIndex.at(j)) ).phi)) );
			EBenergy_j.push_back(EcalRecHitsAK4->at(p).energy);

			//check_ecal_energy += EcalRecHitsAK4->at(p).energy;

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
	for (unsigned int j=0; j<skimmedJets.size(); j++)
	  {
	    //second loop on jets to calculate delta phi/R
	    for (unsigned int k=j+1; k<skimmedJets.size() && k!=j; k++)
	      {
		//std::cout << "Doing general pair: (" << j <<" , "<<k<<")"<<std::endl;

		min_dPhi_jets = std::min(fabs(min_dPhi_jets),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
		min_dEta_jets = std::min(fabs(min_dEta_jets),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
		min_dR_jets = std::min(min_dR_jets,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));

		if( abs(skimmedJets.at(j).eta)<1.0 and abs(skimmedJets.at(k).eta)<1.0)
		  {
		    min_dPhi_jets_eta_1p0 = std::min(fabs(min_dPhi_jets_eta_1p0),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
		    min_dEta_jets_eta_1p0 = std::min(fabs(min_dEta_jets_eta_1p0),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
		    min_dR_jets_eta_1p0 = std::min(min_dR_jets_eta_1p0,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));
		  }

		//0p7
		if(skimmedJets.at(j).sigprob>0.7 and skimmedJets.at(k).sigprob>0.7)
		  {
		    min_dPhi_jets_0p7 = std::min(fabs(min_dPhi_jets_0p7),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
		    min_dEta_jets_0p7 = std::min(fabs(min_dEta_jets_0p7),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
		    min_dR_jets_0p7 = std::min(min_dR_jets_0p7,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));

		    if( abs(skimmedJets.at(j).eta)<1.0 and abs(skimmedJets.at(k).eta)<1.0)
		      {
			min_dPhi_jets_eta_1p0_0p7 = std::min(fabs(min_dPhi_jets_eta_1p0_0p7),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
			min_dEta_jets_eta_1p0_0p7 = std::min(fabs(min_dEta_jets_eta_1p0_0p7),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
			min_dR_jets_eta_1p0_0p7 = std::min(min_dR_jets_eta_1p0_0p7,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));
		      }
		  }

		//0p9
		if(skimmedJets.at(j).sigprob>0.9 and skimmedJets.at(k).sigprob>0.9)
		  {
		    min_dPhi_jets_0p9 = std::min(fabs(min_dPhi_jets_0p9),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
		    min_dEta_jets_0p9 = std::min(fabs(min_dEta_jets_0p9),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
		    min_dR_jets_0p9 = std::min(min_dR_jets_0p9,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));
		    //And both not tagged
		    if(skimmedJets.at(j).sigprob<=0.996 and skimmedJets.at(k).sigprob<=0.996)
		      {
			min_dPhi_jets_0p9_no_tags = std::min(fabs(min_dPhi_jets_0p9_no_tags),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
			min_dEta_jets_0p9_no_tags = std::min(fabs(min_dEta_jets_0p9_no_tags),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
			min_dR_jets_0p9_no_tags = std::min(min_dR_jets_0p9_no_tags,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));

		      }

		    if( abs(skimmedJets.at(j).eta)<1.0 and abs(skimmedJets.at(k).eta)<1.0)
		      {
			min_dPhi_jets_eta_1p0_0p9 = std::min(fabs(min_dPhi_jets_eta_1p0_0p9),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
			min_dEta_jets_eta_1p0_0p9 = std::min(fabs(min_dEta_jets_eta_1p0_0p9),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
			min_dR_jets_eta_1p0_0p9 = std::min(min_dR_jets_eta_1p0_0p9,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));

			//And both not tagged
			if(skimmedJets.at(j).sigprob<=0.996 and skimmedJets.at(k).sigprob<=0.996)
			  {
			    min_dPhi_jets_eta_1p0_0p9_no_tags = std::min(fabs(min_dPhi_jets_eta_1p0_0p9_no_tags),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
			    min_dEta_jets_eta_1p0_0p9_no_tags = std::min(fabs(min_dEta_jets_eta_1p0_0p9_no_tags),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
			    min_dR_jets_eta_1p0_0p9_no_tags = std::min(min_dR_jets_eta_1p0_0p9_no_tags,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));

			  }
		      }
		  }
		//0p996
		if(skimmedJets.at(j).sigprob>0.996 and skimmedJets.at(k).sigprob>0.996)
		  {
		    //std::cout << "Doing min_dPhi_jets_0p996 with jet pair: (" << j <<" , "<<k<<")"<<std::endl;
		    //std::cout << "prev min_dPhi_jets_0p996 " << min_dPhi_jets_0p996 << std::endl;
		    //std::cout << "their distance: "<< fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi))  << std::endl;
		    min_dPhi_jets_0p996 = std::min(fabs(min_dPhi_jets_0p996),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
		    min_dEta_jets_0p996 = std::min(fabs(min_dEta_jets_0p996),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
		    min_dR_jets_0p996 = std::min(min_dR_jets_0p996,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));
		    //std::cout << "post min_dPhi_jets_0p996 " << min_dPhi_jets_0p996 << std::endl;

		    if( abs(skimmedJets.at(j).eta)<1.0 and abs(skimmedJets.at(k).eta)<1.0)
		      {
			min_dPhi_jets_eta_1p0_0p996 = std::min(fabs(min_dPhi_jets_eta_1p0_0p996),fabs(reco::deltaPhi(skimmedJets.at(j).phi,skimmedJets.at(k).phi)));
			min_dEta_jets_eta_1p0_0p996 = std::min(fabs(min_dEta_jets_eta_1p0_0p996),fabs(skimmedJets.at(j).eta - skimmedJets.at(k).eta));
			min_dR_jets_eta_1p0_0p996 = std::min(min_dR_jets_eta_1p0_0p996,reco::deltaR(skimmedJets.at(j).eta,skimmedJets.at(j).phi,skimmedJets.at(k).eta,skimmedJets.at(k).phi));
		      }
		  }

	      }

	  }
	//////
	//Veto objects

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

	//Here select bin 1/2 LISA
	//if(doSR and nTagJets_0p996_JJ<1) continue;
	//At this point, doSR and doZtoMM should be all fulfilled, cross check
	if(doZtoMM) isZtoMM = true;
	if(doZtoEE) isZtoEE = true;
	if(doPho) isPho = true;
	if(doZtoMMPho) isZtoMMPho = true;
	if(doZtoEEPho) isZtoEEPho = true;

	//Observed worse agreement, skip this --> redo
	n_pass->Fill(0.);
	if(EventNumber % 2 == 0) n_even->Fill(0.);
	if(EventNumber % 2 != 0) n_odd->Fill(0.);
	if(skipTrain==true and EventNumber % 2 == 0) continue;
	outputTree->Fill();
	events_passing++;
    }


    // finalize files
    if(events_passing>0)
      {
        TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
        outputFile->cd();
        outputTree->SetDirectory(outputFile);
        outputTree->SetWeight(tree_weight);
        counter->Write();
        n_pass->Write();
        n_odd->Write();
        n_even->Write();
        b_skipTrain->Write();
        outputTree->Write();
        outputTree->Delete();
        outputFile->Write();
        outputFile->Close();
      }
    else
      {
        outputTree->Delete();
      }

    mcPUFile->Close();
    mcTriggerFile->Close();
    inputFile->Close();
    

    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    if(events_passing>0)
      {
	std::cout << "**************************************************" << std::endl;
	std::cout << "Output written: " << outputPath << std::endl;
	std::cout << "\n" << std::endl;
      }
    else
      {
	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "No output " << std::endl;
	std::cout << "\n" << std::endl;
      }

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


