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
    bool doMR(false);
    if(strcmp(argv[9], "doMR")==0) doMR=true;
    bool doMRPho(false);
    if(strcmp(argv[9], "doMRPho")==0) doMRPho=true;
    bool doZtoMM(false);
    if(strcmp(argv[9], "doZtoMM")==0) doZtoMM=true;
    bool doZtoEE(false);
    if(strcmp(argv[9], "doZtoEE")==0) doZtoEE=true;
    bool doTtoEM(false);
    if(strcmp(argv[9], "doTtoEM")==0) doTtoEM=true;
    bool doWtoEN(false);
    if(strcmp(argv[9], "doWtoEN")==0) doWtoEN=true;
    bool doWtoMN(false);
    if(strcmp(argv[9], "doWtoMN")==0) doWtoMN=true;
    bool doEN(false);
    if(strcmp(argv[9], "doEN")==0) doEN=true;
    bool doMN(false);
    if(strcmp(argv[9], "doMN")==0) doMN=true;
    bool doPho(false);
    if(strcmp(argv[9], "doPho")==0) doPho=true;
    bool doJetHT(false);
    if(strcmp(argv[9], "doJetHT")==0) doJetHT=true;
    bool doJetMET(false);
    if(strcmp(argv[9], "doJetMET")==0) doJetMET=true;
    bool doDiJetMET(false);
    if(strcmp(argv[9], "doDiJetMET")==0) doDiJetMET=true;

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
    if(doMR) std::cout << "MR selections" << std::endl;
    if(doMRPho) std::cout << "MR + 1 photon selections" << std::endl;
    if(doZtoMM) std::cout << "ZtoMM selections" << std::endl;
    if(doZtoEE) std::cout << "ZtoEE selections" << std::endl;


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

    bool doPFCand=false;

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


    //PU reweighting
    //TFile *mcPUFile = TFile::Open(mcPUFilename.data(),"READ"); if (!mcPUFile) return 0;
    //TFile *dataFile = TFile::Open(dataFilename.data(),"READ"); if (!dataFile) return 0;
    //TFile *dataFileUp = TFile::Open(dataFilenameUp.data(),"READ"); if (!dataFileUp) return 0;
    //TFile *dataFileDown = TFile::Open(dataFilenameDown.data(),"READ"); if (!dataFileDown) return 0;

    //TH1F  *pileup_mc = (TH1F*)mcPUFile->Get("pileup");
    //TH1F  *pileup_mc_copy = (TH1F*)pileup_mc->Clone("pileup_mc");
    //pileup_mc_copy->SetLineColor(8);
    //pileup_mc_copy->SetLineWidth(2);
    //pileup_mc_copy->SetLineStyle(2);

    //TH1F  *pileup_data = (TH1F*)dataFile->Get("pileup");
    //TH1F  *pileup_data_copy = (TH1F*)pileup_data->Clone("pileup_data");
    //pileup_data_copy->SetLineColor(1);
    //pileup_data_copy->SetLineWidth(2);
    //pileup_data_copy->SetLineStyle(1);

    //TH1F  *pileup_data_up = (TH1F*)dataFileUp->Get("pileup");
    //TH1F  *pileup_data_up_copy = (TH1F*)pileup_data_up->Clone("pileup_data_up");
    //pileup_data_up_copy->SetLineColor(2);
    //pileup_data_up_copy->SetLineWidth(2);
    //pileup_data_up_copy->SetLineStyle(2);

    //TH1F  *pileup_data_down = (TH1F*)dataFileDown->Get("pileup");
    //TH1F  *pileup_data_down_copy = (TH1F*)pileup_data_down->Clone("pileup_data_down");
    //pileup_data_down_copy->SetLineColor(4);
    //pileup_data_down_copy->SetLineWidth(2);
    //pileup_data_down_copy->SetLineStyle(2);
    //Hist normalization
    //NormalizeHist(pileup_mc_copy);
    //NormalizeHist(pileup_data_copy);
    //NormalizeHist(pileup_data_up_copy);
    //NormalizeHist(pileup_data_down_copy);

    //Hist normalization
    //NormalizeHist(pileup_mc);
    //NormalizeHist(pileup_data);
    //NormalizeHist(pileup_data_up);
    //NormalizeHist(pileup_data_down);

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
    std::vector<FatJetType>      *FatJets = 0;
    std::vector<PFCandidateType> *PFCandidatesAK4 = 0;
    std::vector<PFCandidateType> *PFCandidatesAK8 = 0;
    std::vector<ecalRecHitType>  *EcalRecHitsAK4 = 0;
    std::vector<ecalRecHitType>  *EcalRecHitsAK8 = 0;
    //std::vector<hcalRecHitType>  *HcalRecHitsAK8 = 0;
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
    TBranch        *b_FatJets = 0;
    TBranch        *b_PFCandidatesAK4 = 0;
    TBranch        *b_PFCandidatesAK8 = 0;
    TBranch        *b_MEt = 0;
    TBranch        *b_GenHiggs = 0;
    TBranch        *b_GenLLPs = 0;
    TBranch        *b_GenBquarks = 0;
    TBranch        *b_GenGravitinos = 0;
    TBranch        *b_EcalRecHitsAK4 = 0;
    TBranch        *b_EcalRecHitsAK8 = 0;
    TBranch        *b_DTSegments = 0;
    TBranch        *b_CSCSegments = 0;
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
    inputTree->SetBranchAddress("FatJets",           &FatJets,           &b_FatJets);
    inputTree->SetBranchAddress("PFCandidatesAK4",   &PFCandidatesAK4,   &b_PFCandidatesAK4);
    inputTree->SetBranchAddress("PFCandidatesAK8",   &PFCandidatesAK8,   &b_PFCandidatesAK8);
    inputTree->SetBranchAddress("EcalRecHitsAK4",    &EcalRecHitsAK4,    &b_EcalRecHitsAK4);
    inputTree->SetBranchAddress("EcalRecHitsAK8",    &EcalRecHitsAK8,    &b_EcalRecHitsAK8);
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

    //AK8
    std::ifstream finAK8;
    std::string featAK8;
    finAK8.open(MetaDataFileAK8);
    std::vector<std::string> featuresAK8;
    std::string toEraseAK8 = "FatJet_";
    //std::cout << "   -- > Features AK8: " << std::endl;
    while (finAK8 >> featAK8)
      {
	size_t pos = featAK8.find(toEraseAK8);
	if (pos != std::string::npos)
	  {
	    // If found then erase it from string
	    featAK8.erase(pos, toEraseAK8.length());
	  }
	//std::string new_feat = featAK8.substr(position);
	//std::cout << featAK8 << std::endl;
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


    //Flags for SR/CR
    bool isSR(false);
    bool isMR(false);
    bool isMRPho(false);
    bool isZtoMM(false);
    bool isZtoEE(false);
    bool isTtoEM(false);
    bool isWtoEN(false);
    bool isWtoMN(false);
    bool isEN(false);
    bool isMN(false);
    bool isPho(false);
    bool isJetHT(false);
    bool isJetMET(false);
    bool isDiJetMET(false);

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

    std::vector<PFCandidateType> Jet_0_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_1_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_2_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_3_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_4_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_5_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_6_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_7_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_8_PFCandidatesAK4;
    std::vector<PFCandidateType> Jet_9_PFCandidatesAK4;
    std::vector<PFCandidateType> FatJet_0_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_1_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_2_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_3_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_4_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_5_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_6_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_7_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_8_PFCandidatesAK8;
    std::vector<PFCandidateType> FatJet_9_PFCandidatesAK8;
    //std::vector<ecalRecHitType>  Jet_0_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_1_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_2_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_3_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_4_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_5_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_6_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_7_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_8_EcalRecHitsAK4;
    //std::vector<ecalRecHitType>  Jet_9_EcalRecHitsAK4;
    std::vector<ecalRecHitType>  FatJet_0_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_1_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_2_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_3_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_4_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_5_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_6_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_7_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_8_EcalRecHitsAK8;
    std::vector<ecalRecHitType>  FatJet_9_EcalRecHitsAK8;

    std::vector<TauType>    skimmedTaus;
    std::vector<JetType>    skimmedJets;
    std::vector<JetType>    skimmedJetsNegative;
    std::vector<JetCaloType> skimmedJetsCalo;
    std::vector<FatJetType> skimmedFatJets;
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
    float MinFatJetMetDPhi(10.);
    float MinFatJetMetDPhiBarrel(10.);
    float MinFatJetMetDPhiBarrelMatched(10.);
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

    //Gen level studies
    float dR_LLPs(-9.);
    float dR_Higgs(-9.);
    float dR_Gravitinos(-9.);
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
    int nCHSFatJetsAcceptanceCalo;
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
    outputTree->Branch("PUWeight",          &PUWeight,          "PUWeight/F");
    outputTree->Branch("PUReWeight",        &PUReWeight,        "PUReWeight/F");
    outputTree->Branch("PUReWeightUp",      &PUReWeightUp,      "PUReWeightUp/F");
    outputTree->Branch("PUReWeightDown",    &PUReWeightDown,    "PUReWeightDown/F");
    outputTree->Branch("TriggerWeight",     &TriggerWeight,     "TriggerWeight/F");
    outputTree->Branch("isMC",              &isMC,              "isMC/O");
    outputTree->Branch("isCosmic",          &isCosmic,          "isCosmic/O");
    outputTree->Branch("isDT_fit",          &isDT_fit,          "isDT_fit/O");
    outputTree->Branch("isCosmicVetoWithTags", &isCosmicVetoWithTags, "isCosmicVetoWithTags/O");
    outputTree->Branch("isSR",              &isSR,              "isSR/O");
    outputTree->Branch("isMR",              &isMR,              "isMR/O");
    outputTree->Branch("isMRPho",           &isMRPho,           "isMRPho/O");
    outputTree->Branch("isZtoMM",           &isZtoMM,           "isZtoMM/O");
    outputTree->Branch("isZtoEE",           &isZtoEE,           "isZtoEE/O");
    outputTree->Branch("isWtoMN",           &isWtoMN,           "isWtoMN/O");
    outputTree->Branch("isWtoEN",           &isWtoEN,           "isWtoEN/O");
    outputTree->Branch("isMN",           &isMN,           "isMN/O");
    outputTree->Branch("isEN",           &isEN,           "isEN/O");
    outputTree->Branch("isTtoEM",           &isTtoEM,           "isTtoEM/O");
    outputTree->Branch("isPho",             &isPho,             "isPho/O");
    outputTree->Branch("isJetHT",           &isJetHT,           "isJetHT/O");
    outputTree->Branch("isJetMET",           &isJetMET,           "isJetMET/O");
    outputTree->Branch("isDiJetMET",           &isDiJetMET,           "isDiJetMET/O");
    outputTree->Branch("isVBF",             &isVBF,             "isVBF/O");
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

    outputTree->Branch("dR_LLPs", &dR_LLPs, "dR_LLPs/F");
    outputTree->Branch("dR_Higgs", &dR_Higgs, "dR_Higgs/F");
    outputTree->Branch("dR_Gravitinos", &dR_Gravitinos, "dR_Gravitinos/F");
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
    outputTree->Branch("MinFatJetMetDPhi",  &MinFatJetMetDPhi,  "MinFatJetMetDPhi/F");
    outputTree->Branch("MinFatJetMetDPhiBarrel",  &MinFatJetMetDPhiBarrel,  "MinFatJetMetDPhiBarrel/F");
    outputTree->Branch("MinFatJetMetDPhiBarrelMatched",  &MinFatJetMetDPhiBarrelMatched,  "MinFatJetMetDPhiBarrelMatched/F");
    outputTree->Branch("MinLeadingJetMetDPhi", &MinLeadingJetMetDPhi, "MinLeadingJetMetDPhi/F");
    outputTree->Branch("MinSubLeadingJetMetDPhi", &MinSubLeadingJetMetDPhi, "MinSubLeadingJetMetDPhi/F");
    outputTree->Branch("MinSubSubLeadingJetMetDPhi", &MinSubSubLeadingJetMetDPhi, "MinSubSubLeadingJetMetDPhi/F");
    outputTree->Branch("nPV",          &nPV,          "nPV/I");
    outputTree->Branch("nCHSJets",          &nCHSJets,          "nCHSJets/I");
    outputTree->Branch("nCHSFatJets",       &nCHSFatJets,       "nCHSFatJets/I");
    outputTree->Branch("nCHSJetsAcceptanceCalo",          &nCHSJetsAcceptanceCalo,          "nCHSJetsAcceptanceCalo/I");
    outputTree->Branch("nCHSJetsNegativeAcceptanceCalo",          &nCHSJetsNegativeAcceptanceCalo,          "nCHSJetsNegativeAcceptanceCalo/I");
    outputTree->Branch("nCHSFatJetsAcceptanceCalo",       &nCHSFatJetsAcceptanceCalo,       "nCHSFatJetsAcceptanceCalo/I");
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
    outputTree->Branch("m_chi", &m_chi, "m_chi/I");
    outputTree->Branch("ctau", &ctau, "ctau/I");
    outputTree->Branch("is_central", &is_central, "is_central/O");
    outputTree->Branch("Muons", &Muons);
    outputTree->Branch("Electrons", &Electrons);
    outputTree->Branch("Photons", &Photons);
    outputTree->Branch("Taus", &skimmedTaus);
    outputTree->Branch("Jets", &skimmedJets);
    outputTree->Branch("JetsNegative", &skimmedJetsNegative);
    outputTree->Branch("JetsCaloAdd", &skimmedJetsCalo);
    outputTree->Branch("FatJets", &skimmedFatJets);
    outputTree->Branch("EcalRecHitsAK4", &EcalRecHitsAK4);
    outputTree->Branch("skimmedEcalRecHitsAK4", &skimmedEcalRecHitsAK4);
    outputTree->Branch("skimmedAcceptanceEcalRecHitsAK4", &skimmedAcceptanceEcalRecHitsAK4);
    outputTree->Branch("taggedEcalRecHitsAK4", &taggedEcalRecHitsAK4);
    outputTree->Branch("taggedAcceptanceEcalRecHitsAK4", &taggedAcceptanceEcalRecHitsAK4);
    outputTree->Branch("DT_fit_xx", &DT_fit_xx);
    outputTree->Branch("DT_fit_yy", &DT_fit_yy);
    outputTree->Branch("DT_fit_zz", &DT_fit_zz);
    outputTree->Branch("DT_fit_res", &DT_fit_res);
    if(doPFCand) outputTree->Branch("Jet_0_PFCandidatesAK4", &Jet_0_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_1_PFCandidatesAK4", &Jet_1_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_2_PFCandidatesAK4", &Jet_2_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_3_PFCandidatesAK4", &Jet_3_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_4_PFCandidatesAK4", &Jet_4_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_5_PFCandidatesAK4", &Jet_5_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_6_PFCandidatesAK4", &Jet_6_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_7_PFCandidatesAK4", &Jet_7_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_8_PFCandidatesAK4", &Jet_8_PFCandidatesAK4);
    if(doPFCand) outputTree->Branch("Jet_9_PFCandidatesAK4", &Jet_9_PFCandidatesAK4);

    if(doPFCand) outputTree->Branch("FatJet_0_PFCandidatesAK8", &FatJet_0_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_1_PFCandidatesAK8", &FatJet_1_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_2_PFCandidatesAK8", &FatJet_2_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_3_PFCandidatesAK8", &FatJet_3_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_4_PFCandidatesAK8", &FatJet_4_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_5_PFCandidatesAK8", &FatJet_5_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_6_PFCandidatesAK8", &FatJet_6_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_7_PFCandidatesAK8", &FatJet_7_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_8_PFCandidatesAK8", &FatJet_8_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_9_PFCandidatesAK8", &FatJet_9_PFCandidatesAK8);

    if(doPFCand) outputTree->Branch("FatJet_0_PFCandidatesAK8", &FatJet_0_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_1_PFCandidatesAK8", &FatJet_1_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_2_PFCandidatesAK8", &FatJet_2_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_3_PFCandidatesAK8", &FatJet_3_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_4_PFCandidatesAK8", &FatJet_4_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_5_PFCandidatesAK8", &FatJet_5_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_6_PFCandidatesAK8", &FatJet_6_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_7_PFCandidatesAK8", &FatJet_7_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_8_PFCandidatesAK8", &FatJet_8_PFCandidatesAK8);
    if(doPFCand) outputTree->Branch("FatJet_9_PFCandidatesAK8", &FatJet_9_PFCandidatesAK8);

    if(doPFCand) outputTree->Branch("FatJet_0_EcalRecHitsAK8", &FatJet_0_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_1_EcalRecHitsAK8", &FatJet_1_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_2_EcalRecHitsAK8", &FatJet_2_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_3_EcalRecHitsAK8", &FatJet_3_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_4_EcalRecHitsAK8", &FatJet_4_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_5_EcalRecHitsAK8", &FatJet_5_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_6_EcalRecHitsAK8", &FatJet_6_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_7_EcalRecHitsAK8", &FatJet_7_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_8_EcalRecHitsAK8", &FatJet_8_EcalRecHitsAK8);
    if(doPFCand) outputTree->Branch("FatJet_9_EcalRecHitsAK8", &FatJet_9_EcalRecHitsAK8);

    //outputTree->Branch("Jet_0_EcalRecHitsAK4", &Jet_0_EcalRecHitsAK4);
    //outputTree->Branch("Jet_1_EcalRecHitsAK4", &Jet_1_EcalRecHitsAK4);
    //outputTree->Branch("Jet_2_EcalRecHitsAK4", &Jet_2_EcalRecHitsAK4);
    //outputTree->Branch("Jet_3_EcalRecHitsAK4", &Jet_3_EcalRecHitsAK4);
    //outputTree->Branch("Jet_4_EcalRecHitsAK4", &Jet_4_EcalRecHitsAK4);
    //outputTree->Branch("Jet_5_EcalRecHitsAK4", &Jet_5_EcalRecHitsAK4);
    //outputTree->Branch("Jet_6_EcalRecHitsAK4", &Jet_6_EcalRecHitsAK4);
    //outputTree->Branch("Jet_7_EcalRecHitsAK4", &Jet_7_EcalRecHitsAK4);
    //outputTree->Branch("Jet_8_EcalRecHitsAK4", &Jet_8_EcalRecHitsAK4);
    //outputTree->Branch("Jet_9_EcalRecHitsAK4", &Jet_9_EcalRecHitsAK4);

    outputTree->Branch("MEt", &MEt);
    outputTree->Branch("GenHiggs", &GenHiggs);
    outputTree->Branch("GenLLPs", &GenLLPs);
    outputTree->Branch("GenBquarks", &GenBquarks);
    outputTree->Branch("GenGravitinos", &GenGravitinos);
    outputTree->Branch("DTSegments", &DTSegments);
    outputTree->Branch("CSCSegments", &CSCSegments);
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
    outputTree->Branch("nTagJets_0p996_JJ_eta_1p0",     &nTagJets_0p996_JJ_eta_1p0,      "nTagJets_0p996_JJ_eta_1p0/I");
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

    tensorflow::GraphDef* graphDefUnsmearedAK4 = tensorflow::loadGraphDef(graphPathAK4);
    tensorflow::Session* sessionUnsmearedAK4 = tensorflow::createSession(graphDefUnsmearedAK4, nThreads);
    tensorflow::Tensor inputTensorUnsmearedAK4(tensorflow::DT_FLOAT, {1, int(featuresAK4.size()) });
    float outputValueUnsmearedAK4;

    //tensorflow::GraphDef* graphDefAK8 = tensorflow::loadGraphDef(graphPathAK8);
    //tensorflow::Session* sessionAK8 = tensorflow::createSession(graphDefAK8, nThreads);
    //tensorflow::Tensor inputTensorAK8(tensorflow::DT_FLOAT, {1, int(featuresAK8.size()) });
    //float outputValueAK8;


    // Event loop

    //for(int i = 0; i < 10; i++) {
    for(int i = 0; i < inputTree->GetEntriesFast(); i++) {

        TriggerWeight = 1.;
        PUReWeight = 1.;
        PUReWeightUp = 1.;
        PUReWeightDown = 1.;
	//Initialize nTagJets at every event
        nCHSJetsAcceptanceCalo = 0;
        nCHSJetsNegativeAcceptanceCalo = 0;
        nCHSFatJetsAcceptanceCalo = 0;
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
	MinFatJetMetDPhi = 10.;
	MinJetMetDPhi = 10.;
	MinJetMetDPhiBarrel = 10.;
	MinJetMetDPhiStar = 10.;
	MinJetMetDPhiBarrelStar = 10.;
	MinFatJetMetDPhiBarrel = 10.;
	MinFatJetMetDPhiBarrelMatched = 10.;
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
	nTagJets_0p996_JJ_eta_1p0 = 0;
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
	skimmedTaus.clear();
        skimmedJets.clear();
        skimmedJetsNegative.clear();
        skimmedJetsCalo.clear();
        skimmedFatJets.clear();
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

        Jet_0_PFCandidatesAK4.clear();
        Jet_1_PFCandidatesAK4.clear();
        Jet_2_PFCandidatesAK4.clear();
        Jet_3_PFCandidatesAK4.clear();
        Jet_4_PFCandidatesAK4.clear();
        Jet_5_PFCandidatesAK4.clear();
        Jet_6_PFCandidatesAK4.clear();
        Jet_7_PFCandidatesAK4.clear();
        Jet_8_PFCandidatesAK4.clear();
        Jet_9_PFCandidatesAK4.clear();

        FatJet_0_PFCandidatesAK8.clear();
        FatJet_1_PFCandidatesAK8.clear();
        FatJet_2_PFCandidatesAK8.clear();
        FatJet_3_PFCandidatesAK8.clear();
        FatJet_4_PFCandidatesAK8.clear();
        FatJet_5_PFCandidatesAK8.clear();
        FatJet_6_PFCandidatesAK8.clear();
        FatJet_7_PFCandidatesAK8.clear();
        FatJet_8_PFCandidatesAK8.clear();
        FatJet_9_PFCandidatesAK8.clear();


        FatJet_0_PFCandidatesAK8.clear();
        FatJet_1_PFCandidatesAK8.clear();
        FatJet_2_PFCandidatesAK8.clear();
        FatJet_3_PFCandidatesAK8.clear();
        FatJet_4_PFCandidatesAK8.clear();
        FatJet_5_PFCandidatesAK8.clear();
        FatJet_6_PFCandidatesAK8.clear();
        FatJet_7_PFCandidatesAK8.clear();
        FatJet_8_PFCandidatesAK8.clear();
        FatJet_9_PFCandidatesAK8.clear();

        FatJet_0_EcalRecHitsAK8.clear();
        FatJet_1_EcalRecHitsAK8.clear();
        FatJet_2_EcalRecHitsAK8.clear();
        FatJet_3_EcalRecHitsAK8.clear();
        FatJet_4_EcalRecHitsAK8.clear();
        FatJet_5_EcalRecHitsAK8.clear();
        FatJet_6_EcalRecHitsAK8.clear();
        FatJet_7_EcalRecHitsAK8.clear();
        FatJet_8_EcalRecHitsAK8.clear();
        FatJet_9_EcalRecHitsAK8.clear();

        //Jet_0_EcalRecHitsAK4.clear();
        //Jet_1_EcalRecHitsAK4.clear();
        //Jet_2_EcalRecHitsAK4.clear();
        //Jet_3_EcalRecHitsAK4.clear();
        //Jet_4_EcalRecHitsAK4.clear();
        //Jet_5_EcalRecHitsAK4.clear();
        //Jet_6_EcalRecHitsAK4.clear();
        //Jet_7_EcalRecHitsAK4.clear();
        //Jet_8_EcalRecHitsAK4.clear();
        //Jet_9_EcalRecHitsAK4.clear();

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

	Long64_t TagNumber;
	if(isMC) TagNumber=EventNumber;
	else TagNumber=RunNumber;

	if(RunNumber>=319077)
	  {
	    RunNumber_in_HEM = true;
	  }

	//if (not ( (RunNumber==276775 and LumiNumber==437 and EventNumber==686255481) or (RunNumber==276870 and LumiNumber==246 and EventNumber==33815501) or (RunNumber==297050 and LumiNumber==96 and EventNumber==122060825) or (RunNumber==299061 and LumiNumber==43 and EventNumber==17473623) or (RunNumber==300636 and LumiNumber==1570 and EventNumber==1745799131 ) or (RunNumber==301959 and LumiNumber==656 and EventNumber==739086198) or (RunNumber==303948 and LumiNumber==55 and EventNumber==12029612) or (RunNumber==304506 and LumiNumber==115 and EventNumber==197159168) or (RunNumber==305365 and LumiNumber==556 and EventNumber==908806478) or (RunNumber==305862 and LumiNumber==437 and EventNumber==706695236) or (RunNumber==315705 and LumiNumber==693 and EventNumber==434537655) or (RunNumber==322492 and LumiNumber==1266 and EventNumber==2209685676) or (RunNumber==323470 and LumiNumber==104 and EventNumber==161827808 ) or (RunNumber==324841 and LumiNumber==1341 and EventNumber==2446831291) )) continue;
	//if(strcmp(argv[3], "y")==1 || strcmp(argv[3], "yes")==1)
	//{
	//if (EventNumber % 2 == 0)
	//{
	//std::cout << "Skip even EventNumber! " << std::endl;
	//continue;
	//}
	//}



        //std::cout << "======== " << std::endl;
        //std::cout << "Event " << entry << std::endl;
	//std::cout << "======== " << std::endl;

	//Consider PU weight

	//PUReWeight = PUWeightHist->GetBinContent(PUWeightHist->GetXaxis()->FindBin(MeanNumInteractions));
	//PUReWeightUp = PUWeightHistUp->GetBinContent(PUWeightHistUp->GetXaxis()->FindBin(MeanNumInteractions));
	//PUReWeightDown = PUWeightHistDown->GetBinContent(PUWeightHistDown->GetXaxis()->FindBin(MeanNumInteractions));

	if(isMC)
	  {
	    if(doSR) TriggerWeight = tr->GetBinContent(tr->GetXaxis()->FindBin(MEt->pt));//only for SR MC!!
	    PUReWeight = pu->GetBinContent(pu->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightUp = pu_up->GetBinContent(pu_up->GetXaxis()->FindBin(MeanNumInteractions));
	    PUReWeightDown = pu_down->GetBinContent(pu_down->GetXaxis()->FindBin(MeanNumInteractions));
	  }

	if(isMC and doGen)
	  {
            dR_LLPs = reco::deltaR(GenLLPs->at(0).eta,GenLLPs->at(0).phi,GenLLPs->at(1).eta,GenLLPs->at(1).phi);
            dR_Higgs = GenHiggs->size()==2 ? reco::deltaR(GenHiggs->at(0).eta,GenHiggs->at(0).phi,GenHiggs->at(1).eta,GenHiggs->at(1).phi) : -9.;
            dR_Gravitinos = reco::deltaR(GenGravitinos->at(0).eta,GenGravitinos->at(0).phi,GenGravitinos->at(1).eta,GenGravitinos->at(1).phi);
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
	if(doMR and not(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) ) continue;
	if(doMRPho and not(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v or HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v or HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v) ) continue;
	if(doZtoMM and not(HLT_IsoMu24_v or HLT_IsoMu27_v) ) continue;
	if(doZtoEE and not(HLT_Ele32_WPTight_Gsf_v or HLT_Ele35_WPTight_Gsf_v or HLT_Ele32_eta2p1_WPLoose_Gsf_v) ) continue;
	if(doWtoMN and not(HLT_IsoMu24_v or HLT_IsoMu27_v) ) continue;
	if(doWtoEN and not(HLT_Ele32_WPTight_Gsf_v or HLT_Ele35_WPTight_Gsf_v or HLT_Ele32_eta2p1_WPLoose_Gsf_v) ) continue;
	if(doMN and not(HLT_IsoMu24_v or HLT_IsoMu27_v) ) continue;
	if(doEN and not(HLT_Ele32_WPTight_Gsf_v or HLT_Ele35_WPTight_Gsf_v or HLT_Ele32_eta2p1_WPLoose_Gsf_v) ) continue;
	if(doTtoEM and not(HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_v or HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v or HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v or HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v or HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v or HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v or HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v or HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v or HLT_Mu27_Ele37_CaloIdL_MW_v or HLT_Mu37_Ele27_CaloIdL_MW_v) ) continue;
	if(doPho and not(HLT_Photon22_v or HLT_Photon30_v or HLT_Photon33_v or HLT_Photon36_v or HLT_Photon50_v or HLT_Photon75_v or HLT_Photon90_v or HLT_Photon120_v or HLT_Photon125_v or HLT_Photon150_v or HLT_Photon200_v or HLT_Photon175_v or HLT_Photon250_NoHE_v or HLT_Photon300_NoHE_v or HLT_Photon500_v or HLT_Photon600_v) ) continue;
	if(doJetHT and not(HLT_PFJet40_v or HLT_PFJet60_v or HLT_PFJet80_v or HLT_PFJet140_v or HLT_PFJet200_v or HLT_PFJet260_v or HLT_PFJet320_v or HLT_PFJet400_v or HLT_PFJet450_v or HLT_PFJet500_v or HLT_PFJet550_v) ) continue;
	if(doJetMET and not(HLT_PFJet40_v or HLT_PFJet60_v or HLT_PFJet80_v or HLT_PFJet140_v or HLT_PFJet200_v or HLT_PFJet260_v or HLT_PFJet320_v or HLT_PFJet400_v or HLT_PFJet450_v or HLT_PFJet500_v or HLT_PFJet550_v or HLT_DiPFJetAve40_v or HLT_DiPFJetAve60_v or HLT_DiPFJetAve80_v or HLT_DiPFJetAve200_v or HLT_DiPFJetAve500_v) ) continue;
	if(doDiJetMET and not(HLT_PFJet40_v or HLT_PFJet60_v or HLT_PFJet80_v or HLT_PFJet140_v or HLT_PFJet200_v or HLT_PFJet260_v or HLT_PFJet320_v or HLT_PFJet400_v or HLT_PFJet450_v or HLT_PFJet500_v or HLT_PFJet550_v) ) continue;

	//Selection on MET
        if(doSR and MEt->pt<200) continue;
        if(doMR and MEt->pt<200) continue;
        if(doMRPho and MEt->pt<200) continue;
	if(doZtoMM and MEt->pt>=30) continue;
	if(doZtoEE and MEt->pt>=30) continue;
	if(doWtoMN and MEt->pt<40) continue;
	if(doWtoEN and MEt->pt<40) continue;
	if(doMN and MEt->pt<200) continue;
	if(doEN and MEt->pt<200) continue;
	if(doTtoEM and MEt->pt<30) continue;
	if(doPho and MEt->pt>=30) continue;
	if(doJetHT and MEt->pt>=30) continue;
	if(doJetMET and MEt->pt<100) continue;//first attempt, try met 100
	if(doDiJetMET and MEt->pt<60) continue;//first attempt, try met 60

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
	    if( (doZtoMM or doWtoMN or doMN) and (Muons->at(m).pt<30 or !Muons->at(m).isTight or Muons->at(m).pfIso04>=0.15) ) continue;
	    if( (doTtoEM) and (Muons->at(m).pt<30) ) continue;
	    
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
	
	//WtoMN
	if(doWtoMN and MuonsStruct.size()!=1) continue;
	if(doMN and MuonsStruct.size()!=1) continue;
	//ZtoMM
	if(doZtoMM and MuonsStruct.size()!=2) continue;



	//Electrons
	float ele_iso = 0.4;
	//if(Electrons->size()>0) std::cout << "Electrons size: " << Electrons->size() << std::endl;
	for (unsigned int e=0; e<Electrons->size(); e++)
	  {

	    if(Electrons->at(e).eta>-3. and Electrons->at(e).eta<-1.3 and Electrons->at(e).phi>-1.57 and Electrons->at(e).phi<-0.87)
	      {
		nElectrons_in_HEM++;
	      }

	    //WtoEN and ZToEE CR
	    if( (doZtoEE or doWtoEN or doEN) and (Electrons->at(e).pt<37 or !Electrons->at(e).isTight) ) continue;
	    if( (doTtoEM) and (Electrons->at(e).pt<30 or !Electrons->at(e).isLoose) ) continue;

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

	//WtoEN
	if(doWtoEN and ElectronsStruct.size()!=1) continue;
	//ZtoEE
	if(doZtoEE and ElectronsStruct.size()!=2) continue;
	if(doEN and ElectronsStruct.size()!=1) continue;


	//TtoEN
	if(doTtoEM and not(ElectronsStruct.size()==1 and MuonsStruct.size()==1) ) continue;

	//nElectronsPassing = Electrons->size();
	//if(nElectronsPassing!=nElectrons) std::cout << "DIFFERENT! : " << nElectrons  << nElectronsPassing <<  std::endl;

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

	//Pho CR
	if( doPho and PhotonsStruct.size() != 1 ) continue;
	if( doMRPho and PhotonsStruct.size() != 1 ) continue;

	//Transverse mass met + Leptons (e and mu)
	TLorentzVector lepp4;
	for ( auto &tmp : LeptonsStruct )
	  {
	    lepp4 += tmp.vec;
	  }
	dPhi = reco::deltaPhi(MEt->phi, lepp4.Phi());
	MT = sqrt(2*(MEt->pt)*lepp4.Pt()*(1-cos(dPhi)));
	//if(doWtoEN and MT>=100) continue;
	//if(doWtoMN and MT>=100) continue;
	nLeptons = LeptonsStruct.size();
	if(doMR and LeptonsStruct.size()!=1) continue;
	if(doMR and MT>=100) continue;

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




        //if(nCHSJets<1 and nCHSFatJets<1) continue;
        //if(nTaus>0) continue;
        //if(nPhotons>0) continue;
        //if(nMuons>0) continue;
        //if(nElectrons>0) continue;
        ////if(HT<100) continue;

        if(isVerbose) std::cout << "======================================== " << std::endl;
        if(isVerbose) std::cout << "EventNumber " << EventNumber << "\tLumiNumber " << LumiNumber << std::endl;


	//Apply acceptance cuts to jets and fat jets 
	std::vector<int> validJetIndex;
	std::vector<int> validFatJetIndex;

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
	    


	    //Time smearing!
	    //We need:
	    //1. data file --> can pass from python
	    //2. signal file fit --> can pass from python
	    //3. name of the CB --> can pass from python



	    //Don't remove jets with eta>1 now, it messes up with the cosmic veto!
	    if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1.48 and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8)//cleaned jets!
	    //if( Jets->at(j).pt>30 and fabs(Jets->at(j).eta)<1. and Jets->at(j).timeRecHitsEB>-100. and Jets->at(j).muEFrac<0.6 and Jets->at(j).eleEFrac<0.6 and Jets->at(j).photonEFrac<0.8)//cleaned jets!
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

		//Time smearing here
		float pre_time = Jets->at(j).timeRecHitsEB;
		float smearer = smearCB->GetRandom();
		//Keep also the original time if needed
		Jets->at(j).timeRecHitsHB = pre_time;
		Jets->at(j).timeRecHitsEB = pre_time + smearer;

		//std::cout<< "Jet n. " << j << " pt: " << Jets->at(j).pt << " ; sigprob: " << Jets->at(j).sigprob  << std::endl;
		//here build the inputVector for each jet
		std::vector<float> inputValues(featuresAK4.size());
		std::vector<float> inputValuesUnsmeared(featuresAK4.size());

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
		    JetCaloType JetCalo;
		    FillJetCaloType( JetCalo, Jets->at(j), isMC );
		    skimmedJetsCalo.push_back(JetCalo);
		    validJetIndex.push_back(j);
		  }

		//save also jets including negative times
		skimmedJetsNegative.push_back(Jets->at(j));

	      }//acceptance

	  }//jet loop
	
        nCHSJetsAcceptanceCalo = skimmedJets.size();
        nCHSJetsNegativeAcceptanceCalo = skimmedJetsNegative.size();

	if(isVerbose) std::cout << "n. tagged jets " << nTagJets_0p996_JJ << std::endl;
        if(isVerbose) std::cout << "======================================== " << std::endl;

        for (unsigned int j=0; j<FatJets->size(); j++)
          {
	    if(fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi)) < MinFatJetMetDPhi) MinFatJetMetDPhi = fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi));

            if( FatJets->at(j).pt>170 && fabs(FatJets->at(j).eta)<1.48 and FatJets->at(j).timeRecHitsEB>-100.)
              {
		if(fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi)) < MinFatJetMetDPhiBarrel) MinFatJetMetDPhiBarrel = fabs(reco::deltaPhi(FatJets->at(j).phi, MEt->phi));


		//First: compute the eFracRecHitsEB as energyRecHitsEB/energy
		FatJets->at(j).eFracRecHitsEB = (FatJets->at(j).energy>0 and FatJets->at(j).energyRecHitsEB>0) ? FatJets->at(j).energyRecHitsEB/FatJets->at(j).energy : -1.;

		/*

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
		*/


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

			bool count_as_csc_en(false);
			bool count_as_csc_en_0p1(false);
			bool count_as_csc_en_0p04(false);
			//For Beam Halo: look at CSC and DT
			for(unsigned int csc=0; csc<CSCSegments->size(); csc++)
			  {
			    if( abs(reco::deltaPhi(CSCSegments->at(csc).phi,EcalRecHitsAK4->at(p).phi))<0.4  ) 
			      {
				//std::cout << "CSC[" << csc << "]: phi " << CSCSegments->at(csc).phi << std::endl;
				//std::cout<< "      : phi " << CSCSegments->at(csc).phi << std::endl;
				//std::cout<< "      : Dphi CSC-ECAL " << reco::deltaPhi(CSCSegments->at(csc).phi,EcalRecHitsAK4->at(p).phi) << std::endl;
				//std::cout<< "      : DR CSC-ECAL " << reco::deltaR(CSCSegments->at(csc).eta,CSCSegments->at(csc).phi,EcalRecHitsAK4->at(p).eta,EcalRecHitsAK4->at(p).phi) << std::endl;
				//csc_energy+=EcalRecHitsAK4->at(p).energy;
				count_as_csc_en = true;
				if( abs(reco::deltaPhi(CSCSegments->at(csc).phi,EcalRecHitsAK4->at(p).phi))<0.1  )
				  {
				    count_as_csc_en_0p1 = true;

				    if( abs(reco::deltaPhi(CSCSegments->at(csc).phi,EcalRecHitsAK4->at(p).phi))<0.04  )
				      {
					count_as_csc_en_0p04 = true;

				      }

				  } 
				
			      }
			  }

			if(count_as_csc_en) 
			  {
			    //cout << "Has at least 1 CSC associated hence count: " << endl;
			    //cout << "Corresp. ecal energy: " << EcalRecHitsAK4->at(p).energy <<endl;
			    csc_energy+=EcalRecHitsAK4->at(p).energy;
			  }

			if(count_as_csc_en_0p1) 
			  {
			    csc_energy_0p1+=EcalRecHitsAK4->at(p).energy;
			  }

			if(count_as_csc_en_0p04) 
			  {
			    csc_energy_0p04+=EcalRecHitsAK4->at(p).energy;
			  }

			//if(!count_as_csc_en) 
			//{
			//  cout << "Has no CSC associated hence NOT count: " << endl;
			//  cout << "Corresp. ecal energy: " << EcalRecHitsAK4->at(p).energy <<endl;
			//}

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

	    //cout << "Jet [" << j << "]: " << endl;
	    //cout << "ECAL associated CSC energy: " << csc_energy << endl;
	    //cout << "Tot ECAL energy: " << skimmedJetsCalo.at(j).energyRecHitsEB << endl;
	    //cout << "check_ecal_energy : " << check_ecal_energy << endl;

	    skimmedJetsCalo.at(j).energyEB2CSC = csc_energy > 0 ? csc_energy : -1;
	    skimmedJetsCalo.at(j).eFracEB2CSC  = skimmedJetsCalo.at(j).energyRecHitsEB > 0 ? csc_energy / skimmedJetsCalo.at(j).energyRecHitsEB : -1.;

	    skimmedJetsCalo.at(j).energyEB2CSC0p1 = csc_energy_0p1 > 0 ? csc_energy_0p1 : -1;
	    skimmedJetsCalo.at(j).eFracEB2CSC0p1  = skimmedJetsCalo.at(j).energyRecHitsEB > 0 ? csc_energy_0p1 / skimmedJetsCalo.at(j).energyRecHitsEB : -1.;

	    skimmedJetsCalo.at(j).energyEB2CSC0p04 = csc_energy_0p04 > 0 ? csc_energy_0p04 : -1;
	    skimmedJetsCalo.at(j).eFracEB2CSC0p04  = skimmedJetsCalo.at(j).energyRecHitsEB > 0 ? csc_energy_0p04 / skimmedJetsCalo.at(j).energyRecHitsEB : -1.;

	    skimmedJetsCalo.at(j).meanEtaEB = avg(EBeta_j);
	    skimmedJetsCalo.at(j).meanPhiEB = avg(EBphi_j);
	    skimmedJetsCalo.at(j).meanATLASEB = avg(EB_Dphi_j);
	    skimmedJetsCalo.at(j).meanXEB = avg(EBx_j);
	    skimmedJetsCalo.at(j).meanYEB = avg(EBy_j);
	    skimmedJetsCalo.at(j).meanZEB = avg(EBz_j);
	    skimmedJetsCalo.at(j).meanREB = avg(EBr_j);
	    skimmedJetsCalo.at(j).spreadEtaEB = stdev(EBeta_j);
	    skimmedJetsCalo.at(j).spreadPhiEB = stdev(EBphi_j);
	    skimmedJetsCalo.at(j).spreadATLASEB = stdev(EB_Dphi_j);
	    skimmedJetsCalo.at(j).spreadXEB = stdev(EBx_j);
	    skimmedJetsCalo.at(j).spreadYEB = stdev(EBy_j);
	    skimmedJetsCalo.at(j).spreadZEB = stdev(EBz_j);
	    skimmedJetsCalo.at(j).spreadREB = stdev(EBr_j);

	    //Energy weighted
	    skimmedJetsCalo.at(j).meanWeightedEtaEB = weighted_avg(EBeta_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedPhiEB = weighted_avg(EBphi_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedATLASEB = weighted_avg(EB_Dphi_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedXEB = weighted_avg(EBx_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedYEB = weighted_avg(EBy_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedZEB = weighted_avg(EBz_j,EBenergy_j);
	    skimmedJetsCalo.at(j).meanWeightedREB = weighted_avg(EBr_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedEtaEB = biased_weighted_stdev(EBeta_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedPhiEB = biased_weighted_stdev(EBphi_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedATLASEB = biased_weighted_stdev(EB_Dphi_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedXEB = biased_weighted_stdev(EBx_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedYEB = biased_weighted_stdev(EBy_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedZEB = biased_weighted_stdev(EBz_j,EBenergy_j);
	    skimmedJetsCalo.at(j).spreadWeightedREB = biased_weighted_stdev(EBr_j,EBenergy_j);

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

	    /*
	    if(DTSegments->at(d).time > -9999.)
	    {
	        Point p;
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
		points_valid_time.push_back(p);
	    }
	    */
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
	//choose the right pair of cosmic clouds
	if(n_clusters>=2 and nCosmicMuonsOneLeg>0 and nCosmicMuons>1)
	  {
	    //This is clearly an overshooting
	    //Simplify!! TODO
	    //It's not possible to mask arrays or so
	    //can probably create a vector of vectors to keep the size dynamic
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

		    //Opposite emispheres condition plus comsic tracks
		    //cout << "Pair: " << k1 << " " << k2 << endl;
		    //cout << "TagNumber  x1, x2, opp,   y1, y2, opp,    z1, z2, opp " << k1 << " " << k2 << endl;
		    //printf("%llu & %5.2lf & %5.2lf & %d & %5.2lf & %5.2lf & %d & %5.2lf & %5.2lf & %d \\\\ \n",
		    //TagNumber,
		    // avg(vec_xx.at(k1)),avg(vec_xx.at(k2)),(avg(vec_xx.at(k1))*avg(vec_xx.at(k2))<0),
		    // avg(vec_yy.at(k1)),avg(vec_yy.at(k2)),(avg(vec_yy.at(k1))*avg(vec_yy.at(k2))<0),
		    // avg(vec_zz.at(k1)),avg(vec_zz.at(k2)),(avg(vec_zz.at(k1))*avg(vec_zz.at(k2))<0)
		    // );

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

		//printf("%llu & %5.2lf & %5.2lf & %d & %5.2lf & %5.2lf & %d & %5.2lf & %5.2lf & %d \\\\ \n",
		//       TagNumber,
		//       avg(vec_xx.at(ch_k1)),avg(vec_xx.at(ch_k2)),(avg(vec_xx.at(ch_k1))*avg(vec_xx.at(ch_k2))<0),
		//       avg(vec_yy.at(ch_k1)),avg(vec_yy.at(ch_k2)),(avg(vec_yy.at(ch_k1))*avg(vec_yy.at(ch_k2))<0),
		//       avg(vec_zz.at(ch_k1)),avg(vec_zz.at(ch_k2)),(avg(vec_zz.at(ch_k1))*avg(vec_zz.at(ch_k2))<0)
		//       );
		
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
		if(printFit) std::cout << TagNumber<< " & " << nTagJets_0p996_JJ << " & " << DT_fit_xx.size()  << " &" << dt_fit_chi2 << " & " << dt_fit_chi2_reduced << " & " << sqrt(distance2(mean_ECAL_tag_x,mean_ECAL_tag_y,mean_ECAL_tag_z,SolXZ,SolYZ)) << "\\\\ " << std::endl;
		if(printFit) cout << "      " << endl;


		if(dt_ecal_dist<0.5)
		  {
		    isCosmic = true;
		  }

	      }

	  }//if 2 clusters




        if(doPFCand and nCHSJetsAcceptanceCalo>0)
          {
	    std::sort(PFCandidatesAK4->begin(), PFCandidatesAK4->end(), pt_sorter);
            for(unsigned int p=0; p<PFCandidatesAK4->size(); p++)
	      {

		for (unsigned int j=0; j<validJetIndex.size(); j++)
                  {
		    if(PFCandidatesAK4->at(p).jetIndex== int(validJetIndex.at(j)) )
		      {
			if(j==0) 
			  {
			    Jet_0_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			  }
			else if(j==1) Jet_1_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==2) Jet_2_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==3) Jet_3_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==4) Jet_4_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==5) Jet_5_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==6) Jet_6_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==7) Jet_7_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==8) Jet_8_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
			else if(j==9) Jet_9_PFCandidatesAK4.push_back(PFCandidatesAK4->at(p));
		      }//check pf cand and jet indices
		  }//loop on jet indices
	      }//loop on pf candidates
          }//doPfCandidates



        if(doPFCand and nCHSFatJetsAcceptanceCalo>0)
          {
	    std::sort(PFCandidatesAK8->begin(), PFCandidatesAK8->end(), pt_sorter);
            //Loop on PFCandidates
            for(unsigned int p=0; p<PFCandidatesAK8->size(); p++)
	      {
		for (unsigned int j=0; j<validFatJetIndex.size(); j++)
                  {
		    if(PFCandidatesAK8->at(p).fatJetIndex== int(validFatJetIndex.at(j)) )
		      {
			if(j==0) 
			  {
			    FatJet_0_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			  }
			else if(j==1) FatJet_1_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==2) FatJet_2_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==3) FatJet_3_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==4) FatJet_4_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==5) FatJet_5_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==6) FatJet_6_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==7) FatJet_7_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==8) FatJet_8_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
			else if(j==9) FatJet_9_PFCandidatesAK8.push_back(PFCandidatesAK8->at(p));
		      }//check pf cand and jet indices
		  }//loop on jet indices
	      }//loop on pf candidates

	    //Loop on EcalRecHitsAK8
	    std::sort(EcalRecHitsAK8->begin(), EcalRecHitsAK8->end(), energy_sorter);
            for(unsigned int p=0; p<EcalRecHitsAK8->size(); p++)
	      {
		for (unsigned int j=0; j<validFatJetIndex.size(); j++)
                  {
		    if(int(EcalRecHitsAK8->at(p).jetIndex) == int(validFatJetIndex.at(j)) )//only this is complaining...
		      {
			//pf_index++;
			if(j==0) FatJet_0_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==1) FatJet_1_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==2) FatJet_2_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==3) FatJet_3_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==4) FatJet_4_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==5) FatJet_5_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==6) FatJet_6_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==7) FatJet_7_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==8) FatJet_8_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
			else if(j==9) FatJet_9_EcalRecHitsAK8.push_back(EcalRecHitsAK8->at(p));
		      }//check pf cand and jet indices
		  }//loop on jet indices
	      }//loop on EcalRecHitsAK8
          }//if doPFCandidates


	//Veto objects
	if(doSR and nMuonsPassing!=0) continue;
	if(doSR and nElectronsPassing!=0) continue;
	if(doSR and nTausPassing!=0) continue;
	if(doSR and nPhotonsPassing!=0) continue;

	//MinLeadingJetMetDPhi minimal requirement
	if(doJetMET and MinLeadingJetMetDPhi<0) continue;

	//Exactly 2 jets
	if(doDiJetMET and nCHSJetsAcceptanceCalo!=2) continue;

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
	if(doSR) isSR = true;
	if(doMR) isMR = true;
	if(doMRPho) isMRPho = true;
	if(doZtoMM) isZtoMM = true;
	if(doZtoEE) isZtoEE = true;
	if(doTtoEM) isTtoEM = true;
	if(doWtoEN) isWtoEN = true;
	if(doWtoMN) isWtoMN = true;
	if(doEN) isEN = true;
	if(doMN) isMN = true;
	if(doPho) isPho = true;
	if(doJetHT) isJetHT = true;
	if(doJetMET) isJetMET = true;
	if(doDiJetMET) isDiJetMET = true;


	//Observed worse agreement, skip this --> redo
	n_pass->Fill(0.);
	if(EventNumber % 2 == 0) n_even->Fill(0.);
	if(EventNumber % 2 != 0) n_odd->Fill(0.);
	if(skipTrain==true and EventNumber % 2 == 0) continue;
	outputTree->Fill();

        //std::cout << "======================================== " << std::endl;
	//for(unsigned int j=0;j<skimmedJets.size();j++)
	//{
	//std::cout << "Jet ["<<j<<"] eta: " << skimmedJets.at(j).eta << " ; phi: " << skimmedJets.at(j).phi<< " ; DNN: " << skimmedJets.at(j).sigprob << std::endl;
	//}
	//std::cout << "min_jets_dPhi: " << min_dPhi_jets << std::endl;
	//std::cout << "min_jets_dPhi_0p996: " << min_dPhi_jets_0p996 << std::endl;
	//std::cout << "min_jets_dEta_0p996: " << min_dEta_jets_0p996 << std::endl;

	//
	//if(skipTrain==true)
	//  {
	//    if(EventNumber % 2 != 0) outputTree->Fill();
	//    else continue;
	//  }
	//if(skipTrain==false)
	//  {
	//    outputTree->Fill();
	//  }
	//

	//Clear all the vectors
	//
        //skimmedJets.clear();
        //skimmedFatJets.clear();
        //Jet_0_PFCandidatesAK4.clear();
        //Jet_1_PFCandidatesAK4.clear();
        //Jet_2_PFCandidatesAK4.clear();
        //Jet_3_PFCandidatesAK4.clear();
        //Jet_4_PFCandidatesAK4.clear();
        //Jet_5_PFCandidatesAK4.clear();
        //Jet_6_PFCandidatesAK4.clear();
        //Jet_7_PFCandidatesAK4.clear();
        //Jet_8_PFCandidatesAK4.clear();
        //Jet_9_PFCandidatesAK4.clear();

        //FatJet_0_PFCandidatesAK8.clear();
        //FatJet_1_PFCandidatesAK8.clear();
        //FatJet_2_PFCandidatesAK8.clear();
        //FatJet_3_PFCandidatesAK8.clear();
        //FatJet_4_PFCandidatesAK8.clear();
        //FatJet_5_PFCandidatesAK8.clear();
        //FatJet_6_PFCandidatesAK8.clear();
        //FatJet_7_PFCandidatesAK8.clear();
        //FatJet_8_PFCandidatesAK8.clear();
        //FatJet_9_PFCandidatesAK8.clear();


        //FatJet_0_PFCandidatesAK8.clear();
        //FatJet_1_PFCandidatesAK8.clear();
        //FatJet_2_PFCandidatesAK8.clear();
        //FatJet_3_PFCandidatesAK8.clear();
        //FatJet_4_PFCandidatesAK8.clear();
        //FatJet_5_PFCandidatesAK8.clear();
        //FatJet_6_PFCandidatesAK8.clear();
        //FatJet_7_PFCandidatesAK8.clear();
        //FatJet_8_PFCandidatesAK8.clear();
        //FatJet_9_PFCandidatesAK8.clear();

        //FatJet_0_EcalRecHitsAK8.clear();
        //FatJet_1_EcalRecHitsAK8.clear();
        //FatJet_2_EcalRecHitsAK8.clear();
        //FatJet_3_EcalRecHitsAK8.clear();
        //FatJet_4_EcalRecHitsAK8.clear();
        //FatJet_5_EcalRecHitsAK8.clear();
        //FatJet_6_EcalRecHitsAK8.clear();
        //FatJet_7_EcalRecHitsAK8.clear();
        //FatJet_8_EcalRecHitsAK8.clear();
        //FatJet_9_EcalRecHitsAK8.clear();
	//

    }


    // finalize files
    outputTree->SetWeight(tree_weight);
    counter->Write();
    n_pass->Write();
    n_odd->Write();
    n_even->Write();
    b_skipTrain->Write();

    //PUWeightHist->Write();
    //pileup_mc_copy->Write();
    //pileup_data_copy->Write();
    //pileup_data_up_copy->Write();
    //pileup_data_down_copy->Write();

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


