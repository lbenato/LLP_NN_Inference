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

#include "NNInferenceCMSSW/LLP_NN_Inference/plugins/dbscan.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <chrono>//compute time
#include <ctime>//compute time 

using namespace ROOT::Math;
using namespace std;
using namespace Eigen;


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

    if(argc<2)
      {
	std::cout<<"Invalid arguments, exit!" << std::endl;
	return 0;
      }

    bool isVerbose(false);
    bool printFit(false);

    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    auto start = std::chrono::system_clock::now();//time!     

    std::string basePath = std::string(std::getenv("CMSSW_BASE")) + "/src/NNInferenceCMSSW/LLP_NN_Inference/nn_inference";
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];//!!!//"/test_on_real_ntuple.root";
    std::string inputTreeName = "ntuples/llp";
    std::string outputTreeName = "tree";//inputTreeName;
    
    // ================= 
    // Input
    // ================= 

    // open input file, read the tree and register input branches
    TFile* inputFile = new TFile(inputPath.c_str(), "READ");
    TTree* inputTree = (TTree*)inputFile->Get(inputTreeName.c_str());

    float  tree_weight = inputTree->GetWeight();
    if(isVerbose) std::cout << "Tree weight: " << tree_weight << std::endl;

    // Input variables
    Int_t           nGenParticle;
    Int_t           gParticleMotherId[2000];   //[nGenParticle]
    Int_t           gParticleMotherIndex[2000];   //[nGenParticle]
    Int_t           gParticleId[2000];   //[nGenParticle]
    Int_t           gParticleStatus[2000];   //[nGenParticle]
    Float_t         gParticleE[2000];   //[nGenParticle]
    Float_t         gParticlePt[2000];   //[nGenParticle]
    Float_t         gParticleEta[2000];   //[nGenParticle]
    Float_t         gParticlePhi[2000];   //[nGenParticle]
    Float_t         gParticleProdVertexX[2000];   //[nGenParticle]
    Float_t         gParticleProdVertexY[2000];   //[nGenParticle]
    Float_t         gParticleProdVertexZ[2000];   //[nGenParticle]
    Float_t         gParticleDecayVertexX[2000];   //[nGenParticle]
    Float_t         gParticleDecayVertexY[2000];   //[nGenParticle]
    Float_t         gParticleDecayVertexZ[2000];   //[nGenParticle]
    Float_t         gLLP_decay_vertex_x[2];
    Float_t         gLLP_decay_vertex_y[2];
    Float_t         gLLP_decay_vertex_z[2];
    Float_t         gLLP_beta[2];
    Float_t         gLLP_e[2];
    Float_t         gLLP_pt[2];
    Float_t         gLLP_eta[2];
    Float_t         gLLP_phi[2];
    Bool_t          gLLP_csc[2];
    Bool_t          gLLP_dt[2];
    Float_t         gLLP_travel_time[2];
    Int_t           gLLP_daughter_id[4];
    Float_t         gLLP_daughter_pt[4];
    Float_t         gLLP_daughter_eta[4];
    Float_t         gLLP_daughter_phi[4];
    Float_t         gLLP_daughter_eta_ecalcorr[4];
    Float_t         gLLP_daughter_phi_ecalcorr[4];
    Float_t         gLLP_daughter_e[4];
    Float_t         gLLP_daughter_mass[4];
    Int_t           gLLP_grandaughter_id[4];
    Float_t         gLLP_grandaughter_pt[4];
    Float_t         gLLP_grandaughter_eta[4];
    Float_t         gLLP_grandaughter_phi[4];
    Float_t         gLLP_grandaughter_eta_ecalcorr[4];
    Float_t         gLLP_grandaughter_phi_ecalcorr[4];
    Float_t         gLLP_grandaughter_e[4];
    Float_t         gLLP_grandaughter_mass[4];
    Int_t           nCscRechitClusters;
    Int_t           nDtRechitClusters;
    Float_t         cscRechitCluster_match_cscSegCluster_minDeltaR[15];
    Int_t           cscRechitCluster_match_cscSegCluster_index[15];
    Float_t         cscRechitCluster_match_gParticle_minDeltaR[15]; 
    Int_t           cscRechitCluster_match_gParticle_index[15];
    Int_t           cscRechitCluster_match_gParticle_id[15];
    Float_t         cscRechitClusterPhi[15];
    Float_t         cscRechitClusterEta[15];
    Int_t           cscRechitClusterSize[15];
    Float_t         dtRechitClusterPhi[15];
    Float_t         dtRechitClusterEta[15];
    Int_t           dtRechitClusterSize[15];
    

    // Input branches
    TBranch        *b_nGenParticle;   //!
    TBranch        *b_gParticleMotherId;   //!
    TBranch        *b_gParticleMotherIndex;   //!
    TBranch        *b_gParticleId;   //!
    TBranch        *b_gParticleStatus;   //!
    TBranch        *b_gParticleE;   //!
    TBranch        *b_gParticlePt;   //!
    TBranch        *b_gParticleEta;   //!
    TBranch        *b_gParticlePhi;   //!
    TBranch        *b_gParticleProdVertexX;   //!
    TBranch        *b_gParticleProdVertexY;   //!
    TBranch        *b_gParticleProdVertexZ;   //!
    TBranch        *b_gParticleDecayVertexX;   //!
    TBranch        *b_gParticleDecayVertexY;   //!
    TBranch        *b_gParticleDecayVertexZ;   //!
    TBranch        *b_gLLP_decay_vertex_x;   //!
    TBranch        *b_gLLP_decay_vertex_y;   //!
    TBranch        *b_gLLP_decay_vertex_z;   //!
    TBranch        *b_gLLP_beta;   //!
    TBranch        *b_gLLP_e;   //!
    TBranch        *b_gLLP_pt;   //!
    TBranch        *b_gLLP_eta;   //!
    TBranch        *b_gLLP_phi;   //!
    TBranch        *b_gLLP_csc;   //!
    TBranch        *b_gLLP_dt;   //!
    TBranch        *b_gLLP_travel_time;   //!
    TBranch        *b_gLLP_daughter_id;   //!
    TBranch        *b_gLLP_daughter_pt;   //!
    TBranch        *b_gLLP_daughter_eta;   //!
    TBranch        *b_gLLP_daughter_phi;   //!
    TBranch        *b_gLLP_daughter_eta_ecalcorr;   //!
    TBranch        *b_gLLP_daughter_phi_ecalcorr;   //!
    TBranch        *b_gLLP_daughter_e;   //!
    TBranch        *b_gLLP_daughter_mass;   //!
    TBranch        *b_gLLP_grandaughter_id;   //!
    TBranch        *b_gLLP_grandaughter_pt;   //!
    TBranch        *b_gLLP_grandaughter_eta;   //!
    TBranch        *b_gLLP_grandaughter_phi;   //!
    TBranch        *b_gLLP_grandaughter_eta_ecalcorr;   //!
    TBranch        *b_gLLP_grandaughter_phi_ecalcorr;   //!
    TBranch        *b_gLLP_grandaughter_e;   //!
    TBranch        *b_gLLP_grandaughter_mass;   //!
    TBranch        *b_nCscRechitClusters;
    TBranch        *b_nDtRechitClusters;
    TBranch        *b_cscRechitCluster_match_cscSegCluster_minDeltaR;
    TBranch        *b_cscRechitCluster_match_cscSegCluster_index;
    TBranch        *b_cscRechitCluster_match_gParticle_minDeltaR; 
    TBranch        *b_cscRechitCluster_match_gParticle_index;
    TBranch        *b_cscRechitCluster_match_gParticle_id;
    TBranch        *b_cscRechitClusterPhi;
    TBranch        *b_cscRechitClusterEta;
    TBranch        *b_cscRechitClusterSize;
    TBranch        *b_dtRechitClusterPhi;
    TBranch        *b_dtRechitClusterEta;
    TBranch        *b_dtRechitClusterSize;


    
    inputTree->SetBranchAddress("nGenParticle", &nGenParticle, &b_nGenParticle);
    inputTree->SetBranchAddress("gParticleMotherId", gParticleMotherId, &b_gParticleMotherId);
    inputTree->SetBranchAddress("gParticleMotherIndex", gParticleMotherIndex, &b_gParticleMotherIndex);
    inputTree->SetBranchAddress("gParticleId", gParticleId, &b_gParticleId);
    inputTree->SetBranchAddress("gParticleStatus", gParticleStatus, &b_gParticleStatus);
    inputTree->SetBranchAddress("gParticleE", gParticleE, &b_gParticleE);
    inputTree->SetBranchAddress("gParticlePt", gParticlePt, &b_gParticlePt);
    inputTree->SetBranchAddress("gParticleEta", gParticleEta, &b_gParticleEta);
    inputTree->SetBranchAddress("gParticlePhi", gParticlePhi, &b_gParticlePhi);
    inputTree->SetBranchAddress("gParticleProdVertexX", gParticleProdVertexX, &b_gParticleProdVertexX);
    inputTree->SetBranchAddress("gParticleProdVertexY", gParticleProdVertexY, &b_gParticleProdVertexY);
    inputTree->SetBranchAddress("gParticleProdVertexZ", gParticleProdVertexZ, &b_gParticleProdVertexZ);
    inputTree->SetBranchAddress("gParticleDecayVertexX", gParticleDecayVertexX, &b_gParticleDecayVertexX);
    inputTree->SetBranchAddress("gParticleDecayVertexY", gParticleDecayVertexY, &b_gParticleDecayVertexY);
    inputTree->SetBranchAddress("gParticleDecayVertexZ", gParticleDecayVertexZ, &b_gParticleDecayVertexZ);
    inputTree->SetBranchAddress("gLLP_decay_vertex_x", gLLP_decay_vertex_x, &b_gLLP_decay_vertex_x);
    inputTree->SetBranchAddress("gLLP_decay_vertex_y", gLLP_decay_vertex_y, &b_gLLP_decay_vertex_y);
    inputTree->SetBranchAddress("gLLP_decay_vertex_z", gLLP_decay_vertex_z, &b_gLLP_decay_vertex_z);
    inputTree->SetBranchAddress("gLLP_beta", gLLP_beta, &b_gLLP_beta);
    inputTree->SetBranchAddress("gLLP_e", gLLP_e, &b_gLLP_e);
    inputTree->SetBranchAddress("gLLP_pt", gLLP_pt, &b_gLLP_pt);
    inputTree->SetBranchAddress("gLLP_eta", gLLP_eta, &b_gLLP_eta);
    inputTree->SetBranchAddress("gLLP_phi", gLLP_phi, &b_gLLP_phi);
    inputTree->SetBranchAddress("gLLP_csc", gLLP_csc, &b_gLLP_csc);
    inputTree->SetBranchAddress("gLLP_dt", gLLP_dt, &b_gLLP_dt);
    inputTree->SetBranchAddress("gLLP_travel_time", gLLP_travel_time, &b_gLLP_travel_time);
    inputTree->SetBranchAddress("gLLP_daughter_id", gLLP_daughter_id, &b_gLLP_daughter_id);
    inputTree->SetBranchAddress("gLLP_daughter_pt", gLLP_daughter_pt, &b_gLLP_daughter_pt);
    inputTree->SetBranchAddress("gLLP_daughter_eta", gLLP_daughter_eta, &b_gLLP_daughter_eta);
    inputTree->SetBranchAddress("gLLP_daughter_phi", gLLP_daughter_phi, &b_gLLP_daughter_phi);
    inputTree->SetBranchAddress("gLLP_daughter_eta_ecalcorr", gLLP_daughter_eta_ecalcorr, &b_gLLP_daughter_eta_ecalcorr);
    inputTree->SetBranchAddress("gLLP_daughter_phi_ecalcorr", gLLP_daughter_phi_ecalcorr, &b_gLLP_daughter_phi_ecalcorr);
    inputTree->SetBranchAddress("gLLP_daughter_e", gLLP_daughter_e, &b_gLLP_daughter_e);
    inputTree->SetBranchAddress("gLLP_daughter_mass", gLLP_daughter_mass, &b_gLLP_daughter_mass);
    inputTree->SetBranchAddress("gLLP_grandaughter_id", gLLP_grandaughter_id, &b_gLLP_grandaughter_id);
    inputTree->SetBranchAddress("gLLP_grandaughter_pt", gLLP_grandaughter_pt, &b_gLLP_grandaughter_pt);
    inputTree->SetBranchAddress("gLLP_grandaughter_eta", gLLP_grandaughter_eta, &b_gLLP_grandaughter_eta);
    inputTree->SetBranchAddress("gLLP_grandaughter_phi", gLLP_grandaughter_phi, &b_gLLP_grandaughter_phi);
    inputTree->SetBranchAddress("gLLP_grandaughter_eta_ecalcorr", gLLP_grandaughter_eta_ecalcorr, &b_gLLP_grandaughter_eta_ecalcorr);
    inputTree->SetBranchAddress("gLLP_grandaughter_phi_ecalcorr", gLLP_grandaughter_phi_ecalcorr, &b_gLLP_grandaughter_phi_ecalcorr);
    inputTree->SetBranchAddress("gLLP_grandaughter_e", gLLP_grandaughter_e, &b_gLLP_grandaughter_e);
    inputTree->SetBranchAddress("gLLP_grandaughter_mass", gLLP_grandaughter_mass, &b_gLLP_grandaughter_mass);

    inputTree->SetBranchAddress("nCscRechitClusters", &nCscRechitClusters, &b_nCscRechitClusters);
    inputTree->SetBranchAddress("nDtRechitClusters", &nDtRechitClusters, &b_nDtRechitClusters);
    inputTree->SetBranchAddress("cscRechitCluster_match_cscSegCluster_minDeltaR", cscRechitCluster_match_cscSegCluster_minDeltaR, &b_cscRechitCluster_match_cscSegCluster_minDeltaR);
    inputTree->SetBranchAddress("cscRechitCluster_match_cscSegCluster_index", cscRechitCluster_match_cscSegCluster_index, &b_cscRechitCluster_match_cscSegCluster_index);
    inputTree->SetBranchAddress("cscRechitCluster_match_gParticle_minDeltaR", cscRechitCluster_match_gParticle_minDeltaR, &b_cscRechitCluster_match_gParticle_minDeltaR);
    inputTree->SetBranchAddress("cscRechitCluster_match_gParticle_index", cscRechitCluster_match_gParticle_index, &b_cscRechitCluster_match_gParticle_index);
    inputTree->SetBranchAddress("cscRechitCluster_match_gParticle_id", cscRechitCluster_match_gParticle_id, &b_cscRechitCluster_match_gParticle_id);
    inputTree->SetBranchAddress("cscRechitClusterPhi", cscRechitClusterPhi, &b_cscRechitClusterPhi);
    inputTree->SetBranchAddress("cscRechitClusterEta", cscRechitClusterEta, &b_cscRechitClusterEta);
    inputTree->SetBranchAddress("cscRechitClusterSize", cscRechitClusterSize, &b_cscRechitClusterSize);
    inputTree->SetBranchAddress("dtRechitClusterPhi", dtRechitClusterPhi, &b_dtRechitClusterPhi);
    inputTree->SetBranchAddress("dtRechitClusterEta", dtRechitClusterEta, &b_dtRechitClusterEta);
    inputTree->SetBranchAddress("dtRechitClusterSize", dtRechitClusterSize, &b_dtRechitClusterSize);

    // ================= 
    // Output
    // ================= 

    TFile* outputFile = new TFile(outputPath.c_str(), "RECREATE");
    outputFile->cd();
    TTree *outputTree = new TTree(outputTreeName.c_str(), "");

    std::vector<GenPType> tops;
    std::vector<GenPType> higgses;
    std::vector<GenPType> LLPs;
    std::vector<GenPType> taus;

    GenPType top0;
    GenPType top1;
    GenPType higgs;
    GenPType LLP0;
    GenPType LLP1;
    GenPType tau0;
    GenPType tau1;
    GenPType tau2;
    GenPType tau3;

    float dR_tau_top_0(999.);
    float dR_tau_top_1(999.);
    float dPhi_tau_top_0(999.);
    float dPhi_tau_top_1(999.);

    float dR_CSC_clusters_top_0;
    float dR_CSC_clusters_top_1;
    float dPhi_CSC_clusters_top_0;
    float dPhi_CSC_clusters_top_1;

    float dR_CSC_clusters_higgs;
    float dPhi_CSC_clusters_higgs;


    float dR_DT_clusters_top_0;
    float dR_DT_clusters_top_1;
    float dPhi_DT_clusters_top_0;
    float dPhi_DT_clusters_top_1;

    float dR_DT_clusters_higgs;
    float dPhi_DT_clusters_higgs;

    // Output branches 
    outputTree->Branch("dR_tau_top_0",       &dR_tau_top_0,       "dR_tau_top_0/F");
    outputTree->Branch("dPhi_tau_top_0",       &dPhi_tau_top_0,       "dPhi_tau_top_0/F");
    outputTree->Branch("dR_tau_top_1",       &dR_tau_top_1,       "dR_tau_top_1/F");
    outputTree->Branch("dPhi_tau_top_1",       &dPhi_tau_top_1,       "dPhi_tau_top_1/F");

    outputTree->Branch("dR_CSC_clusters_top_0",       &dR_CSC_clusters_top_0,       "dR_CSC_clusters_top_0/F");
    outputTree->Branch("dPhi_CSC_clusters_top_0",       &dPhi_CSC_clusters_top_0,       "dPhi_CSC_clusters_top_0/F");
    outputTree->Branch("dR_CSC_clusters_top_1",       &dR_CSC_clusters_top_1,       "dR_CSC_clusters_top_1/F");
    outputTree->Branch("dPhi_CSC_clusters_top_1",       &dPhi_CSC_clusters_top_1,       "dPhi_CSC_clusters_top_1/F");

    outputTree->Branch("dR_CSC_clusters_higgs",       &dR_CSC_clusters_higgs,       "dR_CSC_clusters_higgs/F");
    outputTree->Branch("dPhi_CSC_clusters_higgs",       &dPhi_CSC_clusters_higgs,       "dPhi_CSC_clusters_higgs/F");

    outputTree->Branch("dR_DT_clusters_top_0",       &dR_DT_clusters_top_0,       "dR_DT_clusters_top_0/F");
    outputTree->Branch("dPhi_DT_clusters_top_0",       &dPhi_DT_clusters_top_0,       "dPhi_DT_clusters_top_0/F");
    outputTree->Branch("dR_DT_clusters_top_1",       &dR_DT_clusters_top_1,       "dR_DT_clusters_top_1/F");
    outputTree->Branch("dPhi_DT_clusters_top_1",       &dPhi_DT_clusters_top_1,       "dPhi_DT_clusters_top_1/F");

    outputTree->Branch("dR_DT_clusters_higgs",       &dR_DT_clusters_higgs,       "dR_DT_clusters_higgs/F");
    outputTree->Branch("dPhi_DT_clusters_higgs",       &dPhi_DT_clusters_higgs,       "dPhi_DT_clusters_higgs/F");

    outputTree->Branch("nCscRechitClusters", &nCscRechitClusters, "nCscRechitClusters/I");
    outputTree->Branch("nDtRechitClusters", &nDtRechitClusters, "nDtRechitClusters/I");

    outputTree->Branch("Top0",&top0);
    outputTree->Branch("Top1",&top1);
    outputTree->Branch("Higgs",&higgs);
    outputTree->Branch("Tau0",&tau0);
    outputTree->Branch("Tau1",&tau1);
    outputTree->Branch("Tau2",&tau2);
    outputTree->Branch("Tau3",&tau3);

    // Event loop

    //for(int ev = 0; ev < 10; ev++) {
    for(int ev = 0; ev < inputTree->GetEntriesFast(); ev++)
      {
	//Clear all the vectors
        tops.clear();
        higgses.clear();
        LLPs.clear();
	taus.clear();

        inputTree->GetEntry(ev);

	std::cout << "======== " << std::endl;
	std::cout << "Event " << ev << std::endl;
	std::cout << "======== " << std::endl;
	std::cout << "tau0 pt: " << top0.pt<< std::endl;
	
	int ntop = 0;
	int nhiggs = 0;
	int nllp = 0;
	int ntau = 0;
	for (int j=0; j<nGenParticle; j++)
	  {
	    if(abs(gParticleId[j]) == 6)
	      {
		ntop++;
		if(ntop<2)
		  {
		    //Fill top 0
		    top0.pt = gParticlePt[j];
		    top0.motherid = gParticleMotherId[j];
		    top0.pdgId = gParticleId[j];
		    top0.status = gParticleStatus[j];
		    top0.energy = gParticleE[j];
		    top0.eta = gParticleEta[j];
		    top0.phi = gParticlePhi[j];
		    top0.vx = gParticleProdVertexX[j];
		    top0.vy = gParticleProdVertexY[j];
		    top0.vz = gParticleProdVertexZ[j];
		    top0.travelX = gParticleDecayVertexX[j];
		    top0.travelY = gParticleDecayVertexY[j];
		    top0.travelZ = gParticleDecayVertexZ[j];
		  }
		else
		  {
		    //Fill top 1
		    top1.pt = gParticlePt[j];
		    top1.motherid = gParticleMotherId[j];
		    top1.pdgId = gParticleId[j];
		    top1.status = gParticleStatus[j];
		    top1.energy = gParticleE[j];
		    top1.eta = gParticleEta[j];
		    top1.phi = gParticlePhi[j];
		    top1.vx = gParticleProdVertexX[j];
		    top1.vy = gParticleProdVertexY[j];
		    top1.vz = gParticleProdVertexZ[j];
		    top1.travelX = gParticleDecayVertexX[j];
		    top1.travelY = gParticleDecayVertexY[j];
		    top1.travelZ = gParticleDecayVertexZ[j];
		  }
	      }

	    if(abs(gParticleId[j]) == 25 and gParticleStatus[j]==62)
	      {
		nhiggs++;
		if(nhiggs>0)
		  {
		    //Fill top 0
		    higgs.pt = gParticlePt[j];
		    higgs.motherid = gParticleMotherId[j];
		    higgs.pdgId = gParticleId[j];
		    higgs.status = gParticleStatus[j];
		    higgs.energy = gParticleE[j];
		    higgs.eta = gParticleEta[j];
		    higgs.phi = gParticlePhi[j];
		    higgs.vx = gParticleProdVertexX[j];
		    higgs.vy = gParticleProdVertexY[j];
		    higgs.vz = gParticleProdVertexZ[j];
		    higgs.travelX = gParticleDecayVertexX[j];
		    higgs.travelY = gParticleDecayVertexY[j];
		    higgs.travelZ = gParticleDecayVertexZ[j];
		  }			  
	      }
	  }

	//Loop for tau:different
	if(abs(gLLP_daughter_id[0]) == 15)
	      {
		//Fill tau 0
		tau0.pt = gLLP_daughter_pt[0];
		tau0.pdgId = gLLP_daughter_id[0];
		tau0.energy = gLLP_daughter_e[0];
		tau0.eta = gLLP_daughter_eta[0];
		tau0.phi = gLLP_daughter_phi[0];
	      }

	if(abs(gLLP_daughter_id[1]) == 15)
	      {
		//Fill tau 1
		tau1.pt = gLLP_daughter_pt[1];
		tau1.pdgId = gLLP_daughter_id[1];
		tau1.energy = gLLP_daughter_e[1];
		tau1.eta = gLLP_daughter_eta[1];
		tau1.phi = gLLP_daughter_phi[1];
	      }

	if(abs(gLLP_daughter_id[2]) == 15)
	      {
		//Fill tau 2
		tau2.pt = gLLP_daughter_pt[2];
		tau2.pdgId = gLLP_daughter_id[2];
		tau2.energy = gLLP_daughter_e[2];
		tau2.eta = gLLP_daughter_eta[2];
		tau2.phi = gLLP_daughter_phi[2];
	      }

	if(abs(gLLP_daughter_id[3]) == 15)
	      {
		//Fill tau 3
		tau3.pt = gLLP_daughter_pt[3];
		tau3.pdgId = gLLP_daughter_id[3];
		tau3.energy = gLLP_daughter_e[3];
		tau3.eta = gLLP_daughter_eta[3];
		tau3.phi = gLLP_daughter_phi[3];
	      }
	


	float tmp_dR_CSC_clusters_top_0(999.);
	float tmp_dR_CSC_clusters_top_1(999.);
	float tmp_dPhi_CSC_clusters_top_0(999.);
	float tmp_dPhi_CSC_clusters_top_1(999.);
	float tmp_dR_CSC_clusters_higgs(999.);
	float tmp_dPhi_CSC_clusters_higgs(999.);
	for (int j=0; j<nCscRechitClusters; j++)
	  {
	    
	    tmp_dR_CSC_clusters_top_0 = min(tmp_dR_CSC_clusters_top_0, reco::deltaR(top0.eta,top0.phi,cscRechitClusterEta[j],cscRechitClusterPhi[j]));
	    tmp_dPhi_CSC_clusters_top_0 = min(tmp_dPhi_CSC_clusters_top_0, reco::deltaPhi(top0.phi,cscRechitClusterPhi[j]));


	    tmp_dR_CSC_clusters_top_1 = min(tmp_dR_CSC_clusters_top_1, reco::deltaR(top1.eta,top1.phi,cscRechitClusterEta[j],cscRechitClusterPhi[j]));
	    tmp_dPhi_CSC_clusters_top_1 = min(tmp_dPhi_CSC_clusters_top_1, reco::deltaPhi(top1.phi,cscRechitClusterPhi[j]));


	    tmp_dR_CSC_clusters_higgs = min(tmp_dR_CSC_clusters_higgs, reco::deltaR(higgs.eta,higgs.phi,cscRechitClusterEta[j],cscRechitClusterPhi[j]));
	    tmp_dPhi_CSC_clusters_higgs = min(tmp_dPhi_CSC_clusters_higgs, reco::deltaPhi(higgs.phi,cscRechitClusterPhi[j]));
	  }

	dR_CSC_clusters_top_0 = tmp_dR_CSC_clusters_top_0;
	dR_CSC_clusters_top_1 = tmp_dR_CSC_clusters_top_1;
	dPhi_CSC_clusters_top_0 = tmp_dPhi_CSC_clusters_top_0;
	dPhi_CSC_clusters_top_1 = tmp_dPhi_CSC_clusters_top_1;

	dR_CSC_clusters_higgs = tmp_dR_CSC_clusters_higgs;
	dPhi_CSC_clusters_higgs = tmp_dPhi_CSC_clusters_higgs;



	float tmp_dR_DT_clusters_top_0(999.);
	float tmp_dR_DT_clusters_top_1(999.);
	float tmp_dPhi_DT_clusters_top_0(999.);
	float tmp_dPhi_DT_clusters_top_1(999.);
	float tmp_dR_DT_clusters_higgs(999.);
	float tmp_dPhi_DT_clusters_higgs(999.);
	for (int j=0; j<nDtRechitClusters; j++)
	  {
	    
	    tmp_dR_DT_clusters_top_0 = min(tmp_dR_DT_clusters_top_0, reco::deltaR(top0.eta,top0.phi,dtRechitClusterEta[j],dtRechitClusterPhi[j]));
	    tmp_dPhi_DT_clusters_top_0 = min(tmp_dPhi_DT_clusters_top_0, reco::deltaPhi(top0.phi,dtRechitClusterPhi[j]));


	    tmp_dR_DT_clusters_top_1 = min(tmp_dR_DT_clusters_top_1, reco::deltaR(top1.eta,top1.phi,dtRechitClusterEta[j],dtRechitClusterPhi[j]));
	    tmp_dPhi_DT_clusters_top_1 = min(tmp_dPhi_DT_clusters_top_1, reco::deltaPhi(top1.phi,dtRechitClusterPhi[j]));


	    tmp_dR_DT_clusters_higgs = min(tmp_dR_DT_clusters_higgs, reco::deltaR(higgs.eta,higgs.phi,dtRechitClusterEta[j],dtRechitClusterPhi[j]));
	    tmp_dPhi_DT_clusters_higgs = min(tmp_dPhi_DT_clusters_higgs, reco::deltaPhi(higgs.phi,dtRechitClusterPhi[j]));
	  }

	dR_DT_clusters_top_0 = tmp_dR_DT_clusters_top_0;
	dR_DT_clusters_top_1 = tmp_dR_DT_clusters_top_1;
	dPhi_DT_clusters_top_0 = tmp_dPhi_DT_clusters_top_0;
	dPhi_DT_clusters_top_1 = tmp_dPhi_DT_clusters_top_1;

	dR_DT_clusters_higgs = tmp_dR_DT_clusters_higgs;
	dPhi_DT_clusters_higgs = tmp_dPhi_DT_clusters_higgs;



	float dR_tau_top_0 = min( dR_tau_top_0, min( min( reco::deltaR(top0.eta,top0.phi,tau0.eta,tau0.phi), reco::deltaR(top0.eta,top0.phi,tau1.eta,tau1.phi)), min( reco::deltaR(top0.eta,top0.phi,tau2.eta,tau2.phi), reco::deltaR(top0.eta,top0.phi,tau3.eta,tau3.phi) )) ); 
	float dR_tau_top_1 = min( dR_tau_top_1, min( min( reco::deltaR(top1.eta,top1.phi,tau0.eta,tau0.phi), reco::deltaR(top1.eta,top1.phi,tau1.eta,tau1.phi)), min( reco::deltaR(top1.eta,top1.phi,tau2.eta,tau2.phi), reco::deltaR(top1.eta,top1.phi,tau3.eta,tau3.phi) )) );

	float dPhi_tau_top_0 = min( dPhi_tau_top_0, min( min( reco::deltaPhi(top0.phi,tau0.phi), reco::deltaPhi(top0.phi,tau1.phi)), min( reco::deltaPhi(top0.phi,tau2.phi), reco::deltaPhi(top0.phi,tau3.phi) )) );
	float dPhi_tau_top_1 = min( dPhi_tau_top_1, min( min( reco::deltaPhi(top1.phi,tau0.phi), reco::deltaPhi(top1.phi,tau1.phi)), min( reco::deltaPhi(top1.phi,tau2.phi), reco::deltaPhi(top1.phi,tau3.phi) )) );

	outputTree->Fill();

      }

    // finalize files
    outputTree->SetWeight(tree_weight);
    outputFile->Write();
    outputFile->Close();
    auto end = std::chrono::system_clock::now();//time!
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

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


