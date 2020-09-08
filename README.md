# Application of TensorFlow interface in CMSSW to LLP NN Inference

### Setup

```bash
cmsrel CMSSW_11_1_3
cd CMSSW_11_1_3/src
cmsenv
git cms-init
scram b -j 32
cd $CMSSW_BASE/src
git clone https://github.com/lbenato/LLP_NN_Inference.git
scram b -j 32
```

### Run
```bash
cd $CMSSW_BASE/src
../bin/slc7_amd64_gcc820/tf_test
```

## Original repo: Marcel Rieger

## TensorFlow interface examples in CMSSW

[![build status](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples/badges/master/pipeline.svg)](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples)
- Code mirror: [github.com/riga/CMSSW-TensorFlowExamples](https://github.com/riga/CMSSW-TensorFlowExamples)

The TensorFlow interface itself is documented at [mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN).

It was merged under [PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow) on Jan 25 2018 into [CMSSW\_10\_1\_X](https://github.com/cms-sw/cmssw/pull/19893) and backported to [CMSSW\_9\_4\_X](https://github.com/cms-sw/cmssw/pull/22042) on Feb 15 2018.

---

### Setup

```bash
# in your CMSSW src directory
git clone https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples.git TensorFlowExamples
scram b
```

See the README files of the particular examples for more info.


### Examples

- [`GraphLoading`](./GraphLoading): Graph loading and evaluation in a CMSSW plugin, demonstrated with and without multi-threading.
- [`StandaloneBinary`](./StandaloneBinary): Standalone binary showing how to evaluate a TensorFlow model using data in a flat input tree.

