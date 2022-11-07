# Application of TensorFlow interface in CMSSW to LLP NN Inference

A small example is included in folder ```nn_inference```: a keras model and a small tree with 100 events are provided.

### Setup

Install CMSSW release and cmsml package [https://cms-ml.github.io/documentation/inference/tensorflow2.html#saving-your-model](https://cms-ml.github.io/documentation/inference/tensorflow2.html#saving-your-model)

Note: SL7 is required

```bash
cmsrel CMSSW_11_1_3
cd CMSSW_11_1_3/src
cmsenv
git cms-init
scram b -j 32
cd $CMSSW_BASE/src
pip install --upgrade --user git+https://github.com/cms-ml/cmsml
```

Clone and compile this repo

```bash
mkdir NNInferenceCMSSW
cd NNInferenceCMSSW
git clone https://github.com/lbenato/LLP_NN_Inference.git
scram b -j 32
```

### Convert keras model to a constant graph
```bash
cd $CMSSW_BASE/src/NNInferenceCMSSW/LLP_NN_Inference
python python_macro/convert_model_to_graph.py
```

### Run NN inference
```bash
cd $CMSSW_BASE/src
../bin/slc7_amd64_gcc820/tf_test
```

### In case of issues with remote URL
https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories#switching-remote-urls-from-ssh-to-https

## Original repo: Marcel Rieger

## TensorFlow interface examples in CMSSW

[![build status](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples/badges/master/pipeline.svg)](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples](https://gitlab.cern.ch/mrieger/CMSSW-TensorFlowExamples)
- Code mirror: [github.com/riga/CMSSW-TensorFlowExamples](https://github.com/riga/CMSSW-TensorFlowExamples)

The TensorFlow interface itself is documented at [mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN).

It was merged under [PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow) on Jan 25 2018 into [CMSSW\_10\_1\_X](https://github.com/cms-sw/cmssw/pull/19893) and backported to [CMSSW\_9\_4\_X](https://github.com/cms-sw/cmssw/pull/22042) on Feb 15 2018.

---
