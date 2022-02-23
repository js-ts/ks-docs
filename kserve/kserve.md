+++
title = "KServe"
description = "Model serving using KServe"
weight = 2
 
+++
 
{{% beta-status
  feedbacklink="https://github.com/KServe/KServe/issues" %}}
 
KServe enables serverless inferencing on Kubernetes and provides performant, high abstraction interfaces for common machine learning (ML) frameworks like TensorFlow, XGBoost, scikit-learn, PyTorch, and ONNX to solve production model serving use cases.
 
You can use KServe to do the following:
 
* Provide a Kubernetes [Custom Resource Definition](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) for serving ML models on arbitrary frameworks.
 
* Encapsulate the complexity of autoscaling, networking, health checking, and server configuration to bring cutting edge serving features like GPU autoscaling, scale to zero, and canary rollouts to your ML deployments.
 
* Enable a simple, pluggable, and complete story for your production ML inference server by providing prediction, pre-processing, post-processing and explainability out of the box.
 
Our strong community contributions help KServe to grow. We have a Technical Steering Committee driven by Bloomberg, IBM Cloud, Seldon, Amazon Web Services (AWS) and NVIDIA. [Browse the KServe GitHub repo](https://github.com/KServe/KServe) to give us feedback!
 
## Install with Kubeflow
 
KServe works with Kubeflow 1.5. Kustomize installation files are [located in the manifests repo](https://github.com/kubeflow/manifests/tree/master/apps/KServe/upstream).
Check the examples running KServe on Istio/Dex in the [`kubeflow/KServe`](https://github.com/KServe/KServe/tree/master/docs/samples/istio-dex) repository. For installation on major cloud providers with Kubeflow, follow their installation docs.
 
Kubeflow 1.5 includes KServe v0.7 which promoted the core InferenceService API from v1alpha2 to v1beta1 stable and added v1alpha1 version of Multi-Model Serving. Additionally, LFAI Trusted AI Projects on AI Fairness, AI Explainability and Adversarial Robustness have been integrated in KServe, and we have made KServe available on OpenShift as well. To know more, please read the [release blog](https://kserve.github.io/website/blog/articles/2021-10-11-KServe-0.7-release/) and follow the [release notes](https://github.com/KServe/KServe/releases/tag/v0.7.0)
 
<img src="./kserve.png" alt="KServe">
 
## Examples
 
### Deploy models with out-of-the-box model servers
 
* [TensorFlow](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/tensorflow)
* [PyTorch](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/torchserve)
* [XGBoost](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/xgboost)
* [Scikit-Learn](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/sklearn/v2)
* [ONNXRuntime](https://github.com/KServe/KServe/tree/master/docs/samples/v1alpha2/onnx)
 
### Deploy models with custom model servers
 
* [Custom](https://github.com/KServe/KServe/tree/master/docs/samples/v1alpha2/custom)
* [BentoML](https://github.com/KServe/KServe/tree/master/docs/samples/bentoml)
 
### Deploy models on GPU
 
* [GPU](https://github.com/KServe/KServe/tree/master/docs/samples/accelerators)
* [Nvidia Triton Inference Server](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/triton)
 
### Autoscaling and Rollouts
 
* [Autoscaling](https://github.com/KServe/KServe/tree/master/docs/samples/autoscaling)
* [Canary Rollout](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/rollout)
 
### Model explainability and outlier detection
 
* [Explainability](https://github.com/KServe/KServe/tree/master/docs/samples/explanation/alibi)
* [OutlierDetection](https://github.com/KServe/KServe/tree/master/docs/samples/outlier-detection/alibi-detect/cifar10)
 
### Integrations
 
* [Transformer](https://github.com/KServe/KServe/tree/master/docs/samples/v1beta1/transformer/torchserve_image_transformer)
* [Kafka](https://github.com/KServe/KServe/tree/master/docs/samples/kafka)
* [Pipelines](https://github.com/KServe/KServe/tree/master/docs/samples/pipelines)
* [Request/Response logging](https://github.com/KServe/KServe/tree/master/docs/samples/logger)
 
### Model Storages
 
* [Azure](https://github.com/KServe/KServe/tree/master/docs/samples/storage/azure)
* [S3](https://github.com/KServe/KServe/tree/master/docs/samples/storage/s3)
* [On-prem cluster](https://github.com/KServe/KServe/tree/master/docs/samples/storage/pvc)
 
### Sample notebooks
 
* [SDK client](https://github.com/KServe/KServe/blob/master/docs/samples/client/kfserving_sdk_v1beta1_sample.ipynb)
* [Transformer (pre/post processing)](https://github.com/KServe/KServe/blob/master/docs/samples/v1alpha2/transformer/image_transformer/kfserving_sdk_transformer.ipynb)
* [ONNX](https://github.com/KServe/KServe/blob/master/docs/samples/v1alpha2/onnx/mosaic-onnx.ipynb)
 
We frequently add examples to our [GitHub repo](https://github.com/KServe/KServe/tree/master/docs/samples/).
 
## Learn more
 
* Join our [working group](https://groups.google.com/forum/#!forum/kfserving) for meeting invitations and discussion.
* [Read the docs](https://github.com/kubeflow/kfserving/tree/master/docs).
* [API docs](https://github.com/kubeflow/kfserving/tree/master/docs/apis/v1beta1/README.md).
* [Debugging guide](https://github.com/kubeflow/kfserving/blob/master/docs/KFSERVING_DEBUG_GUIDE.md).
* [Roadmap](https://github.com/kubeflow/kfserving/tree/master/ROADMAP.md).
* [KFServing 101 slides](https://drive.google.com/file/d/16oqz6dhY5BR0u74pi9mDThU97Np__AFb/view).
* [Kubecon Introducing KFServing](https://kccncna19.sched.com/event/UaZo/introducing-kfserving-serverless-model-serving-on-kubernetes-ellis-bigelow-google-dan-sun-bloomberg).
* [Kubecon Advanced KFServing](https://kccncna19.sched.com/event/UaVw/advanced-model-inferencing-leveraging-knative-istio-and-kubeflow-serving-animesh-singh-ibm-clive-cox-seldon).
* [Nvidia GTC Accelerate and Autoscale Deep Learning Inference on GPUs](https://developer.nvidia.com/gtc/2020/video/s22459-vid).
* [Hands-on serving models using KFserving video](https://youtu.be/VtZ9LWyJPdc) and [slides](https://www.slideshare.net/theofpa/serving-models-using-kfserving).
| [KF Community: KFServing - Enabling Serverless Workloads Across Model Frameworks](https://www.youtube.com/watch?v=hGIvlFADMhU) |Ellis Tarn|
| [KubeflowDojo: Demo - KFServing End to End through Notebook](https://www.youtube.com/watch?v=xg5ar6vSAXY) |Animesh Singh, Tommy Li|
| [KubeflowDojo: Demo - KFServing with Kafka and Kubeflow Pipelines](https://www.youtube.com/watch?v=sVs6gFUddII) |Animesh Singh|
| [Anchor MLOps Podcast: Serving Models with KFServing](https://anchor.fm/mlops/episodes/MLOps-Coffee-Sessions-1-Serving-Models-with-Kubeflow-efbht0) | David Aponte, Demetrios Brinkmann|
| [Kubeflow 101: What is KFserving?](https://www.youtube.com/watch?v=lj_X2ND2BBI) | Stephanie Wong |
| [ICML 2020, Workshop on Challenges in Deploying and Monitoring Machine Learning Systems : Serverless inferencing on Kubernetes](https://slideslive.com/38931706/serverless-inferencing-on-kubernetes?ref=account-folder-55868-folders) | Clive Cox |
| [Serverless Practitioners Summit 2020: Serverless Machine Learning Inference with KFServing](https://www.youtube.com/watch?v=HlKOOgY5OyA) | Clive Cox, Yuzhui Liu|