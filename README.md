<!-- <img src="https://raw.githubusercontent.com/wayneweiqiang/QuakeFlow/master/docs/assets/logo.png"> -->

# QuakeFlow: A Scalable Deep-learning-based Earthquake Monitoring Workflow with Cloud Computing
[![documentation](https://github.com/wayneweiqiang/QuakeFlow/actions/workflows/docs.yml/badge.svg)](https://wayneweiqiang.github.io/QuakeFlow/)

## Overview

![](https://raw.githubusercontent.com/wayneweiqiang/QuakeFlow/master/docs/assets/quakeflow_diagram.png)

QuakeFlow is a scalable deep-learning-based earthquake monitoring system with cloud computing. 
It applies the state-of-art deep learning/machine learning models for earthquake detection. 
With auto-scaling enabled on Kubernetes, our system can balance computational loads with computational resources. 

<!-- Checkout our Twitter Bot for realtime earthquake early warning at [@Quakeflow_Bot](https://twitter.com/QuakeFlow_bot). -->

## Current Modules 

### Models
- [DeepDenoiser](https://wayneweiqiang.github.io/DeepDenoiser/): [(paper)](https://arxiv.org/abs/1811.02695) [(example)](https://wayneweiqiang.github.io/DeepDenoiser/example_interactive/)
- [PhaseNet](https://wayneweiqiang.github.io/PhaseNet/): [(paper)](https://arxiv.org/abs/1803.03211) [(example)](https://wayneweiqiang.github.io/PhaseNet/example_interactive/)
- [GaMMA](https://wayneweiqiang.github.io/GaMMA/): [(paper)](https://arxiv.org/abs/2109.09008) [(example)](https://wayneweiqiang.github.io/GaMMA/example_interactive/)
- [HypoDD](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) [(paper)](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/90/6/1353/120565/A-Double-Difference-Earthquake-Location-Algorithm?redirectedFrom=fulltext) [(example)](https://github.com/wayneweiqiang/QuakeFlow/blob/master/HypoDD/gamma2hypodd.py)
- More models to be added. Contributions are highly welcomed!
  
### Data stream
- [Plotly](https://dash.gallery/Portal/): [ui.quakeflow.com](http://ui.quakeflow.com)
- [Kafka](https://www.confluent.io/what-is-apache-kafka/) 
- [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html)

### Data process
- [Colab example](https://colab.research.google.com/drive/19dC8-Vq0mv1Q9K-OS8VJf3xNEweKv4SN)
- [Kubeflow](https://www.kubeflow.org/): [(example)](https://wayneweiqiang.github.io/QuakeFlow/workflow/)

## Deployment

QuakeFlow can be deployed on any cloud platforms with Kubernetes service.

- For google cloud platform (GCP), check out the [GCP README](gcp_readme.md).
- For on-premise servers, check out the [Kubernetes README](k8s_readme.md).

<!-- ## User-Facing Platform

### Streamlit Web App

<img src="https://i.imgur.com/xL696Yh.jpg" width="800px">


### Twitter Bot

<img src="https://i.imgur.com/50kVK4Q.png" width="400px"> -->

