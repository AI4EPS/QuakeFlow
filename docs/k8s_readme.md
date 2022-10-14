# Quick readme, not detailed

## All-in-one script
You need to preinstall [helm](https://helm.sh/), [kubectl](https://kubernetes.io/docs/tasks/tools/), [docker](https://docs.docker.com/engine/install/) and [minikube](https://minikube.sigs.k8s.io/docs/start/) (or any other local Kubernetes framework)

Then deploy everything with the following script!

```
$ git clone --recurse-submodules -j8 git@github.com:wayneweiqiang/QuakeFlow.git
$ sh deploy_local.sh
```

## Prebuilt Kafka 

1. Install
```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install quakeflow-kafka bitnami/kafka   
```

2. Create topics
```
kubectl run --quiet=true -it --rm quakeflow-kafka-client --restart='Never' --image docker.io/bitnami/kafka:2.7.0-debian-10-r68 --restart=Never --command -- bash -c "kafka-topics.sh --create --topic phasenet_picks --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-topics.sh --create --topic gmma_events --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-topics.sh --create --topic waveform_raw --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092"
```

2. Check status
```
helm status quakeflow-kafka
```

## Our own containers

1. Switch to minikube environment
```
eval $(minikube docker-env)     
```

1.1. Fix metrics-server for auto-scalling (Only for docker)
https://stackoverflow.com/questions/54106725/docker-kubernetes-mac-autoscaler-unable-to-find-metrics

```
kubectl apply -f metrics-server.yaml
```

2. Build the docker images, see the docs for each container

```
docker build --tag quakeflow-spark:1.0 .
...
```

3. Create everything
```
kubectl apply -f quakeflow-delpoyment.yaml     
```

3.1 Add autoscaling
```
kubectl autoscale deployment phasenet-api --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment gmma-api --cpu-percent=80 --min=1 --max=10
```

3.2 Expose API
```
kubectl expose deployment phasenet-api --type=LoadBalancer --name=phasenet-service
```

4. Check the pods
```
kubectl get pods
```

5. Check the logs (an example)
```
kubectl logs quakeflow-spark-7699cd45d8-mvv6r
```

6. Delete a single deployment
```
kubectl delete deploy quakeflow-spark     
```

7. Delete everything
```
kubectl delete -f quakeflow-delpoyment.yaml   
```



