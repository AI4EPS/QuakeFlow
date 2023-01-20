# Deploy Kafka with Helm, create client and add topics
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install quakeflow-kafka bitnami/kafka
kubectl run --quiet=true -it --rm quakeflow-kafka-client --restart='Never' --image docker.io/bitnami/kafka:2.7.0-debian-10-r68 --restart=Never \
    --command -- bash -c "kafka-topics.sh --create --topic phasenet_picks --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-topics.sh --create --topic gmma_events --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-topics.sh --create --topic waveform_raw --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-topics.sh --create --topic phasenet_waveform --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092"
kubectl run --quiet=true -it --rm quakeflow-kafka-client --restart='Never' --image docker.io/bitnami/kafka:2.7.0-debian-10-r68 --restart=Never \
    --command -- bash -c "kafka-configs.sh --alter --entity-type topics --entity-name phasenet_picks --add-config 'retention.ms=-1' --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092 && kafka-configs.sh --alter --entity-type topics --entity-name gmma_events --add-config 'retention.ms=-1' --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092"
## For external access:
# helm upgrade quakeflow-kafka bitnami/kafka --set externalAccess.enabled=true,externalAccess.autoDiscovery.enabled=true,rbac.create=true
## Check topic configs:
# kubectl run --quiet=true -it --rm quakeflow-kafka-client --restart='Never' --image docker.io/bitnami/kafka:2.7.0-debian-10-r68 --restart=Never \
#     --command -- bash -c "kafka-topics.sh --describe --topics-with-overrides --bootstrap-server quakeflow-kafka.default.svc.cluster.local:9092"

# Deploy MongoDB
helm install quakeflow-mongodb --set auth.rootPassword=quakeflow123,auth.username=quakeflow,auth.password=quakeflow123,auth.database=quakeflow,architecture=replicaset,persistence.size=100Gi
 bitnami/mongodb

# Deploy to Kubernetes
kubectl apply -f quakeflow-gcp.yaml

# Add autoscaling
kubectl autoscale deployment phasenet-api --cpu-percent=50 --min=1 --max=365
kubectl autoscale deployment gamma-api --cpu-percent=200 --min=1 --max=365
kubectl autoscale deployment deepdenoiser-api --cpu-percent=50 --min=1 --max=10

# Expose APIs
# kubectl expose deployment phasenet-api --type=LoadBalancer --name=phasenet-service
# kubectl expose deployment gamma-api --type=LoadBalancer --name=gmma-service
# kubectl expose deployment quakeflow-streamlit --type=LoadBalancer --name=streamlit-ui
# kubectl expose deployment quakeflow-ui --type=LoadBalancer --name=quakeflow-ui

# Add MINIO storage
# helm install quakeflow-minio --set accessKey.password=minio --set secretKey.password=minio123 --set persistence.size=1T  bitnami/minio
