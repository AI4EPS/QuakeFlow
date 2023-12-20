#!/bin/bash
helm install quakeflow-redis --set auth.enabled=false oci://registry-1.docker.io/bitnamicharts/redis