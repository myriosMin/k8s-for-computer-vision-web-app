SHELL := /bin/bash
NS ?= xview
K  := kubectl -n $(NS)
KUSTOMIZE_DIR ?= k8s/dev

# ------- Image tags (override after pushing to a registry) -------
UI_IMG     ?= ui:dev
WORKER_IMG ?= worker:dev
INFER_IMG  ?= infer:dev

# CPU-only image tags
UI_CPU_IMG     ?= ui:cpu
WORKER_CPU_IMG ?= worker:cpu
INFER_CPU_IMG  ?= infer:cpu

.PHONY: all build deploy redeploy undeploy down \
        wait ui infer url serve get desc logs \
        rollout-infer rollout-ui scale-ui scale-infer \
        job-preprocess job-train train-and-reload job-clean \
        hpa-on hpa-off tunnel nuke image-clean help \
        build-cpu deploy-cpu redeploy-cpu cpu-all

# ======= 0) One-shot happy path =======
all: build deploy wait url serve  ## Start minikube, build, deploy, wait, print URL

# CPU-only lightweight deployment for testing
cpu-all: build-cpu deploy-cpu-infrastructure prepare-data-cpu deploy-cpu-services wait url serve  ## Start minikube, build CPU images, deploy infrastructure, prepare data, deploy services, wait, print URL

# ======= 1) Cluster/bootstrap =======
# skipped; done separately

# ======= 2) Build & Deploy =======

build:                    ## Build all images inside Minikube's Docker
	@echo "Using Minikube Docker daemon so images are visible to the cluster"
	minikube image build -t localhost/$(UI_IMG)     images/ui
	minikube image build -t localhost/$(WORKER_IMG) images/worker
	minikube image build -t localhost/$(INFER_IMG)  images/infer

build-cpu:                ## Build CPU-only images inside Minikube's Docker (lightweight)
	@echo "Building CPU-only images for lightweight testing"
	minikube image build -t localhost/ui:cpu     images/ui-cpu
	minikube image build -t localhost/worker:cpu images/worker-cpu
	minikube image build -t localhost/infer:cpu  images/infer-cpu

build-enhanced:           ## Build enhanced images with MinIO and authentication support
	@echo "Building enhanced images with full features"
	minikube image build -t localhost/ui:enhanced     images/ui-enhanced
	minikube image build -t localhost/worker:enhanced images/worker-enhanced
	minikube image build -t localhost/infer:enhanced  images/infer-enhanced

deploy:                   ## Apply all manifests (Kustomize overlay)
	- kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
	kubectl apply -k $(KUSTOMIZE_DIR)

deploy-cpu:               ## Apply CPU-only manifests (lightweight for testing)
	- kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
	- kubectl delete job preprocess train -n $(NS) --ignore-not-found=true
	kubectl apply -k k8s/cpu

deploy-cpu-infrastructure: ## Deploy only storage and config first  
	- kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
	- kubectl delete job preprocess train -n $(NS) --ignore-not-found=true
	# Deploy infrastructure components first
	kubectl apply -f k8s/base/namespace.yaml
	kubectl apply -f k8s/base/configmap-app.yaml -f k8s/base/secret-app.yaml -f k8s/base/secret-minio.yaml
	kubectl apply -f k8s/base/pvc-datasets.yaml -f k8s/base/pvc-models.yaml -f k8s/base/pvc-outputs.yaml

deploy-cpu-services:      ## Deploy services and deployments after data preparation
	kubectl apply -k k8s/cpu

redeploy: build deploy wait url  ## Rebuild images, reapply, and wait

redeploy-cpu: build-cpu deploy-cpu wait url  ## Rebuild CPU images, reapply lightweight setup, and wait

undeploy:                 ## Delete all resources from overlay
	kubectl delete -k $(KUSTOMIZE_DIR) --ignore-not-found

down:                     ## Nuke namespace (keeps cluster running)
	kubectl delete ns $(NS) --ignore-not-found || true

# ======= 3) Post-deploy utilities =======
wait:                     ## Wait until UI & Infer are Ready
	$(K) rollout status deploy/ui --timeout=180s
	$(K) rollout status deploy/infer --timeout=180s

ui:                       ## Wait for UI and echo URL
	$(K) rollout status deploy/ui --timeout=180s
	@$(MAKE) url

infer:                    ## Wait for Infer
	$(K) rollout status deploy/infer --timeout=180s

url:                      ## Print ingress URL; need extra configurations to map domain to endpoint locally
	@echo "Open: http://app.localtest.me"
	@echo "Minikube IP: $$(minikube ip)"

serve:					  ## Serve front-end website 
	minikube service ui -n $(NS)

get:                      ## Get high-level objects
	$(K) get deploy,svc,ingress,hpa,pdb,pvc
	@echo ""
	$(K) get pods -o wide

desc/%:                   ## Describe any resource, e.g., make "desc/deploy/ui"
	$(K) describe $*

logs:                     ## Tail UI logs (default); override with name=pod-or-deploy
	@if [ -z "$$name" ]; then \
	  echo "Usage: make logs name=deploy/ui"; \
	else \
	  $(K) logs $$name -f; \
	fi

# ======= 4) Safe rollouts & scaling =======
rollout-ui:               ## Restart UI deployment and wait
	$(K) rollout restart deploy/ui
	$(K) rollout status  deploy/ui --timeout=180s

rollout-infer:            ## Restart Infer deployment and wait
	$(K) rollout restart deploy/infer
	$(K) rollout status  deploy/infer --timeout=300s

scale-ui:                 ## Scale UI (e.g., make scale-ui n=3)
	@if [ -z "$$n" ]; then echo "Usage: make scale-ui n=2"; exit 1; fi
	$(K) scale deploy/ui --replicas=$$n

scale-infer:              ## Scale Infer (e.g., make scale-infer n=3)
	@if [ -z "$$n" ]; then echo "Usage: make scale-infer n=2"; exit 1; fi
	$(K) scale deploy/infer --replicas=$$n

# ======= 5) Jobs workflow =======
# Create a temporary pod that mounts the datasets PVC, copy the zip into it, then delete the pod.
prepare-data: ## Copy data/aiad_data.zip into datasets-pvc
	@echo "⏳ Seeding datasets-pvc with aiad_data.zip..."
	kubectl run tmp-copy -n $(NS) --restart=Never --image=busybox -- sleep 3600 \
	  --overrides='{"spec":{"volumes":[{"name":"datasets","persistentVolumeClaim":{"claimName":"datasets-pvc"}}],"containers":[{"name":"copy","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"datasets","mountPath":"/data"}]}]}}'
	kubectl wait pod/tmp-copy -n $(NS) --for=condition=Ready --timeout=30s
	kubectl cp data/aiad_data.zip $(NS)/tmp-copy:/data/aiad_data.zip
	kubectl delete pod tmp-copy -n $(NS)
	@echo "datasets-pvc is seeded."

# CPU-specific lightweight data preparation (smaller dataset)
prepare-data-cpu: ## Copy initial model for CPU deployment
	@echo "⏳ Preparing models-pvc with initial best.pt for CPU deployment..."
	kubectl run tmp-copy -n $(NS) --restart=Never --image=busybox \
	  --overrides='{"spec":{"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"models-pvc"}}],"containers":[{"name":"copy","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}]}}' \
	  -- sleep 3600
	kubectl wait pod/tmp-copy -n $(NS) --for=condition=Ready --timeout=60s
	# Copy the trained model for inference
	kubectl cp models/best.pt $(NS)/tmp-copy:/models/best.pt
	kubectl delete pod tmp-copy -n $(NS)
	@echo "models-pvc is seeded with best.pt for inference."

# Initialize MinIO with buckets and demo data
init-minio: ## Initialize MinIO with required buckets and demo user
	@echo "⏳ Initializing MinIO..."
	$(K) wait --for=condition=ready pod -l app=minio --timeout=300s
	@echo "Creating MinIO user and buckets..."
	kubectl run minio-init -n $(NS) --restart=Never --image=minio/mc:latest --rm -i --tty -- sh -c '\
		mc alias set xview http://minio:9000 admin minio123 && \
		mc admin user add xview xview-app xview-app-secret-key && \
		mc admin policy attach xview readwrite --user xview-app && \
		mc mb xview/datasets xview/models xview/predictions xview/batch-uploads 2>/dev/null || true && \
		echo "MinIO initialized successfully"'

job-preprocess: prepare-data           ## Run preprocess Job and follow until complete
	$(K) delete job/preprocess --ignore-not-found
	$(K) apply -f k8s/base/job-preprocess.yaml
	$(K) wait --for=condition=complete job/preprocess --timeout=6h
	@echo "preprocess complete"

job-train:                ## Run train Job and follow until complete (publishes /models/best.pt)
	$(K) delete job/train --ignore-not-found
	$(K) create -f k8s/base/job-train.yaml
	$(K) wait --for=condition=complete job/train --timeout=48h
	@echo "train complete (best.pt published to /models/best.pt)"

train-and-reload: job-train rollout-infer ui  ## Train → publish best.pt → restart infer → check UI

job-clean:                ## Remove finished Jobs (TTL also handles this eventually)
	$(K) delete job/preprocess job/train --ignore-not-found || true

# ======= 6) HPAs (optional; only if you included the YAMLs) =======
hpa-on:                   ## Apply autoscalers (UI & Infer)
	-kubectl apply -f k8s/base/hpa-ui.yaml
	-kubectl apply -f k8s/base/hpa-infer.yaml
hpa-off:                  ## Remove autoscalers
	-kubectl delete -f k8s/base/hpa-ui.yaml   --ignore-not-found
	-kubectl delete -f k8s/base/hpa-infer.yaml --ignore-not-found

# ======= 7) Ingress tunneling (rarely needed with localtest.me) =======
tunnel:                   ## Run minikube tunnel (foreground)
	minikube tunnel

# ======= 8) Clean up resources after testing or to restart =======
nuke: ## Delete all deployments, jobs, PVCs, and namespace 
	@echo "!!! Nuking everything in namespace: $(NS)"
	kubectl delete job --all -n $(NS) --ignore-not-found
	kubectl delete -k $(KUSTOMIZE_DIR) --ignore-not-found
	kubectl delete pvc --all -n $(NS) --ignore-not-found
	kubectl delete ns $(NS) --ignore-not-found
	@echo " All resources in namespace '$(NS)' deleted."

image-clean: ## Delete built images from Minikube 
	@echo "!!! Deleting all images in namespace: $(NS)"
	minikube ssh -- docker rmi -f localhost/$(UI_IMG)     || true
	minikube ssh -- docker rmi -f localhost/$(WORKER_IMG) || true
	minikube ssh -- docker rmi -f localhost/$(INFER_IMG)  || true
	minikube ssh -- docker rmi -f localhost/$(UI_CPU_IMG)     || true
	minikube ssh -- docker rmi -f localhost/$(WORKER_CPU_IMG) || true
	minikube ssh -- docker rmi -f localhost/$(INFER_CPU_IMG)  || true

# HELP
help:                     ## Show help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' -e 's/\\$$//' | awk 'BEGIN {FS = ": "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'