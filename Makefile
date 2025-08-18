SHELL := /bin/bash
NS ?= xview
K  := kubectl -n $(NS)
KUSTOMIZE_DIR ?= k8s/dev

# ------- Image tags (override after pushing to a registry) -------
UI_IMG     ?= ui:dev
WORKER_IMG ?= worker:dev
INFER_IMG  ?= infer:dev

.PHONY: all build deploy redeploy undeploy down \
        wait ui infer url serve get desc logs \
        rollout-infer rollout-ui scale-ui scale-infer \
        job-preprocess job-train train-and-reload job-clean \
        hpa-on hpa-off tunnel nuke image-clean help

# ======= 0) One-shot happy path =======
all: minio build deploy clear-pv-claim wait url serve  ## Start minikube, build, deploy, wait, print URL

# ======= 1) Cluster/bootstrap =======
# skipped; done separately

# ======= 2) Build & Deploy =======

clear-pv-claim:
	kubectl patch pv minio-pv -p '{"spec":{"claimRef": null}}'

minio: ## Deploy MinIO
	kubectl apply -f k8s/base/minio-namespace.yaml
	kubectl apply -f k8s/base/minio-secret.yaml -n $(NS)
	kubectl apply -f k8s/base/minio-pvc.yaml -n $(NS)
	kubectl apply -f k8s/base/minio-deployment.yaml -n $(NS)
	kubectl apply -f k8s/base/minio-service.yaml -n $(NS)

build:                    ## Build all images inside Minikube's Docker
	@echo "Using Minikube Docker daemon so images are visible to the cluster"
	minikube image build -t localhost/$(UI_IMG)     images/ui
	minikube image build -t localhost/$(WORKER_IMG) images/worker
	minikube image build -t localhost/$(INFER_IMG)  images/infer

deploy:                   ## Apply all manifests (Kustomize overlay)
	- kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
	kubectl apply -k $(KUSTOMIZE_DIR)

redeploy: build deploy wait url  ## Rebuild images, reapply, and wait

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

# HELP
help:                     ## Show help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' -e 's/\\$$//' | awk 'BEGIN {FS = ": "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'