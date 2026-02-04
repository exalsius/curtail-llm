# Running the Flower app as a federation on exalsius Kubernetes

The Flower operator and federation integration with the exalsius stack are still **beta**. The steps below are partly manual; once the integration is stable, most of them (step 2-6) will be automated.

**Requirements:** `kubectl`, `helm`

---

## 1. Import cluster kubeconfig

```bash
exls clusters import-kubeconfig <cluster-name> --kubeconfig-path=derms.conf
```

## 2. Install the Flower operator

```bash
git clone https://github.com/exalsius/flower-operator
cd flower-operator
helm --kubeconfig derms.conf upgrade --install flower-operator ./dist/chart --namespace flower-operator --create-namespace
```

## 3. Install Redis

```bash
helm repo add pascaliske https://charts.pascaliske.dev
helm --kubeconfig derms.conf install redis pascaliske/redis
```

## 4. Create the federation

Set your Weights & Biases API key in `federation-derms.yaml` and adjust the `supernodes.pools` section if needed, then:

```bash
kubectl --kubeconfig derms.conf apply -f federation-derms.yaml
```

Wait until the federation is up: `kubectl get pods`, `kubectl get federation`.

## 5. Configure `pyproject.toml` with cluster endpoints

Get the NodePort for the superlink (port 9093):

```bash
kubectl --kubeconfig derms.conf get svc
```

Example output (use the NodePort for **9093**, e.g. `31731`):

```
NAME                         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                         AGE
federation-derms-superlink   NodePort    10.102.92.122   <none>        9092:32497/TCP,9093:31731/TCP   34m
kubernetes                   ClusterIP   10.96.0.1       <none>        443/TCP                         19d
redis                        ClusterIP   10.96.196.142   <none>        6379/TCP                        7h34m
redis-headless               ClusterIP   None            <none>        6379/TCP                        7h34m
vessim                       ClusterIP   10.104.118.90   <none>        8800/TCP                        51m
```

Update `pyproject.toml`:

| Section | Key | Value |
|--------|-----|--------|
| `[tool.flwr.federations.exls-federation]` | `address` | `<superlink-external-ip>:<NodePort>` (from the superlink service) |
| `[tool.flwr.app.config]` | `redis_url` | `redis://redis.default.svc.cluster.local:6379` |
| `[tool.flwr.app.config]` | `mci_api_url` | `http://vessim.default.svc.cluster.local:8800` |

## 6. Start Vessim in the cluster

```bash
kubectl --kubeconfig derms.conf apply -f vessim-deployment.yaml
```

## 7. Run the Flower federation

```bash
flwr run . exls-federation --stream
```

Logs can be checked with e.g. `kubectl --kubeconfig derms.conf logs -f federation-derms-supernode-default-0 -c superexec-clientapp`.
