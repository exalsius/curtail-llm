import os
import subprocess
import time
from datetime import datetime
from logging import INFO

import exalsius_api_client as exls
from pilot.logger import log

from pilot.event_log import get_event_log

ExlsNodeId = str


class ExlsProvisioner:
    """Manages Exalsius cloud node provisioning."""

    def __init__(self, cluster_id: str, node_id_mapping: dict[str, ExlsNodeId]):
        self.cluster_id = cluster_id
        self.node_id_mapping = node_id_mapping

        config = exls.Configuration(host="https://api.exalsius.ai/v1")
        config.access_token = os.environ["EXALSIUS_TOKEN"]
        self._api_client = exls.ApiClient(config)
        self._clusters_api = exls.ClustersApi(self._api_client)

    def add_node(self, client_name: str):
        exls_node_id = self.node_id_mapping[client_name]
        self._clusters_api.add_nodes(self.cluster_id, exls.ClusterAddNodeRequest(
            nodes_to_add=[exls.ClusterNodeToAdd(node_id=exls_node_id, node_role="WORKER")]
        ))
        log(INFO, f"Provisioned '{client_name}' (Exalsius node {exls_node_id})")

    def remove_node(self, client_name: str):
        exls_node_id = self.node_id_mapping[client_name]
        self._clusters_api.delete_node_from_cluster(self.cluster_id, exls_node_id)
        log(INFO, f"Deprovisioned '{client_name}' (Exalsius node {exls_node_id})")


class SubprocessProvisioner:
    """Manages local Flower supernode subprocesses."""

    def __init__(self, superlink_address: str = "127.0.0.1:9092"):
        self.superlink_address = superlink_address
        self.current_round = 0
        self.processes = {}  # client_name -> (Popen, slot_id)
        # 8xA100 machine, 4 GPUs per client -> 2 slots
        self.gpu_slots = {
            0: {"gpus": "0,1,2,3", "busy": False},
            1: {"gpus": "4,5,6,7", "busy": False},
        }
        log(INFO, f"SubprocessProvisioner initialized with superlink at {self.superlink_address}")

    def add_node(self, client_name: str):
        if client_name in self.processes:
            log(INFO, f"Client '{client_name}' is already running.")
            return

        slot_id = next((k for k, v in self.gpu_slots.items() if not v["busy"]), None)
        if slot_id is None:
            raise RuntimeError(f"No free GPU slots to provision '{client_name}'!")

        self.gpu_slots[slot_id]["busy"] = True

        provisioning_delay = 240  # 4 minutes artificial overhead
        evt = get_event_log()
        if evt:
            evt.log("PROVISION_START", client=client_name)
        log(INFO, f"Simulating provisioning overhead for {client_name} ({provisioning_delay}s)...")
        time.sleep(provisioning_delay)
        log(INFO, f"Provisioning overhead complete for {client_name}, starting supernode.")

        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/{client_name}_r{self.current_round}_{timestamp}.log"
        log_file = open(log_file_path, "w")

        env = os.environ.copy()
        gpus = self.gpu_slots[slot_id]["gpus"]
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["NANOCHAT_BASE_DIR"] = "/workspace/cache/nanochat/"
        partition_id = int(client_name.split("_")[1])
        cmd = [
            "flower-supernode",
            "--insecure",
            "--superlink", self.superlink_address,
            "--clientappio-api-address", f"127.0.0.1:{9094 + partition_id}",
            "--node-config", f'partition-id={partition_id} name="{client_name}"',
        ]
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        self.processes[client_name] = (proc, slot_id, log_file)
        log(INFO, f"Provisioned {client_name} on GPUs {gpus} (PID {proc.pid}) -> {log_file_path}")
        if evt:
            evt.log("PROVISION_COMPLETE", client=client_name, details=f"pid={proc.pid} gpus={gpus}")

    def remove_node(self, client_name: str):
        if client_name not in self.processes:
            log(INFO, f"{client_name} not found to deprovision.")
            return

        proc, slot_id, log_file = self.processes[client_name]
        log(INFO, f"Deprovisioning {client_name} (PID {proc.pid})...")
        evt = get_event_log()
        if evt:
            evt.log("DEPROVISION", client=client_name)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        self.gpu_slots[slot_id]["busy"] = False
        del self.processes[client_name]
