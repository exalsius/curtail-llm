import os
from logging import INFO

import exalsius_api_client as exls
from flwr.common import log

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

    def get_node_id(self, client_name: str) -> ExlsNodeId:
        return self.node_id_mapping[client_name]

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
