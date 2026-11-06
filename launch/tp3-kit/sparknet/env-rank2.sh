# One-shot routing env for rank 2 (worker B) — EDIT for your fabric (see README.md).
# Find each HCA name with `ibdev2netdev` on the box; every EDIT_ME below must
# be replaced. The peer->local-HCA map and HCA->subnet map must match your
# rendered node map (`sparknet topology render nodes.json <node> ...`).
export GLOO_SOCKET_IFNAME=EDIT_ME_MGMT_IF
export NCCL_SOCKET_IFNAME=EDIT_ME_MGMT_IF
export TP_SOCKET_IFNAME=EDIT_ME_MGMT_IF
export NCCL_IB_HCA="=EDIT_ME_HCA_A,EDIT_ME_HCA_B"
export SPARKNET_ROCE_PEER_HCAS='{"0":["EDIT_ME_HCA_TO_RANK0"],"1":["EDIT_ME_HCA_TO_RANK1"]}'
export SPARKNET_ROCE_SUBNETS='{"EDIT_ME_HCA_TO_RANK0":"10.10.2.0/24","EDIT_ME_HCA_TO_RANK1":"10.10.0.0/24"}'
export SPARKNET_ROCE_TOPOLOGY=direct
# GID index 2: the IPv4 RoCE v2 GID sits at index 2 on every measured box;
# some firmware has index 3 all-zero (see README.md before changing).
export SPARKNET_ROCE_GID_INDEX=2
export SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES=2MB
export SPARKNET_ROCE_ALLGATHER_MAX_BYTES=4MB
