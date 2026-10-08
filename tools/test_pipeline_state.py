import torch
from pipeline_state import *

B = 3

x = torch.arange(6 * 4).reshape(6, 4)
parts = split_cfg_tensor(x, B)
assert torch.equal(merge_cfg_tensor(parts), x)

reordered = merge_cfg_tensor([parts[2], parts[0]])
expected = torch.cat((x[[2, 0]], x[[5, 3]]), dim=0)
assert torch.equal(reordered, expected)

lengths = [2, 3, 1, 4, 2, 5]
cu = torch.tensor([0, 2, 5, 6, 10, 12, 17], dtype=torch.int32)
kv = torch.arange(17 * 4).reshape(17, 4)

ca = (kv, cu, max(lengths))
ca_parts = split_ca_kv(ca, B)
merged = merge_ca_kv(ca_parts)

assert torch.equal(merged[0], kv)
assert torch.equal(merged[1], cu)

reordered_ca = merge_ca_kv([ca_parts[2], ca_parts[0]])

print("CFG_TENSOR_TEST_OK")
print("CA_KV_TEST_OK")
print("PIPELINE_STATE_HELPERS_OK")
print("reordered_cfg_shape =", tuple(reordered.shape))
print("reordered_ca_kv_shape =", tuple(reordered_ca[0].shape))
print("reordered_ca_cu =", reordered_ca[1].tolist())
