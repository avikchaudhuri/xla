import logging
import os
import unittest

import torch
from torch import nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from torch_xla._internal import tpu

if xr.device_type() == 'TPU':
  from torch_xla.experimental.custom_kernel import jax_import_guard
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  from jax.experimental import pallas as pl


class PallasTest(unittest.TestCase):

  def _attention(self, q, k, v):
    attn_weight = q @ k.transpose(-2, -1)
    attn_weight = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ v
    return attn_output

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_add(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, y_ref, o_ref):
    #   x, y = x_ref[...], y_ref[...]
    #   o_ref[...] = x + y
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA2lNDQFLBw8LEw8PDwsPMwsLCwtlCwsLCwsPCw8PEwsTDwsTDwsPDxMLDwUDYQENGwcTDxsPAsICHx0rLQUXAwMnKRURNx1HSRELAQUZHTM1AwsVFxkbHw0hDSMlBRsBAQUdDQlhZmZpbmVfbWFwPChkMCkgLT4gKGQwKT4ABR8FIQUjBSUFJxEDAQUpFS8JHQ8xFwUTAQUrFwUdAR05OwUtFwUlAR0/QQUvFUMJHQ9FFwUVAQUxFREJI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AF0sDIQcdAycDIQcBAgIFBwEBAQEBAgQEpwUBEAEHAwEFAxEBEwcDFScHAQEBAQEBBwMDBwMDCwYDAwUFAQcHAwMHAwMLBgMDBQUDCwkGPQMFBQkNBwMLBwMDCwYLAwUFBRENBAsHDwURBQABBgMBBQEAdgcz2wsTGdkNCxMjIR0pJ0MNCwsTDw8PDQkLEWJ1aWx0aW4AZnVuYwB0cHUAYXJpdGgAdmVjdG9yAG1vZHVsZQByZXR1cm4AY29uc3RhbnQAYWRkaQBsb2FkAHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AGFkZF92ZWN0b3JzX2tlcm5lbABkaW1lbnNpb25fc2VtYW50aWNzAGZ1bmN0aW9uX3R5cGUAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAc3ltX25hbWUAbWFpbgB2YWx1ZQAvZ2V0W3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQBhZGRfdmVjdG9ycwA8bW9kdWxlPgAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\", \"needs_layout_passes\": true}}"

    x = torch.arange(8, dtype=torch.int).to("xla")
    y = torch.arange(8, dtype=torch.int).to("xla")
    expected_output = x + y

    output = torch_xla._XLAC._xla_tpu_custom_call([x, y], payload, [x.shape],
                                                  [x.dtype])
    self.assertTrue(torch.allclose(output[0].cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_add_one(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, o_ref):
    #   o_ref[...] = x_ref[...] + 1
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAEtCwEDBQcJAQMLAwUDDQcFDxEJBxMVFwNlSQ0BRwcPCw8PDxMLDzMLCwsLZQsLCwsPCw8LEw8PCxMPCxMTDwsLBQNhAQ0bDxMHFw8CpgIfFSsxBRkdQwMdRQMRCwEDAw8nBRsdKQMDCxUXGRsfCyELIyUFHQEBBR8NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFIQUjBSUFJxEHAQUpHS0vBSsXBRsBFTM5HTU3BS0XBS8BHTs9BS8XBUUBAwMPQREDBQUxBTMjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAXRwMhAx0BAgInAyEDAwUFAQEBAQIEBKEFARABBwMBBQMRARMHAxMnBQEBAQEHAxENAwcLBhEDBQUBBQcDBz8DAw0GBwMFAwkJBgcDBQUHCwcDCQ0DBwsGCQMFBQMPDwQJBw0DDwUAAQYDAQUBAMIHNdsLEyEv2QsTIyEdKQ1DDRULCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAYnJvYWRjYXN0AHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AHZhbHVlAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBzeW1fbmFtZQBtYWluAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAGFkZF9vbmVfdmVjdG9yc19rZXJuZWwAYWRkX3ZlY3RvcnNfb25lADxtb2R1bGU+AC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAA==\", \"needs_layout_passes\": true}}"

    x = torch.arange(8, dtype=torch.int).to("xla")
    expected_output = x + 1

    output = torch_xla._XLAC._xla_tpu_custom_call([x], payload, [x.shape],
                                                  [x.dtype])
    self.assertTrue(torch.allclose(output[0].cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_raise(self):
    # This payload is generated by the following Pallas code:
    # def add_vectors_kernel(x_ref, o_ref):
    #   o_ref[...] = x_ref[...] + 1
    payload = "{\"custom_call_config\": {\"body\": \"TUzvUgFNTElSMTguMC4wZ2l0AAEtCwEDBQcJAQMLAwUDDQcFDxEJBxMVFwNlSQ0BRwcPCw8PDxMLDzMLCwsLZQsLCwsPCw8LEw8PCxMPCxMTDwsLBQNhAQ0bDxMHFw8CpgIfFSsxBRkdQwMdRQMRCwEDAw8nBRsdKQMDCxUXGRsfCyELIyUFHQEBBR8NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFIQUjBSUFJxEHAQUpHS0vBSsXBRsBFTM5HTU3BS0XBS8BHTs9BS8XBUUBAwMPQREDBQUxBTMjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAXRwMhAx0BAgInAyEDAwUFAQEBAQIEBKEFARABBwMBBQMRARMHAxMnBQEBAQEHAxENAwcLBhEDBQUBBQcDBz8DAw0GBwMFAwkJBgcDBQUHCwcDCQ0DBwsGCQMFBQMPDwQJBw0DDwUAAQYDAQUBAMIHNdsLEyEv2QsTIyEdKQ1DDRULCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAYnJvYWRjYXN0AHN0b3JlAC9ob21lL2p3dGFuL3BhbGxhcy9wYWxsYXNfYWRkLnB5AHZhbHVlAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBzeW1fbmFtZQBtYWluAC9nZXRbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAGFkZF9vbmVfdmVjdG9yc19rZXJuZWwAYWRkX3ZlY3RvcnNfb25lADxtb2R1bGU+AC9hZGQAL3N3YXBbdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhQeVRyZWVEZWYoKEN1c3RvbU5vZGUoU2xpY2VbKDAsIDgpXSwgW10pLCkpLCAoOCwpLCAoKSldLCBbXSksKSldAA==\", \"needs_layout_passes\": true}}"

    # _xla_tpu_custom_call requires at least one input.
    with self.assertRaises(RuntimeError):
      torch_xla._XLAC._xla_tpu_custom_call([], payload, [(8, 1)], [torch.int32])
      output.cpu()

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_flash_attention(self):
    # This payload is generated by the following Pallas code:
    # https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
    # To be noted, set `jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)`` before generating the payload.
    payload = "{\"custom_call_config\": {\"body\": \"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eC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AF9mbGFzaF9hdHRlbnRpb25fa2VybmVsX3NpbmdsZV9iYXRjaF9zaW5nbGVfc3RlcAB2YWx1ZQBmdW5jdGlvbl90eXBlAHN5bV9uYW1lAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwgKiwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgNCldLCBbXSkpKSwgKDEsIDEsIDEyOCwgNCksICgpKV0sIFsqLCAqXSksKSldAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHRyYW5zZm9ybV8zAHByZWNpc2lvbgB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAa2luZAByZWR1Y3Rpb25fZGltcwAvYnJvYWRjYXN0X2luX2RpbVtzaGFwZT0oMTI4LCAxKSBicm9hZGNhc3RfZGltZW5zaW9ucz0oMCwpXQBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAF9mbGFzaF9hdHRlbnRpb25fa2VybmVsAF9mbGFzaF9hdHRlbnRpb25faW1wbABfZmxhc2hfYXR0ZW50aW9uAGZsYXNoX2F0dGVudGlvbgA8bW9kdWxlPgAvbW50L2Rpc2tzL3NzZC93b3JrL3BhbGxhcy9wYWxsYXNfYWRkLnB5AC9kb3RfZ2VuZXJhbFtkaW1lbnNpb25fbnVtYmVycz0oKCgxLCksICgxLCkpLCAoKCksICgpKSkgcHJlY2lzaW9uPSg8UHJlY2lzaW9uLkhJR0hFU1Q6IDI+LCA8UHJlY2lzaW9uLkhJR0hFU1Q6IDI+KSBwcmVmZXJyZWRfZWxlbWVudF90eXBlPWZsb2F0MzJdAC9yZWR1Y2VfbWF4W2F4ZXM9KDEsKV0AL3N1YgBmYXN0bWF0aAAvZXhwAC9yZWR1Y2Vfc3VtW2F4ZXM9KDEsKV0AL2RpdgAvZG90X2dlbmVyYWxbZGltZW5zaW9uX251bWJlcnM9KCgoMSwpLCAoMCwpKSwgKCgpLCAoKSkpIHByZWNpc2lvbj0oPFByZWNpc2lvbi5ISUdIRVNUOiAyPiwgPFByZWNpc2lvbi5ISUdIRVNUOiAyPikgcHJlZmVycmVkX2VsZW1lbnRfdHlwZT1mbG9hdDMyXQAvc3dhcFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoKiwgKiwgQ3VzdG9tTm9kZShTbGljZVsoMCwgMTI4KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoMCwgNCldLCBbXSkpKSwgKDEsIDEsIDEyOCwgNCksICgpKV0sIFsqLCAqXSksKSldAA==\", \"needs_layout_passes\": true}}"

    # The division is to cause potential precision issue on TPU.
    q_mini = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4) / 13
    k_mini = torch.arange(
        1000, 1000 + 128 * 4, dtype=torch.float32).reshape(128, 4) / 13
    q = q_mini.broadcast_to(3, 2, 128, 4).to("xla")
    k = k_mini.broadcast_to(3, 2, 128, 4).to("xla")
    v = torch.ones(3, 2, 128, 4).to("xla")

    expected_o = self._attention(q, k, v)

    o = torch_xla._XLAC._xla_tpu_custom_call([q, k, v], payload, [q.shape],
                                             [q.dtype])
    self.assertTrue(torch.allclose(o[0].cpu(), expected_o.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_extract_add_payload(self):
    import jax._src.pallas.mosaic.pallas_call_registration

    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
                                                             x.dtype))(x, y)

    import torch_xla.experimental.custom_kernel as custom_kernel

    ir = jax.jit(add_vectors).lower(jnp.arange(8), jnp.arange(8)).compiler_ir()
    payload = custom_kernel._extract_backend_config(ir)
    # The payload being generated could vary each time. We just want to make sure
    # the most important fields are present.
    self.assertIn("custom_call_config", payload)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_tpu_custom_call_pallas_wrap_add_payload(self):

    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape,
                                                             x.dtype))(x, y)

    from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
    pt_kernel = make_kernel_from_pallas(add_vectors,
                                        lambda x, y: [(x.shape, x.dtype)])

    dtypes = [
        torch.float32, torch.float
    ]  # Add doesn't support torch.float64, torch.bfloat16, torch.float16.
    for i in range(len(dtypes)):
      x = torch.randn((i + 1, i + 1), dtype=dtypes[i]).to("xla")
      y = torch.randn((i + 1, i + 1), dtype=dtypes[i]).to("xla")
      expected_output = x + y
      output = pt_kernel(x, y)
      self.assertTrue(torch.allclose(output.cpu(), expected_output.cpu()))

    dtypes = [
        torch.int32, torch.int
    ]  # Add doesn't support torch.int64, torch.int16, torch.int8, torch.uint8.
    for i in range(len(dtypes)):
      x = torch.arange(i + 1, dtype=dtypes[i]).to("xla")
      y = torch.arange(i + 1, dtype=dtypes[i]).to("xla")
      expected_output = x + y
      output = pt_kernel(x, y)
      self.assertTrue(torch.allclose(output.cpu(), expected_output.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_tpu_custom_call_pallas_wrap_flash_attention(self):
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
    from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
    flash_attention_kernel = make_kernel_from_pallas(
        flash_attention, lambda q, k, v: [(q.shape, q.dtype)])

    q_mini = torch.arange(128 * 4, dtype=torch.bfloat16).reshape(128, 4) / 13
    k_mini = torch.arange(
        1000, 1000 + 128 * 4, dtype=torch.bfloat16).reshape(128, 4) / 13
    q = q_mini.broadcast_to(3, 2, 128, 4).to("xla")
    k = k_mini.broadcast_to(3, 2, 128, 4).to("xla")
    v = torch.ones(3, 2, 128, 4, dtype=torch.bfloat16).to("xla")

    o = flash_attention_kernel(q, k, v)
    expected_o = self._attention(q, k, v)
    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_wrapper(self):
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    from torch_xla.experimental.custom_kernel import flash_attention

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")

    o = flash_attention(q, k, v)
    expected_o = self._attention(q, k, v)
    self.assertTrue(torch.allclose(o.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_wrapper_with_dynamo(self):
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    from torch_xla.experimental.custom_kernel import flash_attention

    def flash_attention_wrapper(q, k, v, causal=False):
      return torch.ops.xla.flash_attention(q, k, v, causal)

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")

    compiled_flash_attention = torch.compile(
        flash_attention_wrapper, backend="openxla")
    o_no_causal = compiled_flash_attention(q, k, v)
    o_with_causal = compiled_flash_attention(q, k, v, causal=True)
    expected_o = self._attention(q, k, v)
    self.assertTrue(torch.allclose(o_no_causal.cpu(), expected_o.cpu()))
    # The causal mask is turned on by default in the wrapper.
    # It masks out the top right triangle of the attention matrix,
    # therefore it speeds up the compute but also changes the output.
    self.assertFalse(
        torch.allclose(o_with_causal.cpu(), expected_o.cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_wrapper_causal(self):
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    from torch_xla.experimental.custom_kernel import flash_attention

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")

    # The causal mask is turned on by default in the wrapper.
    # It masks out the top right triangle of the attention matrix, therefore it speeds up the compute but also changes the output.
    o = flash_attention(q, k, v, causal=True)
    expected_o = self._attention(q, k, v)
    self.assertFalse(torch.allclose(o.cpu(), expected_o.cpu()))
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  @unittest.mock.patch.dict(os.environ, {"XLA_USE_BF16": "1"})
  def test_flash_attention_wrapper_bf16(self):
    from torch_xla.experimental.custom_kernel import flash_attention

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")

    # No exception being raised.
    o = flash_attention(q, k, v)

  @unittest.skipIf(xr.device_type() != 'TPU', "This test only works on TPU.")
  def test_multiple_returns(self):
    import jax._src.pallas.mosaic.pallas_call_registration

    def add_minus_vectors_kernel(x_ref, y_ref, o1_ref, o2_ref):
      x, y = x_ref[...], y_ref[...]
      o1_ref[...] = x + y
      o2_ref[...] = x - y

    @jax.jit
    def add_minus_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
      return pl.pallas_call(
          add_minus_vectors_kernel, out_shape=[out_shape, out_shape])(x, y)

    from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
    pt_kernel = make_kernel_from_pallas(
        add_minus_vectors, lambda x, y: [(x.shape, x.dtype),
                                         (x.shape, x.dtype)])
    x = torch.arange(8, device="xla", dtype=torch.float)
    o = pt_kernel(x, x)
    self.assertEqual(len(o), 2)

    expected_o0 = x + x
    expected_o1 = x - x
    self.assertTrue(torch.allclose(o[0].cpu(), expected_o0.cpu()))
    self.assertTrue(torch.allclose(o[1].cpu(), expected_o1.cpu()))

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test__flash_attention_impl(self):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl
    from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
    MIN_BLOCK_SIZE = 128

    def shape_dtype(q, *arg):
      res_shape = list(q.shape)
      res_shape[-1] = MIN_BLOCK_SIZE
      return [(q.shape, q.dtype), (res_shape, torch.float32),
              (res_shape, torch.float32)]

    flash_attention_kernel = make_kernel_from_pallas(_flash_attention_impl,
                                                     shape_dtype)

    q = torch.randn(3, 2, 128, 4, dtype=torch.bfloat16).to("xla")
    k = torch.randn(3, 2, 128, 4, dtype=torch.bfloat16).to("xla")
    v = torch.randn(3, 2, 128, 4, dtype=torch.bfloat16).to("xla")

    o, l, m = flash_attention_kernel(
        q,
        k,
        v,
        None,
        None,
        True,
        False,
        1.0,
        2,
        128,
        128,
        128,
        False,
        static_argnums=range(5, 13))
    xm.mark_step()

    # TODO: I don't really know how to test the value. Let's do the shape check for now.
    self.assertEqual(l.shape, (3, 2, 128, MIN_BLOCK_SIZE))
    self.assertEqual(l.dtype, torch.float32)
    self.assertEqual(m.shape, (3, 2, 128, MIN_BLOCK_SIZE))
    self.assertEqual(m.dtype, torch.float32)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test__flash_attention_bwd_dkv(self):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dkv
    from torch_xla.experimental.custom_kernel import trace_pallas
    MIN_BLOCK_SIZE = 128
    DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")
    l = torch.randn(3, 2, 128).to("xla")
    m = torch.randn(3, 2, 128).to("xla")
    grad_i = torch.randn(3, 2, 128, dtype=torch.float32).to("xla")
    grad_o = torch.randn(3, 2, 128, 4).to("xla")

    payload, _ = trace_pallas(
        _flash_attention_bwd_dkv,
        q,
        k,
        v,
        None,
        None,
        l,
        m,
        grad_o,
        grad_i,
        block_q_major=128,
        block_k_major=128,
        block_k=128,
        block_q=128,
        sm_scale=1.0,
        causal=False,
        mask_value=DEFAULT_MASK_VALUE,
        debug=False,
        static_argnames=[
            "block_q_major", "block_k_major", "block_k", "block_q", "sm_scale",
            "causal", "mask_value", "debug"
        ])

    # TODO: Because of the following reshapes, we can't use make_kernel_from_pallas directly.
    l = l.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    m = m.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    grad_i = grad_i.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    output = torch_xla._XLAC._xla_tpu_custom_call(
        [q, k, v, l, m, grad_o, grad_i], payload, [k.shape, v.shape],
        [k.dtype, v.dtype])

    xm.mark_step()

    # TODO: I don't really know how to test the value. Let's do the shape check for now.
    self.assertEqual(output[0].shape, (3, 2, 128, 4))
    self.assertEqual(output[1].shape, (3, 2, 128, 4))

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test__flash_attention_bwd_dkv(self):
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq
    from torch_xla.experimental.custom_kernel import trace_pallas
    MIN_BLOCK_SIZE = 128
    DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")
    l = torch.randn(3, 2, 128).to("xla")
    m = torch.randn(3, 2, 128).to("xla")
    grad_i = torch.randn(3, 2, 128, dtype=torch.float32).to("xla")
    grad_o = torch.randn(3, 2, 128, 4).to("xla")

    payload, _ = trace_pallas(
        _flash_attention_bwd_dq,
        q,
        k,
        v,
        None,
        None,
        l,
        m,
        grad_o,
        grad_i,
        block_q_major=128,
        block_k_major=128,
        block_k=128,
        sm_scale=1.0,
        causal=False,
        mask_value=DEFAULT_MASK_VALUE,
        debug=False,
        static_argnames=[
            "block_q_major", "block_k_major", "block_k", "sm_scale", "causal",
            "mask_value", "debug"
        ])

    # TODO: Because of the following reshapes, we can't use make_kernel_from_pallas directly.
    l = l.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    m = m.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    grad_i = grad_i.unsqueeze(-1).expand(3, 2, 128, MIN_BLOCK_SIZE)
    output = torch_xla._XLAC._xla_tpu_custom_call(
        [q, k, v, l, m, grad_o, grad_i], payload, [q.shape], [q.dtype])

    xm.mark_step()

    # TODO: I don't really know how to test the value. Let's do the shape check for now.
    self.assertEqual(output[0].shape, (3, 2, 128, 4))

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_backward(self):
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    from torch_xla.experimental.custom_kernel import flash_attention

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = flash_attention(q, k, v)
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad

    torch.manual_seed(42)
    q = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    k = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    v = torch.randn(4, 2, 128, 8, requires_grad=True).to("xla")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o = self._attention(q, k, v)
    loss = o.sum()
    loss.backward()
    xm.mark_step()

    for i in [(q, q_grad), (k, k_grad), (v, v_grad)]:
      self.assertTrue(torch.allclose(i[0].grad.cpu(), i[1].cpu(), atol=1e-05))
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)

  @unittest.skipIf(xr.device_type() != 'TPU' or tpu.version() < 3,
                   "This test only works on TPUv3+.")
  def test_flash_attention_wrapper_segment_ids(self):
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    from torch_xla.experimental.custom_kernel import flash_attention

    q = torch.randn(3, 2, 128, 4).to("xla")
    k = torch.randn(3, 2, 128, 4).to("xla")
    v = torch.randn(3, 2, 128, 4).to("xla")
    q_segment_ids = torch.zeros(3, 128).to("xla")
    kv_segment_ids = torch.zeros(3, 128).to("xla")

    o = flash_attention(q, k, v, False, q_segment_ids, kv_segment_ids)
    print(o)
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.DEFAULT)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
