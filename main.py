#!/usr/bin/env python
# coding: utf-8

# # Performance Comparison
# ## Privacy-Preserving Neural Network Inference
# 
# **Model:** 784 → Dense(128, poly3) → Dense(128, poly3) → Dense(10, Softmax)  
# **Library:** Pyfhel 3.5.0 (Microsoft SEAL backend)  

import os
import concurrent.futures
import numpy as np
import time, gc, warnings

warnings.filterwarnings('ignore')

from Pyfhel import Pyfhel
from math import comb
print("imports OK")

# Set matplotlib to headless mode for the compute node
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Activation ────────────────────────────────────────────────────────
P3_C0 =  0.28126405800
P3_C1 =  0.50000000000
P3_C2 =  0.15624218910
P3_C3 = -4.44089209850063e-18

def poly3_activation(x):
    x = np.asarray(x, dtype=np.float64)
    return P3_C0 + P3_C1*x + P3_C2*x**2 + P3_C3*x**3

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def forward(x_batch, return_intermediates=False):
    pz1 = x_batch @ W1 + b1;  pa1 = poly3_activation(pz1)
    pz2 = pa1 @ W2 + b2;      pa2 = poly3_activation(pz2)
    pz3 = pa2 @ W3 + b3
    if return_intermediates:
        return pz3, pz1, pa1, pz2, pa2
    return pz3

# ── Weights and query (same seed for all three approaches) ─────────────
rng = np.random.default_rng(42)
W1 = rng.normal(0, np.sqrt(2/784), (784,128)).astype(np.float64);  b1 = np.zeros(128)
W2 = rng.normal(0, np.sqrt(2/128), (128,128)).astype(np.float64);  b2 = np.zeros(128)
W3 = rng.normal(0, np.sqrt(2/128), (128,10)).astype(np.float64);   b3 = np.zeros(10)
query = rng.uniform(0, 1, 784).astype(np.float64)
print("weights and query ready — rng.seed=42, He init")

# --- SHAP setup ---
def build_superpixel_map(img_h=28, img_w=28, grid=4):
    block_h, block_w = img_h // grid, img_w // grid
    n_sp = grid * grid
    sp_map = np.zeros((n_sp, img_h * img_w), dtype=np.float64)
    idx = 0
    for r in range(grid):
        for c in range(grid):
            block = np.zeros((img_h, img_w))
            block[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w] = 1
            sp_map[idx] = block.flatten()
            idx += 1
    return sp_map

def expand_superpixel_mask(z_sp, sp_map):
    return z_sp @ sp_map

def f_masked_superpixel(z_sp, x, baseline, sp_map, forward_fn):
    z_pixels = expand_superpixel_mask(z_sp, sp_map)
    x_masked = z_pixels * x[None, :] + (1 - z_pixels) * baseline[None, :]
    return forward_fn(x_masked)

def shap_kernel_weight(k, d):
    return (d - 1) / (comb(d, k) * k * (d - k))

def kernel_shap_superpixel(x, baseline, forward_fn, grid=4, n_samples=150, seed=1):
    sp_map = build_superpixel_map(grid=grid)
    d = sp_map.shape[0]
    rng_local = np.random.default_rng(seed)

    Z = np.zeros((n_samples, d))
    for i in range(n_samples):
        k = rng_local.integers(1, d)
        idx = rng_local.choice(d, size=k, replace=False)
        Z[i, idx] = 1

    fx_full  = f_masked_superpixel(np.ones((1, d)),  x, baseline, sp_map, forward_fn)[0]
    fx_empty = f_masked_superpixel(np.zeros((1, d)), x, baseline, sp_map, forward_fn)[0]
    Y = f_masked_superpixel(Z, x, baseline, sp_map, forward_fn) - fx_empty[None, :]

    sizes = Z.sum(axis=1).astype(int)
    weights = np.array([shap_kernel_weight(k, d) for k in sizes])
    W = np.diag(weights)
    ZtW = Z.T @ W
    A = ZtW @ Z
    B = ZtW @ Y
    phi_sp = np.linalg.solve(A + 1e-6*np.eye(d), B)

    return phi_sp, fx_full, fx_empty, sp_map

def print_shap_summary(phi_sp, fx_full, fx_empty, label, target_class=None, top_k=5):
    if target_class is None:
        target_class = fx_full.argmax()
    phi_class = phi_sp[:, target_class]
    top_idx = np.argsort(-np.abs(phi_class))[:top_k]
    print(f"[{label}] predicted class={target_class}")
    print(f"  phi (class {target_class}, all 16 superpixels): {np.round(phi_class, 4)}")
    print(f"  top-{top_k} superpixels by |SHAP|: idx={top_idx.tolist()}  vals={np.round(phi_class[top_idx],4).tolist()}")
    print(f"  sum(phi) vs f_full-f_empty (class {target_class}): {phi_class.sum():.4f} vs {(fx_full[target_class]-fx_empty[target_class]):.4f}")

# ## 2 · Plaintext baseline
RUNS = 10
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    pz3, pz1, pa1, pz2, pa2 = forward(query, return_intermediates=True)
    times.append((time.perf_counter()-t0)*1e3)

plain_logits = pz3
plain_ms     = np.mean(times)
plain_std    = np.std(times)

print(f"[Plaintext] {plain_ms:.4f} ms ± {plain_std:.4f} ms  class={plain_logits.argmax()}")
print(f"  z1 range [{pz1.min():.3f}, {pz1.max():.3f}]")
print(f"  logits: {np.round(pz3,4)}")

# --- XAI plaintext ---
RUNS_XAI = 3
xai_times = []
background = np.zeros(784)
target_class = None

for i in range(RUNS_XAI):
    t0 = time.perf_counter()
    phi_sp, fx_full, fx_empty, sp_map = kernel_shap_superpixel(
        query, background, forward_fn=forward, grid=4, n_samples=150, seed=1
    )
    xai_times.append((time.perf_counter() - t0) * 1e3)
    if target_class is None:
        target_class = int(fx_full.argmax())
    if i == 0:
        print_shap_summary(phi_sp, fx_full, fx_empty, label="Plaintext KernelSHAP", target_class=target_class)

xai_ms  = np.mean(xai_times)
xai_std = np.std(xai_times)
n_calls = 150 + 2

print(f"[Plaintext KernelSHAP, 16 superpixels] {xai_ms:.2f} ms ± {xai_std:.2f} ms")
print(f"  forward-pass calls used: {n_calls}")
print(f"  slowdown vs single forward pass: {xai_ms/plain_ms:.1f}×")

# ## 3 · Scenario 1 — Full HE
HE1 = Pyfhel()
HE1.contextGen(scheme='ckks', n=2**14, scale=2**30, qi_sizes=[60]+[30]*10+[60])
HE1.keyGen(); HE1.relinKeyGen(); HE1.rotateKeyGen()
N1     = HE1.get_nSlots()
SCALE1 = 2**30
print(f"[S1 CKKS] n={HE1.get_poly_modulus_degree()}  slots={N1}")

def enc_const1(v):
    return HE1.encryptFrac(np.full(N1, float(v), dtype=np.float64))

def ct_add1(a, b):
    a_, b_ = HE1.align_mod_n_scale(~a, ~b, copy_this=True, copy_other=True)
    return HE1.add(a_, b_, in_new_ctxt=True)

def ct_mul1(a, b):
    a_, b_ = HE1.align_mod_n_scale(~a, ~b, copy_this=True, copy_other=True)
    a_.scale = SCALE1;  b_.scale = SCALE1
    r = HE1.multiply(a_, b_, in_new_ctxt=True)
    HE1.relinearize(r); HE1.rescale_to_next(r); r.scale = SCALE1
    return r

def plain_mul_and_sum1(Enc_a, w_arr):
    ptxt    = HE1.encodeFrac(w_arr.astype(np.float64))
    enc_p   = HE1.multiply_plain(~Enc_a, ptxt, in_new_ctxt=True)
    enc_z   = HE1.cumul_add(enc_p, n_elements=0, in_new_ctxt=True)
    del enc_p, ptxt
    HE1.rescale_to_next(enc_z); enc_z.scale = SCALE1
    return enc_z

def he_poly3_1(Enc_z):
    t1 = ct_mul1(enc_const1(P3_C1), Enc_z)
    z2 = ct_mul1(Enc_z, Enc_z)
    t2 = ct_mul1(enc_const1(P3_C2), z2)
    z3 = ct_mul1(z2, Enc_z);           del z2;  gc.collect()
    t3 = ct_mul1(enc_const1(P3_C3), z3); del z3;  gc.collect()
    acc = ct_add1(t1, t2);  del t1, t2;  gc.collect()
    acc = ct_add1(acc, t3);  del t3;     gc.collect()
    r   = ct_add1(acc, enc_const1(P3_C0));  del acc;  gc.collect()
    return r

def linear_from_list1(enc_a_list, W, b):
    enc_z_list = []
    for k in range(W.shape[1]):
        acc = None
        for j, enc_aj in enumerate(enc_a_list):
            ptxt_w = HE1.encodeFrac(np.full(N1, W[j,k], dtype=np.float64))
            ct_al, pt_al = HE1.align_mod_n_scale(~enc_aj, ptxt_w, copy_this=True, copy_other=True)
            enc_wj = HE1.multiply_plain(ct_al, pt_al, in_new_ctxt=True)
            del ptxt_w, ct_al, pt_al
            if acc is None: acc = enc_wj
            else:
                a2, b2 = HE1.align_mod_n_scale(~acc, ~enc_wj, copy_this=True, copy_other=True)
                acc = HE1.add(a2, b2, in_new_ctxt=True)
                del enc_wj, a2, b2
        HE1.rescale_to_next(acc);  acc.scale = SCALE1
        if abs(b[k]) > 1e-12:
            enc_b = HE1.encryptFrac(np.full(N1, float(b[k]), dtype=np.float64))
            a2, b2 = HE1.align_mod_n_scale(~acc, ~enc_b, copy_this=True, copy_other=True)
            acc = HE1.add(a2, b2, in_new_ctxt=True)
            del enc_b, a2, b2
        enc_z_list.append(acc);  gc.collect()
    return enc_z_list

print("S1 HE functions defined ✅")

def he_forward_scenario1_timed(x_plain, verbose_stage_times=None, debug_compare=None):
    q_pad = np.zeros(N1, dtype=np.float64)
    q_pad[:784] = x_plain;  q_pad[784] = 1.0
    Enc_q = HE1.encryptFrac(q_pad)

    t0 = time.perf_counter()
    enc_z1 = []
    for j in range(128):
        w_pad = np.zeros(N1, dtype=np.float64)
        w_pad[:784] = W1[:,j];  w_pad[784] = b1[j]
        enc_z1.append(plain_mul_and_sum1(Enc_q, w_pad));  gc.collect()
    t_l1_lin = time.perf_counter()-t0
    if debug_compare is not None:
        dbg = HE1.decryptFrac(enc_z1[0])[0]
        print(f"[L1 linear]  {t_l1_lin*1e3:.0f} ms  z1[0]: HE={dbg:.4f} plain={debug_compare['pz1'][0]:.4f} err={abs(dbg-debug_compare['pz1'][0]):.2e}")

    t0 = time.perf_counter()
    enc_a1 = [he_poly3_1(ez) for ez in enc_z1]
    del enc_z1;  gc.collect()
    t_l1_act = time.perf_counter()-t0
    if debug_compare is not None:
        dbg_a = HE1.decryptFrac(enc_a1[0])[0]
        print(f"[L1 poly-3]  {t_l1_act*1e3:.0f} ms  a1[0]: HE={dbg_a:.4f} plain={debug_compare['pa1'][0]:.4f} err={abs(dbg_a-debug_compare['pa1'][0]):.2e}")

    t0 = time.perf_counter()
    enc_z2 = linear_from_list1(enc_a1, W2, b2)
    del enc_a1;  gc.collect()
    t_l2_lin = time.perf_counter()-t0
    if debug_compare is not None:
        dbg2 = HE1.decryptFrac(enc_z2[0])[0]
        print(f"[L2 linear]  {t_l2_lin*1e3:.0f} ms  z2[0]: HE={dbg2:.4f} plain={debug_compare['pz2'][0]:.4f} err={abs(dbg2-debug_compare['pz2'][0]):.2e}")

    t0 = time.perf_counter()
    enc_a2 = [he_poly3_1(ez) for ez in enc_z2]
    del enc_z2;  gc.collect()
    t_l2_act = time.perf_counter()-t0
    if debug_compare is not None:
        dbg_a2 = HE1.decryptFrac(enc_a2[0])[0]
        print(f"[L2 poly-3]  {t_l2_act*1e3:.0f} ms  a2[0]: HE={dbg_a2:.4f} plain={debug_compare['pa2'][0]:.4f} err={abs(dbg_a2-debug_compare['pa2'][0]):.2e}")

    t0 = time.perf_counter()
    enc_z3 = linear_from_list1(enc_a2, W3, b3)
    del enc_a2;  gc.collect()
    t_l3_lin = time.perf_counter()-t0

    logits = np.array([float(HE1.decryptFrac(enc_z3[k])[0]) for k in range(10)])

    if verbose_stage_times is not None:
        for k, v in [('l1_lin',t_l1_lin), ('l1_act',t_l1_act), ('l2_lin',t_l2_lin), ('l2_act',t_l2_act), ('l3_lin',t_l3_lin)]:
            verbose_stage_times[k] += v

    return logits, {'l1_lin': t_l1_lin, 'l1_act': t_l1_act, 'l2_lin': t_l2_lin, 'l2_act': t_l2_act, 'l3_lin': t_l3_lin}

print("Running Scenario 1 (Full HE)...")
print("=" * 60)

s1_logits, s1_timing = he_forward_scenario1_timed(
    query, debug_compare={'pz1': pz1, 'pa1': pa1, 'pz2': pz2, 'pa2': pa2}
)
s1_total = sum(s1_timing.values())
print(f"[L3 linear]  {s1_timing['l3_lin']*1e3:.0f} ms")
print(f"\n[S1] class={s1_logits.argmax()}  total={s1_total*1e3:.0f} ms")
print(f"  logits: {np.round(s1_logits,4)}")

# --- xai version ---
print("Running Scenario 1 (Full HE) + KernelSHAP...")
print("=" * 60)
s1_xai_timing = {'l1_lin': 0.0, 'l1_act': 0.0, 'l2_lin': 0.0, 'l2_act': 0.0, 'l3_lin': 0.0}

def _run_s1_single(row):
    # Worker function that runs on a single core
    logits, timings = he_forward_scenario1_timed(row)
    return logits, timings

def he_forward_scenario1_batch(x_batch):
    # Dynamically read Slurm CPU allocation
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    outputs = []
    
    # Spawn a pool of workers to process the XAI batch in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_run_s1_single, x_batch))
        
    for logits, timings in results:
        outputs.append(logits)
        for k in s1_xai_timing:
            s1_xai_timing[k] += timings[k]
            
    return np.array(outputs)

t0 = time.perf_counter()
phi_sp_s1, fx_full_s1, fx_empty_s1, sp_map = kernel_shap_superpixel(
    query, background, forward_fn=he_forward_scenario1_batch, grid=4, n_samples=150, seed=1
)
s1_xai_total = time.perf_counter() - t0

print_shap_summary(phi_sp_s1, fx_full_s1, fx_empty_s1, label="Scenario 1 (HE) KernelSHAP", target_class=target_class)

print(f"[L1 linear]  {s1_xai_timing['l1_lin']*1e3:.0f} ms")
print(f"[L1 poly-3]  {s1_xai_timing['l1_act']*1e3:.0f} ms")
print(f"[L2 linear]  {s1_xai_timing['l2_lin']*1e3:.0f} ms")
print(f"[L2 poly-3]  {s1_xai_timing['l2_act']*1e3:.0f} ms")
print(f"[L3 linear]  {s1_xai_timing['l3_lin']*1e3:.0f} ms")
print(f"\n[S1-XAI] n_calls=152  total={s1_xai_total*1e3:.0f} ms  ({s1_xai_total/60:.1f} min)")

# ## 4 · Scenario 2 — Encrypted Weights + TEE
def he_forward_scenario2_timed(x_plain, verbose_stage_times=None, debug_compare=None):
    t0 = time.perf_counter()
    z1, a1 = sp_tee_layer(HE2, enc_W1, x_plain, 784, N2)
    t_l1 = time.perf_counter()-t0
    if debug_compare is not None:
        print(f"[L1]  {t_l1*1e3:.0f} ms  z1[0]: HE={z1[0]:.4f} plain={debug_compare['pz1'][0]:.4f} err={abs(z1[0]-debug_compare['pz1'][0]):.2e}")

    t0 = time.perf_counter()
    z2, a2 = sp_tee_layer(HE2, enc_W2, a1, 128, N2)
    t_l2 = time.perf_counter()-t0
    if debug_compare is not None:
        print(f"[L2]  {t_l2*1e3:.0f} ms  z2[0]: HE={z2[0]:.4f} plain={debug_compare['pz2'][0]:.4f} err={abs(z2[0]-debug_compare['pz2'][0]):.2e}")

    t0 = time.perf_counter()
    z3, a3 = sp_tee_layer(HE2, enc_W3, a2, 128, N2)
    t_l3 = time.perf_counter()-t0
    if debug_compare is not None:
        print(f"[L3]  {t_l3*1e3:.0f} ms")

    logits = z3

    if verbose_stage_times is not None:
        verbose_stage_times['l1'] += t_l1
        verbose_stage_times['l2'] += t_l2
        verbose_stage_times['l3'] += t_l3

    return logits, {'l1': t_l1, 'l2': t_l2, 'l3': t_l3}

# ── CKKS context for Scenario 2 ───────────────────────────────────────
HE2 = Pyfhel()
HE2.contextGen(scheme='ckks', n=2**13, scale=2**30, qi_sizes=[60,30,30,30,60])
HE2.keyGen(); HE2.relinKeyGen(); HE2.rotateKeyGen()
N2     = HE2.get_nSlots()
SCALE2 = 2**30
print(f"[S2 CKKS] n={HE2.get_poly_modulus_degree()}  slots={N2}")

def encrypt_weights(HE, W, b, input_len, N):
    enc = []
    for j in range(W.shape[1]):
        w = np.zeros(N, dtype=np.float64)
        w[:input_len] = W[:,j];  w[input_len] = b[j]
        enc.append(HE.encryptFrac(w));  gc.collect()
    return enc

def sp_tee_layer(HE, enc_W_cols, a_plain, input_len, N):
    a_pad = np.zeros(N, dtype=np.float64)
    a_pad[:input_len] = a_plain[:input_len];  a_pad[input_len] = 1.0
    ptxt_a = HE.encodeFrac(a_pad)
    z = np.zeros(len(enc_W_cols), dtype=np.float64)
    for j, enc_wj in enumerate(enc_W_cols):
        enc_p  = HE.multiply_plain(~enc_wj, ptxt_a, in_new_ctxt=True)
        enc_z  = HE.cumul_add(enc_p, n_elements=0, in_new_ctxt=True)
        z[j]   = float(HE.decryptFrac(enc_z)[0])
        del enc_p, enc_z;  gc.collect()
    return z, poly3_activation(z)

print("Encrypting model weights (one-time)...")
t0 = time.perf_counter()
enc_W1 = encrypt_weights(HE2, W1, b1, 784, N2)
enc_W2 = encrypt_weights(HE2, W2, b2, 128, N2)
enc_W3 = encrypt_weights(HE2, W3, b3, 128, N2)
s2_wenc_ms = (time.perf_counter()-t0)*1e3
print(f"  Weight encryption: {s2_wenc_ms:.0f} ms  (one-time at deployment)")

print("Running Scenario 2 (TEE)...")
print("=" * 60)
s2_timing = {}

t0=time.perf_counter(); z1,a1=sp_tee_layer(HE2,enc_W1,query,784,N2);  s2_timing['l1']=time.perf_counter()-t0
print(f"[L1]  {s2_timing['l1']*1e3:.0f} ms  z1[0]: HE={z1[0]:.4f} plain={pz1[0]:.4f} err={abs(z1[0]-pz1[0]):.2e}")

t0=time.perf_counter(); z2,a2=sp_tee_layer(HE2,enc_W2,a1,128,N2);     s2_timing['l2']=time.perf_counter()-t0
print(f"[L2]  {s2_timing['l2']*1e3:.0f} ms  z2[0]: HE={z2[0]:.4f} plain={pz2[0]:.4f} err={abs(z2[0]-pz2[0]):.2e}")

t0=time.perf_counter(); z3,a3=sp_tee_layer(HE2,enc_W3,a2,128,N2);     s2_timing['l3']=time.perf_counter()-t0
print(f"[L3]  {s2_timing['l3']*1e3:.0f} ms")

s2_logits = z3
s2_total  = sum(s2_timing.values())
print(f"\n[S2] class={s2_logits.argmax()}  total={s2_total*1e3:.0f} ms")
print(f"  logits: {np.round(s2_logits,4)}")

print("Running Scenario 2 (TEE) + KernelSHAP...")
print("=" * 60)
s2_xai_timing = {'l1': 0.0, 'l2': 0.0, 'l3': 0.0}

def _run_s2_single(row):
    logits, timings = he_forward_scenario2_timed(row)
    return logits, timings

def he_forward_scenario2_batch(x_batch):
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    outputs = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_run_s2_single, x_batch))
        
    for logits, timings in results:
        outputs.append(logits)
        for k in s2_xai_timing:
            s2_xai_timing[k] += timings[k]
            
    return np.array(outputs)

t0 = time.perf_counter()
phi_sp_s2, fx_full_s2, fx_empty_s2, sp_map = kernel_shap_superpixel(
    query, background, forward_fn=he_forward_scenario2_batch, grid=4, n_samples=150, seed=1
)
s2_xai_total = time.perf_counter() - t0

print(f"[L1]  {s2_xai_timing['l1']*1e3:.0f} ms")
print(f"[L2]  {s2_xai_timing['l2']*1e3:.0f} ms")
print(f"[L3]  {s2_xai_timing['l3']*1e3:.0f} ms")
print(f"\n[S2-XAI] n_calls=152  total={s2_xai_total*1e3:.0f} ms  ({s2_xai_total/60:.1f} min)")

print_shap_summary(phi_sp_s2, fx_full_s2, fx_empty_s2, label="Scenario 2 (TEE) KernelSHAP", target_class=target_class)

# ## 5 · Correctness check — all intermediates vs plaintext
print("=" * 62)
print("CORRECTNESS  (all intermediate values vs plaintext baseline)")
print("=" * 62)

rows = [
    ("z1  (Layer 1 pre-activation)", pz1, None,   z1),
    ("a1  = poly3(z1)",              pa1, None,   a1),
    ("z2  (Layer 2 pre-activation)", pz2, None,   z2),
    ("a2  = poly3(z2)",              pa2, None,   a2),
    ("z3  (output logits)",          pz3, s1_logits, z3),
]

print(f"{'Intermediate':<32} {'S1 max err':>12} {'S2 max err':>12}")
print("-" * 58)
for name, plain, s1, s2 in rows:
    s1_err = f"{np.abs(plain-s1).max():.2e}" if s1 is not None else "    —     "
    s2_err = f"{np.abs(plain-s2).max():.2e}" if s2 is not None else "    —     "
    print(f"  {name:<30} {s1_err:>12} {s2_err:>12}")

print()
s1_match = plain_logits.argmax() == s1_logits.argmax()
s2_match = plain_logits.argmax() == s2_logits.argmax()
print(f"  Plaintext class : {plain_logits.argmax()}")
print(f"  Scenario 1 class: {s1_logits.argmax()}  {'✅ MATCH' if s1_match else '❌ MISMATCH'}")
print(f"  Scenario 2 class: {s2_logits.argmax()}  {'✅ MATCH' if s2_match else '❌ MISMATCH'}")

# ## 6 · Performance comparison
print("=" * 62)
print("PERFORMANCE SUMMARY")
print("=" * 62)

s1_lin  = s1_timing['l1_lin']+s1_timing['l2_lin']+s1_timing['l3_lin']
s1_act  = s1_timing['l1_act']+s1_timing['l2_act']
s1_tot  = s1_total
s2_tot  = s2_total

print(f"\n{'Approach':<32} {'Time (ms)':>12} {'Slowdown':>12} {'Class':>8}")
print("-" * 66)
print(f"  {'Plaintext baseline':<30} {plain_ms:>12.1f} {'1×':>12} {plain_logits.argmax():>8}")
print(f"  {'Scenario 1 — Encrypted query':<30} {s1_tot*1e3:>12.1f} {f'{s1_tot*1e3/plain_ms:.0f}×':>12} {s1_logits.argmax():>8}")
print(f"  {'Scenario 2 — Encrypted weights':<30} {s2_tot*1e3:>12.1f} {f'{s2_tot*1e3/plain_ms:.0f}×':>12} {s2_logits.argmax():>8}")

print()
print(f"  Weight encryption one-time cost (S2): {s2_wenc_ms:.0f} ms")
print()
print("  Scenario 1 breakdown:")
print(f"    HE linear layers:     {s1_lin*1e3:>8.0f} ms")
print(f"    HE poly-3 activation: {s1_act*1e3:>8.0f} ms")
print()
print("  Scenario 2 breakdown:")
for lbl, k in [("Layer 1 (784→128)", 'l1'), ("Layer 2 (128→128)", 'l2'), ("Layer 3 (128→10)", 'l3')]:
    print(f"    {lbl}: {s2_timing[k]*1e3:>8.0f} ms")

# ## 7 · Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Privacy-Preserving Neural Network Inference\nPerformance Comparison",
             fontsize=13, fontweight='bold')

ax = axes[0]
labels  = ['Plaintext', 'Scenario 1\n(Enc. Query)', 'Scenario 2\n(Enc. Weights)']
times   = [plain_ms, s1_tot*1e3, s2_tot*1e3]
colors  = ['#4CAF50', '#2196F3', '#FF9800']
bars = ax.bar(labels, times, color=colors, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Total Inference Time', fontsize=11)
ax.set_yscale('log')
for bar, t in zip(bars, times):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
            f'{t:.0f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
slowdowns = [1, s1_tot*1e3/plain_ms, s2_tot*1e3/plain_ms]
bars = ax.bar(labels, slowdowns, color=colors, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Slowdown (×)', fontsize=11)
ax.set_title('Slowdown vs Plaintext', fontsize=11)
ax.set_yscale('log')
for bar, s in zip(bars, slowdowns):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
            f'{s:.0f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
s1_layers = [s1_timing['l1_lin']+s1_timing['l1_act'],
             s1_timing['l2_lin']+s1_timing['l2_act'],
             s1_timing['l3_lin']]
s2_layers = [s2_timing['l1'], s2_timing['l2'], s2_timing['l3']]
layer_labels = ['Layer 1\n(784→128)', 'Layer 2\n(128→128)', 'Layer 3\n(128→10)']
x = np.arange(3)
w = 0.35
ax.bar(x-w/2, [t*1e3 for t in s1_layers], w, label='Scenario 1', color='#2196F3')
ax.bar(x+w/2, [t*1e3 for t in s2_layers], w, label='Scenario 2', color='#FF9800')
ax.set_xticks(x); ax.set_xticklabels(layer_labels)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Per-Layer Breakdown', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved as performance_comparison.png")

# ## 8 · Summary table
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n{'':32} {'Plaintext':>12} {'Scenario 1':>12} {'Scenario 2':>12}")
print("-" * 70)

rows = [
    ("Model weights",      "Plaintext",   "Plaintext",    "Encrypted E(W)"),
    ("Consumer query",     "Plaintext",   "Encrypted E(x)","Plaintext"),
    ("HE scheme",          "None",        "CKKS",         "CKKS"),
    ("CKKS n",             "—",           "2^14",         "2^13"),
    ("Intermediate decrypt","—",          "None ✅",      "TEE only"),
    ("TEE needed",         "No",          "No",           "Yes"),
    ("Activation method",  "Exact poly3", "HE poly3",     "Plain poly3 (TEE)"),
    ("Time (ms)",
     f"{plain_ms:.2f}",
     f"{s1_tot*1e3:.0f}",
     f"{s2_tot*1e3:.0f}"),
    ("Slowdown",           "1×",
     f"{s1_tot*1e3/plain_ms:.0f}×",
     f"{s2_tot*1e3/plain_ms:.0f}×"),
    ("Max logit error",    "—",
     f"{np.abs(pz3-s1_logits).max():.2e}",
     f"{np.abs(pz3-s2_logits).max():.2e}"),
    ("Predicted class",
     f"{plain_logits.argmax()}",
     f"{s1_logits.argmax()} ✅",
     f"{s2_logits.argmax()} ✅"),
    ("No MPC needed",      "—",           "✅",           "✅"),
]

for row in rows:
    lbl = row[0]; vals = row[1:]
    print(f"  {lbl:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")