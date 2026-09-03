// SPDX-License-Identifier: LicenseRef-Proprietary
// SPDX-FileCopyrightText: Copyright (c) 2026 DaoTechAi Team. All rights reserved.
// dlhook2 —— 按「agent 句柄」而非下标过滤（dlhook.c 的修正版）
//
//   为什么必须改：实测 HIP 传给 allow_access 的 agent 列表**顺序不固定**——
//   101 次调用里有 2 次把当前设备的 agent 提到了队首，1 次只传 7 个。
//   按下标过滤会在这些调用上挑错卡（KEEP=4,5,6,7 在重排列表上挑到 GPU3/4/6/7，
//   漏掉 GPU5）→ 该卡的缓冲未被授权 → 段错误。
//
//   本版做法：agent 句柄按 GPU 序单调递增，维护一张全局升序句柄表，
//   句柄在表中的名次 = GPU 号。首次调用用完整列表播种。
//
//   DLHOOK2_KEEP=4,5,6,7   只把这些 GPU 号的 agent 传下去（显式指定）
//   DLHOOK2_AUTO=1         ★ 自动：拦 hsa_queue_create 得知本进程用哪张卡，
//                          只保留与它同 socket 的 agent。用于 vLLM / sglang
//                          这类「worker 由框架自己 spawn、无法逐 rank 设环境变量」的场景。
//                          （不要改成拦 hipSetDevice —— 试过，拿不到真函数时会把 torch 搞崩）
//   DLHOOK2_SOCKETS=0,1,2,3|4,5,6,7   socket 分组（AUTO 模式用，默认即此值）
//   DLHOOK2_LOG=1          打印每次调用（含解析出的 GPU 号）
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

typedef struct { uint64_t handle; } hsa_agent_t;
typedef struct { uint32_t handle[8]; } hsa_amd_ipc_memory_t;
typedef int hsa_status_t;

static void* (*real_dlsym)(void*, const char*) = NULL;
static void init_real_dlsym(void){
  if (!real_dlsym) real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  if (!real_dlsym) real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
}
static int logon(void){ static int v=-1; if(v<0){const char*e=getenv("DLHOOK2_LOG"); v=e&&*e!='0';} return v; }

/* ── 句柄 → GPU 号 ── */
static pthread_mutex_t tbl_lk = PTHREAD_MUTEX_INITIALIZER;
static uint64_t tbl[64]; static int tbl_n = 0;
static void tbl_add(uint64_t h){                 /* 保持升序插入，去重 */
  int i;
  for (i = 0; i < tbl_n; i++){ if (tbl[i] == h) return; if (tbl[i] > h) break; }
  if (tbl_n >= 64) return;
  memmove(&tbl[i+1], &tbl[i], (size_t)(tbl_n - i) * sizeof tbl[0]);
  tbl[i] = h; tbl_n++;
}
static int tbl_rank(uint64_t h){ for (int i=0;i<tbl_n;i++) if (tbl[i]==h) return i; return -1; }

static int keep_n = -1; static unsigned keep_mask = 0; static int keep_max = -1;

/* ── AUTO 模式：socket 分组 + 当前设备 ── */
static int auto_on = -1;
static unsigned grp[8]; static int grp_n = 0, grp_max = -1;   /* 每组一个位掩码 */
static volatile int cur_dev = -1;                             /* 由 hipSetDevice 记录 */

static void init_groups(void){
  if (grp_n) return;
  const char* e = getenv("DLHOOK2_SOCKETS");
  if (!e || !*e) e = "0,1,2,3|4,5,6,7";
  char b[256]; snprintf(b,sizeof b,"%s",e);
  char* save1 = NULL;
  for (char* g = strtok_r(b,"|",&save1); g && grp_n < 8; g = strtok_r(NULL,"|",&save1)){
    unsigned m = 0; char* save2 = NULL;
    for (char* t = strtok_r(g,",",&save2); t; t = strtok_r(NULL,",",&save2)){
      int v = atoi(t);
      if (v >= 0 && v < 32){ m |= 1u<<v; if (v > grp_max) grp_max = v; }
    }
    if (m) grp[grp_n++] = m;
  }
}
static void init_keep(void){
  if (keep_n >= 0) return;
  keep_n = 0;
  if (auto_on < 0){ const char* a = getenv("DLHOOK2_AUTO"); auto_on = a && *a != '0'; }
  if (auto_on) init_groups();
  const char* e = getenv("DLHOOK2_KEEP");
  if (!e || !*e) return;
  char b[128]; snprintf(b,sizeof b,"%s",e);
  for (char* t = strtok(b,","); t; t = strtok(NULL,",")){
    int g = atoi(t);
    if (g >= 0 && g < 32){ keep_mask |= 1u<<g; keep_n++; if (g > keep_max) keep_max = g; }
  }
}
/* AUTO 模式下当前该保留哪些 GPU；0 = 还不知道，别过滤 */
static unsigned auto_mask(int* need_max){
  int d = cur_dev;
  if (d < 0) return 0;                       /* 还没观察到本进程用哪张卡 */
  for (int i = 0; i < grp_n; i++)
    if (grp[i] & (1u<<d)){ *need_max = grp_max; return grp[i]; }
  return 0;                                  /* 不在任何分组里，不过滤 */
}

/* 返回过滤后个数；-1 = 不过滤 */
static int filter_agents(const hsa_agent_t* ag, uint32_t n, hsa_agent_t* out, int* gpus){
  init_keep();
  /* 无论过不过滤，都先把句柄喂进表里 —— 表要尽早认全，名次才稳定 */
  pthread_mutex_lock(&tbl_lk);
  for (uint32_t i = 0; i < n; i++) tbl_add(ag[i].handle);
  for (uint32_t i = 0; i < n && i < 32; i++) gpus[i] = tbl_rank(ag[i].handle);
  int tn = tbl_n;
  pthread_mutex_unlock(&tbl_lk);

  unsigned mask = keep_mask; int need_max = keep_max;
  if (keep_n <= 0){                        /* 没显式给 KEEP，才走 AUTO */
    if (!auto_on) return -1;
    mask = auto_mask(&need_max);
    if (!mask) return -1;
  }
  if (tn <= need_max) return -1;           /* 句柄表还没认全所有卡时不过滤，避免名次漂移 */
  int c = 0;
  for (uint32_t i = 0; i < n; i++){
    int g = gpus[i];
    if (g >= 0 && g < 32 && (mask & (1u<<g))) out[c++] = ag[i];
  }
  return c > 0 ? c : -1;                   /* 一个都没留下就别过滤，宁可不改也不能传空表 */
}

static void dump(const char* tag, uint32_t n, const hsa_agent_t* ag, const int* gpus,
                 int use, hsa_status_t rc){
  if (!logon()) return;
  fprintf(stderr, "[DLHOOK2] %s num_agents=%u gpus=", tag, n);
  for (uint32_t i=0;i<n && i<32;i++) fprintf(stderr, "%d,", gpus[i]);
  if (use >= 0 && (uint32_t)use != n) fprintf(stderr, " →保留%d个", use);
  fprintf(stderr, " rc=%d\n", rc);
}

static hsa_status_t (*r_allow)(uint32_t, const hsa_agent_t*, const uint32_t*, const void*) = NULL;
static hsa_status_t w_allow(uint32_t n, const hsa_agent_t* ag, const uint32_t* fl, const void* p){
  hsa_agent_t tmp[64]; int gpus[32]; for(int i=0;i<32;i++) gpus[i]=-1;
  uint32_t use = n; const hsa_agent_t* useag = ag;
  int fc = (n && n<=64) ? filter_agents(ag, n, tmp, gpus) : -1;
  if (fc > 0){ useag = tmp; use = (uint32_t)fc; }
  hsa_status_t rc = r_allow(use, useag, fl, p);
  dump("allow_access", n, ag, gpus, (int)use, rc);
  return rc;
}

/* ── hsa_queue_create：第一个参数就是本 worker 用的 agent ──
 *   HIP 为「当前设备」创建 HSA 队列，这是判断本进程归属哪张卡最安全的信号：
 *   它在我们本来就在 hook 的 HSA 层，不用去动 libamdhip64 的导出 ABI
 *   （试过拦 hipSetDevice，拿不到真函数时会把 torch 直接搞崩）。 */
static hsa_status_t (*r_qcreate)(hsa_agent_t, uint32_t, uint32_t, void*, void*,
                                 uint32_t, uint32_t, void**) = NULL;
static hsa_status_t w_qcreate(hsa_agent_t a, uint32_t sz, uint32_t ty, void* cb, void* data,
                              uint32_t pss, uint32_t gss, void** q){
  hsa_status_t rc = r_qcreate(a, sz, ty, cb, data, pss, gss, q);
  if (rc == 0 && cur_dev < 0){
    pthread_mutex_lock(&tbl_lk);
    tbl_add(a.handle);
    int r = tbl_rank(a.handle);
    pthread_mutex_unlock(&tbl_lk);
    if (r >= 0){
      cur_dev = r;
      if (logon()) fprintf(stderr, "[DLHOOK2] queue_create 在 GPU%d 上 → 本进程归属该 socket\n", r);
    }
  }
  return rc;
}

static hsa_status_t (*r_attach)(const hsa_amd_ipc_memory_t*, size_t, uint32_t,
                                const hsa_agent_t*, void**) = NULL;
static hsa_status_t w_attach(const hsa_amd_ipc_memory_t* h, size_t len, uint32_t n,
                             const hsa_agent_t* ag, void** ptr){
  hsa_agent_t tmp[64]; int gpus[32]; for(int i=0;i<32;i++) gpus[i]=-1;
  uint32_t use = n; const hsa_agent_t* useag = ag;
  int fc = (n && n<=64) ? filter_agents(ag, n, tmp, gpus) : -1;
  if (fc > 0){ useag = tmp; use = (uint32_t)fc; }
  hsa_status_t rc = r_attach(h, len, use, useag, ptr);
  dump("ipc_attach", n, ag, gpus, (int)use, rc);
  return rc;
}

void* dlsym(void* handle, const char* name){
  init_real_dlsym();
  if (!real_dlsym) return NULL;
  void* r = real_dlsym(handle, name);
  if (!name || !r) return r;
  if (!strcmp(name, "hsa_amd_agents_allow_access")){
    r_allow = (hsa_status_t(*)(uint32_t,const hsa_agent_t*,const uint32_t*,const void*))r;
    return (void*)w_allow;
  }
  if (!strcmp(name, "hsa_amd_ipc_memory_attach")){
    r_attach = (hsa_status_t(*)(const hsa_amd_ipc_memory_t*,size_t,uint32_t,const hsa_agent_t*,void**))r;
    return (void*)w_attach;
  }
  if (!strcmp(name, "hsa_queue_create")){
    init_keep();
    if (!auto_on) return r;                 /* 只有 AUTO 模式才需要这个信号 */
    r_qcreate = (hsa_status_t(*)(hsa_agent_t,uint32_t,uint32_t,void*,void*,uint32_t,uint32_t,void**))r;
    return (void*)w_qcreate;
  }
  return r;
}
