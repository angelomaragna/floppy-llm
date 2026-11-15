/* floppy.c — single-file C99 inference host for the FLPY v1 int8 model.
 *
 * Pure C99, stdlib only. Reads a FLPY v1 binary (header + int8 tensors with
 * per-tensor float32 scales), dequantizes on load, runs a GPT with RoPE /
 * RMSNorm / GELU and a re-computed KV window, and serves a REPL.
 * BPE-512 pieces are embedded via vocab.c; encoding is the same pair-merge
 * algorithm as floppy_llm/bpe_ref.py (verified token-exact against
 * sentencepiece on the corpus).
 *
 * Usage: floppy model.bin [-p "prompt"] [-t temp] [--logits-debug]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VOCAB 1024
#define MAX_DIM 256
#define MAX_FFN 1024
#define MAX_CTX 512
#define MAX_TENSORS 64
#define MAX_NAME 32
#define UNK 0
#define EOS 2

/* ---------- vocab (generated) ---------- */
extern const int sp_n;
extern const char *sp_piece[];

/* ---------- model ---------- */
typedef struct {
    uint32_t shape[MAX_DIM];
    int ndim;
    float scale;
    int8_t *q;
    float *w;
    int n;
    char name[MAX_NAME];
} Tensor;

typedef struct {
    uint32_t magic, version, vocab_size, d_model, n_layer, n_heads, ffn_dim,
        context_len, n_tensors;
    Tensor t[MAX_TENSORS];
    int count;
} Model;

static Model g_model;

static Tensor *tfind(const char *name) {
    for (int i = 0; i < g_model.count; i++)
        if (strcmp(g_model.t[i].name, name) == 0) return &g_model.t[i];
    fprintf(stderr, "missing tensor %s\n", name);
    exit(1);
}

static uint32_t rdu32(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}
static uint16_t rdu16(const uint8_t *b) {
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}
static float rdf32(const uint8_t *b) {
    uint32_t u = rdu32(b);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
    if (sz < 32) { fprintf(stderr, "file too small\n"); exit(1); }
    g_model.magic = rdu32(buf + 0);
    g_model.version = rdu32(buf + 4);
    g_model.vocab_size = rdu32(buf + 8);
    g_model.d_model = rdu16(buf + 12);
    g_model.n_layer = rdu16(buf + 14);
    g_model.n_heads = rdu32(buf + 16);
    g_model.ffn_dim = rdu32(buf + 20);
    g_model.context_len = rdu32(buf + 24);
    g_model.n_tensors = rdu32(buf + 28);
    if (g_model.magic != 0x464C5059u || g_model.version != 1) {
        fprintf(stderr, "bad magic/version\n"); exit(1);
    }
    long off = 32;
    for (uint32_t i = 0; i < g_model.n_tensors; i++) {
        Tensor *t = &g_model.t[i];
        int name_len = buf[off]; off++;
        if (name_len >= MAX_NAME) { fprintf(stderr, "name too long\n"); exit(1); }
        memcpy(t->name, buf + off, (size_t)name_len);
        t->name[name_len] = 0;
        off += name_len;
        t->ndim = buf[off]; off++;
        t->n = 1;
        for (int d = 0; d < t->ndim; d++) {
            t->shape[d] = rdu32(buf + off); off += 4;
            t->n *= (int)t->shape[d];
        }
        t->scale = rdf32(buf + off); off += 4;
        t->q = malloc((size_t)t->n);
        memcpy(t->q, buf + off, (size_t)t->n);
        off += t->n;
        t->w = malloc(sizeof(float) * (size_t)t->n);
        for (int j = 0; j < t->n; j++) t->w[j] = (float)t->q[j] * t->scale;
    }
    g_model.count = (int)g_model.n_tensors;
    free(buf);
}

/* ---------- BPE tokenizer (pair-merge, identical to bpe_ref.py) ---------- */

#define HASH_MAX 2048
static int g_hash_id[HASH_MAX]; /* piece id + 1; 0 = empty */
static char g_hash_key[HASH_MAX][MAX_NAME * 2 + 8];
static int g_hash_klen[HASH_MAX];
static char g_piece_mem[MAX_VOCAB][MAX_NAME * 2 + 8];
static int g_piece_len[MAX_VOCAB];

static unsigned long fnv1a(const char *s, int n) {
    unsigned long h = 1469598103934665603UL;
    for (int i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211UL;
    }
    return h;
}

static void hash_put(const char *key, int len, int id) {
    unsigned long h = fnv1a(key, len) % HASH_MAX;
    while (g_hash_id[h] != 0) h = (h + 1) % HASH_MAX;
    memcpy(g_hash_key[h], key, (size_t)len);
    g_hash_klen[h] = len;
    g_hash_id[h] = id + 1;
}

static int hash_get(const char *key, int len) {
    unsigned long h = fnv1a(key, len) % HASH_MAX;
    while (g_hash_id[h] != 0) {
        if (g_hash_klen[h] == len && memcmp(g_hash_key[h], key, (size_t)len) == 0)
            return g_hash_id[h] - 1;
        h = (h + 1) % HASH_MAX;
    }
    return -1;
}

static void vocab_init(void) {
    if (sp_n > MAX_VOCAB) { fprintf(stderr, "vocab too big\n"); exit(1); }
    for (int i = 0; i < sp_n; i++) {
        size_t L = strlen(sp_piece[i]);
        if (L >= sizeof(g_piece_mem[0])) { fprintf(stderr, "piece too long\n"); exit(1); }
        strcpy(g_piece_mem[i], sp_piece[i]);
        g_piece_len[i] = (int)L;
        hash_put(g_piece_mem[i], g_piece_len[i], i);
    }
}

static const char *SP = "\xe2\x96\x81"; /* U+2581 */
#define SPL 3

static int utf8_len(unsigned char c) {
    return c < 0x80 ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
}

static int word_encode(const char *word, int wlen, int *out, int out_cap) {
    static char sym[64][16];
    static int syml[64];
    int ns = 0;
    for (int i = 0; i < wlen && ns < 64;) {
        int cl = utf8_len((unsigned char)word[i]);
        if (i + cl > wlen) cl = 1;
        memcpy(sym[ns], word + i, (size_t)cl);
        syml[ns] = cl;
        ns++;
        i += cl;
    }
    int changed = 1;
    while (changed && ns > 1) {
        changed = 0;
        int best = -1, best_rank = 1 << 30;
        for (int i = 0; i < ns - 1; i++) {
            int cl = syml[i] + syml[i + 1];
            if (cl > 15) continue;
            char cat[16];
            memcpy(cat, sym[i], (size_t)syml[i]);
            memcpy(cat + syml[i], sym[i + 1], (size_t)syml[i + 1]);
            if (cl < 2) continue;
            int id = hash_get(cat, cl);
            if (id < 3) continue;
            int rank = id; /* vocab order == merge rank, as in bpe_ref.py */
            if (rank < best_rank) { best_rank = rank; best = i; }
        }
        if (best >= 0) {
            memcpy(sym[best] + syml[best], sym[best + 1], (size_t)syml[best + 1]);
            syml[best] += syml[best + 1];
            for (int i = best + 1; i < ns - 1; i++) {
                memcpy(sym[i], sym[i + 1], (size_t)syml[i + 1]);
                syml[i] = syml[i + 1];
            }
            ns--;
            changed = 1;
        }
    }
    int n = 0;
    for (int i = 0; i < ns; i++) {
        int id = hash_get(sym[i], syml[i]);
        if (id >= 0) {
            if (n < out_cap) out[n++] = id;
        } else {
            int pos = 0;
            while (pos < syml[i] && n < out_cap) {
                int cl = utf8_len((unsigned char)sym[i][pos]);
                if (pos + cl > syml[i]) cl = 1;
                int cid = hash_get(sym[i] + pos, cl);
                out[n++] = (cid >= 0) ? cid : UNK;
                pos += cl;
            }
        }
    }
    return n;
}

static int bpe_encode(const char *text, int *ids, int cap) {
    int n = 0, i = 0, len = (int)strlen(text);
    while (i < len && n < cap) {
        while (i < len && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n')) i++;
        if (i >= len) break;
        char word[MAX_NAME * 2 + 8];
        int wl = SPL;
        memcpy(word, SP, (size_t)SPL);
        while (i < len && text[i] != ' ' && text[i] != '\t' && text[i] != '\n')
            word[wl++] = text[i++];
        n += word_encode(word, wl, ids + n, cap - n);
    }
    return n;
}

static void decode_ids(const int *ids, int n) {
    for (int i = 0; i < n; i++) {
        const char *p = g_piece_mem[ids[i]];
        int L = g_piece_len[ids[i]];
        if (L == SPL && memcmp(p, SP, (size_t)SPL) == 0) continue;
        if (L > SPL && memcmp(p, SP, (size_t)SPL) == 0) {
            fputs(" ", stdout);
            fputs(p + SPL, stdout);
        } else {
            fputs(p, stdout);
        }
    }
}

/* ---------- forward pass ---------- */
static float g_x[MAX_DIM], g_xb[MAX_DIM], g_q[MAX_DIM], g_att[MAX_DIM];
static float g_h[MAX_FFN], g_logits[MAX_VOCAB], g_e1[MAX_DIM], g_e2[MAX_DIM];
static float *g_kcache[MAX_TENSORS], *g_vcache[MAX_TENSORS];

static void rmsnorm(float *out, const float *x, const float *w, int d) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / d + 1e-6f);
    for (int i = 0; i < d; i++) out[i] = x[i] * ss * w[i];
}

static void matvec(float *out, const float *x, const float *W, int nout, int nin) {
    for (int o = 0; o < nout; o++) {
        float s = 0;
        const float *row = W + (size_t)o * (size_t)nin;
        for (int i = 0; i < nin; i++) s += row[i] * x[i];
        out[o] = s;
    }
}

/* process tokens ids[0..n-1]; returns argmax of the last position's logits.
   if next != NULL, fills next[t] with argmax of position t.
   if debug, prints "ARGMAX t id" per position to stderr. */
static void forward(const int *ids, int n, int *next, int debug) {
    int d = (int)g_model.d_model;
    int nh = (int)g_model.n_heads;
    int hd = d / nh;
    int ffn = (int)g_model.ffn_dim;
    int ctx = (int)g_model.context_len;
    if (n > ctx) { ids += n - ctx; n = ctx; }

    Tensor *emb = tfind("tok_emb");
    Tensor *fnorm = tfind("final_norm");

    for (int t = 0; t < n; t++) {
        int pos = t;
        memcpy(g_x, emb->w + (size_t)ids[t] * (size_t)d, sizeof(float) * (size_t)d);
        for (int L = 0; L < (int)g_model.n_layer; L++) {
            char nm[MAX_NAME];
            snprintf(nm, MAX_NAME, "attn_q_L%d", L);
            Tensor *Wq = tfind(nm);
            snprintf(nm, MAX_NAME, "attn_k_L%d", L);
            Tensor *Wk = tfind(nm);
            snprintf(nm, MAX_NAME, "attn_v_L%d", L);
            Tensor *Wv = tfind(nm);
            snprintf(nm, MAX_NAME, "attn_o_L%d", L);
            Tensor *Wo = tfind(nm);
            snprintf(nm, MAX_NAME, "norm1_L%d", L);
            Tensor *N1 = tfind(nm);
            snprintf(nm, MAX_NAME, "norm2_L%d", L);
            Tensor *N2 = tfind(nm);
            snprintf(nm, MAX_NAME, "ffn_fc_L%d", L);
            Tensor *Wf1 = tfind(nm);
            snprintf(nm, MAX_NAME, "ffn_proj_L%d", L);
            Tensor *Wf2 = tfind(nm);

            rmsnorm(g_xb, g_x, N1->w, d);
            matvec(g_q, g_xb, Wq->w, d, d);
            float *k = g_kcache[L] + (size_t)pos * (size_t)d;
            float *v = g_vcache[L] + (size_t)pos * (size_t)d;
            matvec(k, g_xb, Wk->w, d, d);
            matvec(v, g_xb, Wv->w, d, d);

            /* RoPE: pair (2i, 2i+1) has angle pos / 10000^(2i/hd) */
            for (int h = 0; h < nh; h++) {
                float *qh = g_q + h * hd;
                float *kh = k + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float inv = powf(10000.0f, (float)(2 * i) / (float)hd);
                    float angle = (float)pos / inv;
                    float c = cosf(angle), s = sinf(angle);
                    float qe = qh[2 * i], qo = qh[2 * i + 1];
                    qh[2 * i] = qe * c - qo * s;
                    qh[2 * i + 1] = qe * s + qo * c;
                    float ke = kh[2 * i], ko = kh[2 * i + 1];
                    kh[2 * i] = ke * c - ko * s;
                    kh[2 * i + 1] = ke * s + ko * c;
                }
            }

            float scale = 1.0f / sqrtf((float)hd);
            float o[MAX_DIM];
            for (int h = 0; h < nh; h++) {
                float scores[MAX_CTX];
                float mx = -1e30f;
                const float *qh = g_q + h * hd;
                for (int p = 0; p <= pos; p++) {
                    const float *kh = g_kcache[L] + (size_t)p * (size_t)d + h * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
                    scores[p] = s * scale;
                    if (scores[p] > mx) mx = scores[p];
                }
                float sum = 0;
                for (int p = 0; p <= pos; p++) {
                    scores[p] = expf(scores[p] - mx);
                    sum += scores[p];
                }
                float out[MAX_DIM / 4];
                for (int i = 0; i < hd; i++) out[i] = 0;
                for (int p = 0; p <= pos; p++) {
                    const float *vh = g_vcache[L] + (size_t)p * (size_t)d + h * hd;
                    float wgt = scores[p] / sum;
                    for (int i = 0; i < hd; i++) out[i] += wgt * vh[i];
                }
                memcpy(o + h * hd, out, sizeof(float) * (size_t)hd);
            }
            matvec(g_att, o, Wo->w, d, d);
            for (int i = 0; i < d; i++) g_x[i] += g_att[i];

            rmsnorm(g_xb, g_x, N2->w, d);
            matvec(g_h, g_xb, Wf1->w, ffn, d);
            for (int i = 0; i < ffn; i++)
                g_h[i] = 0.5f * g_h[i] * (1.0f + erff(g_h[i] / 1.41421356f));
            matvec(g_e1, g_h, Wf2->w, d, ffn);
            for (int i = 0; i < d; i++) g_x[i] += g_e1[i];
        }
        rmsnorm(g_e2, g_x, fnorm->w, d);
        for (int voc = 0; voc < (int)g_model.vocab_size; voc++) {
            float s = 0;
            const float *row = emb->w + (size_t)voc * (size_t)d;
            for (int i = 0; i < d; i++) s += row[i] * g_e2[i];
            g_logits[voc] = s;
        }
        int argmax = 0;
        float bestv = g_logits[0];
        for (int voc = 1; voc < (int)g_model.vocab_size; voc++)
            if (g_logits[voc] > bestv) { bestv = g_logits[voc]; argmax = voc; }
        if (debug) fprintf(stderr, "ARGMAX %d %d\n", t, argmax);
        if (next) next[t] = argmax;
    }
}

static void generate(const int *prompt, int n, int steps, float temp) {
    int ctx = (int)g_model.context_len;
    int gen[MAX_CTX * 2];
    int total = n;
    if (total > MAX_CTX * 2) total = MAX_CTX * 2;
    memcpy(gen, prompt, sizeof(int) * (size_t)n);
    for (int step = 0; step < steps; step++) {
        int next[MAX_CTX];
        int len = total > ctx ? ctx : total;
        const int *window = gen + (total > ctx ? total - ctx : 0);
        forward(window, len, next, 0);
        int tk = next[len - 1];
        if (tk == EOS) break;
        gen[total++] = tk;
        decode_ids(&tk, 1);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.bin [-p prompt] [-t temp] [--logits-debug]\n",
                argv[0]);
        return 1;
    }
    srand(1234);
    load_model(argv[1]);
    vocab_init();

    const char *prompt = NULL;
    float temp = 0.0f;
    int debug = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--logits-debug") == 0) debug = 1;
    }

    int d = (int)g_model.d_model;
    int ctx = (int)g_model.context_len;
    for (int L = 0; L < (int)g_model.n_layer; L++) {
        g_kcache[L] = calloc((size_t)ctx * (size_t)d, sizeof(float));
        g_vcache[L] = calloc((size_t)ctx * (size_t)d, sizeof(float));
    }

    if (prompt) {
        int ids[MAX_CTX];
        int n = bpe_encode(prompt, ids, MAX_CTX);
        if (debug) {
            int next[MAX_CTX];
            forward(ids, n, next, 1);
        }
        fputs(prompt, stdout);
        generate(ids, n, ctx, temp);
        fputs("\n", stdout);
        return 0;
    }

    static char line[4096];
    for (;;) {
        fputs("> ", stdout); fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;
        if (line[0] == 0) continue;
        int ids[MAX_CTX];
        int n = bpe_encode(line, ids, MAX_CTX);
        generate(ids, n, ctx, temp);
        fputs("\n", stdout);
    }
    return 0;
}
