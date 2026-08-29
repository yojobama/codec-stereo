#include "codec_stereo/cs_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads one whitespace-delimited token from a PNM/PFM header, skipping
   '#' comments, per the Netpbm header grammar. */
static int read_header_token(FILE *f, char *buf, size_t buf_sz) {
    int c;
    size_t n = 0;

    /* skip whitespace and comments */
    for (;;) {
        c = fgetc(f);
        if (c == EOF) return -1;
        if (c == '#') {
            while ((c = fgetc(f)) != '\n' && c != EOF) {}
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        break;
    }

    while (c != EOF && c != ' ' && c != '\t' && c != '\r' && c != '\n') {
        if (n + 1 >= buf_sz) return -1;
        buf[n++] = (char)c;
        c = fgetc(f);
    }
    buf[n] = '\0';
    return 0;
}

int cs_pgm_read(const char *path, uint8_t **data_out, int *w_out, int *h_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[16], tok[32];
    int w, h, maxval;

    if (read_header_token(f, magic, sizeof magic) != 0 || strcmp(magic, "P5") != 0) {
        fclose(f);
        return -1;
    }
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    w = atoi(tok);
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    h = atoi(tok);
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    maxval = atoi(tok);
    if (w <= 0 || h <= 0 || maxval != 255) { fclose(f); return -1; }

    /* Exactly one whitespace byte separates the header from binary data;
       read_header_token already consumed it as the token delimiter. */
    uint8_t *data = (uint8_t *)malloc((size_t)w * (size_t)h);
    if (!data) { fclose(f); return -1; }
    if (fread(data, 1, (size_t)w * (size_t)h, f) != (size_t)w * (size_t)h) {
        free(data);
        fclose(f);
        return -1;
    }
    fclose(f);

    *data_out = data;
    *w_out = w;
    *h_out = h;
    return 0;
}

int cs_pgm_write(const char *path, const uint8_t *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h;
    int ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok ? 0 : -1;
}

int cs_pfm_read(const char *path, float **data_out, int *w_out, int *h_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[16], tok[32];
    int w, h;
    float scale;

    if (read_header_token(f, magic, sizeof magic) != 0 || strcmp(magic, "Pf") != 0) {
        /* Only single-channel PFM ("Pf") is supported; "PF" (color) is not. */
        fclose(f);
        return -1;
    }
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    w = atoi(tok);
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    h = atoi(tok);
    if (read_header_token(f, tok, sizeof tok) != 0) { fclose(f); return -1; }
    scale = (float)atof(tok);
    if (w <= 0 || h <= 0 || scale == 0.0f) { fclose(f); return -1; }

    int little_endian_file = scale < 0.0f; /* PFM: negative scale = little-endian */

    size_t n = (size_t)w * (size_t)h;
    float *raw = (float *)malloc(n * sizeof(float));
    if (!raw) { fclose(f); return -1; }
    if (fread(raw, sizeof(float), n, f) != n) {
        free(raw);
        fclose(f);
        return -1;
    }
    fclose(f);

    /* Byte-swap if the file's endianness doesn't match the host's. This
       library only targets little-endian hosts (x86/ARM), so swap iff the
       file declares itself big-endian. */
    if (!little_endian_file) {
        for (size_t i = 0; i < n; i++) {
            uint8_t *b = (uint8_t *)&raw[i];
            uint8_t t;
            t = b[0]; b[0] = b[3]; b[3] = t;
            t = b[1]; b[1] = b[2]; b[2] = t;
        }
    }

    /* PFM stores rows bottom-to-top; flip to top-to-bottom row-major. */
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) { free(raw); return -1; }
    for (int y = 0; y < h; y++) {
        memcpy(out + (size_t)y * w, raw + (size_t)(h - 1 - y) * w, (size_t)w * sizeof(float));
    }
    free(raw);

    *data_out = out;
    *w_out = w;
    *h_out = h;
    return 0;
}

int cs_pfm_write(const char *path, const float *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "Pf\n%d %d\n-1.0\n", w, h); /* -1.0 => little-endian */
    for (int y = h - 1; y >= 0; y--) {
        if (fwrite(data + (size_t)y * w, sizeof(float), (size_t)w, f) != (size_t)w) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

void cs_shift_gray8(const uint8_t *src, int src_stride,
                     uint8_t *dst, int dst_stride,
                     int w, int h, int shift_px) {
    for (int y = 0; y < h; y++) {
        const uint8_t *srow = src + (size_t)y * src_stride;
        uint8_t *drow = dst + (size_t)y * dst_stride;
        for (int x = 0; x < w; x++) {
            int sx = x - shift_px;
            if (sx < 0) sx = 0;
            if (sx >= w) sx = w - 1;
            drow[x] = srow[sx];
        }
    }
}
