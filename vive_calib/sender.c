// vive_calib/sender.c
// VIVE 位姿推流发送端 —— 跑在 x86，链接 libsurvive。
// 职责：survive_init → 装 pose 回调 → 循环 survive_poll；
//       回调里把每片 tracker 的 6-DoF 位姿打包成固定长度 UDP 消息发往 Jetson。
//
// 编译（在 libsurvive 根目录，按实际 -I/-L 调整）：
//   gcc -O2 -I<libsurvive>/include -I<libsurvive>/redist \
//       sender.c -L<libsurvive.so 所在目录> -lsurvive -lm -lpthread -o sender

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <libsurvive/survive.h>

// ---- 配置：改这里 ----
#define JETSON_IP   "192.168.1.100"  // Jetson 的 IP；本机回环测试改成 "127.0.0.1"
#define JETSON_PORT 12345            // 与 receiver.py 约定一致
// -----------------------

// 固定长度二进制消息（与 receiver.py 的 "<16s d I 7f" 字节对齐）：
//   dev[16] 设备名（"WM0" 等，\0 补齐）
//   ts       double 单调时钟秒
//   seq      uint32 全局递增序号
//   pos[3]   float x y z
//   quat[4]  float qx qy qz qw  （linmath 约定 {x,y,z,w}，w 在末位）
#define DEVNAME_LEN 16
typedef struct __attribute__((packed)) {
    char     dev[DEVNAME_LEN];
    double   ts;
    uint32_t seq;
    float    pos[3];
    float    quat[4];
} pose_msg_t;

static int              sockfd = -1;
static struct sockaddr_in dst;
static uint32_t         g_seq = 0;
static uint32_t         g_sent = 0;

static void pose_cb(SurviveObject *so, unsigned long timecode, const LinmathPose *pose) {
    (void)timecode;  // 灯塔 timecode；本方案用 clock_gettime 单调钟做时间戳
    pose_msg_t m;
    struct timespec t;
    memset(&m, 0, sizeof(m));

    snprintf(m.dev, DEVNAME_LEN, "%s", so->codename);
    clock_gettime(CLOCK_MONOTONIC, &t);
    m.ts  = (double)t.tv_sec + (double)t.tv_nsec / 1e9;
    m.seq = g_seq++;

    for (int i = 0; i < 3; i++) m.pos[i]  = pose->Pos[i];
    for (int i = 0; i < 4; i++) m.quat[i] = pose->Rot[i];

    ssize_t n = sendto(sockfd, &m, sizeof(m), 0,
                       (struct sockaddr *)&dst, sizeof(dst));
    if (n < 0) {
        perror("sendto");
        return;
    }
    if (++g_sent % 500 == 0) {
        printf("[sender] sent %u packets, last seq=%u\n", g_sent, m.seq);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    SurviveContext *ctx = survive_init(argc, argv);
    if (!ctx) { fprintf(stderr, "survive_init failed\n"); return 1; }

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); survive_close(ctx); return 1; }

    memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(JETSON_PORT);
    if (inet_pton(AF_INET, JETSON_IP, &dst.sin_addr) != 1) {
        fprintf(stderr, "bad IP: %s\n", JETSON_IP);
        close(sockfd); survive_close(ctx); return 1;
    }

    survive_install_pose_fn(ctx, pose_cb);

    // 主循环：survive_poll 返回 0 表示继续，非 0 退出
    while (survive_poll(ctx) == 0) { }

    close(sockfd);
    survive_close(ctx);
    return 0;
}
