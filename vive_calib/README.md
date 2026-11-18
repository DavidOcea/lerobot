# VIVE 位姿推流（阶段 3：网络链路 x86↔Jetson）

机器分工（把 Vive SDK 和机器人控制物理隔离）：

| 文件 | 跑哪 | 语言/依赖 | 职责 |
|---|---|---|---|
| `sender.c` | x86 | C，链 libsurvive | 从 libsurvive 拿 6-DoF 位姿 → UDP 推 Jetson |
| `receiver.py` | Jetson（机器人） | Python 纯标准库 | 收 UDP → 按设备+序号去重/乱序丢弃 → 缓存最新位姿 |

## 消息格式（56 字节，小端）

| 偏移 | 字段 | 类型 | 说明 |
|---|---|---|---|
| 0 | dev | `char[16]` | 设备名（"WM0" 等，`\0` 补齐）|
| 16 | ts | `double` | 单调时钟秒 |
| 24 | seq | `uint32` | 全局递增序号 |
| 28 | pos | `float×3` | x, y, z |
| 40 | quat | `float×4` | qx, qy, qz, qw |

对应 Python：`struct.unpack("<16s d I 7f", data)`（共 56 字节）。

## 编译 sender.c（在 x86 上）

先找到 libsurvive 的头文件和库：

```bash
LH=/home/smai/workspace/dc_dir/vive/libsurvive   # 你的 libsurvive 根目录
find "$LH" -name "libsurvive.so"                   # 找到 .so 实际路径
```

然后编译（把 `-I`/`-L` 换成上面找到的实际路径）：

```bash
gcc -O2 -I"$LH/include" -I"$LH/redist" sender.c \
    -L<libsurvive.so 所在目录> -lsurvive -lm -lpthread -o sender
```

> 编译报错（头文件名/库路径对不上）就把报错贴回来，我按你的实际目录调整。

## 本机回环测试（先验证，别急着跨机）

1. 把 `sender.c` 里 `JETSON_IP` 改成 `"127.0.0.1"`，重新编译。
2. 终端 1 起接收端：
   ```bash
   python3 receiver.py --port 12345 --verbose
   ```
3. 终端 2 跑发送端：
   ```bash
   ./sender
   ```

预期：接收端刷出 `[WM0] seq=... pos=(...) quat=(...)`，数值跟着 tracker 动。
如果只有 `[sender] sent N packets` 而接收端没反应，先查 IP/端口/防火墙。

## 跨机联调

1. `sender.c` 里 `JETSON_IP` 改成 Jetson 实际 IP，重编译。
2. Jetson 上：`python3 receiver.py --port 12345`。
3. 两边同一网段，**有线最好**（原计划网线直连）。

## 注意

- `pose->Rot` 顺序假定为 linmath 的 `{x,y,z,w}`（w 在末位）。若后续标定时发现姿态系统性错误，先查这里。
- sender 重启会让 seq 回零，接收端靠「seq 大幅回退」检测重置；正常测试两端一起启停即可。
- 本目录 README 按仓库 `.gitignore` 的 `*.md` 规则会被忽略，提交时需 `git add -f`。
