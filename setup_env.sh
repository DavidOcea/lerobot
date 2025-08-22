#!/bin/bash
conda activate ros2_env
sudo chmod 666 /dev/ttyTHS1
sudo chmod 666 /dev/ttyTHS2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6 python