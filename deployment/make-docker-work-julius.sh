#!/usr/bin/env bash

ip link add br0 type bridge
ip addr add 10.151.0.0/24 dev br0
ip link set br0 up