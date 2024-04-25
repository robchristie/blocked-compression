#!/usr/bin/env bash

cmake -G Ninja -S src -B build
cmake --build build
