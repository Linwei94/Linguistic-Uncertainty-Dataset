#!/bin/bash

PATH=$(pwd)
BUILD_PATH="${PATH}/build"
SCRIPT_PATH="${BUILD_PATH}/scripts"
CACHE_PATH="${BUILD_PATH}/cache"

build_type=$1

if [ "$build_type" = "-c" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_PATH"
    mkdir -p "$BUILD_PATH"
    python build.py -b -dataset='1.csv'
else
    if [ ! -d "$BUILD_PATH" ]; then
        mkdir -p "$BUILD_PATH"
    fi
    python build.py -dataset='question.csv'
fi
