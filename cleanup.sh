#!/bin/sh

find . -iname "*.jpg" -exec echo "{}" \; -exec sh -c 'djpeg -fast -grayscale -onepass "$1" > /dev/null' _ "{}" \;

