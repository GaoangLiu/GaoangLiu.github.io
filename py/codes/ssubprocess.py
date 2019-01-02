#!/usr/local/bin/env python3 
import os, subprocess

# subprocess.run("exit 1", shell=True, check=True)
# output = subprocess.run(["leetcode",  "user"], )

result = subprocess.run(["ls", "-lt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# print(result.stdout)

try:
	subprocess.run(["sleep", "10"], timeout=3)
except subprocess.TimeoutExpired:
	print("Time out. Do something else.")

