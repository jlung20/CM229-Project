#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 04:18:03 2019

@author: jeff
"""

with open('hello.bim', 'r') as f:
    with open('hello_new.bim', 'w') as fw:
        fw.write('    3.00  = FILE FORMAT VERSION NUMBER.')
        print(f.readline())
        for l in f:
            print(l)
            line = l.split()
            print(line)
            lnew = [line[1], line[0], line[3]]
            fw.write(', '.join(lnew) + '\n')
