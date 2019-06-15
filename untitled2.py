#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 05:16:48 2019

@author: jeff
"""

ii = 0
with open('subset20.vcf', 'r') as s20:
    with open('ref20.vcf', 'w') as r20:
        with open('missing20.vcf', 'w') as m20:
            while ii < 252:
                ii += 1
                l = s20.readline()
                r20.write(l)
                m20.write(l)
            ls = s20.readline()
            ln = ls.strip().split()
            r20.write('\t'.join(ln[0:2013]) + '\n')
            m20.write('\t'.join(ln[0:9]) + '\t' + '\t'.join(ln[2013:]) + '\n')
            for l in s20:
                line = l.strip().split()
                r20.write('\t'.join(line[0:2013]) + '\n')
                m20.write('\t'.join(line[0:9]) + '\t' + '\t'.join(line[2013:]) + '\n')
