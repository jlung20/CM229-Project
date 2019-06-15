#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:45:52 2019

@author: jeff
"""

import numpy as np

ii = 0
jj = 0
input_mat = []
masked_input_mat = []
all_masked = []
mask = [1 for ii in range(5)] + [0 if np.random.randint(0, 10) == 9 else 1 for ii in range(990)] + [1 for ii in range(5)]
print(len(mask))
metadata_list = []
with open("ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf", "r") as f:
    with open("subset20.vcf", "w") as s20, open("complete20.vcf", "w") as c20:
        while ii < 253:
            l = f.readline()
            s20.write(l)
            c20.write(l)
            ii += 1
        pFlag = False
        while jj < 1000:
            l = f.readline()
            line = l.strip().split()
            geno = []
            zcount = 0
            ocount = 0
            kk = 0
            for ph in line[9:]:
                if ph == '0|0':
                    geno.append('0')
                    if kk < 2004:
                        zcount += 1
                elif ph == '1|1':
                    geno.append('2')
                    if kk < 2004:
                        ocount += 1
                elif ph == '0|1' or ph == '1|0':
                    geno.append('1')
                # Filtering sites that aren't biallelic
                else:
                    pFlag = True
                    break
                kk += 1
            if not pFlag and zcount < 1984 and ocount < 1984:
                c20.write(l)
                tmet = [line[2], line[0], line[1]]
                tstr = ', '.join(tmet) + '\n'
                metadata_list.append(tstr)
                input_mat.append(geno)
                if mask[jj]:
                    all_masked.append(geno)
                    masked_input_mat.append(geno)
                    s20.write(l)
                else:
                    all_masked.append(['NaN'] * 2504)
                    masked_input_mat.append(geno[0:2004] + ([' '] * 500))             
                    s20.write('\t'.join(line[0:9]) + '\t')
                    s20.write('\t'.join(line[9:2013]) + '\t')
                    s20.write('\t'.join(['.|.'] * 500) + '\n')
                jj += 1
            pFlag = False
        with open("original.txt", "w") as og:
            for line in input_mat:
                og.write(','.join(line) + '\n')
        with open("partial_masked.txt", "w") as pm:
            for line in masked_input_mat:
                pm.write(','.join(line) + '\n')
        with open("all_masked.txt", "w") as am:
            for line in all_masked:
                am.write(','.join(line) + '\n')
        with open("SNP_def.txt", "w") as sd:
            sd.write("    3.00  = FILE FORMAT VERSION NUMBER.\n")
            for line in metadata_list:
                sd.write(line)
