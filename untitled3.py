#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:05:33 2019

@author: jeff
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def calc_geno(haplo):
    if haplo == '1|1':
        return 2
    elif haplo == '0|1' or haplo == '1|0':
        return 1
    elif haplo == '0|0':
        return 0
    else:
        print('ERROR: {}'.format(haplo))
        return 0

with open('complete20.vcf', 'r') as og, open('missing20.vcf', 'r') as missing, open('test.vcf', 'r') as imputed:
    accs = np.load('accs.npy')
    maf_dict = {}
    for ii in range(10):
        imputed.readline()
    for jj in range(253):
        missing.readline()
        og.readline()
    i = 0
    results = []
    results_1 = []
    maf = []
    for l in missing:
        i += 1
        mline = l.split()
        ol = og.readline()
        imp = imputed.readline()
        if mline[9] == '.|.':
            oline = ol.split()
            imline = imp.split()
            correct = 0
            correct_phased = 0
            minor_allele_sum = 0
            for ii in range(9, 2013):
                minor_allele_sum += calc_geno(oline[ii])
            tmaf = minor_allele_sum/4000.0
            maf_dict[i - 1] = tmaf
            if tmaf > 0.5:
                maf.append(1 - tmaf)
                maf_dict[i - 1] = 1 - tmaf
            else:
                maf.append(tmaf)
            for offset in range(500):
                imp_geno = calc_geno(imline[9 + offset])
                og_geno = calc_geno(oline[2013 + offset])
                if imp_geno == og_geno:
                    correct += 1
                if imline[9 + offset] == oline[2013 + offset]:
                    correct_phased += 1
            results.append(correct/500.0)
            results_1.append(correct_phased/500.0)
    print(np.mean(results))
    print(np.mean(results_1))
    print(maf)
    print(results)
    print(results_1)
    print(accs)
    lstm_res = []
    lmaf = []
    for tp in accs:
        lstm_res.append(tp[1])
        lmaf.append(maf_dict[tp[0]])
    plt.plot(maf, results, 'o', label="BEAGLE")
    plt.plot(lmaf, lstm_res, 'ro', label="BiLSTM")
    plt.xlabel('Minor Allele Frequency')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig('beagle_blstm.pdf')