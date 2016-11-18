#!/usr/bin/env python3

import scipy.sparse
import numpy as np
import scipy.stats
import sys
import logging
import operator as op


def n_choose_k(n, k):
    k = min(k, n - k)
    if k == 0:
        return 1
    numerator = 1
    for i in range(n, n - k, -1):
        numerator *= i
    denominator = 1
    for i in range(1, k + 1):
        denominator *= i
    return numerator // denominator


def analyze_mosaic_matrix(phasing_evidence,
                          sequencing_error_rate=0.02,
                          mosaic_prior=0.1):
    '''
    Calculates the probability of:
    1) Finding the observed data
    2) The mosaic variant being on haplotype 1
    3) The variant being mosaic
    Using a bayesian framework.
    '''

    e = sequencing_error_rate
    d = phasing_evidence
    p = mosaic_prior

    # Find the probability that the variant is on Haplotype 1 #
    prob_d_given_h1 = (1 - e) ** d[0, 1] * e ** d[1, 1]
    prob_d_given_h2 = (1 - e) ** d[1, 1] * e ** d[0, 1]
    prob_h1 = ((prob_d_given_h1 * 0.5) /
               (prob_d_given_h1 * 0.5 + prob_d_given_h2 * 0.5))
    prob_h2 = 1 - prob_h1

    # Find the normalized probability of observing this data #
    n_alt_reads = d[0, 1] + d[1, 1]
    n_ml_h1_reads = int((1 - e) * n_alt_reads + 0.499999)
    n_ml_e_reads = n_alt_reads - n_ml_h1_reads
    ml_prob_d0 = (1 - e) ** n_ml_h1_reads * e ** n_ml_e_reads
    ml_prob_d1 = (1 - e) ** n_ml_e_reads * e ** n_ml_h1_reads

    prob_data = ((n_choose_k(n_alt_reads, d[0, 1]) * prob_d_given_h1 +
                  n_choose_k(n_alt_reads, d[1, 1]) * prob_d_given_h2) /
                 (n_choose_k(n_alt_reads, n_ml_h1_reads) * ml_prob_d0 +
                  n_choose_k(n_alt_reads, n_ml_h1_reads) * ml_prob_d1))

    # Estimate the fraction of mosaicism #
    f_h1 = d[0, 0] / (d[0, 0] + d[0, 1])
    f_h2 = d[1, 0] / (d[1, 0] + d[1, 1])

    # Find the probability that the variant is mosaic #
    # Hard cutoffs if very little support for the mosaic variant #
    if prob_h1 > 0.5 and f_h1 < 2 * e:
        return (prob_data, prob_h1, 0)
    elif prob_h1 < 0.5 and f_h2 < 2 * e:
        return (prob_data, prob_h1, 0)
    else:
        prob_d_given_mos_h1 = (
            (1 - e) ** d[1, 0] * (e + f_h1) ** d[0, 0] *
            (1 - f_h1) ** d[0, 1])
        prob_d_given_mos_h2 = (
            (1 - e) ** d[0, 0] * (e + f_h2) ** d[1, 0] *
            (1 - f_h2) ** d[1, 1])
        prob_d_given_nm_h1 = (
            (1 - e) ** d[1, 0] * (2 * e) ** d[0, 0] * (1 - e) ** d[0, 1])
        prob_d_given_nm_h2 = (
            (1 - e) ** d[0, 0] * (2 * e) ** d[1, 0] * (1 - e) ** d[1, 1])

        prob_mos_given_d_h1 = (prob_d_given_mos_h1 * p /
                               (prob_d_given_mos_h1 * p +
                                prob_d_given_nm_h1 * (1 - p)))
        prob_mos_given_d_h2 = (prob_d_given_mos_h2 * p /
                               (prob_d_given_mos_h2 * p +
                                prob_d_given_nm_h2 * (1 - p)))

        prob_mosaic = (prob_mos_given_d_h1 * prob_h1 +
                       prob_mos_given_d_h2 * prob_h2)

        return (prob_data, prob_h1, prob_mosaic)


def matrix_to_str(np_matrix, n_rows=np.inf, n_cols=np.inf):
    '''
    Convert a subsection of a np matrix to a nice table.
    '''
    if n_cols > np_matrix.shape[1]:
        n_cols = np_matrix.shape[1]
    if n_rows > np_matrix.shape[0]:
        n_rows = np_matrix.shape[0]

    out_table = ''
    for index, v in np.ndenumerate(np_matrix[:n_rows, :n_cols]):
        if index[1] == 0 and index[0] > 0:
            out_table += '\n'
        elif index[1] > 0:
            out_table += '\t'
        out_table += str(v)
    return out_table


def phase_mosaic_var(variant_id, variant_barcodes):
    '''
    Performs phasing of the mosaic variant to nearby germline variants.
    Returns the number of barcodes used in the phasing, \
    the p-value of the variant as a mosaic and \
    the p-value of the variant being a false-positive.
    '''
    (variant_matrix, variant_index, barcode_index) = \
        construct_germline_barcode_matrix(variant_barcodes)

    logging.debug("Analyzed {} variants and {} barcodes".format(
            len(variant_index), len(barcode_index)))
#    logging.debug("Table is:\n{}".format(
#            matrix_to_str(variant_matrix, 10000, 50)))

    (haplotypes, confidence, n_seen) = get_haplotypes(variant_matrix)
    skip_barcode_indices = set()
    barcode_not_seen = set()
    barcode_discordant = set()
    for barcode_iter in range(0, len(haplotypes)):
        if n_seen[barcode_iter] == 0:
            barcode_not_seen.add(barcode_iter)
        elif confidence[barcode_iter] / n_seen[barcode_iter] < 0.5:
            barcode_discordant.add(barcode_iter)

    logging.debug("Removed {} barcodes due to discordance".format(
            len(barcode_discordant)))
    logging.debug("Removed {} barcodes due to not seen".format(
            len(barcode_not_seen)))
    skip_barcode_indices = barcode_not_seen | barcode_discordant

    return determine_mosaicism(haplotypes, skip_barcode_indices,
                               barcode_index, variant_id, variant_barcodes)


def construct_germline_barcode_matrix(variant_barcodes,
                                      dtype=np.dtype("int32")):
    '''
    Construct a sparse matrix of the variants \
    supported by the barcode information.
    '''
    # Find the size of the matrix   #
    # Index the matrix rows/columns #
    variant_index = {}
    barcode_index = {}
    variant_counter = 0
    barcode_counter = 0
    for variant, properties in variant_barcodes.items():
        if not properties["is_germline"]:  # Skip non-het germline vars
            continue
        variant_index[variant] = variant_counter
        variant_counter += 1
        for allele, barcode_list in properties.items():
            if not (type(allele) is int and type(barcode_list) is list):
                continue
            for barcode in barcode_list:
                if barcode not in barcode_index:
                    barcode_index[barcode] = barcode_counter
                    barcode_counter += 1

    variant_barcode_matrix = np.zeros(
            (variant_counter, barcode_counter), dtype=np.dtype("int32"))

    # Populate the matrix #
    for variant, properties in variant_barcodes.items():
        if not properties["is_germline"]:
            continue
        for allele, barcode_list in properties.items():
            if not (type(allele) is int and type(barcode_list) is list):
                continue
            for barcode in barcode_list:
                variant_barcode_matrix[variant_index[variant],
                                       barcode_index[barcode]] = allele + 1
    return (variant_barcode_matrix, variant_index, barcode_index)


def get_haplotypes(variant_matrix):
    '''
    Determine the haplotypes of the barcodes.
    Returns the haplotype (0 or 1) the number of variants \
    on the barcode and the haplotype confidence \
    (concordant variants - discordant variants).
    '''
    n_barcodes = variant_matrix.shape[1]
    haplotypes = [0] * n_barcodes
    confidence = [0] * n_barcodes  # n_concordant - n_discordant
    n_seen = [0] * n_barcodes  # Total number of times barcode was seen

    # Iterate over the matrix rows (variants) #
    start, stop = 0, 0
    nonzero = variant_matrix.nonzero()
    while stop < variant_matrix.shape[1]:
        n_discordant = 0
        is_flipped = False
        while stop < len(nonzero[0]) and nonzero[0][stop] == nonzero[0][start]:
            stop += 1

        # Check to see if we flip the variant #
        for nonzero_idx in range(start, stop):
            i, j = nonzero[0][nonzero_idx], nonzero[1][nonzero_idx]
            if variant_matrix[i, j] - 1 != haplotypes[j]:
                n_discordant += 1
        if n_discordant > (stop - start) / 2:
            is_flipped = True

        # Find the haplotypes #
        for nonzero_idx in range(start, stop):
            i, j = nonzero[0][nonzero_idx], nonzero[1][nonzero_idx]
            current = variant_matrix[i, j]
            n_seen[j] += 1
            if ((not is_flipped and haplotypes[j] == current - 1) or
                    (is_flipped and haplotypes[j] != current - 1)):
                confidence[j] += 1
            else:
                confidence[j] -= 1
                if confidence[j] < 0:
                    confidence[j] = abs(confidence[j])
                    haplotypes[j] = current - 1

        # Prepare for the next iteration #
        stop += 1
        start = stop
    return (haplotypes, confidence, n_seen)


def determine_mosaicism(haplotypes, skip_barcode_indices,
                        barcode_index, variant_id, variant_barcodes,
                        sequencing_error_rate=0.05):
    '''
    Given the barcodes that the haplotypes are a part of, \
    identify mosaic variants.
    Returns the number of barcodes used in the phasing, \
    the probability of the variant as a mosaic and the probability \
    of the variant begin a false-positive
    '''

    '''
    Phasing evidence is stored in a 2x2 table. \
    Axis 0 defines the status of the barcode, \
    axis 1 defines the status of the mosaic variant.
    '''
    phasing_evidence = np.array([[0, 0], [0, 0]])
    variant_properties = variant_barcodes[variant_id]
    for allele, barcodes in variant_properties.items():
        if not (type(allele) is int and type(barcodes) is list):
            continue
        for barcode in barcodes:
            # Skip barcodes not in germline variants #
            if barcode not in barcode_index:
                continue
            cur_index = barcode_index[barcode]
            if cur_index in skip_barcode_indices:
                continue
            haplotype = haplotypes[cur_index]
            phasing_evidence[haplotype, allele] += 1

    logging.debug("Constructed phasing matrix {}".format(
            str(phasing_evidence)))

    depth = sum(sum(phasing_evidence))
    prob_data, prob_h1, prob_mosaic = (
        analyze_mosaic_matrix(phasing_evidence, sequencing_error_rate))

    return (depth, prob_mosaic, prob_data)
