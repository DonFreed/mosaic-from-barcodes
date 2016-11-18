#!/usr/bin/env python3

import sys
import argparse
import logging
import re
import phasing
import math

log_format = "%(filename)s::%(funcName)s [%(levelname)s] %(message)s"

vcf_header_lines = '''
##INFO=<ID=MOSAIC,Number=0,Type=Flag,Description="Variant is mosaic"
##INFO=<ID=MOSAICP,Number=1,Type=Integer,Description=\
"Phred-scaled probability that the variant is mosaic"
##INFO=<ID=PHASEDP,Number=1,Type=Integer,Description=\
"Number of phase-informative barcodes"
##INFO=<ID=PHASETP,Number=1,Type=Integer,Description=\
"Phred-scaled probability that the variant is a true-positive from phasing"
'''


class InputError(Exception):
    '''
    An exception class for handling user input errors.

    Attributes:
        message -- Explanation of the error
    '''
    def __init__(self, message):
        self.message = message


class VariantList(list):
    '''
    Contains a list of VCF lines split on tabs.

    Attributes:
        cur_phase_idx -- The index of the next variant to phase

    Methods:
        __init__ -- Initalization
        pop_past_variants -- Pop variants some distance
                             behind a genomic position.
        get_next_phased_var -- The next variant(s) to phase
                               behind some genomic position.
    '''
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.cur_phase_idx = 0

    def pop_past_variants(self, chrom, pos, distance):
        while (len(self) > 0 and
               (self[0][0] != chrom or
                (self[0][0] == chrom and
                 int(self[0][1]) < int(pos) - distance))):
            if self.cur_phase_idx > 0:
                self.cur_phase_idx -= 1
            yield self.pop(0)

    def next_var_to_phase(self, chrom, pos, distance):
        while (self.cur_phase_idx < len(self) and
               (self[self.cur_phase_idx][0] != chrom or
                (self[self.cur_phase_idx][0] == chrom and
                 int(self[self.cur_phase_idx][1]) < int(pos) - distance))):
            yield self[self.cur_phase_idx]
            self.cur_phase_idx += 1


class VariantBarcodes(dict):
    '''
    Contains a dictionary of variants pointing to a dictionary \
of alleles which point to lists of barcodes supporting the alleles.
    '''
    pass


def get_args():
    '''
    Process program arguments using argparse.
    '''

    parser = argparse.ArgumentParser(
        description="Find mosaic variants from barcoded variant calls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input #
    parser.add_argument(
        "--infile",
        help="The input VCF file with barcoded variant calls [stdin]")
    # Variant parsing #
    parser.add_argument(
        "--marks_germline",
        nargs='*',
        help="One or more INFO field tags which mark the variant as germline")
    parser.add_argument(
        "--absent_marks_germline",
        nargs="*",
        help="One or more INFO field tags which mark variants "
        "WITHOUT these tags as germline variants")
    parser.add_argument(
        "--marks_possible_mosaic",
        nargs='*',
        help="One or more INFO field tags which mark the variant "
        "as a possible mosaic")
    parser.add_argument(
        "--absent_marks_possible_mosaic",
        nargs="*",
        help="One or more INFO field tags which mark variants WITHOUT "
        "these tags as possible mosaic variants")
    # Output #
    parser.add_argument(
        "--outfile",
        help="An output VCF file with mosaic variants marked in the "
        "INFO field [stdout]")
#    parser.add_argument(
#        "--out_mosaic_text",
#        help="An optional additonal output tab-delimited text file "
#        "describing identified mosaic variants")
    # Misc #
    parser.add_argument(
        "--ignore_all_filters",
        action="store_true",
        help="Ignore the FILTER column of the input VCF")
    parser.add_argument(
        "--ignore_filter",
        nargs="*",
        help="Ignore the given filters")
    parser.add_argument(
        "--max_phase_distance",
        type=int,
        default=500000,
        help="The maximum distance to attempt to phase a variant")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        help="Print more information, debug with -vv")
    return parser.parse_args()


def perform_phasing(cur_phase_var, variant_id, variant_barcodes):
    phase_chrom, phase_pos, phase_ref, phase_alt = (
            cur_phase_var[:2] + cur_phase_var[3:5])
    variant_id = ';'.join([phase_chrom, phase_pos, phase_ref, phase_alt])
    if variant_id not in variant_barcodes or (
            not variant_barcodes[variant_id]["is_mosaic"]):
        return
    logging.info("Phasing the variant at {}:{}".format(phase_chrom, phase_pos))
    (depth, prob_mosaic, prob_data) = (phasing.phase_mosaic_var(
            variant_id, variant_barcodes))
    additional_tags = []
    logging.debug("depth-{}, prob_mosaic-{}, prob_fp-{}".format(
            depth, prob_mosaic, prob_data))
    if prob_data > 0.01 and prob_mosaic > 0.2:
        additional_tags.append("MOSAIC")
    additional_tags.append("PHASEDP={}".format(depth))
    additional_tags.append("PHASETP={}".format(prob_data))
    additional_tags.append("MOSAICP={}".format(prob_mosaic))
    cur_phase_var[7] = (cur_phase_var[7] + ';' + ';'.join(additional_tags))
    return


def main(args):
    if not args:
        args = get_args()

    log_level = logging.WARNING
    if args.verbose:
        if args.verbose == 1:
            log_level = logging.INFO
        else:
            log_level = logging.DEBUG
    logging.basicConfig(format=log_format, level=log_level)

    if args.infile:
        args.in_fh = open(args.infile)
    else:
        args.in_fh = sys.stdin

    if args.outfile:
        args.out_fh = open(args.outfile, 'w')
    else:
        args.out_fh = sys.stdout

#    if args.out_mosaic_text:
#        args.out_text_fh = open(args.out_mosaic_text, 'w')

    # Handle the INFO field tags #
    if (args.marks_germline and args.absent_marks_germline or
            (not args.marks_germline and
             not args.absent_marks_germline)):
        raise InputError("Please specify one of '--marks_germline' "
                         "or 'absent_marks_germline'")
    if (args.marks_possible_mosaic and args.absent_marks_possible_mosaic or
            (not args.marks_possible_mosaic and
             not args.absent_marks_possible_mosaic)):
        raise InputError("Please specify only '--marks_possible_mosaic' "
                         "or 'absent_marks_possible_mosaic'")
    variant_tags = {
        "marks_germline": set(),
        "absent_marks_germline": set(),
        "marks_possible_mosaic": set(),
        "absent_marks_possible_mosaic": set()
        }
    for criteria, tag_set in variant_tags.items():
        if not getattr(args, criteria):
            continue
        logging.debug(type(getattr(args, criteria)))
        logging.debug(getattr(args, criteria))
        for tag in getattr(args, criteria):
            tag_set.add(tag)
    logging.debug("variant_tags are {}".format(str(variant_tags)))

    logging.info("Starting analysis")

    # Data containers #
    variant_list = VariantList()
    variant_barcodes = VariantBarcodes()

    # main loop #
    header_written = False
    sample_name = ''
    ignore_filters = set()
    if args.ignore_filter:
        ignore_filters = set(args.ignore_filter)
    for line in args.in_fh:
        line = line.rstrip()

        # Parse the header lines #
        if line.startswith("##"):
            if line.startswith("##INFO") and not header_written:
                print(vcf_header_lines, file=args.out_fh)
                header_written = True
            print(line, file=args.out_fh)
            continue
        elif line[0] == '#':
            if not header_written:
                print(vcf_header_lines, file=args.out_fh)
                header_written = True
            line = line.split('\t')
            if len(line) > 10 or len(line) < 10:
                raise InputError("Please only run with a single sample")
            sample_name = line[9]
            print('\t'.join(line), file=args.out_fh)
            continue

        # The variant data #
        current_var_marks = dict(zip(variant_tags.keys(), [False] * 4))
        is_mosaic, is_germline = False, False
        line = line.split('\t')
        chrom, pos, ref, alt = line[0], line[1], line[3], line[4]

        # Parser the FILTER field #
        if (not args.ignore_all_filters and
                (line[6] != 'PASS' and line[6] != '.' and
                 line[6] not in ignore_filters)):
            logging.info("The variant at {}:{} was skipped due to it's "
                         "filter field - {}".format(chrom, pos, line[6]))
            variant_list.append(line)
            continue

        # Parse the INFO field #
        for tag in line[7].split(';'):
            if '=' in tag:
                tag, value = tag.split('=')
            for criteria, tag_set in variant_tags.items():
                if tag in tag_set:
                    current_var_marks[criteria] = True
        if (current_var_marks["marks_germline"] or
                (variant_tags["absent_marks_germline"] and
                 not current_var_marks["absent_marks_germline"])):
            is_germline = True
        if (current_var_marks["marks_possible_mosaic"] or
                (variant_tags["absent_marks_possible_mosaic"] and
                 not current_var_marks["absent_marks_possible_mosaic"])):
            is_mosaic = True
        logging.debug("Found: {}".format(str(current_var_marks)))
        if is_mosaic and is_germline:
            logging.warning("The variant at {}:{} has tags of both a "
                            "mosaic and germline variant. "
                            "Skipping".format(chrom, pos))
            variant_list.append(line)
            continue
        logging.debug("is mosaic: {}, is germline: {}".format(
                is_mosaic, is_germline))

        # Parse the FORMAT field #
        bx_idx = -1
        for i, format_tag in enumerate(line[8].split(':')):
            if format_tag == "BX":
                bx_idx = i
        if bx_idx < 0:
            logging.warning("The variant at {}:{} "
                            "contains no BX tags. Skipping "
                            "phasing for this variant.".format(chrom, pos))
            variant_list.append(line)
            continue

        # Parse sample field #
        genotype = re.split(r'\|', line[9].split(':')[0])
        if len(genotype) <= 1 or genotype[0] == genotype[1]:  # Skip homozygous
            logging.info(
                    "Skipped {}:{} as it is homozygous".format(chrom, pos))
            variant_list.append(line)
            continue
        genotype = {genotype[0]: 0, genotype[1]: 1}

        variant_id = ';'.join([chrom, pos, ref, alt])
        if variant_id in variant_barcodes:
            del variant_barcodes[variant_id]
            logging.warning(
                    "The input VCF contains duplicate "
                    "records for {}:{}".format(chrom, pos))
        variant_barcodes[variant_id] = {}
        variant_barcodes[variant_id]["is_germline"] = is_germline
        variant_barcodes[variant_id]["is_mosaic"] = is_mosaic
        for allele, cur_barcodes in enumerate(
                line[9].split(':')[bx_idx].split(',')):
            if str(allele) not in genotype:  # Skip not genotyped
                continue
            if genotype[str(allele)] not in variant_barcodes[variant_id]:
                variant_barcodes[variant_id][genotype[str(allele)]] = []
            for bc_string in cur_barcodes.split(';'):
                barcode = bc_string.split('_')[0]  # Ignore barcode quality
                variant_barcodes[variant_id][
                        genotype[str(allele)]].append(barcode)

        variant_list.append(line)

        # Perform phasing of previous variants #
        for cur_phase_var in variant_list.next_var_to_phase(
                    line[0], line[1], args.max_phase_distance):
            perform_phasing(cur_phase_var,
                            variant_id, variant_barcodes)

        # Output previous variants #
        for out_var in variant_list.pop_past_variants(
                line[0], line[1], 2 * args.max_phase_distance):
            print('\t'.join(out_var), file=args.out_fh)
            # Clean up #
            out_chrom, out_pos, out_ref, out_alt = out_var[:2] + out_var[3:5]
            variant_id = ';'.join([out_chrom, out_pos, out_ref, out_alt])
            if variant_id not in variant_barcodes:
                continue
            for allele, barcode_list in variant_barcodes[variant_id].items():
                # Skip other fields #
                if not (type(allele) is int and type(barcode_list) is list):
                    continue
            del variant_barcodes[variant_id]

    # Final phasing #
    for cur_phase_var in variant_list.next_var_to_phase(
            'END', 100, args.max_phase_distance):
        perform_phasing(cur_phase_var, variant_id,
                        variant_barcodes)

    # Final output #
    for out_var in variant_list.pop_past_variants(
            line[0], line[1], 2 * args.max_phase_distance):
        print('\t'.join(out_var), file=args.out_fh)

    if args.infile:
        args.in_fh.close()
    if args.outfile:
        args.out_fh.close()
#    if args.out_mosaic_text:
#        args.out_text_fh.close()

    logging.info("Analysis finsihed")


if __name__ == "__main__":
    main(None)
