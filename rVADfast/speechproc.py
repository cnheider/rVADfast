#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import math
from copy import deepcopy

import numpy
from scipy.io import wavfile

__author__ = "Achintya Kumar Sarkar and Zheng-Hua Tan"

__doc__ = r"""
# References
# Z.-H. Tan and B. Lindberg, Low-complexity variable frame rate analysis for speech recognition and voice
# activity detection.
# IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
# Achintya Kumar Sarkar and Zheng-Hua Tan 2017
# Version: 02 Dec 2017

# Editions by Christian Heider Nielsen
"""

__all__ = [
    "speech_wave",
    "enframe",
    "sflux",
    "snre_highenergy",
    "snre_vad",
    "pitchblockdetect",
]


def speech_wave(wav_file_name):
    (fs, sig) = wavfile.read(wav_file_name)
    if sig.dtype == "int16":
        nb = 16  # -> 16-bit wav files
    elif sig.dtype == "int32":
        nb = 32  # -> 32-bit wav files
    max_nb = float(2 ** (nb - 1))
    sig = sig / (max_nb + 1.0)
    return fs, sig


def enframe(speech, fs, winlen, ovrlen):
    N, flth, foVr = (
        len(speech),
        int(numpy.fix(fs * winlen)),
        int(numpy.fix(fs * ovrlen)),
    )

    if len(speech) < flth:
        print("speech file length shorter than window length")
        exit()

    frames = int(math.ceil((N - flth) / foVr))
    slen = (frames - 1) * foVr + flth

    if len(speech) < slen:
        signal = numpy.concatenate((speech, numpy.zeros((slen - N))))
    else:
        signal = deepcopy(speech)

    idx = (
        numpy.tile(numpy.arange(0, flth), (frames, 1))
        + numpy.tile(numpy.arange(0, frames * foVr, foVr), (flth, 1)).T
    )
    idx = numpy.array(idx, dtype=numpy.int64)

    return signal[idx]


def sflux(data, fs, winlen, ovrlen, nftt):
    eps = numpy.finfo(float).eps

    xf = enframe(data, fs, winlen, ovrlen)  # framing
    w = numpy.matrix(numpy.hamming(int(fs * winlen)))
    w = numpy.tile(w, (numpy.size(xf, axis=0), 1))

    xf = numpy.multiply(xf, w)  # apply window
    # fft
    ak = numpy.abs(numpy.fft.fft(xf, nftt))
    idx = range(0, int(nftt / 2) + 1)
    ak = ak[:, idx]
    num = numpy.exp(float(1 / len(idx)) * numpy.sum(numpy.log(ak + eps), axis=1))
    den = float(1 / len(idx)) * numpy.sum(ak, axis=1)

    ft = (num + eps) / (den + eps)

    flen, fsh10 = int(numpy.fix(fs * winlen)), int(numpy.fix(fs * ovrlen))
    nfr10 = int(numpy.floor((len(data) - (flen - fsh10)) / fsh10))

    return ft, flen, fsh10, nfr10


def snre_highenergy(
    fdata,
    nfr10,
    flen,
    fsh10,
    energy_floor,
    pv01,
    pvblk,
    *,
    d_expl=18,
    d_expr=18,
    seg_threshold=0.25
):
    ## ---*******- important *******
    # here [0] index array element has  not used

    fdata_ = deepcopy(fdata)
    pv01_ = deepcopy(pv01)
    pvblk_ = deepcopy(pvblk)

    fdata_ = numpy.insert(fdata_, 0, "inf")
    pv01_ = numpy.insert(pv01_, 0, "inf")
    pvblk_ = numpy.insert(pvblk_, 0, "inf")  # TODO: unused

    # energy estimation
    e = numpy.zeros(nfr10, dtype="float64")
    e = numpy.insert(e, 0, "inf")

    for i in range(1, nfr10 + 1):
        for j in range(1, flen + 1):
            e[i] = e[i] + numpy.square(fdata_[(i - 1) * fsh10 + j])

        if numpy.less_equal(e[i], energy_floor):
            e[i] = energy_floor

    e_min = numpy.ones(nfr10)
    e_min = numpy.insert(e_min, 0, "inf")
    ne_seg = 200

    if numpy.less(nfr10, ne_seg):
        ne_seg = nfr10

    for i in range(1, int(numpy.floor(nfr10 / ne_seg)) + 1):
        e_y = numpy.sort(e[range((i - 1) * ne_seg + 1, (i * ne_seg) + 1)])
        e_y = numpy.insert(e_y, 0, "inf")

        e_min[range((i - 1) * ne_seg + 1, i * ne_seg + 1)] = e_y[
            int(numpy.floor(ne_seg * 0.1))
        ]
        if numpy.not_equal(i, 1):
            e_min[range((i - 1) * ne_seg + 1, i * ne_seg + 1)] = (
                0.9 * e_min[(i - 1) * ne_seg] + 0.1 * e_min[(i - 1) * ne_seg + 1]
            )

    if numpy.not_equal(i * ne_seg, nfr10):
        e_y = numpy.sort(e[range((i - 1) * ne_seg + 1, nfr10 + 1)])
        e_y = numpy.insert(e_y, 0, "inf")

        e_min[range(i * ne_seg + 1, nfr10 + 1)] = e_y[
            int(numpy.floor((nfr10 - (i - 1) * ne_seg) * 0.1))
        ]
        e_min[range(i * ne_seg + 1, nfr10 + 1)] = (
            0.9 * e_min[i * ne_seg] + 0.1 * e_min[i * ne_seg + 1]
        )

    d_ = numpy.zeros(nfr10)
    d_ = numpy.insert(d_, 0, "inf")

    post_snr = numpy.zeros(nfr10)
    post_snr = numpy.insert(post_snr, 0, "inf")

    for i in range(2, nfr10 + 1):
        post_snr[i] = numpy.log10(e[i]) - numpy.log10(e_min[i])
        if numpy.less(post_snr[i], 0):
            post_snr[i] = 0

        d_[i] = numpy.sqrt(numpy.abs(e[i] - e[i - 1]) * post_snr[i])
    d_[1] = d_[2]

    tm1 = numpy.hstack((numpy.ones(d_expl) * d_[1], d_[1 : len(d_)]))
    d_exp = numpy.hstack((tm1, numpy.ones(d_expr) * d_[nfr10]))
    d_exp = numpy.insert(d_exp, 0, "inf")

    d_smth = numpy.zeros(nfr10, dtype="float64")
    d_smth = numpy.insert(d_smth, 0, "inf")

    d_smth_max = deepcopy(d_smth)

    for i in range(1, nfr10 + 1):
        d_smth[i] = sum(d_exp[range(i, i + d_expl + d_expr + 1)])

    for i in range(1, int(numpy.floor(nfr10 / ne_seg)) + 1):
        d_smth_max[range((i - 1) * ne_seg + 1, i * ne_seg + 1)] = numpy.amax(
            e[range((i - 1) * ne_seg + 1, i * ne_seg + 1)]
        )  #
        # numpy.amax(Dsmth[range((i-1)*NESEG+1, i*NESEG+1)])

    if numpy.not_equal(i * ne_seg, nfr10):
        d_smth_max[range(i * ne_seg + 1, nfr10 + 1)] = numpy.amax(
            e[range((i - 1) * ne_seg + 1, nfr10 + 1)]
        )  # numpy.amax(
        # Dsmth[range((i-1)*NESEG+1, nfr10+1)])

    snre_vad_ = numpy.zeros(nfr10)
    snre_vad_ = numpy.insert(snre_vad_, 0, "inf")

    for i in range(1, nfr10 + 1):
        if numpy.greater(d_smth[i], d_smth_max[i] * seg_threshold):
            snre_vad_[i] = 1

    # block based processing to remove noise part by using snre_vad1.
    sign_vad = 0
    noise_seg = numpy.zeros(int(numpy.floor(nfr10 / 1.6)))
    noise_seg = numpy.insert(noise_seg, 0, "inf")

    noise_samp = numpy.zeros((nfr10, 2))
    n_noise_samp = -1

    for i in range(1, nfr10 + 1):
        if (snre_vad_[i] == 1) and (sign_vad == 0):  # % start of a segment
            sign_vad = 1
            nstart = i
        elif ((snre_vad_[i] == 0) or (i == nfr10)) and (
            sign_vad == 1
        ):  # % end of a segment
            sign_vad = 0
            nstop = i - 1
            if numpy.equal(sum(pv01_[range(nstart, nstop + 1)]), 0):
                noise_seg[
                    range(
                        int(numpy.round(nstart / 1.6)),
                        int(numpy.floor(nstop / 1.6)) + 1,
                    )
                ] = 1
                n_noise_samp = n_noise_samp + 1
                noise_samp[n_noise_samp, :] = numpy.array(
                    [(nstart - 1) * fsh10 + 1, nstop * fsh10]
                )

    noise_samp = noise_samp[
        : n_noise_samp + 1,
    ]

    # syn  from [0] index
    noise_samp = noise_samp - 1
    noise_seg = noise_seg[1 : len(noise_seg)]

    return noise_samp, noise_seg, len(noise_samp)


def snre_vad(
    fdata,
    nfr10,
    flen,
    fsh10,
    energy_floor,
    pv01,
    pvblk,
    vad_threshold,
    *,
    d_expl=18,
    d_expr=18
):
    ## ---*******- important *******
    # here [0] index array element has  not used

    d_smth = numpy.zeros(nfr10, dtype="float64")
    d_smth = numpy.insert(d_smth, 0, "inf")

    fdata_ = deepcopy(fdata)
    pv01_ = deepcopy(pv01)
    pvblk_ = deepcopy(pvblk)

    fdata_ = numpy.insert(fdata_, 0, "inf")
    pv01_ = numpy.insert(pv01_, 0, "inf")
    pvblk_ = numpy.insert(pvblk_, 0, "inf")

    # energy estimation
    e = numpy.zeros(nfr10, dtype="float64")
    e = numpy.insert(e, 0, "inf")

    for i in range(1, nfr10 + 1):
        for j in range(1, flen + 1):
            e[i] = e[i] + numpy.square(fdata_[(i - 1) * fsh10 + j])

        if numpy.less_equal(e[i], energy_floor):
            e[i] = energy_floor

    # seg_snr = numpy.zeros(nfr10)
    # seg_snr = numpy.insert(seg_snr, 0, "inf")
    # segsnrsmth = 1
    # sign_segsnr = 0
    d_ = numpy.zeros(nfr10)
    d_ = numpy.insert(d_, 0, "inf")
    post_snr = numpy.zeros(nfr10, dtype="float64")
    post_snr = numpy.insert(post_snr, 0, "inf")
    snre_vad = numpy.zeros(nfr10)
    snre_vad = numpy.insert(snre_vad, 0, "inf")
    sign_pv = 0

    for i in range(1, nfr10 + 1):

        if (pvblk_[i] == 1) and (sign_pv == 0):
            n_start = i
            sign_pv = 1

        elif ((pvblk_[i] == 0) or (i == nfr10)) and (sign_pv == 1):

            n_stop = i - 1
            if i == nfr10:
                n_stop = i
            sign_pv = 0
            data_i = fdata_[
                range(
                    (n_start - 1) * fsh10 + 1, (n_stop - 1) * fsh10 + flen - fsh10 + 1
                )
            ]
            data_i = numpy.insert(data_i, 0, "inf")

            for j in range(
                n_start, n_stop - 1 + 1
            ):  # previously it was for j=nstart:nstop-1
                for h in range(1, flen + 1):
                    e[j] = e[j] + numpy.square(data_i[(j - n_start) * fsh10 + h])
                if numpy.less_equal(e[j], energy_floor):
                    e[j] = energy_floor

            e[n_stop] = e[n_stop - 1]

            e_y = numpy.sort(e[range(n_start, n_stop + 1)])
            e_y = numpy.insert(e_y, 0, "inf")  # as [0] is discarding

            emin = e_y[int(numpy.floor((n_stop - n_start + 1) * 0.1))]

            for j in range(n_start + 1, n_stop + 1):

                post_snr[j] = math.log10(e[j]) - math.log10(emin)

                if numpy.less(post_snr[j], 0):
                    post_snr[j] = 0

                d_[j] = math.sqrt(numpy.abs(e[j] - e[j - 1]) * post_snr[j])

            d_[n_start] = d_[n_start + 1]

            tm1 = numpy.hstack(
                (numpy.ones(d_expl) * d_[n_start], d_[range(n_start, n_stop + 1)])
            )
            d_exp = numpy.hstack((tm1, numpy.ones(d_expr) * d_[n_stop]))

            d_exp = numpy.insert(d_exp, 0, "inf")

            for j in range(0, n_stop - n_start + 1):
                d_smth[n_start + j] = sum(d_exp[range(j + 1, j + d_expl + d_expr + 1)])

            d_smth_threshold = sum(
                d_smth[range(n_start, n_stop + 1)] * pv01_[range(n_start, n_stop + 1)]
            ) / sum(pv01_[range(n_start, n_stop + 1)])

            for j in range(n_start, n_stop + 1):
                if numpy.greater(d_smth[j], d_smth_threshold * vad_threshold):
                    snre_vad[j] = 1

                    #
    pv_vad = deepcopy(snre_vad)

    n_expl = 33
    n_expr = (
        47  # % 29 and 39, estimated statistically, 95% ; 33, 47 %98 for voicebox pitch
    )
    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            n_start = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            n_stop = i - 1
            if i == nfr10:
                n_stop = i
            sign_vad = 0
            for j in range(n_start, n_stop + 1):
                if pv01_[j] == 1:
                    break

            pv_vad[range(n_start, numpy.max([j - n_expl - 1, 1]) + 1)] = 0

            for j in range(0, n_stop - n_start + 1):
                if pv01_[n_stop - j] == 1:
                    break

            pv_vad[range(n_stop - j + 1 + n_expr, n_stop + 1)] = 0

    n_expl = 5
    n_expr = 12  # ; % 9 and 13, estimated statistically 5%; 5, 12 %2 for voicebox pitch
    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            n_start = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            n_stop = i - 1
            if i == nfr10:
                n_stop = i
            sign_vad = 0

            if numpy.greater(sum(pv01_[range(n_start, n_stop + 1)]), 4):
                for j in range(n_start, n_stop + 1):
                    if pv01_[j] == 1:
                        break

                pv_vad[range(numpy.maximum(j - n_expl, 1), j - 1 + 1)] = 1
                for j in range(0, n_stop - n_start + 1):
                    if pv01_[n_stop - j] == 1:
                        break
                pv_vad[range(n_stop - j + 1, min(n_stop - j + n_expr, nfr10) + 1)] = 1

            e_segment = sum(e[range(n_start, n_stop + 1)]) / (n_stop - n_start + 1)
            if numpy.less(e_segment, 0.001):
                pv_vad[range(n_start, n_stop + 1)] = 0

            if numpy.less_equal(sum(pv01_[range(n_start, n_stop + 1)]), 2):
                pv_vad[range(n_start, n_stop + 1)] = 0

    sign_vad = 0
    e_sum = 0
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            n_start = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            n_stop = i - 1
            if i == nfr10:
                n_stop = i
            sign_vad = 0
            e_sum = e_sum + sum(e[range(n_start, n_stop + 1)])

    #
    eps = numpy.finfo(float).eps

    eave = e_sum / (sum(pv_vad[1 : len(pv_vad)]) + eps)  # except [0] index 'inf'

    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            n_start = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            n_stop = i - 1
            if i == nfr10:
                n_stop = i
            sign_vad = 0

            # if numpy.less(sum(e[range(nstart,nstop+1)])/(nstop-nstart+1), eave*0.05):
            # pv_vad[range(nstart,nstop+1)] = 0

    #
    sign_vad = 0
    vad_seg = numpy.zeros((nfr10, 2), dtype="int64")
    n_vad_seg = -1  # for indexing array
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            n_start = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            n_stop = i - 1
            sign_vad = 0
            n_vad_seg = n_vad_seg + 1
            vad_seg[n_vad_seg, :] = numpy.array([n_start, n_stop])

    vad_seg = vad_seg[
        : n_vad_seg + 1,
    ]

    # syn  from [0] index
    vad_seg = vad_seg - 1

    # make one dimension array of (0/1)
    x_yy = numpy.zeros(nfr10, dtype="int64")
    for i in range(len(vad_seg)):
        k = range(vad_seg[i, 0], vad_seg[i, 1] + 1)
        x_yy[k] = 1

    vad_seg = x_yy
    return vad_seg


def pitchblockdetect(pv01, pitch, nfr10, opts):
    pv01_ = deepcopy(pv01)

    if nfr10 == len(pv01_) + 1:
        numpy.append(pv01_, pv01_[nfr10 - 1])
    if opts == 0:
        sign_pv = 0
        for i in range(0, nfr10):

            if (pv01_[i] == 1) and (sign_pv == 0):

                n_start, sign_pv = i, 1

            elif ((pv01_[i] == 0) or (i == nfr10 - 1)) and (sign_pv == 1):

                n_stop = i
                if i == nfr10 - 1:
                    n_stop = i + 1
                sign_pv = 0
                pitch_seg = numpy.zeros(n_stop - n_start)
                for j in range(n_start, n_stop):

                    pitch_seg[j - n_start] = pitch[j]

                if (
                    sum(numpy.abs(numpy.round(pitch_seg - numpy.average(pitch_seg))))
                    == 0
                ) and (n_stop - n_start + 1 >= 10):
                    pv01_[range(n_start, n_stop)] = 0
                    #
    sign_pv = 0
    pv_blk = deepcopy(pv01_)

    # print i
    for i in range(0, nfr10):

        if (pv01_[i] == 1) and (sign_pv == 0):
            # print("i=%s " %(i))
            n_start, sign_pv = i, 1
            pv_blk[range(max([n_start - 60, 0]), n_start + 1)] = 1
            # print("fm P2: i=%s %s % " %(i,max([nstart-60,0]), nstart+1))

        elif ((pv01_[i] == 0) or (i == nfr10 - 1)) and (sign_pv == 1):

            n_stop, sign_pv = i, 0

            pv_blk[range(n_stop, numpy.amin([n_stop + 60, nfr10 - 1]) + 1)] = 1
            # print("fm P2: i=%s %s %s " %(i,n_stop, numpy.amin([n_stop+60,nfr10-1])+1 ))

    return pv_blk
