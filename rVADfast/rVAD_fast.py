#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import sys
from copy import deepcopy

import numpy
from scipy.signal import lfilter

from external.rVADfast.rVADfast import speechproc

__author__ = "Achintya Kumar Sarkar and Zheng-Hua Tan"

__doc__ = r"""

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity
#  detection method," Computer Speech and Language, 2019.
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and
#  voice activity detection."
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
# 2017-12-02, Achintya Kumar Sarkar and Zheng-Hua Tan

# Editions by Christian Heider Nielsen

# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel

"""

__all__ = ["get_rvad"]


def get_rvad(sample_rate,
             data,
             *,
             win_len=0.025,
             ovr_len=0.01,
             pre_coef=0.97,  # Unused, preemphasis
             nfilter=20,  # Unused, num fcc fitlers
             n_ftt=512,
             ft_threshold=0.5,
             vad_threshold=0.4,
             opts=1):
  ft, flen, fsh10, nfr10 = speechproc.sflux(data, sample_rate, win_len, ovr_len, n_ftt)

  # --spectral flatness --
  pv01 = numpy.zeros(nfr10)
  pv01[numpy.less_equal(ft, ft_threshold)] = 1
  pitch = deepcopy(ft)

  pv_block = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

  # --filtering--
  energy_floor = numpy.exp(-50)
  b = numpy.array([0.9770, -0.9770])
  a = numpy.array([1.0000, -0.9540])
  f_data = lfilter(b, a, data, axis=0)

  # --pass 1--
  noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
      f_data, nfr10, flen, fsh10, energy_floor, pv01, pv_block
      )
  # sets noisy segments to zero
  for j in range(n_noise_samp):
    f_data[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

  voice_activity_mask = speechproc.snre_vad(
      f_data, nfr10, flen, fsh10, energy_floor, pv01, pv_block, vad_threshold
      )

  voice_activity_mask_out = numpy.repeat(voice_activity_mask, 160)
  last_two = numpy.array([voice_activity_mask_out[-1]] * 2 * 160)
  voice_activity_mask_out = numpy.hstack([voice_activity_mask_out, last_two])
  # 160 * mask
  # 240 append

  return voice_activity_mask_out


if __name__ == "__main__":

  def main():
    inWaveFile = str(sys.argv[1])
    outputVadLabel = str(sys.argv[2])

    sample_rate, data = speechproc.speech_wave(inWaveFile)

    vad_seg = get_rvad(sample_rate, data)

    numpy.savetxt(outputVadLabel, vad_seg.astype(int), fmt="%i")
    print(f"{inWaveFile} --> {outputVadLabel} ")


  main()
