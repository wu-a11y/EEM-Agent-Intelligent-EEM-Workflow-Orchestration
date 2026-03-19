# -*- coding = utf-8 -*-
# @TIME : 2023/01/10 12:29
# @File : EEMs_toolkit.py
# @Software : PyCharm

import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from tensorly.decomposition import non_negative_parafac_hals
from joblib import Parallel, delayed
import os
import pandas as pd
import datetime
import tlviz
from concurrent.futures import ThreadPoolExecutor

#  read files
def read_eems(sample_log, eem_path):
    """
    Read the EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param sample_log: A DataFrame with the column 'eem' in it.
    :param eem_path: The directory where the EEM files saved.
    :return: A tuple including EEMs, Ex and Em, and a list which representing the file_list of the EEMs.
    """
    file_list = list(sample_log['eem'])
    x = []
    for j, i in enumerate(file_list):
        c = pd.read_excel(eem_path + r'\\' + i, header=None)
        x.append(c.values[1:, 1:])
        print(f'EEM {i} has been read. {j + 1} of {len(file_list)}')
    ex = c.values[0, 1:]
    em = c.values[1:, 0]
    fl = [i.split('.')[0] for i in file_list]
    x = np.array(x)
    return (x, ex, em), fl


def read_abs(sample_log, abs_path):
    """
    Read the Abs samples.

    :param sample_log:  A DataFrame with the column 'abs' in it.
    :param abs_path: The directory where the Abs files saved.
    :return: A matrix of absorbance, and the wavelength of the absorbance.
    """
    file_list = set(sample_log['abs'])
    d = dict()
    for j, i in enumerate(file_list):
        c = pd.read_excel(abs_path + r'\\' + i, header=None)
        print(f'Abs {i} has been read. {j + 1} of {len(file_list)}')
        t = c.values
        if t.shape[0] == 2:
            t = t.T
        d[i] = t[:, 1]
    Abs = np.array([d[i] for i in sample_log['abs']])
    Abs_wave = t[:, 0]
    return Abs, Abs_wave


def read_blank(sample_log, blank_path):
    """
    Read the blank EEM samples with different Ex in columns and different Em in rows.
    The header is Ex and the index is Em.

    :param sample_log: A DataFrame with the column 'blank' in it.
    :param blank_path: The directory where the blank EEMs files saved.
    :return: blank_eem
    """
    file_list = set(sample_log['blank'])
    d = dict()
    for j, i in enumerate(file_list):
        c = pd.read_excel(blank_path + r'\\' + i, header=None)
        print(f'Blank {i} has been read. {j + 1} of {len(file_list)}')
        d[i] = c.values[1:, 1:]
    blank_eem = np.array([d[i] for i in sample_log['blank']])
    return blank_eem


def read_sample_log(path):
    """
    Read the sample log. The example of sample log can be downloaded from GitHub.

    :param path: The directory where the sample log saved.
    :return: A DataFrame of sample log.
    """
    log = pd.read_excel(path, header=0, index_col=None)
    print('Sample log has been read.')
    return log


def slope_fit(Abs, Abs_wave, rsq=0.95, long_range=None):
    """
    Fit spectral slopes in two different areas of the CDOM absorbance spectrum.
    It will create a new excel file named 'slope_fit_result.xlsx' in the current directory.
    Exponential model: y = a350*exp(S/1000*(350-lambda))+k
        Reference DOI: 10.4319/lo.2008.53.3.0955, 10.4319/lo.2001.46.8.2087

    :param Abs: A matrix with different samples in rows and different wavelengths in columns.
    :param Abs_wave: The wavelengths of Abs.
    :param rsq: The lower limit of adjusted R-squared. If below rsq, return nan. Default is 0.95.
    :param long_range: Wavelength range for the long wavelength slope. Default is [300, 600].
    :return: A DataFrame of results.
    """

    def f1(x, *args):
        b1, b2, b3 = args
        return b1 * np.exp(b2 / 1000 * (350 - x)) + b3

    if long_range is None:
        long_range = [300, 600]
    index0 = [np.where(long_range[0] == Abs_wave)[0][0], np.where(long_range[1] == Abs_wave)[0][0]]
    ws0 = Abs_wave[index0[0]:index0[1] + 1]
    abc0 = Abs[:, index0[0]:index0[1] + 1]
    r1 = [curve_fit(f1, ws0, abc0[i, :], p0=[abc0[i, :].mean(), 18, 0], maxfev=2500)[0] for i in range(Abs.shape[0])]
    fit0 = [(f1(ws0, *r1[i]) ** 2).sum() / (abc0[i, :] ** 2).sum() for i in range(Abs.shape[0])]
    judge0 = [i >= rsq for i in fit0]
    r = [[275, 295], [350, 400]]
    index = [[np.where(r[i][0] == Abs_wave)[0][0],
              np.where(r[i][1] == Abs_wave)[0][0]]
             for i in [0, 1]]
    ws = [Abs_wave[i[0]:i[1] + 1] for i in index]
    abc = [np.log(Abs[:, i[0]:i[1] + 1]) for i in index]

    def r_squared(aa, y_true, ww):
        y_p = np.polyval(aa, ww)
        r2 = 1 - ((y_p - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
        r2_adjust = 1 - (1 - r2) * (len(ww) - 1) / (len(ww) - len(aa))
        return r2_adjust

    p1 = [np.polyfit(ws[0], abc[0][i, :], 1) for i in range(Abs.shape[0])]
    fit1 = [r_squared(p1[i], abc[0][i, :], ws[0]) for i in range(Abs.shape[0])]
    judge1 = [i >= rsq for i in fit1]
    p2 = [np.polyfit(ws[1], abc[1][i, :], 1) for i in range(Abs.shape[0])]
    fit2 = [r_squared(p2[i], abc[1][i, :], ws[1]) for i in range(Abs.shape[0])]
    judge2 = [i >= rsq for i in fit2]
    exp_slope_microm = []
    s_275_295 = []
    s_350_400 = []
    sr = []
    for i in range(Abs.shape[0]):
        if judge0[i]:
            exp_slope_microm.append(r1[i][1])
        else:
            exp_slope_microm.append(np.nan)
        if judge1[i]:
            s_275_295.append(p1[i][0] * -1e3)
        else:
            s_275_295.append(np.nan)
        if judge2[i]:
            s_350_400.append(p2[i][0] * -1e3)
        else:
            s_350_400.append(np.nan)
        if judge1[i] and judge2[i]:
            sr.append(p1[i][0] / p2[i][0])
        else:
            sr.append(np.nan)
    d = pd.DataFrame()
    d['exp_slope_microm'] = exp_slope_microm
    d['s_275_295'] = s_275_295
    d['s_350_400'] = s_350_400
    d['SR'] = sr
    d.to_excel('slope_fit_result.xlsx', index=False)
    return d


class EEMs_Dataset:
    """
    The object of EEMs dataset.
    """

    #  Initialization
    def __init__(self, x, ex, em, **kwargs):
        self.x = x
        self.nSample = x.shape[0]
        self.ex = ex
        self.em = em
        self.nem = em.shape[0]
        self.nex = ex.shape[0]
        self.i = np.arange(self.nSample)
        self.file_list = kwargs.get('file_list', self.i)
        for i in range(self.nex):
            for j in range(self.nem):
                if self.ex[i] > self.em[j]:
                    self.x[:, j, i] = 0
        # self.backup = x.copy()
        self.f = dict()
        self.factors = dict()
        # self.weights = dict()
        self.model = dict()
        self.core_consistency = dict()
        self.explanation_rate = dict()
        self.sse = dict()
        self.leverage = dict()
        self.split_result = dict()
        self.model_selection = dict()
        self.iu = 'AU'

    #  Preprocess Method
    def cut_ray_scatter(self, first, second):
        """
        Remove Rayleigh scatter in place, use nans instead.

        :param first: A list [below, above] for 1st Rayleigh scatter.
        :param second: A list [below, above] for 2nd Rayleigh scatter.
        """
        for i in range(self.nex):
            for j in range(self.nem):
                if self.ex[i] - first[0] <= self.em[j] <= self.ex[i] + first[1] and first:
                    self.x[:, j, i] = np.nan
                if 2 * self.ex[i] - second[0] <= self.em[j] <= 2 * self.ex[i] + second[1] and second != 0:
                    self.x[:, j, i] = np.nan

        #
        self._rayleigh_width_1st = first
        self._rayleigh_width_2nd = second
        print('Cut Rayleigh Scatter done.')

    def cut_ram_scatter(self, first, second, freq=3382):
        """
        Remove Raman scatter in place, use nans instead.

        :param first: A list [below, above] for 1st Raman scatter.
        :param second: A list [below, above] for 2nd Raman scatter.
        :param freq: Raman shift (cm^-1), default 3382.
        """
        for i in range(self.nex):
            for j in range(self.nem):
                ram = 1 / (1 / self.ex[i] - freq / 10 ** 7)
                if ram - first[0] <= self.em[j] <= ram + first[1] and first:
                    self.x[:, j, i] = np.nan
                if 2 * ram - second[0] <= self.em[j] <= 2 * ram + second[1] and second:
                    self.x[:, j, i] = np.nan

        #
        self._raman_width_1st = first
        self._raman_width_2nd = second
        self._raman_shift = freq
        print('Cut Raman Scatter done.')

    def miss_value_interpolation(self):
        """
        Use Delaunay triangulation method to interpolate in place,
        then force-fill out-of-bound nan with mean value.
        """
        x, y = np.meshgrid(self.ex, self.em)
        for j, i in enumerate(self.x):
            nan_ed = np.where(~np.isnan(i.astype('float')))
            filled = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (x, y), method='cubic')

            #  NaN
            if np.isnan(filled).any():
                mean_val = np.nanmean(filled)
                filled[np.isnan(filled)] = mean_val

            self.x[j] = filled

        self.x[self.x < 0] = 0  # 
        print('Interpolation done with post-processing for NaN.')

    def smooth_eem(self, sigma=0.5, filter_size=None):
        """
        Use Gaussian kernel to smooth EEMs in place. Suggesting after interpolation.

        :param sigma: Optional, standard deviation of the Gaussian distribution, specified as a positive number.
            Default is 0.5.
        :param filter_size: optional, size of the Gaussian filter, specified as a positive,
            odd integer. The default filter size is 2*ceil(2*sigma)+1.
        """
        if filter_size is None:
            filter_size = 2 * int(np.ceil(2 * sigma)) + 1
        kernel = np.zeros((filter_size, filter_size))
        center = filter_size // 2
        for i in range(filter_size):
            for j in range(filter_size):
                x, y = center - i, center - j
                kernel[i, j] = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
        kernel = np.exp(kernel)
        kernel /= np.sum(kernel)
        t = np.zeros((self.nem, self.nex))
        for j, i in enumerate(self.x):
            xx = np.pad(i, center, mode='reflect')
            for k in range(self.nem):
                for q in range(self.nex):
                    t[k, q] = np.sum(xx[k:k + filter_size, q:q + filter_size] * kernel)
            self.x[j] = t
        print('EEM smooth done.')

    def sub_dataset(self, sample_orders, ex_orders=None, em_orders=None):
        """
        Removing samples and/or wavelengths in place.

        :param sample_orders: A list of positive integers, 1 is the first sample.
        :param ex_orders: e.g. self.ex == 240
        :param em_orders: e.g. self.em > 600
        """
        if sample_orders:
            orders = [a - 1 for a in sample_orders]
            self.i = np.delete(self.i, orders)
            self.x = self.x[self.i]
            self.nSample -= len(sample_orders)
            new_file_list = []
            for i in self.i:
                new_file_list.append(self.file_list[i])
            self.file_list = new_file_list
            # self.i = np.arange(self.nSample)
        if ex_orders is not None:
            self.ex = np.delete(self.ex, ex_orders)
            self.nex = self.ex.shape[0]
            self.x = self.x[:, :, ~ex_orders]
        if em_orders is not None:
            self.em = np.delete(self.em, em_orders)
            self.nem = self.em.shape[0]
            self.x = self.x[:, ~em_orders, :]

    def minus_the_blank(self, blank):
        """
        Minus blank sample in place.

        :param blank: The same shape as self.x or just one sample.
        """
        if len(blank.shape) == 2:
            blank = np.array([blank for _ in range(self.nSample)])
        self.x -= blank
        self.x[self.x < 0] = 0
        print('Blank minus done.')

    def dilute(self, dilution_factors):
        """
        Dilution correction in place.

        :param dilution_factors: A list of dilution factors for every sample.
        """
        # for i in range(self.nSample):
        #     self.x[i] *= dilution_factors[i]
        self.x = np.einsum('ijk,i->ijk', self.x, dilution_factors)
        print('Dilution correction done.')

    def inner_effect_correct(self, Abs_wave, Abs):
        """
        Inner filter effect correction in place.
            Reference DOI: 10.1021/es0155276, 10.4319/lom.2013.11.616

        :param Abs_wave: The wavelengths of Abs.
        :param Abs: A matrix with each row corresponding to EEM and different wavelengths in columns.
        """
        ex_index = []
        for i in self.ex:
            if i not in Abs_wave:
                print('Some Ex wavelengths are not included')
                return
            else:
                ex_index.append(i == Abs_wave)
        em_index = []
        for i in self.em:
            if i not in Abs_wave:
                print('Some Ex wavelengths are not included')
                return
            else:
                em_index.append(i == Abs_wave)
        correct = np.zeros(self.x.shape)
        for k in range(self.nSample):
            for i in range(self.nem):
                for j in range(self.nex):
                    correct[k, i, j] = 0.5 * (Abs[k, em_index[i]] + Abs[k, ex_index[j]])
        self.x *= np.power(10, correct)
        print('Inner effect correction done.')

    def raman_areal(self, blank, ex=350, em_range=None, freq=3382):
        """
        Turning arbitrary units(AU) into Raman units(RU).
        Reference DOI: 10.1366/000370209788964548

        :param blank: The same shape as self.x or just one sample.
        :param ex: Optional, the excitation position of Raman peak. Default is 350.
        :param em_range: Optional, a list of two elements, range used to calculate the area,
            if not given, it will be calculated by ex.
        :param freq: Optional, a constant named Raman shift which is around 3400-3600 cm^-1 for water. Default is 3382.
        """
        if em_range is None:
            em_range = np.round([1e7 * (1e7 / ex - freq + 1800) ** -1,
                                 1e7 * (1e7 / ex - freq - 1800) ** -1])
        em = np.arange(em_range[0], em_range[1] + 1)
        ex_index = np.argmin(np.abs(self.ex - ex))
        em_index = [np.where((self.em - em_range[0]) <= 0)[0][-1],
                    np.where((self.em - em_range[1]) >= 0)[0][0]]
        if len(blank.shape) == 2:
            blank = np.array([blank for _ in range(self.nSample)])
        for i in range(self.nSample):
            areal = blank[i, em_index[0]:em_index[1] + 1, ex_index]
            f = interp1d(self.em[em_index[0]:em_index[1] + 1], areal)
            f_new = f(em)
            self.x[i] /= np.sum(f_new)
        self.iu = 'RU'
        print('Raman Units done.')

    #  Fluorescence Indices
    def __pick_fluorescence_indices(self):
        """
        Calculate fluorescence indices.
            Fluorescence index: Ratio of em wavelengths at 470 nm and 520 nm, obtained at ex370
                Reference DOI: 10.1016/j.gca.2006.06.1554
            Freshness index: em 380 divided by the em maximum between 420 and 435, obtained at ex310
                Reference DOI: 10.1016/S0146-6380(00)00124-8
            Humification index: area under the em spectra 435:480 divided by the peak area 300:345 + 435:480, at ex254
                Reference DOI: 10.1016/S0146-6380(00)00124-8
            Biological index: Ex 310, Em 380 divided by 430
                Reference DOI: 10.1016/j.orggeochem.2009.03.002
        """
        ex = np.arange(np.ceil(self.ex.min()), np.floor(self.ex.max()) + 1)
        em = np.arange(np.ceil(self.em.min()), np.floor(self.em.max()) + 1)
        ex_index = {i: j for j, i in enumerate(ex)}
        em_index = {i: j for j, i in enumerate(em)}
        if np.any(ex == 254):
            hix_bool = True
        else:
            hix_bool = False
            print('Humification index cannot be calculated due to dataset limitations.')
        x, y = np.meshgrid(ex, em)
        fl, fre, hix, bix = [], [], [], []
        for i in self.x:
            index = np.where(~np.isnan(i.astype('float')))
            fitted = griddata((self.ex[index[1]], self.em[index[0]]), i[index],
                              (x, y), method='linear')
            scan370 = savgol_filter(fitted[:, ex_index[370]], 21, 2)
            fl.append(scan370[em_index[470]] / scan370[em_index[520]])
            scan310 = savgol_filter(fitted[:, ex_index[310]], 21, 2)
            fre.append(scan310[em_index[380]] / scan310[em_index[420]:em_index[436]].max())
            if hix_bool:
                scan254 = savgol_filter(fitted[:, ex_index[254]], 21, 2)
                v1 = scan254[em_index[435]:em_index[481]].sum()
                v2 = v1 + scan254[em_index[300]:em_index[346]].sum()
                hix.append(v1 / v2)
            bix.append(scan310[em_index[380]] / scan310[em_index[430]])
        return fl, fre, hix, bix

    #  FRI
    def __fluorescence_regional_integration(self):
        """
        Calculation of FRI, including percent fluorescence response, volume of different regions
        and multiplication factor.
            Reference DOI: 10.1021/es034354c
        """
        if self.ex.min() >= 250:
            print('Regions 1, 2, and 3 do not exist!')
            return
        if self.ex.max() <= 250:
            print('Regions 4 and 5 do not exist!')
            return
        if self.em.min() >= 380:
            print('Regions 1, 2, and 4 do not exist!')
            return
        elif self.em.min() >= 330:
            print('Regions 1 does not exist!')
            return
        if self.em.max() <= 330:
            print('Regions 2, 3, and 5 do not exist!')
            return
        elif self.em.max() <= 380:
            print('Regions 3 and 5 do not exist!')
            return
        x = self.x.copy()
        if np.any(np.isnan(x.astype('float'))):
            xx, yy = np.meshgrid(self.ex, self.em)
            for j, i in enumerate(x):
                nan_ed = np.where(~np.isnan(i.astype('float')))
                x[j] = griddata((self.ex[nan_ed[1]], self.em[nan_ed[0]]), i[nan_ed], (xx, yy), method='cubic')
            x[x < 0] = 0
        index = [[[np.where(self.ex == 250)[0][0], np.argmin(self.ex)],
                  [np.where(self.em == 330)[0][0], np.argmin(self.em)]],
                 [[np.where(self.ex == 250)[0][0], np.argmin(self.ex)],
                  [np.where(self.em == 380)[0][0], np.where(self.em == 330)[0][0]]],
                 [[np.where(self.ex == 250)[0][0], np.argmin(self.ex)],
                  [np.argmax(self.em), np.where(self.em == 380)[0][0]]],
                 [[np.argmax(self.ex), np.where(self.ex == 250)[0][0]],
                  [np.where(self.em == 380)[0][0], np.argmin(self.em)]],
                 [[np.argmax(self.ex), np.where(self.ex == 250)[0][0]],
                  [np.argmax(self.em), np.where(self.em == 380)[0][0]]]]
        s = np.array([(250 - self.ex.min()) * (330 - self.em.min()),
                      (250 - self.ex.min()) * (380 - 330),
                      (250 - self.ex.min()) * (self.em.max() - 380),
                      (self.ex.max() - 250) * (380 - self.em.min()),
                      (self.ex.max() - 250) * (self.em.max() - 380)])
        s = s.sum() / s
        p, phi = np.zeros((self.nSample, 5)), np.zeros((self.nSample, 5))
        for i in range(self.nSample):
            t = x[i]
            for j in range(5):
                ts = t[index[j][1][1]:index[j][1][0] + 1, index[j][0][1]:index[j][0][0] + 1]
                phi[i, j] = np.trapz(np.trapz(ts, self.ex[index[j][0][1]:index[j][0][0] + 1], axis=1),
                                     self.em[index[j][1][1]:index[j][1][0] + 1])
                p[i, j] = phi[i, j] * s[j]
        p = (p.T / p.sum(axis=1).T).T
        s = s.reshape((1, 5))
        return p, phi, s

    #  Parallel Factors Analysis
    def __evaluate_candidate_model(self, candidate_model):
        """
        Evaluate one CP candidate model with robust handling of missing values.
        """
        model_tensor = tl.cp_to_tensor(candidate_model)
        mask = np.isfinite(self.x)
        if not np.any(mask):
            raise ValueError('No finite observations found in dataset.')

        residual = self.x[mask] - model_tensor[mask]
        sse = float(np.sum(residual ** 2))
        ssx = float(np.sum(self.x[mask] ** 2))
        explanation = np.nan if ssx <= 0 else float(1.0 - sse / ssx)

        # Align with N-way-style handling: fill missing values before CC.
        x_for_cc = self.x.copy()
        if not np.all(mask):
            x_for_cc[~mask] = model_tensor[~mask]
        cc = float(tlviz.model_evaluation.core_consistency(candidate_model, x_for_cc, normalised=True))

        return {
            'model': candidate_model,
            'sse': sse,
            'ssx': ssx,
            'explanation': explanation,
            'core_consistency': cc
        }

    def __select_model_from_starts(self, r_all, strategy='lowest_error', cc_sse_relax=0.02):
        """
        Select one model from multi-start results.

        strategy:
            - 'lowest_error': choose the minimum SSE model (original behaviour)
            - 'cc_aware': choose the highest CC model among models with SSE close to best SSE
        cc_sse_relax:
            relative SSE tolerance for 'cc_aware'. 0.02 means within +2% of best SSE.
        """
        evaluated = []
        for idx, candidate in enumerate(r_all):
            try:
                m = self.__evaluate_candidate_model(candidate)
                m['start_idx'] = idx
                evaluated.append(m)
            except Exception:
                continue

        if not evaluated:
            raise RuntimeError('All random-start PARAFAC models failed during evaluation.')

        if strategy == 'lowest_error':
            selected = min(evaluated, key=lambda m: m['sse'])
        elif strategy in ('cc_aware', 'cc_first'):
            best_sse = min(m['sse'] for m in evaluated)
            cutoff = best_sse * (1.0 + cc_sse_relax)
            pool = [m for m in evaluated if m['sse'] <= cutoff]
            selected = max(pool, key=lambda m: (m['core_consistency'], m['explanation'], -m['sse']))
        else:
            raise ValueError(f"Unknown model selection strategy: {strategy}")

        return selected, evaluated

    def __non_parafac_cal(self, f: int, tol=1e-6, max_iter=2500, start=10,
                          select_strategy='lowest_error', cc_sse_relax=0.02):
        """
        HALSCPALS
        
        

        :param f: int 
        :param tol: float  
        tol1e-6
        :param max_iter: int 
        :param start: int 
        """

        def npc(x, ff):
            return non_negative_parafac_hals(tl.tensor(x.astype('float')), rank=ff, init='random',
                                             n_iter_max=max_iter, tol=tol, verbose=False)

        self.f[f] = f
        print(f'The {f}-component model starts to fit.')
        # r_all = [non_negative_parafac_hals(tl.tensor(self.x.astype('float')),
        #                                    rank=f, verbose=False, init='random',
        #                                    n_iter_max=max_iter, tol=tol)
        #          for _ in range(start)]
        n = start if start < os.cpu_count() else -1
        r_all = Parallel(n_jobs=n)(delayed(npc)(self.x, f) for _ in range(start))
        selected, evaluated = self.__select_model_from_starts(
            r_all, strategy=select_strategy, cc_sse_relax=cc_sse_relax
        )
        r = selected['model']
        # r  None 
        # if r is None:
        #     print(f"{f}-component PARAFAC ")
        #     return  # ←  r.factors
        # print(f'The {f}-component model has been fitted.')

        self.factors[f] = r.factors
        # self.weights[f] = r.weights
        self.__normalise_factors(f)
        self.__explanation_rate(f)
        self.leverage[f] = tlviz.outliers.compute_leverage(self.factors[f][0])
        self.sse[f] = tlviz.outliers.compute_slabwise_sse(self.model[f], self.x)
        # tlviz.visualisation.outlier_plot(r, self.x)
        # plt.show()
        self.core_consistency[f] = selected['core_consistency']
        self.model_selection[f] = {
            'strategy': select_strategy,
            'cc_sse_relax': cc_sse_relax,
            'n_starts_evaluated': len(evaluated),
            'selected_start': int(selected['start_idx']) + 1,
            'selected_sse': selected['sse'],
            'selected_explanation_rate': selected['explanation'],
            'selected_core_consistency': selected['core_consistency']
        }
        print(
            f"{f}-component selected start {self.model_selection[f]['selected_start']}/"
            f"{self.model_selection[f]['n_starts_evaluated']} "
            f"(strategy={select_strategy}, CC={selected['core_consistency']:.2f}, "
            f"ER={100 * selected['explanation']:.2f}%)"
        )
        # tlviz.visualisation.core_element_plot(r, self.x)
        # plt.show()

    def multi_non_parafac_cal(self, f=None, tol=1e-6, max_iter=2500, start=10,
                              select_strategy='lowest_error', cc_sse_relax=0.02):
        """
        Run 'start' times Non-negative CP decomposition via HALS. Uses Hierarchical ALS (Alternating Least Squares)
        which updates each factor column-wise (one column at a time while keeping all other
        columns fixed)

        :param f: int, number of components
        :param tol: float, optional, the algorithm stops when the variation in the reconstruction
            error is less than the tol. Default: 1e-6
        :param max_iter: int, maximum number of iteration
        :param start: int, number of models to fit
        :param select_strategy: str, 'lowest_error' (default) or 'cc_aware'
        :param cc_sse_relax: float, relative SSE tolerance used when select_strategy='cc_aware'
        """
        if f is None:
            f = [2, 5]
            print('Parameter f is not given, use [2, 5] instead.')
        elif isinstance(f, int):
            f = [f, f]
        elif len(f) == 1:
            f = [f[0], f[0]]
        for i in range(f[0], f[1] + 1):
            self.__non_parafac_cal(
                i, tol=tol, max_iter=max_iter, start=start,
                select_strategy=select_strategy, cc_sse_relax=cc_sse_relax
            )

    def __factors_to_model(self, f: int):
        """
        Create EEM model from factors.

        :param f: int, number of components
        """
        qx = self.__finger_model(f)
        self.model[f] = np.einsum('ij,jkl->ikl', self.factors[f][0], qx)

    def split_analysis(self, f, random_state=None, start=10):
        """
        Generate PARAFAC models nested in dataset splits. TCC above 0.95 can be validated.
            Reference DOI: 10.1027/1614-2241.2.2.57

        :param f: int, number of components
        :param random_state: {None(default), int, np.random.RandomState}
        :param start: int, number of models to fit
        """
        n = start if start < os.cpu_count() else -1

        def parafac_model_fit(x, fac):
            def npc(xx, ff):
                return non_negative_parafac_hals(xx, rank=ff, init='random', n_iter_max=2500)

            # r_all = [
            #     non_negative_parafac_hals(tl.tensor(x.astype('float')), rank=fac, init='random', n_iter_max=2500)
            #     for _ in range(5)
            # ]
            r_all = Parallel(n_jobs=n)(delayed(npc)(tl.tensor(x.astype('float')), fac) for _ in range(start))
            r = tlviz.multimodel_evaluation.get_model_with_lowest_error(r_all, x)
            r = tlviz.factor_tools.distribute_weights(r, weight_behaviour='one_mode', weight_mode=0)
            return r[1]

        if isinstance(f, int):
            f = [f, f]
        elif len(f) == 1:
            f = [f[0], f[0]]
        print('Split analysis starts.')
        index = np.arange(self.nSample)
        np.random.seed(random_state)
        np.random.shuffle(index)
        splits = [self.x[index[i * self.nSample // 4:(i + 1) * self.nSample // 4]] for i in range(4)]
        cross = [[[0, 1], [2, 3]], [[0, 2], [1, 3]]]

        for i in range(f[0], f[1] + 1):
            print(f'Split analysis for {i}-component:')
            split_models = []
            for j in cross:
                model = []
                for k in j:
                    comb = np.concatenate((splits[k[0]], splits[k[1]]), axis=0)
                    model.append(parafac_model_fit(comb, i))
                order = tlviz.factor_tools.get_factor_matrix_permutation(model[0][1], model[1][1])
                model[1] = [model[1][k][:, order] for k in range(3)]
                split_models.append(model)
            self.split_result[i] = split_models
            em_cs = [tlviz.factor_tools.cosine_similarity(split_models[i][0][1], split_models[i][1][1]) for i in [0, 1]]
            ex_cs = [tlviz.factor_tools.cosine_similarity(split_models[i][0][2], split_models[i][1][2]) for i in [0, 1]]
            compare_to_all = [
                tlviz.factor_tools.cosine_similarity(split_models[0][0][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][0][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][2], self.factors[i][2])
            ]
            print(f'The TCC of Ex loadings for {i}-component split analysis: {ex_cs}')
            print(f'The TCC of Em loadings for {i}-component split analysis: {em_cs}')
            if min(compare_to_all) <= 0.95:
                print('Not Validated!')
            else:
                print(f'The {i}-component Model Validated!')

    def __explanation_rate(self, f: int):
        """
        Calculate explanation rate.

        :param f: int, number of components
        """
        self.__factors_to_model(f)
        mask = np.isfinite(self.x)
        if not np.any(mask):
            self.explanation_rate[f] = np.nan
            return
        numerator = np.sum((self.x[mask] - self.model[f][mask]) ** 2)
        denominator = np.sum(self.x[mask] ** 2)
        if denominator == 0:
            self.explanation_rate[f] = np.nan
            return
        self.explanation_rate[f] = 1.0 - numerator / denominator

    def __finger_model(self, f: int):
        """
        Calculate components from factors.

        :param f: int, number of components
        :return: components matrix
        """
        qx = np.zeros((f, self.nem, self.nex))
        for i in range(f):
            # qx[i, :, :] = self.factors[f][1][:, i].reshape((self.nem, 1)) @ self.factors[f][2][:, i].reshape(
            #     (1, self.nex))
            qx[i, :, :] = np.einsum('i,j->ij', self.factors[f][1][:, i], self.factors[f][2][:, i])
        return qx

    def __normalise_factors(self, f: int):
        """
        Ensure that the Ex and Em factor matrices have unit norm.

        :param f: int, number of components
        """
        self.factors[f][0] = self.factors[f][0] * \
                             np.linalg.norm(self.factors[f][1], axis=0) * \
                             np.linalg.norm(self.factors[f][2], axis=0)
        self.factors[f][1] /= np.linalg.norm(self.factors[f][1], axis=0)
        self.factors[f][2] /= np.linalg.norm(self.factors[f][2], axis=0)

    #  Plot Functions
    def plot_eem_by1(self, f=None, sample_id=None, save_dir=None):
        """
        Plot EEMs with Emission on X-axis, Excitation on Y-axis, jet colormap, smooth gradient, high resolution, and show images.

        :param f: Optional, use model data if given, else raw data self.x.
        :param sample_id: Optional, list of sample indices to plot. Default all.
        :param save_dir: Optional, directory to save plots.
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        if sample_id is None:
            sample_id = range(self.nSample)
        else:
            sample_id = np.array(sample_id) - 1

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        contour_levels = 100  # 

        if f is None:
            for i in sample_id:
                fig = plt.figure(figsize=(6, 4.8))
                fig.canvas.manager.set_window_title('Plot Raw EEMs')
                cs = plt.contourf(self.em, self.ex, self.x[i].T, levels=contour_levels, cmap='jet')
                plt.xlabel('Emission/nm', fontsize=11)
                plt.ylabel('Excitation/nm', fontsize=11)
                plt.title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
                plt.colorbar(cs)
                plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

                # ===  +  ===
                if save_dir:
                    filename = f"{i + 1:03d}_{self.file_list[i].replace(' ', '_')}.png"
                    plt.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight')
                    print(f"Saved: {filename}")
                plt.show()

        else:
            for i in sample_id:
                fig = plt.figure(figsize=(6, 4.8))
                fig.canvas.manager.set_window_title(f'{f}-component model')
                cs = plt.contourf(self.em, self.ex, self.model[f][i].T, levels=contour_levels, cmap='jet')
                plt.xlabel('Emission/nm', fontsize=11)
                plt.ylabel('Excitation/nm', fontsize=11)
                plt.title(f'Model {f} for sample {i + 1}', fontsize=13)
                plt.colorbar(cs)
                plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)

                if save_dir:
                    filename = f"{i + 1:03d}_model{f}_{self.file_list[i].replace(' ', '_')}.png"
                    plt.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight')
                    print(f"Saved: {filename}")
                plt.show()

    def plot_3deem_by1(self, f=None, sample_id=None):
        """
        Plot 3d-EEMs

        :param f: Optional, if given, plot model from factors instead self.x. Default is None.
        :param sample_id: Optional, which samples to plot, if not given, plot all. Default is None.
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        if sample_id is None:
            sample_id = range(self.nSample)
        else:
            sample_id = np.array(sample_id) - 1
        x, y = np.meshgrid(self.ex, self.em)
        if f is None:
            for i in sample_id:
                fig = plt.figure(figsize=(6, 4.8))
                ax = fig.add_subplot(projection='3d')
                fig.canvas.manager.set_window_title('Plot Raw EEMs')
                # ax.contourf3D(self.ex, self.em, self.x[i], level)
                ax.plot_surface(x, y, self.x[i], cmap='viridis')
                ax.set_xlabel('Excitation/nm', fontsize=11)
                ax.set_ylabel('Emission/nm', fontsize=11)
                ax.set_title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
                plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
                plt.show()
        else:
            for i in sample_id:
                fig = plt.figure(figsize=(6, 4.8))
                ax = plt.axes(projection='3d')
                fig.canvas.manager.set_window_title(f'{f}-component model')
                ax.plot_surface(x, y, self.model[f][i], cmap='viridis')
                ax.set_xlabel('Excitation/nm', fontsize=11)
                ax.set_ylabel('Emission/nm', fontsize=11)
                ax.set_title(f'Model {f} for sample {i + 1}', fontsize=13)
                plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
                plt.show()

    def plot_fri(self):
        """
        Plot FRI
            Reference DOI: 10.1021/es034354c
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        for i in range(self.nSample):
            fig = plt.figure(figsize=(6, 4.8))
            fig.canvas.manager.set_window_title('Plot Fluorescence Region')
            plt.contourf(self.em, self.ex, self.x[i].T)
            plt.axvline(380, color='k', linestyle='--')
            plt.axhline(250, color='k', linestyle='--')
            plt.axvline(330, ymax=(250 - self.ex.min()) / (self.ex.max() - self.ex.min()), color='k', linestyle='--', )
            plt.text((self.em.min() + 330) / 2, (self.ex.min() + 250) / 2, 'Ⅰ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((380 + 330) / 2, (self.ex.min() + 250) / 2, 'Ⅱ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.max() + 380) / 2, (self.ex.min() + 250) / 2, 'Ⅲ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.min() + 380) / 2, (self.ex.max() + 250) / 2, 'Ⅳ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.text((self.em.max() + 380) / 2, (self.ex.max() + 250) / 2, 'Ⅴ',
                     ha='center', va='center', font='SimHei', color='w', fontsize=15)
            plt.ylabel('Excitation/nm', fontsize=11)
            plt.xlabel('Emission/nm', fontsize=11)
            plt.title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            plt.colorbar()
            plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
            plt.show()

    def plot_outlier_test(self, f: int, save_path=None):
        """
        Create a new figure to show outliers.

        :param f: int, number of components
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(figsize=(6, 4.8))
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        fig.canvas.manager.set_window_title(f'Plot the outlier test of {f}-component model')
        plt.scatter(self.leverage[f], self.sse[f])
        for i in range(len(self.sse[f])):
            plt.annotate(f'{self.i[i] + 1}: {self.file_list[i]}', (self.leverage[f][i], self.sse[f][i]))
        plt.xlabel('Leverage', fontsize=11)
        plt.ylabel('SSE', fontsize=11)
        plt.title(f'Outlier plot for {f} components', fontsize=13)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        #  figure
        try:
            plt.close(fig)
        except Exception:
            pass

    def plot_residual_error(self, f: int, sample_id=None):
        """
        Plot sample, model from factors and residual in one figure.

        :param f: int, number of component
        :param sample_id: Optional, which samples to plot, if not given, plot all. Default is None.
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        if sample_id is None:
            sample_id = range(self.nSample)
        else:
            sample_id = np.array(sample_id) - 1
        for i in sample_id:
            fig = plt.figure(figsize=(4.5 * 3, 4))
            fig.canvas.manager.set_window_title('Plot model and residual')
            plt.subplot(1, 3, 1)
            plt.contourf(self.ex, self.em, self.x[i])
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Sample {i + 1}: {self.file_list[i]}', fontsize=13)
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.contourf(self.ex, self.em, self.model[f][i])
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Model {f} for Sample {i + 1}', fontsize=13)
            plt.colorbar()
            plt.subplot(1, 3, 3)
            #  (nSample, nEm, nEx) 
            err_i = self.x[i] - self.model[f][i]
            plt.contourf(self.ex, self.em, err_i)
            plt.xlabel('Excitation/nm', fontsize=11)
            plt.ylabel('Emission/nm', fontsize=11)
            plt.title(f'Residual', fontsize=13)
            plt.colorbar()
            plt.subplots_adjust(left=0.07, bottom=0.13, right=0.94, top=0.9, wspace=0.24, hspace=0.2)
            plt.show()
            #  figure
            try:
                plt.close(fig)
            except Exception:
                pass

    def plot_core_consistency_and_explanation(self):
        """
        Plot core consistency and explanation rate, with f in self.f
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(figsize=(9, 4.3))
        fig.canvas.manager.set_window_title('Core consistency and explanation rate')
        fs = sorted(self.f.keys())
        ep = [100 * self.explanation_rate[i] for i in fs]
        cc = [self.core_consistency[i] for i in fs]
        plt.subplot(1, 2, 1)
        plt.title('Core consistency', fontsize=13)
        plt.plot(fs, cc, '-o')
        plt.xlabel('Number of component', fontsize=11)
        plt.ylabel('%', fontsize=11)
        plt.xticks(fs)
        print(f'Core consistency: {self.core_consistency}')
        plt.subplot(1, 2, 2)
        plt.title('Explanation rate', fontsize=13)
        plt.plot(fs, ep, '-o')
        plt.xlabel('Number of component', fontsize=11)
        # plt.ylabel('%', fontsize=11)
        plt.xticks(fs)
        print(f'Explanation rate: {self.explanation_rate}')
        plt.subplots_adjust(left=0.07, bottom=0.13, right=0.94, top=0.9, wspace=0.24, hspace=0.2)
        plt.show()

    def plot_factor_similarity(self):
        """
        Plot factor similarity with f in self.f, above 0.95 can be a good model.
        """
        ss = []
        fs = sorted(self.split_result.keys())
        for i in fs:
            split_models = self.split_result[i]
            compare_to_all = [
                tlviz.factor_tools.cosine_similarity(split_models[0][0][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][1], self.factors[i][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][0][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][2], self.factors[i][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][2], self.factors[i][2])
            ]
            ss.append(min(compare_to_all))
        print('Factor similarity:', {i: j for i, j in zip(fs, ss)})
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(figsize=(6, 4.8))
        fig.canvas.manager.set_window_title('Factor similarity for different components')
        plt.plot(fs, ss, '-o')
        plt.axhline(0.95, c='k', ls='--', label='Good similarity')
        plt.xticks(fs)
        plt.title('The smallest factor similarity', fontsize=13)
        plt.xlabel('Number of component', fontsize=11)
        plt.ylabel('Split half stability', fontsize=11)
        plt.legend(framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
        plt.show()

    def plot_split_result(self, f: int, save_path=None, save_path_1=None):
        """
        Plot split analysis result

        :param f: int, number of components
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.manager.set_window_title(f'Split analysis result for {f}-component')
        plt.subplot(2, 2, 1)
        for i in range(f):
            line = plt.plot(self.em, self.split_result[f][0][0][1][:, i], linestyle='-', label=f'AB comp {i + 1}')
            plt.plot(self.em, self.split_result[f][0][1][1][:, i], c=line[0].get_color(), linestyle='--',
                     label=f'CD comp {i + 1}')
        lines, labels = fig.axes[0].get_legend_handles_labels()
        lg1 = fig.legend(lines, labels, framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False,
                         bbox_to_anchor=(1.0, 0.93))
        plt.ylim(0)
        plt.ylabel('Em loadings', fontsize=11)
        plt.title(f'Model {f}\nAB VS CD - Em', fontsize=13)
        plt.subplot(2, 2, 3)
        for i in range(f):
            line = plt.plot(self.em, self.split_result[f][1][0][1][:, i], linestyle='-')
            plt.plot(self.em, self.split_result[f][1][1][1][:, i], c=line[0].get_color(), linestyle='--')
        plt.title(f'Model {f}\nAC VS BD - Em', fontsize=13)
        plt.ylim(0)
        plt.xlabel('Emission/nm', fontsize=11)
        plt.ylabel('Em loadings', fontsize=11)
        plt.subplot(2, 2, 2)
        for i in range(f):
            line = plt.plot(self.ex, self.split_result[f][0][0][2][:, i], linestyle='-')
            plt.plot(self.ex, self.split_result[f][0][1][2][:, i], c=line[0].get_color(), linestyle='--')
        plt.title(f'Model {f}\nAB VS CD - Ex', fontsize=13)
        plt.ylim(0)
        plt.ylabel('Ex loadings', fontsize=11)
        plt.subplot(2, 2, 4)
        for i in range(f):
            line = plt.plot(self.ex, self.split_result[f][1][0][2][:, i], linestyle='-', label=f'AC comp {i + 1}')
            plt.plot(self.ex, self.split_result[f][1][1][2][:, i], c=line[0].get_color(), linestyle='--',
                     label=f'BD comp {i + 1}')
        plt.title(f'Model {f}\nAC VS BD - Ex', fontsize=13)
        plt.ylim(0)
        plt.xlabel('Excitation/nm', fontsize=11)
        plt.ylabel('Ex loadings', fontsize=11)
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels, framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False,
                   bbox_to_anchor=(1.0, 0.46))
        plt.gca().add_artist(lg1)
        plt.subplots_adjust(left=0.08, bottom=0.09, right=0.85, top=0.92, wspace=0.23, hspace=0.32)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f": {save_path}")

        #   Em  Ex 
        save_dir = os.path.dirname(save_path_1)

        # ---  Em Loadings ---
        em_data = {}
        for idx in range(f):
            em_data[f'AB_Em_Comp{idx + 1}'] = self.split_result[f][0][0][1][:, idx]
            em_data[f'CD_Em_Comp{idx + 1}'] = self.split_result[f][0][1][1][:, idx]
            em_data[f'AC_Em_Comp{idx + 1}'] = self.split_result[f][1][0][1][:, idx]
            em_data[f'BD_Em_Comp{idx + 1}'] = self.split_result[f][1][1][1][:, idx]
        em_df = pd.DataFrame(em_data, index=self.em)
        em_df.index.name = 'Emission_Wavelength'
        em_df.to_excel(os.path.join(save_dir, f'split_result_data_Em_{f}.xlsx'))

        # ---  Ex Loadings ---
        ex_data = {}
        for idx in range(f):
            ex_data[f'AB_Ex_Comp{idx + 1}'] = self.split_result[f][0][0][2][:, idx]
            ex_data[f'CD_Ex_Comp{idx + 1}'] = self.split_result[f][0][1][2][:, idx]
            ex_data[f'AC_Ex_Comp{idx + 1}'] = self.split_result[f][1][0][2][:, idx]
            ex_data[f'BD_Ex_Comp{idx + 1}'] = self.split_result[f][1][1][2][:, idx]
        ex_df = pd.DataFrame(ex_data, index=self.ex)
        ex_df.index.name = 'Excitation_Wavelength'
        ex_df.to_excel(os.path.join(save_dir, f'split_result_data_Ex_{f}.xlsx'))
        print(f" (EmissionExcitation)")

        plt.show()

    def plot_fmax(self, f: int, use_file_list=False, save_path=None, save_path_1=None):
        """
        Plot FMax to see components' variation among samples.
        :param f: int, number of components
        :param use_file_list: Optional, if True, use file_list in xticks. Default is False
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        fig = plt.figure(figsize=(6, 4.8))
        fig.canvas.manager.set_window_title(f'FMax plot of {f}-component')
        fmax = self.factors[f][0] * self.factors[f][1].max(axis=0) * self.factors[f][2].max(axis=0)
        for i in range(f):
            plt.plot(fmax[:, i], label=f'Component {i + 1}')
        plt.xlabel('Sample', fontsize=11)
        if use_file_list:
            plt.xticks(range(self.nSample), self.file_list)
        plt.ylabel('FMax', fontsize=11)
        plt.title('FMax for the samples', fontsize=13)
        plt.legend(framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False)
        plt.subplots_adjust(left=0.13, bottom=0.11, right=0.95, top=0.9)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fmax: {save_path}")
            # FMax
            save_dir = os.path.dirname(save_path_1)
            df = pd.DataFrame(fmax, columns=[f'FMax_Component_{i + 1}' for i in range(f)])
            df.insert(0, 'Sample', self.file_list)
            df.to_excel(os.path.join(save_dir, f'fmax_data_{f}.xlsx'), index=False)
            print(f"FMax")
        plt.show()

    def plot_fingers(self, f: int, save_path=None, save_path_1=None):
        """
        Plot components in one figure, with Emission on X-axis, Excitation on Y-axis, smooth 'jet' colormap, and high resolution.

        :param f: int, number of components
        :param save_path: optional, path to save the figure
        """
        from scipy.ndimage import gaussian_filter
        plt.rcParams['font.sans-serif'] = 'Arial'
        model = self.__finger_model(f)
        print(f'Description of PARAFAC components - Model {f}')
        peak = [[], []]
        for i in [1, 2]:
            w = self.em if i == 1 else self.ex
            for j in range(f):
                index = find_peaks(self.factors[f][i][:, j])
                peak[i - 1].append(w[index[0]])
        for j in range(f):
            print(f'Component {j + 1}')
            for i in [1, 2]:
                if i == 1:
                    print('Em:', end='\t')
                else:
                    print('Ex:', end='\t')
                for k in peak[i - 1][j]:
                    print(k, end='\t')
                print('')
        r, c, rr = 1, f, 4.3
        left, bottom, top = 0.07, 0.13, 0.9
        if f > 3:
            r, c, rr = 2, int(np.ceil(f / 2)), 7.5
            left, bottom, top = 0.09, 0.076, 0.92
        fig = plt.figure(figsize=(4.5 * c, rr))
        fig.canvas.manager.set_window_title('Plot components')
        contour_levels = 100
        for ii in range(1, f + 1):
            plt.subplot(r, c, ii)
            Z = model[ii - 1].T
            Z_smooth = gaussian_filter(Z, sigma=1.5)
            cs = plt.contourf(self.em, self.ex, Z_smooth, levels=contour_levels, cmap='jet')
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Excitation/nm', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('Emission/nm', fontsize=11)
        plt.subplots_adjust(left=left, bottom=bottom, right=0.94, top=top, wspace=0.24, hspace=0.2)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f": {save_path}")
            # 
            if save_path_1:
                save_dir = os.path.dirname(save_path_1)
                os.makedirs(save_dir, exist_ok=True)
                for idx, m in enumerate(model):
                    df = pd.DataFrame(m, index=self.em, columns=self.ex)
                    df.index.name = 'Emission'
                    df.columns.name = 'Excitation'
                    df.to_excel(os.path.join(save_dir, f'finger_component_{idx + 1}_{f}.xlsx'))
                print(f"")
        plt.show()

    def plot_loadings(self, f: int, save_path=None, save_path_1=None):
        """
        Plot loadings of components

        :param f: int, number of components
        """
        plt.rcParams['font.sans-serif'] = 'Arial'
        r, c, rr = 1, f, 4.3
        left, bottom, top = 0.07, 0.13, 0.9
        if f > 3:
            r, c, rr = 2, int(np.ceil(f / 2)), 7.5
            left, bottom, top = 0.09, 0.076, 0.92
        fig = plt.figure(figsize=(4.5 * c, rr))
        fig.canvas.manager.set_window_title('Plot loadings for all components')
        for ii in range(1, f + 1):
            plt.subplot(r, c, ii)
            plt.plot(self.ex, self.factors[f][2][:, ii - 1], label='Ex')
            plt.plot(self.em, self.factors[f][1][:, ii - 1], label='Em')
            plt.legend(framealpha=0.6, loc='upper right', fontsize=11, frameon=True, shadow=False)
            plt.xlim(min(self.ex.min(), self.em.min()),
                     max(self.ex.max(), self.em.max()))
            plt.ylim(0)
            plt.title(f'Component {ii}', fontsize=13)
            if ii % c == 1:
                plt.ylabel('Loadings', fontsize=11)
            if ii > (r - 1) * c:
                plt.xlabel('$\lambda$/nm', fontsize=11)
        plt.subplots_adjust(left=left, bottom=bottom, right=0.94, top=top, wspace=0.24, hspace=0.2)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f": {save_path}")
            # 
            save_dir = os.path.dirname(save_path_1)
            em_df = pd.DataFrame(self.factors[f][1], index=self.em, columns=[f'Em_Component_{i+1}' for i in range(f)])
            ex_df = pd.DataFrame(self.factors[f][2], index=self.ex, columns=[f'Ex_Component_{i+1}' for i in range(f)])
            em_df.to_excel(os.path.join(save_dir, f'em_loadings_{f}.xlsx'))
            ex_df.to_excel(os.path.join(save_dir, f'ex_loadings_{f}.xlsx'))
            print(f"")
        plt.show()

    #  Data output
    def open_fluor(self, f: int, output_path=None, **kwargs):
        """
        Output a text file which can be uploaded to the OpenFluor to compare.

        :param f: int, number of components
        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        :param kwargs: Optional, consist of personal information
        """
        if output_path is None:
            file_name = f'{f}-component model.txt'
        else:
            file_name = output_path + r'\\' + f'{f}-component model.txt'
        ex_str = ''
        for i in range(self.nex):
            ex_str += f'EX\t{self.ex[i]}\t'
            s = '\t'.join(f'{self.factors[f][2][i, j]}' for j in range(f))
            ex_str = ex_str + s + '\n'
        em_str = ''
        for i in range(self.nem):
            em_str += f'EM\t{self.em[i]}\t'
            s = '\t'.join(f'{self.factors[f][1][i, j]}' for j in range(f))
            em_str = em_str + s + '\n'
        em_str = em_str[:-1]
        name = kwargs.get('name', '')
        creator = kwargs.get('creator', '')
        email = kwargs.get('email', '')
        doi = kwargs.get('doi', '')
        reference = kwargs.get('reference', '')
        fluorometer = kwargs.get('fluorometer', '')
        sources = kwargs.get('sources', '')
        ecozones = kwargs.get('ecozones', '')
        method = kwargs.get('method', '')
        preprocess = kwargs.get('preprocess', '')
        description = kwargs.get('description', '')
        with open(file_name, 'a') as f:
            f.write(f'#\n# Fluorescence Model\n#\n'
                    f'name\t{name}\ncreator\t{creator}\nemail\t{email}\ndoi\t{doi}\n'
                    f'reference\t{reference}\nunit\t{self.iu}\ntoolbox\tEEM toolkit for Python\n'
                    f'date\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                    f'fluorometer\t{fluorometer}\nnSample\t{self.nSample}\nconstraints\tnon-negativity\n'
                    f'validation\tSplit-half analysis (S4C4T2)\nmethods\t{method}\n'
                    f'preprocess\t{preprocess}\nsources\t{sources}\necozones\t{ecozones}\n'
                    f'description\t{description}\n'
                    f'#\n# Excitation/Emission (Ex, Em), wavelength (nm), component[n] (intensity)\n#\n'
                    f'{ex_str}{em_str}')

    def eems_output(self, fac, output_root):
        """
        Output the EEMs
        :param output_root: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        output_path = os.path.join(output_root, str(fac))
        os.makedirs(output_path, exist_ok=True)

        def save_sample(i):
            out = np.full((self.nem + 1, self.nex + 1), np.nan)
            out[1:, 1:] = self.x[i]
            out[0, 1:] = self.ex
            out[1:, 0] = self.em
            out = pd.DataFrame(out)

            if isinstance(self.file_list[i], (int, np.integer)):
                sample_name = f"sample_{i + 1}"
            else:
                sample_name = str(self.file_list[i])
            file_name = os.path.join(output_path, sample_name + '.xlsx')
            out.to_excel(file_name, header=False, index=False)
            print(f' {file_name}')

        with ThreadPoolExecutor() as executor:
            executor.map(save_sample, range(self.nSample))

        print(f' {fac}  {output_path}')

    def parafac_result_output(self, f: int, output_path=None):
        """
        Output PARAFAC results, including FMax, Em loadings, Ex loadings and components.

        :param f: int, number of components
        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        fmax = pd.DataFrame(self.factors[f][0] * self.factors[f][1].max(axis=0) * self.factors[f][2].max(axis=0),
                            columns=[f'FMax{i + 1}' for i in range(f)])
        fmax.insert(loc=0, column='EEM', value=self.file_list)
        em_loadings = pd.DataFrame(self.factors[f][1],
                                   columns=[f'Em{i + 1}' for i in range(f)])
        em_loadings.insert(loc=0, column='Em', value=self.em)
        ex_loadings = pd.DataFrame(self.factors[f][2],
                                   columns=[f'Ex{i + 1}' for i in range(f)])
        ex_loadings.insert(loc=0, column='Ex', value=self.ex)
        qx = self.__finger_model(f)
        out_model = []
        for j in qx:
            out = np.full((self.nem + 1, self.nex + 1), np.nan)
            out[1:, 1:] = j
            out[0, 1:] = self.ex
            out[1:, 0] = self.em
            out_model.append(pd.DataFrame(out))
        if output_path is None:
            file_name = f'{f}-component result.xlsx'
        else:
            file_name = output_path + r'\\' + f'{f}-component result.xlsx'
        with pd.ExcelWriter(file_name) as writer:
            fmax.to_excel(writer, sheet_name='FMax', index=False)
            em_loadings.to_excel(writer, sheet_name='Em loadings', index=False)
            ex_loadings.to_excel(writer, sheet_name='Ex loadings', index=False)
            for j in range(len(out_model)):
                out_model[j].to_excel(writer, sheet_name=f'C{j + 1}', header=False, index=False)

    def fri_result_output(self, output_path=None):
        """
        Calculate FRI and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        p, phi, mf = self.__fluorescence_regional_integration()
        cols = np.arange(1, 6)
        pfr = pd.DataFrame(p, columns=cols)
        pfr.insert(loc=0, column='Sample', value=self.file_list)
        volume = pd.DataFrame(phi, columns=cols)
        volume.insert(loc=0, column='Sample', value=self.file_list)
        mf = pd.DataFrame(mf, columns=cols)
        if output_path is None:
            file_name = 'FRI.xlsx'
        else:
            file_name = output_path + r'\\FRI.xlsx'
        with pd.ExcelWriter(file_name) as writer:
            pfr.to_excel(writer, sheet_name='percent fluorescence response', index=False)
            volume.to_excel(writer, sheet_name='volume', index=False)
            mf.to_excel(writer, sheet_name='multiplication factor', index=False)

    def fluorescence_indices_output(self, output_path=None):
        """
        Calculate fluorescence indices and output

        :param output_path: Optional, the directory to save the text. If not given,
            save in the current directory.
        """
        fl, fre, hix, bix = self.__pick_fluorescence_indices()
        df = pd.DataFrame()
        df['Sample'] = self.file_list
        df['FluI'] = fl
        df['FreI'] = fre
        df['BIX'] = bix
        if output_path is None:
            file_name = 'fluorescence_indices.xlsx'
        else:
            file_name = output_path + r'\\fluorescence_indices.xlsx'
        if hix:
            df['HIX'] = hix
        df.to_excel(file_name, index=False)

    def classify_components(self, f: int, output_path=None):
        """
        drEEM
        :param f: int
        :param output_path: Optional
        """
        results = []
        ex_loadings = self.factors[f][2]  # excitation loading
        em_loadings = self.factors[f][1]  # emission loading
        for i in range(f):
            ex_wavelength = self.ex
            em_wavelength = self.em
            ex_peak_idx = np.argmax(ex_loadings[:, i])
            em_peak_idx = np.argmax(em_loadings[:, i])
            ex_peak = ex_wavelength[ex_peak_idx]
            em_peak = em_wavelength[em_peak_idx]
            # 
            if 270 <= ex_peak <= 280 and 330 <= em_peak <= 350:
                group = 'Tryptophan-like ( Peak T)'
            elif 270 <= ex_peak <= 280 and 300 <= em_peak <= 310:
                group = 'Tyrosine-like ( Peak B)'
            elif 240 <= ex_peak <= 270 and 380 <= em_peak <= 460:
                group = 'Humic-like ( Peak A)'
            elif 310 <= ex_peak <= 350 and 420 <= em_peak <= 480:
                group = 'Humic-like ( Peak C)'
            elif 300 <= ex_peak <= 320 and 370 <= em_peak <= 390:
                group = 'Microbial-like ( Peak M)'
            elif 340 <= ex_peak <= 360 and 440 <= em_peak <= 460:
                group = 'Marine humic-like ( Peak N)'
            else:
                group = 'Unknown ()'
            results.append({
                'Component': f'C{i+1}',
                'Excitation Peak (nm)': ex_peak,
                'Emission Peak (nm)': em_peak,
                'Classification': group
            })
        df = pd.DataFrame(results)
        if output_path:
            df.to_excel(output_path, index=False)
            print(f': {output_path}')
        else:
            print(df)

    def clear_scatter_after_interpolation_v2(self):
        """
         cut_ray_scatter  cut_ram_scatter 

        
        -  cut_ray_scatter  cut_ram_scatter
        - 
        -  0
        """
        try:
            width_primary = self._rayleigh_width_1st  # [, ] 
            width_secondary = self._rayleigh_width_2nd  # 
            width_raman_1st = self._raman_width_1st  # 
            width_raman_2nd = self._raman_width_2nd  # 
            raman_shift = self._raman_shift
        except AttributeError:
            raise ValueError(" cut_ray_scatter  cut_ram_scatter ")

        em_wavelengths = self.em
        ex_wavelengths = self.ex
        em_grid, ex_grid = np.meshgrid(em_wavelengths, ex_wavelengths, indexing='ij')

        # ===  ===
        rayleigh1_mask = (em_grid >= ex_grid - width_primary[0]) & (em_grid <= ex_grid + width_primary[1])
        rayleigh2_mask = (em_grid >= 2 * ex_grid - width_secondary[0]) & (em_grid <= 2 * ex_grid + width_secondary[1])

        # ===  ===
        raman_center = 1 / (1 / ex_grid - raman_shift / 1e7)
        raman1_mask = (em_grid >= raman_center - width_raman_1st[0]) & (em_grid <= raman_center + width_raman_1st[1])
        raman2_mask = (em_grid >= 2 * raman_center - width_raman_2nd[0]) & (
                    em_grid <= 2 * raman_center + width_raman_2nd[1])

        # 
        total_mask = rayleigh1_mask | rayleigh2_mask | raman1_mask | raman2_mask

        for i in range(self.nSample):
            self.x[i][total_mask] = 0  #  0

        print('')

    def save_split_analysis_summary(self, save_path="split_analysis_full_summary.xlsx"):
        """
         TCC
        """
        # === Step 1:  ===
        records = []
        for comp in sorted(self.split_result.keys()):
            # Split Factor Similarity
            split_models = self.split_result[comp]
            compare_to_all = [
                tlviz.factor_tools.cosine_similarity(split_models[0][0][1], self.factors[comp][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][1], self.factors[comp][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][1], self.factors[comp][1]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][1], self.factors[comp][1]),
                tlviz.factor_tools.cosine_similarity(split_models[0][0][2], self.factors[comp][2]),
                tlviz.factor_tools.cosine_similarity(split_models[0][1][2], self.factors[comp][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][0][2], self.factors[comp][2]),
                tlviz.factor_tools.cosine_similarity(split_models[1][1][2], self.factors[comp][2])
            ]
            factor_similarity = min(compare_to_all)
            # TCC
            em_cs = [tlviz.factor_tools.cosine_similarity(split_models[i][0][1], split_models[i][1][1]) for i in [0, 1]]
            ex_cs = [tlviz.factor_tools.cosine_similarity(split_models[i][0][2], split_models[i][1][2]) for i in [0, 1]]
            # 
            validated = "Yes" if factor_similarity > 0.95 else "No"
            #   
            core_consistency = self.core_consistency.get(comp, None)
            explanation = self.explanation_rate.get(comp, None)
            records.append({
                "Component": comp,
                "Ex_TCC_1": ex_cs[0],
                "Ex_TCC_2": ex_cs[1],
                "Em_TCC_1": em_cs[0],
                "Em_TCC_2": em_cs[1],
                "Validated": validated,
                "Factor_Similarity": factor_similarity,
                "Core_Consistency": core_consistency,
                "Explanation_Rate": explanation
            })
        # === Step 2:  Excel ===
        df = pd.DataFrame(records)
        df.to_excel(save_path, index=False)
        print(f": {save_path}")
