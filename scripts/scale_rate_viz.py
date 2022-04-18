# -*- coding: utf-8 -*-
import numpy as np
import warnings
from kymatio.numpy import TimeFrequencyScattering1D, Scattering1D
from kymatio.toolkit import echirp, pack_coeffs_jtfs
from kymatio.visuals import plot, imshow
from numpy.fft import ifft, ifftshift
import matplotlib.pyplot as plt

# sampling rate - only affects axis labels
#  None: will label axes with discrete units (cycles/sample), `xi`
#  int: will label axes with physical units, `xi *= Fs`
FS = 4096
got_fs = bool(FS is not None)
if FS is None:
    FS = 1

#%% Generate echirp and create scattering object #############################
N = 4096
# span low to Nyquist; assume duration of 1 second
x = echirp(N, fmin=64, fmax=N/2) / 2

# xe = echirp(N//2, fmin=64, fmax=N/4)
x += np.cos(2*np.pi * 364 * np.linspace(0, 1, N)) / 2
x[N//2-16:N//2+16] += 5


#%% Show joint wavelets on a smaller filterbank ##############################
o = (0, 999)[1]
rp = np.sqrt(.5)
ckw = dict(shape=N, J=(8, 8), Q=(16, 1), T=2**8)
jtfs = TimeFrequencyScattering1D(**ckw, J_fr=3, Q_fr=1,
                                 sampling_filters_fr='resample', analytic=1,
                                 average=0, average_fr=0, F=8,
                                 r_psi=(rp, rp, rp), out_type='dict:list',
                                 pad_mode_fr=('conj-reflect-zero', 'zero')[1],
                                 oversampling=o, oversampling_fr=o,
                                  paths_exclude={'n2':    [3, 4],
                                                 'n1_fr': []},
                                 max_noncqt_fr=0)
#%% scalogram
sc = Scattering1D(**ckw, average=False, out_type='list', oversampling=999)
S1 = np.array([c['coef'].squeeze() for c in sc(x)])[sc.meta()['order'] == 1]

#%% configure styling ########################################################
def format_ticks(ticks, max_digits=3):
    # `max_digits` not strict
    not_iterable = bool(not isinstance(ticks, (tuple, list, np.ndarray)))
    if not_iterable:
        ticks = [ticks]
    _ticks = []
    for tk in ticks:
        negative = False
        if tk < 0:
            negative = True
            tk = abs(tk)

        n_nondecimal = np.log10(tk)
        if n_nondecimal < 0:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)) + 1)
            n_total = n_nondecimal + 2
            tk = f"%.{n_total - 1}f" % tk
        else:
            n_nondecimal = int(np.ceil(abs(n_nondecimal)))
            n_decimal = max(0, max_digits - n_nondecimal)
            tk = round(tk, n_decimal)
            tk = f"%.{n_decimal}f" % tk

        if negative:
            tk = "-" + tk
        _ticks.append(tk)
    if not_iterable:
        _ticks = _ticks[0]
    return _ticks

# ticks & units
if got_fs:
    f_units = "[Hz]"
    t_units = "[sec]"
else:
    f_units = "[cycles/sample]"
    t_units = "[samples]"

yticks = np.array([p['xi'] for p in sc.psi1_f])
if got_fs:
    t = np.linspace(0, N/FS, N, endpoint=False)
    yticks *= FS
else:
    t = np.arange(N)

# axis labels
lkw = {'fontsize': 18, 'weight': 'bold'}
xlabel  = (f"Time {t_units}", lkw)
ylabel0 = ("Amplitude", lkw)
ylabel1 = (f"Frequency {f_units}", lkw)
# titles
tkw = {'fontsize': 20, 'weight': 'bold'}
title0 = ("Exponential chirp + pure sine + pulse", tkw)
title1 = ("Scalogram", tkw)
# format yticks (limit # of shown decimal digits, and round the rest)
yticks = (format_ticks(yticks), {'fontsize': 14})

# plot #######################################################################
fig, ax = plt.gcf(), plt.gca()
plot(t, x, xlabel=xlabel, ylabel=ylabel0, fig=fig, ax=ax,
     title=title0, show=0)
ax.tick_params(labelsize=16)
plt.show()

imshow(S1, abs=1, xlabel=xlabel, ylabel=ylabel1, title=title1,
       yticks=yticks, xticks=(t, {'fontsize': 15}))

#%% scatter
Scx_orig = jtfs(x)
jmeta = jtfs.meta()

#%% pack
Scx = pack_coeffs_jtfs(Scx_orig, jmeta, structure=2, out_3D=False)
Scx = Scx[:, :, ::-1]

#%%
psi_id = 0
n_n2s = sum(p['j'] > 0 for p in jtfs.psi2_f)
n2s    = np.unique(jmeta['n']['psi_t * psi_f_up'][:, 0])
n1_frs = np.unique(jmeta['n']['psi_t * psi_f_up'][:, 1])
n_n2s, n_n1_frs = len(n2s), len(n1_frs)
# drop spin up
# Scx = Scx[:, n_n1_frs:]
# drop spin dn
# Scx = Scx[:, :n_n1_frs + 1]

#%%
psi2s = [p for n2, p in enumerate(jtfs.psi2_f) if n2 in n2s]
psis_up, psis_dn = [[p for n1_fr, p in enumerate(psi1_f_fr[psi_id])
                     if n1_fr in n1_frs]
                    for psi1_f_fr in (jtfs.psi1_f_fr_up, jtfs.psi1_f_fr_dn)]
pdn_meta = {field: [value for n1_fr, value in
                    enumerate(jtfs.psi1_f_fr_dn[field][psi_id])
                    if n1_fr in n1_frs]
            for field in jtfs.psi1_f_fr_dn if isinstance(field, str)}

# reverse ordering (meta already done)
psi2s   = psi2s
psis_dn = psis_dn

#%% Visualize ################################################################
viz_coef = 1
viz_filterbank = 1
viz_spin_up = 1
viz_spin_dn = 1
axis_labels = 1
phi_t_blank = 1

phi_t_loc = 'bottom'
xylabel_kw = dict(fontsize=20)
title_kw = dict(weight='bold', fontsize=26, y=1.02)
xy_suplabel_kw = dict(weight='bold', fontsize=24)

assert phi_t_loc in ('top', 'bottom', 'both')
if phi_t_loc == 'both':
    raise NotImplementedError
    if phi_t_blank:
        warnings.warn("`phi_t_blank` does nothing if `phi_t_loc='blank'`")
        phi_t_blank = 0

def no_border(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

def to_time(p_f):
    while isinstance(p_f, (dict, list)):
        p_f = p_f[0]
    return ifftshift(ifft(p_f.squeeze()))

if viz_filterbank:
    imshow_kw0 = dict(aspect='auto', cmap='bwr')#, interpolation='none')
if viz_coef:
    imshow_kw1 = dict(aspect='auto', cmap='turbo')#, interpolation='none')

if viz_spin_up and viz_spin_dn:
    n_rows = 2*n_n1_frs + 1
else:
    n_rows = n_n1_frs + 1
n_cols = n_n2s + 1

w = 11
h = 11 * n_rows / n_cols

if viz_filterbank:
    fig0, axes0 = plt.subplots(n_rows, n_cols, figsize=(w, h))
if viz_coef:
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(w, h))

# compute common params to zoom on wavelets based on largest wavelet
# centers
n1_fr_largest = n_n1_frs - 1
n2_largest = n_n2s - 1
pf_f = psis_dn[n1_fr_largest].squeeze()
pt_f = psi2s[n2_largest][0].squeeze()
ct = len(pt_f) // 2
cf = len(pf_f) // 2
# supports
st = int(psi2s[n2_largest]['support'][0]    / 1.8)
sf = int(pdn_meta['support'][n1_fr_largest] / 1.8)

# coeff max
cmx = Scx.max() * .8

def plot_spinned(up):
    def label_axis(ax, n1_fr_idx, n2_idx):
        at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
        if at_border:
            xi2 = psi2s[::-1][n2_idx]['xi']
            if got_fs:
                xi2 = xi2 * FS
            xi2 = format_ticks(xi2)
            ax.set_xlabel(xi2, **xylabel_kw)

    if up:
        psi1_frs = psis_up
    else:
        psi1_frs = psis_dn[::-1]

    for n2_idx, pt_f in enumerate(psi2s[::-1]):
        for n1_fr_idx, pf_f in enumerate(psi1_frs):
            # compute axis & coef indices ####################################
            if up:
                row_idx = n1_fr_idx
                coef_n1_fr_idx = n1_fr_idx
            else:
                if viz_spin_up:
                    row_idx = n1_fr_idx + 1 + n_n1_frs
                else:
                    row_idx = n1_fr_idx + 1
                coef_n1_fr_idx = n1_fr_idx + n_n1_frs + 1
            col_idx = n2_idx + 1
            coef_n2_idx = n2_idx + 1

            # visualize ######################################################
            # filterbank
            if viz_filterbank:
                pt = to_time(pt_f)
                pf = to_time(pf_f)
                # trim to zoom on wavelet
                pt = pt[ct - st:ct + st + 1]
                pf = pf[cf - sf:cf + sf + 1]

                Psi = pf[:, None] * pt[None]
                mx = np.abs(Psi).max()

                ax0 = axes0[row_idx, col_idx]
                ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)

                # axis styling
                no_border(ax0)
                if axis_labels:
                    label_axis(ax0, n1_fr_idx, n2_idx)

            # coeffs
            if viz_coef:
                c = Scx[coef_n2_idx, coef_n1_fr_idx]

                ax1 = axes1[row_idx, col_idx]
                ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)

                # axis styling
                no_border(ax1)
                if axis_labels:
                    label_axis(ax1, n1_fr_idx, n2_idx)

if viz_spin_up:
    plot_spinned(up=True)
if viz_spin_dn:
    plot_spinned(up=False)

# psi_t * phi_f ##############################################################
if viz_filterbank:
    phif = to_time(jtfs.phi_f_fr)
    phif = phif[cf - sf:cf + sf + 1]

if viz_spin_up:
    row_idx = n_n1_frs
else:
    row_idx = 0
coef_n1_fr_idx = n_n1_frs

for n2_idx, pt_f in enumerate(psi2s[::-1]):
    # compute axis & coef indices
    col_idx = n2_idx + 1
    coef_n2_idx = n2_idx + 1

    # filterbank
    if viz_filterbank:
        pt = to_time(pt_f)
        pt = pt[ct - st:ct + st + 1]

        Psi = phif[:, None] * pt[None]
        mx = np.abs(Psi).max()

        ax0 = axes0[row_idx, col_idx]
        ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)
        no_border(ax0)

    # coeffs
    if viz_coef:
        ax1 = axes1[row_idx, col_idx]
        c = Scx[coef_n2_idx, coef_n1_fr_idx]
        ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)
        no_border(ax1)

# phi_t * psi_f ##############################################################
def plot_phi_t(up):
    def label_axis(ax, n1_fr_idx):
        """`xi1_fr` units

        Meta stores discrete, [cycles/sample].
        Want [cycles/octave].
        To get physical, we do `xi * fs`, where `fs [samples/second]`.
        Hence, find `fs` equivalent for `octaves`.

        If `Q1` denotes "number of first-order wavelets per octave", we realize
        that "samples" of `psi_fr` are actually "first-order wavelets":
            `xi [cycles/(first-order wavelets)]`
        Hence, set
            `fs [(first-order wavelets)/octave]`
        and so
            `xi1_fr = xi*fs = xi*Q1 [cycles/octave]`

         - This is consistent with raising `Q1` being equivalent of raising
           the physical sampling rate (i.e. sample `n1` more densely without
           changing the number of octaves).
         - Raising `J1` is then equivalent to increasing physical duration
           (seconds) without changing sampling rate, so `xi1_fr` is only a
           function of `Q1`.
        """
        if up:
            filter_n1_fr_idx = n1_fr_idx
        else:
            filter_n1_fr_idx = n_n1_frs - n1_fr_idx - 1

        xi1_fr = pdn_meta['xi'][filter_n1_fr_idx] * jtfs.Q[0]
        if not up:
            xi1_fr = -xi1_fr
        xi1_fr = format_ticks(xi1_fr)
        ax.set_ylabel(xi1_fr, **xylabel_kw)

        at_border = bool(n1_fr_idx == len(psi1_frs) - 1)
        if at_border:
            ax.set_xlabel("0", **xylabel_kw)

    if phi_t_loc == 'top':
        if up:
            psi1_frs = psis_up
            assert not viz_spin_dn or (viz_spin_up and viz_spin_dn)
        else:
            if viz_spin_up and viz_spin_dn:
                # don't show stuff if both spins given
                psi1_frs = [p*0 for p in psis_up]
            else:
                psi1_frs = psis_dn[::-1]
    elif phi_t_loc == 'bottom':
        if up:
            if viz_spin_up and viz_spin_dn:
                # don't show stuff if both spins given
                psi1_frs = [p*0 for p in psis_up]
            else:
                psi1_frs = psis_up
        else:
            psi1_frs = psis_dn[::-1]
            assert not viz_spin_up or (viz_spin_up and viz_spin_dn)

    col_idx = 0
    coef_n2_idx = 0
    for n1_fr_idx, pf_f in enumerate(psi1_frs):
        if up:
            row_idx = n1_fr_idx
            coef_n1_fr_idx = n1_fr_idx
        else:
            if viz_spin_up and viz_spin_dn:
                row_idx = n1_fr_idx + 1 + n_n1_frs
            else:
                row_idx = n1_fr_idx + 1
            coef_n1_fr_idx = n1_fr_idx + 1 + n_n1_frs

        if viz_filterbank:
            pf = to_time(pf_f)
            pf = pf[cf - sf:cf + sf + 1]

            Psi = pf[:, None] * phit[None]

            ax0 = axes0[row_idx, col_idx]

            if phi_t_loc != 'both':
                # energy norm (no effect if color norm adjusted to Psi)
                Psi *= np.sqrt(2)

            if phi_t_loc == 'top':
                if not up and (viz_spin_up and viz_spin_dn):
                    # actually zero but that defaults the plot to max negative
                    mx = 1
                else:
                    mx = np.abs(Psi).max()
            elif phi_t_loc == 'bottom':
                if up and (viz_spin_up and viz_spin_dn):
                    # actually zero but that defaults the plot to max negative
                    mx = 1
                else:
                    mx = np.abs(Psi).max()
            ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)

            # axis styling
            no_border(ax0)
            if axis_labels:
                label_axis(ax0, n1_fr_idx)

        if viz_coef:
            ax1 = axes1[row_idx, col_idx]
            skip_coef = bool(phi_t_blank and ((phi_t_loc == 'top' and not up) or
                                              (phi_t_loc == 'bottom' and up)))

            if not skip_coef:
                c = Scx[coef_n2_idx, coef_n1_fr_idx]
                if phi_t_loc != 'both':
                    # energy norm since we viz only once;
                    # did /= sqrt(2) in pack_coeffs_jtfs
                    c = c * np.sqrt(2)
                if phi_t_loc == 'top':
                    if not up and (viz_spin_up and viz_spin_dn):
                        c = c * 0  # viz only once
                elif phi_t_loc == 'bottom':
                    if up and (viz_spin_up and viz_spin_dn):
                        c = c * 0  # viz only once
                ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)

            # axis styling
            no_border(ax1)
            if axis_labels:
                label_axis(ax1, n1_fr_idx)

if viz_filterbank:
    phit = to_time(jtfs.phi_f)
    phit = phit[ct - st:ct + st + 1]

if viz_spin_up:
    plot_phi_t(up=True)
if viz_spin_dn:
    plot_phi_t(up=False)

# phi_t * phi_f ##############################################################
def label_axis(ax):
    ax.set_ylabel("0", **xylabel_kw)

if viz_spin_up:
    row_idx = n_n1_frs
else:
    row_idx = 0
col_idx = 0
coef_n2_idx = 0
coef_n1_fr_idx = n_n1_frs

# filterbank
if viz_filterbank:
    Psi = phif[:, None] * phit[None]
    mx = np.abs(Psi).max()

    ax0 = axes0[row_idx, col_idx]
    ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)

    # axis styling
    no_border(ax0)
    label_axis(ax0)

# coeffs
if viz_coef:
    c = Scx[coef_n2_idx, coef_n1_fr_idx]
    ax1 = axes1[row_idx, col_idx]
    ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)

    # axis styling
    no_border(ax1)
    label_axis(ax1)

# finalize ###################################################################
def fig_adjust(fig):
    if axis_labels:
        fig.supxlabel(f"Temporal modulation {f_units}",
                      **xy_suplabel_kw, y=-.037)
        fig.supylabel("Freqential modulation [cycles/octave]",
                      **xy_suplabel_kw, x=-.066)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.02, hspace=.02)

if viz_filterbank:
    fig_adjust(fig0)
    if axis_labels:
        fig0.suptitle("JTFS filterbank (real part)", **title_kw)
if viz_coef:
    fig_adjust(fig1)
    if axis_labels:
        fig1.suptitle("JTFS coefficients", **title_kw)

base_name = 'jtfs_echirp_wavelets'
if viz_filterbank:
    fig0.savefig(f'{base_name}0.png', bbox_inches='tight')
if viz_coef:
    fig1.savefig(f'{base_name}1.png', bbox_inches='tight')
