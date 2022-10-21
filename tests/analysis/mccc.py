import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.signal import tukey
from scipy.sparse.linalg import lsmr


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def moving_average(data, ma):
    """
    moving average with AvgPool1d along axis=0
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    m = torch.nn.AvgPool1d(ma, stride=1, padding=ma // 2)
    data_ma = m(data.transpose(1, 0))[:, : data.shape[0]].transpose(1, 0)
    return data_ma


def taper_time(data, alpha=0.8):
    taper = tukey(data.shape[-1], alpha)
    return data * torch.tensor(taper, device=data.device)


def normalize(x):
    x -= torch.mean(x, dim=-1, keepdims=True)
    x /= x.square().sum(dim=-1, keepdims=True).sqrt()
    return x


def gather_roll(data, shift_index):
    """
    roll data[irow, :] along axis 1 by the amount of shift[irow]
    """
    nrow, ncol = data.shape
    index = torch.arange(ncol, device=data.device).view([1, ncol]).repeat((nrow, 1))
    index = (index - shift_index.view([nrow, 1])) % ncol
    return torch.gather(data, 1, index)


class MCCCPicker:
    """"""

    def __init__(
        self,
        data,
        dt,
        taper=0.8,
        ma=40,
        scale_factor=10,
        damp=1,
        mccc_mincc=0.7,
        mccc_maxlag=0.04,
        mccc_maxwin=10,
        chunk_size=50000,
        whitening=True,
        whitening_waterlevel=0.1,
        win_main=0.3,
        win_side=0.1,
        w0=10,
        max_niter=5,
        refine_ma=60,
        mode="pick",
        return_data_align=True,
    ):
        """
        pick/align data using multi-channel cross-correlation
        Ref: VanDecar-1990-Determination of teleseismic relative phase arrival times using
        multil-channel cross-correlation and least squares
        Args:
            data: multi-channel data, torch.Tensor, shape=[nchan, ntime]
                if the data is cross-correlation, use mode="pick" to pick the maximum absolute cross-correlations
                if the data is seismic waveform, use mode="align" to obtain the time shift
            dt: sampling interval
            taper: taper for input data
            ma: moving average window for input data
            scale_factor: upsampling factor for sub-sampling rate resolution
            damp: damping factor for smoothing the picking
            mccc_mincc: minimum cc for mccc
            mccc_maxlag: maxlag for mccc
            mccc_maxwin: maximum channel spanning for mccc
            chunk_size: maximum chunk_size
            whitening: if True, do spectral whitening
            whitening_waterlevel: waterlevel of spectral whitening
            win_main: in the "pick" mode: initial maximum window lag for picking the maximum absolute xcor of the main lobe
            win_side: in the "pick" mode: maximum window lag for picking the maximum absolute xcor of the side lobe
            w0: in the "pick" mode: weighting factor of mccc sparse matrix
            max_niter: in the "pick" mode: maximum number of iterations of refining the picking
            refine_ma: moving average window for iteratively refining the pick
            mode: "pick" or "align"
            return_data_align: if True, return the aligned data
        Returns:
            solution dictionary
        """
        self.device = data.device
        self.taper = taper
        self.ma = ma
        self.nchan, self.ntime = data.shape
        self.dt = dt
        self.fN = 0.5 / dt
        self.damp = damp
        # mccc control parameters
        self.chunk_size = chunk_size
        self.mccc_mincc = mccc_mincc
        self.mccc_maxlag = mccc_maxlag
        self.mccc_nlag = int(mccc_maxlag / dt)
        self.mccc_maxwin = mccc_maxwin
        # xcor parameters in frequency domain:
        assert int(scale_factor) >= 1
        self.scale_factor = int(scale_factor)
        self.npts_xcor = nextpow2(2 * self.ntime - 1)
        self.npts_xcor_pad = nextpow2(2 * self.scale_factor * self.ntime - 1)
        self.resample_factor = self.npts_xcor_pad / self.npts_xcor
        self.resample_dt = self.dt / self.resample_factor
        self.resample_mccc_nlag = int(mccc_maxlag / self.resample_dt)
        self.zero_complex = torch.complex(torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device))
        # mccc refine pick parameters
        self.refine_ma = refine_ma
        self.win_main = win_main
        self.win_side = win_side
        self.w0 = w0
        self.max_niter = max_niter
        # preprocessing
        if self.taper is not None:
            data = taper_time(data, self.taper)
        self.data_raw = data
        if self.ma is not None:
            data = moving_average(data, self.ma)
        self.data = normalize(data)
        self.data_freq = self.fft_real_xcor(self.data)
        # spectral whitening parameters
        self.whitening = whitening
        self.whitening_waterlevel = whitening_waterlevel
        # mode: "pick" or "align"
        self.mode = mode
        self.return_data_align = return_data_align

    def set_data(self, data):
        """
        set new data for mccc
        """
        assert self.data.shape == data.shape
        if self.taper is not None:
            data = taper_time(data, self.taper)
        self.data_raw = data
        if self.ma is not None:
            data = moving_average(data, self.ma)
        self.data = normalize(data)
        self.data_freq = self.fft_real_xcor(self.data)

    def fft_real_xcor(self, data):
        """
        prepare xcor in frequency domain
        """
        return torch.fft.rfft(data, n=self.npts_xcor, dim=-1)

    @property
    def fft_real_xcor_freq_axis(self):
        """
        xcor positive frequency axis
        """
        return torch.linspace(0, 0.5 / self.dt, self.npts_xcor // 2 + 1)

    def spectral_whitening(self, data_freq):
        """
        smooth the spectrum and enhance the xcor peak
        """
        # data_freq_smooth = moving_average(torch.abs(data_freq), 10)
        # data_freq /= (data_freq_smooth + self.whitening_waterlevel)
        data_freq_energy_1 = torch.mean(torch.abs(data_freq) ** 2, dim=-1, keepdim=True).sqrt()
        data_freq = data_freq / (torch.abs(data_freq) + self.whitening_waterlevel)
        data_freq_energy_2 = torch.mean(torch.abs(data_freq) ** 2, dim=-1, keepdim=True).sqrt()
        data_freq = data_freq / data_freq_energy_2 * data_freq_energy_1
        return data_freq

    def pick_data_win_maxabs(self, data, dt, maxlag, update_vmin=True):
        """"""
        nlag = int(maxlag / dt)
        nt = data.shape[-1]
        ic = nt // 2
        ib = max([0, ic - nlag + 1])
        ie = min([nt, ic + nlag])
        vmax, imax = torch.max(data[:, ib:ie], dim=-1)
        vmin, imin = torch.min(data[:, ib:ie], dim=-1)
        ineg = torch.abs(vmin) > vmax
        if update_vmin:
            vmax[ineg], vmin[ineg] = vmin[ineg], vmax[ineg]
        else:
            vmax[ineg] = vmin[ineg]
        imax[ineg] = imin[ineg]
        tmax = (imax - nlag + 1) * dt
        return (vmax, vmin, tmax)

    def pick_data_win_maxabs_side_lobe(self, data, dt, win_side):
        nt = data.shape[-1]
        ic = nt // 2
        # window including main and side lobes
        nwin2 = int(win_side / dt)
        indx_lb = max([0, ic - nwin2 + 1])
        indx_le = max([0, ic - nwin2 // 2 + 1])
        indx_rb = min([nt, ic + nwin2 // 2])
        indx_re = min([nt, ic + nwin2])
        vmin = torch.maximum(
            torch.max(torch.abs(data[:, indx_lb:indx_le]), dim=-1).values,
            torch.max(torch.abs(data[:, indx_rb:indx_re]), dim=-1).values,
        )
        return vmin

    def resample_data_in_freq_domain(self, data):
        """
        upsample the data padding zeros in frequency domain
        data_resample[..., 0] = data[..., 0]
        data_resample[..., -1] = data[..., -1]
        """
        ntime_b = data.shape[-1]
        ntime_q = ntime_b * self.scale_factor
        dt_resample = self.dt / self.scale_factor
        data_freq = torch.fft.rfft(data, dim=-1)
        if self.whitening:
            data_freq = self.spectral_whitening(data_freq)
        data_freq_pad = F.pad(data_freq, (0, ntime_q // 2 - ntime_b // 2), "constant", self.zero_complex)
        data_resample = torch.fft.irfft(data_freq_pad, dim=-1, n=ntime_b * self.scale_factor) * self.scale_factor
        npts_resample = (data.shape[-1] - 1) * self.scale_factor + 1
        return data_resample[:, :npts_resample], dt_resample

    def form_coo(self):
        """
        calculate the difference matrix based on neighboring channel cross-correlation
        """
        index_i = torch.tensor(
            [i for i in range(self.nchan - 1) for _ in range(i + 1, min(i + self.mccc_maxwin, self.nchan))],
            device=self.device,
        )
        index_j = torch.tensor(
            [j for i in range(self.nchan - 1) for j in range(i + 1, min(i + self.mccc_maxwin, self.nchan))],
            device=self.device,
        )
        npair = len(index_i)
        mccc_cc = torch.zeros(npair, device=self.device)
        mccc_dt = torch.zeros(npair, device=self.device)
        nchunk = int(np.ceil(npair / self.chunk_size))
        ib = 0
        for ichunk in range(nchunk):
            ie = min(ib + self.chunk_size, npair)
            ii = index_i[ib:ie]
            jj = index_j[ib:ie]
            xcor_freq = F.pad(
                self.data_freq[ii, :] * torch.conj(self.data_freq[jj, :]),
                (0, self.npts_xcor_pad // 2 - self.npts_xcor // 2),
                "constant",
                self.zero_complex,
            )
            xcor_time = torch.roll(
                torch.fft.irfft(xcor_freq, n=self.npts_xcor_pad, dim=-1),
                self.npts_xcor_pad // 2,
                dims=-1,
            )
            mccc_cc[ib:ie], _, mccc_dt[ib:ie] = self.pick_data_win_maxabs(
                xcor_time, self.resample_dt, self.mccc_maxlag, update_vmin=False
            )
            ib = ie
        mccc_cc *= self.resample_factor
        del xcor_freq
        del xcor_time
        return index_i, index_j, mccc_cc, mccc_dt

    def form_A_b_scipy(self, index_i, index_j, mccc_cc, mccc_dt):
        """
        A = [D; S; I]
        b = [dt; 0; tmax]
        ls problem: At = b
        """
        index_i = index_i.cpu()
        index_j = index_j.cpu()
        mccc_cc = mccc_cc.cpu()
        mccc_dt = mccc_dt.cpu()
        # G matrix, dt_ij = ti-tj = mccc_dt
        igood = torch.where(torch.abs(mccc_cc) > self.mccc_mincc)[0]
        ngood = len(igood)
        # cc[d[ti], d[tj], dt_ij=ti-tj
        mccc_cc = mccc_cc[igood]
        mccc_dt = mccc_dt[igood]
        # weight = torch.abs(value_cc).numpy()
        weight = np.ones(len(mccc_cc))
        index_ii = np.tile(np.arange(ngood), 2)
        index_jj = torch.concat([index_i[igood], index_j[igood]]).numpy()
        value_ij = np.concatenate([np.ones(ngood) * weight, -np.ones(ngood) * weight])
        # difference operator
        D = sparse.coo_matrix((value_ij, (index_ii, index_jj)), shape=(ngood, self.nchan))
        d = np.concatenate([mccc_dt.numpy() * weight, []])
        # smooth operator:  t[i+1] - t[i] = 0
        S = (np.diag(np.ones(self.nchan)) - np.diag(np.ones(self.nchan - 1), k=-1))[1:, :]
        S = sparse.csr_matrix(S) * self.damp
        # A * x = b
        A = sparse.vstack((D, S))
        b = np.concatenate([d, np.zeros(S.shape[0])])
        return A, b

    def concat_A_b_with_picking(self, A, b, pick_dt, pick_cc, w):
        """"""
        assert A.shape[1] == len(pick_dt)
        isel = torch.where(torch.abs(pick_cc) > torch.quantile(torch.abs(pick_cc), 0.15))[0]
        nsel = len(isel)
        P = sparse.coo_matrix(
            (np.ones(nsel), (np.arange(nsel), isel.cpu().numpy())),
            shape=(nsel, len(pick_dt)),
        )
        p = pick_dt[isel].cpu().numpy()
        return sparse.vstack([w * A, P]), np.concatenate([w * b, p])

    def solve(self):
        """
        Solve the least-square problem
        """
        # sparse matrix coords
        index_i, index_j, value_cc, value_dt = self.form_coo()
        # sparse matrix from mccc
        A0, b0 = self.form_A_b_scipy(index_i, index_j, value_cc, value_dt)
        # upsample the data in frequency domain
        data_resample, dt_resample = self.resample_data_in_freq_domain(self.data)
        # data_resample, dt_resample = self._resample_data_in_freq_domain(self.data_raw)
        niter = 0
        shift_index = torch.arange(self.nchan, dtype=int, device=self.device)
        w = self.w0
        win_main = self.win_main
        pick_dt = torch.zeros(self.nchan, device=self.device)
        while niter < self.max_niter:
            if niter == self.max_niter - 1:
                win_main = min([0.05, win_main])
            if niter == 0:
                pick_cc, _, pick_dt_refine = self.pick_data_win_maxabs(
                    # moving_average(data_resample, self.refine_ma),
                    data_resample,
                    dt_resample,
                    win_main,
                    update_vmin=False,
                )
            else:
                pick_cc, _, pick_dt_refine = self.pick_data_win_maxabs(
                    moving_average(gather_roll(data_resample, shift_index), self.refine_ma),
                    dt_resample,
                    win_main,
                    update_vmin=False,
                )
            # add picked value to sparse matrix
            if self.mode == "pick":
                A, b = self.concat_A_b_with_picking(A0, b0, pick_dt + pick_dt_refine, pick_cc, w)
            elif self.mode == "align":
                A, b = A0, b0
            # least-square
            solution = lsmr(A, b)
            pick_dt[:] = torch.tensor(solution[0], device=self.device)
            shift_index[:] = -torch.round(pick_dt / dt_resample).int()
            w /= 1.1
            win_main /= 2
            niter += 1
        # solution
        sol = {"cc_dt": pick_dt, "cc_main": pick_cc, "cc_mean": torch.mean(torch.abs(pick_cc))}
        if self.mode == "pick":
            pick_side = self.pick_data_win_maxabs_side_lobe(data_resample, dt_resample, self.win_side)
            sol["cc_side"] = pick_side
        if self.return_data_align:
            sol["data_align"] = moving_average(gather_roll(data_resample, shift_index), 40)
            nt_resample = data_resample.shape[-1]
            time_axis = (np.arange(nt_resample) - nt_resample // 2) * dt_resample
            sol["data_align_info"] = {
                "time_axis": time_axis,
                "nx": self.nchan,
                "dt": dt_resample,
            }
        return sol

    def form_A_b_torch(self, index_i, index_j, value_cc, value_dt):
        """"""
        # with torch:
        igood = torch.where(torch.abs(value_cc) > self.mccc_mincc)[0]
        value_cc = value_cc[igood]
        value_dt = -value_dt[igood]
        ngood = len(igood)
        weight = torch.ones(ngood, device=self.device)
        index_ii = torch.tile(torch.arange(ngood, device=self.device), (2,))
        index_jj = torch.cat([index_i[igood], index_j[igood]])
        index_ij = torch.vstack((index_ii, index_jj))
        value_ij = torch.cat(
            [
                torch.ones(ngood, device=self.device) * weight,
                -torch.ones(ngood, device=self.device) * weight,
            ]
        )
        # difference operator
        D = torch.sparse_coo_tensor(index_ij, value_ij, size=(ngood, self.nchan))
        d = value_dt * weight
        # smooth operator
        S = (
            torch.diag(torch.ones(self.nchan, device=self.device))
            - torch.diag(torch.ones(self.nchan - 1, device=self.device), diagonal=-1)
        )[1:, :].to_sparse()
        s = torch.zeros(S.shape[0], device=self.device)
        return D, d.unsqueeze(1), S, s.unsqueeze(1)

    def form_fun_jac(self, D, d, S, s, P, p):
        """"""
        from torch import sparse as t_sparse

        def fun(t):
            """"""
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, device=self.device, dtype=torch.float32).unsqueeze(1)
            r = ((t_sparse.mm(D, t) - d) ** 2).sum().sqrt()
            r += ((t_sparse.mm(S, t) - s) ** 2).sum().sqrt()
            r += ((t_sparse.mm(P, t) - p) ** 2).sum().sqrt()
            r *= 0.5
            return float(r.numpy())

        def jac(t):
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, device=self.device, dtype=torch.float32).unsqueeze(1)
            # print(f"jac: {t=}, {t.shape=}")
            g = (
                t_sparse.mm(D.transpose(0, 1), (t_sparse.mm(D, t) - d))
                + t_sparse.mm(S.transpose(0, 1), (t_sparse.mm(S, t) - s))
                + t_sparse.mm(P.transpose(0, 1), (t_sparse.mm(P, t) - p))
            )
            return g.numpy().flatten()

        return fun, jac

    def solve_torch(self):
        """"""
        from scipy.optimize import minimize

        # sparse matrix coords
        index_i, index_j, value_cc, value_dt = self.form_coo()
        D, d, S, s = self.form_A_b_torch(index_i, index_j, value_cc, value_dt)
        # upsample the data in frequency domain
        data_resample, dt_resample = self.resample_data_in_freq_domain(self.data)
        niter = 0
        shift_index = torch.arange(self.nchan, dtype=int)
        w = self.w0
        win_main = self.win_main
        pick_dt = torch.zeros(self.nchan, device=self.device)
        while niter < self.max_niter:
            if niter == self.max_niter - 1:
                win_main = min([0.05, win_main])
            if niter == 0:
                pick_cc, _, pick_dt_refine = self.pick_data_win_maxabs(
                    data_resample, dt_resample, win_main, update_vmin=False
                )
            else:
                pick_cc, _, pick_dt_refine = self.pick_data_win_maxabs(
                    moving_average(gather_roll(data_resample, shift_index), 40),
                    dt_resample,
                    win_main,
                    update_vmin=False,
                )
            # pick
            isel = torch.where(torch.abs(pick_cc) > torch.quantile(torch.abs(pick_cc), 0.15))[0]
            nsel = len(isel)
            P = torch.sparse_coo_tensor(
                torch.vstack([torch.arange(nsel, device=self.device), isel]),
                torch.ones(nsel, device=self.device),
                size=(nsel, len(pick_dt_refine)),
            )
            p = (pick_dt + pick_dt_refine)[isel].unsqueeze(1)
            fun, jac = self.form_fun_jac(D, d, S, s, P, p)
            # info = {'D': D, 'S': S, 'P': P, 'd': d, 's': s, 'p':p}
            res = minimize(fun, pick_dt.unsqueeze(1), jac=jac, method="CG")
            pick_dt[:] = torch.tensor(res.x)
            shift_index = -torch.round(pick_dt / dt_resample).int()
            w /= 1.1
            win_main /= 2
            niter += 1
        pick_side = self.pick_data_win_maxabs_side_lobe(data_resample, dt_resample, self.win_side)
        sol = {
            "cc_dt": pick_dt,
            "cc_main": pick_cc,
            "cc_side": pick_side,
            "data_shift": gather_roll(data_resample, shift_index),
        }
        return sol
