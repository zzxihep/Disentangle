import numpy as np
import spectool
import matplotlib.pyplot as plt
from tqdm import tqdm


class Spectrum:
    def __init__(self, wave, flux, ivar, obstime):
        self.wavelength = wave
        self.flux = flux
        self.ivar = ivar
        self.obstime = obstime
        self.RVs = []
        self.original_flux = None


class StarSystem:
    def __init__(self):
        self._T0 = None
        self._period = None
        self._RV_component = []
        self._specs = []

    def add_spec(self, spec: Spectrum):
        self._specs.append(spec)

    def set_T0(self, T0: float):
        """set the phase zero point

        Args:
            T0 (float): the phase zero point
        """
        self._T0 = T0

    def set_period(self, period: float):
        """set the period

        Args:
            period (float): the period
        """
        self._period = period
    
    def add_rv_component(self, RV_type: str, RV_params: list):
        """add a RV component to the system
        we only allow two RV types now: 'sine' or 'const'

        For RV_type = 'sine', RV_params = [K],
        where K is the semi-amplitude
        the RV is given by K * sin(2 * pi * phase)
        The phase is defined as phase = (t - T0) % period

        For RV_type = 'const', RV_params = [RV],
        where RV is the constant RV

        Args:
            RV_type (str): the type of the RV component
            RV_params (list): the parameters of the RV component
        """
        if RV_type not in ['sine', 'const']:
            raise ValueError('RV_type must be sine or const')
        if RV_type == 'sine':
            if len(RV_params) != 1:
                raise ValueError('RV_params must be [K]')
        if RV_type == 'const':
            if len(RV_params) != 1:
                raise ValueError('RV_params must be [RV]')
        self._RV_component.append([RV_type, RV_params])

    def set_rv_component_par(self, ind_rv, RV_params):
        """set the parameters of the RV component

        Args:
            ind_rv (int): the index of the RV component
            RV_params (list): the parameters of the RV component
        """
        self._RV_component[ind_rv][1] = RV_params

    def get_rv_component_par(self, ind_rv):
        """get the parameters of the RV component

        Args:
            ind_rv (int): the index of the RV component

        Returns:
            list: the parameters of the RV component
        """
        return self._RV_component[ind_rv][1]

    def get_rv_component_type(self, ind_rv):
        """get the type of the RV component

        Args:
            ind_rv (int): the index of the RV component

        Returns:
            str: the type of the RV component
        """
        return self._RV_component[ind_rv][0]

    def flush_rv(self):
        """flush the RVs of the spectra
        """
        for spec in self._specs:
            rvs = []
            for rv_compoent in self._RV_component:
                rv_type, rv_params = rv_compoent
                if rv_type == 'sine':
                    K = rv_params[0]
                    phase = ((spec.obstime - self._T0) % self._period) / self._period
                    spec.phase = phase
                    rv = K * np.sin(2 * np.pi * phase)
                    rvs.append(rv)
                elif rv_type == 'const':
                    rv = rv_params[0]
                    rvs.append(rv)
            spec.RVs = rvs

    def plot_ccf_rv_curve(self):
        """plot the RV curve from the CCF
        """
        specs = self._specs
        spec_ref = specs[0]
        # times = []
        rvs = []
        phases = []
        for ind, spec in enumerate(specs):
            rv = spectool.ccf.find_radial_velocity2(spec.wavelength, spec.original_flux, spec_ref.wavelength, spec_ref.original_flux, plot=False)
            # times.append(spec.obstime)
            rvs.append(rv)
            phases.append(spec.phase)
            # print('fitname = %s, rv = %f' % (spec.fn, rv))
        # times = np.array(times)
        # times = times - self._T0
        rvs = np.array(rvs)
        phases = np.array(phases)
        arg = np.argsort(phases)
        phases = phases[arg]
        rvs = rvs[arg]
        # arg = np.argsort(times)
        # times = times[arg]
        # rvs = rvs[arg]
        plt.errorbar(phases, rvs, fmt='o')
        plt.show()


class Disentangle(StarSystem):
    def __init__(self):
        super().__init__()
        self._single_components = []

    def add_wave_ref(self, wave_ref: np.ndarray):
        """add the reference wavelength

        Args:
            wave_ref (np.ndarray): the reference wavelength
        """
        self._wave_ref = wave_ref
        self._single_components = []
        print('Number of RV components = ', len(self._RV_component))
        for ind in range(len(self._RV_component)):
            self._single_components.append(np.zeros(len(wave_ref)))

    def get_single_component(self, ind_comp: int):
        """get the single component by remove all the other components and combined the residual spectra"""
        specs = []
        ivars = []
        for spec in self._specs:
            wave = spec.wavelength
            flux = spec.flux
            ivar = spec.ivar
            for ind_rv, rv in enumerate(spec.RVs):
                if ind_rv != ind_comp:
                    wave_ref_shift = spectool.spec_func.shift_wave(self._wave_ref, rv)
                    single_comp_ind = self._single_components[ind_rv]
                    single_comp_shift = spectool.rebin.rebin_padvalue(wave_ref_shift, single_comp_ind, wave)
                    flux = flux - single_comp_shift
            rv_comp = spec.RVs[ind_comp]
            nwave = spectool.spec_func.shift_wave(wave, -rv_comp)
            flux_rest = spectool.rebin.rebin_padvalue(nwave, flux, self._wave_ref)
            specs.append(flux_rest)
            nivars = spectool.rebin.rebin_padvalue(nwave, ivar, self._wave_ref)
            ivars.append(nivars)
        specs = np.array(specs)
        ivars = np.array(ivars)
        # combined_spec = np.sum(specs, axis=0) / len(specs)
        combined_spec = np.sum(specs*ivars, axis=0) / np.sum(ivars, axis=0)
        self._single_components[ind_comp] = combined_spec

    def plot_disentangled_result(self, ind_spec: int, correct_profile=False, combined_componet=None):
        spec = self._specs[ind_spec]
        wave = spec.wavelength
        flux = spec.flux
        ivar = spec.ivar
        plot_shift = 5 * np.std(flux)
        fig = plt.figure(figsize=(14, 9))
        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.2])
        ax2 = fig.add_axes([0.1, 0.31, 0.85, 0.65], sharex=ax1)
        ax2.plot(wave, flux, label='original')
        flux_construct = np.zeros(len(wave))
        if combined_componet is not None:
            flux_combined = np.zeros(len(wave))
        if correct_profile:
            invert_scale = np.zeros(len(self._wave_ref))
        for ind_rv, rv in enumerate(spec.RVs):
            wave_ref_shift = spectool.spec_func.shift_wave(self._wave_ref, rv)
            single_comp = self._single_components[ind_rv]
            scale_tmp = np.zeros(len(self._wave_ref))
            if correct_profile:
                if ind_rv < len(spec.RVs) - 1:
                    pars = spectool.spec_func.fit_profile_par(self._wave_ref, single_comp, degree=15)
                    scale = spectool.spec_func.get_profile(self._wave_ref, pars)
                    invert_scale -= scale
                    scale_tmp = scale
                else:
                    scale_tmp = invert_scale
            plt.plot(wave_ref_shift, single_comp - scale_tmp - plot_shift * (ind_rv+1), label='single component {}, RV = {:.2f}'.format(ind_rv, rv))
            single_comp_rebin = spectool.rebin.rebin_padvalue(wave_ref_shift, single_comp, wave)
            flux_construct += single_comp_rebin
            if combined_componet is not None and ind_rv in combined_componet:
                flux_combined += single_comp_rebin
        ax2.plot(wave, flux_construct, label='constructed')
        residual = flux - flux_construct
        ax1.plot(wave, residual, label='residual')
        if combined_componet is not None:
            text = 'combinded: '
            for ind in combined_componet:
                text += '%d, ' % ind
            ax2.plot(wave, flux_combined + plot_shift, label=text)
        ax1.legend()
        ax2.legend()
        plt.show()

    def get_chisq(self):
        chisq = 0
        for spec in self._specs:
            wave = spec.wavelength
            flux = spec.flux
            ivar = spec.ivar
            flux_construct = np.zeros(len(wave))
            for ind_rv, rv in enumerate(spec.RVs):
                wave_ref_shift = spectool.spec_func.shift_wave(self._wave_ref, rv)
                single_comp = self._single_components[ind_rv]
                single_comp_rebin = spectool.rebin.rebin_padvalue(wave_ref_shift, single_comp, wave)
                flux_construct += single_comp_rebin
            chisq += np.sum(ivar * (flux - flux_construct) ** 2)
        return chisq

    def run_distangle(self, steps=600):
        self.step_distangles = []
        for ind in tqdm(range(steps)):
            for ind in range(len(self._RV_component)):
                self.get_single_component(ind)
            # self.get_single_component(0)
            # self.get_single_component(1)
            # chisq = self.get_chisq()
            # self.step_distangles.append([chisq, [val.copy() for val in self._single_components]])
        # self._history_distangles = [val[1] for val in self.step_distangles]
        # self.chisqs = np.array([val[0] for val in self.step_distangles])

