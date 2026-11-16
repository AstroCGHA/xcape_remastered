!-----------------------------------------------------------------------
!  xcaper_kernels - single-column numerical kernels shared by every
!                  xcaper Fortran extension module.
!
!  In xcape 0.1 each of these routines existed twice (one copy for
!  model levels, one for pressure levels).  They are now kept here once
!  and the thin, f2py-wrapped drivers (CAPE_CODE_*.f90, Bunkers_*.f90,
!  SREH_*.f90, stdheight_2D_*.f90) loop over the grid and call them.
!
!  Everything in this module is thread safe: no SAVEd state, no I/O and
!  no STOP statements, so the drivers can release the Python GIL and be
!  run concurrently from dask worker threads.
!
!  Authors of the original routines:
!     getcape            George H. Bryan (NCAR/MMM), see notice below
!     stdheight,
!     bunkers, srh       Chiara Lepore @chiaral, John Allen @xebadir
!
!  Disclaimer:  This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
module xcaper_kernels

    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64

    implicit none
    private

    public :: sp, dp
    public :: getcape, qvs_liquid, qvs_ice, theta_e
    public :: stdheight
    public :: bunkers_calc, interp_to_height
    public :: calc_srh, linear_interp

    ! value used by the legacy code to flag "not available"
    real(sp), parameter, public :: missing_sp = -999999.0_sp
    real(dp), parameter, public :: missing_dp = -999999.0_dp

contains

!-----------------------------------------------------------------------
!  getcape - Convective Available Potential Energy (CAPE) and
!            Convective Inhibition (CIN) from a sounding.
!
!  Version 1.02                           Last modified:  10 October 2008
!
!  Author:  George H. Bryan
!           Mesoscale and Microscale Meteorology Division
!           National Center for Atmospheric Research
!           Boulder, Colorado, USA
!           gbryan@ucar.edu
!
!  References:  Bolton (1980, MWR, p. 1046) (constants and definitions)
!               Bryan and Fritsch (2004, MWR, p. 2421) (ice processes)
!
!  FROM ORIGINAL getcape.F available at
!  https://www2.mmm.ucar.edu/people/bryan/Code/getcape.F
!  (c) 2019 - University Corporation for Atmospheric Research
!
!  Permission is hereby granted, free of charge, to any person obtaining a copy
!  of this software and associated documentation files (the "Software"), to deal
!  in the Software without restriction, including without limitation the rights
!  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!  copies of the Software, and to permit persons to whom the Software is furnished
!  to do so, subject to the following conditions:
!
!  The above copyright notice and this permission notice shall be included in all
!  copies or substantial portions of the Software.
!
!  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
!  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
!  THE SOFTWARE.
!-----------------------------------------------------------------------
!  Input:   nk_in    - number of levels above the surface level
!           p_in_a   - pressure (hPa) of those levels, bottom to top
!           t_in_a   - temperature (C)
!           td_in_a  - dewpoint temperature (C)
!           ps_in, ts_in, tds_in - surface pressure (hPa), temperature (C)
!                      and dewpoint (C); used as the lowest level
!           pinc     - pressure increment (Pa) of the parcel ascent
!           source   - 1 surface, 2 most unstable (max theta-e below
!                      500 hPa), 3 mixed layer (depth ml_depth)
!           ml_depth - depth (m) of the mixed layer for source = 3
!           adiabat  - 1 pseudoadiabatic liquid, 2 reversible liquid,
!                      3 pseudoadiabatic ice,    4 reversible ice
!
!  Output:  cape  - Convective Available Potential Energy (J/kg)
!           cin   - Convective Inhibition (J/kg, positive number)
!           mulvl - 1-based index (surface = 1) of the most unstable
!                   level; -999999 unless source = 2
!           z_out - height (m above the surface) of the last level
!                   reached by the parcel ascent (legacy behaviour)
!
!  Changes with respect to the xcape 0.1 copies:
!    * unknown source / adiabat and lack of convergence return zeros
!      instead of calling STOP / PRINT (which killed the interpreter or
!      garbled output when called from threads);
!    * kmax is initialised, so profiles containing NaN can no longer
!      index out of bounds in the most-unstable search;
!    * the mixed-layer integration cannot read past the last level.
!  The arithmetic is otherwise unchanged (single precision).
!-----------------------------------------------------------------------
    subroutine getcape(nk_in, p_in_a, t_in_a, td_in_a, ps_in, ts_in, tds_in, &
                       pinc, source, ml_depth, adiabat, cape, cin, mulvl, z_out)

        integer, intent(in) :: nk_in
        real(sp), dimension(nk_in), intent(in) :: p_in_a, t_in_a, td_in_a
        real(sp), intent(in) :: ps_in, ts_in, tds_in
        real(sp), intent(in) :: pinc
        integer, intent(in) :: source
        real(sp), intent(in) :: ml_depth
        integer, intent(in) :: adiabat
        real(sp), intent(out) :: cape, cin, z_out
        integer, intent(out) :: mulvl

        logical :: doit, ice, cloud, not_converged
        integer :: k, kmax, n, nloop, i, nk

        real(sp), dimension(nk_in + 1) :: p, t, td, pi, q, z, th, thv

        real(sp) :: the, maxthe, parea, narea
        real(sp) :: th1, p1, t1, qv1, ql1, qi1, b1, pi1, thv1, qt, dp_, dz, frac
        real(sp) :: th2, p2, t2, qv2, ql2, qi2, b2, pi2, thv2
        real(sp) :: thlast, fliq, fice, tbar, qvbar, qlbar, qibar, lhv, lhs, rm, cpm
        real(dp) :: avgth, avgqv

        real(sp), parameter :: g     = 9.81_sp
        real(sp), parameter :: p00   = 100000.0_sp
        real(sp), parameter :: cp    = 1005.7_sp
        real(sp), parameter :: rd    = 287.04_sp
        real(sp), parameter :: rv    = 461.5_sp
        real(sp), parameter :: xlv   = 2501000.0_sp
        real(sp), parameter :: xls   = 2836017.0_sp
        real(sp), parameter :: t0    = 273.15_sp
        real(sp), parameter :: cpv   = 1875.0_sp
        real(sp), parameter :: cpl   = 4190.0_sp
        real(sp), parameter :: cpi   = 2118.636_sp
        real(sp), parameter :: lv1   = xlv + (cpl - cpv)*t0
        real(sp), parameter :: lv2   = cpl - cpv
        real(sp), parameter :: ls1   = xls + (cpi - cpv)*t0
        real(sp), parameter :: ls2   = cpi - cpv

        real(sp), parameter :: rp00  = 1.0_sp/p00
        real(sp), parameter :: reps  = rv/rd
        real(sp), parameter :: rddcp = rd/cp
        real(sp), parameter :: cpdg  = cp/g

        real(sp), parameter :: converge = 0.0002_sp

        cape  = 0.0_sp
        cin   = 0.0_sp
        mulvl = -999999
        z_out = missing_sp

        if (source < 1 .or. source > 3) return
        if (adiabat < 1 .or. adiabat > 4) return

        ! number of levels of the 3d array PLUS the surface level
        nk = nk_in + 1

        !---- convert p,t,td to mks units; get pi,q,th,thv ----!
        p(1)  = 100.0_sp*ps_in
        t(1)  = 273.15_sp + ts_in
        td(1) = 273.15_sp + tds_in
        do k = 2, nk
            p(k)  = 100.0_sp*p_in_a(k - 1)
            t(k)  = 273.15_sp + t_in_a(k - 1)
            td(k) = 273.15_sp + td_in_a(k - 1)
        end do
        do k = 1, nk
            pi(k)  = (p(k)*rp00)**rddcp
            q(k)   = qvs_liquid(p(k), td(k))
            th(k)  = t(k)/pi(k)
            thv(k) = th(k)*(1.0_sp + reps*q(k))/(1.0_sp + q(k))
        end do

        !---- get height using the hydrostatic equation ----!
        z(1) = 0.0_sp
        do k = 2, nk
            dz = -cpdg*0.5_sp*(thv(k) + thv(k - 1))*(pi(k) - pi(k - 1))
            z(k) = z(k - 1) + dz
        end do

        !---- find source parcel ----!
        kmax  = 1
        avgth = 0.0_dp
        avgqv = 0.0_dp

        if (source == 1) then
            ! use surface parcel
            kmax = 1

        else if (source == 2) then
            ! use most unstable parcel (max theta-e)
            if (p(1) < 50000.0_sp) then
                ! first report is above 500 mb ... just use the first level
                kmax = 1
            else
                ! find max theta-e below 500 mb
                maxthe = 0.0_sp
                do k = 1, nk
                    if (p(k) >= 50000.0_sp) then
                        the = theta_e(p(k), t(k), td(k), q(k))
                        if (the > maxthe) then
                            mulvl  = k
                            maxthe = the
                            kmax   = k
                        end if
                    end if
                end do
            end if

        else
            ! use mixed layer
            if ((z(2) - z(1)) > ml_depth) then
                ! the second level is above the mixed-layer depth:
                ! just use the lowest level
                avgth = th(1)
                avgqv = q(1)
                kmax  = 1
            else if (z(nk) < ml_depth) then
                ! the top-most level is within the mixed layer:
                ! just use the upper-most level
                avgth = th(nk)
                avgqv = q(nk)
                kmax  = nk
            else
                ! calculate the mixed-layer properties
                k = 2
                do while (k <= nk)
                    if (z(k) > ml_depth) exit
                    avgth = avgth + 0.5_sp*(z(k) - z(k - 1))*(th(k) + th(k - 1))
                    avgqv = avgqv + 0.5_sp*(z(k) - z(k - 1))*(q(k) + q(k - 1))
                    k = k + 1
                end do

                if (k <= nk) then
                    th2 = th(k - 1) + (th(k) - th(k - 1))*(ml_depth - z(k - 1))/(z(k) - z(k - 1))
                    qv2 = q(k - 1) + (q(k) - q(k - 1))*(ml_depth - z(k - 1))/(z(k) - z(k - 1))
                    avgth = avgth + 0.5_sp*(ml_depth - z(k - 1))*(th2 + th(k - 1))
                    avgqv = avgqv + 0.5_sp*(ml_depth - z(k - 1))*(qv2 + q(k - 1))
                end if

                avgth = avgth/ml_depth
                avgqv = avgqv/ml_depth
                kmax  = 1
            end if
        end if

        !---- define parcel properties at initial location ----!
        narea = 0.0_sp

        k = kmax
        if (source == 3) then
            th2  = real(avgth, sp)
            qv2  = real(avgqv, sp)
            thv2 = th2*(1.0_sp + reps*qv2)/(1.0_sp + qv2)
            pi2  = pi(kmax)
            p2   = p(kmax)
            t2   = th2*pi2
            b2   = g*(thv2 - thv(kmax))/thv(kmax)
        else
            th2  = th(kmax)
            pi2  = pi(kmax)
            p2   = p(kmax)
            t2   = t(kmax)
            thv2 = thv(kmax)
            qv2  = q(kmax)
            b2   = 0.0_sp
        end if

        ql2 = 0.0_sp
        qi2 = 0.0_sp
        qt  = qv2

        doit  = .true.
        cloud = .false.
        ice   = (adiabat == 3 .or. adiabat == 4)

        !---- begin ascent of parcel ----!
        do while (doit .and. (k < nk))

            k  = k + 1
            b1 = b2

            dp_ = p(k - 1) - p(k)

            if (dp_ < pinc) then
                nloop = 1
            else
                nloop = 1 + int(dp_/pinc)
                dp_ = dp_/real(nloop, sp)
            end if

            do n = 1, nloop

                p1   = p2
                t1   = t2
                pi1  = pi2
                th1  = th2
                qv1  = qv2
                ql1  = ql2
                qi1  = qi2
                thv1 = thv2

                p2  = p2 - dp_
                pi2 = (p2*rp00)**rddcp

                thlast = th1
                i = 0
                not_converged = .true.

                do while (not_converged)
                    i  = i + 1
                    t2 = thlast*pi2
                    if (ice) then
                        fliq = max(min((t2 - 233.15_sp)/(273.15_sp - 233.15_sp), 1.0_sp), 0.0_sp)
                        fice = 1.0_sp - fliq
                    else
                        fliq = 1.0_sp
                        fice = 0.0_sp
                    end if
                    qv2 = min(qt, fliq*qvs_liquid(p2, t2) + fice*qvs_ice(p2, t2))
                    qi2 = max(fice*(qt - qv2), 0.0_sp)
                    ql2 = max(qt - qv2 - qi2, 0.0_sp)

                    tbar  = 0.5_sp*(t1 + t2)
                    qvbar = 0.5_sp*(qv1 + qv2)
                    qlbar = 0.5_sp*(ql1 + ql2)
                    qibar = 0.5_sp*(qi1 + qi2)

                    lhv = lv1 - lv2*tbar
                    lhs = ls1 - ls2*tbar

                    rm  = rd + rv*qvbar
                    cpm = cp + cpv*qvbar + cpl*qlbar + cpi*qibar
                    th2 = th1*exp(lhv*(ql2 - ql1)/(cpm*tbar)   &
                                  + lhs*(qi2 - qi1)/(cpm*tbar) &
                                  + (rm/cpm - rd/cp)*log(p2/p1))

                    if (i > 100) then
                        ! lack of convergence: give up on this column
                        cape = 0.0_sp
                        cin  = 0.0_sp
                        return
                    end if
                    if (abs(th2 - thlast) > converge) then
                        thlast = thlast + 0.3_sp*(th2 - thlast)
                    else
                        not_converged = .false.
                    end if
                end do

                ! Latest pressure increment is complete.
                if (ql2 >= 1.0e-10_sp) cloud = .true.

                if (adiabat == 1 .or. adiabat == 3) then
                    ! pseudoadiabat
                    qt  = qv2
                    ql2 = 0.0_sp
                    qi2 = 0.0_sp
                end if

            end do

            thv2 = th2*(1.0_sp + reps*qv2)/(1.0_sp + qv2 + ql2 + qi2)
            b2   = g*(thv2 - thv(k))/thv(k)
            dz   = -cpdg*0.5_sp*(thv(k) + thv(k - 1))*(pi(k) - pi(k - 1))

            ! Get contributions to CAPE and CIN:
            if ((b2 >= 0.0_sp) .and. (b1 < 0.0_sp)) then
                ! first trip into positive area
                frac  = b2/(b2 - b1)
                parea = 0.5_sp*b2*dz*frac
                narea = narea - 0.5_sp*b1*dz*(1.0_sp - frac)
                cin   = cin + narea
                narea = 0.0_sp
            else if ((b2 < 0.0_sp) .and. (b1 > 0.0_sp)) then
                ! first trip into neg area
                frac  = b1/(b1 - b2)
                parea = 0.5_sp*b1*dz*frac
                narea = -0.5_sp*b2*dz*(1.0_sp - frac)
            else if (b2 < 0.0_sp) then
                ! still collecting negative buoyancy
                parea = 0.0_sp
                narea = narea - 0.5_sp*dz*(b1 + b2)
            else
                ! still collecting positive buoyancy
                parea = 0.5_sp*dz*(b1 + b2)
                narea = 0.0_sp
            end if

            cape = cape + max(0.0_sp, parea)

            if ((p(k) <= 10000.0_sp) .and. (b2 < 0.0_sp)) then
                ! stop if b < 0 and p < 100 mb
                doit = .false.
            end if
            z_out = z(k)
        end do

    end subroutine getcape

!-----------------------------------------------------------------------
!  Saturation mixing ratio (kg/kg) over liquid water.  p in Pa, t in K.
!-----------------------------------------------------------------------
    pure elemental function qvs_liquid(p, t) result(qvs)
        real(sp), intent(in) :: p, t
        real(sp) :: qvs
        real(sp) :: es
        real(sp), parameter :: eps = 287.04_sp/461.5_sp

        es  = 611.2_sp*exp(17.67_sp*(t - 273.15_sp)/(t - 29.65_sp))
        qvs = eps*es/(p - es)
    end function qvs_liquid

!-----------------------------------------------------------------------
!  Saturation mixing ratio (kg/kg) over ice.  p in Pa, t in K.
!-----------------------------------------------------------------------
    pure elemental function qvs_ice(p, t) result(qvi)
        real(sp), intent(in) :: p, t
        real(sp) :: qvi
        real(sp) :: es
        real(sp), parameter :: eps = 287.04_sp/461.5_sp

        es  = 611.2_sp*exp(21.8745584_sp*(t - 273.15_sp)/(t - 7.66_sp))
        qvi = eps*es/(p - es)
    end function qvs_ice

!-----------------------------------------------------------------------
!  Equivalent potential temperature (K), Bolton (1980).
!  p in Pa, t and td in K, q in kg/kg.
!-----------------------------------------------------------------------
    pure elemental function theta_e(p, t, td, q) result(the)
        real(sp), intent(in) :: p, t, td, q
        real(sp) :: the
        real(sp) :: tlcl

        if ((td - t) >= -0.1_sp) then
            tlcl = t
        else
            tlcl = 56.0_sp + ((td - 56.0_sp)**(-1) + 0.00125_sp*log(t/td))**(-1)
        end if

        the = t*((100000.0_sp/p)**(0.2854_sp*(1.0_sp - 0.28_sp*q))) &
              *exp(((3376.0_sp/tlcl) - 2.54_sp)*q*(1.0_sp + 0.81_sp*q))
    end function theta_e

!-----------------------------------------------------------------------
!  stdheight - height above ground of every level of a sounding from
!              the hypsometric equation, assuming layer-mean virtual
!              temperature.
!
!  Author(s): Chiara Lepore @chiaral, John Allen @xebadir
!
!  References: Bolton (1980, MWR); Holton and Hakim (2013).
!
!  Input:  nk  - number of levels above the surface level
!          p   - pressure (hPa)
!          t   - temperature (C)
!          td  - dewpoint temperature (C)
!          ps, ts, tds - surface pressure (hPa), temperature (C) and
!                dewpoint (C)
!          hin - height above ground (m) of the surface level
!
!  Output: h   - height above ground (m) of every level
!          hs  - height above ground (m) of the surface level (= hin)
!-----------------------------------------------------------------------
    pure subroutine stdheight(nk, p, t, td, ps, ts, tds, hin, h, hs)

        integer, intent(in) :: nk
        real(dp), dimension(nk), intent(in) :: p, t, td
        real(dp), intent(in) :: ps, ts, tds
        real(dp), intent(in) :: hin
        real(dp), dimension(nk), intent(out) :: h
        real(dp), intent(out) :: hs

        integer :: k
        real(dp), dimension(nk) :: tv
        real(dp) :: tvs

        ! xcape 0.1 wrote the constants of this routine as default-real
        ! literals.  Their single-precision values are kept on purpose so
        ! that heights (and therefore SRH) are reproduced bit for bit.
        real(dp), parameter :: r = real(287.04_sp, dp)
        real(dp), parameter :: g = real(-9.80665_sp, dp)

        tvs = virtual_temperature(ts, tds, ps)
        do k = 1, nk
            tv(k) = virtual_temperature(t(k), td(k), p(k))
        end do

        hs = hin
        if (nk < 1) return

        h(1) = hs + ((r*((tv(1) + tvs)/2)/g))*(log(p(1)/ps))
        do k = 2, nk
            h(k) = h(k - 1) + ((r*((tv(k) + tv(k - 1))/2)/g))*(log(p(k)/p(k - 1)))
        end do

    end subroutine stdheight

!-----------------------------------------------------------------------
!  Virtual temperature (K) from temperature (C), dewpoint (C) and
!  pressure (hPa).
!-----------------------------------------------------------------------
    pure function virtual_temperature(t_c, td_c, p_hpa) result(tv)
        real(dp), intent(in) :: t_c, td_c, p_hpa
        real(dp) :: tv
        real(dp) :: tin, tdin, e, w
        ! single-precision literals kept on purpose, see stdheight
        real(dp), parameter :: eps  = real(0.6219800858985514_sp, dp)
        real(dp), parameter :: t0   = real(273.15_sp, dp)
        real(dp), parameter :: e0   = real(6.112_sp, dp)
        real(dp), parameter :: c1   = real(53.49_sp, dp)
        real(dp), parameter :: c3   = real(5.09_sp, dp)

        tin  = t_c + t0
        tdin = td_c + t0
        e  = e0*exp((c1 - (6808/tdin) - c3*log(tdin)))
        w  = eps*(e/(p_hpa - e))
        tv = tin*((w + eps)/(eps*(1 + w)))
    end function virtual_temperature

!-----------------------------------------------------------------------
!  bunkers_calc - Bunkers et al. (2000) internal-dynamics storm motion
!                 for one wind profile.
!
!  Author:  Chiara Lepore @chiaral - John Allen @xebadir
!
!  The profile is sampled at 13 fixed heights (lowest level, then every
!  500 m from 500 m to 6 km above ground).  The 0-6 km mean wind is the
!  plain (non pressure weighted) average of the 13 samples; the shear
!  vector joins the mean of the two lowest samples to the mean of the
!  two highest ones; right/left movers deviate 7.5 m/s from the mean
!  wind, perpendicular to the shear.
!
!  Input:   nk   - number of levels (lowest level first)
!           u, v - wind components (m/s)
!           aglh - height above ground (m)
!
!  Output:  rm, lm      - right / left mover motion (u, v) in m/s
!           mean6kmwind - 0-6 km mean wind (u, v) in m/s
!
!  Fixed in xcaper 1.0: the accumulators (mean wind and the four shear sums)
!  were never initialised, so results depended on whatever happened to
!  be in memory.  They now start from zero.
!-----------------------------------------------------------------------
    pure subroutine bunkers_calc(nk, u, v, aglh, rm, lm, mean6kmwind)

        integer, intent(in) :: nk
        real(sp), dimension(nk), intent(in) :: u, v, aglh
        real(sp), dimension(2), intent(out) :: rm, lm, mean6kmwind

        integer :: i
        real(sp) :: ushr, vshr, ushr_u, vshr_u, ushr_d, vshr_d, shr
        real(sp), dimension(13) :: u_surf6km, v_surf6km
        real(sp), parameter :: deviation = 7.5_sp

        ! Interpolate to the 500 m - 6000 m layers
        u_surf6km(1) = u(1)
        v_surf6km(1) = v(1)
        do i = 2, 13
            u_surf6km(i) = interp_to_height(nk, u, aglh, 500.0_sp*real(i - 1, sp))
            v_surf6km(i) = interp_to_height(nk, v, aglh, 500.0_sp*real(i - 1, sp))
        end do

        ! Mean wind 0-6 km - non-pressure weighted
        mean6kmwind(1) = 0.0_sp
        mean6kmwind(2) = 0.0_sp
        do i = 1, 13
            mean6kmwind(1) = mean6kmwind(1) + u_surf6km(i)
            mean6kmwind(2) = mean6kmwind(2) + v_surf6km(i)
        end do
        mean6kmwind(1) = mean6kmwind(1)/13
        mean6kmwind(2) = mean6kmwind(2)/13

        ! Shear between the 0-500 m mean and the 5500-6000 m mean
        ushr_u = (u_surf6km(12) + u_surf6km(13))/2
        vshr_u = (v_surf6km(12) + v_surf6km(13))/2
        ushr_d = (u_surf6km(1) + u_surf6km(2))/2
        vshr_d = (v_surf6km(1) + v_surf6km(2))/2

        ushr = ushr_u - ushr_d
        vshr = vshr_u - vshr_d
        shr  = (ushr**2 + vshr**2)**0.5_sp

        ! Left and right mover components, Bunkers et al. (2000)
        rm(1) = mean6kmwind(1) + deviation*vshr/shr
        rm(2) = mean6kmwind(2) - deviation*ushr/shr
        lm(1) = mean6kmwind(1) - deviation*vshr/shr
        lm(2) = mean6kmwind(2) + deviation*ushr/shr

    end subroutine bunkers_calc

!-----------------------------------------------------------------------
!  interp_to_height - linear interpolation of a profile to one height.
!  Works for coordinates that increase or decrease with index.  Returns
!  -999999 when the requested height is not bracketed by the profile.
!  (Single-column equivalent of the legacy DINTERP2DZ.)
!-----------------------------------------------------------------------
    pure function interp_to_height(nz, v, z, height) result(vout)

        integer, intent(in) :: nz
        real(sp), dimension(nz), intent(in) :: v, z
        real(sp), intent(in) :: height
        real(sp) :: vout

        integer :: kp, ip, im
        real(sp) :: w1, w2

        ! does the vertical coordinate increase or decrease with k?
        ip = 0
        im = 1
        if (z(1) > z(nz)) then
            ip = 1
            im = 0
        end if

        vout = missing_sp
        do kp = nz, 2, -1
            if ((z(kp - im) <= height) .and. (z(kp - ip) > height)) then
                w2 = (height - z(kp - im))/(z(kp - ip) - z(kp - im))
                w1 = 1.0_sp - w2
                vout = w1*v(kp - im) + w2*v(kp - ip)
                exit
            end if
        end do

    end function interp_to_height

!-----------------------------------------------------------------------
!  calc_srh - storm relative helicity between the lowest level and
!             `top` (m above ground) for a right and a left mover.
!
!  Input:   nk   - number of levels (lowest level first)
!           u, v - wind components (m/s)
!           aglh - height above ground (m)
!           cu_rm, cv_rm, cu_lm, cv_lm - storm motions (m/s)
!           top  - top of the integration layer (m)
!
!  Output:  sreh_rm, sreh_lm - storm relative helicity (m2/s2)
!
!  When the profile never exceeds `top` both outputs are zero (legacy
!  behaviour).
!-----------------------------------------------------------------------
    pure subroutine calc_srh(nk, u, v, aglh, cu_rm, cv_rm, cu_lm, cv_lm, top, &
                             sreh_rm, sreh_lm)

        integer, intent(in) :: nk
        real(dp), dimension(nk), intent(in) :: u, v, aglh
        real(dp), intent(in) :: cu_rm, cv_rm, cu_lm, cv_lm
        real(dp), intent(in) :: top
        real(dp), intent(out) :: sreh_rm, sreh_lm

        real(dp), dimension(nk) :: utop, vtop
        real(dp) :: total
        integer :: k, ktop

        utop = u
        vtop = v
        ktop = 0
        do k = 2, nk
            if (aglh(k) > top) then
                ktop = k
                utop(ktop) = linear_interp(u(ktop), u(ktop - 1), aglh(ktop), top, aglh(ktop - 1))
                vtop(ktop) = linear_interp(v(ktop), v(ktop - 1), aglh(ktop), top, aglh(ktop - 1))
                exit
            end if
        end do

        total = 0.0_dp
        do k = 2, ktop
            total = total + ((utop(k) - cu_rm)*(vtop(k) - vtop(k - 1))) - &
                            ((vtop(k) - cv_rm)*(utop(k) - utop(k - 1)))
        end do
        sreh_rm = -total

        total = 0.0_dp
        do k = 2, ktop
            total = total + ((utop(k) - cu_lm)*(vtop(k) - vtop(k - 1))) - &
                            ((vtop(k) - cv_lm)*(utop(k) - utop(k - 1)))
        end do
        sreh_lm = -total

    end subroutine calc_srh

!-----------------------------------------------------------------------
!  Given y1 at x1 and y3 at x3, return y at x2.
!-----------------------------------------------------------------------
    pure function linear_interp(y1, y3, x1, x2, x3) result(y2)
        real(dp), intent(in) :: y1, y3, x1, x2, x3
        real(dp) :: y2
        real(dp) :: x1_

        x1_ = x1
        if (x3 == x1_) x1_ = x1_ - 0.01_dp
        y2 = y1 + ((y3 - y1)*((x2 - x1_)/(x3 - x1_)))
    end function linear_interp

end module xcaper_kernels
