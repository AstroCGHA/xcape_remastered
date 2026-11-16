!-----------------------------------------------------------------------
!  Bunkers_pressure_lev - Bunkers et al. (2000) storm motion drivers for data
!  on fixed pressure levels (levels below the surface are skipped).
!
!  Author:  Chiara Lepore @chiaral - John Allen @xebadir
!  The single-column algorithm lives in xcaper_kernels.f90.
!
!  Disclaimer:  This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
!  Applies the Bunkers et al. (2000) internal dynamics method for
!  estimating supercell motion to n2 wind profiles.  Each profile is the
!  near-surface wind (lowest model level or 10 m wind) followed by the
!  nk levels of the 3-D arrays.  The profile is sampled every 500 m up
!  to 6 km; right (RM) and left (LM) movers deviate 7.5 m/s from the
!  0-6 km mean wind, perpendicular to the 0-6 km shear vector:
!
!     V_RM = V_mean + 7.5 * (V_shear x k) / |V_shear|
!     V_LM = V_mean - 7.5 * (V_shear x k) / |V_shear|
!
!  Reference: Bunkers et al. (2000): Predicting supercell motion using a
!  new hodograph technique. Wea. Forecasting, 15, 61-79.
!-----------------------------------------------------------------------
!  Inputs
!     u3d, v3d   winds (m/s), shape (nk, n2)
!     aglh3d     height above ground (m), shape (nk, n2)
!     us, vs     near-surface winds (m/s), shape (n2)
!     aglhs      height above ground of the near-surface level (m)
!     start_3d   1-based index of the first level at or above the
!                surface for every grid point
!  Outputs
!     rm, lm       right / left mover motion (u, v), shape (2, n2), m/s
!     mean6kmwind  0-6 km mean wind (u, v), shape (2, n2), m/s
!-----------------------------------------------------------------------
subroutine bunkers_loop_pl(u3d, v3d, aglh3d, us, vs, aglhs, start_3d, nk, n2, rm, lm, mean6kmwind)

    use xcaper_kernels, only: sp, bunkers_calc
    implicit none

    integer, intent(in) :: nk, n2
    real(sp), dimension(nk, n2), intent(in) :: u3d, v3d, aglh3d
    real(sp), dimension(n2), intent(in) :: us, vs, aglhs
    integer, dimension(n2), intent(in) :: start_3d
    real(sp), dimension(2, n2), intent(out) :: rm, lm, mean6kmwind

    integer :: i, nk_all, nk_start
    ! at most, the number of levels is 3d + surface
    real(sp), dimension(nk + 1) :: u_all, v_all, aglh_all

    do i = 1, n2
        ! use only levels at or above the surface
        nk_start = min(max(start_3d(i), 1), nk + 1)
        ! number of used levels in 3d + surface
        nk_all = nk - nk_start + 2

        u_all(1) = us(i)
        v_all(1) = vs(i)
        aglh_all(1) = aglhs(i)
        u_all(2:nk_all) = u3d(nk_start:nk, i)
        v_all(2:nk_all) = v3d(nk_start:nk, i)
        aglh_all(2:nk_all) = aglh3d(nk_start:nk, i)

        call bunkers_calc(nk_all, u_all(1:nk_all), v_all(1:nk_all), aglh_all(1:nk_all), &
                          rm(:, i), lm(:, i), mean6kmwind(:, i))
    end do

end subroutine bunkers_loop_pl

!-----------------------------------------------------------------------
!  bunkers_calc_pl - storm motion of a single profile
!-----------------------------------------------------------------------
subroutine bunkers_calc_pl(u, v, aglh, nk, rm, lm, mean6kmwind)

    use xcaper_kernels, only: sp, bunkers_calc
    implicit none

    integer, intent(in) :: nk
    real(sp), dimension(nk), intent(in) :: u, v, aglh
    real(sp), dimension(2), intent(out) :: rm, lm, mean6kmwind

    call bunkers_calc(nk, u, v, aglh, rm, lm, mean6kmwind)

end subroutine bunkers_calc_pl

!-----------------------------------------------------------------------
!  dinterp2dz - interpolate n2 profiles v3d(nz, n2), given at heights
!  z(nz, n2), to the height loc.  Points whose profile does not bracket
!  loc are returned as -999999.
!-----------------------------------------------------------------------
subroutine dinterp2dz(v3d, z, loc, v2d, n2, nz)

    use xcaper_kernels, only: sp, interp_to_height
    implicit none

    integer, intent(in) :: n2, nz
    real(sp), dimension(nz, n2), intent(in) :: v3d, z
    real(sp), intent(in) :: loc
    real(sp), dimension(n2), intent(out) :: v2d

    integer :: j

    do j = 1, n2
        v2d(j) = interp_to_height(nz, v3d(:, j), z(:, j), loc)
    end do

end subroutine dinterp2dz
