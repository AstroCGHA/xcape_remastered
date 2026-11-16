!-----------------------------------------------------------------------
!  SREH_pressure_lev - storm relative helicity drivers for data on
!  fixed pressure levels (levels below the surface are skipped).
!
!  Author:  Chiara Lepore @chiaral
!  The single-column algorithm lives in xcaper_kernels.f90.
!
!  Disclaimer:  This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
!  Inputs
!     u, v        winds (m/s), shape (nk, n2)
!     aglh        height above ground (m), shape (nk, n2)
!     us, vs      near-surface winds (m/s), shape (n2)
!     aglhs       height above ground of the near-surface level (m)
!     cu_rm, cv_rm, cu_lm, cv_lm
!                 right / left mover storm motion (m/s), shape (n2)
!     top         top of the integration layer (m above ground)
!     start_3d   1-based index of the first level at or above the
!                surface for every grid point
!  Outputs
!     sreh_rm, sreh_lm  storm relative helicity (m2/s2), shape (n2)
!-----------------------------------------------------------------------
subroutine loop_sreh_pl(u, v, aglh, us, vs, aglhs, cu_rm, cv_rm, cu_lm, cv_lm, top, start_3d, nk, n2, &
                        sreh_rm, sreh_lm)

    use xcaper_kernels, only: dp, calc_srh
    implicit none

    integer, intent(in) :: nk, n2
    real(dp), dimension(nk, n2), intent(in) :: u, v, aglh
    real(dp), dimension(n2), intent(in) :: us, vs, aglhs
    real(dp), dimension(n2), intent(in) :: cu_rm, cv_rm, cu_lm, cv_lm
    real(dp), intent(in) :: top
    integer, dimension(n2), intent(in) :: start_3d
    real(dp), dimension(n2), intent(out) :: sreh_rm, sreh_lm

    integer :: i, nk_all, nk_start
    ! at most, the number of levels is 3d + surface
    real(dp), dimension(nk + 1) :: u_all, v_all, aglh_all

    do i = 1, n2
        ! use only levels at or above the surface
        nk_start = min(max(start_3d(i), 1), nk + 1)
        ! number of used levels in 3d + surface
        nk_all = nk - nk_start + 2

        u_all(1) = us(i)
        v_all(1) = vs(i)
        aglh_all(1) = aglhs(i)
        u_all(2:nk_all) = u(nk_start:nk, i)
        v_all(2:nk_all) = v(nk_start:nk, i)
        aglh_all(2:nk_all) = aglh(nk_start:nk, i)

        call calc_srh(nk_all, u_all(1:nk_all), v_all(1:nk_all), aglh_all(1:nk_all), &
                      cu_rm(i), cv_rm(i), cu_lm(i), cv_lm(i), top, &
                      sreh_rm(i), sreh_lm(i))
    end do

end subroutine loop_sreh_pl

!-----------------------------------------------------------------------
!  dcalrelhl_pl - storm relative helicity of a single profile
!-----------------------------------------------------------------------
subroutine dcalrelhl_pl(u, v, aglh, cu_rm, cv_rm, cu_lm, cv_lm, top, nk, &
                        sreh_rm, sreh_lm)

    use xcaper_kernels, only: dp, calc_srh
    implicit none

    integer, intent(in) :: nk
    real(dp), dimension(nk), intent(in) :: u, v, aglh
    real(dp), dimension(1), intent(in) :: cu_rm, cv_rm, cu_lm, cv_lm
    real(dp), intent(in) :: top
    real(dp), dimension(1), intent(out) :: sreh_rm, sreh_lm

    call calc_srh(nk, u, v, aglh, cu_rm(1), cv_rm(1), cu_lm(1), cv_lm(1), top, &
                  sreh_rm(1), sreh_lm(1))

end subroutine dcalrelhl_pl

!-----------------------------------------------------------------------
!  interp1 - given y1 at x1 and y3 at x3, return y at x2.
!  Kept for backward compatibility of the Python bindings.
!-----------------------------------------------------------------------
function interp1(y1, y3, x1, x2, x3)
    use xcaper_kernels, only: dp, linear_interp
    implicit none
    real(dp), intent(in) :: y1, y3, x1, x2, x3
    real(dp) :: interp1
    interp1 = linear_interp(y1, y3, x1, x2, x3)
end function interp1
