!-----------------------------------------------------------------------
!  stdheight_2D_model_lev - height above ground from the hypsometric
!  equation for data on model (sigma / hybrid) levels.
!
!  Author(s): Chiara Lepore @chiaral, John Allen @xebadir
!  The single-column algorithm lives in xcaper_kernels.f90.
!
!  Disclaimer: This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
!  Inputs
!     p          pressure (hPa), shape (nk, nx)
!     t, td      temperature and dewpoint (C), shape (nk, nx)
!     ps, ts, tds surface pressure (hPa), temperature and dewpoint (C)
!     hin        height above ground (m) of the surface level
!  Outputs
!     h          height above ground (m), shape (nk, nx)
!     hs         height above ground (m) of the surface level
!-----------------------------------------------------------------------
subroutine loop_stdheight_ml(p, t, td, ps, ts, tds, hin, nk, nx, h, hs)

    use xcaper_kernels, only: dp, stdheight
    implicit none

    integer, intent(in) :: nk, nx
    real(dp), dimension(nk, nx), intent(in) :: p, t, td
    real(dp), dimension(nx), intent(in) :: ps, ts, tds, hin
    real(dp), dimension(nk, nx), intent(out) :: h
    real(dp), dimension(nx), intent(out) :: hs

    integer :: i

    do i = 1, nx
        call stdheight(nk, p(:, i), t(:, i), td(:, i), ps(i), ts(i), tds(i), hin(i), &
                       h(:, i), hs(i))
    end do

end subroutine loop_stdheight_ml

!-----------------------------------------------------------------------
!  stdheight_ml - heights of a single profile (see xcaper_kernels)
!-----------------------------------------------------------------------
subroutine stdheight_ml(p, t, td, ps, ts, tds, hin, nk, h, hs)

    use xcaper_kernels, only: dp, stdheight
    implicit none

    integer, intent(in) :: nk
    real(dp), dimension(nk), intent(in) :: p, t, td
    real(dp), intent(in) :: ps, ts, tds, hin
    real(dp), dimension(nk), intent(out) :: h
    real(dp), intent(out) :: hs

    call stdheight(nk, p, t, td, ps, ts, tds, hin, h, hs)

end subroutine stdheight_ml
