!-----------------------------------------------------------------------
!  stdheight_2D_pressure_lev - height above ground from the hypsometric
!  equation for data on fixed pressure levels.
!
!  Author(s): Chiara Lepore @chiaral, John Allen @xebadir
!  The single-column algorithm lives in xcaper_kernels.f90.
!
!  Disclaimer: This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
!  Inputs
!     p          pressure (hPa): (nk, nx), or (nk, 1) for the 1d variant
!     t, td      temperature and dewpoint (C), shape (nk, nx)
!     ps, ts, tds surface pressure (hPa), temperature and dewpoint (C)
!     hin        height above ground (m) of the surface level
!     start_3d   1-based index of the first level at or above the
!                surface for every grid point
!  Outputs
!     h          height above ground (m), shape (nk, nx); levels below
!                the surface are set to -999999
!     hs         height above ground (m) of the surface level
!-----------------------------------------------------------------------
subroutine loop_stdheight_pl(p, t, td, ps, ts, tds, hin, start_3d, nk, nx, h, hs)

    use xcaper_kernels, only: dp, stdheight, missing_dp
    implicit none

    integer, intent(in) :: nk, nx
    real(dp), dimension(nk, nx), intent(in) :: p, t, td
    real(dp), dimension(nx), intent(in) :: ps, ts, tds, hin
    integer, dimension(nx), intent(in) :: start_3d
    real(dp), dimension(nk, nx), intent(out) :: h
    real(dp), dimension(nx), intent(out) :: hs

    integer :: i, nk_pl_in, nk_start

    do i = 1, nx
        nk_start = min(max(start_3d(i), 1), nk + 1)
        nk_pl_in = nk - nk_start + 1
        if (nk_start > 1) h(1:nk_start - 1, i) = missing_dp
        call stdheight(nk_pl_in, p(nk_start:nk, i), t(nk_start:nk, i), td(nk_start:nk, i), &
                       ps(i), ts(i), tds(i), hin(i), h(nk_start:nk, i), hs(i))
    end do

end subroutine loop_stdheight_pl

!-----------------------------------------------------------------------
!  loop_stdheight_pl1d - as loop_stdheight_pl, with one pressure vector
!  p(nk, 1) shared by all grid points.
!-----------------------------------------------------------------------
subroutine loop_stdheight_pl1d(t, td, p, ps, ts, tds, hin, start_3d, nk, nx, h, hs)

    use xcaper_kernels, only: dp, stdheight, missing_dp
    implicit none

    integer, intent(in) :: nk, nx
    real(dp), dimension(nk, nx), intent(in) :: t, td
    real(dp), dimension(nk, 1), intent(in) :: p
    real(dp), dimension(nx), intent(in) :: ps, ts, tds, hin
    integer, dimension(nx), intent(in) :: start_3d
    real(dp), dimension(nk, nx), intent(out) :: h
    real(dp), dimension(nx), intent(out) :: hs

    integer :: i, nk_pl_in, nk_start

    do i = 1, nx
        nk_start = min(max(start_3d(i), 1), nk + 1)
        nk_pl_in = nk - nk_start + 1
        if (nk_start > 1) h(1:nk_start - 1, i) = missing_dp
        call stdheight(nk_pl_in, p(nk_start:nk, 1), t(nk_start:nk, i), td(nk_start:nk, i), &
                       ps(i), ts(i), tds(i), hin(i), h(nk_start:nk, i), hs(i))
    end do

end subroutine loop_stdheight_pl1d

!-----------------------------------------------------------------------
!  stdheight_pl - heights of a single profile (see xcaper_kernels)
!-----------------------------------------------------------------------
subroutine stdheight_pl(p, t, td, ps, ts, tds, hin, nk, h, hs)

    use xcaper_kernels, only: dp, stdheight
    implicit none

    integer, intent(in) :: nk
    real(dp), dimension(nk), intent(in) :: p, t, td
    real(dp), intent(in) :: ps, ts, tds, hin
    real(dp), dimension(nk), intent(out) :: h
    real(dp), intent(out) :: hs

    call stdheight(nk, p, t, td, ps, ts, tds, hin, h, hs)

end subroutine stdheight_pl
