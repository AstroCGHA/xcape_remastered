!-----------------------------------------------------------------------
!  CAPE_CODE_pressure_lev - CAPE / CIN drivers for data on fixed
!  pressure levels (levels below the surface are skipped).
!
!  Author:  Chiara Lepore @chiaral
!  The single-column algorithm (George H. Bryan's getcape) lives in
!  xcaper_kernels.f90.
!
!  Disclaimer:  This code is made available WITHOUT WARRANTY.
!-----------------------------------------------------------------------
!  Common arguments
!     nk_in    number of pressure levels
!     n2       number of grid points
!     pinc     pressure increment (Pa) of the parcel ascent; smaller is
!              more accurate, larger is faster (500-1000 is sensible)
!     source   1 = surface, 2 = most unstable, 3 = mixed layer
!     ml_depth depth (m) of the mixed layer for source = 3
!     adiabat  1 = pseudoadiabatic liquid, 2 = reversible liquid,
!              3 = pseudoadiabatic ice,    4 = reversible ice
!     start_3d 1-based index of the first level at or above the surface
!              (p3d_in <= ps_in) for every grid point
!
!  Grid points whose surface temperature is not above 0 C (which
!  includes NaN) are returned as cape = cin = z_out = 0, mulvl = 0.
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!  loopcape_pl - pressure given for every grid point: p3d_in(nk_in,n2)
!-----------------------------------------------------------------------
subroutine loopcape_pl(p3d_in, t3d_in, td3d_in, ps1d_in, ts1d_in, tds1d_in, &
                       pinc, source, ml_depth, adiabat, start_3d, &
                       nk_in, n2, cape3d, cin3d, mulvl3d, z_out3d)

    use xcaper_kernels, only: sp, getcape
    implicit none

    integer, intent(in) :: nk_in, n2
    real(sp), dimension(nk_in, n2), intent(in) :: p3d_in, t3d_in, td3d_in
    real(sp), dimension(n2), intent(in) :: ps1d_in, ts1d_in, tds1d_in
    real(sp), intent(in) :: pinc
    integer, intent(in) :: source
    real(sp), intent(in) :: ml_depth
    integer, intent(in) :: adiabat
    integer, dimension(n2), intent(in) :: start_3d
    real(sp), dimension(n2), intent(out) :: cape3d, cin3d, z_out3d
    integer, dimension(n2), intent(out) :: mulvl3d

    integer :: i, nk_start, nk_pl_in

    do i = 1, n2
        if (ts1d_in(i) > 0.0_sp) then
            ! use only levels where p3d_in <= ps_in
            nk_start = min(max(start_3d(i), 1), nk_in + 1)
            nk_pl_in = nk_in - nk_start + 1
            call getcape(nk_pl_in, p3d_in(nk_start:nk_in, i), t3d_in(nk_start:nk_in, i), &
                         td3d_in(nk_start:nk_in, i), ps1d_in(i), ts1d_in(i), tds1d_in(i), &
                         pinc, source, ml_depth, adiabat, &
                         cape3d(i), cin3d(i), mulvl3d(i), z_out3d(i))
        else
            cape3d(i)  = 0.0_sp
            cin3d(i)   = 0.0_sp
            z_out3d(i) = 0.0_sp
            mulvl3d(i) = 0
        end if
    end do

end subroutine loopcape_pl

!-----------------------------------------------------------------------
!  loopcape_pl1d - one pressure vector shared by all grid points:
!                  p3d_in(nk_in,1)
!-----------------------------------------------------------------------
subroutine loopcape_pl1d(t3d_in, td3d_in, p3d_in, ps1d_in, ts1d_in, tds1d_in, &
                         pinc, source, ml_depth, adiabat, start_3d, &
                         nk_in, n2, cape3d, cin3d, mulvl3d, z_out3d)

    use xcaper_kernels, only: sp, getcape
    implicit none

    integer, intent(in) :: nk_in, n2
    real(sp), dimension(nk_in, n2), intent(in) :: t3d_in, td3d_in
    real(sp), dimension(nk_in, 1), intent(in) :: p3d_in
    real(sp), dimension(n2), intent(in) :: ps1d_in, ts1d_in, tds1d_in
    real(sp), intent(in) :: pinc
    integer, intent(in) :: source
    real(sp), intent(in) :: ml_depth
    integer, intent(in) :: adiabat
    integer, dimension(n2), intent(in) :: start_3d
    real(sp), dimension(n2), intent(out) :: cape3d, cin3d, z_out3d
    integer, dimension(n2), intent(out) :: mulvl3d

    integer :: i, nk_start, nk_pl_in

    do i = 1, n2
        if (ts1d_in(i) > 0.0_sp) then
            ! use only levels where p3d_in <= ps_in
            nk_start = min(max(start_3d(i), 1), nk_in + 1)
            nk_pl_in = nk_in - nk_start + 1
            call getcape(nk_pl_in, p3d_in(nk_start:nk_in, 1), t3d_in(nk_start:nk_in, i), &
                         td3d_in(nk_start:nk_in, i), ps1d_in(i), ts1d_in(i), tds1d_in(i), &
                         pinc, source, ml_depth, adiabat, &
                         cape3d(i), cin3d(i), mulvl3d(i), z_out3d(i))
        else
            cape3d(i)  = 0.0_sp
            cin3d(i)   = 0.0_sp
            z_out3d(i) = 0.0_sp
            mulvl3d(i) = 0
        end if
    end do

end subroutine loopcape_pl1d

!-----------------------------------------------------------------------
!  getcape_pl - CAPE / CIN of a single profile (see xcaper_kernels)
!-----------------------------------------------------------------------
subroutine getcape_pl(p_in_a, t_in_a, td_in_a, ps_in_b, ts_in_b, tds_in_b, &
                      pinc, source, ml_depth, adiabat, nk_in, cape, cin, mulvl, z_out)

    use xcaper_kernels, only: sp, getcape
    implicit none

    integer, intent(in) :: nk_in
    real(sp), dimension(nk_in), intent(in) :: p_in_a, t_in_a, td_in_a
    real(sp), dimension(1), intent(in) :: ps_in_b, ts_in_b, tds_in_b
    real(sp), intent(in) :: pinc
    integer, intent(in) :: source
    real(sp), intent(in) :: ml_depth
    integer, intent(in) :: adiabat
    real(sp), intent(out) :: cape, cin, z_out
    integer, intent(out) :: mulvl

    call getcape(nk_in, p_in_a, t_in_a, td_in_a, ps_in_b(1), ts_in_b(1), tds_in_b(1), &
                 pinc, source, ml_depth, adiabat, cape, cin, mulvl, z_out)

end subroutine getcape_pl

!-----------------------------------------------------------------------
!  Thermodynamic helper functions kept for backward compatibility of
!  the Python bindings.  p in Pa, t and td in K, q in kg/kg.
!-----------------------------------------------------------------------
function getqvs(p, t)
    use xcaper_kernels, only: sp, qvs_liquid
    implicit none
    real(sp), intent(in) :: p, t
    real(sp) :: getqvs
    getqvs = qvs_liquid(p, t)
end function getqvs

function getqvi(p, t)
    use xcaper_kernels, only: sp, qvs_ice
    implicit none
    real(sp), intent(in) :: p, t
    real(sp) :: getqvi
    getqvi = qvs_ice(p, t)
end function getqvi

function getthe(p, t, td, q)
    use xcaper_kernels, only: sp, theta_e
    implicit none
    real(sp), intent(in) :: p, t, td, q
    real(sp) :: getthe
    getthe = theta_e(p, t, td, q)
end function getthe
