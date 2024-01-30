import sys, os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import cmasher as cmr
plt.style.use('kiral.mplstyle')

import astropy.units as u
import velocity_tools.velocity_tools.stream_lines as SL

savedir      = './_view_model_3D/'

#########################
# CONVENIENCE FUNCTIONS #
#########################

def get_streamer(mass, r0, theta0, phi0, omega0, v_r0, inc, PA):
    (x1, y1, z1), (vx1, vy1, vz1) = SL.xyz_stream(mass=mass,      # Central mass
                                                  r0=r0,          # Initial radius of streamline
                                                  theta0=theta0,  # Initial polar angle of streamline
                                                  phi0=phi0,      # Initial azimuthal angle of streamline
                                                  omega=omega0,   # Angular rotation. (defined positive)
                                                  v_r0=v_r0,      # Initial radial velocity of the streamline
                                                  inc=inc,        # inclination with respect of line-of-sight, inc=0 is an edge-on-disk
                                                  pa=PA,          # Position angle of the rotation axis, measured due East from North.
                                                  deltar=10*u.au) # spacing between two consecutive radii in the sampling of the streamer, in au
    # # we obtain the distance of each point in the sky
    # d_sky_au = np.sqrt(x1**2 + z1**2)
    # # Stream line into arcsec
    # dra_stream = -x1.value / dist_Per50
    # ddec_stream = z1.value / dist_Per50
    # fil = SkyCoord(dra_stream*u.arcsec, ddec_stream*u.arcsec,
    #                frame=Per50_ref).transform_to(FK5)
    # velocity = v_lsr + vy1
    # return fil, d_sky_au, velocity
    return x1, y1, z1, vx1, vy1, vz1 # Jess



##############
# PARAMETERS #
##############

# Properties of the system
Mstar       = 2.4 * u.Msun      # Central mass concentrated at center
inc         = 90.  * u.deg      # Disk inclination (i=0 is edge on)
PA_ang      = 0.  * u.deg       # Disk position angle
v_lsr       = 0.  * u.km/u.s    # Systemic velocity (Fixed parameter)

# Initial position and velocity of infalling particle ("tail" of streamer)
r0          = 800  * u.au       # Initial distance from star (radius of sphere undergoing solid-body rotation)
theta0      = 45.  * u.deg      # Initial polar angle (subtended between trajectory's plane and the z-axis/rotation-axis)
phi0        = 0.0  * u.deg      # Initial azimuthal angle position (x-axis is phi=0)
v_r0        = 5.   * u.km/u.s   # Initial radial velocity
omega0      = 8.e-12 / u.s      # Initial rotation frequency of "cloud" (sphere undergoing solid-body rotation)

# Centrifugal radius or disk radius in the Ulrich (1976)'s model
r_c         = SL.r_cent(Mstar, omega0, r0)



###############
# MAKE FIGURE #
###############

fig     = plt.figure(figsize=(7.09*1.5*1.1, 7.09*1.5*1.1))
gs      = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[0.8, 1, 0.8], height_ratios=[0.03, 1])
ax1     = fig.add_subplot(gs[1,:], projection='3d')
cax1    = fig.add_subplot(gs[0,1])

cmap      = cmr.get_sub_cmap('cmr.cosmic', 0.0, 0.8)
getcolour = cm.get_cmap(cmap)

######### Plot the xy plane for visual clarity #########
xx, yy = np.meshgrid(np.arange(-1000, 1000, 1), np.arange(-1000, 1000, 1))
ax1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='grey')

for ax in [ax1]:
    ax.set_xlabel(r'R.A. (au)', fontweight='bold')
    ax.set_ylabel(r'Dec. (au)', fontweight='bold')
    ax.set_zlabel(r'$v_{\rm los}$ (km/s)', fontweight='bold')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-5, 5)
    ax.plot([-1000, 1000],[0, 0],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
    ax.plot([0, 0],[-1000, 1000],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
    ax.plot([0, 0],[0, 0],[-1000, 1000], color='k', lw=1) # plot xyz axes for visual clarity
    ax.grid(alpha=0.5, color='grey')
    ax.scatter([0],[0],[0], marker='*', color='k', s=50, zorder=10000)
    ax.scatter([0],[0],[0], marker='*', color='w', s=25, zorder=10000)
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.zaxis.set_major_locator(MultipleLocator(1.))
    ax.tick_params(axis='x', which='minor', length=0)
    # ax.axis('off')
    ax.invert_xaxis()


####################################
# VARY THE INITIAL AZIMUTHAL ANGLE #
####################################

savedir += 'phi0_ppv/'

phi0s        =  np.array([0., 45, 90, 135, 180, 225, 270, 315, 360]) * u.deg
norm           = mpl.colors.Normalize(vmin=np.min(phi0s.value), vmax=np.max(phi0s.value))
cb = mpl.colorbar.ColorbarBase(ax=cax1, cmap=cmap, orientation='horizontal', ticks=phi0s.value, norm=norm, extend='both', ticklocation='bottom', format='%.0f')
cb.set_label(r'Initial azimuthal angle $\phi_{0}$ ($^{\circ}$)', rotation=0, fontsize=12, fontweight='bold', labelpad=6, color='k')

for phi0 in phi0s:
    colour = getcolour(norm(phi0.value))

    ############## Get the streamer and plot its x,y,vlos trajectory ##############
    RA, LOS, Dec, vRA, v_LOS, vDec = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
    ax1.plot(RA, Dec, -v_LOS, color=colour, lw=2)

    if phi0 == phi0s[-1]:

        cax1.text(-0.6, -3., r'System properties', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
        cax1.text(-0.6, -4., r'$M_{\rm central} = $'+str(Mstar), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
        cax1.text(-0.6, -5., r'$i = $ (face-on)'+str(inc), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
        cax1.text(-0.6, -6., r'${\rm PA} = $'+str(PA_ang), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
        cax1.text(-0.6, -7., r'$v_{\rm LSR} = $'+str(v_lsr), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')

        cax1.text(1.6, -3., r'Particle initial parameters', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -4., r'$r_{0} = $'+str(r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -5., r'$\theta_{0} = $'+str(theta0 ), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -6., r'$\phi_{0} = $'+str(phi0), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -7., r'$v_{r, 0} = $'+str(v_r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -8., r'$\Omega_{0} = $'+str(omega0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        cax1.text(1.6, -9., r'$r_{\rm centrifugal} = %.2f$ au'%(r_c.value), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

if not os.path.isdir(savedir):
    os.system('mkdir -p '+savedir)
gs.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
plt.tight_layout()

for ii in np.arange(0,360,1):
        ax.view_init(elev=15., azim=ii)
        plt.savefig(savedir+"img_vr_%04i.png"%ii, bbox_inches='tight')




#
# ###########################
# # VARY THE INITIAL RADIUS #
# ###########################
#
# savedir += 'r0_ppv/'
#
# r0s        = np.array([250, 500, 750, 1000]) * u.au
# norm       = mpl.colors.Normalize(vmin=np.min(r0s.value), vmax=np.max(r0s.value))
# cb = mpl.colorbar.ColorbarBase(ax=cax1, cmap=cmap, orientation='horizontal', ticks=r0s.value, norm=norm, extend='both', ticklocation='bottom', format='%.0f')
# cb.set_label(r'Initial radius $r_{0}$ (au)', rotation=0, fontsize=12, fontweight='bold', labelpad=6, color='k')
#
# for r0 in r0s:
#     colour = getcolour(norm(r0.value))
#
#     ############## Get the streamer and plot its x,y,vlos trajectory ##############
#     RA, LOS, Dec, vRA, v_LOS, vDec = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
#     ax1.plot(RA, Dec, -v_LOS, color=colour, lw=2)
#
#     if r0 == r0s[-1]:
#         cax1.text(-0.6, -3., r'System properties', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -4., r'$M_{\rm central} = $'+str(Mstar), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -5., r'$i = $ (face-on)'+str(inc), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -6., r'${\rm PA} = $'+str(PA_ang), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -7., r'$v_{\rm LSR} = $'+str(v_lsr), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#
#         cax1.text(1.6, -3., r'Particle initial parameters', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -4., r'$r_{0} = $'+str(r0), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -5., r'$\theta_{0} = $'+str(theta0 ), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -6., r'$\phi_{0} = $'+str(phi0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -7., r'$v_{r, 0} = $'+str(v_r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -8., r'$\Omega_{0} = $'+str(omega0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -9., r'$r_{\rm centrifugal} = %.2f$ au'%(r_c.value), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#
# if not os.path.isdir(savedir):
#     os.system('mkdir -p '+savedir)
# gs.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
# plt.tight_layout()
#
# for ii in np.arange(0,360,1):
#         ax.view_init(elev=15., azim=ii)
#         plt.savefig(savedir+"img_%04i.png"%ii, bbox_inches='tight')

# #####################################
# # VARY THE INITIAL ANGULAR VELOCITY #
# #####################################
#
# savedir += 'omega0_ppv/'
#
# omega0s        =  np.array([1., 5, 7, 9, 10.]) / u.s # Angular rotation. (defined positive)
# norm           = mpl.colors.Normalize(vmin=np.min(omega0s.value), vmax=np.max(omega0s.value))
# cb = mpl.colorbar.ColorbarBase(ax=cax1, cmap=cmap, orientation='horizontal', ticks=omega0s.value, norm=norm, extend='both', ticklocation='bottom', format='%.0f')
# cb.set_label(r'Initial angular frequency $\Omega_{0}$ ($\times 10^{-12}$ s$^{-1}$)', rotation=0, fontsize=12, fontweight='bold', labelpad=6, color='k')
#
# for omega0 in omega0s*1e-12:
#     colour = getcolour(norm(omega0.value/1e-12))
#
#     ############## Get the streamer and plot its x,y,vlos trajectory ##############
#     RA, LOS, Dec, vRA, v_LOS, vDec = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
#     ax1.plot(RA, Dec, -v_LOS, color=colour, lw=2)
#
#     if omega0/1e-12 == omega0s[-1]:
#
#         cax1.text(-0.6, -3., r'System properties', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -4., r'$M_{\rm central} = $'+str(Mstar), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -5., r'$i = $ (face-on)'+str(inc), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -6., r'${\rm PA} = $'+str(PA_ang), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -7., r'$v_{\rm LSR} = $'+str(v_lsr), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#
#         cax1.text(1.6, -3., r'Particle initial parameters', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -4., r'$r_{0} = $'+str(r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -5., r'$\theta_{0} = $'+str(theta0 ), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -6., r'$\phi_{0} = $'+str(phi0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -7., r'$v_{r, 0} = $'+str(v_r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -8., r'$\Omega_{0} = $'+str(omega0), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -9., r'$r_{\rm centrifugal} = %.2f$ au'%(r_c.value), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#
# if not os.path.isdir(savedir):
#     os.system('mkdir -p '+savedir)
# gs.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
# plt.tight_layout()
#
# for ii in np.arange(0,360,1):
#         ax.view_init(elev=15., azim=ii)
#         plt.savefig(savedir+"img_%04i.png"%ii, bbox_inches='tight')

# ################################
# # VARY THE INITIAL POLAR ANGLE #
# ################################
#
# savedir += 'theta0_ppv/'
#
# theta0s        =  np.array([1., 20, 40, 60, 80, 89]) * u.deg
# norm           = mpl.colors.Normalize(vmin=np.min(theta0s.value), vmax=np.max(theta0s.value))
# cb = mpl.colorbar.ColorbarBase(ax=cax1, cmap=cmap, orientation='horizontal', ticks=theta0s.value, norm=norm, extend='both', ticklocation='bottom', format='%.0f')
# cb.set_label(r'Initial polar angle $\theta_{0}$ ($^{\circ}$)', rotation=0, fontsize=12, fontweight='bold', labelpad=6, color='k')
#
# for theta0 in theta0s:
#     colour = getcolour(norm(theta0.value))
#
#     ############## Get the streamer and plot its x,y,vlos trajectory ##############
#     RA, LOS, Dec, vRA, v_LOS, vDec = get_streamer(Mstar, r0, theta0, phi0, omega0, v_r0, inc, PA_ang)
#     ax1.plot(RA, Dec, -v_LOS, color=colour, lw=2)
#
#     if theta0 == theta0s[-1]:
#
#         cax1.text(-0.6, -3., r'System properties', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -4., r'$M_{\rm central} = $'+str(Mstar), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -5., r'$i = $ (face-on)'+str(inc), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -6., r'${\rm PA} = $'+str(PA_ang), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#         cax1.text(-0.6, -7., r'$v_{\rm LSR} = $'+str(v_lsr), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
#
#         cax1.text(1.6, -3., r'Particle initial parameters', fontweight='bold', transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -4., r'$r_{0} = $'+str(r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -5., r'$\theta_{0} = $'+str(theta0 ), transform=cax1.transAxes, color=colour, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -6., r'$\phi_{0} = $'+str(phi0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -7., r'$v_{r, 0} = $'+str(v_r0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -8., r'$\Omega_{0} = $'+str(omega0), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#         cax1.text(1.6, -9., r'$r_{\rm centrifugal} = %.2f$ au'%(r_c.value), transform=cax1.transAxes, color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
#
# if not os.path.isdir(savedir):
#     os.system('mkdir -p '+savedir)
# gs.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
# plt.tight_layout()
#
# for ii in np.arange(0,360,1):
#         ax.view_init(elev=15., azim=ii)
#         plt.savefig(savedir+"img_%04i.png"%ii, bbox_inches='tight')
#





sys.exit()
