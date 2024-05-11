"""
Implementation of stream lines using the prescription from Mendoza et al. (2009).
doi:10.1111/j.1365-2966.2008.14210.x

Functions here are sourced from original implementation by Jaime Pineda.
https://github.com/jpinedaf/velocity_tools/blob/master/velocity_tools/stream_lines.py

Adapted into be object oriented by Jess Speedie 01/2024.
"""

import numpy as np
import astropy.units as u
from scipy import optimize
from astropy.constants import G
import matplotlib.pyplot as plt

class streamline(object):
    """
    Base class containing all the properties of a streamline.

    Args:
        mass (float):     [Msun]
        r0 (float):        [au]
        omega (float):     [Hz]
        theta0 (float):    [deg]
        phi0 (float):      [deg]
        v_r0 (float):      [km/s]
        rmin (Optional[float]):   [au]
        delta_r (Optional[float]):   [au]

        inc (Optional[float]):      [deg]
        pa (Optional[float]):       [deg]

        dist (Optional[float]):     [pc]
        v_sys (Optional[float]):    [km/s]

    Returns:
        A `streamline` instance, with the following attributes:
            r_cent (float): Centrifugal radius of the streamline, in [au].
            x (1D array): Cartesian x-position along the streamline, in [au].
            y (1D array): Cartesian y-position along the streamline, in [au].
            z (1D array): Cartesian z-position along the streamline, in [au].
            v_x (1D array): Cartesian x-velocity along the streamline, in [km/s].
            v_y (1D array): Cartesian y-velocity along the streamline, in [km/s].
            v_z (1D array): Cartesian z-velocity along the streamline, in [km/s].
            r (1D array): Spherical r-position along the streamline, in [au]. In
                other words, (x**2 + y**2 + z**2)**0.5.
            theta (1D array): Spherical theta-position along the streamline, in [rad].
            phi (1D array): Spherical phi-position along the streamline, in [deg].
            v_r (1D array): Spherical r-velocity along the streamline, in [km/s].
            v_theta (1D array): Spherical theta-velocity along the streamline, in [km/s].
            v_phi (1D array): Spherical phi-velocity along the streamline, in [km/s].
        If the disk inclination `inc` and position angle `pa` are provided:
            x_sky (1D array): Cartesian x-position along the streamline projected
                onto the sky, in [au]. Increasing x is westward.
            y_sky (1D array): Cartesian y-position along the streamline projected
                onto the sky, in [au]. Increasing y is northward.
            z_sky (1D array): Cartesian z-position along the streamline projected
                onto the sky, in [au]. Increasing z is into the plane of the sky,
                away from the observer. ***CHECK THIS, SEEMS OPPOSITE
            r_sky (1D array): Cartesian distance from the star along the streamline
                projected onto the sky, in [au]. Hypotenuse of `x_sky` and `y_sky`.
            v_x_sky (1D array): Cartesian x-velocity along the streamline projected
                onto the sky, in [au]. Increasing x is westward.
            v_y_sky (1D array): Cartesian y-velocity along the streamline projected
                onto the sky, in [au]. Increasing y is northward.
            v_z_sky (1D array): Cartesian z-velocity along the streamline projected
                onto the sky, in [au]. Increasing z is into the plane of the sky,
                away from the observer.
        If the distance to the system `dist` and the systemic velocity `v_sys` are ALSO provided:
            RA (1D array): Angular RA-position along the streamline projected
                onto the sky, in [arcsec]. Increasing RA is eastward.
            Dec (1D array): Angular Dec-position along the streamline projected
                onto the sky, in [arcsec]. Increasing Dec is northward.
            LOS (1D array): LOS-position along the streamline projected
                onto the sky, in [arcsec]. Increasing LOS is into the plane of the sky,
                away from the observer.
            r_proj (1D array): Angular distance from the star along the streamline
                projected onto the sky, in [arcsec]. Hypotenuse of `RA` and `Dec`.
            v_RA (1D array): Angular RA-velocity along the streamline projected
                onto the sky, in [km/s]. Increasing RA is eastward.
            v_Dec (1D array): Angular Dec-velocity along the streamline projected
                onto the sky, in [km/s]. Increasing Dec is northward.
            v_LOS (1D array): LOS-velocity along the streamline projected
                onto the sky, in [km/s]. Increasing LOS is into the plane of the sky,
                away from the observer.
        In disk frame coordinates, the central `mass` [i.e. the star] is at
            (x,y,z)=(0,0,0).
        In sky frame coordinates, the central `mass` [i.e. the star] is at
            (x_sky,y_sky,z_sky)=(0,0,0) and (RA,Dec,LOS)=(0,0,0).
    """

    def __init__(self, mass=0.5, r0=1e4, omega=1e-14, theta0=30., phi0=0.,
                 v_r0=0., rmin=None, delta_r=None,
                 inc=None, pa=None,
                 dist=None, v_sys=None):

        # Properties of the disk/system
        self.mass       = mass   *u.Msun

        # Properties of the cloud/streamer
        self.r0         = r0      *u.au
        self.theta0     = theta0  *u.deg
        self.phi0       = phi0    *u.deg
        self.omega      = omega   /u.s
        self.v_r0       = v_r0    *u.km/u.s

        # Centrifual radius of the cloud/streamer [au]
        self.r_cent = self._get_r_cent(mass=self.mass, omega=self.omega, r0=self.r0)

        # Check the centrifugal radius is inside the initial radius
        if self.r0.value <= self.r_cent.value:
            raise ValueError("Outer radius (%.2f) cannot be smaller than centrifugal radius (%.2f)."%(self.r0.value, self.r_cent.value))

        # Check the final radius is inside the initial radius
        self.rmin = self.r_cent*0.5 if rmin is None else rmin*u.au
        if self.r0.value <= self.rmin.value:
            raise ValueError("Outer radius (%.2f) cannot be smaller than inner radius (%.2f)."%(self.r0.value, self.rmin.value))
        self.delta_r = self.r_cent*0.01 if delta_r is None else delta_r *u.au


        """Streamline properties in the disk frame: Cartesian and spherical."""
        (self.x, self.y, self.z, self.v_x, self.v_y, self.v_z, \
        self.r, self.theta, self.phi, self.v_r, self.v_theta, self.v_phi) \
        = self._get_streamline(mass=self.mass, r0=self.r0, theta0=self.theta0,
                       phi0=self.phi0, omega=self.omega, v_r0=self.v_r0,
                       rmin=self.rmin, delta_r=self.delta_r)

        """Streamline properties in the sky frame: Cartesian coordinates."""
        if inc is not None and pa is not None:
            self.inc        = inc    *u.deg
            self.pa         = pa     *u.deg

            (self.x_sky, self.z_sky, self.y_sky) \
            = streamline.rotate_xyz(self.x, self.y, self.z, inc=self.inc, pa=self.pa)
            self.r_sky = (self.x_sky**2. + self.y_sky**2.)**0.5

            (self.v_x_sky, self.v_z_sky, self.v_y_sky) \
            = streamline.rotate_xyz(self.v_x, self.v_y, self.v_z, inc=self.inc, pa=self.pa)

            """Streamline properties in the sky frame: Angular coordinates."""
            if dist is not None and v_sys is not None:
                self.dist       = dist   *u.pc
                self.v_sys      = v_sys  *u.km/u.s

                self.RA     = (-self.x_sky.value / self.dist.value) *u.arcsec
                self.Dec    =  (self.y_sky.value / self.dist.value) *u.arcsec
                self.LOS    =  (self.z_sky.value / self.dist.value) *u.arcsec
                self.r_proj =  (self.r_sky.value / self.dist.value) *u.arcsec

                self.v_RA   = -self.v_x_sky # TODO: check this one
                self.v_Dec  =  self.v_y_sky
                self.v_LOS  =  self.v_z_sky + self.v_sys

            else:
                print("Distance `dist` and systemic velocity `v_sys` were not provided.")
                self.dist, self.v_sys = np.nan, np.nan
                self.RA, self.Dec, self.LOS, self.r_proj, self.v_RA, self.v_Dec, self.v_LOS \
                = [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]

        else:
            print("Disk inclination `inc` and position angle `pa` were not provided.")
            self.inc, self.pa = np.nan, np.nan
            self.x_sky, self.y_sky, self.z_sky, self.r_sky, self.v_x_sky, self.v_y_sky, self.v_z_sky \
            = [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
            # print("Distance `dist` and systemic velocity `v_sys` may have been provided, but will not be used.")
            self.dist, self.v_sys = np.nan, np.nan
            self.RA, self.Dec, self.LOS, self.r_proj, self.v_RA, self.v_Dec, self.v_LOS \
            = [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]

    def __str__(self, frame=''):
        """
        Returns/prints the properties of the streamline.
        """
        string = '\n-------------------------------'
        string += '\nStreamline system properties:'
        string += '\n>> (mass) Central mass: '+str(self.mass)
        string += '\n>> (omega) Angular frequency: '+str(self.omega)
        string += '\n>> (r0) Initial radius: '+str(self.r0)
        string += '\n>> (theta0) Initial polar angle: '+str(self.theta0)
        string += '\n>> (phi0) Initial azimuthal angle: '+str(self.phi0)
        string += '\n>> (v_r0) Initial radial velocity: '+str(self.v_r0)
        string += '\n>> (r_cent) Centrifugal radius: '+str(self.r_cent)
        string += '\n>> (rmin) Minimum radius: '+str(self.rmin)
        string += '\n>> (delta_r) Radial increments: '+str(self.delta_r)
        string += '\n-------------------------------'
        string += '\nDisk frame coordinates (Cartesian):'
        string += '\n>> (x) Initial x-position: '+str(self.x[0])
        string += '\n>> (y) Initial y-position: '+str(self.y[0])
        string += '\n>> (z) Initial z-position: '+str(self.z[0])
        string += '\n>> (v_x) Initial x-velocity: '+str(self.v_x[0])
        string += '\n>> (v_y) Initial y-velocity: '+str(self.v_y[0])
        string += '\n>> (v_z) Initial z-velocity: '+str(self.v_z[0])
        string += '\n-------------------------------'
        string += '\nDisk frame coordinates (Polar):'
        string += '\n>> (r) Initial r-position: '+str(self.r[0])
        string += '\n>> (theta) Initial theta-position: '+str(self.theta[0])
        string += '\n>> (theta) Initial theta-position in [degrees]: '+str(self.theta[0].to(u.deg))
        string += '\n>> (phi) Initial phi-position: '+str(self.phi[0])
        string += '\n>> (v_r) Initial r-velocity: '+str(self.v_r[0])
        string += '\n>> (v_theta) Initial theta-velocity: '+str(self.v_theta[0])
        string += '\n>> (v_phi) Initial phi-velocity: '+str(self.v_phi[0])
        string += '\n-------------------------------'
        string += '\nSky frame coordinates (Cartesian):'
        string += '\n>> (inc) Disk inclination: '+str(self.inc)
        string += '\n>> (pa) Disk position angle: '+str(self.pa)
        string += '\n>> (x_sky) Initial x-position on sky: '+str(self.x_sky[0])
        string += '\n>> (y_sky) Initial y-position on sky: '+str(self.y_sky[0])
        string += '\n>> (z_sky) Initial z-position on sky: '+str(self.z_sky[0])
        string += '\n>> (r_sky) Initial r-position on sky: '+str(self.r_sky[0])
        string += '\n>> (v_x_sky) Initial x-velocity on sky: '+str(self.v_x_sky[0])
        string += '\n>> (v_y_sky) Initial y-velocity on sky: '+str(self.v_y_sky[0])
        string += '\n>> (v_z_sky) Initial z-velocity on sky: '+str(self.v_z_sky[0])
        string += '\n-------------------------------'
        string += '\nSky frame coordinates (Angular):'
        string += '\n>> (dist) Distance to the system: '+str(self.dist)
        string += '\n>> (v_sys) Systemic velocity: '+str(self.v_sys)
        string += '\n>> (RA) Initial RA-position on sky: '+str(self.RA[0])
        string += '\n>> (Dec) Initial Dec-position on sky: '+str(self.Dec[0])
        string += '\n>> (LOS) Initial LOS-position on sky: '+str(self.LOS[0])
        string += '\n>> (r_proj) Initial R-position on sky: '+str(self.r_proj[0])
        string += '\n>> (v_RA) Initial RA-velocity on sky: '+str(self.v_RA[0])
        string += '\n>> (v_Dec) Initial Dec-velocity on sky: '+str(self.v_Dec[0])
        string += '\n>> (v_LOS) Initial LOS-velocity on sky: '+str(self.v_LOS[0])
        string += '\n-------------------------------\n'

        return string


    def _get_vk(self, radius, mass=0.5 * u.Msun):
        """
        Velocity term that is repeated in all velocity component.
        It corresponds to v_k in Mendoza+(2009)
        """
        return np.sqrt(G * mass / radius).to(u.km / u.s)


    def _get_r_cent(self, mass=0.5 * u.Msun, omega=1e-14 / u.s, r0=1e4 * u.au):
        """
        Centrifugal radius or disk radius in the Ulrich (1976)'s model.
        r_u in Mendoza's nomenclature.

        :param mass: Central mass for the protostar
        :param omega: Angular speed at the r0 radius
        :param r0: Initial radius of the streamline
        :return:
        """
        return (r0 ** 4 * omega ** 2 / (G * mass)).to(u.au)


    def _get_dphi(self, theta, theta0=np.radians(30)):
        """
        Gets the difference in Phi.

        :param theta: radians
        :param theta0: radians

        :return difference in Phi angle, in radians
        """
        # print('np.tan(theta0) / np.tan(theta): ', np.tan(theta0) / np.tan(theta))
        ratio = np.tan(theta0) / np.tan(theta)
        if np.any((ratio < -1) | (ratio > 1)):
            print('WARNING: Polar angle is going to have trouble with arccos.')
        return np.arccos(np.tan(theta0) / np.tan(theta))
    # def _get_dphi(self, theta, theta0=np.radians(30)):
    #     """
    #     Gets the difference in Phi.
    #
    #     :param theta: radians
    #     :param theta0: radians
    #
    #     :return difference in Phi angle, in radians
    #     """
    #     return np.arctan2(np.sin(theta - theta0), np.cos(theta - theta0))


    def _calculate_streamline_trajectory(self, r, mass=0.5 * u.Msun, r0=1e4 * u.au, theta0=30 * u.deg,
                    omega=1e-14 / u.s, v_r0=0 * u.km / u.s):
        """
        It calculates the stream line following Mendoza et al. (2009)
        It takes the radial velocity and rotation at the streamline
        initial radius and it describes the entire trajectory.

        :param r:
        :param mass:
        :param r0:
        :param theta0:
        :param phi0:
        :param omega:
        :param v_r0: Initial radial velocity
        :return: theta
        """

        def theta_abs(theta, r_to_rc=0.1, theta0=np.radians(30), ecc=1.,
                      orb_ang=90*u.deg):
            """
            function to determine theta numerically by finding the root of a function
            This is equation (9) in Mendoza+(2009)

            :param theta: angle in radians
            :param r_to_rc: radius in units of the centrifugal radius
            :param theta0: Initial angle of the streamline
            :param ecc: eccentricity of the orbit (equation 6)
            :param orb_ang: angle in the orbital motion (equation 7)
            :return: returns the difference between the radius and the predicted one,
                   a value of 0 corresponds to a proper streamline
            """
            cos_ratio = np.cos(theta) / np.cos(theta0)
            if cos_ratio > 1.:
                print('WARNING: Bad arcos calculation. theta0={0}, theta_try={1}, ratio={2}'.format(theta0, theta, cos_ratio))
                return np.nan
            xi = np.arccos(cos_ratio) + orb_ang.to(u.rad).value
            geom = np.sin(theta0)**2 / (1 - ecc * np.cos(xi))
            return np.abs(r_to_rc - geom)

        # Convert theta0 into radians
        theta0_rad = self.theta0.to(u.rad).value

        # Initialize the trajectory
        theta = np.zeros_like(r.value) + np.nan
        # The first element in the streamline is the starting point
        theta[0] = theta0_rad

        # Dimensionless parameters mu and nu; Eqn. 3 of Mendoza et al. (2009)
        # mu represents the ratio of Ulrich's disc radius (r_d=r_u) to the original cloud’s radius
        mu = (self.r_cent / self.r0).decompose().value
        # nu represents the initial amount of radial velocity measured in units of the Keplerian velocity at position r_u (=r_cent)
        nu = (self.v_r0 * np.sqrt(self.r_cent / (G * self.mass))).decompose().value
        self.mu, self.nu = mu, nu

        # Jess: p represents distance from the star at which the particle hits the midplane?
        p = self.r_cent * (np.sin(self.theta0)**2.)

        # Specific energy in dimensionless form; Eqn. 4 of Mendoza et al. (2009)
        epsilon = nu**2. + (mu**2. * np.sin(self.theta0)**2.) - (2. * mu)
        self.epsilon = epsilon

        # Eccentricity of the orbit; Eqn. 6 of Mendoza et al. (2009)
        self.ecc = np.sqrt(1 + epsilon * np.sin(self.theta0)**2.)

        # Condition on the initial azimuthal angle; Eqn. 7 of Mendoza et al. (2009)
        arccos_argument = np.clip((1. - (mu * np.sin(self.theta0)**2.)) / self.ecc, -1, 1) # Jess
        # orb_ang = np.arccos((1. - (mu * np.sin(theta0)**2.)) / self.ecc)
        orb_ang = np.arccos(arccos_argument) # Jess
        # print('***This should be between -1 and 1: ', (1. - (mu * np.sin(self.theta0)**2.)) / self.ecc)
        # print('Initial orb_ang: ', orb_ang.to(u.deg))

        # Prepare to set the initial guess at the next theta value
            # tol = (6.e-12 * self.delta_r * self.omega / (self.v_r0+ (0.1 * u.km/u.s))).decompose().value # in radians
            # initguess = 2. * tol # in radians
        tol = (6.e-5 * self.delta_r * self.omega / (self.v_r0+ (0.1 * u.km/u.s))).decompose().value # in radians
        initguess = 10. * tol # in radians
        # In Jaime's original implementation, tol = 6.e-5 * [same business], for an epsilon of 0.01 km/s
        # In Jaime's original implementation, initguess = 10. * tol
        print('Computing solution... Tolerance: ', tol)

        # Set the initial guess at the next theta value
        if theta0_rad < np.radians(90): # The streamline starts "above" the midplane (z>0)
            theta_i = theta0_rad + initguess # Guess theta goes downwards
            theta_bracket = [(theta0_rad, np.pi/2.)] # Cannot pass the midplane
        else: # The streamline starts "below" the midplane (z<0)
            theta_i = theta0_rad - initguess # Guess theta goes upwards
            theta_bracket = [(np.pi/2., theta0_rad)] # Cannot pass the midplane

        # Iterate through the radial array and solve for theta
        for ind in np.arange(1, len(r)):

            # Find the current radius in units of r_cent
            r_i = (r[ind] / self.r_cent).decompose().value

            # While the current radius is outside the set minimum
            if r_i > (self.rmin / self.r_cent).decompose().value:
                # print('initial guess of theta_i = {0}'.format(theta_i))
                # result = optimize.minimize(theta_abs, theta_i,
                #                            bounds=theta_bracket,
                #                            args=(r_i, theta0_rad, ecc, orb_ang))
                # By default, when minimize receives bounds and no constrains,
                # it uses the L-BFGS-B method:
                # ftol is the tolerance in the function evaluation
                # "The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol"
                # gtol corresponds to the parameter pgtol in fmin_l_bfgs_b
                # "The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol"
                # eps corresponds to the absolute step size used for numerical approximation of the jacobian via forward differences.
                options_dict = {'gtol': tol/10., 'eps': tol, 'ftol': tol}
                result = optimize.minimize(theta_abs, theta_i,
                                           bounds=theta_bracket,
                                           args=(r_i, theta0_rad, self.ecc, orb_ang),
                                           options=options_dict)
                theta_i = result.x
                # These prints are to diagnose if the minimization is converging
                # print(ind, result.success)
                # print(result.message, result.status, result.nit)
                theta[ind] = theta_i
        return theta * u.rad


    def _calculate_streamline_velocity(self, r, theta, mass=0.5*u.Msun, r0=1e4*u.au, theta0=30*u.deg,
                    omega=1e-14/u.s, v_r0=0*u.km/u.s):
        """
        It calculates the velocity along the stream line following Mendoza+(2009)
        It takes the radial velocity and rotation at the streamline
        initial radius and it describes the entire trajectory.

        :param theta:
        :param r:
        :param mass:
        :param r0:
        :param theta0:
        :param phi0:
        :param omega:
        :param v_r0: Initial radial velocity
        :return: v_r, v_theta, v_phi in units of km/s
        """
        rc = self.r_cent #r_cent(mass=mass, omega=omega, r0=r0)
        r_to_rc = (r / rc).decompose().value
        v_k0 = self._get_vk(rc, mass=mass)
        # mu and nu are dimensionless
        mu = (rc / r0).decompose().value
        nu = (v_r0 * np.sqrt(rc / (G * mass))).decompose().value
        epsilon = nu**2 + mu**2 * np.sin(theta0)**2 - 2 * mu
        ecc = np.sqrt(1 + epsilon*np.sin(theta0)**2)
        orb_ang = np.arccos((1 - mu * np.sin(theta0)**2) / ecc)
        cos_ratio = np.cos(theta) / np.cos(theta0)
        xi = np.arccos(cos_ratio) + orb_ang.to(u.rad)#.value
        #
        v_r_all = -ecc * np.sin(theta0) * np.sin(xi) / r_to_rc /(1 - ecc*np.cos(xi))
        v_theta_all = np.sin(theta0) / np.sin(theta) / r_to_rc \
                      * np.sqrt(np.cos(theta0)**2 - np.cos(theta)**2)
        v_phi_all = np.sin(theta0)**2 / np.sin(theta) / r_to_rc

        return v_r_all * v_k0, v_theta_all * v_k0, v_phi_all * v_k0

    @staticmethod
    def rotate_xyz(x, y, z, inc=30 * u.deg, pa=30 * u.deg):
        """
        Rotate on inclination and PA
        x-axis and y-axis are on the plane on the sky,
        z-axis is the

        Rotation around x is inclination angle
        Rotation around y is PA angle

        Using example matrices as described in:
        https://en.wikipedia.org/wiki/3D_projection

        :param x: cartesian x-coordinate, in the direction of decreasing RA
        :param y: cartesian y-coordinate, in the direction away of the observer
        :param z: cartesian z-coordinate, in the direction of increasing Dec.
        :param inc: Inclination angle. 0=no change
        :param pa: Change the PA angle. Measured from North due East.
        :return: new x, y, and z-coordinates as observed on the sky, with the
        same units as the input ones.

        """
        xyz = np.column_stack((x, y, z))
        rot_inc = np.array([[1, 0, 0],
                            [0, np.cos(inc), np.sin(inc)],
                            [0, -np.sin(inc), np.cos(inc)]])
        rot_pa = np.array([[np.cos(pa), 0, -np.sin(pa)],
                           [0, 1, 0],
                           [np.sin(pa), 0, np.cos(pa)]])
        xyz_new = rot_pa.dot(rot_inc.dot(xyz.T))
        return xyz_new[0], xyz_new[1], xyz_new[2]


    def _get_streamline(self, mass=0.5*u.Msun, r0=1e4*u.au, theta0=30*u.deg,
                   phi0=15*u.deg, omega=1e-14/u.s, v_r0=0*u.km/u.s,
                   rmin=None, delta_r=1*u.au):
                   # Missing: inc=0*u.deg, pa=0*u.deg,
        """
        it gets xyz coordinates and velocities for a stream line.

        Spherical into cartesian transformation is done for position and velocity
        using:
        https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

        :param mass: Central mass
        :param r0: Initial radius of streamline
        :param theta0: Initial polar angle of streamline
        :param phi0: Initial azimuthal angle of streamline
        :param omega: Angular rotation. (defined positive)
        :param v_r0: Initial radial velocity of the streamline
        :param rmin: smallest radius for calculation [jess: must be smaller than r0]
        :param delta_r: spacing between two consecutive radii in the sampling of the streamer, in au
        :return:
        """
        #
        if self.r_cent > self.r0:
            print('WARNING: Centrifugal radius is larger than initial radius of streamline.')

        #
        r = np.arange(self.r0.to(u.au).value, self.rmin.to(u.au).value, \
                      step=-1*self.delta_r.to(u.au).value) * u.au

        theta = self._calculate_streamline_trajectory(r, mass=mass, r0=r0, theta0=theta0,
                            omega=omega, v_r0=v_r0)
        d_phi = self._get_dphi(theta, theta0=theta0)
        phi = phi0 + d_phi
        #
        v_r, v_theta, v_phi = self._calculate_streamline_velocity(r, theta, mass=mass, r0=r0,
                                              theta0=theta0, omega=omega, v_r0=v_r0)
        v_x = v_r * np.sin(theta) * np.cos(phi) \
              + v_theta * np.cos(theta) * np.cos(phi) \
              - v_phi * np.sin(phi)
        v_y = v_r * np.sin(theta) * np.sin(phi) \
              + v_theta * np.cos(theta) * np.sin(phi) \
              + v_phi * np.cos(phi)
        v_z = v_r * np.cos(theta) \
              - v_theta * np.sin(theta)
        # Convert from spherical into cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Does the streamline hit the midplane before rmin?
        idxs = (np.abs(z) > 1.*u.au)
        # idx = np.argmax(z<1e0*u.au)+1
        # if idx!=1:
        if idxs.sum() > 0:
            # If the answer is yes, don't return those non-sensical values
            print('NOTE: The streamline hit the midplane before reaching ``rmin``.')
            return x[idxs], y[idxs], z[idxs], v_x[idxs], v_y[idxs], v_z[idxs], \
                   r[idxs], theta[idxs], phi[idxs], v_r[idxs], v_theta[idxs], v_phi[idxs]
            # return x[:idx], y[:idx], z[:idx], v_x[:idx], v_y[:idx], v_z[:idx], \
            #        r[:idx], theta[:idx], phi[:idx], v_r[:idx], v_theta[:idx], v_phi[:idx]
        else:
            return x, y, z, v_x, v_y, v_z, r, theta, phi, v_r, v_theta, v_phi

    def get_midplane_ring(self, coord_type='Cartesian', r=None, \
                           inc=None, pa=None):
        """
        Calculate the sky frame coordinates of a ring that lies in the disk
        midplane. This is helpful for visualization of the streamer geometry.

        Args:
            coord_type (Optional[str]): Either 'Cartesian' or 'Angular'. If
                'Cartesian', the returned coordinates will be in the Cartesian
                sky frame. If 'Angular', the returned coordinates will be in the
                angular sky frame. Defaults to Cartesian.
            r (Optional[float]): The radius of the ring [au], in disk frame
                coordinates. If not provided, the radius will be the initial
                radius of the streamline, ``r0``.
            inc (Optional[float]): The inclination of the ring [deg]. Defaults
                to that of the streamline, ``inc``.
            pa (Optional[float]): The position angle of the ring [deg]. Defaults
                to that of the streamline, ``pa``.

        Returns:
            x_sky (1D array): x-coordinates of the ring (Cartesian sky frame).
            y_sky (1D array): y-coordinates of the ring (Cartesian sky frame).
            z_sky (1D array): z-coordinates of the ring (Cartesian sky frame).
            RA (1D array): x-coordinates of the ring (angular sky frame).
            Dec (1D array): y-coordinates of the ring (angular sky frame).
            LOS (1D array): z-coordinates of the ring (angular sky frame).
        """
        r = self.r0 if r is None else r
        inc = self.inc if inc is None else inc
        pa = self.pa if pa is None else pa

        r_ring    = np.zeros(10000) + r.value
        t_ring    = np.linspace(0, 2*np.pi, r_ring.size)
        x_ring    = r_ring * np.sin(t_ring) * u.au
        y_ring    = r_ring * np.cos(t_ring) * u.au
        z_ring    = np.zeros(r_ring.size)   * u.au # Lies in the midplane

        (x_sky, z_sky, y_sky) \
        = streamline.rotate_xyz(x_ring, y_ring, z_ring, inc=inc, pa=pa)

        if coord_type=='Cartesian':
            return x_sky, y_sky, z_sky
        elif coord_type=='Angular':
            RA     = (-x_sky.value / self.dist.value) *u.arcsec
            Dec    =  (y_sky.value / self.dist.value) *u.arcsec
            LOS    =  (z_sky.value / self.dist.value) *u.arcsec
            return RA, Dec, LOS
        else:
            print("Supported options for ``coord_type`` are 'Cartesian' or 'Angular'.")



    ######################### Plotting functions #########################

    def plot_xyz_diskframe(self, ax=None, elev=30, azim=45):
        """
        Plot the streamer in xyz coordinates (the disk frame), 3D view.
        """

        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d

        # Generate the axes.
        if ax is None:
            fig     = plt.figure(figsize=(7.09, 7.09))
            gs      = fig.add_gridspec(ncols=1, nrows=1, width_ratios=[1], height_ratios=[1])
            ax      = fig.add_subplot(gs[0,0], projection='3d')

        r_max = self.r0.value
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-1.0*r_max, 1.0*r_max)
        ax.set_ylim(-1.0*r_max, 1.0*r_max)
        ax.set_zlim(-1.0*r_max, 1.0*r_max)
        ax.set_xlabel('x (au)')
        ax.set_ylabel('y (au)')
        ax.set_zlabel('z (au)')
        ax.plot([-1.0*r_max, 1.0*r_max],[0, 0],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
        ax.plot([0, 0],[-1.0*r_max, 1.0*r_max],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
        ax.plot([0, 0],[0, 0],[-1.0*r_max, 1.0*r_max], color='k', lw=1) # plot xyz axes for visual clarity
        ax.grid(alpha=0.5, color='grey')
        ax.scatter([0],[0],[0], marker='*', color='k', s=50, zorder=10000)
        ax.scatter([0],[0],[0], marker='*', color='w', s=25, zorder=10000)

        # Plot the xy plane for visual clarity
        xx, yy = np.meshgrid(np.arange(-1.0*r_max, 1.0*r_max, 1), np.arange(-1.0*r_max, 1.0*r_max, 1))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='grey')

        ax.plot(self.x, self.y, self.z, color='orange', lw=2)
        ax.scatter(self.x[0], self.y[0], self.z[0], color='r')
        ax.scatter(self.x[-1], self.y[-1], self.z[-1], color='b')

        # Plot the projection of the streamline onto the disk plane
        ax.plot(self.x, self.y, np.zeros_like(self.x.value), color='orange', lw=2, linestyle='dotted')

        ######### Plot the centrifugal radius #########
        p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ############## Plot the "cloud" (sphere) for visual clarity ###############
        # Create a meshgrid for spherical coordinates
        phi         = np.linspace(0, 2 * np.pi, 100)
        theta       = np.linspace(0, np.pi, 50)
        phi, theta  = np.meshgrid(phi, theta)
        # Convert from spherical coordinates to cartesian
        x = self.r0 * np.sin(theta) * np.cos(phi)
        y = self.r0 * np.sin(theta) * np.sin(phi)
        z = self.r0 * np.cos(theta)
        ax.plot_surface(x, y, z, alpha=0.075, color='gainsboro', edgecolor='k', lw=0.5, rstride=3, cstride=3)
        p = Circle((0, 0), self.r0.value, color='k', fill=False, lw=1, alpha=1) # Sphere's "equator"
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ax.view_init(elev=elev, azim=azim)

        return #ax

    def plot_xyz_diskframe_skyframe(self, ax=None, elev=30, azim=45):
        """
        Plot the streamer in xyz coordinates (the disk frame), and a second time
        from the observer point of view.
        """
        if self.inc == np.nan:
            if self.pa == np.nan:
                raise ValueError("Cannot use this function if streamline was not initialized with ``inc`` or ``pa``.")

        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d

        # Generate the axes.
        fig     = plt.figure(figsize=(7.09*2, 7.09))
        gs      = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 1], height_ratios=[1])
        ax1      = fig.add_subplot(gs[0,0], projection='3d')
        ax2      = fig.add_subplot(gs[0,1], projection='3d')

        r_max = self.r0.value
        for ax in [ax1, ax2]:
            ax.set_box_aspect([1,1,1])
            ax.set_xlim(-1.0*r_max, 1.0*r_max)
            ax.set_ylim(-1.0*r_max, 1.0*r_max)
            ax.set_zlim(-1.0*r_max, 1.0*r_max)
            ax.plot([-1.0*r_max, 1.0*r_max],[0, 0],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
            ax.plot([0, 0],[-1.0*r_max, 1.0*r_max],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
            ax.plot([0, 0],[0, 0],[-1.0*r_max, 1.0*r_max], color='k', lw=1) # plot xyz axes for visual clarity
            ax.grid(alpha=0.5, color='grey')
            ax.scatter([0],[0],[0], marker='*', color='k', s=50, zorder=10000)
            ax.scatter([0],[0],[0], marker='*', color='w', s=25, zorder=10000)
            # Plot the xy plane for visual clarity
            xx, yy = np.meshgrid(np.arange(-1.0*r_max, 1.0*r_max, 1), np.arange(-1.0*r_max, 1.0*r_max, 1))
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='grey')
        ax1.set_xlabel('x (au)')
        ax1.set_ylabel('y (au)')
        ax1.set_zlabel('z (au)')
        ax2.set_xlabel(r'$x_{\rm sky}$ (au)')
        ax2.set_ylabel(r'$y_{\rm sky}$ (au)')
        ax2.set_zlabel(r'$z_{\rm sky}$ (au)')

        ax1.plot(self.x, self.y, self.z, color='orange', lw=2)
        ax1.scatter(self.x[0], self.y[0], self.z[0], color='r')
        ax1.scatter(self.x[-1], self.y[-1], self.z[-1], color='b')
        ax1.plot(self.x, self.y, np.zeros_like(self.x.value), color='orange', lw=1) # projection onto disk plane

        ax2.plot(self.x_sky, self.y_sky, self.z_sky, color='orange', lw=2)
        ax2.scatter(self.x_sky[0], self.y_sky[0], self.z_sky[0], color='r')
        ax2.scatter(self.x_sky[-1], self.y_sky[-1], self.z_sky[-1], color='b')

        # Plot the projection of the streamline onto the disk plane
        ax2.plot(self.x_sky, self.y_sky, np.zeros_like(self.x_sky.value), color='orange', lw=2, linestyle='dotted')


        ######### Plot the centrifugal radius #########
        p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        ax1.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        ax2.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ############## Plot the "cloud" (sphere) for visual clarity ###############
        # Create a meshgrid for spherical coordinates
        phi         = np.linspace(0, 2 * np.pi, 100)
        theta       = np.linspace(0, np.pi, 50)
        phi, theta  = np.meshgrid(phi, theta)
        # Convert from spherical coordinates to cartesian
        x = self.r0 * np.sin(theta) * np.cos(phi)
        y = self.r0 * np.sin(theta) * np.sin(phi)
        z = self.r0 * np.cos(theta)
        ax1.plot_surface(x, y, z, alpha=0.075, color='gainsboro', edgecolor='k', lw=0.5, rstride=3, cstride=3)
        ax2.plot_surface(x, y, z, alpha=0.075, color='gainsboro', edgecolor='k', lw=0.5, rstride=3, cstride=3)
        p = Circle((0, 0), self.r0.value, color='k', fill=False, lw=1, alpha=1) # Sphere's "equator"
        ax1.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        p = Circle((0, 0), self.r0.value, color='k', fill=False, lw=1, alpha=1) # Sphere's "equator"
        ax2.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # ax1.view_init(elev=self.inc.value, azim=self.pa.value-90)
        ax1.set_title("Disk frame")
        ax1.view_init(elev=elev, azim=azim)
        ax2.set_title("Sky (observer's) frame")
        ax2.view_init(elev=elev, azim=azim)

        return #ax

    def plot_coordinate_transformations(self, ax=None, elev=30, azim=45):
        """
        Plot three panels: (1) The streamer in xyz coordinates (the disk frame),
        which is the same as plot_xyz_diskframe(); (2) The projected sky frame
        Cartesian coordinates and line-of-sight velocity; (3) The projected sky
        frame angular coordinates and line-of-sight velocity. This provides a
        view of all three coordinate systems.
        """
        if self.dist == np.nan:
            if self.v_sys == np.nan:
                raise ValueError("Cannot use this function if streamline was not initialized with ``dist`` or ``v_sys``.")

        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d

        # Generate the axes.
        fig     = plt.figure(figsize=(7.09*1.5, 7.09*0.5))
        gs      = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1], height_ratios=[1])
        ax1      = fig.add_subplot(gs[0,0], projection='3d')
        ax2      = fig.add_subplot(gs[0,1])
        ax3      = fig.add_subplot(gs[0,2])

        r_max = self.r0.value
        for ax in [ax1]:
            ax.set_box_aspect([1,1,1])
            ax.set_xlim(-1.0*r_max, 1.0*r_max)
            ax.set_ylim(-1.0*r_max, 1.0*r_max)
            ax.set_zlim(-1.0*r_max, 1.0*r_max)
            ax1.set_xlabel('x (au)')
            ax1.set_ylabel('y (au)')
            ax1.set_zlabel('z (au)')
            ax.plot([-1.0*r_max, 1.0*r_max],[0, 0],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
            ax.plot([0, 0],[-1.0*r_max, 1.0*r_max],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
            ax.plot([0, 0],[0, 0],[-1.0*r_max, 1.0*r_max], color='k', lw=1) # plot xyz axes for visual clarity
            ax.grid(alpha=0.5, color='grey')
            ax.scatter([0],[0],[0], marker='*', color='k', s=50, zorder=10000)
            ax.scatter([0],[0],[0], marker='*', color='w', s=25, zorder=10000)
            # Plot the xy plane for visual clarity
            xx, yy = np.meshgrid(np.arange(-1.0*r_max, 1.0*r_max, 1), np.arange(-1.0*r_max, 1.0*r_max, 1))
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='grey')
        for ax in [ax2]:
            ax.set_xlim(-1.0*r_max, 1.0*r_max)
            ax.set_ylim(-1.0*r_max, 1.0*r_max)
        for ax in [ax3]:
            ax.set_xlim(-1.0*r_max/self.dist.value, 1.0*r_max/self.dist.value)
            ax.set_ylim(-1.0*r_max/self.dist.value, 1.0*r_max/self.dist.value)
            ax.invert_xaxis()
        for ax in [ax2, ax3]:
            ax.scatter([0],[0], marker='*', color='k', s=50, zorder=10000)
            ax.scatter([0],[0], marker='*', color='w', s=25, zorder=10000)
            ax.set_aspect('equal')

        ax2.set_xlabel(r'$x_{\rm sky}$ (au)')
        ax2.set_ylabel(r'$y_{\rm sky}$ (au)')
        ax3.set_xlabel(r'RA offset (arcsec)')
        ax3.set_ylabel(r'Dec offset (arcsec)')

        ax1.plot(self.x, self.y, self.z, color='orange', lw=2)
        ax1.scatter(self.x[0], self.y[0], self.z[0], color='r')
        ax1.scatter(self.x[-1], self.y[-1], self.z[-1], color='b')
        ax1.plot(self.x, self.y, np.zeros_like(self.x.value), color='orange', lw=1) # projection onto disk plane

        ax2.scatter(self.x_sky[0], self.y_sky[0], color='r')
        ax2.scatter(self.x_sky[-1], self.y_sky[-1], color='b')
        ax2.plot(self.x_sky, self.y_sky, color='orange', lw=2)

        ax3.scatter(self.RA[0], self.Dec[0], color='r')
        ax3.scatter(self.RA[-1], self.Dec[-1], color='b')
        ax3.plot(self.RA, self.Dec, color='orange', lw=2)

        ######### Plot the centrifugal radius #########
        p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        ax1.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        # TODO: INCLINE AND ROTATE THESE CIRCLES TO ELLIPSES ACCORDING TO INC, PA
        # p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        # ax2.add_patch(p)
        # p = Circle((0, 0), self.r_cent.value/self.dist.value, color='purple', fill=False, lw=1, linestyle='dashed')
        # ax3.add_patch(p)

        ############## Plot the "cloud" (sphere) for visual clarity ###############
        # Create a meshgrid for spherical coordinates
        phi         = np.linspace(0, 2 * np.pi, 100)
        theta       = np.linspace(0, np.pi, 50)
        phi, theta  = np.meshgrid(phi, theta)
        # Convert from spherical coordinates to cartesian
        x = self.r0 * np.sin(theta) * np.cos(phi)
        y = self.r0 * np.sin(theta) * np.sin(phi)
        z = self.r0 * np.cos(theta)
        ax1.plot_surface(x, y, z, alpha=0.075, color='gainsboro', edgecolor='k', lw=0.5, rstride=3, cstride=3)
        p = Circle((0, 0), self.r0.value, color='k', fill=False, lw=1, alpha=1) # Sphere's "equator"
        ax1.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # ax1.view_init(elev=self.inc.value, azim=self.pa.value-90)
        ax1.set_title("Disk frame")
        ax1.view_init(elev=elev, azim=azim)
        ax2.set_title("Sky frame (Cartesian)")
        ax3.set_title("Sky frame (Angular)")

        plt.tight_layout()

        return #ax

    def plot_model(self, ax=None):
        """
        Plot the streamer on the sky (RA, Dec coordinates) and in PV space
        (line-of-sight velocity as a function of projected radial coordinate).
        """
        if self.inc == np.nan:
            if self.pa == np.nan:
                raise ValueError("Cannot use this function if streamline was not initialized with ``inc`` or ``pa``.")

        from matplotlib.patches import Circle

        # Generate the axes.
        fig     = plt.figure(figsize=(7.09*1.25, 7.09*0.5))
        gs      = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 1.5], height_ratios=[1])
        ax1      = fig.add_subplot(gs[0,0])
        ax2      = fig.add_subplot(gs[0,1])

        r_max = self.r0.value/self.dist.value
        ax1.set_xlabel('R.A. offset (arcsec)')
        ax1.set_ylabel('Dec. offset (arcsec)')
        ax1.scatter([0],[0], marker='*', color='k', s=50, zorder=10000)
        ax1.scatter([0],[0], marker='*', color='w', s=25, zorder=10000)
        ax1.set_xlim(-1.0*r_max, 1.0*r_max)
        ax1.set_ylim(-1.0*r_max, 1.0*r_max)
        ax1.set_aspect('equal')
        ax1.invert_xaxis()
        ax2.set_xlabel(r'Projected distance (au)')
        ax2.set_ylabel(r'$v_{\rm los}$ (km/s)')

        ax1.plot(self.RA, self.Dec, color='orange', lw=2)
        ax1.scatter(self.RA[0], self.Dec[0], color='r')
        ax1.scatter(self.RA[-1], self.Dec[-1], color='b')

        ax2.axhline(self.v_sys.value, color='k', linestyle='dashed')
        ax2.plot(self.r_sky, self.v_LOS, color='orange', lw=2)
        ax2.scatter(self.r_sky[0], self.v_LOS[0], color='r')
        ax2.scatter(self.r_sky[-1], self.v_LOS[-1], color='b')

        plt.tight_layout()
        return ax1, ax2


    def plot_xyz_sky(self, ax=None, elev=30, azim=45):
        """
        Plot the streamer in xyz_sky coordinates (the sky frame).
        """

        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d

        # Generate the axes.
        if ax is None:
            fig     = plt.figure(figsize=(7.09, 7.09))
            gs      = fig.add_gridspec(ncols=1, nrows=1, width_ratios=[1], height_ratios=[1])
            ax      = fig.add_subplot(gs[0,0], projection='3d')

        r_max = self.r0.value
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-1.0*r_max, 1.0*r_max)
        ax.set_ylim(-1.0*r_max, 1.0*r_max)
        ax.set_zlim(-1.0*r_max, 1.0*r_max)
        ax.set_xlabel(r'$x_{\rm sky}$ (au)')
        ax.set_ylabel(r'$y_{\rm sky}$ (au)')
        ax.set_zlabel(r'$z_{\rm sky}$ (au)')
        ax.plot([-1.0*r_max, 1.0*r_max],[0, 0],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
        ax.plot([0, 0],[-1.0*r_max, 1.0*r_max],[0, 0], color='k', lw=1) # plot xyz axes for visual clarity
        ax.plot([0, 0],[0, 0],[-1.0*r_max, 1.0*r_max], color='k', lw=1) # plot xyz axes for visual clarity
        ax.grid(alpha=0.5, color='grey')
        ax.scatter([0],[0],[0], marker='*', color='k', s=50, zorder=10000)
        ax.scatter([0],[0],[0], marker='*', color='w', s=25, zorder=10000)

        # Plot the xz plane (disk plane) for visual clarity
        xx, yy = np.meshgrid(np.arange(-1.0*r_max, 1.0*r_max, 1), np.arange(-1.0*r_max, 1.0*r_max, 1))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='grey')

        ax.plot(self.x_sky, self.y_sky, self.z_sky, color='orange', lw=2)
        ax.scatter(self.x_sky[0], self.y_sky[0], self.z_sky[0], color='r')
        ax.scatter(self.x_sky[-1], self.y_sky[-1], self.z_sky[-1], color='b')
        ax.plot(self.x_sky, self.y_sky, np.zeros_like(self.x_sky.value), color='orange', lw=1) # projection onto disk plane

        ######### Plot the centrifugal radius #########
        p = Circle((0, 0), self.r_cent.value, color='purple', fill=False, lw=1, linestyle='dashed')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ############## Plot the "cloud" (sphere) for visual clarity ###############
        # Create a meshgrid for spherical coordinates
        phi         = np.linspace(0, 2 * np.pi, 100)
        theta       = np.linspace(0, np.pi, 50)
        phi, theta  = np.meshgrid(phi, theta)
        # Convert from spherical coordinates to cartesian
        x = self.r0 * np.sin(theta) * np.cos(phi)
        y = self.r0 * np.sin(theta) * np.sin(phi)
        z = self.r0 * np.cos(theta)
        ax.plot_surface(x, y, z, alpha=0.075, color='gainsboro', edgecolor='k', lw=0.5, rstride=3, cstride=3)
        p = Circle((0, 0), self.r0.value, color='k', fill=False, lw=1, alpha=1) # Sphere's "equator"
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        ax.view_init(elev=elev, azim=azim)

        return #ax

    def plot_6panel_cartesian(self, ax=None):
        """
        Plot the streamer xyz coordinates and xyz velocities, as a function
        of the radial coordinate.
        """

        # Generate the axes.
        if ax is None:
            fig     = plt.figure(figsize=(7.09*2, 7.09))
            gs      = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], height_ratios=[1, 1])
            ax1      = fig.add_subplot(gs[0,0])
            ax2      = fig.add_subplot(gs[0,1])
            ax3      = fig.add_subplot(gs[0,2])
            ax4      = fig.add_subplot(gs[1,0])
            ax5      = fig.add_subplot(gs[1,1])
            ax6      = fig.add_subplot(gs[1,2])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]

        for ax in [ax4, ax5, ax6]:
            ax.set_xlim(0, self.r0.value)
            ax.set_xlabel('r [au]')
            ax.axhline(0, color='k', lw=1)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, self.r0.value)
            ax.set_ylim(-self.r0.value, self.r0.value)
            ax.axhline(0, color='k', lw=1)

        ax1.set_ylabel('x [au]')
        ax2.set_ylabel('y [au]')
        ax3.set_ylabel('z [au]')

        ax4.set_ylabel('v_x [km/s]')
        ax5.set_ylabel('v_y [km/s]')
        ax6.set_ylabel('v_z [km/s]')

        for i, curve in enumerate([self.x, self.y, self.z, self.v_x, self.v_y, self.v_z]):
            axes[i].axvline(self.r_cent.value, color='purple', lw=2, linestyle='dashed')
            axes[i].plot(self.r, curve, color='orange', lw=2)
            axes[i].scatter(self.r[0], curve[0], color='r', zorder=10)
            axes[i].scatter(self.r[-1], curve[-1], color='b', zorder=10)

        plt.tight_layout()

        return #ax

    def plot_6panel_polar(self, ax=None):
        """
        Plot the streamer r,theta,phi coordinates and r,theta,phi velocities, as
        a function of the radial coordinate.
        """

        # Generate the axes.
        if ax is None:
            fig     = plt.figure(figsize=(7.09*2, 7.09))
            gs      = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[1, 1, 1], height_ratios=[1, 1])
            ax1      = fig.add_subplot(gs[0,0])
            ax2      = fig.add_subplot(gs[0,1])
            ax3      = fig.add_subplot(gs[0,2])
            ax4      = fig.add_subplot(gs[1,0])
            ax5      = fig.add_subplot(gs[1,1])
            ax6      = fig.add_subplot(gs[1,2])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]

        for ax in [ax4, ax5, ax6]:
            ax.set_xlim(0, self.r0.value)
            ax.set_xlabel('r [au]')
            ax.axhline(0, color='k', lw=1)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, self.r0.value)
            ax.axhline(0, color='k', lw=1)

        ax1.set_ylabel('r [au]')
        ax1.set_ylim(-self.r0.value, self.r0.value)
        ax2.set_ylabel('theta [degrees]')
        ax2.set_ylim(0, 180)
        ax2.axhline(90, color='k', lw=1)
        ax3.set_ylabel('phi [degrees]')
        ax3.set_ylim(0, 360)

        ax4.set_ylabel('v_r [km/s]')
        ax5.set_ylabel('v_theta [km/s]')
        ax6.set_ylabel('v_phi [km/s]')

        for i, curve in enumerate([self.r, self.theta.to(u.deg), self.phi, self.v_r, self.v_theta, self.v_phi]):
            axes[i].axvline(self.r_cent.value, color='purple', lw=2, linestyle='dashed')
            axes[i].plot(self.r, curve, color='orange', lw=2)
            axes[i].scatter(self.r[0], curve[0], color='r', zorder=10)
            axes[i].scatter(self.r[-1], curve[-1], color='b', zorder=10)

        plt.tight_layout()

        return #ax

    def plot_disk_axes_sky(self, ax, coord_type='Cartesian', rmax=2., pa=0., inc=0.,
        color='lightgrey', lw=1.2, linestyle='dashed', alpha=1.0, zorder=1):
        """
        Plot the major and minor axes of the disk, projected on the sky. Can be
        either in the Cartesian sky frame (``x_sky``, ``y_sky``) or Angular
        sky frame (``RA``, ``Dec``).

        Args:
            ax (matplotib axis instance): Axis to plot the disk axis lines.
            coord_type (Optional[str]): Either 'Cartesian' or 'Angular'. If
                'Cartesian', the returned coordinates will be in the Cartesian
                sky frame. If 'Angular', the returned coordinates will be in the
                angular sky frame. Defaults to Cartesian.
            rmax ([float]): Disk-frame radius to end the disk axis lines. Must
                be in [au] (but does not need astropy units attached.)
            inc ([float]): Inclination of the disk in [degrees].
            pa ([float]): Position angle of the disk in [degrees],
                measured east-of-north towards the redshifted major axis.
        """

        major_axis_diskframe_x = ([-rmax, rmax])
        major_axis_diskframe_y = ([0, 0])
        minor_axis_diskframe_x = ([0, 0])
        minor_axis_diskframe_y = ([-rmax, rmax])

        major_axis_skyframe_x, _, major_axis_skyframe_y = \
        streamline.rotate_xyz(major_axis_diskframe_x, major_axis_diskframe_y, ([0,0]), inc=inc, pa=pa)
        minor_axis_skyframe_x, _, minor_axis_skyframe_y = \
        streamline.rotate_xyz(minor_axis_diskframe_x, minor_axis_diskframe_y, ([0,0]), inc=inc, pa=pa)

        if coord_type=='Cartesian':
            ax.plot(major_axis_skyframe_x, major_axis_skyframe_y, color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=zorder)
            ax.plot(minor_axis_skyframe_x, minor_axis_skyframe_y, color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=zorder)
        elif coord_type=='Angular':
            major_axis_skyframe_RA     = (-major_axis_skyframe_x / self.dist.value)
            major_axis_skyframe_Dec    =  (major_axis_skyframe_y / self.dist.value)
            minor_axis_skyframe_RA     = (-minor_axis_skyframe_x / self.dist.value)
            minor_axis_skyframe_Dec    =  (minor_axis_skyframe_y / self.dist.value)
            ax.plot(major_axis_skyframe_RA, major_axis_skyframe_Dec, color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=zorder)
            ax.plot(minor_axis_skyframe_RA, minor_axis_skyframe_Dec, color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=zorder)
        else:
            print("Supported options for ``coord_type`` are 'Cartesian' or 'Angular'.")
