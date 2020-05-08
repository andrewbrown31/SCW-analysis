from __future__ import (absolute_import, division, print_function)
    
import numpy as np

import wrapt

from .util import iter_left_indexes, to_np
from .py3compat import py3range
from .config import xarray_enabled
from .constants import default_fill

if xarray_enabled():
    from xarray import DataArray


def uvmet_left_iter(alg_dtype=np.float64):
    """A decorator to handle iterating over the leftmost dimensions for the
    uvmet diagnostic.

    For example, if a wrapped function works with three-dimensional arrays, but
    the variables include a 4th leftmost dimension for 'Time', this decorator
    will iterate over all times, call the 3D Fortran routine, and aggregate the
    results in to a 4D output array.

    It is also important to note that the final output array is allocated
    first, and then views are passed to the wrapped function so that values
    do not need to get copied in to the final output array.

    Args:

        alg_dtype (:class:`np.dtype` or :obj:`str`): The numpy data type used
            in the wrapped function.

    Returns:

        :class:`numpy.ndarray`: The aggregated uvmet output array that includes
        all extra leftmost dimensions.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        u = args[0]
        v = args[1]
        lat = args[2]
        lon = args[3]
        cen_long = args[4]
        cone = args[5]

        orig_dtype = u.dtype

        lat_lon_fixed = False
        if lat.ndim == 2:
            lat_lon_fixed = True

        if lon.ndim == 2 and not lat_lon_fixed:
            raise ValueError("'lat' and 'lon' shape mismatch")

        num_left_dims_u = u.ndim - 2
        num_left_dims_lat = lat.ndim - 2

        if (num_left_dims_lat > num_left_dims_u):
            raise ValueError("number of 'lat' dimensions is greater than 'u'")

        if lat_lon_fixed:
            mode = 0  # fixed lat/lon
        else:
            if num_left_dims_u == num_left_dims_lat:
                mode = 1  # lat/lon same as u
            else:
                mode = 2  # probably 3D with 2D lat/lon plus Time

        has_missing = False
        u_arr = to_np(u)

        v_arr = to_np(v)

        umissing = default_fill(np.float64)
        if isinstance(u_arr, np.ma.MaskedArray):
            has_missing = True
            umissing = u_arr.fill_value

        vmissing = default_fill(np.float64)
        if isinstance(v_arr, np.ma.MaskedArray):
            has_missing = True
            vmissing = v_arr.fill_value

        uvmetmissing = umissing

        is_stag = 0
        if (u.shape[-1] != lat.shape[-1] or u.shape[-2] != lat.shape[-2]):
            is_stag = 1
            # Sanity check
            if (v.shape[-1] == lat.shape[-1] or v.shape[-2] == lat.shape[-2]):
                raise ValueError("u is staggered but v is not")

        if (v.shape[-1] != lat.shape[-1] or v.shape[-2] != lat.shape[-2]):
            is_stag = 1
            # Sanity check
            if (u.shape[-1] == lat.shape[-1] or u.shape[-2] == lat.shape[-2]):
                raise ValueError("v is staggered but u is not")

        # No special left side iteration, return the function result
        if (num_left_dims_u == 0):
            return wrapped(u, v, lat, lon, cen_long, cone, isstag=is_stag,
                           has_missing=has_missing, umissing=umissing,
                           vmissing=vmissing, uvmetmissing=uvmetmissing)

        # Initial output is time,nz,2,ny,nx to create contiguous views
        outdims = u.shape[0:num_left_dims_u]
        extra_dims = tuple(outdims)  # Copy the left-most dims for iteration

        outdims += (2,)

        outdims += lat.shape[-2:]

        outview_array = np.empty(outdims, alg_dtype)

        # Final Output moves the u_v dimension to left side
        output_dims = (2,)
        output_dims += extra_dims
        output_dims += lat.shape[-2:]
        output = np.empty(output_dims, orig_dtype)

        for left_idxs in iter_left_indexes(extra_dims):
            left_and_slice_idxs = left_idxs + (slice(None),)

            if mode == 0:
                lat_left_and_slice = (slice(None),)
            elif mode == 1:
                lat_left_and_slice = left_and_slice_idxs
            elif mode == 2:
                # Only need the left-most
                lat_left_and_slice = tuple(
                    left_idx for left_idx in left_idxs[0:num_left_dims_lat])

            u_output_idxs = (0,) + left_idxs + (slice(None),)
            v_output_idxs = (1,) + left_idxs + (slice(None),)
            u_view_idxs = left_idxs + (0, slice(None))
            v_view_idxs = left_idxs + (1, slice(None))

            new_u = u[left_and_slice_idxs]
            new_v = v[left_and_slice_idxs]
            new_lat = lat[lat_left_and_slice]
            new_lon = lon[lat_left_and_slice]
            outview = outview_array[left_and_slice_idxs]

            # Skip the possible empty/missing arrays for the join method
            skip_missing = False
            for arg in (new_u, new_v, new_lat, new_lon):
                if isinstance(arg, np.ma.MaskedArray):
                    if arg.mask.all():
                        output[u_output_idxs] = uvmetmissing
                        output[v_output_idxs] = uvmetmissing

                        skip_missing = True
                        has_missing = True

            if skip_missing:
                continue

            # Call the numerical routine
            result = wrapped(new_u, new_v, new_lat, new_lon, cen_long, cone,
                             isstag=is_stag, has_missing=has_missing,
                             umissing=umissing, vmissing=vmissing,
                             uvmetmissing=uvmetmissing, outview=outview)

            # Make sure the result is the same data as what got passed in
            # Can delete this once everything works
            if (result.__array_interface__["data"][0] !=
                    outview.__array_interface__["data"][0]):
                raise RuntimeError("output array was copied")

            output[u_output_idxs] = (
                            outview_array[u_view_idxs].astype(orig_dtype))
            output[v_output_idxs] = (
                            outview_array[v_view_idxs].astype(orig_dtype))

        if has_missing:
            output = np.ma.masked_values(output, uvmetmissing)

        return output

    return func_wrapper


def cape_left_iter(alg_dtype=np.float64):
    """A decorator to handle iterating over the leftmost dimensions for the
    cape diagnostic.

    For example, if a wrapped function works with three-dimensional arrays, but
    the variables include a 4th leftmost dimension for 'Time', this decorator
    will iterate over all times, call the 3D Fortran routine, and aggregate the
    results in to a 4D output array.

    It is also important to note that the final output array is allocated
    first, and then views are passed to the wrapped function so that values
    do not need to get copied in to the final output array.

    Args:

        alg_dtype (:class:`np.dtype` or :obj:`str`): The numpy data type used
            in the wrapped function.

    Returns:

        :class:`numpy.ndarray`: The aggregated cape output array that includes
        all extra leftmost dimensions.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        # The cape calculations use an ascending vertical pressure coordinate

        new_args = list(args)
        new_kwargs = dict(kwargs)

        p_hpa = args[0]
        tk = args[1]
        qv = args[2]
        ht = args[3]
        ter = args[4]
        sfp = args[5]
        missing = args[6]
        i3dflag = args[7]
        ter_follow = args[8]

        is2d = i3dflag == 0
        # Note: This should still work with DataArrays
        is1d = np.isscalar(sfp) or np.size(sfp) == 1

        # Make sure sfp and terrain are regular floats for 1D case
        # This should also work with DataArrays
        if is1d:
            ter = float(ter)
            sfp = float(sfp)

        orig_dtype = p_hpa.dtype

        if not is1d:
            # Need to order in ascending pressure order
            flip = False
            bot_idxs = (0,) * p_hpa.ndim
            top_idxs = list(bot_idxs)
            top_idxs[-3] = -1
            top_idxs = tuple(top_idxs)

            if p_hpa[bot_idxs] > p_hpa[top_idxs]:
                flip = True
                p_hpa = np.ascontiguousarray(p_hpa[..., ::-1, :, :])
                tk = np.ascontiguousarray(tk[..., ::-1, :, :])
                qv = np.ascontiguousarray(qv[..., ::-1, :, :])
                ht = np.ascontiguousarray(ht[..., ::-1, :, :])
                new_args[0] = p_hpa
                new_args[1] = tk
                new_args[2] = qv
                new_args[3] = ht

            num_left_dims = p_hpa.ndim - 3
        else:
            # Need to order in ascending pressure order
            flip = False

            if p_hpa[0] > p_hpa[-1]:
                flip = True
                p_hpa = np.ascontiguousarray(p_hpa[::-1])
                tk = np.ascontiguousarray(tk[::-1])
                qv = np.ascontiguousarray(qv[::-1])
                ht = np.ascontiguousarray(ht[::-1])

            # Need to make 3D views for the fortran code.
            # Going to make these fortran ordered, since the f_contiguous and
            # c_contiguous flags are broken in numpy 1.11 (always false).  This
            # should work across all numpy versions.
            new_args[0] = p_hpa.reshape((1, 1, p_hpa.shape[0]), order='F')
            new_args[1] = tk.reshape((1, 1, tk.shape[0]), order='F')
            new_args[2] = qv.reshape((1, 1, qv.shape[0]), order='F')
            new_args[3] = ht.reshape((1, 1, ht.shape[0]), order='F')
            new_args[4] = np.full((1, 1), ter, orig_dtype)
            new_args[5] = np.full((1, 1), sfp, orig_dtype)

            num_left_dims = 0

        # No special left side iteration, build the output from the cape,cin
        # result
        if (num_left_dims == 0):
          try:
            cape, cin, lfc, lcl, el = wrapped(*new_args, **new_kwargs)

            output_dims = (6,)
            if not is1d:
                output_dims += p_hpa.shape[-3:]
            else:
                output_dims += (p_hpa.shape[0], 1, 1)

            output = np.empty(output_dims, orig_dtype)

            if flip and not is2d:
                output[0, :] = cape[::-1, :, :]
                output[1, :] = cin[::-1, :, :]
                output[2, :] = lfc[::-1, :, :]
                output[3, :] = lcl[::-1, :, :]
                output[4, :] = el[::-1, :, :]
            else:
                output[0, :] = cape[:]
                output[1, :] = cin[:]
                output[2, :] = lfc[:]
                output[3, :] = lfc[:]
                output[4, :] = el[:]

            return output
          except:
            cape, cin = wrapped(*new_args, **new_kwargs)

            output_dims = (2,)
            if not is1d:
                output_dims += p_hpa.shape[-3:]
            else:
                output_dims += (p_hpa.shape[0], 1, 1)

            output = np.empty(output_dims, orig_dtype)

            if flip and not is2d:
                output[0, :] = cape[::-1, :, :]
                output[1, :] = cin[::-1, :, :]
            else:
                output[0, :] = cape[:]
                output[1, :] = cin[:]

            return output

        # Initial output is ...,cape_cin,nz,ny,nx to create contiguous views
        outdims = p_hpa.shape[0:num_left_dims]
        extra_dims = tuple(outdims)  # Copy the left-most dims for iteration

        outdims += (2,)  # cape_cin

        outdims += p_hpa.shape[-3:]

        outview_array = np.empty(outdims, alg_dtype)

        # Create the output array where the leftmost dim is the product type
        output_dims = (2,)
        output_dims += extra_dims
        output_dims += p_hpa.shape[-3:]
        output = np.empty(output_dims, orig_dtype)

        for left_idxs in iter_left_indexes(extra_dims):
            left_and_slice_idxs = left_idxs + (slice(None),)
            cape_idxs = left_idxs + (0, slice(None))
            cin_idxs = left_idxs + (1, slice(None))

            cape_output_idxs = (0,) + left_idxs + (slice(None),)
            cin_output_idxs = (1,) + left_idxs + (slice(None),)
            view_cape_reverse_idxs = left_idxs + (0, slice(None, None, -1),
                                                  slice(None))
            view_cin_reverse_idxs = left_idxs + (1, slice(None, None, -1),
                                                 slice(None))

            new_args[0] = p_hpa[left_and_slice_idxs]
            new_args[1] = tk[left_and_slice_idxs]
            new_args[2] = qv[left_and_slice_idxs]
            new_args[3] = ht[left_and_slice_idxs]
            new_args[4] = ter[left_and_slice_idxs]
            new_args[5] = sfp[left_and_slice_idxs]
            capeview = outview_array[cape_idxs]
            cinview = outview_array[cin_idxs]

            # Skip the possible empty/missing arrays for the join method
            # Note: Masking handled by cape.py or computation.py, so only
            # supply the fill values here.
            skip_missing = False
            for arg in (new_args[0:6]):
                if isinstance(arg, np.ma.MaskedArray):
                    if arg.mask.all():
                        if flip and not is2d:
                            output[cape_output_idxs] = missing
                            output[cin_output_idxs] = missing
                        else:
                            output[cape_output_idxs] = missing
                            output[cin_output_idxs] = missing

                        skip_missing = True

            if skip_missing:
                continue

            # Call the numerical routine
            new_kwargs["capeview"] = capeview
            new_kwargs["cinview"] = cinview

            cape, cin = wrapped(*new_args, **new_kwargs)

            # Make sure the result is the same data as what got passed in
            # Can delete this once everything works
            if (cape.__array_interface__["data"][0] !=
                    capeview.__array_interface__["data"][0]):
                raise RuntimeError("output array was copied")

            if flip and not is2d:
                output[cape_output_idxs] = (
                    outview_array[view_cape_reverse_idxs].astype(orig_dtype))
                output[cin_output_idxs] = (
                    outview_array[view_cin_reverse_idxs].astype(orig_dtype))
            else:
                output[cape_output_idxs] = (
                    outview_array[cape_idxs].astype(orig_dtype))
                output[cin_output_idxs] = (
                    outview_array[cin_idxs].astype(orig_dtype))

        return output

    return func_wrapper


def cloudfrac_left_iter(alg_dtype=np.float64):
    """A decorator to handle iterating over the leftmost dimensions for the
    cloud fraction diagnostic.

    For example, if a wrapped function works with three-dimensional arrays, but
    the variables include a 4th leftmost dimension for 'Time', this decorator
    will iterate over all times, call the 3D Fortran routine, and aggregate the
    results in to a 4D output array.

    It is also important to note that the final output array is allocated
    first, and then views are passed to the wrapped function so that values
    do not need to get copied in to the final output array.

    Args:

        alg_dtype (:class:`np.dtype` or :obj:`str`): The numpy data type used
            in the wrapped function.

    Returns:

        :class:`numpy.ndarray`: The aggregated cloud fraction output array
        that includes all extra leftmost dimensions.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        new_args = list(args)
        new_kwargs = dict(kwargs)

        vert = args[0]
        rh = args[1]

        num_left_dims = vert.ndim - 3
        orig_dtype = vert.dtype

        # No special left side iteration, build the output from the
        # low, mid, high results.
        if (num_left_dims == 0):
            low, mid, high = wrapped(*new_args, **new_kwargs)

            output_dims = (3,)
            output_dims += vert.shape[-2:]
            output = np.empty(output_dims, orig_dtype)

            output[0, :] = low[:]
            output[1, :] = mid[:]
            output[2, :] = high[:]

            return output

        # Initial output is ...,low_mid_high,nz,ny,nx to create contiguous
        # views
        outdims = vert.shape[0:num_left_dims]
        extra_dims = tuple(outdims)  # Copy the left-most dims for iteration

        outdims += (3,)  # low_mid_high

        outdims += vert.shape[-2:]

        outview_array = np.empty(outdims, alg_dtype)

        # Create the output array where the leftmost dim is the cloud type
        output_dims = (3,)
        output_dims += extra_dims
        output_dims += vert.shape[-2:]
        output = np.empty(output_dims, orig_dtype)

        has_missing = False
        missing = default_fill(np.float64)
        for left_idxs in iter_left_indexes(extra_dims):
            left_and_slice_idxs = left_idxs + (slice(None),)
            low_idxs = left_idxs + (0, slice(None))
            mid_idxs = left_idxs + (1, slice(None))
            high_idxs = left_idxs + (2, slice(None))

            low_output_idxs = (0,) + left_idxs + (slice(None),)
            mid_output_idxs = (1,) + left_idxs + (slice(None),)
            high_output_idxs = (2,) + left_idxs + (slice(None),)

            new_args[0] = vert[left_and_slice_idxs]
            new_args[1] = rh[left_and_slice_idxs]

            # Skip the possible empty/missing arrays for the join method
            # Note: Masking handled by cloudfrac.py or computation.py, so only
            # supply the fill values here.
            skip_missing = False
            for arg in (new_args[0:2]):
                if isinstance(arg, np.ma.MaskedArray):
                    if arg.mask.all():
                        output[low_output_idxs] = missing
                        output[mid_output_idxs] = missing
                        output[high_output_idxs] = missing

                        skip_missing = True
                        has_missing = True

            if skip_missing:
                continue

            lowview = outview_array[low_idxs]
            midview = outview_array[mid_idxs]
            highview = outview_array[high_idxs]

            new_kwargs["lowview"] = lowview
            new_kwargs["midview"] = midview
            new_kwargs["highview"] = highview

            low, mid, high = wrapped(*new_args, **new_kwargs)

            # Make sure the result is the same data as what got passed in
            # Can delete this once everything works
            if (low.__array_interface__["data"][0] !=
                    lowview.__array_interface__["data"][0]):
                raise RuntimeError("output array was copied")

            output[low_output_idxs] = (
                outview_array[low_idxs].astype(orig_dtype))
            output[mid_output_idxs] = (
                outview_array[mid_idxs].astype(orig_dtype))
            output[high_output_idxs] = (
                outview_array[high_idxs].astype(orig_dtype))

        if has_missing:
            output = np.ma.masked_values(output, missing)

        return output

    return func_wrapper


def interplevel_left_iter(is2dlev, alg_dtype=np.float64):
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        new_args = list(args)
        new_kwargs = dict(kwargs)

        field3d = args[0]
        z = args[1]
        levels = args[2]

        num_left_dims = z.ndim - 3
        orig_dtype = field3d.dtype
        left_dims = z.shape[0:num_left_dims]
        multiproduct = True if field3d.ndim - z.ndim == 1 else False

        # No special left side iteration, build the output from the
        # low, mid, high results.
        if (num_left_dims == 0):
            if multiproduct:
                if not is2dlev:
                    outshape = (field3d.shape[0:-3] + levels.shape +
                                field3d.shape[-2:])
                else:
                    outshape = (field3d.shape[0:-3] + field3d.shape[-2:])

                output = np.empty(outshape, dtype=alg_dtype)
                for i in py3range(field3d.shape[0]):
                    new_args[0] = field3d[i, :]
                    new_kwargs["outview"] = output[i, :]
                    _ = wrapped(*new_args, **new_kwargs)
            else:
                output = wrapped(*args, **kwargs)

            return output

        if multiproduct:
            outdims = field3d.shape[0:1] + left_dims
        else:
            outdims = left_dims

        extra_dims = tuple(outdims)

        if not is2dlev:
            outdims += levels.shape

        outdims += z.shape[-2:]

        outview_array = np.empty(outdims, alg_dtype)

        for left_idxs in iter_left_indexes(extra_dims):

            field_out_slice_idxs = left_idxs + (slice(None),)

            if multiproduct:
                z_slice_idxs = left_idxs[1:] + (slice(None),)
            else:
                z_slice_idxs = left_idxs + (slice(None),)

            new_args[0] = field3d[field_out_slice_idxs]
            new_args[1] = z[z_slice_idxs]

            if is2dlev:
                if levels.ndim > 2:
                    new_args[2] = levels[z_slice_idxs]

            new_kwargs["outview"] = outview_array[field_out_slice_idxs]

            _ = wrapped(*new_args, **new_kwargs)

        output = outview_array.astype(orig_dtype)

        return output

    return func_wrapper


def check_cape_args():
    """A decorator to check that the cape_3d arguments are valid.

    An exception is raised when an invalid argument is found.

    Returns:

        None

    Raises:

        :class:`ValueError`: Raised when an invalid argument is detected.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):

        p_hpa = args[0]
        tk = args[1]
        qv = args[2]
        ht = args[3]
        ter = args[4]
        sfp = args[5]
        missing = args[6]
        i3dflag = args[7]
        ter_follow = args[8]

        is2d = False if i3dflag != 0 else True
        is1d = ((np.isscalar(sfp) or np.size(sfp) == 1) or
                (np.isscalar(ter) or np.size(ter) == 1))

        if not (p_hpa.shape == tk.shape == qv.shape == ht.shape):
            raise ValueError("arguments 0, 1, 2, 3 must be the same shape")

        # 2D CAPE does not allow for scalars
        if not is1d:
            if ter.ndim != p_hpa.ndim-1 or sfp.ndim != p_hpa.ndim-1:
                raise ValueError("arguments 4 and 5 must have "
                                 "{} dimensions".format(p_hpa.ndim-1))
        else:
            if np.size(ter) != np.size(sfp):
                raise ValueError("arguments 4 and 5 must both be scalars or "
                                 "both be arrays")

            # Only need to test p_hpa since we assured args 0-3 have same ndim
            if p_hpa.ndim != 1:
                raise ValueError("arguments 0-3 "
                                 "must be 1-dimensional when "
                                 "arguments 4 and 5 are scalars")

        return wrapped(*args, **kwargs)

    return func_wrapper


def check_interplevel_args(is2dlev):
    """A decorator to check that the interplevel arguments are valid.

    An exception is raised when an invalid argument is found.

    Returns:

        None

    Raises:

        :class:`ValueError`: Raised when an invalid argument is detected.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):

        field3d = args[0]
        z = args[1]
        levels = args[2]

        multiproduct = True if (field3d.ndim - z.ndim) == 1 else False

        if not multiproduct:
            if field3d.shape != z.shape:
                raise ValueError("arguments 0 and 1 must have the same shape")
        else:
            if field3d.shape[1:] != z.shape:
                raise ValueError("argument 0 and 1 must have same rightmost "
                                 "dimensions")

        if is2dlev:
            if levels.ndim != 2:
                if (levels.shape[0:-2] != z.shape[0:-3] or
                        levels.shape[-2:] != z.shape[-2:]):
                    raise ValueError("argument 1 and 2 must have "
                                     "the same leftmost and rightmost "
                                     "dimensions")

        return wrapped(*args, **kwargs)

    return func_wrapper
