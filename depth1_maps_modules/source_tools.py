from pixell import enmap
from pixell import utils as pixell_utils
import numpy as np

## requires astropy and photutils


def load_act_catalog(source_cat_file:str='act_sample_cat.fits',
                    ):
    '''
    source_cat_file is path to source catalog

    '''
    from astropy.table import Table 
    sourcecat=None
    sourcecat = Table.read(source_cat_file)
    sourcecat['RADeg'][sourcecat["RADeg"]<0]+=360.
    print(len(sourcecat["decDeg"]), 'sources in catalog.')
    return sourcecat


def radec_to_str_name(ra:float, 
                      dec:float, 
                      source_class="pointsource", 
                      observatory="ACT"
                      ):
    """
    Convert RA & dec (in radians) to IAU-approved string name.
    Allow for prepending of observatory name and source class, i.e. a cluster or variable.
    
    Arguments
    ---------
    ra : float
        Source right ascension in radians
    dec : float
        Source declination in radians
    source_class : str
        The class of source to which this name is assigned.  Supported classes
        are ``pointsource``, ``cluster`` or ``transient``.  Shorthand class
        names for these sources are also allowed (``S``, ``CL``, or ``SV``,
        respectively).  Alternatively, the class can be ``None`` or ``short``,
        indicating a simple source identifier in the form ``"HHMM-DD"``.

    Returns
    -------
    name : str
        A unique identifier for the source with coordinates truncated to the
        appropriate number of significant figures for the source class.
    """
    from astropy.coordinates.angles import Angle
    from astropy import units as u
    
    source_class = str(source_class).lower()
    if source_class in ["sv", "t", "tr", "transient", "v", "var", "variable"]:
        source_class = "SV"
    elif source_class in ["c", "cl", "cluster"]:
        source_class = "CL"
    elif source_class in ["s", "p", "ps", "pointsource", "point_source"]:
        source_class = "S"
    elif source_class in ["none", "short"]:
        source_class = "short"
    else:
        print("Unrecognized source class {}".format(source_class))
        print("Defaulting to [ S ]")
        source_class = "S"
    # ra in (0, 360)
    ra = np.mod(ra, 360)

    opts = dict(sep="", pad=True, precision=3)
    rastr = Angle(ra * u.deg).to_string(**opts, unit="hour")
    decstr = Angle(dec * u.deg).to_string(alwayssign=True, **opts)

    if source_class == "SV":
        rastr = rastr[:8]
        decstr = decstr.split(".")[0]
    elif source_class == "CL":
        rastr = rastr[:4]
        decstr = decstr[:5]
    elif source_class == "S":
        rastr = rastr.split(".")[0]
        decr = "{:.3f}".format(dec * 60).split(".")[1]
        decstr = decstr[:5] + ".{}".format(decr[:1])
    elif source_class == "short":
        rastr = rastr[:4]
        decstr = decstr[:3]
        return "{}{}".format(rastr, decstr)

    name = "{}-{} J{}{}".format(observatory, source_class, rastr, decstr)
    return name

def get_source_sky_positions(extracted_sources:dict,
                             skymap:enmap.enmap
                             ):
    '''
    from output of extract_sources, use xpeak and ypeak to 
    convert to sky coordinates given the enmap `skymap`.

    returns `extracted_sources` with `ra` and `dec` keys.
    '''
    for f in extracted_sources:
        x,y = extracted_sources[f]['xpeak'],extracted_sources[f]['ypeak']
        dec,ra = skymap.pix2sky(np.asarray([[y],[x]]))
        extracted_sources[f]['ra'] = ra[0]%(360*pixell_utils.degree)
        extracted_sources[f]['dec'] = dec[0]
        
    return extracted_sources


def get_source_observation_time(extracted_sources:dict,
                                timemap:np.ndarray
                                ):
    '''
    from output of extract_sources, use xpeak and ypeak to 
    get the observed time given the map `timemap`.

    returns `extracted_sources` with a `time` key.
    '''
    for f in extracted_sources:
        x,y = int(extracted_sources[f]['xpeak']),int(extracted_sources[f]['ypeak'])
        extracted_sources[f]['time'] = timemap[y,x]
        
    return extracted_sources
    

def extract_sources(inmap:enmap,
                    timemap:enmap=None,
                    maprms:float=None,
                    nsigma:float=5.0,
                    minrad:list=[0.5],
                    sigma_thresh_for_minrad:list=[0.0],
                    res:float=None,
                    pixel_mask:np.ndarray=None,
                    ):
    """
    Source finding using photutils, assuming mean background is zero.

    Arguments
    ---------
    inmap : 2d array or enmap
        2d-array representing an unweighted flux map.
        Must be an enmap to get sky coordinates.
    timemap : enmap (or 2D array)
        time at each pixel, used to get observed time if provided
    maprms : float
        The 1-sigma noise level in the map. If not provided, will be calculated.
    nsigma :
        Required signal-to-noise to detect a source.
    minrad : array-like
        The required separation between detected sources in arcmin. If given as
        a list, provides different radii for different-sigma sources.
    sigma_thresh_for_minrad : array-like
        The source detection strengths corresponding to different exclusion
        radii. Only used if more than one element in ``minrad``.
    res :
        Resolution of map, in arcmin.  Required if ``inmap`` is an array.
    pixel_mask : 2d array or enmap
        Optional mask applied to map before source finding.
    
    Returns
    -------
    output_struct: dict
        Contains a dictionary for each source, labelled by sequential integers,
        with keys 'xpeak', 'ypeak','peakval','peaksig'.
        Also the various photutils output which may be useful:
            "area", "ellipticity", "elongation", "fwhm", "kron_aperture",
            "kron_flux", "kron_fluxerr", "kron_radius"
            
    Notes
    -----
    Jan 2019: DPD ported from spt_analysis/sources/find_sources_quick.pro
    Jan 2025: AF porting to sotrplib
    """

    if res is None:
        try:
            res = np.abs(inmap.wcs.wcs.cdelt[0])
        except AttributeError:
            raise ValueError("Argument `res` required if inmap is an array")

    minrad = np.atleast_1d(minrad)
    sigma_thresh_for_minrad = np.atleast_1d(sigma_thresh_for_minrad)

    
    if len(sigma_thresh_for_minrad) != len(minrad):
        raise ValueError(
            "If you are specifying multiple avoidance radii,"
            + "please supply a threshold level for each one."
        )

    if pixel_mask is not None:
        imap*=pixel_mask 

    # get rms in map if not supplied
    if maprms is None:
        whn0 = np.where(np.abs(inmap) > 1.0e-8)
        if len(whn0[0]) == 0:
            maprms = np.nanstd(inmap)
        else:
            maprms = np.nanstd(np.asarray(inmap)[whn0])
            
    
    
    peaks = find_using_photutils(np.asarray(inmap),
                                 maprms,
                                 nsigma=nsigma,
                                 minnum=1,
                                )
    
    npeaks = peaks["n_detected"]
    if npeaks == 0:
        print("No sources found")
        return

    # gather detected peaks into output structure, ignoring repeat
    # detections of same object
    xpeaks = peaks["xcen"]
    ypeaks = peaks["ycen"]
    peakvals = peaks["maxvals"]
    peaksigs = peaks["sigvals"]
    areas = peaks["area"]
    ellipticities = peaks["ellipticity"]
    elongations = peaks["elongation"]
    fwhms = peaks["fwhm"]
    kron_apertures = peaks["kron_aperture"]
    kron_fluxes = peaks["kron_flux"]
    kron_fluxerrs = peaks["kron_fluxerr"]
    kron_radii = peaks["kron_radius"]
    peak_assoc = np.zeros(npeaks)

    output_struct = dict()
    for i in np.arange(npeaks):
        output_struct[i] = {"xpeak": xpeaks[0],
                            "ypeak": ypeaks[0],
                            "peakval": peakvals[0],
                            "peaksig": peaksigs[0],
                            "area": areas[0].value*res**2,
                            "ellipticity": ellipticities[0],
                            "elongation": elongations[0],
                            "fwhm": fwhms[0].value*res,
                            "kron_aperture": kron_apertures[0],
                            "kron_flux": kron_fluxes[0],
                            "kron_fluxerr": kron_fluxerrs[0],
                            "kron_radius": kron_radii[0].value*res
                           }
            
    minrad_pix = minrad / res

    ksource = 1

    # different accounting if exclusion radius is specified as a function
    # of significance
    if len(minrad) > 1:
        minrad_pix_all = np.zeros(npeaks)
        sthresh = np.argsort(sigma_thresh_for_minrad)
        for j in np.arange(len(minrad)):
            i = sthresh[j]
            whgthresh = np.where(peaksigs >= sigma_thresh_for_minrad[i])[0]
            if len(whgthresh) > 0:
                minrad_pix_all[whgthresh] = minrad_pix[i]
        minrad_pix_os = np.zeros(npeaks)
        minrad_pix_os[0] = minrad_pix_all[0]
        for j in np.arange(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            whclose = np.where(distpix <= minrad_pix_os[0:ksource])[0]
            if len(whclose) == 0:
                output_struct[ksource] = {"xpeak": xpeaks[j],
                                          "ypeak": ypeaks[j],
                                          "peakval": peakvals[j],
                                          "peaksig": peaksigs[j],
                                          "area": areas[j].value*res**2,
                                          "ellipticity": ellipticities[j],
                                          "elongation": elongations[j],
                                          "fwhm": fwhms[j].value*res,
                                          "kron_aperture": kron_apertures[j],
                                          "kron_flux": kron_fluxes[j],
                                          "kron_fluxerr": kron_fluxerrs[j],
                                          "kron_radius": kron_radii[j].value*res
                                         }
                    
                peak_assoc[j] = ksource
                minrad_pix_os[ksource] = minrad_pix_all[j]
                ksource += 1
            else:
                mindist = min(distpix)
                peak_assoc[j] = distpix.argmin()
    else:
        for j in range(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            mindist = min(distpix)
            if mindist > minrad_pix:
                output_struct[ksource] = {"xpeak": xpeaks[j],
                                          "ypeak": ypeaks[j],
                                          "peakval": peakvals[j],
                                          "peaksig": peaksigs[j],
                                          "area": areas[j].value*res**2,
                                          "ellipticity": ellipticities[j],
                                          "elongation": elongations[j],
                                          "fwhm": fwhms[j].value*res,
                                          "kron_aperture": kron_apertures[j],
                                          "kron_flux": kron_fluxes[j],
                                          "kron_fluxerr": kron_fluxerrs[j],
                                          "kron_radius": kron_radii[j].value*res
                                         }
                peak_assoc[j] = ksource
                ksource += 1
            else:
                peak_assoc[j] = distpix.argmin()
    for src in list(output_struct.keys()):
        if src >= ksource:
            del output_struct[src]

    output_struct = get_source_sky_positions(output_struct,
                                             inmap
                                            )
    if not isinstance(timemap,type(None)):
        output_struct = get_source_observation_time(output_struct,
                                                    timemap
                                                   )
    return output_struct


def find_using_photutils(Tmap:np.ndarray, 
                         signoise:float=None,
                         minnum:int=2, 
                         nsigma:float=5.0
                         ):
    """
    Use the photutils package to find sources and extract their fluxes.
    Utilizing astropy.photutils, one can deblend sources close to each other.  
    Exactly how the deblending is done uses the
    default arguments of the package, influenced by SExtractor.

    From ``find_groups``: given a 2d array (a map), will find groups of elements
    (pixels) that are spatially associated (sources).

    Arguments
    ---------
    Tmap: enamp,ndarray
        enmap,ndarray, e.g. representing a flux map.
    signoise : float
        global rms of the map.
        ## need to update this to use tiled rms map - it's all set up, just need to do it.
    offset : float
        Zero point of map.
    minnum : int
        Minimum number of pixels needed to form a group.
    nsigma : float
        Required detection threshold for a group.

    Returns
    -------
    Dictionary with following keys:
        * maxvals - array of heights (in map units) of found objects.
        * sigvals - array of heights (in significance units) of found objects.
        * xcen - array of x-location of group centers. From the documentation,
          the "centroid is computed as the center of mass of the unmasked pixels
          within the source segment."
        * ycen - array of Y-location of group centers.
        * n_detected - Number of detected sources.

    The keys below are outputs of the astropy source finding functionality,
    being used to test if any of them indicate extendedness of a source.
        * area - units of pixels**2
        * ellipticity
        * elongation
        * fwhm - units of pixels. From the documentation, "circularized FWHM of
          2D Gaussian function with same second order moments as the source."
        * kron_aperture
        * kron_flux - Parameter requires that Tmap already be in units of mJy.
        * kron_fluxerr - Parameter requires that Tmap already be in units of mJy.
        * kron_radius - units of pixels.

    Function written by Melanie Archipley, adapted by Allen Foster Jan 2025
    """
    # this import is here instead of at the beginning because it has strange
    # dependencies that I (Melanie) do not want to cause problems for other people
    from photutils import segmentation as pseg

    default_keys = {
        "maxvals": "max_value",
        "sigvals": None,
        "xcen": "xcentroid",
        "ycen": "ycentroid",
        "n_detected": None,
    }

    extra_keys = [
        "area",
        "ellipticity",
        "elongation",
        "fwhm",
        "kron_aperture",
        "kron_flux",
        "kron_fluxerr",
        "kron_radius",
    ]

    groups = {k: 0 for k in list(default_keys) + extra_keys}

    if not isinstance(Tmap, np.ndarray):
        Tmap = np.asarray(Tmap)
    assert len(Tmap.shape) == 2
    if not isinstance(signoise, np.ndarray):
        signoise = np.asarray(signoise)
    assert signoise.shape == Tmap.shape

    img = pseg.detect_sources(Tmap, 
                              threshold=nsigma * signoise, 
                              npixels=minnum
                              )
    if img is None:
        return groups

    img = pseg.deblend_sources(Tmap, 
                               img, 
                               npixels=minnum
                               )
    if img is None:
        return groups

    cat = pseg.SourceCatalog(Tmap,
                             img,
                             error=signoise,
                             apermask_method="correct",
                             kron_params=(2.5, 1.0),
                            )

    # convert catalog to table
    columns = [v for _, v in default_keys.items() if v] + extra_keys
    tbl = cat.to_table(columns=columns)
    # rename keys to match find_groups output
    for k, v in default_keys.items():
        if v is not None:
            tbl.rename_column(v, k)
    tbl.sort("maxvals", reverse=True)

    # store signal at each source location
    ix, iy = [np.floor(tbl[k] + 0.5).astype(int) for k in ("xcen", "ycen")]
    tbl["sigvals"] = Tmap[iy, ix] / signoise[iy, ix]

    # populate output dictionary
    groups["n_detected"] = len(tbl)
    for k in groups:
        if k in tbl.columns:
            groups[k] = tbl[k]

    return groups


def crossmatch_mask(sources, 
                    crosscat, 
                    radius:float,
                    mode:str='all',
                    return_matches:bool=False
                   ):
    """Determines if source matches with masked objects

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        crosscat: catalog of masked objects [[dec, ra]] in deg
        radius: radius to search for matches in arcmin, float or list of length sources.
        mode: return `all` pairs, or just `closest`
    Returns:
        mask column for sources, 1 matches with at least one source, 0 no matches

    """    
    crosspos_ra = crosscat[:, 1] * pixell_utils.degree
    crosspos_dec = crosscat[:, 0] * pixell_utils.degree
    crosspos = np.array([crosspos_ra, crosspos_dec]).T

    # if only one source
    if len(sources.shape) == 1:
        source_ra = sources[1] * pixell_utils.degree
        source_dec = sources[0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius,(list,np.array)):
            r = radius[0] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)
        
        if len(match) > 0:
            return True if not return_matches else True,match
        else:
            return False if not return_matches else False,match

    mask = np.zeros(len(sources), dtype=bool)
    matches = []
    for i, _ in enumerate(mask):
        source_ra = sources[i, 1] * pixell_utils.degree
        source_dec = sources[i, 0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius,(list,np.array)):
            r = radius[i] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)
        if len(match) > 0:
            mask[i] = True
        matches.append([(i,m[1]) for m in match])
    if return_matches:
        return mask,matches
    else:
        return mask


def crossmatch_sources(extracted_sources,
                       catalog_sources,
                       radius1Jy:float=30.0,
                       min_match_radius:float=1.5,
                       source_fluxes:list = None,
                       fwhm_cut = 5.0
                      ):
    
    """
     Perform crossmatching of extracted sources from `extract_sources` and the cataloged sources.
     Return lists of dictionaries containing each source which matches the catalog, may be noise, or appears to be a transient.

     Uses a flux-based matching radius with `radius1Jy` the radius, in arcmin, for a 1Jy source and 
     `min_match_radius` the radius, in arcmin, for a zero flux source, up to a max of 2 degrees.

     Args:
       extracted_sources:dict
           sources returned from extract_sources function
       catalog_sources:astropy table
           source catalog returned from load_act_catalog
       radius1Jy:float=30.0
           matching radius for a 1Jy source, arcmin
       min_match_radius:float=1.5
           minimum matching radius, i.e. for a zero flux source, arcmin
       source_fluxes:list = None,
           a list of the fluxes of the extracted sources, if None will pull it from `extracted_sources` dict. 
       fwhm_cut = 5.0
           a simple cut on fwhm, in arcmin, above which something is considered noise.
     Returns:
        source_candidates, transient_candidates, noise_candidates : list
            list of dictionaries with information about the detected source.
    """    
    if isinstance(source_fluxes,type(None)):
        source_fluxes = np.asarray([extracted_sources[f]['peakval']/1000. for f in extracted_sources])
    
    extracted_ra = np.asarray([extracted_sources[f]['ra'] for f in extracted_sources])
    extracted_dec = np.asarray([extracted_sources[f]['dec'] for f in extracted_sources])
    
    crossmatch_radius = np.minimum(np.maximum(source_fluxes*radius1Jy,min_match_radius),120)
    isin_cat,catalog_match = crossmatch_mask(np.asarray([extracted_dec/ pixell_utils.degree,extracted_ra/ pixell_utils.degree ]).T,
                                             np.asarray([catalog_sources["decDeg"], catalog_sources["RADeg"]]).T,
                                             list(crossmatch_radius),
                                             mode='closest',
                                             return_matches=True
                                            )
    
    
    source_candidates = []
    transient_candidates = []
    noise_candidates = []
    for source,cand_pos in enumerate(zip(extracted_ra,extracted_dec)):
        forced_photometry_info = extracted_sources[source]
        source_string_name = radec_to_str_name(cand_pos[0]/pixell_utils.degree,
                                               cand_pos[1]/pixell_utils.degree
                                              )
            
        if isin_cat[source]:
            crossmatch_name = catalog_sources['name'][catalog_match[source][0][1]]
        else:
            crossmatch_name = ''
    
        cand = {'ra':cand_pos[0]/pixell_utils.degree%360,
                'dec':cand_pos[1]/pixell_utils.degree,
                'flux':forced_photometry_info['peakval'],
                'dflux':forced_photometry_info['peakval']/forced_photometry_info['peaksig'],
                'kron_flux':forced_photometry_info['kron_flux'],
                'kron_fluxerr':forced_photometry_info['kron_fluxerr'],
                'kron_radius' : forced_photometry_info['kron_radius'],
                'snr':forced_photometry_info['peaksig'],
                'ctime':forced_photometry_info['time'],
                'sourceID':source_string_name,
                'catalog_crossmatch':isin_cat[source],
                'crossmatch_name':crossmatch_name,
                'ellipticity':forced_photometry_info['ellipticity'],
                'elongation':forced_photometry_info['elongation'],
                'fwhm':forced_photometry_info['fwhm']
                }
         ## do sifting operations here...
        if not np.isfinite(cand['kron_flux']) or not np.isfinite(cand['kron_fluxerr']) or cand['fwhm']>=fwhm_cut or not np.isfinite(cand['flux']) :
            noise_candidates.append(cand)
        elif isin_cat[source]:
            source_candidates.append(cand)
        else:
            transient_candidates.append(cand)
        del cand
        
    print(len(source_candidates),'catalog matches')
    print(len(transient_candidates),'transient candidates')
    print(len(noise_candidates),'probably noise') 
    return source_candidates, transient_candidates, noise_candidates


