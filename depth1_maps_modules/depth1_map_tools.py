import numpy as np
from pixell import enmap
import warnings


def kappa_clean(kappa:np.ndarray, 
                rho:np.ndarray
                ):
    ## set the inverse-variance below a minimum value to 1e-3 * max of kappa
    ## this effectively thresholds the variance, and reduces SNR for high variance pixels.
    ## also make variance infinite if pixels aren't hit.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    kappa = np.maximum(kappa, np.nanmax(kappa) * 1e-3)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(imap:np.ndarray, 
              inverse_variance:np.ndarray,
              fraction:float=0.01,
              cut_on:str='max'
              ):
    ## zero out regions of the map with variance above a relative threshold.
    ##
    ## cut_on can be `max` or `median`, this sets the imap to zero for values of inverse variance
    ## which are below fraction*max or fraction*median of inverse variance map.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if cut_on=='median' or cut_on=='med':
        imap[inverse_variance < (np.nanmedian(inverse_variance) * fraction)] = 0
    else:
        if cut_on!='max':
            print('%s cut_on not supported, defaulting to max cut'%cut_on)
        imap[inverse_variance < (np.nanmax(inverse_variance) * fraction)] = 0
    return imap


def mask_edge(imap:enmap.ndmap, 
              pix_num: int
             ):
    """Get a mask that masking off edge pixels

    Args:
        imap:ndmap to create mask for,usually kappa map
        pix_num: pixels within this number from edge will be cutoff

    Returns:
        binary ndmap with 1 == unmasked, 0 == masked
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import binary_fill_holes
    edge = enmap.enmap(imap, imap.wcs)  # Create map geometry
    edge[np.abs(edge) > 0] = 1  # Convert to binary
    edge = binary_fill_holes(edge)  # Fill holes
    edge = enmap.enmap(edge.astype("ubyte"), imap.wcs)

    dmap = enmap.enmap(distance_transform_edt(edge), edge.wcs)
    del edge
    mask = enmap.zeros(imap.shape, imap.wcs)
    mask[np.where(dmap > pix_num)] = 1
    return mask


def preprocess_map(rho_map:enmap.enmap,
                   kappa_map:enmap.enmap,
                   time_map:enmap.enmap=None,
                   galmask_file='galactic_dust_mask.fits',
                   flatfield=False
                   ):
    '''
    Clean the maps by cutting on variance, masking the galaxy and masking map edges.

    Flatfield the noise so that regions of low weight with highly non-gaussian noise are de-weighted.
    Defualt is to tile the map in 1deg chunks.
    Both flux and snr are reduced by the relative amplitude of the variance in the snr of the tile.

    returns the cleaned, masked and flatfielded flux and snr maps.
    '''
    from tiles import get_medrat, get_tmap_tiles
    
    print('Cleaning maps...')
    kappa_map  = kappa_clean(kappa_map,
                             rho_map
                            )
    rho_map = clean_map(rho_map,
                        kappa_map,
                        cut_on='median',
                        fraction=0.05
                       )
    
    flux =rho_map/kappa_map
    snr = rho_map*kappa_map**(-0.5)
    del kappa_map,rho_map
    
    print('Masking maps...')
    if galmask_file:
        galaxy_mask = enmap.read_map(galmask_file)
        gal_mask = enmap.extract(galaxy_mask,flux.shape,flux.wcs)
        flux *= gal_mask
        snr *= gal_mask
        del gal_mask,galaxy_mask
        
    edge_mask = mask_edge(flux,40)
    flux *= edge_mask
    snr *= edge_mask
    del edge_mask
        
    if flatfield and not isinstance(time_map,type(None)):
        print('Flatfielding maps...')
        med_ratio = get_medrat(snr,
                               get_tmap_tiles(np.nan_to_num(time_map)-np.nanmin(time_map),
                                              1.0, ## 1deg tiles
                                              snr,
                                             ),
                                  )
            
        flux*=med_ratio
        snr*=med_ratio
        del med_ratio
    
    return flux,snr

