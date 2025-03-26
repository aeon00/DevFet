import numpy as np

def local_determinance_map(
        coefficients, f2analyse, nlevels, group_indices, eig_vec):
    """
    Parameters
    ----------
    coefficients : Array of floats
        Fourier coefficients of the input function f2analyse
    f2analyse : Array of floats
        Scalar function to analyze (e.g. mean curvature)
    nlevels : Array of ints
        number of spectral bands
    group_indices : Array of ints
        indices of spectral bands
    eig_vec : Array of floats
        eigenvectors (reversed order for computation and memory reasons)

    Returns
    -------
    loc_det_band : Array of floats
        texture with the differential contribution of each frequency band
    frecomposed : Array of floats
        recomposition of f2analyse in each frequency band
    """
    N = np.size(coefficients)

    frecomposed = np.zeros((len(f2analyse), nlevels - 1), dtype='object')
    eig_vec = np.flip(eig_vec, 1)

    # band by band recomposition
    for i in range(nlevels - 1):
        # levels_ii: number of frequency band wihin the compact Band i
        levels_i = np.arange(
            group_indices[i + 1, 0], group_indices[i + 1, 1] + 1)
        # np.array((number of vertices, number of levels_ii))
        f_ii = np.dot(eig_vec[:, N - levels_i - 1], coefficients[levels_i].T)
        frecomposed[:, i] = f_ii

    # locally determinant band

    '''
    diff_recomposed = frecomposed[:, 0]
    diff_recomposed = np.concatenate(
        (np.expand_dims(
            diff_recomposed, axis=1), np.diff(
            frecomposed, axis=1)), axis=1)
    '''
    SMk = np.zeros((len(f2analyse),nlevels))
    # Formulas C.2, C.3, C.4 of Spangy 2012 article
    fcumulative = np.zeros((f2analyse.shape))
    for i in range(nlevels-1): 
        sign_fcumulative = (fcumulative<0)
        fcumulative+= frecomposed[:,i]
        SMk[:,i] = (fcumulative >0) - sign_fcumulative
       
    loc_det_band = np.zeros((f2analyse.shape))
    

    # sulci 
    '''
    idx = np.argmin(diff_recomposed, axis=1)
    idx = idx + 1
    loc_dom_band[f2analyse <= 0] = idx[f2analyse <= 0] * (-1)
    '''
    for i in range(1,nlevels-1,1):
        indices = np.where(SMk[:,i]<0 & SMk[:,i-1]>=0)
        loc_det_band[indices] = - i

    # gyri
    '''
    idx = np.argmax(diff_recomposed, axis=1)
    idx = idx + 1
    loc_dom_band[f2analyse > 0] = idx[f2analyse > 0]
    '''
    for i in range(1,nlevels-1,1):
        indices = np.where(SMk[:,i]>0 & SMk[:,i-1]<=0)
        loc_det_band[indices] =  i


    return loc_det_band, frecomposed
