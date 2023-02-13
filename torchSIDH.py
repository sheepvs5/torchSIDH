import torchOptics.optics as tt
import torch
import torch.nn.functional as F
import numpy as np

def simulate_long(x, d, dstep=100e-3, padding=500, band_limit=False, **kwargs):
    if d<dstep:
        return tt.simulate(x, d, padding=padding, band_limit=band_limit, **kwargs)
    else:
        tmp = x.clone()
        dtmps = [dstep for _ in range(int(d//dstep))] + [d%dstep]
        for dtmp in dtmps:
            tmp = tt.simulate(tmp, dtmp, padding=padding, band_limit=band_limit, **kwargs)
        return tmp
    
def get_radius(intensity):
    x,y = np.meshgrid(np.arange(intensity.shape[-2]), np.arange(intensity.shape[-1]))
    intensity = np.where(intensity>intensity.max()*0.01, intensity, 0)
    intensity = intensity/intensity.sum()
    xc,yc = np.sum(x*intensity), np.sum(y*intensity)
    r = ((x-xc)**2 + (y-yc)**2)**0.5
    return (r*intensity).sum()

def get_rms_radius(intensity):
    x,y = np.meshgrid(np.arange(intensity.shape[-2]), np.arange(intensity.shape[-1]))
    intensity = np.where(intensity>intensity.max()*0.01, intensity, 0)
    intensity = intensity/intensity.sum()
    xc,yc = np.sum(x*intensity), np.sum(y*intensity)
    r2 = ((x-xc)**2 + (y-yc)**2)
    return np.sqrt((r*intensity).sum())

def get_psf(rgb, depth, config={'f1':1000e-3, 'f2':1000e-3, 'd2':10e-3}, wl_range=0.2, angle_range=[[-0.02, 0.02], [-0.02, 0.02]], noi=5, dstep=100e-3, psf_slice=[slice(0,3), slice(0,3), slice(500,1500), slice(500,1500)], blocker=None):
    wl_original = rgb.meta['wl'].copy()
    to_tensor = lambda x: torch.tensor(x, device=rgb.device)
    f1, f2, d2 = map(to_tensor, [config['f1'], config['f2'], config['d2']])
    d1s = torch.unique(depth)
    zeros = torch.zeros_like(rgb)
    for i in range(2):
        for j in range(2):
            zeros[..., zeros.shape[-2]//2+i, zeros.shape[-1]//2+j]=25
    
    psfs = []
    for d1 in d1s:
        imgs = 0
        for a0 in np.linspace(*angle_range[0], noi):
            for a1 in np.linspace(*angle_range[1], noi):
                zeros.meta['wl'] = (1+(np.random.rand(3)-0.5)*wl_range)*wl_original
                tmp = simulate_long(zeros * tt.getPrismPhase(zeros, angle=(a0, a1)), d1)
                if blocker is not None:
                    tmp = tmp*blocker
                imgs += simulate_long(tt.applylens(tmp, f1), d2)*simulate_long(tt.applylens(tmp, f2), d2).conj()
                
        imgs.meta['wl'] = wl_original
        psfs.append(imgs.squeeze(0).unsqueeze(1)[psf_slice])
    return [d1s, psfs]

def simulate_SIDH(rgb, depth, config={'f1':1000e-3, 'f2':1000e-3, 'd2':10e-3}, wl_range=0.2, angle_range=[[-0.02, 0.02], [-0.02, 0.02]], noi=5, dstep=100e-3, psf_slice=[slice(0,3), slice(0,3), slice(500,1500), slice(500,1500)], blocker=None):
    d1s, psfs = get_psf(rgb=rgb, depth=depth, config=config, wl_range=wl_range, angle_range=angle_range, noi=noi, dstep=dstep, psf_slice=psf_slice)
    tmp = 0
    for psf, d1 in zip(psfs, d1s):
        timg = torch.where(depth==d1, rgb, torch.zeros_like(rgb))
        tmp += F.conv2d(timg, psf.real, padding=psf.shape[-2:], groups=3) + 1j* F.conv2d(timg, psf.imag, padding=psf.shape[-2:], groups=3)
        # tmp += conv_complex(F.conv2d, timg, psf, padding=psf.shape[-2:], groups=3)
    return tmp

def conv_complex(func, timg, weight, **kwargs):
    return func(timg, weight.real, **kwargs) + 1j* func(timg, weight.imag, **kwargs)